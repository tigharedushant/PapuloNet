"""
modules/training.py

AEF-CRC Phase 3: Training orchestration.

*** NOT EXECUTED IN THIS SANDBOX *** -- see modules/image_loader.py's
module docstring for why.

Reuses modules/fold_loader.py's FoldPlan directly (Part 16: "Do NOT
create a second independent CV splitter") and modules/evaluation.py
for all metrics. Never touches plan.holdout_test_records -- test stays
completely locked through every fold of every experiment.

--- Three fixes made on review, before any real training run ---

1. class_weight removed from model.fit() entirely. See
   image_loader.py's docstring for the full explanation: Keras's own
   docs say class_weight is not supported when x is a tf.data.Dataset.
   Weighting now flows through sample_weight, computed inside
   build_dataset() from the same fold.class_weights values -- one
   formula, one place, just a different (correct) delivery mechanism.

2. Fresh ModelCheckpoint instances per stage (NO EarlyStopping). In
   accordance with the pre-registered protocol, EarlyStopping is omitted
   so models train for the full fixed epoch budget (15 epochs Stage 1,
   10 epochs Stage 2), checkpointing ONLY the lowest validation loss
   (save_best_only=True). Constructing fresh callback instances per stage
   ensures clean state tracking across stages.

3. Global seed synchronization added (Part 20: "Synchronize where
   practical: Python, NumPy, TensorFlow"). The original version only
   passed config.random_seed to ds.shuffle() -- weight initialization,
   dropout masks, and any other framework-level randomness were left
   on whatever ambient seed state existed. set_global_seeds() now runs
   once per fold, before model construction, covering all three.

--- What the backbone contract does NOT abstract (disclosed, not hidden) ---

modules.backbones.BackboneSpec abstracts model CONSTRUCTION
(build_stage1, unfreeze_stage2), introspection (describe), and feature
EXTRACTION (extract_features) -- the four things downstream phases
(4/5/6) actually depend on, since they only ever consume the resulting
cached deep-feature vectors, never the model object itself. It does
NOT abstract the training loop below: configure_gpu(), set_global_seeds(),
_build_callbacks(), and the two model.fit(train_ds, ..., callbacks=...)
calls in run_experiment() are TensorFlow/Keras-specific and assume
build_stage1 returns something with a Keras-compatible .fit() method
over a tf.data.Dataset. A genuinely different-framework backbone (e.g.
PyTorch) could not just implement the four BackboneSpec functions and
plug in here -- it would need its own training-loop orchestration, not
just its own build_stage1/unfreeze_stage2. This is deliberately NOT
solved this phase: EfficientNet is the only real implementation, and a
training-loop abstraction designed around a single concrete case would
be speculative generality, not a tested contract. If a second backbone
in a genuinely different framework is ever added, THAT is when this
boundary needs revisiting -- not before.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.config import PSDConfig
from modules.fold_loader import FoldPlan
from modules.image_loader import build_dataset
from modules.backbones import get_backbone
from modules.evaluation import compute_fold_metrics, aggregate_fold_metrics, FoldMetrics, ClassMetrics
from modules.experiment_config import representation_id
import numpy as np
from utils.logger import get_module_logger


from modules.device_utils import (
    DeviceInfo,
    detect_device_environment,
    configure_execution_device,
    detect_and_configure_device,
    device_smoke_test,
    format_device_manifest,
)


def configure_gpu() -> str:
    """Part 21 (GPU memory efficiency): backwards-compatible wrapper returning status string."""
    return str(detect_and_configure_device())


def set_global_seeds(seed: int) -> None:
    """Part 20. Called once per fold, before model construction, so
    weight initialization and dropout are reproducible given the same
    seed -- not just the data shuffle order."""
    import os
    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _build_callbacks(config: PSDConfig, checkpoint_path: Path):
    """Saves the best checkpoint for the fold according to validation loss.
    Fixed epochs protocol: exactly stage1_epochs and stage2_epochs are run,
    with ModelCheckpoint capturing the minimum val_loss checkpoint."""
    import tensorflow as tf

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True,
        ),
    ]


def _safe_load_model(ckpt_path: Path, max_retries: int = 15, delay: float = 1.0):
    """Safely loads a Keras model checkpoint, allowing OS file/zip buffers to flush.
    Prevents race conditions where ModelCheckpoint has finished writing but the OS
    filesystem cache (e.g. WSL-to-Windows 9P mount) has not yet closed the zip archive.
    If full-model deserialization fails due to Keras config schema differences across
    minor versions (e.g. VarianceScaling / BatchNormalization attributes), falls back
    to reconstructing the standard architecture and loading weights from model.weights.h5.
    """
    import time
    import zipfile
    import tempfile
    import tensorflow as tf
    from config.config import get_config
    from modules.backbones import get_backbone

    ckpt_path = Path(ckpt_path)
    for attempt in range(max_retries):
        try:
            if ckpt_path.exists() and zipfile.is_zipfile(str(ckpt_path)):
                try:
                    return tf.keras.models.load_model(ckpt_path, compile=False)
                except Exception:
                    config = get_config()
                    backbone_spec = get_backbone(config.backbone)
                    model, backbone = backbone_spec.build_stage1(config, num_classes=len(config.target_classes))
                    model = backbone_spec.unfreeze_stage2(model, backbone, config)
                    with zipfile.ZipFile(ckpt_path) as z:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            extracted = z.extract("model.weights.h5", tmpdir)
                            model.load_weights(extracted)
                            return model
        except Exception:
            pass
        time.sleep(delay)

    try:
        return tf.keras.models.load_model(ckpt_path, compile=False)
    except Exception:
        config = get_config()
        backbone_spec = get_backbone(config.backbone)
        model, backbone = backbone_spec.build_stage1(config, num_classes=len(config.target_classes))
        model = backbone_spec.unfreeze_stage2(model, backbone, config)
        with zipfile.ZipFile(ckpt_path) as z:
            with tempfile.TemporaryDirectory() as tmpdir:
                extracted = z.extract("model.weights.h5", tmpdir)
                model.load_weights(extracted)
                return model


def safe_copy_file(src: Path, dst: Path) -> None:
    """Safely copies files in chunks, avoiding os.sendfile bugs across virtual 9P mounts."""
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        while True:
            chunk = fsrc.read(4 * 1024 * 1024)
            if not chunk:
                break
            fdst.write(chunk)


def validate_cached_fold(
    fold_dir: Path,
    config: PSDConfig,
    fold_index: int,
    experiment_name: str,
    apply_training_time_augmentation: bool,
) -> Tuple[bool, str]:
    """Validates that a cached fold directory contains complete, uncorrupted artifacts
    strictly matching the CURRENT execution environment, dataset freeze, and fold plan.

    Returns (True, "OK") if safe to reuse.
    Returns (False, reason) if any mismatch, missing file, or corruption is detected.
    """
    best_model_path = fold_dir / "best_model.keras"
    manifest_path = fold_dir / "manifest.json"
    cm_path = fold_dir / "confusion_matrix.json"

    # 1. Check all required artifact files exist
    if not best_model_path.exists():
        return False, f"Missing best_model.keras in {fold_dir.name}"
    if not manifest_path.exists():
        return False, f"Missing manifest.json in {fold_dir.name}"
    if not cm_path.exists():
        return False, f"Missing confusion_matrix.json in {fold_dir.name}"

    # 2. Parse manifest.json
    try:
        m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Corrupt manifest.json: {exc}"

    # 3. Parse confusion_matrix.json
    try:
        cm_data = json.loads(cm_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Corrupt confusion_matrix.json: {exc}"

    # 4. Check dataset freeze hash against current authoritative report
    freeze_p = config.aef_crc_reports_dir / "dataset_freeze.json"
    if not freeze_p.exists():
        return False, "Current dataset_freeze.json does not exist on disk"
    expected_freeze_hash = hashlib.sha256(freeze_p.read_bytes()).hexdigest()
    actual_freeze_hash = m_data.get("dataset_freeze_hash")
    if not actual_freeze_hash:
        return False, "Missing dataset_freeze_hash in manifest"
    if actual_freeze_hash != expected_freeze_hash:
        return False, (
            f"dataset_freeze_hash mismatch (cached: {actual_freeze_hash[:12]}..., "
            f"expected: {expected_freeze_hash[:12]}...)"
        )

    # 5. Check fold plan hash against current authoritative report
    plan_p = config.aef_crc_reports_dir / "fold_plan.csv"
    if not plan_p.exists():
        return False, "Current fold_plan.csv does not exist on disk"
    expected_plan_hash = hashlib.sha256(plan_p.read_bytes()).hexdigest()
    actual_plan_hash = m_data.get("fold_plan_hash")
    if not actual_plan_hash:
        return False, "Missing fold_plan_hash in manifest"
    if actual_plan_hash != expected_plan_hash:
        return False, (
            f"fold_plan_hash mismatch (cached: {actual_plan_hash[:12]}..., "
            f"expected: {expected_plan_hash[:12]}...)"
        )

    # 6. Check fold_index
    if m_data.get("fold_index") != fold_index:
        return False, f"fold_index mismatch (cached: {m_data.get('fold_index')}, requested: {fold_index})"

    # 7. Check representation_id
    expected_repr_id = representation_id(config)
    actual_repr_id = m_data.get("representation_id")
    if not actual_repr_id:
        return False, "Missing representation_id in manifest"
    if actual_repr_id != expected_repr_id:
        return False, f"representation_id mismatch (cached: {actual_repr_id}, expected: {expected_repr_id})"

    # 8. Check random_seed
    if m_data.get("random_seed") != config.random_seed:
        return False, f"random_seed mismatch (cached: {m_data.get('random_seed')}, expected: {config.random_seed})"

    # 9. Check preprocessing_mode & training_time_augmentation
    if m_data.get("preprocessing_mode") != config.preprocessing_mode:
        return False, f"preprocessing_mode mismatch (cached: {m_data.get('preprocessing_mode')}, expected: {config.preprocessing_mode})"
    if m_data.get("training_time_augmentation") != apply_training_time_augmentation:
        return False, f"training_time_augmentation mismatch (cached: {m_data.get('training_time_augmentation')}, expected: {apply_training_time_augmentation})"

    # 10. Check metric fields exist in manifest and confusion matrix
    if "macro_f1" not in m_data or "accuracy" not in m_data:
        return False, "Missing macro_f1 or accuracy in manifest"
    if "matrix" not in cm_data or "per_class" not in cm_data:
        return False, "Missing matrix or per_class in confusion_matrix.json"
    if cm_data.get("class_order") != list(config.target_classes):
        return False, "class_order mismatch in confusion_matrix.json"

    return True, "Verified matching current provenance, fold plan, and configuration"


def run_experiment(
    config: PSDConfig,
    plan: FoldPlan,
    experiment_name: str,
    apply_training_time_augmentation: bool,
    extract_features: bool = False,
) -> List[FoldMetrics]:
    """Runs ALL folds in plan (never a 'best fold' shortcut -- Part 16)
    for one named experiment (e.g. 'P3-BASE', 'P3-PRE', 'P3-AUG'), and
    returns per-fold metrics for aggregate_fold_metrics() to summarize.

    extract_features: if True, also caches deep features (Task 7) for
    this experiment's fold.train_records and fold.val_records after
    training completes. Intentionally opt-in and off by default --
    the brief is explicit that features should come from the WINNING
    experiment, not every experiment; the caller decides that after
    seeing all experiments' metrics, then re-invokes with this flag
    (see run_aef_crc_phase3.py) rather than this function guessing."""
    from modules.experiment_config import validate_experiment_config
    validate_experiment_config(config)  # fail fast, before any GPU/training work -- not deep inside a fold loop

    logger = get_module_logger("training", config.aef_crc_phase3_logs_dir, config.log_level)
    class_order = config.target_classes
    experiment_dir = config.aef_crc_artifacts_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    device_info = detect_and_configure_device()
    logger.info(f"Compute configuration: {device_info.status_message}")

    fold_metrics: List[FoldMetrics] = []

    for fold in plan.folds:
        fold_dir = experiment_dir / f"fold_{fold.fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"=== {experiment_name} | Fold {fold.fold_index} starting ===")
        start_time = time.time()

        set_global_seeds(config.random_seed)  # fix 3 -- before model construction, every fold

        # Class weights come ONLY from this fold's own training portion
        # (Part 12), already computed that way in fold_loader.py -- not
        # recomputed here, just passed through, so there is exactly one
        # place in the codebase that formula lives. Delivered via
        # sample_weight now, not class_weight -- see fix 1.
        train_ds = build_dataset(
            config, fold.train_records, class_order, training=True,
            apply_training_time_augmentation=apply_training_time_augmentation,
            class_weights=fold.class_weights,
        )
        val_ds = build_dataset(
            config, fold.val_records, class_order, training=False,
            apply_training_time_augmentation=False,
            class_weights=None,  # never weight validation -- see image_loader.py's refusal check
        )


        backbone_spec = get_backbone(config.backbone)

        best_model_path = fold_dir / "best_model.keras"
        manifest_path = fold_dir / "manifest.json"
        cm_path = fold_dir / "confusion_matrix.json"

        is_valid, validation_reason = validate_cached_fold(
            fold_dir=fold_dir,
            config=config,
            fold_index=fold.fold_index,
            experiment_name=experiment_name,
            apply_training_time_augmentation=apply_training_time_augmentation,
        )

        if is_valid:
            logger.info(f"Fold {fold.fold_index}: reusing verified cached fold artifacts ({validation_reason}) from {manifest_path}")
            m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            cm_data = json.loads(cm_path.read_text(encoding="utf-8"))
            per_class = {
                cls_name: ClassMetrics(
                    precision=float(cm_data["per_class"][cls_name]["precision"]),
                    recall=float(cm_data["per_class"][cls_name]["recall"]),
                    f1=float(cm_data["per_class"][cls_name]["f1"]),
                    support=int(cm_data["per_class"][cls_name]["support"]),
                )
                for cls_name in cm_data["class_order"]
            }
            total_support = sum(pc.support for pc in per_class.values())
            weighted_f1 = sum(pc.f1 * pc.support for pc in per_class.values()) / max(1, total_support)
            metrics = FoldMetrics(
                fold_index=fold.fold_index,
                macro_f1=float(m_data["macro_f1"]),
                accuracy=float(m_data["accuracy"]),
                balanced_accuracy=float(m_data["balanced_accuracy"]),
                weighted_f1=weighted_f1,
                mcc=float(m_data["mcc"]),
                per_class=per_class,
                confusion=np.array(cm_data["matrix"]),
                class_order=cm_data["class_order"],
            )
            fold_metrics.append(metrics)

            if extract_features:
                model = _safe_load_model(best_model_path)
                backbone_layer = model.get_layer("efficientnetb0") if "efficientnetb0" in [l.name for l in model.layers] else model
                for records, split_name in ((fold.train_records, "train"), (fold.val_records, "val")):
                    backbone_spec.extract_features(config, backbone_layer, records, experiment_name, fold.fold_index, split_name, best_model_path)
                logger.info(f"Deep features cached for fold {fold.fold_index} using reloaded best checkpoint {best_model_path}")
            continue
        else:
            if manifest_path.exists() or best_model_path.exists() or cm_path.exists():
                logger.warning(
                    f"Fold {fold.fold_index}: existing cached artifacts REJECTED ({validation_reason}). "
                    f"Treating fold as stale and retraining from scratch."
                )

        model, backbone = backbone_spec.build_stage1(config, num_classes=len(class_order))
        model_info = backbone_spec.describe(model)
        logger.info(
            f"Backbone: {backbone_spec.name} | Model built: {model_info.total_params:,} total params "
            f"({model_info.trainable_params:,} trainable, {model_info.frozen_params:,} frozen), "
            f"pretrained: {model_info.pretrained_source}"
        )

        stage1_ckpt = fold_dir / "stage1_best.keras"
        stage2_ckpt = fold_dir / "stage2_best.keras"

        history_stage1 = model.fit(
            train_ds, validation_data=val_ds, epochs=config.stage1_epochs,
            callbacks=_build_callbacks(config, stage1_ckpt), verbose=2,
        )

        val_loss_stage1 = min(history_stage1.history.get("val_loss", [float("inf")]))

        model = backbone_spec.unfreeze_stage2(model, backbone, config)
        history_stage2 = model.fit(
            train_ds, validation_data=val_ds, epochs=config.stage2_epochs,
            callbacks=_build_callbacks(config, stage2_ckpt), verbose=2,
        )

        val_loss_stage2 = min(history_stage2.history.get("val_loss", [float("inf")]))

        # Select the superior checkpoint based on predefined val_loss selection
        if val_loss_stage2 <= val_loss_stage1 and stage2_ckpt.exists():
            selected_ckpt = stage2_ckpt
            selected_stage = 2
            best_val_loss = float(val_loss_stage2)
            v_losses = history_stage2.history.get("val_loss", [])
            selected_epoch = int(v_losses.index(val_loss_stage2)) if val_loss_stage2 in v_losses else -1
        elif stage1_ckpt.exists():
            selected_ckpt = stage1_ckpt
            selected_stage = 1
            best_val_loss = float(val_loss_stage1)
            v_losses = history_stage1.history.get("val_loss", [])
            selected_epoch = int(v_losses.index(val_loss_stage1)) if val_loss_stage1 in v_losses else -1
        elif stage2_ckpt.exists():
            selected_ckpt = stage2_ckpt
            selected_stage = 2
            best_val_loss = float(val_loss_stage2)
            v_losses = history_stage2.history.get("val_loss", [])
            selected_epoch = int(v_losses.index(val_loss_stage2)) if val_loss_stage2 in v_losses else -1
        else:
            selected_ckpt = None
            selected_stage = 0
            best_val_loss = float("inf")
            selected_epoch = -1

        # Authoritative best model copy for fold
        best_model_path = fold_dir / "best_model.keras"
        if selected_ckpt is not None and selected_ckpt.exists():
            logger.info(f"Fold {fold.fold_index}: reloading selected {selected_ckpt.name} (stage {selected_stage}) before evaluation")
            model = _safe_load_model(selected_ckpt)
            safe_copy_file(selected_ckpt, best_model_path)
            checkpoint_used = str(best_model_path)
        else:
            checkpoint_used = "live_unrestored"

        # Re-derive backbone reference from loaded model for feature extraction
        backbone_layer = model.get_layer("efficientnetb0") if "efficientnetb0" in [l.name for l in model.layers] else backbone

        # Predict on fold-validation for this fold's metrics -- test set
        # is never referenced anywhere in this function.
        y_true, y_pred = [], []
        for images, one_hot_labels in val_ds:
            probs = model.predict(images, verbose=0)
            pred_idx = probs.argmax(axis=1)
            true_idx = one_hot_labels.numpy().argmax(axis=1)
            y_pred.extend(class_order[i] for i in pred_idx)
            y_true.extend(class_order[i] for i in true_idx)

        metrics = compute_fold_metrics(y_true, y_pred, class_order, fold.fold_index)
        fold_metrics.append(metrics)

        if extract_features:
            for records, split_name in ((fold.train_records, "train"), (fold.val_records, "val")):
                backbone_spec.extract_features(config, backbone_layer, records, experiment_name, fold.fold_index, split_name, best_model_path)
            logger.info(f"Deep features cached for fold {fold.fold_index} using reloaded best checkpoint {selected_ckpt}")

        _write_fold_artifacts(
            fold_dir, config, fold, model_info, history_stage1, history_stage2,
            metrics, apply_training_time_augmentation, time.time() - start_time,
            selected_stage=selected_stage, checkpoint_used=checkpoint_used,
            selected_epoch=selected_epoch, best_val_loss=best_val_loss,
            device_info=device_info,
        )
        logger.info(f"=== {experiment_name} | Fold {fold.fold_index} done: macro_f1={metrics.macro_f1:.4f} ===")

        # Clean session between folds to prevent GPU/RAM memory buildup
        import tensorflow as tf
        tf.keras.backend.clear_session()
        del model, backbone, backbone_layer, train_ds, val_ds
        gc.collect()

    return fold_metrics


def _write_fold_artifacts(
    fold_dir, config, fold, model_info, history1, history2, metrics, aug_flag, runtime_sec,
    selected_stage: int = 0, checkpoint_used: str = "",
    selected_epoch: int = -1, best_val_loss: float = 0.0,
    device_info: DeviceInfo = None,
):
    """Part 19: config, seed, fold ID, class weights, preprocessing
    config, training history, validation metrics, confusion matrix, and device metadata --
    all persisted per fold, not just the checkpoint weights."""
    import hashlib
    import sklearn

    freeze_p = config.aef_crc_reports_dir / "dataset_freeze.json"
    dataset_freeze_hash = hashlib.sha256(freeze_p.read_bytes()).hexdigest() if freeze_p.exists() else None

    meta_p = config.reports_dir / "metadata.csv"
    metadata_hash = hashlib.sha256(meta_p.read_bytes()).hexdigest() if meta_p.exists() else None

    plan_p = config.aef_crc_reports_dir / "fold_plan.csv"
    fold_plan_hash = hashlib.sha256(plan_p.read_bytes()).hexdigest() if plan_p.exists() else None

    manifest = {
        "fold_index": fold.fold_index,
        "random_seed": config.random_seed,
        "class_weights": fold.class_weights,
        "preprocessing_mode": config.preprocessing_mode,
        "hair_coverage_threshold": config.hair_coverage_threshold,
        "contrast_std_threshold": config.contrast_std_threshold,
        "training_time_augmentation": aug_flag,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "stage1_learning_rate": config.stage1_learning_rate,
        "stage2_learning_rate": config.stage2_learning_rate,
        "stage1_epochs": config.stage1_epochs,
        "stage2_epochs": config.stage2_epochs,
        "unfrozen_layers": config.unfrozen_layers,
        "dropout_rate": config.dropout_rate,
        "representation_id": representation_id(config),
        "run_id": getattr(config, "run_id", None) or os.environ.get("AEFCRC_RUN_ID", None),
        "dataset_freeze_hash": dataset_freeze_hash,
        "metadata_hash": metadata_hash,
        "fold_plan_hash": fold_plan_hash,
        "model_total_params": model_info.total_params,
        "model_trainable_params": model_info.trainable_params,
        "runtime_seconds": runtime_sec,
        "selected_stage": selected_stage,
        "selected_epoch": selected_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_used": checkpoint_used,
        "macro_f1": metrics.macro_f1,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "mcc": metrics.mcc,
        "device_type": device_info.device_type if device_info else "Unknown",
        "device_name": device_info.device_name if device_info else "Unknown",
        "device_count": device_info.device_count if device_info else 0,
        "execution_strategy": device_info.execution_strategy if device_info else "Unknown",
        "device_fallback": device_info.device_fallback if device_info else False,
        "tensorflow_version": device_info.tensorflow_version if device_info else "Unknown",
        "keras_version": getattr(device_info, "keras_version", "Unknown") if device_info else "Unknown",
        "python_version": device_info.python_version if device_info else "Unknown",
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "platform_system": getattr(device_info, "platform_system", "Unknown") if device_info else "Unknown",
        "platform_release": getattr(device_info, "platform_release", "Unknown") if device_info else "Unknown",
        "cuda_available": getattr(device_info, "cuda_available", False) if device_info else False,
        "mixed_precision_policy": getattr(device_info, "mixed_precision_policy", "float32") if device_info else "Unknown",
        "physical_devices": getattr(device_info, "physical_devices", []) if device_info else [],
    }
    (fold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Persist complete 4x4 confusion matrix as CSV and JSON
    cm_path = fold_dir / "confusion_matrix.csv"
    with cm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class \\ pred_class"] + list(metrics.class_order))
        for cls_name, row in zip(metrics.class_order, metrics.confusion):
            writer.writerow([cls_name] + [int(v) for v in row])

    cm_json_path = fold_dir / "confusion_matrix.json"
    cm_dict = {
        "class_order": list(metrics.class_order),
        "matrix": [[int(v) for v in row] for row in metrics.confusion],
        "per_class": {
            cls: {
                "precision": metrics.per_class[cls].precision,
                "recall": metrics.per_class[cls].recall,
                "f1": metrics.per_class[cls].f1,
                "support": metrics.per_class[cls].support,
            } for cls in metrics.class_order if cls in metrics.per_class
        }
    }
    cm_json_path.write_text(json.dumps(cm_dict, indent=2), encoding="utf-8")

    with (fold_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "epoch", "loss", "val_loss", "accuracy", "val_accuracy"])
        for stage_num, hist in ((1, history1), (2, history2)):
            for epoch in range(len(hist.history["loss"])):
                writer.writerow([
                    stage_num, epoch,
                    hist.history["loss"][epoch], hist.history.get("val_loss", [None])[epoch],
                    hist.history.get("accuracy", [None])[epoch], hist.history.get("val_accuracy", [None])[epoch],
                ])
