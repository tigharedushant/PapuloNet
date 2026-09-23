"""
modules/efficientnet_model.py

AEF-CRC Phase 3: EfficientNet-B0 backbone + two-stage transfer learning.

*** NOT EXECUTED IN THIS SANDBOX *** -- see modules/image_loader.py's
module docstring for why (no TensorFlow, no network access here).

Explicitly does NOT add: SE-Net, CBAM, ECA, coordinate attention, or
any other attention mechanism (Part 26 -- EfficientNet-B0 already
contains squeeze-and-excitation internally; adding another attention
layer here would be exactly the "novelty for its own sake" this
project has repeatedly rejected elsewhere).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.config import PSDConfig


@dataclass
class ModelInfo:
    total_params: int
    trainable_params: int
    frozen_params: int
    pretrained_source: str
    input_resolution: int


def build_stage1_model(config: PSDConfig, num_classes: int):
    """Stage 1 (Part 11): EfficientNet-B0 backbone frozen, ImageNet
    weights, train only a new classification head."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",  # requires network access to download on first use, in the user's own environment
        input_shape=(config.image_size, config.image_size, 3),
        pooling="avg",
    )
    backbone.trainable = False  # frozen for stage 1

    inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3))
    x = backbone(inputs, training=False)
    x = layers.Dropout(config.dropout_rate)(x)
    # dtype="float32" here is required, not cosmetic, when
    # mixed_float16 is active (see training.py's configure_gpu()):
    # softmax + categorical_crossentropy computed in float16 is a real
    # precision risk near 0/1 probabilities. Harmless no-op if mixed
    # precision isn't active.
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = models.Model(inputs, outputs, name="efficientnet_b0_stage1")

    clipnorm = getattr(config, "adam_clipnorm", 1.0)
    optimizer = (
        tf.keras.optimizers.Adam(learning_rate=config.stage1_learning_rate, clipnorm=clipnorm)
        if clipnorm is not None and clipnorm > 0
        else tf.keras.optimizers.Adam(learning_rate=config.stage1_learning_rate)
    )

    loss_obj = getattr(config, "loss_function", None)
    if loss_obj is None:
        loss_name = getattr(config, "loss_name", "categorical_crossentropy")
        focal_gamma = getattr(config, "focal_gamma", 2.0)
        from modules.losses import get_loss
        loss_obj = get_loss(loss_name, gamma=focal_gamma)

    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=["accuracy"],
    )
    return model, backbone


def unfreeze_for_stage2(model, backbone, config: PSDConfig):
    """Stage 2 (Part 11): unfreeze the top config.unfrozen_layers of the
    backbone, recompile with a much smaller learning rate. Does NOT
    unfreeze the entire network at once (Part 11: 'Do not immediately
    unfreeze the entire network')."""
    import tensorflow as tf

    backbone.trainable = True
    for layer in backbone.layers[:-config.unfrozen_layers]:
        layer.trainable = False
    # BatchNorm layers are kept frozen even within the unfrozen region --
    # standard transfer-learning practice to avoid destabilizing running
    # statistics learned on ImageNet's much larger dataset.
    for layer in backbone.layers[-config.unfrozen_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    clipnorm = getattr(config, "adam_clipnorm", 1.0)
    optimizer = (
        tf.keras.optimizers.Adam(learning_rate=config.stage2_learning_rate, clipnorm=clipnorm)
        if clipnorm is not None and clipnorm > 0
        else tf.keras.optimizers.Adam(learning_rate=config.stage2_learning_rate)
    )

    loss_obj = getattr(config, "loss_function", None)
    if loss_obj is None:
        loss_name = getattr(config, "loss_name", "categorical_crossentropy")
        focal_gamma = getattr(config, "focal_gamma", 2.0)
        from modules.losses import get_loss
        loss_obj = get_loss(loss_name, gamma=focal_gamma)

    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=["accuracy"],
    )
    return model


def extract_deep_features(config: PSDConfig, backbone, records, experiment_name: str, fold_index: int, split_name: str, checkpoint_path):
    """Task 7: deterministic deep-feature cache from a TRAINED model's
    backbone, for later fusion with GLCM/LBP/HOG (Phase 5 -- not
    implemented here).

    Takes `backbone` directly (the same object returned by
    build_stage1_model() and mutated in place by unfreeze_for_stage2()
    during fine-tuning) rather than looking it up from the compiled
    model by layer index -- an earlier version used
    `model.get_layer(index=1)`, which is fragile (silently wrong if
    the functional model's layer order ever changes) and unnecessary,
    since training.py already has a live reference to the exact
    backbone object that was actually trained.

    With pooling="avg" set at construction (build_stage1_model),
    backbone(images) already returns the pooled (batch, 1280) vector
    directly -- EfficientNet-B0's final block has 1280 filters, so no
    additional pooling step is needed or performed here. Verified
    explicitly below (not just assumed) via an assertion on the
    output shape.

    Deliberately does NOT touch plan.holdout_test_records anywhere in
    this project yet -- only call this with fold.train_records or
    fold.val_records. Extracting from test now, before it's actually
    needed for final evaluation, is an unnecessary leakage-adjacent
    risk for zero present benefit (Task 7's own success criteria:
    "never mix train/validation/test features").

    Caches to artifacts/phase3/deep_features/<experiment>/fold_XX/<split>/<psd_id>.npy
    plus one manifest.json per (experiment, fold, split) recording the
    checkpoint path used -- Task 7: "record the model/checkpoint used
    to generate the features," so a feature cache can never be
    silently reused against the wrong trained model.
    """
    import json
    import tensorflow as tf
    from modules.image_loader import build_dataset

    out_dir = config.aef_crc_artifacts_dir / "deep_features" / experiment_name / f"fold_{fold_index:02d}" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    import hashlib
    from modules.experiment_config import representation_id

    manifest_p = out_dir / "manifest.json"
    if manifest_p.exists():
        try:
            m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
            expected_repr_id = representation_id(config)
            if (
                m_data.get("experiment_name") == experiment_name
                and m_data.get("fold_index") == fold_index
                and m_data.get("split") == split_name
                and m_data.get("n_features") == len(records)
                and m_data.get("feature_dim") == 1280
                and m_data.get("backbone_name") == config.backbone
                and m_data.get("representation_id") == expected_repr_id
            ):
                if (out_dir / f"{records[0].psd_id}.npy").exists() and (out_dir / f"{records[-1].psd_id}.npy").exists():
                    return out_dir
        except Exception:
            pass

    import tempfile
    import shutil
    import subprocess

    is_9p = str(out_dir).startswith("/mnt/")
    target_write_dir = Path(tempfile.mkdtemp(prefix="deep_feat_")) if is_9p else out_dir
    target_write_dir.mkdir(parents=True, exist_ok=True)

    # Implementation detail for deep feature extraction: use extraction_batch_size=16
    # to ensure safe GPU memory headroom on 4 GB VRAM devices without altering the training protocol.
    extraction_batch_size = getattr(config, "extraction_batch_size", 16)
    ds = build_dataset(
        config, records, config.target_classes, training=False,
        apply_training_time_augmentation=False, batch_size=extraction_batch_size,
    )
    psd_ids_in_order = [r.psd_id for r in records]  # build_dataset preserves order when training=False

    idx = 0
    batch_features = None
    for images, _ in ds:
        batch_features = backbone(images, training=False).numpy()
        assert batch_features.shape[-1] == 1280, (
            f"Expected EfficientNet-B0's pooled representation to be 1280-D, got {batch_features.shape[-1]}. "
            f"Check that the backbone was constructed with pooling='avg' (build_stage1_model)."
        )
        for feat in batch_features:
            psd_id = psd_ids_in_order[idx]
            np.save(target_write_dir / f"{psd_id}.npy", feat.astype(np.float32))
            idx += 1

    freeze_p = config.aef_crc_reports_dir / "dataset_freeze.json"
    dataset_freeze_hash = hashlib.sha256(freeze_p.read_bytes()).hexdigest() if freeze_p.exists() else None

    plan_p = config.aef_crc_reports_dir / "fold_plan.csv"
    fold_plan_hash = hashlib.sha256(plan_p.read_bytes()).hexdigest() if plan_p.exists() else None

    ckpt_p = Path(checkpoint_path)
    ckpt_hash = hashlib.sha256(ckpt_p.read_bytes()).hexdigest() if ckpt_p.exists() and ckpt_p.is_file() else None

    manifest = {
        "experiment_name": experiment_name,
        "fold_index": fold_index,
        "split": split_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_hash,
        "dataset_freeze_hash": dataset_freeze_hash,
        "fold_plan_hash": fold_plan_hash,
        "n_features": idx,
        "feature_dim": int(batch_features.shape[-1]) if batch_features is not None else None,
        "psd_ids": psd_ids_in_order,
        "backbone_name": config.backbone,
        "representation_id": representation_id(config),
    }
    (target_write_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if is_9p:
        shutil.copytree(target_write_dir, out_dir, dirs_exist_ok=True)
        shutil.rmtree(target_write_dir, ignore_errors=True)

    return out_dir


def _resolve_deep_feature_root(config: PSDConfig, experiment_name: str) -> Path:
    """Resolves the root directory containing deep feature caches for experiment_name.
    If the experiment is a V2 experiment (e.g. 'P3-V2-*') or exists under
    config.aef_crc_phase3_v2_artifacts_dir, use the V2 artifacts directory;
    otherwise fall back to config.aef_crc_artifacts_dir."""
    v2_dir = getattr(config, "aef_crc_phase3_v2_artifacts_dir", None)
    if v2_dir is not None:
        v2_candidate = v2_dir / "deep_features" / experiment_name
        if v2_candidate.exists() or experiment_name.startswith("P3-V2-"):
            return v2_dir
    return config.aef_crc_artifacts_dir


def load_deep_feature(config: PSDConfig, experiment_name: str, fold_index: int, split_name: str, psd_id: str):
    """Reads one cached deep feature vector -- the read-side counterpart
    to extract_deep_features(), for Phase 5's fusion code to call
    without needing to know the cache directory structure directly."""
    root_dir = _resolve_deep_feature_root(config, experiment_name)
    path = (root_dir / "deep_features" / experiment_name /
            f"fold_{fold_index:02d}" / split_name / f"{psd_id}.npy")
    if not path.exists():
        raise FileNotFoundError(
            f"No cached deep feature for psd_id={psd_id} at {path} -- "
            f"run extract_deep_features() for this experiment/fold/split first."
        )
    return np.load(path)


def load_deep_feature_manifest(config: PSDConfig, experiment_name: str, fold_index: int, split_name: str) -> dict:
    """Reads the manifest.json extract_deep_features() writes alongside
    its .npy files -- the read-side counterpart used by
    modules.backbones.validate_deep_feature_cache() to check a cache's
    identity BEFORE any individual vector is loaded. Raises
    FileNotFoundError (not a silent {}) if the manifest is missing --
    an absent manifest means this cache predates the metadata contract
    or was never actually produced by extract_deep_features()."""
    import json
    root_dir = _resolve_deep_feature_root(config, experiment_name)
    path = (root_dir / "deep_features" / experiment_name /
            f"fold_{fold_index:02d}" / split_name / "manifest.json")
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest.json for experiment={experiment_name!r} fold={fold_index} split={split_name!r} "
            f"at {path} -- run extract_deep_features() for this experiment/fold/split first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def describe_model(model) -> ModelInfo:
    trainable = sum(int(w.shape.num_elements()) for w in model.trainable_weights)
    total = model.count_params()
    return ModelInfo(
        total_params=total,
        trainable_params=trainable,
        frozen_params=total - trainable,
        pretrained_source="ImageNet (tensorflow.keras.applications.EfficientNetB0, weights='imagenet')",
        input_resolution=model.input_shape[1],
    )
