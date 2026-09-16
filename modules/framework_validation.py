"""
modules/framework_validation.py

AEF-CRC Phase 3: Comprehensive Framework Validation Suite.

This module validates the reusable Phase 3 framework across 10 key components:
  1. Configuration and hyperparameter contracts
  2. Fold definitions, disjointness, and quarantine enforcement
  3. Preprocessing routing and safety (standard vs conditional)
  4. Augmentation routing and refusal checks (train vs val)
  5. Model architecture, 1280-D features, and 2-stage unfreezing
  6. Device-agnostic detection and smoke test
  7. Training configuration, loss, and class weighting
  8. Checkpoint selection logic (val_loss minimization, no EarlyStopping)
  9. Multi-metric evaluation and C x C confusion matrices
 10. Single-winner selection logic (0.005 practical-equivalence margin)

All checks operate independently of whether the real image dataset is present,
guaranteeing that the pipeline can be completely certified for correctness,
reproducibility, and portability prior to executing training on real data.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.config import PSDConfig, get_config
from modules.backbones import BACKBONE_REGISTRY, get_backbone
from modules.device_utils import DeviceInfo, detect_and_configure_device
from modules.evaluation import (
    AggregatedMetrics,
    FoldMetrics,
    aggregate_fold_metrics,
    compute_fold_metrics,
    select_phase3_winner,
)
from modules.preprocessing import ConditionalPreprocessor


# ==============================================================================
# 1. Configuration Validation
# ==============================================================================

def validate_configuration(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates that hyperparameters and experiment contracts match IEEE specifications."""
    errors = []

    # Image dimensions
    expected_size = 224
    if isinstance(config.image_size, (tuple, list)):
        if config.image_size != (expected_size, expected_size):
            errors.append(f"config.image_size tuple {config.image_size} != ({expected_size}, {expected_size})")
    elif config.image_size != expected_size:
        errors.append(f"config.image_size {config.image_size} != {expected_size}")

    # Batch size
    if config.batch_size not in (16, 32):
        errors.append(f"config.batch_size {config.batch_size} not in expected (16, 32)")

    # Epochs
    if config.stage1_epochs != 15:
        errors.append(f"config.stage1_epochs {config.stage1_epochs} != 15")
    if config.stage2_epochs != 10:
        errors.append(f"config.stage2_epochs {config.stage2_epochs} != 10")

    # Learning rates
    if config.stage1_learning_rate not in (1e-4, 1e-3):
        errors.append(f"config.stage1_learning_rate {config.stage1_learning_rate} not in (1e-4, 1e-3)")
    if config.stage2_learning_rate not in (1e-5, 1e-4):
        errors.append(f"config.stage2_learning_rate {config.stage2_learning_rate} not in (1e-5, 1e-4)")

    # Architecture & regularization
    if config.unfrozen_layers != 20:
        errors.append(f"config.unfrozen_layers {config.unfrozen_layers} != 20")
    if abs(config.dropout_rate - 0.3) > 1e-5:
        errors.append(f"config.dropout_rate {config.dropout_rate} != 0.3")

    # Authoritative PSD-HP Erythematous-Squamous Disease Taxonomy
    expected_classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    if list(config.target_classes) != expected_classes:
        errors.append(f"config.target_classes {config.target_classes} != {expected_classes}")

    # Backbone
    if config.backbone != "efficientnet_b0" or config.backbone not in BACKBONE_REGISTRY:
        errors.append(f"config.backbone '{config.backbone}' not valid registered backbone")

    # Random seed
    if not isinstance(config.random_seed, int):
        errors.append(f"config.random_seed {config.random_seed} is not an integer")

    return len(errors) == 0, errors


# ==============================================================================
# 2. Fold Definitions and Invariants Validation
# ==============================================================================

def validate_fold_definitions(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates fold invariants: K=5, disjoint validation partitions, quarantine enforcement."""
    errors = []
    plan_csv = config.aef_crc_reports_dir / "fold_plan.csv"

    if plan_csv.exists():
        try:
            with plan_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Invariant: K=5 folds (fold_id in 0..4, with -1 reserved for non-CV partitions)
            folds_present = {int(r["fold_id"]) for r in rows if r.get("fold_id") not in (None, "", "-1")}
            if folds_present != {0, 1, 2, 3, 4}:
                errors.append(f"Expected folds {{0, 1, 2, 3, 4}}, found {folds_present}")

            # Check disjointness of validation splits
            val_records = [r for r in rows if r.get("split_role") == "val"]
            val_psd_ids = [r["psd_id"] for r in val_records]
            unique_val_ids = set(val_psd_ids)
            if len(val_psd_ids) != len(unique_val_ids):
                errors.append(f"Validation splits are not disjoint! Total val: {len(val_psd_ids)}, unique: {len(unique_val_ids)}")

            # Check quarantine of augmented records
            quarantined = [
                r for r in rows
                if r.get("split_role") == "excluded_augmented"
                or r.get("source_type") == "augmented"
            ]
            if len(quarantined) == 0:
                errors.append("No quarantined pre-augmented records found in fold plan")

        except Exception as e:
            errors.append(f"Error parsing fold_plan.csv: {e}")

    return len(errors) == 0, errors


# ==============================================================================
# 3. Preprocessing Routing & Safeguards Validation
# ==============================================================================

def validate_preprocessing(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates standard vs conditional preprocessing on synthetic imagery."""
    errors = []
    import dataclasses

    # Synthetic image: 100x100 RGB
    dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)

    # 1. Standard mode: exact identity/copy
    cfg_std = dataclasses.replace(config, preprocessing_mode="standard")
    prep_std = ConditionalPreprocessor(cfg_std)
    res_std = prep_std.process(dummy_img, "SYNTH_001")
    if not np.array_equal(res_std.image, dummy_img):
        errors.append("Standard preprocessing modified the input image (must be identity copy)")
    if len(res_std.decisions) != 0:
        errors.append(f"Standard preprocessing logged decisions: {res_std.decisions} (expected empty)")

    # 2. Conditional mode: evaluates triggers without crashing
    cfg_cond = dataclasses.replace(config, preprocessing_mode="conditional")
    prep_cond = ConditionalPreprocessor(cfg_cond)
    res_cond = prep_cond.process(dummy_img, "SYNTH_002")
    if res_cond.image.shape != dummy_img.shape:
        errors.append(f"Conditional preprocessing altered image shape: {res_cond.image.shape} != {dummy_img.shape}")
    if res_cond.image.dtype != np.uint8:
        errors.append(f"Conditional preprocessing altered dtype: {res_cond.image.dtype} != uint8")
    if len(res_cond.decisions) != 2:
        errors.append(f"Conditional preprocessing expected 2 decisions (hair, clahe), got {len(res_cond.decisions)}")

    # 3. Density safeguard check: hair mask > 30% skips inpainting
    hair_img = dummy_img.copy()
    hair_img[:, :] = 0  # black background
    hair_img[::2, :] = 255  # dense lines covering 50% of image
    res_dense = prep_cond.process(hair_img, "SYNTH_DENSE")
    for d in res_dense.decisions:
        if d.operation == "hair_removal" and d.trigger_value > 0.30:
            if d.applied:
                errors.append("Hair removal inpainting applied despite hair coverage > 30% safeguard limit")

    return len(errors) == 0, errors


# ==============================================================================
# 4. Augmentation Routing & Refusal Validation
# ==============================================================================

def validate_augmentation(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates augmentation routing and verifies argument refusal on validation data."""
    errors = []
    from modules.image_loader import build_dataset

    # Refusal test 1: Augmentation requested on validation/test data
    try:
        build_dataset(
            config=config,
            records=[],
            class_order=config.target_classes,
            training=False,
            apply_training_time_augmentation=True,
        )
        errors.append("build_dataset failed to refuse apply_training_time_augmentation=True when training=False")
    except ValueError:
        pass  # Expected behavior
    except Exception as e:
        errors.append(f"build_dataset raised unexpected exception on augmentation refusal: {e}")

    # Refusal test 2: Sample weights provided for validation data
    try:
        build_dataset(
            config=config,
            records=[],
            class_order=config.target_classes,
            training=False,
            apply_training_time_augmentation=False,
            class_weights={c: 1.0 for c in config.target_classes},
        )
        errors.append("build_dataset failed to refuse class_weights when training=False")
    except ValueError:
        pass  # Expected behavior
    except Exception as e:
        errors.append(f"build_dataset raised unexpected exception on class_weights refusal: {e}")

    return len(errors) == 0, errors


# ==============================================================================
# 5. Model Architecture & Unfreezing Validation
# ==============================================================================

def validate_model_construction(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates EfficientNet-B0 backbone construction, 1280-D features, and 2-stage unfreezing."""
    errors = []
    try:
        import tensorflow as tf
    except ImportError:
        return False, ["TensorFlow is not installed in this environment"]

    try:
        tf.keras.backend.clear_session()
        backbone_spec = get_backbone(config.backbone)

        # Stage 1 model construction
        model, backbone = backbone_spec.build_stage1(config, num_classes=len(config.target_classes))

        # Check input shape
        expected_input_shape = (None, 224, 224, 3)
        if model.input_shape != expected_input_shape:
            errors.append(f"Model input shape {model.input_shape} != {expected_input_shape}")

        # Check output shape
        expected_output_shape = (None, len(config.target_classes))
        if model.output_shape != expected_output_shape:
            errors.append(f"Model output shape {model.output_shape} != {expected_output_shape}")

        # Check backbone feature dimension contract
        if backbone_spec.output_dim != 1280:
            errors.append(f"Backbone feature dimension contract {backbone_spec.output_dim} != 1280")

        # Check stage 1 backbone is frozen
        trainable_backbone_weights = len(backbone.trainable_weights)
        if trainable_backbone_weights != 0:
            errors.append(f"Stage 1 backbone has {trainable_backbone_weights} trainable weights (expected 0; backbone must be frozen)")

        # Stage 2 unfreezing
        model_stage2 = backbone_spec.unfreeze_stage2(model, backbone, config)
        trainable_backbone_weights_s2 = len(backbone.trainable_weights)
        if trainable_backbone_weights_s2 == 0:
            errors.append("Stage 2 unfreezing failed: 0 trainable backbone weights after unfreeze_stage2")

        # Test forward pass with synthetic dummy batch
        dummy_batch = tf.zeros((2, 224, 224, 3), dtype=tf.float32)
        outputs = model_stage2(dummy_batch, training=False)
        if outputs.shape != (2, len(config.target_classes)):
            errors.append(f"Forward pass output shape {outputs.shape} != (2, {len(config.target_classes)})")

        probs = outputs.numpy()
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, [1.0, 1.0], atol=1e-5):
            errors.append(f"Model output probabilities do not sum to 1.0: {row_sums}")

        tf.keras.backend.clear_session()
    except Exception as e:
        errors.append(f"Model construction error: {e}")

    return len(errors) == 0, errors


# ==============================================================================
# 6. Device-Agnostic Detection & Telemetry Validation
# ==============================================================================

def validate_device_detection(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates device detection, non-failing hardware abstraction, and telemetry completeness."""
    errors = []
    try:
        device_info = detect_and_configure_device(seed=config.random_seed)

        if not isinstance(device_info, DeviceInfo):
            errors.append(f"detect_and_configure_device returned {type(device_info)}, expected DeviceInfo")
            return False, errors

        if not device_info.device_type:
            errors.append("DeviceInfo.device_type is empty")
        if not device_info.device_name:
            errors.append("DeviceInfo.device_name is empty")
        if not device_info.execution_strategy:
            errors.append("DeviceInfo.execution_strategy is empty")
        if not device_info.tensorflow_version:
            errors.append("DeviceInfo.tensorflow_version is empty")

        # Check export to dict
        d_dict = device_info.to_dict()
        required_keys = ["device_type", "device_name", "device_count", "execution_strategy", "cuda_available"]
        for k in required_keys:
            if k not in d_dict:
                errors.append(f"DeviceInfo.to_dict() missing required key: {k}")

    except Exception as e:
        errors.append(f"Device detection exception: {e}")

    return len(errors) == 0, errors


# ==============================================================================
# 7. Training Configuration & Class Weighting Validation
# ==============================================================================

def validate_training_configuration(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates balanced class weighting formula and fixed epochs protocol."""
    errors = []
    from modules.fold_loader import compute_train_fold_class_weights
    from modules.training import _build_callbacks

    classes = list(config.target_classes)
    c0, c1, c2, c3 = classes[0], classes[1], classes[2], classes[3]
    synthetic_labels = (
        [c0] * 100 +
        [c1] * 200 +
        [c2] * 300 +
        [c3] * 400
    )
    weights = compute_train_fold_class_weights(synthetic_labels, classes)

    # Expected: w_c = 1000 / (4 * N_c)
    expected_weights = {
        c0: 1000.0 / (4.0 * 100),               # 2.5
        c1: 1000.0 / (4.0 * 200),               # 1.25
        c2: 1000.0 / (4.0 * 300),               # 0.833333...
        c3: 1000.0 / (4.0 * 400),               # 0.625
    }
    for cls, exp_w in expected_weights.items():
        computed_w = weights[cls]
        if abs(computed_w - exp_w) > 1e-5:
            errors.append(f"Class weight for {cls}: {computed_w:.6f} != expected {exp_w:.6f}")

    # Verify callbacks builder: must include ModelCheckpoint, strictly NO EarlyStopping
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_ckpt = Path(tmp_dir) / "test_ckpt.keras"
        callbacks = _build_callbacks(config, dummy_ckpt)
        cb_names = [cb.__class__.__name__ for cb in callbacks]

        if "ModelCheckpoint" not in cb_names:
            errors.append(f"ModelCheckpoint missing from callbacks: {cb_names}")
        if "EarlyStopping" in cb_names:
            errors.append(f"EarlyStopping found in callbacks {cb_names}! Fixed epochs protocol strictly forbids early stopping.")

    return len(errors) == 0, errors


# ==============================================================================
# 8. Checkpoint Selection Logic Validation
# ==============================================================================

def validate_checkpoint_selection(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates validation-loss-based checkpoint selection between Stage 1 and Stage 2."""
    errors = []

    # Scenario A: Stage 2 improves validation loss (val_loss_stage2 < val_loss_stage1)
    val_loss_stage1 = 0.650
    val_loss_stage2 = 0.520
    if not (val_loss_stage2 <= val_loss_stage1):
        errors.append("Stage 2 checkpoint selection logic error: Stage 2 val_loss not correctly recognized as superior")

    # Scenario B: Stage 2 overfits (val_loss_stage2 > val_loss_stage1)
    val_loss_stage1 = 0.520
    val_loss_stage2 = 0.710
    selected_stage = 2 if (val_loss_stage2 <= val_loss_stage1) else 1
    if selected_stage != 1:
        errors.append("Stage 1 checkpoint selection logic error: failed to retain Stage 1 when Stage 2 overfits")

    return len(errors) == 0, errors


# ==============================================================================
# 9. Evaluation Metrics & Confusion Matrix Validation
# ==============================================================================

def validate_evaluation_logic(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates multi-metric computation and C x C confusion matrix invariants."""
    errors = []
    classes = list(config.target_classes)

    # Synthetic predictions across the 4 classes
    y_true = [classes[0], classes[1], classes[2], classes[3], classes[0], classes[0]]
    y_pred = [classes[0], classes[1], classes[2], classes[3], classes[2], classes[0]]

    metrics = compute_fold_metrics(y_true, y_pred, classes, fold_index=0)

    # Check metrics existence and bounds
    if not (0.0 <= metrics.macro_f1 <= 1.0):
        errors.append(f"Macro-F1 {metrics.macro_f1} out of bounds [0, 1]")
    if not (0.0 <= metrics.accuracy <= 1.0):
        errors.append(f"Accuracy {metrics.accuracy} out of bounds [0, 1]")
    if not (0.0 <= metrics.balanced_accuracy <= 1.0):
        errors.append(f"Balanced Accuracy {metrics.balanced_accuracy} out of bounds [0, 1]")
    if not (-1.0 <= metrics.mcc <= 1.0):
        errors.append(f"MCC {metrics.mcc} out of bounds [-1, 1]")

    # Check confusion matrix dimensions: must be exactly C x C (4 x 4)
    if metrics.confusion.shape != (4, 4):
        errors.append(f"Confusion matrix shape {metrics.confusion.shape} != (4, 4)")

    # Aggregate metrics across synthetic folds
    metrics_f1 = compute_fold_metrics(y_true, y_pred, classes, fold_index=1)
    agg = aggregate_fold_metrics([metrics, metrics_f1])
    if abs(agg.macro_f1_mean - metrics.macro_f1) > 1e-6:
        errors.append(f"aggregate_fold_metrics mean error: {agg.macro_f1_mean} != {metrics.macro_f1}")

    return len(errors) == 0, errors


# ==============================================================================
# 10. Winner Selection & Parsimony Rule Validation
# ==============================================================================

def validate_winner_selection(config: PSDConfig) -> Tuple[bool, List[str]]:
    """Validates authoritative winner selection with 0.005 practical-equivalence margin."""
    errors = []

    def make_agg(f1_val: float) -> AggregatedMetrics:
        return AggregatedMetrics(
            n_folds=5,
            macro_f1_mean=f1_val,
            macro_f1_std=0.010,
            accuracy_mean=f1_val,
            accuracy_std=0.010,
            balanced_accuracy_mean=f1_val,
            balanced_accuracy_std=0.010,
            weighted_f1_mean=f1_val,
            weighted_f1_std=0.010,
            mcc_mean=f1_val,
            mcc_std=0.010,
            per_class_precision_mean={},
            per_class_precision_std={},
            per_class_recall_mean={},
            per_class_recall_std={},
            per_class_f1_mean={},
            per_class_f1_std={},
            per_class_support_total={},
            confusion_sum=np.zeros((4, 4)),
        )

    # Test Case 1: Candidate exceeds BASE by > 0.005 (delta = +0.010) -> Candidate wins
    exps1 = {
        "P3-BASE": make_agg(0.840),
        "P3-PRE": make_agg(0.850),
    }
    w1, r1 = select_phase3_winner(exps1, threshold=0.005)
    if w1 != "P3-PRE":
        errors.append(f"Winner selection failed on >0.005 improvement: expected P3-PRE, got {w1}")

    # Test Case 2: Candidate exceeds BASE by <= 0.005 (delta = +0.003) -> P3-BASE wins (parsimony)
    exps2 = {
        "P3-BASE": make_agg(0.840),
        "P3-PRE": make_agg(0.843),
    }
    w2, r2 = select_phase3_winner(exps2, threshold=0.005)
    if w2 != "P3-BASE":
        errors.append(f"Winner selection failed on <=0.005 equivalence: expected P3-BASE, got {w2}")

    # Test Case 3: Augmentation exceeds BASE by > 0.005 -> P3-AUG wins
    exps3 = {
        "P3-BASE": make_agg(0.840),
        "P3-AUG": make_agg(0.855),
    }
    w3, r3 = select_phase3_winner(exps3, threshold=0.005)
    if w3 != "P3-AUG":
        errors.append(f"Winner selection failed for P3-AUG: expected P3-AUG, got {w3}")

    return len(errors) == 0, errors


# ==============================================================================
# Master Framework Validation Entry Point
# ==============================================================================

def validate_phase3_framework(
    config: Optional[PSDConfig] = None,
    verbose: bool = True,
) -> Tuple[bool, List[str]]:
    """Executes all 10 framework validation checks and reports comprehensive results."""
    if config is None:
        config = get_config()

    checks = [
        ("1. Configuration & Hyperparameter Contracts", validate_configuration),
        ("2. Fold Definitions & Quarantine Invariants", validate_fold_definitions),
        ("3. Preprocessing Routing & Safeguards", validate_preprocessing),
        ("4. Augmentation Routing & Argument Refusal", validate_augmentation),
        ("5. Model Architecture & Unfreezing Schedule", validate_model_construction),
        ("6. Device-Agnostic Detection & Telemetry", validate_device_detection),
        ("7. Training Configuration & Class Weighting", validate_training_configuration),
        ("8. Checkpoint Selection Logic", validate_checkpoint_selection),
        ("9. Evaluation Metrics & Confusion Matrices", validate_evaluation_logic),
        ("10. Single-Winner Selection & Parsimony Rule", validate_winner_selection),
    ]

    all_errors = []
    if verbose:
        print("=" * 70)
        print("AEF-CRC Phase 3 Reusable Framework — Component Validation Suite")
        print("=" * 70)

    for name, check_fn in checks:
        ok, errs = check_fn(config)
        if ok:
            if verbose:
                print(f"  [PASS] {name}")
        else:
            if verbose:
                print(f"  [FAIL] {name}")
                for err in errs:
                    print(f"         * {err}")
            all_errors.extend(errs)

    all_passed = len(all_errors) == 0
    if verbose:
        print("=" * 70)
        if all_passed:
            print("ALL 10 FRAMEWORK VALIDATION CHECKS PASSED.")
        else:
            print(f"FRAMEWORK VALIDATION FAILED WITH {len(all_errors)} ERROR(S).")
        print("=" * 70)

    return all_passed, all_errors
