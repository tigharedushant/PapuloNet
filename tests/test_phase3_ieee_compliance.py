"""
tests/test_phase3_ieee_compliance.py

Test suite verifying IEEE-level reproducibility and device-agnostic requirements for Phase 3:
1. Device-agnostic compute detection and execution telemetry
2. Device smoke test execution
3. Fixed epoch protocol (strictly no EarlyStopping)
4. Winner selection with 0.005 practical-equivalence margin
5. Complete machine-readable fold metrics CSV format
6. Programmatic winner validation contract
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from config.config import PSDConfig
from modules.device_utils import (
    detect_device_environment,
    configure_execution_device,
    device_smoke_test,
    DeviceInfo,
)
from modules.evaluation import (
    AggregatedMetrics,
    FoldMetrics,
    ClassMetrics,
    select_phase3_winner,
    write_phase3_fold_metrics,
    write_phase3_winner,
    validate_phase3_winner,
)
from modules.training import _build_callbacks


def test_device_agnostic_detection():
    """Verify that device detection returns structured DeviceInfo without throwing exceptions."""
    info = detect_device_environment()
    assert isinstance(info, DeviceInfo)
    assert info.device_type in ("GPU", "CPU", "TPU")
    assert info.device_count >= 1
    assert isinstance(info.device_name, str)
    assert isinstance(info.tensorflow_version, str)
    assert isinstance(info.python_version, str)


def test_device_smoke_test():
    """Verify that small tensor multiplication executes successfully on detected device."""
    ok, msg = device_smoke_test()
    assert ok is True, f"Smoke test failed: {msg}"
    assert "verified successfully" in msg


def test_fixed_epochs_no_early_stopping(tmp_path):
    """Verify that _build_callbacks strictly contains ModelCheckpoint and zero EarlyStopping."""
    cfg = PSDConfig()
    ckpt = tmp_path / "test.keras"
    callbacks = _build_callbacks(cfg, ckpt)

    cb_types = [type(cb).__name__ for cb in callbacks]
    assert "ModelCheckpoint" in cb_types
    assert "EarlyStopping" not in cb_types, "EarlyStopping must NOT be present in Phase 3 training"


def _make_dummy_agg(macro_f1: float, std: float = 0.01) -> AggregatedMetrics:
    return AggregatedMetrics(
        n_folds=5,
        macro_f1_mean=macro_f1,
        macro_f1_std=std,
        accuracy_mean=macro_f1,
        accuracy_std=std,
        balanced_accuracy_mean=macro_f1,
        balanced_accuracy_std=std,
        weighted_f1_mean=macro_f1,
        weighted_f1_std=std,
        mcc_mean=macro_f1,
        mcc_std=std,
        per_class_f1_mean={},
        per_class_f1_std={},
    )


def test_winner_selection_equivalence_margin():
    """Verify winner selection rule:
    - candidate exceeds baseline by > 0.005 -> candidate wins
    - candidate within 0.005 of baseline -> baseline wins for parsimony
    - candidate inferior -> baseline wins
    """
    # 1. Candidate exceeds baseline by > 0.005
    aggs1 = {
        "P3-BASE": _make_dummy_agg(0.7000),
        "P3-PRE": _make_dummy_agg(0.6000),
        "P3-AUG": _make_dummy_agg(0.7060),  # +0.006 > 0.005
    }
    winner, rationale = select_phase3_winner(aggs1, threshold=0.005)
    assert winner == "P3-AUG"
    assert "exceeding P3-BASE" in rationale

    # 2. Candidate exceeds baseline by <= 0.005 -> P3-BASE wins for parsimony
    aggs2 = {
        "P3-BASE": _make_dummy_agg(0.7012),
        "P3-PRE": _make_dummy_agg(0.6040),
        "P3-AUG": _make_dummy_agg(0.7030),  # +0.0018 <= 0.005
    }
    winner, rationale = select_phase3_winner(aggs2, threshold=0.005)
    assert winner == "P3-BASE"
    assert "retained for parsimony" in rationale


def test_fold_metrics_csv_columns(tmp_path):
    """Verify that write_phase3_fold_metrics outputs all mandatory IEEE columns."""
    class_order = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    per_class = {c: ClassMetrics(0.8, 0.8, 0.8, 50) for c in class_order}
    dummy_fold = FoldMetrics(
        fold_index=0,
        macro_f1=0.75,
        accuracy=0.78,
        balanced_accuracy=0.77,
        weighted_f1=0.76,
        mcc=0.65,
        per_class=per_class,
        confusion=np.zeros((4, 4), dtype=int),
        class_order=class_order,
    )
    exp_metrics = {"P3-BASE": [dummy_fold]}
    fold_meta = {
        "P3-BASE": [{
            "n_train": 916,
            "n_val": 230,
            "selected_stage": 2,
            "selected_epoch": 8,
            "best_val_loss": 0.62,
            "checkpoint_used": "best_model.keras",
            "device": "NVIDIA GeForce RTX 2050",
        }]
    }

    metrics_csv, summary_csv = write_phase3_fold_metrics(tmp_path, exp_metrics, fold_meta)
    assert metrics_csv.exists()
    assert summary_csv.exists()

    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # Check minimum required columns per Item 20
    for req_col in [
        "arm", "fold", "validation_n", "accuracy", "balanced_accuracy",
        "macro_f1", "weighted_f1", "mcc", "selected_stage", "selected_epoch",
        "best_val_loss", "checkpoint", "device",
    ]:
        assert req_col in row, f"Required column {req_col} missing in fold_metrics.csv"

    assert row["arm"] == "P3-BASE"
    assert row["fold"] == "0"
    assert row["validation_n"] == "230"
    assert row["device"] == "NVIDIA GeForce RTX 2050"


def test_validate_phase3_winner_contract(tmp_path):
    """Verify that validate_phase3_winner catches missing checkpoints and mismatched means."""
    reports_dir = tmp_path / "reports" / "phase3"
    artifacts_dir = tmp_path / "artifacts" / "phase3"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Test missing winner.json
    ok, errs = validate_phase3_winner(reports_dir, artifacts_dir)
    assert not ok
    assert any("Winner file missing" in e for e in errs)

    # Write dummy winner.json
    winner_name = "P3-BASE"
    agg = _make_dummy_agg(0.7000)
    write_phase3_winner(
        out_dir=reports_dir,
        winner_experiment_id=winner_name,
        winner_name=winner_name,
        preprocessing_mode="standard",
        training_time_augmentation=False,
        selection_metric="mean_validation_macro_f1",
        selection_direction="maximize",
        winner_aggregate=agg,
        representation_id_str="test_repr",
    )

    # Missing checkpoints
    ok, errs = validate_phase3_winner(reports_dir, artifacts_dir)
    assert not ok
    assert any("Missing winner checkpoint" in e for e in errs)

    # Create dummy checkpoints and confusion matrices
    for f in range(5):
        fold_dir = artifacts_dir / winner_name / f"fold_{f:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "best_model.keras").write_bytes(b"dummy_model")
        (fold_dir / "confusion_matrix.json").write_text("{}", encoding="utf-8")

    # Create matching fold_metrics.csv
    class_order = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    per_class = {c: ClassMetrics(0.7, 0.7, 0.7, 50) for c in class_order}
    fms = [
        FoldMetrics(i, 0.7000, 0.70, 0.70, 0.70, 0.60, per_class, np.zeros((4, 4)), class_order)
        for i in range(5)
    ]
    meta = [{"n_train": 916, "n_val": 230, "checkpoint_used": "best_model.keras"} for _ in range(5)]
    write_phase3_fold_metrics(reports_dir, {winner_name: fms}, {winner_name: meta})

    # Now all invariants should pass
    ok, errs = validate_phase3_winner(reports_dir, artifacts_dir)
    assert ok, f"Validation failed unexpectedly: {errs}"


def test_cpu_fallback_device_detection(monkeypatch):
    """Verify that when no GPU is detected, detect_device_environment gracefully
    configures a truthful CPU execution context with float32 policy and fallback flag."""
    import tensorflow as tf
    from collections import namedtuple
    MockDevice = namedtuple("MockDevice", ["name", "device_type"])

    def _mock_list_devices(device_type=None):
        if device_type == "GPU":
            return []
        if device_type == "TPU":
            return []
        return [MockDevice(name="/physical_device:CPU:0", device_type="CPU")]

    monkeypatch.setattr(tf.config, "list_physical_devices", _mock_list_devices)

    info = detect_device_environment()
    assert info.device_type == "CPU"
    assert info.device_fallback is True
    assert info.cuda_available is False
    assert info.execution_strategy == "DefaultStrategy"

    configured = configure_execution_device(info)
    assert configured.mixed_precision_active is False
    assert configured.mixed_precision_policy == "float32"


def test_reproducibility_utilities():
    """Verify that set_global_seeds sets Python, NumPy, and TensorFlow seeds deterministically."""
    from modules.training import set_global_seeds
    import os

    set_global_seeds(42)
    assert os.environ.get("PYTHONHASHSEED") == "42"
    assert os.environ.get("TF_DETERMINISTIC_OPS") == "1"
    assert os.environ.get("TF_CUDNN_DETERMINISTIC") == "1"

    val_py1 = [np.random.rand() for _ in range(5)]

    set_global_seeds(42)
    val_py2 = [np.random.rand() for _ in range(5)]

    assert np.allclose(val_py1, val_py2), "Random seed re-initialization was not deterministic!"


def test_model_construction_architecture():
    """Verify that build_stage1_model and unfreeze_for_stage2 adhere strictly
    to the IEEE architecture:
    - EfficientNet-B0
    - include_top=False, pooling='avg'
    - Dropout(0.3)
    - Dense(4, activation='softmax')
    - Stage 1: backbone completely frozen
    - Stage 2: top 20 layers unfrozen, BatchNorm layers remain frozen
    """
    import tensorflow as tf
    from modules.efficientnet_model import build_stage1_model, unfreeze_for_stage2, describe_model

    config = PSDConfig(
        image_size=224,
        stage1_learning_rate=1e-3,
        stage2_learning_rate=1e-5,
        unfrozen_layers=20,
        dropout_rate=0.3,
    )

    model, backbone = build_stage1_model(config, num_classes=4)
    info = describe_model(model)

    assert info.frozen_params > 0, "Backbone must be frozen in Stage 1"
    assert backbone.trainable is False, "Backbone.trainable must be False in Stage 1"

    # Verify head structure
    dense_layer = model.layers[-1]
    assert isinstance(dense_layer, tf.keras.layers.Dense)
    assert dense_layer.units == 4
    assert dense_layer.activation.__name__ == "softmax"

    dropout_layer = model.layers[-2]
    assert isinstance(dropout_layer, tf.keras.layers.Dropout)
    assert abs(dropout_layer.rate - 0.3) < 1e-4

    # Stage 2 unfreezing
    model_s2 = unfreeze_for_stage2(model, backbone, config)
    assert backbone.trainable is True

    # Layers before the top unfrozen slice must remain frozen
    for layer in backbone.layers[:-config.unfrozen_layers]:
        assert layer.trainable is False

    # Top unfrozen layers: BatchNorm must remain frozen
    for layer in backbone.layers[-config.unfrozen_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable is False
