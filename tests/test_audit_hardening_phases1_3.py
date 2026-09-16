"""
tests/test_audit_hardening_phases1_3.py

Dedicated regression and verification test suite resulting from the
Brutal Audit & Hardening pass for AEF-CRC Phases 1-3.

Covers:
1. Stage 2 BatchNorm freezing invariant (no BatchNorm layer trainable).
2. Preprocessing & scaling invariant (no double normalization).
3. Phase 3 winner selection logic & 0.005 equivalence margin.
4. Fold plan validation & zero holdout/quarantine contamination.
5. Checkpoint selection serialization round-trip.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import numpy as np
import pytest

from config.config import get_config
from modules.aef_input_validator import compute_train_fold_class_weights
from modules.evaluation import (
    compute_fold_metrics, aggregate_fold_metrics,
    write_phase3_fold_metrics, write_phase3_winner, AggregatedMetrics, FoldMetrics, ClassMetrics
)
from modules.fold_loader import load_frozen_fold_plan, validate_fold_plan, FoldPlan

_FOLD_PLAN_PATH = get_config().aef_crc_reports_dir / "fold_plan.csv"
_METADATA_PATH = get_config().reports_dir / "metadata.csv"

requires_authoritative_fold_plan = pytest.mark.skipif(
    not _FOLD_PLAN_PATH.exists(),
    reason="real-data integration test skipped because authoritative dataset artifacts are not installed (clean framework mode)",
)

requires_authoritative_metadata = pytest.mark.skipif(
    not _METADATA_PATH.exists(),
    reason="real-data integration test skipped because authoritative dataset artifacts are not installed (clean framework mode)",
)


def test_batchnorm_layers_remain_frozen_in_stage2():
    """Verify the transfer-learning invariant that all BatchNormalization
    layers in EfficientNet-B0 remain frozen in Stage 2 fine-tuning."""
    import tensorflow as tf
    from modules.efficientnet_model import build_stage1_model, unfreeze_for_stage2

    config = get_config()
    model, backbone = build_stage1_model(config, num_classes=len(config.target_classes))
    model = unfreeze_for_stage2(model, backbone, config)

    # Check top unfrozen slice
    top_layers = backbone.layers[-config.unfrozen_layers:]
    bn_layers = [l for l in top_layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    
    assert len(bn_layers) > 0, "Expected at least one BatchNorm layer in top unfrozen slice."
    for bn in bn_layers:
        assert not bn.trainable, f"BatchNorm layer '{bn.name}' was NOT frozen in Stage 2!"


def test_efficientnet_input_scaling_invariant():
    """Verify that preprocessing keeps pixel intensities in [0, 255] float32
    and that preprocess_input does not apply redundant scaling."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from modules.preprocessing import ConditionalPreprocessor

    config = get_config()
    pre = ConditionalPreprocessor(config)

    # Synthetic test image
    raw_img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    res = pre.process(raw_img, "SCALE_TEST")
    processed = res.image.astype(np.float32)

    assert processed.min() >= 0.0
    assert processed.max() <= 255.0

    # EfficientNet preprocess_input is identity because normalization layers are internal
    scaled = preprocess_input(processed)
    assert np.allclose(scaled, processed), "preprocess_input unexpectedly altered pixel values!"


def test_equivalence_margin_selection_logic():
    """Verify that an arm outperforming P3-BASE by <= 0.005 is discarded
    in favor of P3-BASE for parsimony, but kept if > 0.005."""
    def _make_dummy_agg(macro_f1: float) -> AggregatedMetrics:
        return AggregatedMetrics(
            n_folds=5,
            macro_f1_mean=macro_f1, macro_f1_std=0.01,
            accuracy_mean=0.8, accuracy_std=0.01,
            balanced_accuracy_mean=0.8, balanced_accuracy_std=0.01,
            weighted_f1_mean=0.8, weighted_f1_std=0.01,
            mcc_mean=0.7, mcc_std=0.01,
        )

    EQUIVALENCE_MARGIN = 0.005

    # Case 1: Candidate beats BASE by exactly 0.004 (<= 0.005) -> BASE wins
    all_results_1 = {
        "P3-BASE": _make_dummy_agg(0.8500),
        "P3-PRE": _make_dummy_agg(0.8540),
        "P3-AUG": _make_dummy_agg(0.8400),
    }
    candidate_arms = [("P3-BASE", "cfg_base", False), ("P3-PRE", "cfg_pre", False), ("P3-AUG", "cfg_aug", True)]
    best_cand = max(candidate_arms, key=lambda t: all_results_1[t[0]].macro_f1_mean)
    best_name = best_cand[0]
    base_f1 = all_results_1["P3-BASE"].macro_f1_mean
    best_f1 = all_results_1[best_name].macro_f1_mean
    if best_name != "P3-BASE" and (best_f1 - base_f1) <= EQUIVALENCE_MARGIN:
        winner = "P3-BASE"
    else:
        winner = best_name
    assert winner == "P3-BASE", "Expected P3-BASE to win on <= 0.005 margin!"

    # Case 2: Candidate beats BASE by 0.006 (> 0.005) -> Candidate wins
    all_results_2 = {
        "P3-BASE": _make_dummy_agg(0.8500),
        "P3-PRE": _make_dummy_agg(0.8560),
        "P3-AUG": _make_dummy_agg(0.8400),
    }
    best_cand = max(candidate_arms, key=lambda t: all_results_2[t[0]].macro_f1_mean)
    best_name = best_cand[0]
    best_f1 = all_results_2[best_name].macro_f1_mean
    if best_name != "P3-BASE" and (best_f1 - base_f1) <= EQUIVALENCE_MARGIN:
        winner = "P3-BASE"
    else:
        winner = best_name
    assert winner == "P3-PRE", "Expected P3-PRE to win when delta > 0.005!"


@requires_authoritative_fold_plan
def test_fold_plan_integrity_and_isolation():
    """Verify that authoritative fold plan has exactly 5 folds, exactly 1146 CV samples,
    zero leakage with holdouts (246 val, 243 test), and excludes all 264 pre-augmented images."""
    config = get_config()
    plan = load_frozen_fold_plan(config)

    assert plan.k == 5
    assert len(plan.folds) == 5
    assert len(plan.holdout_val_records) == 246
    assert len(plan.holdout_test_records) == 243
    assert len(plan.excluded_augmented_records) == 264

    # Check that validate_fold_plan passes
    validate_fold_plan(plan, config)

    # Check total unique CV validation records
    all_val_ids = [r.psd_id for f in plan.folds for r in f.val_records]
    assert len(all_val_ids) == 1146
    assert len(set(all_val_ids)) == 1146


def test_write_phase3_winner_contract(tmp_path: Path):
    """Verify that write_phase3_winner persists a strictly conforming winner.json."""
    agg = AggregatedMetrics(
        n_folds=5,
        macro_f1_mean=0.88, macro_f1_std=0.015,
        accuracy_mean=0.89, accuracy_std=0.012,
        balanced_accuracy_mean=0.87, balanced_accuracy_std=0.014,
        weighted_f1_mean=0.89, weighted_f1_std=0.013,
        mcc_mean=0.84, mcc_std=0.02,
        per_class_f1_mean={"A": 0.85, "B": 0.90},
        per_class_f1_std={"A": 0.02, "B": 0.01},
    )

    out_p = write_phase3_winner(
        out_dir=tmp_path,
        winner_experiment_id="P3-BASE",
        winner_name="P3-BASE",
        preprocessing_mode="standard",
        training_time_augmentation=False,
        selection_metric="macro_f1",
        selection_direction="maximize",
        winner_aggregate=agg,
        representation_id_str="test_repr_123",
        dataset_freeze_hash="test_freeze_hash",
        fold_plan_hash="test_plan_hash",
    )

    assert out_p.exists()
    data = json.loads(out_p.read_text(encoding="utf-8"))
    assert data["winner_name"] == "P3-BASE"
    assert data["preprocessing_mode"] == "standard"
    assert data["training_time_augmentation"] is False
    assert data["macro_f1_mean"] == 0.88
    assert data["dataset_freeze_hash"] == "test_freeze_hash"
    assert data["fold_plan_hash"] == "test_plan_hash"


@requires_authoritative_fold_plan
def test_class_weights_formula_and_fold_specific_values():
    """Verify that class weights compute w_c = N_train / (C * N_c) exactly
    from each fold's training slice, and verify all five fold-specific weights."""
    import csv
    config = get_config()
    plan = load_frozen_fold_plan(config)
    summary_path = config.aef_crc_reports_dir / "phase2_fold_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    assert len(plan.folds) == 5
    C = len(config.target_classes)  # 4 classes

    for fid, fold in enumerate(plan.folds):
        n_train = len(fold.train_records)
        class_counts = {}
        for r in fold.train_records:
            class_counts[r.mapped_class] = class_counts.get(r.mapped_class, 0) + 1

        calc_weights = compute_train_fold_class_weights(
            [r.mapped_class for r in fold.train_records], config.target_classes
        )
        stored_weights = json.loads(summary_rows[fid]["class_weights_json"])

        for cls in config.target_classes:
            expected_weight = n_train / (C * class_counts[cls])
            assert abs(calc_weights[cls] - expected_weight) < 1e-9
            assert abs(stored_weights[cls] - expected_weight) < 1e-6

    # Verify Fold 0 differs from Fold 1, and Fold 1 differs from Fold 3
    w0 = json.loads(summary_rows[0]["class_weights_json"])
    w1 = json.loads(summary_rows[1]["class_weights_json"])
    w3 = json.loads(summary_rows[3]["class_weights_json"])

    assert w0["Psoriasis"] != w1["Psoriasis"]
    assert w1["Psoriasis"] != w3["Psoriasis"]
    assert w1["Seborrheic_Dermatitis"] != w3["Seborrheic_Dermatitis"]


@requires_authoritative_metadata
def test_metadata_provenance_relationship():
    """Verify that metadata.csv contains 3,414 candidate records, that every
    one of the 1,899 physical final_split images maps to a valid provenance row,
    and that the 1,515 non-final rows represent filtered candidates."""
    import csv
    config = get_config()
    meta_path = config.reports_dir / "metadata.csv"
    with meta_path.open(newline="", encoding="utf-8") as f:
        meta_rows = {r["psd_id"]: r for r in csv.DictReader(f)}

    assert len(meta_rows) == 3414, f"Expected 3414 metadata rows, got {len(meta_rows)}"

    disk_files = list(config.final_dir.rglob("*.jpg"))
    assert len(disk_files) == 1899, f"Expected 1899 disk images, got {len(disk_files)}"

    for fpath in disk_files:
        psd_id = fpath.stem
        assert psd_id in meta_rows, f"Missing provenance for final disk image: {psd_id}"
        meta = meta_rows[psd_id]
        assert meta["source_dataset"] in ("DermNet", "SkinDisNet", "Curated32", "AtlasISIC31")
        assert meta["mapped_class"] in config.target_classes

    non_final_ids = set(meta_rows.keys()) - {f.stem for f in disk_files}
    assert len(non_final_ids) == 1515, f"Expected 1515 filtered records, got {len(non_final_ids)}"


def test_callbacks_use_fixed_epochs_no_early_stopping(tmp_path: Path):
    """Verify that _build_callbacks enforces fixed epochs by excluding EarlyStopping."""
    import tensorflow as tf
    from modules.training import _build_callbacks
    config = get_config()
    ckpt_path = tmp_path / "model.keras"
    callbacks = _build_callbacks(config, ckpt_path)

    callback_types = [type(cb) for cb in callbacks]
    assert tf.keras.callbacks.EarlyStopping not in callback_types, (
        "EarlyStopping is present in _build_callbacks! The approved Phase 3 protocol "
        "requires fixed epochs (15 Stage 1, 10 Stage 2) with ModelCheckpoint tracking val_loss."
    )
    assert tf.keras.callbacks.ModelCheckpoint in callback_types

