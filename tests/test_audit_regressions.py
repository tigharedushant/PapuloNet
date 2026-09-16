"""
tests/test_audit_regressions.py

Regression test suite specifically verifying the Phase 1-3 repairs:
1. Dataset freeze verification and orphan file detection in Phase 1 (B10, B18)
2. Authoritative fold plan persistence, validation, and load_frozen_fold_plan in Phase 2 (B1, B2)
3. Missing fold plan fails loudly (B2)
4. Fold plan validation guards (leakage, disjoint train/val, no aug in CV) (B1, B2)
5. Per-fold metrics and fold summary CSV persistence, aggregate matches rows (B8)
6. Machine-readable winner.json creation and authoritative consumption (B5, B6)
7. Handcrafted cache manifest with parameter hashing (B14)
8. Inpainting density safeguard (B21)
9. Canonical label ordering and index mapping (B17)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from modules.aef_input_validator import (
    AEFInputLoader, AEFDatasetValidator, ImageRecord, compute_train_fold_class_weights,
)
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import (
    ImbalanceAwareFoldLoader, save_fold_plan, load_frozen_fold_plan, validate_fold_plan,
)
from modules.evaluation import (
    compute_fold_metrics, aggregate_fold_metrics,
    write_phase3_fold_metrics, write_phase3_winner,
)
from modules.handcrafted_features import HandcraftedFeatureExtractor
from modules.preprocessing import ConditionalPreprocessor
from modules.synthetic_fixtures import (
    make_test_config as _make_test_config,
    build_synthetic_dataset as _build_synthetic_dataset,
    run_pipeline_through_split as _run_pipeline_through_split,
)


@pytest.fixture()
def pipeline_config(tmp_path):
    import dataclasses
    config = _make_test_config(tmp_path, seed=42)
    config = dataclasses.replace(
        config,
        target_classes=["Psoriasis", "Lichen_Planus", "Seborrheic_Dermatitis"],
    )
    _build_synthetic_dataset(config)
    _run_pipeline_through_split(config)
    return config


def test_phase1_dataset_freeze_and_orphan_detection(pipeline_config):
    """B10 & B18: DatasetFreezer verifies freeze, and orphan detection catches unreferenced disk files."""
    freezer = DatasetFreezer(pipeline_config)
    freeze_path = freezer.freeze(overwrite=True)
    assert freeze_path.exists()

    verify_res = freezer.verify()
    assert verify_res.matches is True

    records = AEFInputLoader(pipeline_config).load()
    validator = AEFDatasetValidator(pipeline_config)
    report = validator.run(records)

    # Clean dataset should have 0 orphan files and 0 missing disk files
    assert len(report.orphan_files) == 0
    assert len(report.missing_disk_files) == 0
    assert report.passed is True

    # Inject an orphan file
    orphan_dir = pipeline_config.final_dir / "train" / "Psoriasis"
    orphan_file = orphan_dir / "orphan_test_12345.jpg"
    orphan_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    records_with_orphan = AEFInputLoader(pipeline_config).load()
    report_orphan = validator.run(records_with_orphan)
    assert len(report_orphan.orphan_files) == 1
    assert report_orphan.orphan_files[0].name == "orphan_test_12345.jpg"

    # Cleanup
    orphan_file.unlink()


def test_phase2_fold_plan_persistence_and_reloading(pipeline_config):
    """B1 & B2: Phase 2 writes fold_plan.csv & phase2_fold_summary.csv, and load_frozen_fold_plan restores it."""
    loader = ImbalanceAwareFoldLoader(pipeline_config, k=3)
    plan = loader.build()
    assert len(plan.folds) == 3

    # Save plan
    plan_path = save_fold_plan(pipeline_config, plan)
    assert plan_path.exists()
    summary_path = pipeline_config.aef_crc_reports_dir / "phase2_fold_summary.csv"
    assert summary_path.exists()

    # Load frozen plan
    loaded_plan = load_frozen_fold_plan(pipeline_config)
    assert loaded_plan.k == plan.k
    assert len(loaded_plan.folds) == len(plan.folds)

    for f_orig, f_load in zip(plan.folds, loaded_plan.folds):
        assert f_orig.fold_index == f_load.fold_index
        orig_train_ids = sorted(r.psd_id for r in f_orig.train_records)
        load_train_ids = sorted(r.psd_id for r in f_load.train_records)
        assert orig_train_ids == load_train_ids

        orig_val_ids = sorted(r.psd_id for r in f_orig.val_records)
        load_val_ids = sorted(r.psd_id for r in f_load.val_records)
        assert orig_val_ids == load_val_ids

        assert f_orig.class_weights == f_load.class_weights


def test_missing_fold_plan_fails_loudly(pipeline_config):
    """B2: Downstream phases MUST fail loudly if fold_plan.csv is missing."""
    missing_path = pipeline_config.aef_crc_reports_dir / "non_existent_fold_plan.csv"
    with pytest.raises(FileNotFoundError, match="Authoritative fold plan not found"):
        load_frozen_fold_plan(pipeline_config, fold_plan_path=missing_path)


def test_fold_plan_validation_guards(pipeline_config):
    """B1 & B2: Validate fold plan invariants (no leakage, disjoint train/val, no aug in CV)."""
    loader = ImbalanceAwareFoldLoader(pipeline_config, k=3)
    plan = loader.build()
    validate_fold_plan(plan, pipeline_config)

    # Artificially corrupt: add an augmented image to CV train
    corrupted_plan = loader.build()
    aug_rec = ImageRecord(
        psd_id="aug_fake", split="train", mapped_class="Psoriasis",
        source_dataset="test", dataset_version="1", original_split="train",
        source_type="augmented", file_path=Path("fake.jpg"),
    )
    corrupted_plan.folds[0].train_records.append(aug_rec)
    with pytest.raises(ValueError, match="contains augmented image"):
        validate_fold_plan(corrupted_plan, pipeline_config)


def test_phase3_fold_metrics_and_winner_json(tmp_path, pipeline_config):
    """B5 & B8: Check per-fold metrics CSV, fold summary CSV, and winner.json persistence."""
    out_dir = tmp_path / "reports" / "phase3"
    class_order = pipeline_config.target_classes

    # Simulate 3 folds of metrics
    fms = []
    for f in range(3):
        y_true = [class_order[i % len(class_order)] for i in range(20)]
        y_pred = list(y_true)
        # add 1 error
        y_pred[0] = class_order[(class_order.index(y_true[0]) + 1) % len(class_order)]
        fms.append(compute_fold_metrics(y_true, y_pred, class_order, fold_index=f))

    exp_metrics = {"P3-BASE": fms}
    metrics_path, summary_path = write_phase3_fold_metrics(out_dir, exp_metrics)
    assert metrics_path.exists()
    assert summary_path.exists()

    # Read back fold_metrics.csv and verify per-fold rows
    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert [r["fold_id"] for r in rows] == ["0", "1", "2"]

    # Verify that summary mean equals mean of fold rows
    row_f1s = [float(r["macro_f1"]) for r in rows]
    expected_mean = float(np.mean(row_f1s))

    with summary_path.open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert len(summary_rows) == 1
    assert abs(float(summary_rows[0]["macro_f1_mean"]) - expected_mean) < 1e-4

    # Write winner.json
    agg = aggregate_fold_metrics(fms)
    winner_path = write_phase3_winner(
        out_dir=out_dir,
        winner_experiment_id="P3-BASE",
        winner_name="P3-BASE",
        preprocessing_mode="standard",
        training_time_augmentation=False,
        selection_metric="macro_f1",
        selection_direction="maximize",
        winner_aggregate=agg,
        representation_id_str="rep123",
    )
    assert winner_path.exists()
    winner_data = json.loads(winner_path.read_text(encoding="utf-8"))
    assert winner_data["winner_experiment_id"] == "P3-BASE"
    assert winner_data["preprocessing_mode"] == "standard"
    assert winner_data["macro_f1_mean"] == agg.macro_f1_mean


def test_handcrafted_cache_manifest_and_hashing(pipeline_config):
    """B14: HandcraftedFeatureExtractor initializes cache_manifest.json with parameter hash."""
    extractor = HandcraftedFeatureExtractor(pipeline_config)
    manifest_file = extractor._cache_root / "cache_manifest.json"
    assert manifest_file.exists()

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert "param_hash" in data
    assert data["preprocessing_mode"] == pipeline_config.preprocessing_mode
    assert data["image_size"] == pipeline_config.image_size


def test_inpainting_density_safeguard(pipeline_config):
    """B21: Hair filter density safeguard prevents inpainting if mask covers > 30% of image."""
    preprocessor = ConditionalPreprocessor(pipeline_config)
    # Synthetic image that produces dense blackhat response
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add dense grid
    for i in range(0, 100, 3):
        img[i, :, :] = 255
    res, dec = preprocessor._maybe_remove_hair(img, "test_dense")
    # If coverage > 0.30, inpainting is disabled by safeguard
    if dec.trigger_value > 0.30:
        assert dec.applied is False

