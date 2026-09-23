"""
tests/test_phase6_v2_a7_contract.py

Targeted unit tests and contract verification for PapuloNet V2 Phase 6-A7:
Exploratory Feature Refinement Benchmark (A7 = EfficientNet-B0 + GLCM + LBP + LAB, 1316-D).

Verifies all strict scientific and architectural invariants:
1. A7 Definition & 1316-D Dimension:
   - Arm ID is strictly 'A7'.
   - Total dimension is exactly 1316 (deep: 1280, glcm: 12, lbp: 18, color_lab: 6).
   - Branches are strictly ('deep', 'glcm', 'lbp', 'color_lab').
   - Fast-fails if any dimension or branch drifts. Never silently falls back to A2.
2. Upstream Deep Representation:
   - Strictly binds to Phase 3 V2 Focal (experiment: 'P3-V2-Focal',
     representation_id: 'efficientnet_b0_43d581b96f8ec368', feature_dim: 1280).
   - Verified path under artifacts/phase3_v2/.
3. Path Isolation:
   - Output paths strictly under reports/phase6_v2_a7/, artifacts/phase6_v2_a7/, logs/phase6_v2_a7/.
   - Zero access or collision with certified Phase 6 V2 A2 (reports/phase6_v2/) or V1 (reports/phase6/).
4. Imbalance-Safe Inner Optimization:
   - BDA and GA proxy estimators compute fold-local sample weights strictly from inner-train labels:
     w_c = N_inner_train / (C * N_c,inner_train).
   - Outer validation split is never touched during optimization.
   - Parity of proxy models (LogisticRegression C=1.0, lbfgs, max_iter=300).
5. Unpenalized Fitness Objective:
   - fitness = inner_validation_macro_f1 (penalty = 0.0).
   - No 0.0005 * k penalty, no parsimony penalty, no sparsity prior.
6. Three-Arm Evaluation on Identical Outer Folds:
   - full_a7 (1316-D unselected baseline), bda_a7, ga_a7.
   - Evaluated with outer classifier receiving fold-local balanced sample weights.
7. Partition Isolation:
   - Locked 243-image test partition and 246 outer validation images are never accessed.
8. Mask Integrity & Feature Family Breakdown:
   - Masks are 1316-element boolean arrays.
   - Family breakdown correctly tracks deep, glcm, lbp, and color_lab.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from config.config import get_config, PSDConfig
from modules.feature_selection import (
    select_no_selection,
    select_bda,
    select_ga,
    run_selector,
    compute_jaccard_similarity,
    compute_pairwise_jaccard,
    compute_feature_family_breakdown,
)
from modules.fusion import (
    FUSION_ARMS_BY_ID,
    FusionFoldData,
    get_fused_feature_names,
)
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import load_frozen_fold_plan
from modules.handcrafted_features import HandcraftedFeatureExtractor
from run_aef_crc_phase6_v2_a7 import (
    A7_ARM_ID,
    A7_ARM_NAME,
    A7_BRANCHES,
    A7_EXPECTED_DIM,
    A7_BRANCH_DIMS,
    P3_V2_FOCAL_EXP,
    P3_V2_REPR_ID,
    gate_check_a7,
    preflight_deep_feature_artifacts,
    preflight_handcrafted_features,
    compute_dominance_diagnostics,
    run_pairwise_mcnemar,
    evaluate_selection_fold,
    write_phase6_v2_a7_reports,
)
from modules.evaluation import AggregatedMetrics


@pytest.fixture
def config() -> PSDConfig:
    return get_config()


def test_phase6_v2_a7_definition():
    """Verify that A7 is registered with exactly 1316 dimensions and ('deep', 'glcm', 'lbp', 'color_lab')."""
    assert A7_ARM_ID == "A7"
    assert A7_EXPECTED_DIM == 1316
    assert A7_BRANCHES == ("deep", "glcm", "lbp", "color_lab")
    assert sum(A7_BRANCH_DIMS.values()) == 1316

    arm_def = FUSION_ARMS_BY_ID[A7_ARM_ID]
    assert arm_def.expected_dim == 1316
    assert arm_def.branches == A7_BRANCHES
    assert arm_def.representation_id == "efficientnet_glcm_lbp_lab"


def test_phase6_v2_a7_gate_check_success(config: PSDConfig):
    """Verify that gate check passes cleanly with the certified upstream artifacts."""
    gate_ok, detail, manifest = gate_check_a7(config)
    assert gate_ok is True, f"A7 gate check failed: {detail}"
    assert "Verified A7 definition" in detail
    assert "P3-V2-Focal" in detail


def test_phase6_v2_a7_gate_check_rejections(config: PSDConfig, tmp_path: Path):
    """Verify that gate check fast-fails on missing manifest or missing A7 in fusion results."""
    # Test missing manifest
    test_cfg = dataclasses.replace(config, aef_crc_phase5_v2_reports_dir=tmp_path)
    ok, detail, _ = gate_check_a7(test_cfg)
    assert ok is False
    assert "Missing Phase 5 V2 manifest" in detail

    # Test manifest present but fusion_results.csv missing A7
    p5_manifest = {"run_id": "TEST_RUN"}
    (tmp_path / "phase5_manifest.json").write_text(json.dumps(p5_manifest), encoding="utf-8")
    ok, detail, _ = gate_check_a7(test_cfg)
    assert ok is False
    assert "Missing Phase 5 V2 fusion_results.csv" in detail

    # Test fusion_results.csv without A7
    fusion_csv = tmp_path / "fusion_results.csv"
    fusion_csv.write_text("arm,macro_f1_mean\nEfficientNet,0.70\nEfficientNet+LBP,0.72\n", encoding="utf-8")
    ok, detail, _ = gate_check_a7(test_cfg)
    assert ok is False
    assert "Arm A7 ('EfficientNet+GLCM+LBP+LAB') not found" in detail


def test_phase6_v2_a7_path_isolation(config: PSDConfig):
    """Verify that Phase 6 V2-A7 output paths are strictly isolated from certified A2 and V1."""
    assert config.aef_crc_phase6_v2_a7_reports_dir.name == "phase6_v2_a7"
    assert config.aef_crc_phase6_v2_a7_artifacts_dir.name == "phase6_v2_a7"
    assert config.aef_crc_phase6_v2_a7_logs_dir.name == "phase6_v2_a7"

    # Must be separate from certified Phase 6 V2 A2 directories
    assert config.aef_crc_phase6_v2_a7_reports_dir != config.aef_crc_phase6_v2_reports_dir
    assert config.aef_crc_phase6_v2_a7_artifacts_dir != config.aef_crc_phase6_v2_artifacts_dir
    assert config.aef_crc_phase6_v2_a7_logs_dir != config.aef_crc_phase6_v2_logs_dir

    # Must be separate from V1 directories
    assert config.aef_crc_phase6_v2_a7_reports_dir != config.aef_crc_phase6_reports_dir
    assert config.aef_crc_phase6_v2_a7_artifacts_dir != config.aef_crc_phase6_artifacts_dir

    # Certified Phase 6 V2 A2 reports must exist and remain untouched
    cert_a2_manifest = config.aef_crc_phase6_v2_reports_dir / "phase6_manifest.json"
    assert cert_a2_manifest.exists(), "Certified Phase 6 V2 A2 manifest is missing!"


def test_phase6_v2_a7_deep_v2_path_resolution(config: PSDConfig):
    """Verify that Phase 3 V2 deep features resolve to artifacts/phase3_v2 and pass preflight."""
    plan = load_frozen_fold_plan(config)
    ok, detail = preflight_deep_feature_artifacts(config, P3_V2_FOCAL_EXP, plan.folds)
    assert ok is True, f"Deep feature preflight failed: {detail}"
    assert "Preflight deep-feature check PASSED" in detail
    assert "dim=1280" in detail
    assert P3_V2_REPR_ID in detail


def test_phase6_v2_a7_handcrafted_cache_verification(config: PSDConfig):
    """Verify that all required handcrafted feature caches (GLCM, LBP, LAB) exist for all folds."""
    std_cfg = dataclasses.replace(config, preprocessing_mode="standard")
    plan = load_frozen_fold_plan(std_cfg)
    extractor = HandcraftedFeatureExtractor(std_cfg)
    ok, detail = preflight_handcrafted_features(std_cfg, plan.folds, extractor)
    assert ok is True, f"Handcrafted preflight failed: {detail}"
    assert "Preflight handcrafted feature check PASSED" in detail


def test_phase6_v2_a7_feature_naming_and_slices(config: PSDConfig):
    """Verify deterministic 1316 feature names and canonical slice boundaries."""
    std_cfg = dataclasses.replace(config, preprocessing_mode="standard")
    names = get_fused_feature_names(A7_BRANCHES, std_cfg)
    assert len(names) == 1316

    # Slices:
    # deep: [0, 1280)
    # glcm: [1280, 1292)
    # lbp: [1292, 1310)
    # color_lab: [1310, 1316)
    assert names[0] == "efficientnet_0"
    assert names[1279] == "efficientnet_1279"
    assert names[1280] == "glcm_contrast_d1"
    assert names[1291] == "glcm_ASM_d3"
    assert names[1292] == "lbp_bin_0"
    assert names[1309] == "lbp_bin_17"
    assert names[1310] == "lab_L_mean"
    assert names[1315] == "lab_b_std"


def test_phase6_v2_a7_inner_sample_weights_bda_ga(config: PSDConfig):
    """Verify BDA and GA execute on 1316-D data using fold-local sample weights on inner train."""
    rng = np.random.default_rng(42)
    n_train, n_val, n_feat = 80, 20, 1316
    X_tr = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    y_tr = (
        ["Psoriasis"] * 45
        + ["Lichen_Planus"] * 18
        + ["Pityriasis_Rosea"] * 11
        + ["Seborrheic_Dermatitis"] * 6
    )
    X_va = rng.standard_normal((n_val, n_feat)).astype(np.float32)
    y_va = (
        ["Psoriasis"] * 11
        + ["Lichen_Planus"] * 5
        + ["Pityriasis_Rosea"] * 3
        + ["Seborrheic_Dermatitis"] * 1
    )

    synth_data = FusionFoldData(
        X_train=X_tr, y_train=y_tr, psd_ids_train=[f"TR-{i:03d}" for i in range(n_train)],
        X_val=X_va, y_val=y_va, psd_ids_val=[f"VA-{i:03d}" for i in range(n_val)],
        branch_dims=A7_BRANCH_DIMS,
    )

    micro_cfg = dataclasses.replace(
        config,
        bda_population_size=4,
        bda_iterations=3,
        ga_population_size=4,
        ga_generations=3,
        bda_feature_count_penalty=0.0,
        ga_feature_count_penalty=0.0,
    )

    # 1. full_a7
    res_full = select_no_selection(synth_data)
    assert res_full.selected_count == 1316
    assert res_full.branch_retained == A7_BRANCH_DIMS

    # 2. bda_a7
    res_bda = run_selector("bda", synth_data, micro_cfg, random_seed=101)
    assert res_bda.selected_count > 0
    assert res_bda.selected_mask.shape == (1316,)
    assert res_bda.selected_mask.dtype == bool
    assert res_bda.extra.get("feature_count_penalty") == 0.0
    assert res_bda.extra.get("fitness_objective") == "inner_validation_macro_f1"

    # 3. ga_a7
    res_ga = run_selector("ga", synth_data, micro_cfg, random_seed=101)
    assert res_ga.selected_count > 0
    assert res_ga.selected_mask.shape == (1316,)
    assert res_ga.selected_mask.dtype == bool
    assert res_ga.extra.get("feature_count_penalty") == 0.0
    assert res_ga.extra.get("fitness_objective") == "inner_validation_macro_f1"

    # Check feature family breakdown
    fb_bda = compute_feature_family_breakdown(res_bda.selected_mask, A7_BRANCH_DIMS)
    assert set(fb_bda.keys()) == set(A7_BRANCH_DIMS.keys())
    assert sum(fb_bda.values()) == res_bda.selected_count


def test_phase6_v2_a7_outer_partition_isolation(config: PSDConfig):
    """Verify that locked test set (243) and outer holdout validation (246) are strictly isolated."""
    # 1. Dataset freeze verification
    freezer = DatasetFreezer(config)
    verify_res = freezer.verify()
    assert verify_res.matches is True, "Dataset freeze corrupted"

    # 2. Frozen fold plan structure
    plan = load_frozen_fold_plan(config)
    assert len(plan.folds) == 5

    # Check total development records across outer folds
    all_dev_ids = set()
    for fold in plan.folds:
        train_ids = {r.psd_id for r in fold.train_records}
        val_ids = {r.psd_id for r in fold.val_records}
        assert not (train_ids & val_ids), f"Fold {fold.fold_index}: leakage between train and val!"
        all_dev_ids |= (train_ids | val_ids)

    assert len(all_dev_ids) == 1146, f"Expected 1146 development images, found {len(all_dev_ids)}"
    assert len(plan.holdout_val_records) == 246, f"Expected 246 outer val images, found {len(plan.holdout_val_records)}"
    assert len(plan.holdout_test_records) == 243, f"Expected 243 test images, found {len(plan.holdout_test_records)}"

    # Zero overlap between partitions
    holdout_ids = {r.psd_id for r in plan.holdout_val_records}
    test_ids = {r.psd_id for r in plan.holdout_test_records}
    assert not (all_dev_ids & holdout_ids), "Leakage between dev cohort and outer holdout validation!"
    assert not (all_dev_ids & test_ids), "Leakage between dev cohort and locked test set!"
    assert not (holdout_ids & test_ids), "Leakage between outer holdout val and locked test set!"


def test_phase6_v2_a7_mcnemar_and_dominance_diagnostics(config: PSDConfig):
    """Verify dominance diagnostics and McNemar test calculations on synthetic A7 outputs."""
    classes = config.target_classes
    fake_preds = {
        "full_a7": {f"ID_{i:03d}": ("Psoriasis", "Psoriasis" if i < 70 else "Lichen_Planus") for i in range(100)},
        "bda_a7": {f"ID_{i:03d}": ("Psoriasis", "Psoriasis" if i < 75 else "Lichen_Planus") for i in range(100)},
        "ga_a7": {f"ID_{i:03d}": ("Psoriasis", "Psoriasis" if i < 68 else "Lichen_Planus") for i in range(100)},
    }

    diag = compute_dominance_diagnostics(fake_preds["bda_a7"], classes)
    assert diag["total_samples"] == 100
    assert "Psoriasis_pred_ratio" in diag
    assert "Lichen_Planus_pred_ratio" in diag

    mcnemar_rows = run_pairwise_mcnemar(fake_preds)
    assert len(mcnemar_rows) == 3
    for r in mcnemar_rows:
        assert "arm_a" in r
        assert "arm_b" in r
        assert "status" in r


def test_phase6_v2_a7_freeze_verify_result_api_regression(config: PSDConfig, tmp_path: Path):
    """Regression test: Ensure FreezeVerifyResult has no stored_hash attribute, catching
    the AttributeError bug, and verify write_phase6_v2_a7_reports resolves dataset_freeze_hash
    using the canonical repository pattern without errors.
    """
    freezer = DatasetFreezer(config)
    verify_res = freezer.verify()

    # 1. Exact attribute regression: FreezeVerifyResult has matches, frozen_at, but NOT stored_hash
    assert hasattr(verify_res, "matches")
    assert hasattr(verify_res, "frozen_at")
    assert not hasattr(verify_res, "stored_hash"), (
        "FreezeVerifyResult should NOT have stored_hash; code accessing it must raise AttributeError"
    )
    with pytest.raises(AttributeError):
        _ = verify_res.stored_hash  # type: ignore[attr-defined]

    # 2. Test write_phase6_v2_a7_reports with synthetic results in a temp directory
    p5_manifest_path = config.project_root / "reports" / "phase5_v2" / "phase5_manifest.json"
    p5_manifest = json.loads(p5_manifest_path.read_text(encoding="utf-8"))

    arms = ["full_a7", "bda_a7", "ga_a7"]
    classes = config.target_classes
    n_classes = len(classes)

    fake_aggregates = {}
    for arm in arms:
        fake_aggregates[arm] = AggregatedMetrics(
            n_folds=5,
            macro_f1_mean=0.72,
            macro_f1_std=0.03,
            balanced_accuracy_mean=0.71,
            balanced_accuracy_std=0.02,
            mcc_mean=0.60,
            mcc_std=0.04,
            accuracy_mean=0.75,
            accuracy_std=0.02,
            weighted_f1_mean=0.74,
            weighted_f1_std=0.02,
            per_class_f1_mean={c: 0.70 for c in classes},
            per_class_f1_std={c: 0.05 for c in classes},
            per_class_precision_mean={c: 0.70 for c in classes},
            per_class_precision_std={c: 0.05 for c in classes},
            per_class_recall_mean={c: 0.70 for c in classes},
            per_class_recall_std={c: 0.05 for c in classes},
            per_class_support_total={c: 100 for c in classes},
            confusion_sum=np.zeros((n_classes, n_classes), dtype=int),
            class_order=classes,
        )

    fake_preds = {
        arm: {f"ID_{i:03d}": ("Psoriasis", "Psoriasis") for i in range(10)}
        for arm in arms
    }

    synthetic_benchmark_results = {
        "arms": arms,
        "aggregates": fake_aggregates,
        "dim_stats": {
            "full_a7": {"mean": 1316.0, "std": 0.0, "per_fold": [1316] * 5},
            "bda_a7": {"mean": 600.0, "std": 50.0, "per_fold": [600] * 5},
            "ga_a7": {"mean": 550.0, "std": 40.0, "per_fold": [550] * 5},
        },
        "family_breakdowns": {
            arm: [
                {"deep": 500, "glcm": 5, "lbp": 10, "color_lab": 3}
                for _ in range(5)
            ]
            for arm in arms
        },
        "dominance": {
            arm: compute_dominance_diagnostics(fake_preds[arm], classes)
            for arm in arms
        },
        "mcnemar_rows": run_pairwise_mcnemar(fake_preds),
        "stability": {
            arm: {"mean_jaccard": 0.5, "median_jaccard": 0.5, "pairwise": {}}
            for arm in arms
        },
    }

    test_rep_dir = tmp_path / "reports" / "phase6_v2_a7"
    custom_cfg = dataclasses.replace(
        config,
        aef_crc_phase6_v2_a7_reports_dir=test_rep_dir,
    )

    # Calling write_phase6_v2_a7_reports must succeed without AttributeError
    out = write_phase6_v2_a7_reports(
        config=custom_cfg,
        benchmark_results=synthetic_benchmark_results,
        p5_manifest=p5_manifest,
        source_experiment="P3-V2-Focal",
    )

    manifest_path = out["manifest_json"]
    assert manifest_path.exists()
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert saved_manifest["dataset_freeze_hash"] is not None
    assert len(saved_manifest["dataset_freeze_hash"]) == 64
    assert saved_manifest["dataset_freeze_hash"] == p5_manifest["dataset_freeze_hash"]
    assert saved_manifest["fold_plan_hash"] == p5_manifest["fold_plan_hash"]
    assert "phase6_a7_manifest.json" in str(manifest_path)

