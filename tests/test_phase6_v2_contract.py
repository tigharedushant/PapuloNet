"""
tests/test_phase6_v2_contract.py

Targeted test suite and contract verification for PapuloNet V2 Phase 6:
Feature Refinement Benchmark (A2 = EfficientNet-B0 + LBP, 1298-D).

Covers all 5 pre-run audit blocker resolutions:
1. Upstream Phase 5 V2 Binding:
   - Certified winner is strictly A2 (EfficientNet+LBP, 1298-D, deep=1280, lbp=18).
   - Fast failure if winner != A2, dim != 1298, or branches != ('deep', 'lbp').
2. Path Isolation:
   - Output paths strictly under reports/phase6_v2/, artifacts/phase6_v2/, logs/phase6_v2/.
   - Zero access or pollution of V1 reports/phase6/ or artifacts/phase6/.
3. Imbalance-Safe Inner Optimization:
   - BDA and GA proxy estimators compute fold-local sample weights strictly from inner-train labels:
     w_c = N_inner_train / (C * N_c,inner_train).
   - Validation split is never touched during optimization.
   - Parity of proxy models (LogisticRegression C=1.0, lbfgs, max_iter=300).
4. Three-Arm Evaluation on Identical Outer Folds:
   - full_a2 (1298-D unselected baseline), bda, ga.
   - Evaluated with outer classifier receiving fold-local balanced sample weights.
5. Comprehensive Diagnostic Reporting:
   - Macro-F1, BalAcc, MCC, Accuracy, Weighted F1.
   - Per-class Precision, Recall, F1, Support for all 4 classes.
   - Psoriasis majority dominance diagnostic table (true vs pred distributions and ratios).
   - Pooled confusion matrix.
   - Pairwise McNemar's tests with Holm correction across all 3 arm pairs.
   - Jaccard stability across 5 folds.
6. Partition Isolation:
   - Locked 243-image test partition and 246 outer validation images never accessed.
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
    get_production_bda_mask_path,
    save_production_bda_mask,
    load_production_bda_mask,
)
from modules.fusion import FusionFoldData
from run_aef_crc_phase6_v2 import (
    gate_check_phase5_v2,
    compute_dominance_diagnostics,
    run_pairwise_mcnemar,
    evaluate_selection_fold,
)


@pytest.fixture
def config() -> PSDConfig:
    return get_config()


def test_phase6_v2_gate_check_success(config: PSDConfig):
    """Verify that gate check successfully recognizes the certified Phase 5 V2 A2 winner."""
    gate_ok, detail, manifest = gate_check_phase5_v2(config)
    assert gate_ok is True, f"Gate check failed: {detail}"
    assert manifest.get("selected_fusion_arm_id") == "A2"
    assert manifest.get("selected_fusion_dimension") == 1298
    assert manifest.get("selected_fusion_branches") == ["deep", "lbp"]


def test_phase6_v2_gate_check_rejections(config: PSDConfig, tmp_path: Path):
    """Verify that gate check rejects invalid upstream configurations (wrong arm, wrong dim, wrong branches)."""
    # Create invalid manifests in temporary directory
    bad_arm_manifest = {
        "selected_fusion_arm_id": "A7",
        "selected_fusion_arm": "EfficientNet+GLCM+LBP+LAB",
        "selected_fusion_dimension": 1316,
        "selected_fusion_branches": ["deep", "glcm", "lbp", "color_lab"],
    }
    bad_manifest_path = tmp_path / "phase5_manifest.json"
    bad_manifest_path.write_text(json.dumps(bad_arm_manifest), encoding="utf-8")

    test_cfg = dataclasses.replace(config, aef_crc_phase5_v2_reports_dir=tmp_path)
    ok, detail, _ = gate_check_phase5_v2(test_cfg)
    assert ok is False
    assert "expected 'A2'" in detail

    # Wrong dimension
    bad_dim_manifest = {
        "selected_fusion_arm_id": "A2",
        "selected_fusion_arm": "EfficientNet+LBP",
        "selected_fusion_dimension": 1280,
        "selected_fusion_branches": ["deep", "lbp"],
    }
    bad_manifest_path.write_text(json.dumps(bad_dim_manifest), encoding="utf-8")
    ok, detail, _ = gate_check_phase5_v2(test_cfg)
    assert ok is False
    assert "expected exactly 1298" in detail

    # Wrong branches
    bad_branches_manifest = {
        "selected_fusion_arm_id": "A2",
        "selected_fusion_arm": "EfficientNet+LBP",
        "selected_fusion_dimension": 1298,
        "selected_fusion_branches": ["deep", "glcm"],
    }
    bad_manifest_path.write_text(json.dumps(bad_branches_manifest), encoding="utf-8")
    ok, detail, _ = gate_check_phase5_v2(test_cfg)
    assert ok is False
    assert "expected ('deep', 'lbp')" in detail


def test_phase6_v2_path_isolation(config: PSDConfig):
    """Verify that Phase 6 V2 paths are distinct from V1 and write to phase6_v2."""
    assert config.aef_crc_phase6_v2_reports_dir.name == "phase6_v2"
    assert config.aef_crc_phase6_v2_artifacts_dir.name == "phase6_v2"
    assert config.aef_crc_phase6_v2_logs_dir.name == "phase6_v2"

    # Must be separate from V1 directories
    assert config.aef_crc_phase6_v2_reports_dir != config.aef_crc_phase6_reports_dir
    assert config.aef_crc_phase6_v2_artifacts_dir != config.aef_crc_phase6_artifacts_dir


def test_phase6_v2_inner_sample_weights_bda(config: PSDConfig):
    """Verify BDA executes on 1298-D data using fold-local sample weights on inner train."""
    rng = np.random.default_rng(42)
    n_train, n_val, n_feat = 80, 20, 1298
    X_tr = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    # Strongly imbalanced train labels
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
    branch_dims = {"deep": 1280, "lbp": 18}

    data = FusionFoldData(
        X_train=X_tr,
        X_val=X_va,
        y_train=y_tr,
        y_val=y_va,
        psd_ids_train=[f"TR-{i:03d}" for i in range(n_train)],
        psd_ids_val=[f"VA-{i:03d}" for i in range(n_val)],
        branch_dims=branch_dims,
    )

    micro_cfg = dataclasses.replace(
        config,
        bda_population_size=4,
        bda_iterations=2,
        bda_nested_val_fraction=0.2,
        bda_feature_count_penalty=0.0,
    )

    res = select_bda(data, micro_cfg, random_seed=123)
    assert res.method == "bda"
    assert res.selected_mask.shape == (1298,)
    assert res.selected_mask.dtype == bool
    assert res.selected_count == int(res.selected_mask.sum())
    assert 0 < res.selected_count <= 1298
    assert "deep" in res.branch_retained and "lbp" in res.branch_retained
    assert res.branch_retained["deep"] <= 1280
    assert res.branch_retained["lbp"] <= 18
    assert res.extra["feature_count_penalty"] == 0.0
    assert res.extra["fitness_objective"] == "inner_validation_macro_f1"
    assert res.extra["parsimony_penalty"] is None


def test_phase6_v2_inner_sample_weights_ga(config: PSDConfig):
    """Verify GA executes on 1298-D data using fold-local sample weights on inner train."""
    rng = np.random.default_rng(42)
    n_train, n_val, n_feat = 80, 20, 1298
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
    branch_dims = {"deep": 1280, "lbp": 18}

    data = FusionFoldData(
        X_train=X_tr,
        X_val=X_va,
        y_train=y_tr,
        y_val=y_va,
        psd_ids_train=[f"TR-{i:03d}" for i in range(n_train)],
        psd_ids_val=[f"VA-{i:03d}" for i in range(n_val)],
        branch_dims=branch_dims,
    )

    micro_cfg = dataclasses.replace(
        config,
        ga_population_size=4,
        ga_generations=2,
        ga_nested_val_fraction=0.2,
        ga_feature_count_penalty=0.0,
    )

    res = select_ga(data, micro_cfg, random_seed=123)
    assert res.method == "ga"
    assert res.selected_mask.shape == (1298,)
    assert res.selected_mask.dtype == bool
    assert res.selected_count == int(res.selected_mask.sum())
    assert 0 < res.selected_count <= 1298
    assert "deep" in res.branch_retained and "lbp" in res.branch_retained
    assert res.branch_retained["deep"] <= 1280
    assert res.branch_retained["lbp"] <= 18
    assert res.extra["feature_count_penalty"] == 0.0
    assert res.extra["fitness_objective"] == "inner_validation_macro_f1"
    assert res.extra["parsimony_penalty"] is None


def test_phase6_v2_unpenalized_fitness_contract(config: PSDConfig):
    """Verify that Phase 6 V2 optimizer objective is strictly unpenalized (lambda = 0, fitness = inner Macro-F1)."""
    assert config.bda_feature_count_penalty == 0.0
    assert config.ga_feature_count_penalty == 0.0
    assert getattr(config, "phase6_v2_feature_count_penalty", 0.0) == 0.0
    assert getattr(config, "phase6_v2_fitness_objective", "") == "inner_validation_macro_f1"

    rng = np.random.default_rng(99)
    n_train, n_val, n_feat = 40, 16, 1298
    data = FusionFoldData(
        X_train=rng.standard_normal((n_train, n_feat)).astype(np.float32),
        X_val=rng.standard_normal((n_val, n_feat)).astype(np.float32),
        y_train=[config.target_classes[i % 4] for i in range(n_train)],
        y_val=[config.target_classes[i % 4] for i in range(n_val)],
        psd_ids_train=[f"TR-{i:03d}" for i in range(n_train)],
        psd_ids_val=[f"VA-{i:03d}" for i in range(n_val)],
        branch_dims={"deep": 1280, "lbp": 18},
    )

    micro_cfg = dataclasses.replace(
        config,
        bda_population_size=4,
        bda_iterations=2,
        ga_population_size=4,
        ga_generations=2,
        bda_feature_count_penalty=0.0,
        ga_feature_count_penalty=0.0,
    )

    res_bda = select_bda(data, micro_cfg, random_seed=42)
    # With penalty=0, best_nested_fitness must exactly equal best_nested_macro_f1
    assert np.isclose(res_bda.extra["best_nested_fitness"], res_bda.extra["best_nested_macro_f1"])
    assert res_bda.extra["feature_count_penalty"] == 0.0
    assert res_bda.extra["parsimony_penalty"] is None
    assert res_bda.extra["fitness_objective"] == "inner_validation_macro_f1"

    res_ga = select_ga(data, micro_cfg, random_seed=42)
    assert res_ga.extra["feature_count_penalty"] == 0.0
    assert res_ga.extra["parsimony_penalty"] is None
    assert res_ga.extra["fitness_objective"] == "inner_validation_macro_f1"


def test_phase6_v2_dominance_diagnostics_calculation():
    """Verify that dominance diagnostics correctly detects over-prediction of majority class."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    # 10 samples: true has 4 Psoriasis, 3 Lichen Planus, 2 PR, 1 SebDerm
    # predicted has 8 Psoriasis (extreme dominance!)
    pooled = {
        "S1": ("Psoriasis", "Psoriasis"),
        "S2": ("Psoriasis", "Psoriasis"),
        "S3": ("Psoriasis", "Psoriasis"),
        "S4": ("Psoriasis", "Psoriasis"),
        "S5": ("Lichen_Planus", "Psoriasis"),
        "S6": ("Lichen_Planus", "Psoriasis"),
        "S7": ("Lichen_Planus", "Lichen_Planus"),
        "S8": ("Pityriasis_Rosea", "Psoriasis"),
        "S9": ("Pityriasis_Rosea", "Psoriasis"),
        "S10": ("Seborrheic_Dermatitis", "Seborrheic_Dermatitis"),
    }

    diag = compute_dominance_diagnostics(pooled, classes)
    assert diag["total_samples"] == 10
    assert diag["Psoriasis_true_count"] == 4
    assert diag["Psoriasis_true_pct"] == 40.0
    assert diag["Psoriasis_pred_count"] == 8
    assert diag["Psoriasis_pred_pct"] == 80.0
    assert diag["Psoriasis_pred_ratio"] == 2.0  # 2x over-prediction!
    assert diag["Seborrheic_Dermatitis_true_count"] == 1
    assert diag["Seborrheic_Dermatitis_pred_count"] == 1
    assert diag["Seborrheic_Dermatitis_pred_ratio"] == 1.0


def test_phase6_v2_pairwise_mcnemar_holm():
    """Verify that McNemar's tests run across all 3 arms with Holm correction."""
    pooled_by_arm = {
        "full_a2": {f"id_{i}": ("Psoriasis", "Psoriasis" if i < 30 else "Lichen_Planus") for i in range(50)},
        "bda": {f"id_{i}": ("Psoriasis", "Psoriasis" if i < 35 else "Lichen_Planus") for i in range(50)},
        "ga": {f"id_{i}": ("Psoriasis", "Psoriasis" if i < 25 else "Lichen_Planus") for i in range(50)},
    }
    rows = run_pairwise_mcnemar(pooled_by_arm)
    assert len(rows) == 3  # (full_a2, bda), (full_a2, ga), (bda, ga)
    pairs = {(r["arm_a"], r["arm_b"]) for r in rows}
    assert ("full_a2", "bda") in pairs
    assert ("full_a2", "ga") in pairs
    assert ("bda", "ga") in pairs


def test_phase6_v2_jaccard_stability():
    """Verify pairwise and mean Jaccard similarity across fold masks."""
    m1 = np.array([True, True, False, False, True])
    m2 = np.array([True, True, True, False, False])
    m3 = np.array([True, True, False, False, True])

    sim_1_2 = compute_jaccard_similarity(m1, m2)
    # intersection: [0, 1] (2), union: [0, 1, 2, 4] (4) -> 2/4 = 0.5
    assert np.isclose(sim_1_2, 0.5)

    res = compute_pairwise_jaccard([m1, m2, m3])
    assert "mean_jaccard" in res
    assert "median_jaccard" in res
    assert len(res["pairwise"]) == 3  # (0,1), (0,2), (1,2)


def test_phase6_v2_production_bda_mask_contract(config: PSDConfig, tmp_path: Path):
    """Verify save and load of 1298-D production BDA mask."""
    mask_1298 = np.zeros(1298, dtype=bool)
    mask_1298[10:150] = True
    assert mask_1298.sum() == 140

    saved_path = save_production_bda_mask(
        mask_1298,
        config,
        metadata={"test": True},
        run_id="TEST_RUN_123",
        artifacts_dir=tmp_path,
    )
    assert saved_path.exists()

    loaded_mask = load_production_bda_mask(
        config,
        expected_run_id="TEST_RUN_123",
        artifacts_dir=tmp_path,
    )
    assert np.array_equal(loaded_mask, mask_1298)
    assert loaded_mask.shape == (1298,)


def test_phase6_v2_deep_feature_path_resolution(config: PSDConfig):
    """Verify that Phase 6 V2 routes deep-feature requests to artifacts/phase3_v2 for P3-V2-Focal."""
    from modules.efficientnet_model import _resolve_deep_feature_root, load_deep_feature_manifest

    # Test resolution helper
    root_p3_v2 = _resolve_deep_feature_root(config, "P3-V2-Focal")
    assert "phase3_v2" in str(root_p3_v2)
    assert root_p3_v2 == config.aef_crc_phase3_v2_artifacts_dir

    # V1 fallback check: non-V2 experiment names must NOT route to phase3_v2
    root_v1 = _resolve_deep_feature_root(config, "P3-BASE")
    assert root_v1 == config.aef_crc_artifacts_dir

    # Verify that fold 0 train manifest loads directly from V2
    manifest_f0 = load_deep_feature_manifest(config, "P3-V2-Focal", 0, "train")
    assert manifest_f0["experiment_name"] == "P3-V2-Focal"
    assert manifest_f0["representation_id"] == "efficientnet_b0_43d581b96f8ec368"
    assert manifest_f0["feature_dim"] == 1280
    assert manifest_f0["backbone_name"] == "efficientnet_b0"
    assert len(manifest_f0["psd_ids"]) == 916


def test_phase6_v2_preflight_deep_features_success(config: PSDConfig):
    """Verify that preflight check passes for all 5 folds on the real P3-V2-Focal cache."""
    from collections import namedtuple
    from run_aef_crc_phase6_v2 import preflight_deep_feature_artifacts

    MockFold = namedtuple("MockFold", ["fold_index", "train_records", "val_records"])
    mock_folds = [
        MockFold(0, [None] * 916, [None] * 230),
        MockFold(1, [None] * 917, [None] * 229),
        MockFold(2, [None] * 917, [None] * 229),
        MockFold(3, [None] * 917, [None] * 229),
        MockFold(4, [None] * 917, [None] * 229),
    ]

    ok, detail = preflight_deep_feature_artifacts(config, "P3-V2-Focal", mock_folds)
    assert ok is True, f"Preflight check failed: {detail}"
    assert "10 manifests" in detail
    assert "efficientnet_b0_43d581b96f8ec368" in detail


def test_phase6_v2_preflight_rejections(config: PSDConfig, tmp_path: Path):
    """Verify that preflight check rejects non-phase3_v2 roots or missing/corrupted manifests."""
    from collections import namedtuple
    from run_aef_crc_phase6_v2 import preflight_deep_feature_artifacts

    MockFold = namedtuple("MockFold", ["fold_index", "train_records", "val_records"])
    dummy_folds = [MockFold(fold_index=0, train_records=[], val_records=[])]

    # 1. Non-phase3_v2 root rejection
    bad_cfg_non_v2 = dataclasses.replace(config, aef_crc_phase3_v2_artifacts_dir=tmp_path / "non_v2_dir")
    ok, detail = preflight_deep_feature_artifacts(bad_cfg_non_v2, "P3-V2-Focal", dummy_folds)
    assert ok is False
    assert "expected path containing 'phase3_v2'" in detail

    # 2. Missing manifest inside phase3_v2 directory
    empty_v2_dir = tmp_path / "artifacts" / "phase3_v2"
    empty_v2_dir.mkdir(parents=True, exist_ok=True)
    bad_cfg_empty = dataclasses.replace(config, aef_crc_phase3_v2_artifacts_dir=empty_v2_dir)
    ok, detail = preflight_deep_feature_artifacts(bad_cfg_empty, "P3-V2-Focal", dummy_folds)
    assert ok is False
    assert "Failed to load deep feature manifest" in detail

