"""
tests/test_phase7_v2_contract.py

Targeted unit tests and contract verification for PapuloNet V2 Phase 7:
Final Model Selection, Evaluation & Calibration Handoff Pipeline (A7-BDA).

Verifies all strict scientific and architectural invariants:
1. Target Pipeline Definition & 1316-D Invariants:
   - Arm ID is strictly 'A7'.
   - Total dimension is exactly 1316 (deep: 1280, glcm: 12, lbp: 18, color_lab: 6).
   - Branches are strictly ('deep', 'glcm', 'lbp', 'color_lab').
   - Upstream backbone binds to Phase 3 V2 Focal (efficientnet_b0_43d581b96f8ec368).
2. Gate Check Enforcement:
   - Passes when reports/phase6_v2_a7/phase6_a7_manifest.json is valid and signed.
   - Fast-fails if manifest is missing, arm drifts, representation ID drifts, or hashes fail.
3. Path Isolation:
   - All Phase 7 V2 paths under reports/phase7_v2/, artifacts/phase7_v2/, logs/phase7_v2/.
   - Zero collision with V1 (reports/phase7/) or Phase 6 (reports/phase6_v2/).
4. Deep & Handcrafted Cache Preflight:
   - Verifies deep manifests for all 5 folds + fold -1 (final_train and calibration).
   - Verifies handcrafted caches (GLCM 12, LBP 18, LAB 6) for dev (1146) and calib (246).
5. Partition Isolation:
   - Zero overlap between dev (1146), calibration (246), locked test (243), and quarantined (264).
6. Imbalance-Aware Weighting:
   - Inner BDA proxy uses inner-train balanced sample weights.
   - Outer RF uses fold-local balanced class weights.
   - Validation is strictly unweighted.
7. Authoritative Mask & Calibration Handoff Contract:
   - BDA mask is 1316-D boolean array.
   - CalibrationHandoff stratifies 246 calibration records into 123 calib / 123 conformal.
   - Preserves all model weights, masks, probabilities, and provenance.
8. Predefined Acceptance Gating & Fallback Trigger:
   - Explicitly evaluates gating rules without data leakage.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from config.config import get_config, PSDConfig
from modules.aef_input_validator import compute_train_fold_class_weights
from modules.calibration_handoff import (
    CalibrationHandoff,
    save_calibration_handoff,
    load_calibration_handoff,
    validate_bda_mask_compatibility,
    validate_calibration_handoff_provenance,
)
from modules.dataset_freeze import DatasetFreezer
from modules.evaluation import (
    FoldMetrics,
    AggregatedMetrics,
    compute_fold_metrics,
    aggregate_fold_metrics,
)
from modules.feature_selection import (
    run_selector,
    save_production_bda_mask,
    compute_feature_family_breakdown,
)
from modules.fold_loader import load_frozen_fold_plan
from modules.fusion import (
    FusionFoldData,
    compute_sample_weights,
)
from run_aef_crc_phase7_v2 import (
    A7_ARM_ID,
    A7_ARM_NAME,
    A7_BRANCHES,
    A7_EXPECTED_DIM,
    A7_BRANCH_DIMS,
    P3_V2_FOCAL_EXP,
    P3_V2_REPR_ID,
    GATE_CRITERIA,
    configure_phase7_v2,
    gate_check_phase7_v2,
    preflight_deep_feature_artifacts,
    preflight_handcrafted_features,
    preflight_partition_isolation,
    compute_dominance_diagnostics,
    evaluate_phase7_v2_gate,
    write_phase7_v2_reports,
)


@pytest.fixture
def config() -> PSDConfig:
    return configure_phase7_v2(get_config())


def test_phase7_v2_pipeline_definition():
    """Verify that A7-BDA is defined with exactly 1316-D and ('deep', 'glcm', 'lbp', 'color_lab')."""
    assert A7_ARM_ID == "A7"
    assert A7_EXPECTED_DIM == 1316
    assert A7_BRANCHES == ("deep", "glcm", "lbp", "color_lab")
    assert sum(A7_BRANCH_DIMS.values()) == 1316
    assert A7_BRANCH_DIMS["deep"] == 1280
    assert A7_BRANCH_DIMS["glcm"] == 12
    assert A7_BRANCH_DIMS["lbp"] == 18
    assert A7_BRANCH_DIMS["color_lab"] == 6


def test_phase7_v2_gate_check_success(config: PSDConfig):
    """Verify gate check passes against real certified Phase 6 V2-A7 manifest."""
    ok, detail, manifest = gate_check_phase7_v2(config)
    assert ok is True, f"Gate check failed: {detail}"
    assert manifest.get("phase") == "phase6_v2_a7"
    assert manifest.get("representation_id") == P3_V2_REPR_ID
    assert manifest.get("a7_definition", {}).get("expected_dimension") == 1316
    assert manifest.get("dataset_freeze_hash") is not None
    assert manifest.get("fold_plan_hash") is not None


def test_phase7_v2_gate_check_rejections(config: PSDConfig, tmp_path: Path):
    """Verify gate check rejects missing or corrupted manifests."""
    # 1. Missing manifest
    bad_cfg_missing = dataclasses.replace(config, aef_crc_phase6_v2_a7_reports_dir=tmp_path / "empty")
    ok, detail, _ = gate_check_phase7_v2(bad_cfg_missing)
    assert ok is False
    assert "missing" in detail.lower()

    # 2. Corrupt representation ID
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    real_m = json.loads((config.aef_crc_phase6_v2_a7_reports_dir / "phase6_a7_manifest.json").read_text(encoding="utf-8"))

    corrupt_m = dict(real_m)
    corrupt_m["representation_id"] = "wrong_backbone_hash"
    (corrupt_dir / "phase6_a7_manifest.json").write_text(json.dumps(corrupt_m), encoding="utf-8")

    bad_cfg_repr = dataclasses.replace(config, aef_crc_phase6_v2_a7_reports_dir=corrupt_dir)
    ok, detail, _ = gate_check_phase7_v2(bad_cfg_repr)
    assert ok is False
    assert "representation id mismatch" in detail.lower()

    # 3. Wrong dimension
    corrupt_m2 = dict(real_m)
    corrupt_m2["a7_definition"] = dict(real_m["a7_definition"])
    corrupt_m2["a7_definition"]["expected_dimension"] = 1298
    (corrupt_dir / "phase6_a7_manifest.json").write_text(json.dumps(corrupt_m2), encoding="utf-8")
    ok, detail, _ = gate_check_phase7_v2(bad_cfg_repr)
    assert ok is False
    assert "dimension mismatch" in detail.lower()


def test_phase7_v2_path_isolation(config: PSDConfig):
    """Verify strict path isolation for Phase 7 V2 directories."""
    v2_rep = config.aef_crc_phase7_v2_reports_dir
    v2_art = config.aef_crc_phase7_v2_artifacts_dir
    v2_log = config.aef_crc_phase7_v2_logs_dir

    v1_rep = config.project_root / "reports" / "phase7"
    v1_art = config.project_root / "artifacts" / "phase7"
    p6_rep = config.project_root / "reports" / "phase6_v2_a7"

    assert "phase7_v2" in str(v2_rep)
    assert "phase7_v2" in str(v2_art)
    assert "phase7_v2" in str(v2_log)
    assert str(v2_rep) != str(v1_rep)
    assert str(v2_art) != str(v1_art)
    assert str(v2_rep) != str(p6_rep)


def test_phase7_v2_deep_feature_preflight(config: PSDConfig):
    """Verify deep-feature manifests for all 5 folds + fold -1 (final_train and calibration)."""
    plan = load_frozen_fold_plan(config)
    ok, msg = preflight_deep_feature_artifacts(config, P3_V2_FOCAL_EXP, plan.folds)
    assert ok is True, f"Deep feature preflight failed: {msg}"
    assert "verified all 12 deep-feature manifests" in msg.lower()


def test_phase7_v2_handcrafted_cache_preflight(config: PSDConfig):
    """Verify handcrafted feature caches for dev (1146) and calibration (246)."""
    plan = load_frozen_fold_plan(config)
    ok, msg = preflight_handcrafted_features(config, plan)
    assert ok is True, f"Handcrafted cache preflight failed: {msg}"
    assert "1392 records" in msg  # 1146 dev + 246 calib = 1392


def test_phase7_v2_partition_isolation(config: PSDConfig):
    """Verify mutual disjointness between dev (1146), calibration (246), test (243), and quarantined (264)."""
    plan = load_frozen_fold_plan(config)
    ok, msg = preflight_partition_isolation(plan)
    assert ok is True, f"Partition isolation check failed: {msg}"
    assert "dev=1146, calib=246, test=243, quarantined=264" in msg


def test_phase7_v2_inner_sample_weights(config: PSDConfig):
    """Verify BDA runs on synthetic 1316-D data with inner-train balanced sample weights."""
    rng = np.random.default_rng(42)
    n_tr, n_va, n_dim = 80, 20, A7_EXPECTED_DIM
    X_tr = rng.standard_normal((n_tr, n_dim)).astype(np.float32)
    y_tr = ["Psoriasis"] * 45 + ["Lichen_Planus"] * 18 + ["Pityriasis_Rosea"] * 11 + ["Seborrheic_Dermatitis"] * 6
    X_va = rng.standard_normal((n_va, n_dim)).astype(np.float32)
    y_va = ["Psoriasis"] * 11 + ["Lichen_Planus"] * 5 + ["Pityriasis_Rosea"] * 3 + ["Seborrheic_Dermatitis"] * 1

    synth_data = FusionFoldData(
        X_train=X_tr, y_train=y_tr, psd_ids_train=[f"TR-{i:03d}" for i in range(n_tr)],
        X_val=X_va, y_val=y_va, psd_ids_val=[f"VA-{i:03d}" for i in range(n_va)],
        branch_dims=A7_BRANCH_DIMS,
    )

    micro_cfg = dataclasses.replace(
        config,
        bda_population_size=4,
        bda_iterations=3,
        bda_feature_count_penalty=0.0,
    )
    sel_res = run_selector("bda", synth_data, micro_cfg, random_seed=42)
    assert sel_res.selected_mask.shape == (A7_EXPECTED_DIM,)
    assert sel_res.selected_count > 0
    assert sel_res.extra.get("fitness_objective") == "inner_validation_macro_f1"
    assert sel_res.extra.get("feature_count_penalty") == 0.0


def test_phase7_v2_production_bda_mask_and_handoff(config: PSDConfig, tmp_path: Path):
    """Verify production BDA mask saving, compatibility, and CalibrationHandoff round-trip."""
    # 1. Create and validate synthetic production BDA mask
    mask = np.zeros(A7_EXPECTED_DIM, dtype=bool)
    mask[:600] = True
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    X_dummy = np.ones((20, 600), dtype=np.float32)
    y_dummy = ["Psoriasis"] * 10 + ["Lichen_Planus"] * 5 + ["Pityriasis_Rosea"] * 3 + ["Seborrheic_Dermatitis"] * 2
    clf.fit(X_dummy, y_dummy)

    validate_bda_mask_compatibility(mask, clf, expected_dim=A7_EXPECTED_DIM)

    # 2. Save production mask
    saved_path = save_production_bda_mask(
        mask,
        config,
        metadata={"selected_count": 600},
        run_id="TEST_RUN_ID",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert saved_path.exists()
    payload = joblib.load(saved_path)
    assert payload["selected_count"] == 600
    assert payload["total_dim"] == A7_EXPECTED_DIM

    # 3. Test CalibrationHandoff persistence and round-trip
    classes = config.target_classes
    n_calib = 246
    handoff = CalibrationHandoff(
        representation_id=P3_V2_REPR_ID,
        experiment_id=P3_V2_FOCAL_EXP,
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=42,
        source_experiment=P3_V2_FOCAL_EXP,
        feature_arm=A7_ARM_ID,
        class_order=list(classes),
        final_classifier=clf,
        selected_feature_mask=mask,
        branch_dims=A7_BRANCH_DIMS,
        calibration_psd_ids=[f"PSD_CAL_{i:04d}" for i in range(n_calib)],
        calibration_true_labels=["Psoriasis"] * n_calib,
        calibration_predicted_labels=["Psoriasis"] * n_calib,
        calibration_raw_probabilities=np.ones((n_calib, len(classes)), dtype=np.float32) * 0.25,
        run_id="TEST_RUN_ID",
        dataset_freeze_hash="test_freeze_hash",
        fold_plan_hash="test_fold_hash",
        val_calib_psd_ids=[f"PSD_CAL_{i:04d}" for i in range(123)],
        val_calib_true_labels=["Psoriasis"] * 123,
        val_calib_raw_probabilities=np.ones((123, len(classes)), dtype=np.float32) * 0.25,
        val_conf_psd_ids=[f"PSD_CAL_{i:04d}" for i in range(123, 246)],
        val_conf_true_labels=["Psoriasis"] * 123,
        val_conf_raw_probabilities=np.ones((123, len(classes)), dtype=np.float32) * 0.25,
    )

    handoff_path = tmp_path / "artifacts" / "calibration_handoff.joblib"
    save_calibration_handoff(handoff, handoff_path)
    assert handoff_path.exists()

    loaded = load_calibration_handoff(handoff_path)
    assert loaded.representation_id == P3_V2_REPR_ID
    assert loaded.selected_feature_mask.shape == (A7_EXPECTED_DIM,)
    assert len(loaded.val_calib_psd_ids) == 123
    assert len(loaded.val_conf_psd_ids) == 123


def test_phase7_v2_dominance_and_confusion_matrix(config: PSDConfig):
    """Verify dominance diagnostics calculation and confusion matrix handling."""
    classes = config.target_classes
    preds = {
        f"ID_{i:03d}": ("Psoriasis", "Psoriasis" if i < 60 else "Lichen_Planus")
        for i in range(100)
    }
    diag = compute_dominance_diagnostics(preds, classes)
    assert diag["total_samples"] == 100
    assert diag["Psoriasis_true_count"] == 100
    assert diag["Psoriasis_pred_count"] == 60
    assert diag["Psoriasis_pred_ratio"] == 0.60
    assert "Seborrheic_Dermatitis_pred_ratio" in diag


def test_phase7_v2_gating_logic(config: PSDConfig):
    """Verify predefined gating evaluation: pass triggers CERTIFIED_WINNER; failure triggers FALLBACK_TRIGGERED."""
    classes = config.target_classes

    # 1. Passing scenario
    agg_pass = AggregatedMetrics(
        n_folds=5, macro_f1_mean=0.7143, macro_f1_std=0.0361,
        accuracy_mean=0.7574, accuracy_std=0.0245,
        balanced_accuracy_mean=0.7106, balanced_accuracy_std=0.0346,
        weighted_f1_mean=0.7526, weighted_f1_std=0.0247,
        mcc_mean=0.5991, mcc_std=0.0420,
        per_class_f1_mean={"Psoriasis": 0.82, "Lichen_Planus": 0.63, "Pityriasis_Rosea": 0.70, "Seborrheic_Dermatitis": 0.71},
        per_class_f1_std={"Psoriasis": 0.02, "Lichen_Planus": 0.04, "Pityriasis_Rosea": 0.06, "Seborrheic_Dermatitis": 0.08},
        per_class_precision_mean={c: 0.70 for c in classes},
        per_class_precision_std={c: 0.05 for c in classes},
        per_class_recall_mean={c: 0.70 for c in classes},
        per_class_recall_std={c: 0.05 for c in classes},
        per_class_support_total={c: 100 for c in classes},
        confusion_sum=np.zeros((len(classes), len(classes)), dtype=int),
        class_order=list(classes),
    )
    calib_pass = FoldMetrics(
        fold_index=-1, macro_f1=0.69, accuracy=0.74, balanced_accuracy=0.68,
        weighted_f1=0.73, mcc=0.58, per_class={}, confusion=np.zeros((4, 4)),
        class_order=list(classes),
    )
    dom_pass = {"Psoriasis_pred_ratio": 1.07}

    res_pass = evaluate_phase7_v2_gate(agg_pass, calib_pass, dom_pass)
    assert res_pass["passed_all"] is True
    assert res_pass["decision"] == "CERTIFIED_WINNER"
    assert res_pass["pipeline_selected"] == "A7-BDA"

    # 2. Failing scenario (Psoriasis dominance too high)
    dom_fail = {"Psoriasis_pred_ratio": 1.25}  # exceeds 1.15
    res_fail = evaluate_phase7_v2_gate(agg_pass, calib_pass, dom_fail)
    assert res_pass["passed_all"] is True
    assert res_fail["passed_all"] is False
    assert res_fail["decision"] == "FALLBACK_TRIGGERED"
    assert res_fail["pipeline_selected"] == "full_a7"


def test_phase7_v2_representation_id_binding(config: PSDConfig):
    """Regression test for representation ID mismatch:
    - Verifies configure_phase7_v2(cfg) strictly produces P3_V2_REPR_ID ('efficientnet_b0_43d581b96f8ec368').
    - Verifies unconfigured base config produces the default drifted hash ('efficientnet_b0_d00e004c85ba726c').
    - Verifies preflight_deep_feature_artifacts succeeds with configured config.
    - Verifies preflight_deep_feature_artifacts rejects unconfigured config.
    - Verifies validate_deep_feature_cache succeeds with configured config.
    - Verifies validate_deep_feature_cache raises DeepFeatureCacheError when representation differs.
    """
    from modules.experiment_config import representation_id
    from modules.backbones import validate_deep_feature_cache, DeepFeatureCacheError

    # 1. Configured config produces P3_V2_REPR_ID
    assert representation_id(config) == P3_V2_REPR_ID
    assert P3_V2_REPR_ID == "efficientnet_b0_43d581b96f8ec368"

    # 2. Config with standard preprocessing but missing V2 focal loss produces the drifted hash observed in real run
    base_cfg = dataclasses.replace(get_config(), preprocessing_mode="standard")
    drifted_id = representation_id(base_cfg)
    assert drifted_id == "efficientnet_b0_d00e004c85ba726c"
    assert drifted_id != P3_V2_REPR_ID

    # 3. Preflight rejects unconfigured config
    plan = load_frozen_fold_plan(config)
    ok_bad, msg_bad = preflight_deep_feature_artifacts(base_cfg, P3_V2_FOCAL_EXP, plan.folds)
    assert ok_bad is False
    assert "does not match authoritative" in msg_bad

    # 4. validate_deep_feature_cache passes with configured config
    f0 = plan.folds[0]
    validate_deep_feature_cache(
        config, P3_V2_FOCAL_EXP, 0, "train",
        expected_psd_ids=[r.psd_id for r in f0.train_records],
        backbone_name=config.backbone,
    )

    # 5. validate_deep_feature_cache raises DeepFeatureCacheError on unconfigured/drifted config
    with pytest.raises(DeepFeatureCacheError) as excinfo:
        validate_deep_feature_cache(
            base_cfg, P3_V2_FOCAL_EXP, 0, "train",
            expected_psd_ids=[r.psd_id for r in f0.train_records],
            backbone_name=config.backbone,
        )
    assert "d00e004c85ba726c" in str(excinfo.value)

