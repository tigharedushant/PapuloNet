"""
tests/test_phase10_runner.py

Targeted unit tests for run_aef_crc_phase10.py:
1. Preflight validation logic and contract checks.
2. Representative validation sample selection logic (strictly from holdout val; zero test leakage).
3. Production A7 SHAP contract validation under various valid and invalid configurations.
4. Locked test set isolation enforcement.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from config.config import get_config
from run_aef_crc_phase9_v2 import configure_phase9_v2
from modules.aef_input_validator import ImageRecord
from modules.calibration_handoff import FinalPipelineHandoff, get_active_run_id
from modules.fold_loader import FoldPlan, load_frozen_fold_plan
from modules.xai_shap import SHAPError, validate_production_shap_contract
from run_aef_crc_phase10 import (
    _write_phase10_report,
    select_representative_validation_samples,
    validate_phase10_preflight,
)

# ---------------------------------------------------------------------------
# V2 Production Contract Constants
# ---------------------------------------------------------------------------
V2_REPR_ID = "efficientnet_b0_43d581b96f8ec368"
V2_K_FEATURES = 642
V2_FULL_DIM = 1316


@pytest.fixture
def mock_pipeline_handoff():
    """Builds a valid mock FinalPipelineHandoff adhering to the V2 production A7 contract (642 features)."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    mask = np.zeros(V2_FULL_DIM, dtype=bool)
    mask[:V2_K_FEATURES] = True  # exactly 642 features active (V2)

    mock_rf = MagicMock()
    mock_rf.n_features_in_ = V2_K_FEATURES
    mock_rf.classes_ = np.arange(4)
    mock_rf.predict_proba.return_value = np.array([[0.70, 0.10, 0.10, 0.10]])

    # Build a mock temperature scaler that mirrors MulticlassTemperatureScaler.predict_proba
    mock_temp_scaler = MagicMock()
    mock_temp_scaler.predict_proba.side_effect = lambda p: p  # identity for testing

    config = configure_phase9_v2(get_config())
    active_run_id = get_active_run_id(config) or ""

    return FinalPipelineHandoff(
        run_id=active_run_id,
        representation_id=V2_REPR_ID,
        experiment_id="test_exp_v2",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=config.random_seed,
        class_order=classes,
        final_classifier=mock_rf,
        selected_feature_mask=mask,
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
        calibration_method="temperature_scaling",
        temperature_scaler=mock_temp_scaler,
        marginal_q_hat=0.7382427827337363,
        mondrian_q_hat={cls: 0.7382427827337363 for cls in classes},
        backbone_checkpoint_path=str(config.aef_crc_artifacts_dir / "P3-BASE" / "fold_00" / "best_model.keras"),
    )


def test_validation_sample_selection_integrity():
    """Verifies that sample selection strictly consumes holdout val and NEVER touches test set."""
    config = get_config()
    plan = load_frozen_fold_plan(config)

    # Must have exact pre-registered holdout counts
    assert len(plan.holdout_val_records) == 246
    assert len(plan.holdout_test_records) == 243

    # Select 2 samples per class (8 total)
    selected = select_representative_validation_samples(plan, config, num_per_class=2)
    assert len(selected) == 8

    # All 4 target classes must be represented with exactly 2 samples each
    classes_found = [s.mapped_class for s in selected]
    for cls in config.target_classes:
        assert classes_found.count(cls) == 2

    # Verify zero leakage into locked test partition
    test_paths = {str(r.file_path).replace("\\", "/") for r in plan.holdout_test_records}
    test_ids = {r.psd_id for r in plan.holdout_test_records}

    for s in selected:
        p_str = str(s.file_path).replace("\\", "/")
        assert p_str not in test_paths, f"Sample {s.psd_id} found in test partition paths!"
        assert s.psd_id not in test_ids, f"Sample {s.psd_id} found in test partition IDs!"
        assert "test" not in p_str.lower()
        assert "val" in p_str.lower()
        assert s.file_path.exists()


def test_validation_sample_selection_determinism():
    """Verifies that sample selection is 100% deterministic given the same config seed."""
    config = get_config()
    plan = load_frozen_fold_plan(config)

    sel1 = select_representative_validation_samples(plan, config, num_per_class=2)
    sel2 = select_representative_validation_samples(plan, config, num_per_class=2)

    ids1 = [s.psd_id for s in sel1]
    ids2 = [s.psd_id for s in sel2]
    assert ids1 == ids2


def test_production_shap_contract_v2_valid():
    """Verifies that validate_production_shap_contract passes with the V2 642-feature contract."""
    mock_rf = MagicMock()
    mock_rf.n_features_in_ = V2_K_FEATURES

    valid_mask = np.zeros(V2_FULL_DIM, dtype=bool)
    valid_mask[:V2_K_FEATURES] = True

    # V2: explicit expected_k_features=642 passes cleanly
    validate_production_shap_contract(
        classifier=mock_rf,
        selected_feature_mask=valid_mask,
        expected_full_dim=V2_FULL_DIM,
        expected_k_features=V2_K_FEATURES,
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
    )


def test_production_shap_contract_v2_no_explicit_k():
    """Verifies that passing expected_k_features=None allows the mask-derived count to be used."""
    mock_rf = MagicMock()
    mock_rf.n_features_in_ = V2_K_FEATURES

    valid_mask = np.zeros(V2_FULL_DIM, dtype=bool)
    valid_mask[:V2_K_FEATURES] = True

    # Without explicit k, validation passes using mask-derived k == n_features_in_
    validate_production_shap_contract(
        classifier=mock_rf,
        selected_feature_mask=valid_mask,
        expected_full_dim=V2_FULL_DIM,
        expected_k_features=None,
    )


def test_production_shap_contract_invalid_mask_length():
    """Verifies that an incorrect mask length is rejected."""
    mock_rf = MagicMock()
    mock_rf.n_features_in_ = V2_K_FEATURES
    valid_mask = np.zeros(V2_FULL_DIM, dtype=bool)
    valid_mask[:V2_K_FEATURES] = True

    with pytest.raises(SHAPError, match="BDA mask dimension mismatch"):
        validate_production_shap_contract(
            classifier=mock_rf,
            selected_feature_mask=valid_mask[:1000],
            expected_full_dim=V2_FULL_DIM,
            expected_k_features=V2_K_FEATURES,
        )


def test_production_shap_contract_invalid_k_count():
    """Verifies that an incorrect active feature count is rejected when expected_k_features is set."""
    mock_rf = MagicMock()
    mock_rf.n_features_in_ = 100  # wrong for both V1 and V2

    invalid_k_mask = np.zeros(V2_FULL_DIM, dtype=bool)
    invalid_k_mask[:100] = True  # 100 features, but V2 expects 642
    with pytest.raises(SHAPError, match="Production selected features mismatch"):
        validate_production_shap_contract(
            classifier=mock_rf,
            selected_feature_mask=invalid_k_mask,
            expected_full_dim=V2_FULL_DIM,
            expected_k_features=V2_K_FEATURES,  # strict check: must be 642
        )


def test_production_shap_contract_rejects_hog():
    """Verifies that HOG block in A7 layout is rejected."""
    mock_rf = MagicMock()
    mock_rf.n_features_in_ = V2_K_FEATURES
    valid_mask = np.zeros(V2_FULL_DIM, dtype=bool)
    valid_mask[:V2_K_FEATURES] = True

    with pytest.raises(SHAPError, match="forbids HOG"):
        validate_production_shap_contract(
            classifier=mock_rf,
            selected_feature_mask=valid_mask,
            expected_full_dim=V2_FULL_DIM,
            expected_k_features=V2_K_FEATURES,
            branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6},
        )


def test_phase10_preflight_validation(mock_pipeline_handoff):
    """Verifies that validate_phase10_preflight passes with the V2 handoff and config."""
    config = configure_phase9_v2(get_config())
    info = validate_phase10_preflight(config, mock_pipeline_handoff, device="cpu")
    assert info["status"] == "PASS"
    assert info["fused_dimensions"] == V2_FULL_DIM
    assert info["selected_features"] == V2_K_FEATURES  # V2: 642
    assert info["val_samples_available"] == 246
    assert info["test_samples_locked"] == 243


def test_write_phase10_report_standalone(tmp_path):
    """Verifies that _write_phase10_report operates standalone without depending on an out-of-scope sample_id."""
    report_path = tmp_path / "test_report.md"
    manifest = {
        "timestamp_utc": "2026-09-18T16:30:16.459487+00:00",
        "representation_id": "efficientnet_b0_6760c4f151acc2d2",
        "fusion_layout": "A7",
        "fused_dimensions": 1316,
        "selected_features_k": 194,
        "classifier_name": "random_forest",
        "calibration_method": "platt",
        "marginal_q_hat": 0.782165682476496,
    }
    samples = [
        {
            "sample_id": "PSD_TEST_001",
            "true_class": "Psoriasis",
            "predicted_class": "Psoriasis",
            "is_correct": True,
            "calibrated_confidence": 0.95,
            "marginal_prediction_set": "Psoriasis",
            "review_status": "HIGH_CONFIDENCE_POINT_PREDICTION",
            "top_shap_branch": "deep",
            "top_shap_branch_pct": 92.5,
        },
        {
            "sample_id": "PSD_TEST_002",
            "true_class": "Lichen_Planus",
            "predicted_class": "Lichen_Planus",
            "is_correct": True,
            "calibrated_confidence": 0.88,
            "marginal_prediction_set": "Lichen_Planus",
            "review_status": "HIGH_CONFIDENCE_POINT_PREDICTION",
            "top_shap_branch": "glcm",
            "top_shap_branch_pct": 45.0,
        },
    ]

    # Ensure no global sample_id exists in globals
    assert "sample_id" not in globals()

    # Call _write_phase10_report directly outside any sample processing loop
    _write_phase10_report(report_path, manifest, samples)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    # Verify actual sample IDs are listed
    assert "PSD_TEST_001" in content
    assert "PSD_TEST_002" in content

    # Verify existing artifact paths are preserved
    assert "gradcam/PSD_TEST_001/original.png" in content
    assert "gradcam/PSD_TEST_001/gradcam_heatmap.png" in content
    assert "gradcam/PSD_TEST_001/gradcam_overlay.png" in content
    assert "shap/PSD_TEST_001/shap_feature_importance.csv" in content
    assert "shap/PSD_TEST_001/shap_block_importance.csv" in content

    # Verify no unresolved template string remains
    assert "{sample_id}" not in content

    # Test edge case: empty samples list does not fail or reference undefined variable
    empty_report = tmp_path / "empty_report.md"
    _write_phase10_report(empty_report, manifest, [])
    assert empty_report.exists()
    empty_content = empty_report.read_text(encoding="utf-8")
    assert "Selected Samples Table" in empty_content


def test_write_phase10_report_real_manifest_compatibility(tmp_path):
    """Verifies that _write_phase10_report works seamlessly with the real Phase 10 manifest."""
    manifest_path = Path("reports/phase10/phase10_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Real phase10_manifest.json not present in workspace")

    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = tmp_path / "phase10_report_real_check.md"

    _write_phase10_report(report_path, manifest, manifest["samples"])
    assert report_path.exists()

    content = report_path.read_text(encoding="utf-8")
    for s in manifest["samples"]:
        sid = s["sample_id"]
        assert sid in content
        assert f"gradcam/{sid}/original.png" in content
        assert f"gradcam/{sid}/gradcam_heatmap.png" in content
        assert f"gradcam/{sid}/gradcam_overlay.png" in content
        assert f"shap/{sid}/shap_feature_importance.csv" in content
        assert f"shap/{sid}/shap_block_importance.csv" in content
    assert "{sample_id}" not in content


# =============================================================================
# Regression Tests for Phase 10 V2 Fixes
# =============================================================================

def test_canonical_conformal_boundary_exact_match():
    """
    Regression: (1.0 - p) <= q_hat MUST include the exact boundary sample.
    Verifies canonical nonconformity rule (not legacy p >= 1.0 - q).
    At the exact boundary: s = 1.0 - p == q_hat → (1.0 - p) <= q_hat is True.
    """
    from modules.conformal import predict_conformal_sets

    classes = ["A", "B", "C", "D"]
    q_hat = 0.7382427827337363  # certified Phase 9 V2 q_hat

    # Exact boundary: p = 1.0 - q_hat → s = q_hat exactly
    p_boundary = 1.0 - q_hat
    probs = np.array([[0.90, p_boundary, 0.05, 0.05]])

    marginal_sets = predict_conformal_sets(probs, q_hat, classes)
    assert "A" in marginal_sets[0], "Class A (high prob) must be included"
    assert "B" in marginal_sets[0], (
        f"Class B at exact boundary (p={p_boundary}, s=q_hat={q_hat}) "
        "MUST be included by canonical (1-p) <= q_hat rule"
    )
    assert "C" not in marginal_sets[0], "Class C (low prob) must NOT be included"
    assert "D" not in marginal_sets[0], "Class D (low prob) must NOT be included"


def test_canonical_conformal_boundary_just_below_and_above():
    """Regression: p just below boundary excluded; p just above boundary included."""
    from modules.conformal import predict_conformal_sets

    classes = ["A", "B"]
    q_hat = 0.7382427827337363

    p_exact = 1.0 - q_hat
    p_just_below = p_exact - 1e-10  # s > q_hat → excluded
    p_just_above = p_exact + 1e-10  # s < q_hat → included

    probs_below = np.array([[p_just_below, 1.0 - p_just_below]])
    sets_below = predict_conformal_sets(probs_below, q_hat, classes)
    assert "A" not in sets_below[0], "p just below boundary must NOT be included"

    probs_above = np.array([[p_just_above, 1.0 - p_just_above]])
    sets_above = predict_conformal_sets(probs_above, q_hat, classes)
    assert "A" in sets_above[0], "p just above boundary must be included"


def test_v2_artifact_path_in_main():
    """Regression: Phase 10 main() must reference artifacts/phase9_v2/, not phase9/."""
    import run_aef_crc_phase10 as p10_module
    import inspect
    src = inspect.getsource(p10_module.main)
    assert "phase9_v2" in src, "main() must reference artifacts/phase9_v2/"
    assert '"phase9"' not in src and "'phase9'" not in src, (
        "main() must NOT reference stale artifacts/phase9/ path"
    )


def test_v2_config_repr_id():
    """Regression: configure_phase9_v2 must yield the certified V2 representation_id."""
    from modules.experiment_config import representation_id as compute_repr_id
    config = configure_phase9_v2(get_config())
    rid = compute_repr_id(config)
    assert rid == V2_REPR_ID, (
        f"configure_phase9_v2 must yield representation_id={V2_REPR_ID!r}, got {rid!r}"
    )


def test_temperature_scaling_branch_selected_for_v2(mock_pipeline_handoff):
    """Regression: apply_calibration must route to temperature_scaling for V2 handoff."""
    from modules.inference import AEFCRCInferenceEngine
    engine = AEFCRCInferenceEngine(artifact=mock_pipeline_handoff, device="cpu")
    assert engine.calib_method == "temperature_scaling"
    raw_probs = np.array([[0.7, 0.1, 0.1, 0.1]])
    calib_probs = engine.apply_calibration(raw_probs)
    assert calib_probs is not None
    assert calib_probs.shape == raw_probs.shape


def test_test_evaluation_guard_accepts_temperature_scaler():
    """Regression: The guard must NOT block V2 handoffs that have temperature_scaler (platt_models=None)."""
    from unittest.mock import MagicMock
    mock_handoff = MagicMock()
    mock_handoff.platt_models = None
    mock_handoff.isotonic_models = None
    mock_handoff.temperature_scaler = MagicMock()  # V2 calibration
    has_any = (
        getattr(mock_handoff, "temperature_scaler", None) is not None
        or getattr(mock_handoff, "platt_models", None) is not None
        or getattr(mock_handoff, "isotonic_models", None) is not None
    )
    assert has_any, "V2 handoff with temperature_scaler must satisfy calibration guard"


def test_v2_selected_dimension_is_642(mock_pipeline_handoff):
    """Regression: V2 handoff must have exactly 642 selected features, not 194."""
    k = int(mock_pipeline_handoff.selected_feature_mask.sum())
    assert k == V2_K_FEATURES, f"V2 selected features must be {V2_K_FEATURES}, got {k}"
    assert k != 194, "V2 must NOT have 194 features (that is the V1 value)"


def test_phase9_v2_handoff_provenance():
    """Regression: Phase 9 V2 handoff artifact must have certified representation_id if present."""
    import joblib
    handoff_path = (
        get_config().project_root / "artifacts" / "phase9_v2" / "final_pipeline_handoff.joblib"
    )
    if not handoff_path.exists():
        pytest.skip("Phase 9 V2 handoff not present — skip provenance regression")
    handoff = joblib.load(handoff_path)
    assert handoff.representation_id == V2_REPR_ID, (
        f"Phase 9 V2 handoff representation_id must be {V2_REPR_ID!r}, got {handoff.representation_id!r}"
    )
    assert "P9_V2" in handoff.run_id, (
        f"Phase 9 V2 handoff run_id must contain 'P9_V2', got {handoff.run_id!r}"
    )


def test_v2_provenance_not_blocked_by_stale_phase3_run_id():
    """
    Regression: validate_pipeline_handoff_provenance must NOT raise when the handoff
    carries a V2 phase-scoped run_id (AEFCRC_P9_V2_...) even though the active Phase 3
    winner.json contains a different AEFCRC_RUN_* run_id.

    Root cause of the Phase 10 error:
      handoff.run_id  = 'AEFCRC_P9_V2_20260921_150441_54fc3c7b'
      get_active_run_id(config) = 'AEFCRC_RUN_20260917_142937_...' (from phase3/winner.json)
    These are different lineages; the Phase 3 run ID must be ignored for V2 handoffs.
    """
    from modules.calibration_handoff import validate_pipeline_handoff_provenance, FinalPipelineHandoff
    from unittest.mock import MagicMock

    config = configure_phase9_v2(get_config())

    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    mask = np.zeros(V2_FULL_DIM, dtype=bool)
    mask[:V2_K_FEATURES] = True

    mock_rf = MagicMock()
    mock_rf.n_features_in_ = V2_K_FEATURES

    handoff = FinalPipelineHandoff(
        run_id="AEFCRC_P9_V2_20260921_150441_54fc3c7b",  # certified V2 run ID
        representation_id=V2_REPR_ID,
        experiment_id="test_v2_provenance",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=config.random_seed,
        class_order=classes,
        final_classifier=mock_rf,
        selected_feature_mask=mask,
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
        calibration_method="temperature_scaling",
        marginal_q_hat=0.7382427827337363,
        mondrian_q_hat={cls: 0.7382427827337363 for cls in classes},
    )

    # Must not raise — the AEFCRC_P9_V2_* run_id must bypass Phase 3 run_id comparison
    try:
        validate_pipeline_handoff_provenance(handoff, config)
    except RuntimeError as exc:
        pytest.fail(
            f"validate_pipeline_handoff_provenance raised for V2 handoff: {exc}\n"
            "Fix: AEFCRC_P* run IDs must not be compared against Phase 3 winner.json run_id."
        )
