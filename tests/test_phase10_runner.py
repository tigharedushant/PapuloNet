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
from run_aef_crc_phase6 import align_phase3_winner_config
from modules.aef_input_validator import ImageRecord
from modules.calibration_handoff import FinalPipelineHandoff, get_active_run_id
from modules.fold_loader import FoldPlan, load_frozen_fold_plan
from modules.xai_shap import SHAPError, validate_production_shap_contract
from run_aef_crc_phase10 import (
    _write_phase10_report,
    select_representative_validation_samples,
    validate_phase10_preflight,
)


@pytest.fixture
def mock_pipeline_handoff():
    """Builds a valid mock FinalPipelineHandoff adhering to the production A7 contract."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    mask = np.zeros(1316, dtype=bool)
    mask[:194] = True  # exactly 194 features active

    mock_rf = MagicMock()
    mock_rf.n_features_in_ = 194
    mock_rf.classes_ = np.arange(4)
    mock_rf.predict_proba.return_value = np.array([[0.70, 0.10, 0.10, 0.10]])

    config, _ = align_phase3_winner_config(get_config())
    active_run_id = get_active_run_id(config) or ""

    return FinalPipelineHandoff(
        run_id=active_run_id,
        representation_id="efficientnet_b0_6760c4f151acc2d2",
        experiment_id="test_exp",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=config.random_seed,
        class_order=classes,
        final_classifier=mock_rf,
        selected_feature_mask=mask,
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
        calibration_method="platt",
        marginal_q_hat=0.78,
        mondrian_q_hat={cls: 0.78 for cls in classes},
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


def test_production_shap_contract_valid_and_invalid():
    """Verifies that validate_production_shap_contract strictly enforces the A7 contract."""
    mock_rf = MagicMock()
    mock_rf.n_features_in_ = 194

    valid_mask = np.zeros(1316, dtype=bool)
    valid_mask[:194] = True

    # Valid contract passes cleanly
    validate_production_shap_contract(
        classifier=mock_rf,
        selected_feature_mask=valid_mask,
        expected_full_dim=1316,
        expected_k_features=194,
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
    )

    # 1. Invalid mask length
    with pytest.raises(SHAPError, match="BDA mask dimension mismatch"):
        validate_production_shap_contract(
            classifier=mock_rf,
            selected_feature_mask=valid_mask[:1000],
            expected_full_dim=1316,
            expected_k_features=194,
        )

    # 2. Invalid active feature count
    invalid_k_mask = np.zeros(1316, dtype=bool)
    invalid_k_mask[:100] = True
    with pytest.raises(SHAPError, match="Production selected features mismatch"):
        validate_production_shap_contract(
            classifier=mock_rf,
            selected_feature_mask=invalid_k_mask,
            expected_full_dim=1316,
            expected_k_features=194,
        )

    # 3. Disallowed HOG block in A7
    with pytest.raises(SHAPError, match="forbids HOG"):
        validate_production_shap_contract(
            classifier=mock_rf,
            selected_feature_mask=valid_mask,
            expected_full_dim=1316,
            expected_k_features=194,
            branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6},
        )


def test_phase10_preflight_validation(mock_pipeline_handoff):
    """Verifies that validate_phase10_preflight passes with the valid handoff and config."""
    config, _ = align_phase3_winner_config(get_config())
    info = validate_phase10_preflight(config, mock_pipeline_handoff, device="cpu")
    assert info["status"] == "PASS"
    assert info["fused_dimensions"] == 1316
    assert info["selected_features"] == 194
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

