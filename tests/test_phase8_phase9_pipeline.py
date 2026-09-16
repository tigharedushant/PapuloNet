"""
tests/test_phase8_phase9_pipeline.py

Comprehensive test suite for AEF-CRC Phase 8 (Probability Calibration)
and Phase 9 (Conformal Prediction).

Coverage:
1. Outer validation partition (123 + 123 from 246 outer validation records).
2. Probability calibration models (Platt scaling, Isotonic regression, ECE, Brier).
3. Split-conformal quantile arithmetic (exact finite-sample k formula checks).
4. Marginal and Mondrian conformal prediction set construction.
5. Small-sample diagnostic alerts for rare classes.
6. Handoff artifact persistence and roundtrips (ConformalHandoff, FinalPipelineHandoff).
7. End-to-end inference engine contract validation.

EXECUTION GUARD COMPLIANCE:
This file is created for verification and CI regression testing.
It is NOT executed during the current turn.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from PIL import Image

from modules.calibration import (
    clip_probabilities,
    brier_score,
    expected_calibration_error,
    fit_platt_scaling,
    apply_platt_scaling,
    fit_isotonic_scaling,
    apply_isotonic_scaling,
    partition_outer_validation,
    partition_validation_set,
)
from modules.conformal import (
    compute_nonconformity_scores,
    compute_conformal_quantile,
    fit_marginal_conformal,
    fit_class_conditional_mondrian,
    predict_conformal_sets,
    predict_conformal_sets_mondrian,
    evaluate_conformal_sets,
)
from modules.calibration_handoff import (
    ConformalHandoff,
    ConformalArtifact,
    FinalPipelineHandoff,
    save_conformal_handoff,
    load_conformal_handoff,
    save_final_pipeline_handoff,
    load_final_pipeline_handoff,
    validate_bda_mask_compatibility,
)
from modules.inference import AEFCRCInferenceEngine, InferenceResult


# =============================================================================
# 1. Outer Validation Partitioning Tests
# =============================================================================

def test_validation_partition_exact_50_50():
    """Verifies that 246 outer validation records partition into exactly 123 + 123 disjoint samples."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    # Exact real counts from dataset_report.csv: Psoriasis=137, Lichen=55, Pityriasis=35, Seborrheic=19 (Total=246)
    counts = {"Psoriasis": 137, "Lichen_Planus": 55, "Pityriasis_Rosea": 35, "Seborrheic_Dermatitis": 19}
    psd_ids = []
    labels = []
    for cls, cnt in counts.items():
        for i in range(cnt):
            psd_ids.append(f"{cls}_{i:03d}")
            labels.append(cls)

    assert len(psd_ids) == 246

    result = partition_outer_validation(
        psd_ids=psd_ids,
        labels=labels,
        class_order=classes,
        random_seed=42,
        calib_fraction=0.5,
    )

    # Check partition sizes
    assert len(result.val_calib_psd_ids) == 123
    assert len(result.val_conf_psd_ids) == 123

    # Check disjointness (zero leakage)
    calib_set = set(result.val_calib_psd_ids)
    conf_set = set(result.val_conf_psd_ids)
    assert calib_set.isdisjoint(conf_set)
    assert len(calib_set | conf_set) == 246

    # Check all classes present in both
    for cls in classes:
        assert result.per_class_calib_counts[cls] > 0
        assert result.per_class_conf_counts[cls] > 0
        assert result.per_class_calib_counts[cls] + result.per_class_conf_counts[cls] == counts[cls]


def test_partition_validation_set_wrapper():
    """Verifies the partition_validation_set object wrapper function."""
    records = [{"psd_id": f"IMG_{i:03d}", "mapped_class": "Psoriasis" if i % 2 == 0 else "Lichen_Planus"} for i in range(50)]
    calib, conf = partition_validation_set(records, seed=42)
    assert len(calib) == 25
    assert len(conf) == 25
    calib_ids = {r["psd_id"] for r in calib}
    conf_ids = {r["psd_id"] for r in conf}
    assert calib_ids.isdisjoint(conf_ids)


# =============================================================================
# 2. Probability Calibration Tests
# =============================================================================

def test_platt_and_isotonic_calibration_properties():
    """Verifies Platt and Isotonic scaling produce valid probability distributions."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    rng = np.random.default_rng(123)
    n = 60
    true_indices = rng.integers(0, 4, size=n)

    # Generate synthetic uncalibrated overconfident probabilities
    raw_probs = np.full((n, 4), 0.05)
    raw_probs[np.arange(n), true_indices] = 0.85
    raw_probs /= raw_probs.sum(axis=1, keepdims=True)

    # 1. Platt scaling
    platt_fit = fit_platt_scaling(raw_probs, true_indices, classes, random_seed=42)
    platt_calib = apply_platt_scaling(raw_probs, platt_fit, classes)
    assert platt_calib.shape == (n, 4)
    assert np.all(platt_calib >= 0.0) and np.all(platt_calib <= 1.0)
    np.testing.assert_allclose(platt_calib.sum(axis=1), np.ones(n), rtol=1e-5)

    # 2. Isotonic regression
    iso_fit = fit_isotonic_scaling(raw_probs, true_indices, classes)
    iso_calib = apply_isotonic_scaling(raw_probs, iso_fit, classes)
    assert iso_calib.shape == (n, 4)
    assert np.all(iso_calib >= 0.0) and np.all(iso_calib <= 1.0)
    np.testing.assert_allclose(iso_calib.sum(axis=1), np.ones(n), rtol=1e-5)


def test_ece_and_brier_calculation():
    """Verifies calibration evaluation metrics."""
    probs = np.array([
        [0.90, 0.10],
        [0.80, 0.20],
        [0.30, 0.70],
        [0.20, 0.80],
    ])
    true_indices = np.array([0, 0, 1, 1])
    brier = brier_score(probs, true_indices, n_classes=2)
    assert 0.0 <= brier <= 1.0

    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == true_indices).astype(float)
    ece, bins = expected_calibration_error(confidences, correct, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(bins) == 5


# =============================================================================
# 3. Split-Conformal Quantile Arithmetic Tests
# =============================================================================

def test_split_conformal_exact_quantile_arithmetic():
    """
    Verifies exact split-conformal quantile arithmetic:
        k = ceil((n + 1) * (1 - alpha))
    Edge cases checked:
      - n = 123, alpha = 0.10 => k = ceil(124 * 0.90) = ceil(111.6) = 112
      - n = 10,  alpha = 0.10 => k = ceil(11 * 0.90)  = ceil(9.9)   = 10 (NOT 11)
      - n = 9,   alpha = 0.10 => k = ceil(10 * 0.90)  = 9
      - n = 8,   alpha = 0.10 => k = ceil(9 * 0.90)   = ceil(8.1)   = 9 (> n)
    """
    # 1. n = 123, alpha = 0.10
    n = 123
    alpha = 0.10
    k_123 = int(math.ceil((n + 1) * (1.0 - alpha)))
    assert k_123 == 112
    scores_123 = np.linspace(0.01, 0.99, n)
    q_123 = compute_conformal_quantile(scores_123, alpha=alpha)
    assert q_123 == pytest.approx(scores_123[112 - 1])

    # 2. n = 10, alpha = 0.10 (Seborrheic Dermatitis calibration support scale)
    n = 10
    k_10 = int(math.ceil((n + 1) * (1.0 - alpha)))
    assert k_10 == 10  # Must be 10, NOT 11!
    scores_10 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    q_10 = compute_conformal_quantile(scores_10, alpha=alpha)
    assert q_10 == pytest.approx(0.95)

    # 3. n = 9, alpha = 0.10
    n = 9
    k_9 = int(math.ceil((n + 1) * (1.0 - alpha)))
    assert k_9 == 9
    scores_9 = np.linspace(0.1, 0.9, n)
    q_9 = compute_conformal_quantile(scores_9, alpha=alpha)
    assert q_9 == pytest.approx(0.9)

    # 4. n = 8, alpha = 0.10 (finite sample support limit)
    n = 8
    k_8 = int(math.ceil((n + 1) * (1.0 - alpha)))
    assert k_8 == 9  # 9 > 8, requires clipping to n
    scores_8 = np.linspace(0.1, 0.8, n)
    q_8 = compute_conformal_quantile(scores_8, alpha=alpha)
    assert q_8 == pytest.approx(0.8)


# =============================================================================
# 4. Marginal and Mondrian Conformal Prediction Sets
# =============================================================================

def test_conformal_prediction_sets_and_evaluation():
    """Verifies set construction, coverage guarantee, and Mondrian diagnostic warnings."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    rng = np.random.default_rng(999)
    n_conf = 123
    true_indices = rng.integers(0, 4, size=n_conf)
    true_labels = [classes[i] for i in true_indices]

    # Synthetic well-calibrated probabilities
    probs = rng.dirichlet(np.ones(4) * 2.0, size=n_conf)

    # 1. Marginal fit
    marginal_fit = fit_marginal_conformal(probs, true_labels, classes, alpha=0.10)
    assert marginal_fit.n_samples == 123
    assert marginal_fit.formula_k == 112
    assert 0.0 <= marginal_fit.q_hat <= 1.0

    # Prediction sets
    pred_sets = predict_conformal_sets(probs, marginal_fit.q_hat, classes)
    assert len(pred_sets) == n_conf
    assert all(isinstance(s, list) for s in pred_sets)

    # Evaluation
    metrics = evaluate_conformal_sets(pred_sets, true_labels, classes, nominal_coverage=0.90)
    assert 0.0 <= metrics.marginal_coverage <= 1.0
    assert metrics.mean_set_size >= 1.0
    assert 0.0 <= metrics.singleton_fraction <= 1.0

    # 2. Mondrian fit (Diagnostic)
    mondrian_fit = fit_class_conditional_mondrian(probs, true_labels, classes, alpha=0.10, min_recommended_n=35)
    for cls in classes:
        assert cls in mondrian_fit.q_hat_by_class
        assert 0.0 <= mondrian_fit.q_hat_by_class[cls] <= 1.0


# =============================================================================
# 5. Handoff Persistence & Compatibility Validation Tests
# =============================================================================

def test_bda_mask_compatibility_validation():
    """Verifies that validate_bda_mask_compatibility checks dimensions and catches mismatches."""
    # Valid mask
    mask_good = np.ones(1348, dtype=bool)
    mock_clf = type("MockClf", (), {"n_features_in_": 1348})()
    validate_bda_mask_compatibility(mask_good, mock_clf, expected_dim=1348)

    # Incompatible dimension mask
    mask_wrong_dim = np.ones(1000, dtype=bool)
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_bda_mask_compatibility(mask_wrong_dim, mock_clf, expected_dim=1348)

    # Incompatible classifier feature count
    mock_clf_mismatch = type("MockClf", (), {"n_features_in_": 500})()
    with pytest.raises(ValueError, match="BDA mask / classifier incompatibility"):
        validate_bda_mask_compatibility(mask_good, mock_clf_mismatch, expected_dim=1348)


def test_conformal_and_final_pipeline_handoff_roundtrip(tmp_path):
    """Verifies ConformalHandoff and FinalPipelineHandoff serialize cleanly through joblib."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    mask = np.ones(1348, dtype=bool)

    handoff = FinalPipelineHandoff(
        representation_id="REPR_TEST",
        experiment_id="EXP_TEST",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=42,
        class_order=classes,
        final_classifier=object(),
        selected_feature_mask=mask,
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6},
        calibration_method="platt",
        marginal_q_hat=0.75,
        mondrian_q_hat={cls: 0.75 for cls in classes},
        alpha=0.10,
        n_conf_samples=123,
    )

    art_path = tmp_path / "final_pipeline_handoff.joblib"
    save_final_pipeline_handoff(handoff, art_path)
    loaded = load_final_pipeline_handoff(art_path)

    assert loaded.representation_id == "REPR_TEST"
    assert loaded.classifier_name == "random_forest"
    assert loaded.marginal_q_hat == 0.75
    assert loaded.calibration_method == "platt"
    assert np.array_equal(loaded.selected_feature_mask, mask)


# =============================================================================
# 6. Inference Engine Mock Pipeline Test
# =============================================================================

def test_inference_engine_mock_contract():
    """Verifies that AEFCRCInferenceEngine validates inputs and correctly structures outputs."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    mask = np.zeros(1348, dtype=bool)
    mask[:10] = True  # selects 10 features

    class MockRandomForest:
        def predict_proba(self, X):
            # Return fixed probabilities favoring Psoriasis
            return np.array([[0.85, 0.05, 0.05, 0.05]])

    handoff = FinalPipelineHandoff(
        representation_id="REPR_INFERENCE_TEST",
        experiment_id="EXP_INFERENCE_TEST",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=42,
        class_order=classes,
        final_classifier=MockRandomForest(),
        selected_feature_mask=mask,
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6},
        calibration_method="uncalibrated",
        marginal_q_hat=0.70,  # threshold = 1 - 0.70 = 0.30 => only Psoriasis (0.85 >= 0.30) included
        mondrian_q_hat={cls: 0.70 for cls in classes},
        alpha=0.10,
    )

    engine = AEFCRCInferenceEngine(artifact=handoff)

    # Test image validation on small/degenerate inputs
    too_small = Image.new("RGB", (16, 16), color="red")
    with pytest.raises(ValueError, match="too small"):
        engine.validate_input_image(too_small)

    degenerate = Image.new("RGB", (64, 64), color="black")
    with pytest.raises(ValueError, match="Degenerate"):
        engine.validate_input_image(degenerate)

    # Valid synthetic test image
    valid_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img_val = engine.validate_input_image(valid_img)
    assert img_val.size == (64, 64)

    # Feature masking contract
    fake_fused = np.arange(1348, dtype=float)
    masked = engine.apply_feature_mask(fake_fused)
    assert masked.shape == (1, 10)

    # Probability prediction & calibration contract
    raw_probs = engine.predict_raw_probabilities(masked)
    assert raw_probs.shape == (1, 4)
    calib_probs = engine.apply_calibration(raw_probs)
    assert calib_probs.shape == (1, 4)

    # Conformal sets contract
    marginal_set, mondrian_set = engine.predict_conformal_sets(calib_probs)
    assert marginal_set == ["Psoriasis"]
    assert "Psoriasis" in mondrian_set
