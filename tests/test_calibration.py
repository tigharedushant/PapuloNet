"""
tests/test_calibration.py

Targeted tests for modules/calibration.py's core math.

Does not re-test run_aef_crc_phase8.py's orchestration, the
ConformalHandoff round-trip, or classifier-agnosticism -- see
tests/test_phase8_calibration.py for those.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.calibration import (
    clip_probabilities, negative_log_likelihood, brier_score, expected_calibration_error,
    per_class_ece, fit_isotonic_scaling, apply_isotonic_scaling,
    fit_platt_scaling, apply_platt_scaling, evaluate_calibration,
    select_calibration_method, select_calibration_method_cv,
)

CLASSES = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]


# ============================================================
# Isotonic Regression: Secondary Comparator Protocol
# ============================================================

def test_isotonic_scaling_monotonicity_and_shape():
    rng = np.random.default_rng(1)
    n = 40
    probs = rng.dirichlet([1, 1, 1, 1], size=n)
    true_indices = rng.integers(0, 4, size=n)

    fit = fit_isotonic_scaling(probs, true_indices, CLASSES)
    calibrated = apply_isotonic_scaling(probs, fit, CLASSES)

    assert calibrated.shape == probs.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(calibrated >= 0.0)


def test_isotonic_scaling_handles_unseen_class():
    rng = np.random.default_rng(2)
    n = 20
    probs = rng.dirichlet([1, 1, 1, 1], size=n)
    true_indices = rng.integers(0, 2, size=n)  # classes 2 and 3 never appear

    fit = fit_isotonic_scaling(probs, true_indices, CLASSES)
    assert fit.models["Pityriasis_Rosea"] is None
    assert fit.models["Seborrheic_Dermatitis"] is None

    calibrated = apply_isotonic_scaling(probs, fit, CLASSES)
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)


# ============================================================
# Numerical safety: exact zeros (the Phase-7 "class absent from
# training" case) must not break anything.
# ============================================================

def test_clip_probabilities_handles_exact_zeros():
    probs = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    clipped = clip_probabilities(probs)
    assert np.all(clipped > 0.0)
    assert np.allclose(clipped.sum(axis=1), 1.0)


def test_isotonic_scaling_does_not_crash_on_exact_zero_probabilities():
    probs = np.array([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    fit = fit_isotonic_scaling(probs, np.array([0, 1]), CLASSES)
    result = apply_isotonic_scaling(probs, fit, CLASSES)
    assert np.all(np.isfinite(result))



# ============================================================
# NLL / Brier basic sanity
# ============================================================

def test_nll_is_near_zero_for_confident_correct_predictions():
    probs = clip_probabilities(np.array([[0.999, 0.0003, 0.0003, 0.0004]] * 5))
    true_indices = np.zeros(5, dtype=int)
    assert negative_log_likelihood(probs, true_indices) < 0.01


def test_nll_for_uniform_predictions_equals_log_n_classes():
    n_classes = 4
    probs = clip_probabilities(np.full((10, n_classes), 1.0 / n_classes))
    true_indices = np.zeros(10, dtype=int)
    assert abs(negative_log_likelihood(probs, true_indices) - np.log(n_classes)) < 1e-6


def test_brier_score_is_zero_for_perfect_one_hot_predictions():
    probs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    true_indices = np.array([0, 1])
    assert brier_score(probs, true_indices, 4) < 1e-9


# ============================================================
# ECE correctness
# ============================================================

def test_ece_is_zero_when_confidence_exactly_matches_accuracy_per_bin():
    # Every sample in bin [0.7,0.8) is correct with probability exactly
    # matching its confidence, by construction: 70 confident-0.75
    # samples, 75% (52.5 -> use a clean 4-of-4-ish ratio instead)
    confidences = np.array([0.75] * 8)
    correct = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])  # 6/8 = 0.75, matches confidence exactly
    ece, bins = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece < 1e-9


def test_ece_is_positive_when_systematically_overconfident():
    confidences = np.full(20, 0.95)
    correct = np.array([1.0] * 10 + [0.0] * 10)  # actual accuracy 0.5, confidence 0.95
    ece, _ = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece > 0.4


def test_per_class_ece_returns_one_entry_per_class():
    rng = np.random.default_rng(4)
    probs = rng.dirichlet([1, 1, 1, 1], size=12)
    true_indices = rng.integers(0, 4, size=12)
    result = per_class_ece(probs, true_indices, CLASSES)
    assert set(result.keys()) == set(CLASSES)


# ============================================================
# Platt scaling: degenerate per-class fits
# ============================================================

def test_platt_scaling_passes_through_a_class_with_no_positive_examples():
    """A class that never appears as the true label in the calibration
    set has nothing for its sigmoid to discriminate -- must pass the
    raw probability through unchanged (before the shared per-row
    renormalization), not crash or fabricate a fit."""
    rng = np.random.default_rng(5)
    probs = rng.dirichlet([1, 1, 1, 1], size=10)
    true_indices = rng.integers(0, 2, size=10)  # classes 2 and 3 NEVER appear
    fit = fit_platt_scaling(probs, true_indices, CLASSES, random_seed=0)
    assert fit.models["Pityriasis_Rosea"] is None
    assert fit.models["Seborrheic_Dermatitis"] is None
    calibrated = apply_platt_scaling(probs, fit, CLASSES)
    # Both column 2 and column 3 are raw pass-through (unfitted), so the
    # per-row renormalization (a single scalar per row) preserves their
    # RATIO exactly, regardless of what the fitted columns 0/1 do.
    assert np.allclose(calibrated[:, 2] / calibrated[:, 3], probs[:, 2] / probs[:, 3], atol=1e-9)


def test_apply_platt_scaling_rows_sum_to_one():
    rng = np.random.default_rng(6)
    probs = rng.dirichlet([1, 1, 1, 1], size=30)
    true_indices = rng.integers(0, 4, size=30)
    fit = fit_platt_scaling(probs, true_indices, CLASSES, random_seed=0)
    calibrated = apply_platt_scaling(probs, fit, CLASSES)
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)


# ============================================================
# Selection: macro_f1 must never affect the decision
# ============================================================

# ============================================================
# Selection: frozen Platt primary protocol
# ============================================================

def test_select_calibration_method_returns_platt_primary():
    from modules.calibration import CalibrationMetrics
    common = dict(reliability_bins=[], per_class_ece={})
    uncal = CalibrationMetrics(method="uncalibrated", nll=1.0, brier=1.0, ece=0.30, macro_f1=0.90, **common)
    platt = CalibrationMetrics(method="platt", nll=0.5, brier=0.5, ece=0.10, macro_f1=0.85, **common)

    chosen, reason = select_calibration_method(uncal, platt, min_improvement=0.01)
    assert chosen == "platt"
    assert "f1" not in reason.lower()


# ============================================================
# Cross-validated selection: frozen protocol check
# ============================================================

def test_select_calibration_method_cv_preserves_frozen_platt_primary():
    rng = np.random.default_rng(8)
    n_per_class = 25
    true_indices = np.repeat(np.arange(4), n_per_class)
    n = len(true_indices)
    probs = np.full((n, 4), 0.02)
    probs[np.arange(n), true_indices] = 0.94
    probs = probs / probs.sum(axis=1, keepdims=True)

    chosen, reason, mean_ece = select_calibration_method_cv(probs, true_indices, CLASSES, random_seed=0, min_improvement=0.01)
    assert chosen == "platt"
    assert "platt" in reason.lower()


# ============================================================
# evaluate_calibration bundling
# ============================================================

def test_evaluate_calibration_recomputes_predictions_from_the_given_probs():
    """Platt scaling CAN change the argmax -- evaluate_calibration must
    derive 'correct' from whichever probs it was actually given, not
    from a stale prediction computed elsewhere."""
    probs_a = np.array([[0.6, 0.4, 0.0, 0.0]])
    probs_b = np.array([[0.4, 0.6, 0.0, 0.0]])  # argmax flips class 0 -> class 1
    true_labels = ["Lichen_Planus"]
    m_a = evaluate_calibration(probs_a, true_labels, CLASSES, "a")
    m_b = evaluate_calibration(probs_b, true_labels, CLASSES, "b")
    assert m_a.macro_f1 < m_b.macro_f1  # a: predicted Psoriasis (wrong); b: predicted Lichen_Planus (right)
