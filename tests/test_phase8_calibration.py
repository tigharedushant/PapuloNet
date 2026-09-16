# SYNTHETIC_TEST_FIXTURE = True
# This test uses strictly synthetic, generated, or mock fixtures.
# Zero real dataset images or private research artifacts are required or accessed.
"""
tests/test_phase8_calibration.py

Targeted tests for run_aef_crc_phase8.py's orchestration:

  - ConformalHandoff round-trips through disk with a still-usable model.
  - Insufficient calibration data honestly falls back to uncalibrated
    (config.calibration_min_samples gate) rather than fitting anything.
  - Calibration is classifier-agnostic (Task 4): run_calibration's
    behavior depends only on the probabilities in the handoff, not on
    which classifier_name produced them (verified with two handoffs
    differing ONLY in classifier_name).
  - run_calibration never receives or reads anything from
    FoldPlan.holdout_test_records -- it only ever sees a
    CalibrationHandoff, which structurally cannot carry the test set
    (Task: "The holdout test set remains untouched").

Does not re-test modules/calibration.py's own math -- see
tests/test_calibration.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.calibration_handoff import CalibrationHandoff, ConformalHandoff, save_conformal_handoff, load_conformal_handoff
import run_aef_crc_phase8 as phase8


def _make_handoff(classifier_name="logistic_regression", n=30, seed=0, min_class_count=None):
    """A CalibrationHandoff with hand-built, deliberately-overconfident-
    but-correctly-ranked probabilities (so calibration has something
    genuine to fix), evenly spread across all 4 classes so
    cross-validated selection can actually run unless min_class_count
    is set smaller."""
    rng = np.random.default_rng(seed)
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    per_class = min_class_count if min_class_count is not None else n // 4
    true_indices = np.repeat(np.arange(4), per_class)
    total = len(true_indices)
    probs = np.full((total, 4), 0.02)
    probs[np.arange(total), true_indices] = 0.94
    probs = probs / probs.sum(axis=1, keepdims=True)
    true_labels = [classes[i] for i in true_indices]
    predicted_labels = [classes[i] for i in probs.argmax(axis=1)]

    class DummyClassifier:
        n_features_in_ = 200

    mask = np.zeros(1348, dtype=bool)
    mask[:200] = True

    return CalibrationHandoff(
        representation_id="repr123", experiment_id="exp123",
        classifier_name=classifier_name, feature_selection_method="bda",
        random_seed=42, source_experiment="TEST-EXP", feature_arm="EfficientNet",
        class_order=classes,
        final_classifier=DummyClassifier(),
        selected_feature_mask=mask,
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "lab": 6},
        calibration_psd_ids=[f"C{i}" for i in range(total)],
        calibration_true_labels=true_labels, calibration_predicted_labels=predicted_labels,
        calibration_raw_probabilities=probs,
    )


# ============================================================
# ConformalHandoff round-trip
# ============================================================

def test_conformal_handoff_round_trips_through_disk(tmp_path):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = ["Psoriasis", "Lichen_Planus"] * 5
    clf = LogisticRegression().fit(X, y)

    handoff = ConformalHandoff(
        representation_id="r", experiment_id="e", classifier_name="logistic_regression",
        feature_selection_method="none", random_seed=42, class_order=["Psoriasis", "Lichen_Planus"],
        final_classifier=clf, selected_feature_mask=np.ones(3, dtype=bool), branch_dims={"deep": 3},
        calibration_method="platt", calibration_method_reason="Frozen primary Platt calibration",
        platt_models={"Psoriasis": None, "Lichen_Planus": None}, isotonic_models=None,
        calibration_psd_ids=["A", "B"], calibration_true_labels=["Psoriasis", "Lichen_Planus"],
        calibration_calibrated_probabilities=clf.predict_proba(X[:2]),
    )
    path = tmp_path / "conformal.joblib"
    save_conformal_handoff(handoff, path)
    reloaded = load_conformal_handoff(path)

    assert reloaded.calibration_method == "platt"
    assert np.array_equal(reloaded.calibration_calibrated_probabilities, handoff.calibration_calibrated_probabilities)
    assert list(reloaded.final_classifier.predict(X[:1])) == list(clf.predict(X[:1]))


def test_load_conformal_handoff_raises_clearly_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_conformal_handoff(tmp_path / "does_not_exist.joblib")


def test_load_conformal_handoff_rejects_wrong_artifact_type(tmp_path):
    import joblib
    path = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a conformal handoff"}, path)
    with pytest.raises(TypeError):
        load_conformal_handoff(path)


# ============================================================
# Insufficient calibration data -- honest fallback
# ============================================================

def test_run_calibration_falls_back_when_below_min_samples():
    from config.config import get_config
    import dataclasses
    config = dataclasses.replace(get_config(), calibration_min_samples=1000)  # deliberately unreachable
    handoff = _make_handoff(n=20)
    result = phase8.run_calibration(config, handoff, label_prefix="[TEST] ")
    assert result.calibration_method == "uncalibrated"
    assert "below the minimum" in result.calibration_method_reason
    assert result.platt_models is None
    # Raw probabilities are still clipped and carried forward, not dropped.
    assert result.calibration_calibrated_probabilities is not None
    assert np.allclose(result.calibration_calibrated_probabilities.sum(axis=1), 1.0)


def test_run_calibration_gate_uses_configured_threshold_not_a_hardcoded_one(tmp_path):
    """The min-samples gate reads config.calibration_min_samples -- a
    lower threshold on the SAME handoff must let calibration proceed."""
    from config.config import get_config
    import dataclasses
    handoff = _make_handoff(n=40, min_class_count=10)
    lenient_config = dataclasses.replace(
        get_config(), calibration_min_samples=5, aef_crc_phase8_reports_dir=tmp_path / "reports" / "phase8",
    )
    result = phase8.run_calibration(lenient_config, handoff, label_prefix="[TEST] ")
    assert result.calibration_method_reason  # a real decision was made, not the below-minimum message
    assert "below the minimum" not in result.calibration_method_reason


# ============================================================
# Task 4: classifier-agnosticism
# ============================================================

@pytest.mark.parametrize("classifier_name", ["random_forest", "logistic_regression"])
def test_run_calibration_behavior_depends_only_on_probabilities_not_classifier_name(classifier_name, tmp_path):
    """Two handoffs with IDENTICAL probabilities but different
    classifier_name must produce identical calibration outcomes --
    calibration operates on probabilities, never on which registered
    classifier produced them (Task 4)."""
    from config.config import get_config
    import dataclasses
    config = dataclasses.replace(get_config(), aef_crc_phase8_reports_dir=tmp_path / "reports" / "phase8")
    reference = phase8.run_calibration(config, _make_handoff(classifier_name="logistic_regression", min_class_count=10), label_prefix="[TEST] ")
    other = phase8.run_calibration(config, _make_handoff(classifier_name=classifier_name, min_class_count=10), label_prefix="[TEST] ")
    assert reference.calibration_method == other.calibration_method
    assert np.array_equal(reference.calibration_calibrated_probabilities, other.calibration_calibrated_probabilities)


# ============================================================
# The test set is structurally unreachable from run_calibration
# ============================================================

def test_calibration_handoff_has_no_test_set_field_at_all():
    """The strongest form of 'never touches the test set': there is no
    field on CalibrationHandoff a test-set record could even occupy.
    run_aef_crc_phase7.py builds this handoff exclusively from
    FoldPlan.holdout_val_records (see tests/test_phase7_final_model.py
    for the proof that holdout_test_records is never read); Phase 8
    receives only this handoff, so nothing test-set-derived can reach
    it even in principle."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(CalibrationHandoff)}
    assert not any("test" in name for name in field_names)


# ============================================================
# Isotonic Scaling & Outer Validation Partitioning
# ============================================================

def test_fit_and_apply_isotonic_scaling():
    from modules.calibration import fit_isotonic_scaling, apply_isotonic_scaling
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    probs = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.2, 0.6, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ])
    true_indices = np.array([0, 1, 2, 3])
    iso_fit = fit_isotonic_scaling(probs, true_indices, classes)
    calibrated = apply_isotonic_scaling(probs, iso_fit, classes)

    assert calibrated.shape == probs.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_outer_validation_partition_50_50():
    from modules.calibration import partition_outer_validation
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    psd_ids = [f"val_{i:03d}" for i in range(246)]
    labels = ["Psoriasis"] * 80 + ["Lichen_Planus"] * 80 + ["Pityriasis_Rosea"] * 50 + ["Seborrheic_Dermatitis"] * 36
    res = partition_outer_validation(psd_ids, labels, classes, random_seed=42, calib_fraction=0.5)

    assert len(res.val_calib_psd_ids) == 123
    assert len(res.val_conf_psd_ids) == 123
    assert set(res.val_calib_psd_ids).isdisjoint(set(res.val_conf_psd_ids))

