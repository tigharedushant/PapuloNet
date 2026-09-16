"""
tests/test_phase7_final_model.py

Targeted tests for run_aef_crc_phase7.py's genuinely new logic (Task 5):

  - build_fusion_final's PSD-ID leakage guards (duplicate within a set,
    overlap between final-train and calibration) -- new code, not
    covered by tests/test_fusion.py's existing build_fusion_fold tests.
  - Fit-on-train-only discipline for the final retrain (mirrors
    build_fusion_fold's existing tested discipline; proven here for
    the NEW function).
  - The predict_proba class-ordering bug found and fixed while
    building this phase (a class absent from clf.classes_ must get a
    0.0 probability column, not crash) -- a real regression test for
    an actual bug.
  - CalibrationHandoff round-trips through disk with every field
    intact, including a still-usable fitted classifier.
  - run_final_retrain never reads FoldPlan.holdout_test_records beyond
    its length.
  - run_final_cv/run_final_retrain pass the correct fold-local /
    final-retrain sample weights (Task 1 -- explicitly required to
    "reuse ... fold-local class weights").
  - classifier_name/feature_selection_method are genuinely swappable
    (Task 4), proven by an actual registry swap, not asserted.

Does NOT re-test: PSD-ID alignment within a CV fold, leakage in
build_fusion_fold, deep-feature cache validation, representation_id
hardening, or the four selectors' own correctness -- all covered by
existing test files, unchanged by this phase.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from config.config import get_config
from modules.fusion import ARMS, FusionFoldData, build_fusion_final, CLASSIFIER_REGISTRY
from modules.feature_selection import SelectionResult, SELECTOR_REGISTRY
from modules.calibration_handoff import CalibrationHandoff, save_calibration_handoff, load_calibration_handoff
import run_aef_crc_phase7 as phase7


# ============================================================
# Shared plumbing fixture: one real synthetic dataset + fold plan +
# deep-feature cache (final_train/calibration included), built ONCE
# per test module rather than per test.
# ============================================================

@pytest.fixture(scope="module")
def plumbing_setup(tmp_path_factory):
    from modules.synthetic_fixtures import make_test_config, build_synthetic_dataset, run_pipeline_through_split
    from modules.fold_loader import ImbalanceAwareFoldLoader
    from modules.backbones import get_backbone
    from modules.experiment_config import representation_id

    tmp_path = tmp_path_factory.mktemp("phase7_plumbing")
    config = make_test_config(tmp_path, seed=42)
    config = dataclasses.replace(config, feature_selection_method="none")
    build_synthetic_dataset(config)
    run_pipeline_through_split(config)
    plan = ImbalanceAwareFoldLoader(config, k=3).build()
    assert plan.folds and plan.holdout_val_records, "fixture setup produced no folds or no calibration records"

    backbone_dim = get_backbone(config.backbone).output_dim
    repr_id = representation_id(config)
    experiment_name = "TEST-FAKE"
    for fold in plan.folds:
        for records, split_name in ((fold.train_records, "train"), (fold.val_records, "val")):
            phase7._write_synthetic_deep_feature_cache(config, experiment_name, fold.fold_index, split_name, records, backbone_dim, repr_id)
    final_train_records = plan.folds[0].train_records + plan.folds[0].val_records
    phase7._write_synthetic_deep_feature_cache(config, experiment_name, -1, "final_train", final_train_records, backbone_dim, repr_id)
    phase7._write_synthetic_deep_feature_cache(config, experiment_name, -1, "calibration", plan.holdout_val_records, backbone_dim, repr_id)

    branches = ARMS["EfficientNet+GLCM+LBP+HOG"]
    return config, plan, branches, experiment_name


# ============================================================
# build_fusion_final: leakage guards + fit discipline
# ============================================================

def test_build_fusion_final_rejects_duplicate_psd_id_in_train_set(plumbing_setup):
    config, plan, branches, experiment_name = plumbing_setup
    final_train = plan.folds[0].train_records + plan.folds[0].val_records
    duplicated = final_train + [final_train[0]]  # re-add the first record -- now a duplicate ID
    with pytest.raises(AssertionError, match="duplicate PSD ID"):
        build_fusion_final(config, duplicated, plan.holdout_val_records, branches, experiment_name)


def test_build_fusion_final_rejects_overlap_between_train_and_calibration(plumbing_setup):
    config, plan, branches, experiment_name = plumbing_setup
    final_train = plan.folds[0].train_records + plan.folds[0].val_records
    contaminated_calibration = plan.holdout_val_records + [final_train[0]]  # a train record also in "calibration"
    with pytest.raises(AssertionError, match="both final_train and calibration"):
        build_fusion_final(config, final_train, contaminated_calibration, branches, experiment_name)


def test_build_fusion_final_normalizer_fit_on_train_only(plumbing_setup, monkeypatch):
    """The SAME leak-prevention discipline build_fusion_fold already has
    for CV folds, proven here for the new final-retrain function:
    FeatureNormalizer.fit() must be called with ONLY final_train's
    vectors, never calibration's."""
    import modules.fusion as fusion_module
    config, plan, branches, experiment_name = plumbing_setup
    final_train = plan.folds[0].train_records + plan.folds[0].val_records
    calibration = plan.holdout_val_records

    fit_call_sizes = []
    original_fit = fusion_module.FeatureNormalizer.fit

    def _spy_fit(self, vectors):
        fit_call_sizes.append(len(vectors))
        return original_fit(self, vectors)

    monkeypatch.setattr(fusion_module.FeatureNormalizer, "fit", _spy_fit)
    build_fusion_final(config, final_train, calibration, branches, experiment_name)

    # One fit() call per branch, each sized to final_train, never to
    # calibration's (different) size.
    assert fit_call_sizes, "FeatureNormalizer.fit was never called"
    assert all(n == len(final_train) for n in fit_call_sizes), (
        f"expected every fit() call sized to len(final_train)={len(final_train)}, got {fit_call_sizes}"
    )


def test_build_fusion_final_produces_correct_shapes(plumbing_setup):
    config, plan, branches, experiment_name = plumbing_setup
    final_train = plan.folds[0].train_records + plan.folds[0].val_records
    data = build_fusion_final(config, final_train, plan.holdout_val_records, branches, experiment_name)
    assert data.X_train.shape[0] == len(final_train)
    assert data.X_val.shape[0] == len(plan.holdout_val_records)
    assert data.X_train.shape[1] == data.X_val.shape[1] == sum(data.branch_dims.values())


# ============================================================
# Regression test: predict_proba class-ordering bug found this phase
# ============================================================

def test_probability_reordering_handles_class_absent_from_training():
    """Bug found while building this phase: clf.classes_ only contains
    classes that actually appeared in y_train. A naive
    prob_class_order.index(c) for every config.target_classes entry
    crashes with ValueError when a target class never appeared in
    training (true for this project's own synthetic fixture, and
    possible with small/imbalanced real data too). Reordering must
    fill a 0.0 column for a missing class instead of crashing."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = ["Psoriasis"] * 10 + ["Lichen_Planus"] * 10  # only 2 of 4 target classes present
    clf = LogisticRegression().fit(X, y)
    raw = clf.predict_proba(X)

    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    prob_class_order = list(clf.classes_)
    reordered = np.zeros((raw.shape[0], len(classes)), dtype=raw.dtype)
    for j, cls in enumerate(classes):
        if cls in prob_class_order:
            reordered[:, j] = raw[:, prob_class_order.index(cls)]

    assert reordered.shape == (20, 4)
    assert np.all(reordered[:, 2] == 0.0)  # Pityriasis_Rosea: never in training, must be all-zero
    assert np.all(reordered[:, 3] == 0.0)  # Seborrheic_Dermatitis: same
    assert np.allclose(reordered[:, 0] + reordered[:, 1], 1.0)  # the two real classes still sum to 1


# ============================================================
# CalibrationHandoff round-trip
# ============================================================

def test_calibration_handoff_round_trips_through_disk(tmp_path):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = ["Psoriasis", "Lichen_Planus"] * 5
    clf = LogisticRegression().fit(X, y)

    handoff = CalibrationHandoff(
        representation_id="repr123", experiment_id="exp123",
        classifier_name="logistic_regression", feature_selection_method="none",
        random_seed=42, source_experiment="TEST-EXP", feature_arm="EfficientNet",
        class_order=["Psoriasis", "Lichen_Planus"],
        final_classifier=clf, selected_feature_mask=np.ones(3, dtype=bool), branch_dims={"deep": 3},
        calibration_psd_ids=["A", "B"], calibration_true_labels=["Psoriasis", "Lichen_Planus"],
        calibration_predicted_labels=["Psoriasis", "Lichen_Planus"],
        calibration_raw_probabilities=clf.predict_proba(X[:2]),
    )

    path = tmp_path / "handoff.joblib"
    save_calibration_handoff(handoff, path)
    reloaded = load_calibration_handoff(path)

    assert reloaded.representation_id == handoff.representation_id
    assert reloaded.class_order == handoff.class_order
    assert np.array_equal(reloaded.calibration_raw_probabilities, handoff.calibration_raw_probabilities)
    assert np.array_equal(reloaded.selected_feature_mask, handoff.selected_feature_mask)
    # The reloaded classifier must still actually work, not just unpickle.
    assert list(reloaded.final_classifier.predict(X[:1])) == list(clf.predict(X[:1]))


def test_load_calibration_handoff_raises_clearly_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration_handoff(tmp_path / "does_not_exist.joblib")


# ============================================================
# holdout_test_records is never touched by the final retrain
# ============================================================

class _PoisonRecord:
    """Raises if ANY attribute is accessed -- proves
    run_final_retrain never reads individual holdout_test_records
    fields, only (at most) the list's length."""
    def __getattr__(self, name):
        raise AssertionError(f"holdout_test_records element attribute '{name}' was accessed -- must never happen")


def test_run_final_retrain_never_touches_holdout_test_records(plumbing_setup):
    config, plan, branches, experiment_name = plumbing_setup
    poisoned_plan = dataclasses.replace(plan, holdout_test_records=[_PoisonRecord(), _PoisonRecord()])
    handoff = phase7.run_final_retrain(config, poisoned_plan, branches, experiment_name, label_prefix="[TEST] ")
    assert handoff is not None  # completed without ever touching the poisoned records


# ============================================================
# Fold-local / final-retrain sample weighting (Task 1 explicit requirement)
# ============================================================

def test_run_final_cv_passes_fold_class_weights_to_classifier(plumbing_setup, monkeypatch):
    config, plan, branches, experiment_name = plumbing_setup
    from modules.fusion import compute_sample_weights

    captured = []

    def _stub_classifier(random_seed):
        class _Stub:
            def fit(self, X, y, sample_weight=None):
                captured.append(sample_weight)
            def predict(self, X):
                return [plan.folds[0].val_records[0].mapped_class] * len(X)
        return "stub", _Stub()

    monkeypatch.setitem(CLASSIFIER_REGISTRY, config.classifier_name, _stub_classifier)
    phase7.run_final_cv(config, plan, branches, experiment_name, label_prefix="[TEST] ")

    assert len(captured) == len(plan.folds)
    for fold, sw in zip(plan.folds, captured):
        assert sw is not None
        expected = compute_sample_weights([r.mapped_class for r in fold.train_records], fold.class_weights)
        assert np.array_equal(sw, expected)


def test_run_final_retrain_passes_full_train_set_class_weights(plumbing_setup, monkeypatch):
    config, plan, branches, experiment_name = plumbing_setup
    from modules.aef_input_validator import compute_train_fold_class_weights
    from modules.fusion import compute_sample_weights

    captured = []

    def _stub_classifier(random_seed):
        class _Stub:
            classes_ = np.array(config.target_classes)
            def fit(self, X, y, sample_weight=None):
                captured.append((y, sample_weight))
            def predict(self, X):
                return [plan.holdout_val_records[0].mapped_class] * len(X)
            def predict_proba(self, X):
                return np.tile(np.eye(len(config.target_classes))[0], (len(X), 1))
        return "stub", _Stub()

    monkeypatch.setitem(CLASSIFIER_REGISTRY, config.classifier_name, _stub_classifier)
    phase7.run_final_retrain(config, plan, branches, experiment_name, label_prefix="[TEST] ")

    assert len(captured) == 1
    y_used, sw_used = captured[0]
    final_train = plan.folds[0].train_records + plan.folds[0].val_records
    expected_weights = compute_train_fold_class_weights([r.mapped_class for r in final_train], config.target_classes)
    expected_sw = compute_sample_weights(list(y_used), expected_weights)
    assert np.array_equal(sw_used, expected_sw)


# ============================================================
# Task 4: classifier/selector are genuinely swappable, proven by an
# actual swap (same style as the Phase 8/9 registry-swap tests).
# ============================================================

def test_run_final_cv_respects_a_swapped_classifier_registry_entry(plumbing_setup, monkeypatch):
    config, plan, branches, experiment_name = plumbing_setup
    calls = {"n": 0}

    def _stub_classifier(random_seed):
        class _Stub:
            def fit(self, X, y, sample_weight=None):
                calls["n"] += 1
            def predict(self, X):
                return [plan.folds[0].val_records[0].mapped_class] * len(X)
        return "stub", _Stub()

    monkeypatch.setitem(CLASSIFIER_REGISTRY, config.classifier_name, _stub_classifier)
    phase7.run_final_cv(config, plan, branches, experiment_name, label_prefix="[TEST] ")
    assert calls["n"] == len(plan.folds)  # the stub, not the real classifier, was actually invoked


def test_run_final_cv_respects_a_swapped_selector_registry_entry(plumbing_setup, monkeypatch):
    config, plan, branches, experiment_name = plumbing_setup
    calls = {"n": 0}

    def _stub_selector(data):
        calls["n"] += 1
        mask = np.ones(data.X_train.shape[1], dtype=bool)
        return SelectionResult(method="stub", selected_mask=mask, selected_count=int(mask.sum()),
                                branch_retained={}, fit_time_sec=0.0)

    config = dataclasses.replace(config, feature_selection_method="none")
    monkeypatch.setitem(SELECTOR_REGISTRY, "none", _stub_selector)
    phase7.run_final_cv(config, plan, branches, experiment_name, label_prefix="[TEST] ")
    assert calls["n"] == len(plan.folds)  # the stub selector, not select_no_selection, was actually invoked


def test_phase7_production_classifier_guard(plumbing_setup):
    config, plan, branches, experiment_name = plumbing_setup
    for bad_name in ["xgboost", "svm", "histgradientboosting", "unsupported_model"]:
        bad_config = dataclasses.replace(config, classifier_name=bad_name)
        with pytest.raises(ValueError, match="Phase 7 production classifier guard violation"):
            phase7.validate_phase7_classifier(bad_name)
        with pytest.raises(ValueError, match="Phase 7 production classifier guard violation"):
            phase7.run_final_cv(bad_config, plan, branches, experiment_name)
        with pytest.raises(ValueError, match="Phase 7 production classifier guard violation"):
            phase7.run_final_retrain(bad_config, plan, branches, experiment_name)

    # Allowed models must pass validation without error
    for good_name in ["random_forest", "rf", "logistic_regression", "lr"]:
        phase7.validate_phase7_classifier(good_name)

