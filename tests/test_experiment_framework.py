# SYNTHETIC_TEST_FIXTURE = True
# This test uses strictly synthetic, generated, or mock fixtures.
# Zero real dataset images or private research artifacts are required or accessed.
"""
tests/test_experiment_framework.py

Targeted tests for the modular-experiment-framework change: registry
resolution (backbone/classifier/selector), experiment_id isolation,
fail-fast validation, and that Phase 5's existing classifier call sites
are unaffected by build_classifier's new optional second argument.

All of this is testable without TensorFlow -- get_backbone()
returns a spec object (doesn't need to CALL the TF-dependent functions
to prove registry correctness), and CLASSIFIER_REGISTRY's non-default
entries (logistic_regression, svm, histgradientboosting) are all
sklearn, real and instantiable here.
"""

from __future__ import annotations

import dataclasses

from config.config import get_config
from modules.backbones import BACKBONE_REGISTRY, get_backbone, BackboneSpec
from modules.fusion import CLASSIFIER_REGISTRY, build_classifier
from modules.feature_selection import SELECTOR_REGISTRY, select_no_selection, select_xgboost_importance, select_rfe, select_ga
from modules.experiment_config import experiment_id, validate_experiment_config


# ============================================================
# Backbone registry
# ============================================================

def test_get_backbone_returns_efficientnet_spec():
    spec = get_backbone("efficientnet_b0")
    assert isinstance(spec, BackboneSpec)
    assert spec.name == "efficientnet_b0"
    assert spec.output_dim == 1280


def test_get_backbone_rejects_unknown_name():
    try:
        get_backbone("convnext_tiny")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "convnext_tiny" in str(e)


def test_backbone_spec_exposes_all_four_required_callables():
    spec = get_backbone("efficientnet_b0")
    for attr in ("build_stage1", "unfreeze_stage2", "describe", "extract_features"):
        assert callable(getattr(spec, attr)), f"{attr} is not callable"


# ============================================================
# Classifier registry
# ============================================================

def test_build_classifier_default_unchanged_signature_still_works():
    """Phase 5's existing call sites (build_classifier(config.random_seed),
    no second argument) must default to the primary Random Forest classifier."""
    backend, clf = build_classifier(42)
    assert backend == "random_forest"


def test_build_classifier_random_forest_contract():
    backend, clf = build_classifier(42, "random_forest")
    assert backend == "random_forest"
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(clf, RandomForestClassifier)
    params = clf.get_params()
    assert params["n_estimators"] == 300
    assert params["criterion"] == "gini"
    assert params["max_depth"] is None
    assert params["min_samples_split"] == 2
    assert params["min_samples_leaf"] == 1
    assert params["max_features"] == "sqrt"
    assert params["bootstrap"] is True
    assert params["class_weight"] is None
    assert params["random_state"] == 42
    assert params["n_jobs"] == -1


def test_build_classifier_logistic_regression_real_and_instantiable():
    backend, clf = build_classifier(42, "logistic_regression")
    assert backend == "logistic_regression"
    from sklearn.linear_model import LogisticRegression
    assert isinstance(clf, LogisticRegression)
    params = clf.get_params()
    assert params["penalty"] == "l2"
    assert params["C"] == 1.0
    assert params["solver"] == "lbfgs"
    assert params["max_iter"] == 1000
    assert params["class_weight"] is None
    assert params["random_state"] == 42


def test_build_classifier_svm_real_and_instantiable():
    backend, clf = build_classifier(42, "svm")
    assert backend == "svm"
    from sklearn.svm import SVC
    assert isinstance(clf, SVC)


def test_build_classifier_histgradientboosting_real_and_instantiable():
    backend, clf = build_classifier(42, "histgradientboosting")
    assert backend == "histgradientboosting"


def test_build_classifier_rejects_unknown_name():
    try:
        build_classifier(42, "unknown_unregistered_classifier")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "unknown_unregistered_classifier" in str(e)


def test_build_classifier_same_seed_same_registry_key_gives_identical_config():
    """Same seed + same classifier_name must be deterministic (same
    hyperparameters set), even though the object identity differs."""
    _, clf_a = build_classifier(42, "logistic_regression")
    _, clf_b = build_classifier(42, "logistic_regression")
    assert clf_a.get_params()["random_state"] == clf_b.get_params()["random_state"] == 42


# ============================================================
# Selector registry
# ============================================================

def test_selector_registry_keys_match_actual_functions():
    assert SELECTOR_REGISTRY["none"] is select_no_selection
    assert SELECTOR_REGISTRY["xgboost_importance"] is select_xgboost_importance
    assert SELECTOR_REGISTRY["rfe"] is select_rfe
    assert SELECTOR_REGISTRY["ga"] is select_ga


# ============================================================
# Experiment identity / isolation
# ============================================================

def test_experiment_id_deterministic_for_identical_config():
    cfg = get_config()
    assert experiment_id(cfg) == experiment_id(cfg)


def test_experiment_id_differs_when_backbone_differs():
    cfg = get_config()
    cfg2 = dataclasses.replace(cfg, backbone="a_different_backbone_name")
    assert experiment_id(cfg) != experiment_id(cfg2)


def test_experiment_id_differs_when_classifier_differs():
    cfg = get_config()
    cfg2 = dataclasses.replace(cfg, classifier_name="logistic_regression")
    assert experiment_id(cfg) != experiment_id(cfg2)


def test_experiment_id_differs_when_feature_selection_differs():
    cfg = get_config()
    cfg2 = dataclasses.replace(cfg, feature_selection_method="rfe")
    assert experiment_id(cfg) != experiment_id(cfg2)


def test_experiment_id_differs_when_preprocessing_differs():
    cfg = get_config()
    cfg2 = dataclasses.replace(cfg, preprocessing_mode="standard")
    assert experiment_id(cfg) != experiment_id(cfg2)


def test_experiment_id_differs_when_seed_differs():
    cfg = get_config()
    cfg2 = dataclasses.replace(cfg, random_seed=99)
    assert experiment_id(cfg) != experiment_id(cfg2)


def test_experiment_id_same_when_only_a_downstream_only_field_differs():
    """feature_selection_top_k only affects Phase 6 -- it has no bearing
    on what gets trained/cached in Phase 3, so it must not change the
    id. (This replaces a prior version of this test that used
    batch_size as its 'unrelated field' example -- batch_size is
    training configuration that materially affects the trained
    representation and was a real gap; see
    test_representation_id_now_includes_training_hyperparameters
    in tests/test_phase_replaceability.py for the fix and the
    regression test proving batch_size DOES now change the id.)"""
    cfg = get_config()
    cfg2 = dataclasses.replace(cfg, feature_selection_top_k=cfg.feature_selection_top_k + 50)
    assert experiment_id(cfg) == experiment_id(cfg2)


# ============================================================
# Fail-fast validation
# ============================================================

def test_validate_accepts_default_config():
    validate_experiment_config(get_config())  # must not raise


def test_validate_rejects_unknown_backbone():
    cfg = dataclasses.replace(get_config(), backbone="nonexistent")
    try:
        validate_experiment_config(cfg)
        assert False, "should have raised"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_validate_rejects_unknown_classifier():
    cfg = dataclasses.replace(get_config(), classifier_name="nonexistent")
    try:
        validate_experiment_config(cfg)
        assert False, "should have raised"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_validate_rejects_unknown_feature_selection_method():
    cfg = dataclasses.replace(get_config(), feature_selection_method="nonexistent")
    try:
        validate_experiment_config(cfg)
        assert False, "should have raised"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_validate_reports_all_errors_at_once_not_just_the_first():
    cfg = dataclasses.replace(get_config(), backbone="bad1", classifier_name="bad2")
    try:
        validate_experiment_config(cfg)
        assert False, "should have raised"
    except ValueError as e:
        assert "bad1" in str(e) and "bad2" in str(e)
