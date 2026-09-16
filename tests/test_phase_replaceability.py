"""
tests/test_phase_replaceability.py

Targeted tests for THIS phase's changes:

  1. The sample-weight bug fix: evaluate_selection now passes the
     EXACT fold-local per-sample weights to the classifier's fit(),
     not just "some non-None value."
  2. run_selector's uniform (name, data, config, seed) interface
     produces results identical to calling the underlying select_*
     function directly -- proving the dispatch table is a pure
     re-routing layer, not a behavior change.
  3. Adding a brand-new handcrafted feature branch (never seen before)
     resolves correctly through modules.fusion._load_branch_vectors
     WITHOUT any edit to fusion.py -- Step 6 item 3, proven, not
     claimed.
  4. representation_id/experiment_id separation: changing classifier
     or selector leaves representation_id unchanged (the cache-reuse
     property); changing backbone/preprocessing/seed changes it.
  5. Structural decoupling: modules.feature_selection never imports
     anything classifier-related from modules.fusion.

Does not re-test what tests/test_deep_feature_contract.py already
covers (cache validation, backbone identity, the classifier/selector
swap-proof pattern) or what tests/test_experiment_framework.py already
covers (registry existence, validate_experiment_config).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from config.config import get_config
from modules.feature_selection import (
    SelectionResult, run_selector, select_no_selection, select_rfe,
)
from modules.experiment_config import representation_id, experiment_id


def _fake_fusion_fold_data(n_train=40, n_val=10, dim=10):
    from modules.fusion import FusionFoldData
    rng = np.random.default_rng(0)
    classes = ["Psoriasis", "Lichen_Planus"]
    return FusionFoldData(
        X_train=rng.normal(size=(n_train, dim)), X_val=rng.normal(size=(n_val, dim)),
        y_train=[classes[i % 2] for i in range(n_train)], y_val=[classes[i % 2] for i in range(n_val)],
        psd_ids_train=[f"T{i}" for i in range(n_train)], psd_ids_val=[f"V{i}" for i in range(n_val)],
        branch_dims={"deep": dim},
    )


# ============================================================
# 1. Fold-local sample weighting (the found bug), precisely
# ============================================================

def test_evaluate_selection_passes_the_exact_fold_local_sample_weights(monkeypatch):
    import modules.fusion as fusion_module
    import run_aef_crc_phase6 as phase6
    from modules.fusion import compute_sample_weights

    data = _fake_fusion_fold_data(dim=6)
    mask = np.ones(6, dtype=bool)
    selection = SelectionResult(method="no_selection", selected_mask=mask, selected_count=6,
                                 branch_retained={"deep": 6}, fit_time_sec=0.0)
    class_weights = {"Psoriasis": 2.5, "Lichen_Planus": 0.5}  # deliberately imbalanced, not 1.0/1.0

    captured = {}

    def _stub_classifier(random_seed):
        class _Stub:
            def fit(self, X, y, sample_weight=None):
                captured["sample_weight"] = sample_weight
            def predict(self, X):
                return [data.y_val[0]] * len(X)
        return "stub_backend", _Stub()

    monkeypatch.setitem(fusion_module.CLASSIFIER_REGISTRY, "logistic_regression", _stub_classifier)
    config = dataclasses.replace(get_config(), classifier_name="logistic_regression")
    phase6.evaluate_selection(config, data, selection, config.target_classes, class_weights)

    expected = compute_sample_weights(data.y_train, class_weights)
    assert np.array_equal(captured["sample_weight"], expected)
    assert len(set(captured["sample_weight"])) > 1  # genuinely non-uniform, not accidentally all-1.0


def test_evaluate_selection_requires_class_weights_explicitly():
    """No default -- a caller cannot silently regress to the
    no-sample-weight bug by forgetting the argument; it's a TypeError,
    not a silent fallback."""
    import run_aef_crc_phase6 as phase6
    data = _fake_fusion_fold_data(dim=4)
    mask = np.ones(4, dtype=bool)
    selection = SelectionResult(method="no_selection", selected_mask=mask, selected_count=4,
                                 branch_retained={"deep": 4}, fit_time_sec=0.0)
    config = get_config()
    with pytest.raises(TypeError):
        phase6.evaluate_selection(config, data, selection, config.target_classes)


# ============================================================
# 2. run_selector's uniform interface == the direct call
# ============================================================

def test_run_selector_no_selection_matches_direct_call():
    data = _fake_fusion_fold_data(dim=10)
    config = get_config()
    direct = select_no_selection(data)
    via_dispatch = run_selector("none", data, config, config.random_seed)
    assert direct.method == via_dispatch.method
    assert np.array_equal(direct.selected_mask, via_dispatch.selected_mask)


def test_run_selector_rfe_matches_direct_call_with_the_same_config_values():
    data = _fake_fusion_fold_data(dim=10, n_train=40)
    config = get_config()
    direct = select_rfe(data, config.feature_selection_top_k, config.rfe_step, config.random_seed)
    via_dispatch = run_selector("rfe", data, config, config.random_seed)
    assert np.array_equal(direct.selected_mask, via_dispatch.selected_mask)


def test_run_selector_rejects_unknown_method_by_name():
    data = _fake_fusion_fold_data(dim=6)
    config = get_config()
    with pytest.raises(ValueError, match="Unknown feature_selection method"):
        run_selector("not_a_real_method", data, config, config.random_seed)


def test_run_aef_crc_phase6_no_longer_imports_selector_registry_directly():
    """The runner calls run_selector() only -- it never indexes
    SELECTOR_REGISTRY with a method-specific argument list of its own
    (that knowledge now lives entirely in feature_selection.py)."""
    import run_aef_crc_phase6 as phase6
    assert not hasattr(phase6, "SELECTOR_REGISTRY")
    assert hasattr(phase6, "run_selector")


# ============================================================
# 3. Adding a new feature branch requires no fusion.py edit
# ============================================================

def test_adding_a_dummy_handcrafted_branch_requires_no_fusion_changes(tmp_path):
    """Registers a brand-new branch name -- not glcm/lbp/hog -- directly
    in HANDCRAFTED_BRANCH_REGISTRY, and confirms
    modules.fusion._load_branch_vectors resolves it correctly through
    the exact same code path used for the three real descriptors,
    without any edit to fusion.py. This is the concrete proof for Step
    6 item 3, not just that the registry exists."""
    from modules.fusion import _load_branch_vectors
    from modules.handcrafted_features import HANDCRAFTED_BRANCH_REGISTRY, _no_reduce
    from modules.synthetic_fixtures import make_test_config

    class _StubRecord:
        def __init__(self, psd_id):
            self.psd_id = psd_id
            self.file_path = tmp_path / f"{psd_id}.jpg"  # deliberately does not exist -- see extractor stub below

    class _StubFeatureVectors:
        dummy_descriptor = np.array([1.0, 2.0, 3.0])

    class _StubExtractor:
        def extract_and_cache(self, image, psd_id):
            return _StubFeatureVectors()  # ignores the (None, since the file doesn't exist) image -- a pure stub

    HANDCRAFTED_BRANCH_REGISTRY["dummy_descriptor"] = (lambda fv: fv.dummy_descriptor, _no_reduce)
    try:
        config = make_test_config(tmp_path, seed=1)
        records = [_StubRecord("A"), _StubRecord("B")]
        vecs = _load_branch_vectors(config, records, "dummy_descriptor", "EXP", 0, "train", _StubExtractor())
        assert len(vecs) == 2
        assert np.array_equal(vecs[0], np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(vecs[1], np.array([1.0, 2.0, 3.0]))
    finally:
        del HANDCRAFTED_BRANCH_REGISTRY["dummy_descriptor"]


def test_unregistered_branch_name_still_fails_loudly():
    """The registry lookup replaced an if-chain that ended in `raise
    ValueError` -- confirm that failure mode survived the refactor."""
    from modules.fusion import _load_branch_vectors
    with pytest.raises(ValueError, match="Unknown branch"):
        _load_branch_vectors(get_config(), [], "not_a_real_branch", "EXP", 0, "train", extractor=None)


# ============================================================
# 4. representation_id vs experiment_id separation
# ============================================================

def test_representation_id_stable_across_classifier_and_selector_changes():
    base = get_config()
    changed_classifier = dataclasses.replace(base, classifier_name="svm")
    changed_selector = dataclasses.replace(base, feature_selection_method="rfe")

    assert representation_id(base) == representation_id(changed_classifier) == representation_id(changed_selector)
    # But the FULL experiment identity still tells them apart -- this is
    # not "the id is broken," it's "the id has two granularities now."
    assert experiment_id(base) != experiment_id(changed_classifier)
    assert experiment_id(base) != experiment_id(changed_selector)


def test_representation_id_changes_with_backbone_preprocessing_or_seed():
    base = get_config()
    assert representation_id(base) != representation_id(dataclasses.replace(base, backbone="convnext_tiny"))
    assert representation_id(base) != representation_id(dataclasses.replace(base, preprocessing_mode="standard"))
    assert representation_id(base) != representation_id(dataclasses.replace(base, random_seed=7))


# ============================================================
# 5. Structural decoupling
# ============================================================

def test_feature_selection_module_has_no_classifier_coupling():
    """modules.feature_selection must never import build_classifier or
    CLASSIFIER_REGISTRY from modules.fusion -- selection and
    classification are separate concerns; only run_aef_crc_phase6.py
    (the runner) is allowed to know about both."""
    import inspect
    import modules.feature_selection as fs
    source = inspect.getsource(fs)
    assert "build_classifier" not in source
    assert "CLASSIFIER_REGISTRY" not in source
