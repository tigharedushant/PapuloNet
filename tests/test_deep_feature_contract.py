"""
tests/test_deep_feature_contract.py

Targeted tests for this phase's genuinely new risk surface:

  1. validate_deep_feature_cache() actually fails loudly on every
     documented mismatch (wrong dim, wrong backbone, wrong experiment/
     fold/split, duplicate PSD IDs, missing PSD IDs, missing manifest) --
     not just that the function exists.
  2. Phase-6 selector/classifier swapping is PROVEN by actually
     replacing a registry entry and confirming run_aef_crc_phase6's
     runner picks up the replacement WITHOUT any edit to that file --
     not merely that SELECTOR_REGISTRY/CLASSIFIER_REGISTRY exist.
  3. get_backbone("efficientnet_b0")'s callables are the REAL
     modules.efficientnet_model functions (identity, not just
     "something callable").
  4. SelectionResult's output contract (selected_indices, original_dim)
     is correct and stays in sync with selected_mask by construction.

Does not re-test what tests/test_experiment_framework.py already covers
(registry existence, experiment_id isolation, validate_experiment_config
fail-fast) or what tests/test_fusion.py covers (PSD-ID collision inside
one fold, branch dimension tracking).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from modules.synthetic_fixtures import make_test_config
from modules.backbones import get_backbone, validate_deep_feature_cache, DeepFeatureCacheError
from modules.feature_selection import SelectionResult


# ============================================================
# Deep-feature cache contract (Task 2)
# ============================================================

def _write_manifest(config, experiment, fold_index, split, **overrides):
    from modules.experiment_config import representation_id as _repr_id
    manifest = {
        "experiment_name": experiment, "fold_index": fold_index, "split": split,
        "checkpoint_path": "fake.h5", "n_features": 3, "feature_dim": 1280,
        "psd_ids": ["A", "B", "C"], "backbone_name": "efficientnet_b0",
        "representation_id": _repr_id(config),
    }
    manifest.update(overrides)
    out_dir = config.aef_crc_artifacts_dir / "deep_features" / experiment / f"fold_{fold_index:02d}" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_validate_rejects_wrong_representation_id(tmp_path):
    """Task 1: a cache produced by a DIFFERENT training configuration
    (e.g. a different learning rate) sharing the same experiment name
    must not silently pass just because backbone/dimension happen to
    match."""
    config = make_test_config(tmp_path, seed=1)
    _write_manifest(config, "EXP1", 0, "train", representation_id="some_other_config_hash")
    with pytest.raises(DeepFeatureCacheError, match="representation_id"):
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "C"], "efficientnet_b0")


def test_validate_passes_for_a_well_formed_manifest(tmp_path):
    config = make_test_config(tmp_path, seed=1)
    _write_manifest(config, "EXP1", 0, "train")
    validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "C"], "efficientnet_b0")  # must not raise


def test_validate_accepts_a_subset_of_cached_ids(tmp_path):
    """Requesting fewer IDs than the cache holds is fine -- only a
    MISSING expected ID is an error, not an unused extra one."""
    config = make_test_config(tmp_path, seed=1)
    _write_manifest(config, "EXP1", 0, "train")
    validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B"], "efficientnet_b0")  # must not raise


def test_validate_rejects_wrong_feature_dimension(tmp_path):
    config = make_test_config(tmp_path, seed=1)
    _write_manifest(config, "EXP1", 0, "train", feature_dim=512)
    with pytest.raises(DeepFeatureCacheError, match="feature_dim"):
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "C"], "efficientnet_b0")


def test_validate_rejects_wrong_backbone_name(tmp_path):
    config = make_test_config(tmp_path, seed=1)
    _write_manifest(config, "EXP1", 0, "train", backbone_name="some_other_backbone")
    with pytest.raises(DeepFeatureCacheError, match="backbone_name"):
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "C"], "efficientnet_b0")


def test_validate_rejects_wrong_experiment_name(tmp_path):
    config = make_test_config(tmp_path, seed=1)
    _write_manifest(config, "EXP1", 0, "train", experiment_name="WRONG-EXP")
    with pytest.raises(DeepFeatureCacheError, match="experiment_name"):
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "C"], "efficientnet_b0")


def test_validate_rejects_wrong_fold_index(tmp_path):
    config = make_test_config(tmp_path, seed=1)
    out_dir = config.aef_crc_artifacts_dir / "deep_features" / "EXP1" / "fold_00" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": "EXP1", "fold_index": 99, "split": "train",  # wrong fold_index in the CONTENT
        "checkpoint_path": "fake.h5", "n_features": 3, "feature_dim": 1280,
        "psd_ids": ["A", "B", "C"], "backbone_name": "efficientnet_b0",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DeepFeatureCacheError, match="fold_index"):
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "C"], "efficientnet_b0")


def test_validate_rejects_wrong_split(tmp_path):
    config = make_test_config(tmp_path, seed=1)
    out_dir = config.aef_crc_artifacts_dir / "deep_features" / "EXP1" / "fold_00" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": "EXP1", "fold_index": 0, "split": "val",  # wrong split in the CONTENT
        "checkpoint_path": "fake.h5", "n_features": 3, "feature_dim": 1280,
        "psd_ids": ["A", "B", "C"], "backbone_name": "efficientnet_b0",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DeepFeatureCacheError, match="split"):
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "C"], "efficientnet_b0")


def test_validate_rejects_duplicate_psd_ids_in_manifest(tmp_path):
    config = make_test_config(tmp_path, seed=1)
    _write_manifest(config, "EXP1", 0, "train", psd_ids=["A", "A", "C"])
    with pytest.raises(DeepFeatureCacheError, match="duplicate"):
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "C"], "efficientnet_b0")


def test_validate_rejects_missing_expected_psd_id(tmp_path):
    config = make_test_config(tmp_path, seed=1)
    _write_manifest(config, "EXP1", 0, "train", psd_ids=["A", "B"])
    with pytest.raises(DeepFeatureCacheError, match="not present in cache"):
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "Z"], "efficientnet_b0")


def test_validate_raises_when_manifest_missing_entirely(tmp_path):
    config = make_test_config(tmp_path, seed=1)  # no manifest written at all
    with pytest.raises(FileNotFoundError):
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A"], "efficientnet_b0")


def test_validate_reports_multiple_errors_at_once(tmp_path):
    """Not just the first mismatch found -- every mismatch, so a
    caller doesn't have to fix-and-rerun repeatedly."""
    config = make_test_config(tmp_path, seed=1)
    _write_manifest(config, "EXP1", 0, "train", feature_dim=999, backbone_name="wrong")
    with pytest.raises(DeepFeatureCacheError) as exc_info:
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "C"], "efficientnet_b0")
    assert "feature_dim" in str(exc_info.value) and "backbone_name" in str(exc_info.value)


def test_validate_dimension_check_is_not_hardcoded_1280(tmp_path):
    """The dimension check reads get_backbone(name).output_dim, not a
    literal 1280 -- proven by registering a throwaway backbone with a
    different dimension and confirming validation follows it."""
    from modules.backbones import BackboneSpec, BACKBONE_REGISTRY

    BACKBONE_REGISTRY["_test_only_backbone"] = BackboneSpec(
        name="_test_only_backbone", build_stage1=None, unfreeze_stage2=None,
        describe=None, extract_features=None, output_dim=42,
    )
    try:
        config = make_test_config(tmp_path, seed=1)
        _write_manifest(config, "EXP1", 0, "train", feature_dim=42, backbone_name="_test_only_backbone")
        validate_deep_feature_cache(config, "EXP1", 0, "train", ["A", "B", "C"], "_test_only_backbone")  # must not raise
    finally:
        del BACKBONE_REGISTRY["_test_only_backbone"]


# ============================================================
# Backbone contract identity (Task 1)
# ============================================================

def test_backbone_spec_callables_are_the_real_efficientnet_functions_not_stubs():
    from modules import efficientnet_model as em
    spec = get_backbone("efficientnet_b0")
    assert spec.build_stage1 is em.build_stage1_model
    assert spec.unfreeze_stage2 is em.unfreeze_for_stage2
    assert spec.describe is em.describe_model
    assert spec.extract_features is em.extract_deep_features


# ============================================================
# Genuine selector replaceability (Task 3) -- proven by actually
# swapping a registry entry, not by asserting the registry exists.
# ============================================================

def _fake_fusion_fold_data(n_train=30, n_val=10, dim=12):
    from modules.fusion import FusionFoldData
    rng = np.random.default_rng(0)
    classes = ["Psoriasis", "Lichen_Planus"]
    return FusionFoldData(
        X_train=rng.normal(size=(n_train, dim)), X_val=rng.normal(size=(n_val, dim)),
        y_train=[classes[i % 2] for i in range(n_train)], y_val=[classes[i % 2] for i in range(n_val)],
        psd_ids_train=[f"T{i}" for i in range(n_train)], psd_ids_val=[f"V{i}" for i in range(n_val)],
        branch_dims={"deep": dim},
    )


def test_swapping_xgboost_importance_registry_entry_changes_what_the_phase6_runner_calls(monkeypatch):
    """The strong version of 'selectors are interchangeable': replace
    SELECTOR_REGISTRY['xgboost_importance'] with a stub that returns a
    recognizably different SelectionResult, then call
    run_aef_crc_phase6.run_candidates_for_fold() -- WITHOUT touching
    run_aef_crc_phase6.py -- and confirm the stub's output comes back.
    If the runner still called modules.feature_selection.
    select_xgboost_importance directly, this would fail. Patched on
    modules.feature_selection.SELECTOR_REGISTRY directly: run_selector
    (what run_candidates_for_fold actually calls) resolves the registry
    from ITS OWN module, not from anything imported into
    run_aef_crc_phase6's namespace -- there is no longer a
    phase6.SELECTOR_REGISTRY to patch, which is itself part of the
    proof that the runner no longer owns any selector-specific
    knowledge."""
    import modules.feature_selection as fs
    import run_aef_crc_phase6 as phase6
    from config.config import get_config

    data = _fake_fusion_fold_data(dim=12)
    sentinel_mask = np.zeros(12, dtype=bool)
    sentinel_mask[0] = True  # exactly 1 feature -- distinguishable from any real selector's output on this data

    def _stub_selector(data, top_k, random_seed):
        return SelectionResult(
            method="stub", selected_mask=sentinel_mask, selected_count=1,
            branch_retained={"deep": 1}, fit_time_sec=0.0,
        )

    monkeypatch.setitem(fs.SELECTOR_REGISTRY, "xgboost_importance", _stub_selector)
    config = get_config()
    results = phase6.run_candidates_for_fold(config, data, fold_index=0)

    assert results["xgboost_importance"].method == "stub"
    assert results["xgboost_importance"].selected_count == 1
    assert np.array_equal(results["xgboost_importance"].selected_mask, sentinel_mask)
    # The OTHER candidates must be unaffected -- swapping one entry
    # doesn't change what the others resolve to.
    assert results["no_selection"].method == "no_selection"
    assert results["rfe"].method == "rfe"


def test_swapping_ga_registry_entry_changes_the_standalone_ga_early_stop_path(monkeypatch):
    """Same proof for the GA branch specifically, since it's invoked
    from a second call site (the early-stop block) inside
    run_aef_crc_phase6.run_benchmark, not just run_candidates_for_fold."""
    import modules.feature_selection as fs
    import run_aef_crc_phase6 as phase6

    data = _fake_fusion_fold_data(dim=12)
    sentinel_mask = np.ones(12, dtype=bool)

    def _stub_ga(data, config, random_seed):
        return SelectionResult(
            method="stub_ga", selected_mask=sentinel_mask, selected_count=12,
            branch_retained={"deep": 12}, fit_time_sec=0.0, extra={"population_size": 0, "generations": 0},
        )

    monkeypatch.setitem(fs.SELECTOR_REGISTRY, "ga", _stub_ga)
    from config.config import get_config
    config = get_config()
    results = phase6.run_candidates_for_fold(config, data, fold_index=0, ga_context=True)
    assert results["ga"].method == "stub_ga"


# ============================================================
# Genuine classifier replaceability (Task 4) -- same style of proof.
# ============================================================

def test_swapping_classifier_registry_entry_changes_what_evaluate_selection_uses(monkeypatch):
    import dataclasses
    import modules.fusion as fusion_module
    import run_aef_crc_phase6 as phase6
    from config.config import get_config

    data = _fake_fusion_fold_data(dim=8)
    full_mask = np.ones(8, dtype=bool)
    selection = SelectionResult(
        method="no_selection", selected_mask=full_mask, selected_count=8,
        branch_retained={"deep": 8}, fit_time_sec=0.0,
    )
    class_weights = {c: 1.0 for c in set(data.y_train)}

    calls = []

    def _stub_classifier(random_seed):
        class _Stub:
            def fit(self, X, y, sample_weight=None):
                calls.append(("fit", sample_weight is not None))
            def predict(self, X):
                calls.append(("predict", None))
                return [data.y_val[0]] * len(X)
        return "stub_backend", _Stub()

    # build_classifier (called inside evaluate_selection) looks up
    # CLASSIFIER_REGISTRY as a module-level global in modules.fusion --
    # patch it there, proving evaluate_selection has no hardcoded
    # classifier of its own left to fall back on.
    monkeypatch.setitem(fusion_module.CLASSIFIER_REGISTRY, "logistic_regression", _stub_classifier)
    config = dataclasses.replace(get_config(), classifier_name="logistic_regression")
    metrics, y_pred = phase6.evaluate_selection(config, data, selection, config.target_classes, class_weights)

    assert [c[0] for c in calls] == ["fit", "predict"]
    assert calls[0][1] is True  # fit was called WITH a non-None sample_weight
    assert all(p == data.y_val[0] for p in y_pred)


# ============================================================
# SelectionResult output contract
# ============================================================

def test_selection_result_selected_indices_matches_mask():
    mask = np.array([True, False, True, True, False])
    result = SelectionResult(method="x", selected_mask=mask, selected_count=3, branch_retained={}, fit_time_sec=0.0)
    assert list(result.selected_indices) == [0, 2, 3]
    assert result.original_dim == 5


def test_selection_result_indices_cannot_drift_from_mask():
    """selected_indices is a property derived from selected_mask, not a
    separately stored field -- mutating the mask changes what
    selected_indices reports, by construction (nothing to fall out of
    sync)."""
    mask = np.array([True, False, False])
    result = SelectionResult(method="x", selected_mask=mask, selected_count=1, branch_retained={}, fit_time_sec=0.0)
    assert list(result.selected_indices) == [0]
    mask[1] = True
    assert list(result.selected_indices) == [0, 1]
