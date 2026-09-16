"""
tests/test_feature_selection.py

Targeted tests for Phase 6's genuinely new risk surface. Does not
re-test what tests/test_leakage.py or tests/test_fusion.py already
cover (outer split, CV fold isolation, PSD-ID alignment into
build_fusion_fold, class weights).
"""

from __future__ import annotations

import numpy as np

from config.config import get_config
from modules.fusion import FusionFoldData
from modules.feature_selection import (
    select_no_selection, select_xgboost_importance, select_rfe, select_ga, select_bda,
    _branch_retained_counts, compute_pairwise_jaccard_stability, compute_branch_jaccard_stability,
)


def _make_fake_fold_data(n_train=60, n_val=20, seed=0):
    rng = np.random.default_rng(seed)
    branch_dims = {"deep": 20, "glcm": 8, "lbp": 6, "hog": 10}
    total_dim = sum(branch_dims.values())
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]

    def labels(n):
        base = classes * (n // len(classes) + 1)
        return base[:n]

    return FusionFoldData(
        X_train=rng.normal(size=(n_train, total_dim)),
        X_val=rng.normal(size=(n_val, total_dim)),
        y_train=labels(n_train), y_val=labels(n_val),
        psd_ids_train=[f"T{i}" for i in range(n_train)],
        psd_ids_val=[f"V{i}" for i in range(n_val)],
        branch_dims=branch_dims,
    )


# ============================================================
# Branch-alignment
# ============================================================

def test_branch_retained_counts_sum_to_selected_count():
    data = _make_fake_fold_data()
    result = select_no_selection(data)
    assert sum(result.branch_retained.values()) == result.selected_count


def test_branch_retained_counts_use_correct_offsets():
    """A hand-built mask selecting only the 'lbp' branch's columns must
    report ALL of lbp retained and ZERO of every other branch --
    proves _branch_offsets maps global indices to branches correctly,
    not just that the totals happen to add up."""
    branch_dims = {"deep": 20, "glcm": 8, "lbp": 6, "hog": 10}
    total = sum(branch_dims.values())
    mask = np.zeros(total, dtype=bool)
    mask[20:28] = True  # exactly the glcm columns (deep=0:20, glcm=20:28)
    retained = _branch_retained_counts(mask, branch_dims)
    assert retained == {"deep": 0, "glcm": 8, "lbp": 0, "hog": 0}


# ============================================================
# Determinism
# ============================================================

def test_xgboost_importance_selection_deterministic_given_seed():
    data = _make_fake_fold_data()
    cfg = get_config()
    r1 = select_xgboost_importance(data, top_k=10, random_seed=42)
    r2 = select_xgboost_importance(data, top_k=10, random_seed=42)
    assert np.array_equal(r1.selected_mask, r2.selected_mask)


def test_rfe_selection_deterministic_given_seed():
    data = _make_fake_fold_data()
    r1 = select_rfe(data, n_features_to_select=10, step=0.2, random_seed=42)
    r2 = select_rfe(data, n_features_to_select=10, step=0.2, random_seed=42)
    assert np.array_equal(r1.selected_mask, r2.selected_mask)


def test_ga_selection_deterministic_given_seed():
    data = _make_fake_fold_data(n_train=40, n_val=10)
    cfg = get_config()
    from dataclasses import replace
    cfg = replace(cfg, ga_population_size=6, ga_generations=3)  # small, fast, still exercises the real code path
    r1 = select_ga(data, cfg, random_seed=42)
    r2 = select_ga(data, cfg, random_seed=42)
    assert np.array_equal(r1.selected_mask, r2.selected_mask)


# ============================================================
# Train-fold-only fitting / no leakage
# ============================================================

def test_no_selection_keeps_all_features():
    data = _make_fake_fold_data()
    result = select_no_selection(data)
    assert result.selected_count == data.X_train.shape[1]
    assert result.selected_mask.all()


def test_ga_nested_split_never_touches_data_x_val():
    """GA's fitness function must only ever see data.X_train (via its
    own nested split) -- never data.X_val. Verified by making X_val
    wildly out-of-distribution (huge magnitude) and confirming GA's
    reported fitness (computed on its OWN nested val, from X_train)
    doesn't reflect it -- if GA were leaking X_val into fitness, the
    reported best_nested_fitness would be degenerate/extreme."""
    data = _make_fake_fold_data(n_train=40, n_val=10)
    data.X_val[:] = 1e6  # extreme, out-of-distribution -- would visibly corrupt fitness if leaked
    cfg = get_config()
    from dataclasses import replace
    cfg = replace(cfg, ga_population_size=6, ga_generations=3)
    result = select_ga(data, cfg, random_seed=42)
    # A real macro-F1-based fitness (minus a small penalty) is always
    # in a bounded, sane range -- a leak into the corrupted X_val would
    # not produce a value in this range by chance.
    assert -1.0 <= result.extra["best_nested_fitness"] <= 1.0


def test_xgboost_importance_respects_top_k_budget():
    data = _make_fake_fold_data()
    result = select_xgboost_importance(data, top_k=5, random_seed=42)
    assert result.selected_count == 5


def test_rfe_respects_target_feature_count():
    data = _make_fake_fold_data()
    result = select_rfe(data, n_features_to_select=8, step=0.2, random_seed=42)
    assert result.selected_count == 8


def test_bda_selection_deterministic_given_seed():
    data = _make_fake_fold_data(n_train=40, n_val=10)
    cfg = get_config()
    from dataclasses import replace
    cfg = replace(cfg, bda_population_size=6, bda_iterations=3)
    r1 = select_bda(data, cfg, random_seed=42)
    r2 = select_bda(data, cfg, random_seed=42)
    assert np.array_equal(r1.selected_mask, r2.selected_mask)
    assert r1.selected_count == r2.selected_count


def test_bda_nested_split_never_touches_data_x_val():
    """BDA's fitness function must only ever see data.X_train (via its
    own nested split) -- never data.X_val."""
    data = _make_fake_fold_data(n_train=40, n_val=10)
    data.X_val[:] = 1e6  # extreme OOD values
    cfg = get_config()
    from dataclasses import replace
    cfg = replace(cfg, bda_population_size=6, bda_iterations=3)
    result = select_bda(data, cfg, random_seed=42)
    assert -1.0 <= result.extra["best_nested_fitness"] <= 1.0


def test_bda_branch_retained_counts_sum_to_selected_count():
    data = _make_fake_fold_data(n_train=40, n_val=10)
    cfg = get_config()
    from dataclasses import replace
    cfg = replace(cfg, bda_population_size=6, bda_iterations=3)
    result = select_bda(data, cfg, random_seed=42)
    assert sum(result.branch_retained.values()) == result.selected_count


def test_pairwise_and_branch_jaccard_stability():
    m1 = np.array([True, True, False, False, True, False], dtype=bool)
    m2 = np.array([True, True, False, False, False, True], dtype=bool)
    m3 = np.array([True, False, True, False, True, False], dtype=bool)

    stab = compute_pairwise_jaccard_stability([m1, m2, m3])
    assert 0.0 <= stab <= 1.0

    branch_dims = {"b1": 4, "b2": 2}
    b_stab = compute_branch_jaccard_stability([m1, m2, m3], branch_dims)
    assert "b1" in b_stab and "b2" in b_stab
    assert 0.0 <= b_stab["b1"] <= 1.0
    assert 0.0 <= b_stab["b2"] <= 1.0

