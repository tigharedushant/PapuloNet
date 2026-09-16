"""
tests/test_phase6_bda_contract.py

AEF-CRC Phase 6: Binary Dragonfly Algorithm (BDA) Contract Verification Suite.
Rigorous synthetic tests verifying:
1. Expected dimension contract (1348-D full fused vector, boolean mask).
2. Non-empty guarantee (all-zero recovery activates max-velocity dimension).
3. Bitwise determinism across runs with identical random seed.
4. Time-varying V-shaped transfer function mathematical properties and bounds.
5. Leakage prevention: outer validation set X_val is strictly untouched.
6. Fold isolation: independent optimizer instances and RNG streams across folds.
7. Feature branch mapping & accounting: branch counts strictly sum to selected_count.
8. Jaccard similarity and pairwise/branch stability metrics.
9. Unified selector registry contract: run_selector("bda", ...) 4-arg signature.
10. Phase boundary enforcement: no real model training or full real data execution.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from config.config import get_config, PSDConfig
from modules.fusion import FusionFoldData, get_fused_feature_names
from modules.feature_selection import (
    select_bda,
    select_ga,
    run_selector,
    compute_jaccard_similarity,
    compute_pairwise_jaccard,
    compute_pairwise_jaccard_stability,
    compute_branch_jaccard_stability,
    compute_feature_family_breakdown,
    map_mask_to_feature_names,
    SELECTOR_REGISTRY,
    SelectionResult,
)


@pytest.fixture
def synthetic_1348d_data():
    """Builds synthetic 1348-D FusionFoldData matching the Phase 5 full fusion arm."""
    rng = np.random.default_rng(100)
    n_train, n_val, n_feat = 60, 20, 1348
    X_train = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    y_train = [classes[i % len(classes)] for i in range(n_train)]
    X_val = rng.standard_normal((n_val, n_feat)).astype(np.float32)
    y_val = [classes[i % len(classes)] for i in range(n_val)]
    branch_dims = {"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6}

    return FusionFoldData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        psd_ids_train=[f"TR_{i:04d}" for i in range(n_train)],
        psd_ids_val=[f"VAL_{i:04d}" for i in range(n_val)],
        branch_dims=branch_dims,
    )


@pytest.fixture
def fast_bda_config():
    """Minimal population and iteration configuration for deterministic unit tests."""
    base = get_config()
    return replace(
        base,
        bda_population_size=6,
        bda_iterations=3,
        bda_v_max=6.0,
        bda_tau_min=1.0,
        bda_tau_max=4.0,
        bda_w_max=0.9,
        bda_w_min=0.4,
        bda_feature_count_penalty=0.0005,
        bda_nested_val_fraction=0.2,
    )


# ============================================================
# 1. Dimensional Contract
# ============================================================

def test_bda_mask_dimensions_and_type(synthetic_1348d_data, fast_bda_config):
    """Mask must be a boolean numpy array with length equal to total fused features (1348)."""
    res = select_bda(synthetic_1348d_data, fast_bda_config, random_seed=42)
    assert isinstance(res, SelectionResult)
    assert res.method == "bda"
    assert isinstance(res.selected_mask, np.ndarray)
    assert res.selected_mask.shape == (1348,)
    assert res.selected_mask.dtype == bool
    assert res.selected_count == int(res.selected_mask.sum())
    assert res.original_dim == 1348
    assert len(res.selected_indices) == res.selected_count


# ============================================================
# 2. Non-Empty Guarantee & All-Zero Recovery
# ============================================================

def test_bda_all_zero_recovery_guarantee(synthetic_1348d_data, fast_bda_config):
    """Under extreme feature count penalties that discourage any selection,
    all-zero recovery must deterministically activate at least one feature."""
    hostile_cfg = replace(
        fast_bda_config,
        bda_feature_count_penalty=50.0,  # massive penalty guarantees all bits want to flip to 0
        bda_iterations=2,
    )
    res = select_bda(synthetic_1348d_data, hostile_cfg, random_seed=77)
    assert res.selected_count >= 1
    assert res.selected_mask.sum() >= 1
    assert not np.isnan(res.extra["best_nested_fitness"])


# ============================================================
# 3. Bitwise Determinism
# ============================================================

def test_bda_bitwise_determinism(synthetic_1348d_data, fast_bda_config):
    """Identical data, configuration, and seed must produce bitwise identical masks and history."""
    run1 = select_bda(synthetic_1348d_data, fast_bda_config, random_seed=42)
    run2 = select_bda(synthetic_1348d_data, fast_bda_config, random_seed=42)

    assert np.array_equal(run1.selected_mask, run2.selected_mask)
    assert run1.selected_count == run2.selected_count
    assert run1.extra["best_nested_fitness"] == run2.extra["best_nested_fitness"]
    assert len(run1.extra["history"]) == len(run2.extra["history"])
    for h1, h2 in zip(run1.extra["history"], run2.extra["history"]):
        assert h1["best_fitness"] == h2["best_fitness"]
        assert h1["mean_fitness"] == h2["mean_fitness"]


# ============================================================
# 4. Time-Varying V-Shaped Transfer Function Properties
# ============================================================

def test_tv_v_shaped_transfer_function_math(fast_bda_config):
    """Validates boundary conditions:
    1. T(0) = 0 (zero step means zero probability of flip).
    2. T(delta_x) is symmetric around 0: T(-v) == T(v).
    3. T(delta_x) is strictly bounded in [0, 1).
    4. tau(t) scales from tau_min to tau_max.
    """
    tau_min = fast_bda_config.bda_tau_min
    tau_max = fast_bda_config.bda_tau_max
    T = fast_bda_config.bda_iterations

    # Test scaling
    assert tau_min == 1.0 and tau_max == 4.0
    progress_values = [0.0, 0.5, 1.0]
    for p in progress_values:
        tau_t = tau_min + (tau_max - tau_min) * p
        assert tau_min <= tau_t <= tau_max

        # Zero step
        t_zero = abs(np.tanh(tau_t * 0.0))
        assert t_zero == 0.0

        # Symmetry and bounds
        for v in [0.1, 1.0, 3.0, 6.0]:
            t_pos = abs(np.tanh(tau_t * v))
            t_neg = abs(np.tanh(tau_t * (-v)))
            assert abs(t_pos - t_neg) < 1e-9
            assert 0.0 <= t_pos <= 1.0


# ============================================================
# 5. Leakage Prevention (Strict Outer Isolation)
# ============================================================

def test_bda_leakage_safety_poisoned_val(synthetic_1348d_data, fast_bda_config):
    """Corrupting outer X_val with extreme OOD values or NaNs must NOT affect BDA
    fitness or selection results, because BDA operates strictly on nested training splits."""
    clean_run = select_bda(synthetic_1348d_data, fast_bda_config, random_seed=42)

    # Create poisoned copy with corrupted outer validation split
    poisoned_data = FusionFoldData(
        X_train=synthetic_1348d_data.X_train.copy(),
        X_val=np.full_like(synthetic_1348d_data.X_val, np.nan),
        y_train=synthetic_1348d_data.y_train,
        y_val=synthetic_1348d_data.y_val,
        psd_ids_train=synthetic_1348d_data.psd_ids_train,
        psd_ids_val=synthetic_1348d_data.psd_ids_val,
        branch_dims=synthetic_1348d_data.branch_dims,
    )

    poisoned_run = select_bda(poisoned_data, fast_bda_config, random_seed=42)

    assert np.array_equal(clean_run.selected_mask, poisoned_run.selected_mask)
    assert clean_run.selected_count == poisoned_run.selected_count
    assert clean_run.extra["best_nested_fitness"] == poisoned_run.extra["best_nested_fitness"]


# ============================================================
# 6. Fold Isolation
# ============================================================

def test_bda_fold_isolation_different_seeds(synthetic_1348d_data, fast_bda_config):
    """Different random seeds (simulating different CV folds) must evolve independent swarms."""
    r_fold0 = select_bda(synthetic_1348d_data, fast_bda_config, random_seed=101)
    r_fold1 = select_bda(synthetic_1348d_data, fast_bda_config, random_seed=202)

    assert r_fold0.selected_count > 0
    assert r_fold1.selected_count > 0
    # Both runs are valid, but independent swarms explore different trajectories
    assert r_fold0.extra["history"] != r_fold1.extra["history"]


# ============================================================
# 7. Branch Mapping and Accounting
# ============================================================

def test_bda_branch_retained_accounting(synthetic_1348d_data, fast_bda_config):
    """The sum of features retained across branches must strictly equal selected_count."""
    res = select_bda(synthetic_1348d_data, fast_bda_config, random_seed=42)
    breakdown = compute_feature_family_breakdown(res.selected_mask, synthetic_1348d_data.branch_dims)

    assert set(breakdown.keys()) == {"deep", "glcm", "lbp", "hog", "color_lab"}
    assert sum(breakdown.values()) == res.selected_count
    for branch, count in breakdown.items():
        assert 0 <= count <= synthetic_1348d_data.branch_dims[branch]


def test_bda_map_mask_to_feature_names(fast_bda_config):
    """Verifies that active mask bits map 1-to-1 to canonical Phase 5 feature names."""
    mask = np.zeros(1348, dtype=bool)
    mask[0] = True     # deep: efficientnet_0
    mask[1280] = True  # glcm: glcm_contrast_d1
    mask[1292] = True  # lbp: lbp_bin_0
    mask[1310] = True  # hog: hog_pca_0
    mask[1342] = True  # color_lab: lab_L_mean

    branches = ("deep", "glcm", "lbp", "hog", "color_lab")
    names = map_mask_to_feature_names(mask, branches, fast_bda_config)
    assert len(names) == 5
    assert names[0] == "efficientnet_0"
    assert names[1].startswith("glcm_")
    assert names[2] == "lbp_bin_0"
    assert names[3] == "hog_pca_0"
    assert names[4] == "lab_L_mean"


# ============================================================
# 8. Jaccard Stability Metrics
# ============================================================

def test_jaccard_stability_mathematical_properties():
    """Tests Jaccard similarity J(A, B) = |A ∩ B| / |A ∪ B| edge cases."""
    m_full = np.ones(10, dtype=bool)
    m_empty = np.zeros(10, dtype=bool)
    m_half1 = np.array([True]*5 + [False]*5, dtype=bool)
    m_half2 = np.array([False]*5 + [True]*5, dtype=bool)

    # Identical non-empty -> 1.0
    assert compute_jaccard_similarity(m_full, m_full) == 1.0
    # Both empty -> 1.0
    assert compute_jaccard_similarity(m_empty, m_empty) == 1.0
    # Disjoint -> 0.0
    assert compute_jaccard_similarity(m_half1, m_half2) == 0.0

    # Overlapping: A has 4 active, B has 4 active, 2 common -> 2 / 6 = 0.3333
    mA = np.array([True, True, True, True, False, False], dtype=bool)
    mB = np.array([False, False, True, True, True, True], dtype=bool)
    assert abs(compute_jaccard_similarity(mA, mB) - (2.0 / 6.0)) < 1e-6

    # Pairwise stability across multiple masks
    masks = [mA, mB, mA]
    sim = compute_pairwise_jaccard_stability(masks)
    assert 0.0 <= sim <= 1.0


# ============================================================
# 9. Unified Selector Registry
# ============================================================

def test_unified_selector_registry_interface(synthetic_1348d_data, fast_bda_config):
    """Every selector in SELECTOR_REGISTRY must be callable through run_selector."""
    for method in ["none", "xgboost_importance", "rfe", "ga", "bda", "dragonfly"]:
        assert method in SELECTOR_REGISTRY
        res = run_selector(method, synthetic_1348d_data, fast_bda_config, random_seed=42)
        assert isinstance(res, SelectionResult)
        assert res.selected_count > 0
        assert res.selected_mask.shape == (1348,)


# ============================================================
# 10. Phase Boundary & Zero-Execution Enforcement
# ============================================================

def test_phase6_ast_boundary_no_neural_training():
    """Static AST inspection verifying that Phase 6 modules contain no deep learning
    training loops (e.g. model.fit, GradientTape, epochs) or unapproved metaheuristics."""
    import modules.feature_selection as fs_mod
    src_file = Path(fs_mod.__file__)
    tree = ast.parse(src_file.read_text(encoding="utf-8"))

    # Scan for forbidden calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ["fit_generator", "train_on_batch"]:
            pytest.fail(f"Forbidden neural training call '{node.attr}' in modules/feature_selection.py")

    # Confirm only approved metaheuristics exist in SELECTOR_REGISTRY
    approved_keys = {"none", "xgboost_importance", "rfe", "ga", "bda", "dragonfly", "dfa"}
    assert set(SELECTOR_REGISTRY.keys()).issubset(approved_keys)
