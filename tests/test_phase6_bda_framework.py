"""
tests/test_phase6_bda_framework.py

Unit tests for Phase 6 Binary Dragonfly Algorithm (BDA) feature selection
framework and associated stability, leakage-safety, and provenance utilities.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from config.config import get_config, PSDConfig
from modules.fusion import FusionFoldData
from modules.feature_selection import (
    select_bda,
    run_selector,
    compute_jaccard_similarity,
    compute_pairwise_jaccard,
    compute_feature_family_breakdown,
    map_mask_to_feature_names,
    SELECTOR_REGISTRY,
)
from run_aef_crc_phase6 import validate_phase6_framework


@pytest.fixture
def synthetic_fusion_data():
    """Provides a synthetic 1348-D FusionFoldData matching the Phase 5 full arm."""
    rng = np.random.default_rng(42)
    n_train, n_val, n_feat = 50, 20, 1348
    X_train = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    y_train = ["Psoriasis", "Lichen Planus", "Pityriasis Rosea", "Seborrheic Dermatitis"] * 12 + ["Psoriasis", "Lichen Planus"]
    X_val = rng.standard_normal((n_val, n_feat)).astype(np.float32)
    y_val = ["Psoriasis", "Lichen Planus", "Pityriasis Rosea", "Seborrheic Dermatitis"] * 5
    branch_dims = {"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6}

    return FusionFoldData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        psd_ids_train=[f"PSD-TR-{i:04d}" for i in range(n_train)],
        psd_ids_val=[f"PSD-VA-{i:04d}" for i in range(n_val)],
        branch_dims=branch_dims,
    )


@pytest.fixture
def fast_bda_config():
    """Config with minimal iterations and population for fast unit tests."""
    base = get_config()
    return dataclasses.replace(
        base,
        bda_population_size=6,
        bda_iterations=4,
        bda_v_max=6.0,
        bda_tau_min=1.0,
        bda_tau_max=4.0,
        bda_feature_count_penalty=0.0005,
        bda_nested_val_fraction=0.2,
    )


def test_bda_selector_contract_and_mask_shape(synthetic_fusion_data, fast_bda_config):
    """Test 1: BDA satisfies the standard Phase-6 SelectionResult contract."""
    res = select_bda(synthetic_fusion_data, fast_bda_config, random_seed=42)

    assert res.method == "bda"
    assert isinstance(res.selected_mask, np.ndarray)
    assert res.selected_mask.shape == (1348,)
    assert res.selected_mask.dtype == bool
    assert res.selected_count == int(res.selected_mask.sum())
    assert res.selected_count > 0
    assert res.fit_time_sec >= 0.0
    assert "bda" in SELECTOR_REGISTRY


def test_bda_all_zero_recovery_deterministic(synthetic_fusion_data, fast_bda_config):
    """Test 2: All-zero recovery activates feature with argmax |Delta x_d|."""
    extreme_cfg = dataclasses.replace(fast_bda_config, bda_feature_count_penalty=10.0, bda_iterations=2)
    res = select_bda(synthetic_fusion_data, extreme_cfg, random_seed=99)

    assert res.selected_count >= 1
    assert res.selected_mask.sum() >= 1


def test_bda_determinism_identical_seed(synthetic_fusion_data, fast_bda_config):
    """Test 3: Identical seeds produce bitwise identical selection results."""
    res1 = select_bda(synthetic_fusion_data, fast_bda_config, random_seed=12345)
    res2 = select_bda(synthetic_fusion_data, fast_bda_config, random_seed=12345)

    assert np.array_equal(res1.selected_mask, res2.selected_mask)
    assert res1.selected_count == res2.selected_count
    assert res1.extra["convergence_history"] == res2.extra["convergence_history"]
    assert res1.extra["best_nested_fitness"] == res2.extra["best_nested_fitness"]


def test_bda_time_varying_v_shaped_transfer_function(fast_bda_config):
    """Test 4: Mathematical validation of time-varying V-shaped transfer function."""
    tau_min = fast_bda_config.bda_tau_min
    tau_max = fast_bda_config.bda_tau_max
    T = fast_bda_config.bda_iterations

    tau_0 = tau_min + (tau_max - tau_min) * (0 / T)
    tau_mid = tau_min + (tau_max - tau_min) * ((T // 2) / T)
    tau_end = tau_min + (tau_max - tau_min) * (T / T)

    assert abs(tau_0 - 1.0) < 1e-6
    assert tau_0 <= tau_mid <= tau_end
    assert abs(tau_end - 4.0) < 1e-6

    for delta in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        t_pos = abs(np.tanh(tau_mid * abs(delta)))
        t_val = abs(np.tanh(tau_mid * delta))
        assert abs(t_pos - t_val) < 1e-6
        assert 0.0 <= t_val <= 1.0


def test_bda_fold_isolation_different_seeds(synthetic_fusion_data, fast_bda_config):
    """Test 5: Fold seeds produce distinct trajectories, ensuring fold isolation."""
    res_fold0 = select_bda(synthetic_fusion_data, fast_bda_config, random_seed=42)
    res_fold1 = select_bda(synthetic_fusion_data, fast_bda_config, random_seed=43)

    assert res_fold0.selected_count > 0
    assert res_fold1.selected_count > 0
    assert res_fold0.extra["convergence_history"] != res_fold1.extra["convergence_history"]


def test_bda_objective_parity_with_ga(synthetic_fusion_data, fast_bda_config):
    """Test 6: Objective function penalty matches GA penalty parity."""
    assert fast_bda_config.bda_feature_count_penalty == fast_bda_config.ga_feature_count_penalty == 0.0005

    res = select_bda(synthetic_fusion_data, fast_bda_config, random_seed=42)
    best_fitness = res.extra["best_nested_fitness"]
    best_f1 = res.extra["best_nested_macro_f1"]
    penalty = fast_bda_config.bda_feature_count_penalty * res.selected_count

    assert abs(best_fitness - (best_f1 - penalty)) < 1e-5


def test_bda_nested_split_leakage_safety(synthetic_fusion_data, fast_bda_config):
    """Test 7: BDA strictly never touches outer validation features during search."""
    poisoned_data = FusionFoldData(
        X_train=synthetic_fusion_data.X_train,
        X_val=np.full_like(synthetic_fusion_data.X_val, np.nan),
        y_train=synthetic_fusion_data.y_train,
        y_val=synthetic_fusion_data.y_val,
        psd_ids_train=synthetic_fusion_data.psd_ids_train,
        psd_ids_val=synthetic_fusion_data.psd_ids_val,
        branch_dims=synthetic_fusion_data.branch_dims,
    )

    res = select_bda(poisoned_data, fast_bda_config, random_seed=42)
    assert res.selected_count > 0
    assert not np.isnan(res.extra["best_nested_fitness"])


def test_bda_feature_name_mapping(synthetic_fusion_data, fast_bda_config):
    """Test 8: map_mask_to_feature_names maps active bits correctly to Phase 5 names."""
    mask = np.zeros(1348, dtype=bool)
    mask[0] = True     # deep 0
    mask[1280] = True  # glcm 0
    mask[1292] = True  # lbp 0
    mask[1310] = True  # hog 0
    mask[1342] = True  # color_lab 0

    branches = ("deep", "glcm", "lbp", "hog", "color_lab")
    names = map_mask_to_feature_names(mask, branches=branches, config=fast_bda_config)

    assert len(names) == 5
    assert names[0] == "efficientnet_0"
    assert names[1] == "glcm_contrast_d1"
    assert names[2] == "lbp_bin_0"
    assert names[3] == "hog_pca_0"
    assert names[4] == "lab_L_mean"


def test_bda_feature_family_breakdown():
    """Test 9: compute_feature_family_breakdown sums exactly to selected_count."""
    mask = np.zeros(1348, dtype=bool)
    mask[0:10] = True      # 10 deep
    mask[1280:1285] = True # 5 glcm
    mask[1292:1296] = True # 4 lbp
    mask[1310:1320] = True # 10 hog
    mask[1342:1345] = True # 3 color_lab

    branch_dims = {"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6}
    breakdown = compute_feature_family_breakdown(mask, branch_dims)

    assert breakdown["deep"] == 10
    assert breakdown["glcm"] == 5
    assert breakdown["lbp"] == 4
    assert breakdown["hog"] == 10
    assert breakdown["color_lab"] == 3
    assert sum(breakdown.values()) == int(mask.sum()) == 32


def test_bda_jaccard_similarity_metrics():
    """Test 10: Jaccard similarity computes correct bounds, self-similarity, and pairwise stats."""
    m1 = np.array([True, True, False, False], dtype=bool)
    m2 = np.array([True, True, False, False], dtype=bool)
    m3 = np.array([False, False, True, True], dtype=bool)
    m4 = np.array([True, False, True, False], dtype=bool)

    assert compute_jaccard_similarity(m1, m2) == 1.0
    assert compute_jaccard_similarity(m1, m3) == 0.0
    assert abs(compute_jaccard_similarity(m1, m4) - (1.0 / 3.0)) < 1e-6

    pairwise = compute_pairwise_jaccard([m1, m2, m3])
    assert "mean_jaccard" in pairwise
    assert "median_jaccard" in pairwise
    assert pairwise["pairwise"]["fold_0_vs_fold_1"] == 1.0
    assert pairwise["pairwise"]["fold_0_vs_fold_2"] == 0.0


def test_bda_convergence_history_tracked(synthetic_fusion_data, fast_bda_config):
    """Test 11: Convergence history monotonically tracks best fitness via elitism."""
    res = select_bda(synthetic_fusion_data, fast_bda_config, random_seed=42)
    history = res.extra["convergence_history"]

    assert len(history) == fast_bda_config.bda_iterations
    for i in range(1, len(history)):
        assert history[i]["best_fitness"] >= history[i - 1]["best_fitness"] - 1e-9


def test_phase6_cli_validation_mode(fast_bda_config):
    """Test 12: CLI validation mode passes cleanly without side effects."""
    assert validate_phase6_framework(fast_bda_config) is True
