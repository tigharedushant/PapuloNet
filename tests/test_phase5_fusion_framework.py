"""
tests/test_phase5_fusion_framework.py

Comprehensive unit tests for AEF-CRC Phase 5 Deep + Handcrafted Feature Fusion:
- 8-arm controlled fusion hierarchy and dimensions
- Deterministic branch ordering and slice boundaries
- Stable 1-to-1 feature naming contract
- Single-branch ablation isolation and additive invariance
- Fold-safety (train-only fit, val-transform-only, no leakage)
- Single PCA discipline (HOG only, no second PCA on fused matrix)
- Phase 5 vs Phase 6 boundary enforcement (zero feature selection)
- Equivalence margin constant (0.005)
- Provenance metadata contract
- CLI framework validation execution
"""

from __future__ import annotations

import numpy as np
import pytest

from config.config import get_config
from modules.fusion import (
    FUSION_ARMS,
    FUSION_ARMS_BY_ID,
    FUSION_ARMS_BY_NAME,
    FUSION_ARMS_BY_REPR_ID,
    ARMS,
    CANONICAL_BRANCH_ORDER,
    BRANCH_ORDER,
    BRANCH_DIMENSIONS,
    EQUIVALENCE_MARGIN,
    sort_branches_deterministically,
    get_fusion_slice_boundaries,
    get_fusion_feature_names,
    get_fusion_provenance,
    validate_fusion_dimensions,
)
from modules.handcrafted_features import FeatureNormalizer, FoldSafeFeatureReducer
from run_aef_crc_phase5 import validate_fusion_framework


def test_all_eight_arms_dimensions():
    assert len(FUSION_ARMS) == 8
    expected_contracts = {
        "A0": ("EfficientNet", "efficientnet", ("deep",), 1280),
        "A1": ("EfficientNet+GLCM", "efficientnet_glcm", ("deep", "glcm"), 1292),
        "A2": ("EfficientNet+LBP", "efficientnet_lbp", ("deep", "lbp"), 1298),
        "A3": ("EfficientNet+HOG-PCA", "efficientnet_hog", ("deep", "hog"), 1312),
        "A4": ("EfficientNet+LAB", "efficientnet_lab", ("deep", "color_lab"), 1286),
        "A5": ("EfficientNet+GLCM+LBP", "efficientnet_glcm_lbp", ("deep", "glcm", "lbp"), 1310),
        "A6": ("EfficientNet+GLCM+LBP+HOG-PCA+LAB", "efficientnet_glcm_lbp_hog_lab", ("deep", "glcm", "lbp", "hog", "color_lab"), 1348),
        "A7": ("EfficientNet+GLCM+LBP+LAB", "efficientnet_glcm_lbp_lab", ("deep", "glcm", "lbp", "color_lab"), 1316),
    }

    for arm_id, (name, repr_id, branches, exp_dim) in expected_contracts.items():
        arm = FUSION_ARMS_BY_ID[arm_id]
        assert arm.name == name
        assert arm.representation_id == repr_id
        assert arm.branches == branches
        assert arm.expected_dim == exp_dim
        assert ARMS[name] == branches


def test_deterministic_branch_ordering():
    test_permutations = [
        ("color_lab", "hog", "deep", "lbp", "glcm"),
        ("hog", "deep"),
        ("glcm", "deep", "lbp"),
        ("color_lab", "glcm", "deep"),
        ("lab", "deep"),
    ]
    for perm in test_permutations:
        sorted_branches = sort_branches_deterministically(perm)
        indices = [CANONICAL_BRANCH_ORDER.index(b) for b in sorted_branches]
        assert indices == sorted(indices)
        assert len(indices) == len(set(indices))


def test_deterministic_slice_boundaries():
    a6 = FUSION_ARMS_BY_ID["A6"]
    slices = get_fusion_slice_boundaries(a6.branches)
    assert slices["deep"] == (0, 1280)
    assert slices["glcm"] == (1280, 1292)
    assert slices["lbp"] == (1292, 1310)
    assert slices["hog"] == (1310, 1342)
    assert slices["color_lab"] == (1342, 1348)

    last_end = 0
    for b in ("deep", "glcm", "lbp", "hog", "color_lab"):
        start, end = slices[b]
        assert start == last_end
        last_end = end
    assert last_end == 1348


def test_deterministic_feature_names():
    config = get_config()
    for arm in FUSION_ARMS:
        fnames = get_fusion_feature_names(arm.branches, config)
        assert len(fnames) == arm.expected_dim
        assert len(set(fnames)) == arm.expected_dim


def test_ablation_isolation():
    a0 = FUSION_ARMS_BY_ID["A0"]
    a1 = FUSION_ARMS_BY_ID["A1"]
    a2 = FUSION_ARMS_BY_ID["A2"]
    a3 = FUSION_ARMS_BY_ID["A3"]
    a4 = FUSION_ARMS_BY_ID["A4"]
    a5 = FUSION_ARMS_BY_ID["A5"]
    a6 = FUSION_ARMS_BY_ID["A6"]
    a7 = FUSION_ARMS_BY_ID["A7"]

    assert set(a1.branches) - set(a0.branches) == {"glcm"}
    assert set(a2.branches) - set(a0.branches) == {"lbp"}
    assert set(a3.branches) - set(a0.branches) == {"hog"}
    assert set(a4.branches) - set(a0.branches) == {"color_lab"}
    assert set(a5.branches) - set(a0.branches) == {"glcm", "lbp"}
    assert set(a7.branches) - set(a5.branches) == {"color_lab"}
    assert set(a6.branches) - set(a7.branches) == {"hog"}


def test_frozen_branches_additivity():
    config = get_config()
    names_a5 = get_fusion_feature_names(FUSION_ARMS_BY_ID["A5"].branches, config)
    names_a7 = get_fusion_feature_names(FUSION_ARMS_BY_ID["A7"].branches, config)
    assert names_a7[:1310] == names_a5


def test_fold_safe_normalization_discipline():
    rng = np.random.default_rng(123)
    train_vecs = rng.normal(loc=5.0, scale=2.0, size=(50, 10))
    val_vecs = rng.normal(loc=10.0, scale=4.0, size=(25, 10))

    normalizer = FeatureNormalizer().fit(train_vecs)
    val_norm = normalizer.transform(val_vecs)

    assert np.allclose(normalizer.mean_, train_vecs.mean(axis=0))
    assert np.allclose(normalizer.std_, train_vecs.std(axis=0))

    with pytest.raises(RuntimeError, match="called twice"):
        normalizer.fit(train_vecs)


def test_no_second_pca():
    import modules.fusion as fusion_mod
    import inspect
    source = inspect.getsource(fusion_mod.build_fusion_fold)
    assert "PCA(" not in source
    assert "FoldSafeFeatureReducer" in source


def test_phase_boundary_no_feature_selection():
    import modules.fusion as fusion_mod
    for forbidden in ("DFASelector", "GeneticSelector", "RFESelector", "run_selector"):
        assert not hasattr(fusion_mod, forbidden), f"Phase 5 fusion exposes forbidden selector: {forbidden}"


def test_equivalence_margin_constant():
    assert EQUIVALENCE_MARGIN == 0.005


def test_fusion_provenance_metadata():
    config = get_config()
    for arm in FUSION_ARMS:
        prov = get_fusion_provenance(arm.name, config, fold_index=1, split="val")
        assert prov["arm_id"] == arm.arm_id
        assert prov["representation_id"] == arm.representation_id
        assert prov["expected_dim"] == arm.expected_dim
        assert prov["actual_feature_count"] == arm.expected_dim
        assert prov["fold_index"] == 1
        assert prov["split"] == "val"
        assert prov["leakage_guarantees"]["val_transform_only"] is True


def test_framework_validation_cli():
    config = get_config()
    assert validate_fusion_framework(config) is True
