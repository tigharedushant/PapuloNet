"""
tests/test_phase4_handcrafted_framework.py

AEF-CRC Phase 4: Handcrafted Feature Framework Test Suite.
Verifies the IEEE-grade 68-D representation:
- GLCM (12-D Haralick texture, 4-angle averaged)
- LBP (18-D Uniform texture histogram)
- HOG (1296-D raw -> Fold-Safe PCA 32-D)
- Color LAB (6-D CIELAB channel statistics from color image)
- Total default: 68-D (or 62-D if color_feature_enabled=False)
- Fold-safe dimensionality reduction (train-only fit)
- Fold-safe feature normalization (train-only fit)
- Deterministic feature naming contract
- Parameter-hashed cache isolation and stale cache rejection
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.config import get_config, PSDConfig
from modules.handcrafted_features import (
    HandcraftedFeatureExtractor,
    FeatureNormalizer,
    FoldSafeFeatureReducer,
    FeatureVectors,
    get_handcrafted_feature_names,
    _GLCM_PROPS,
    HANDCRAFTED_BRANCH_REGISTRY,
)
from run_aef_crc_phase4 import validate_framework_components


def _make_synthetic_image(size: int = 224, color_bgr=(180, 160, 140)) -> np.ndarray:
    """Deterministic synthetic test image."""
    img = np.full((size, size, 3), 190, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 4, color_bgr, -1)
    return cv2.GaussianBlur(img, (9, 9), 0)


# ============================================================
# 1. GLCM Extraction Contract (12-D)
# ============================================================

def test_glcm_extraction_contract():
    config = get_config()
    extractor = HandcraftedFeatureExtractor(config)
    img = _make_synthetic_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_vec = extractor._extract_glcm(gray)
    assert isinstance(glcm_vec, np.ndarray)
    assert glcm_vec.dtype == np.float32
    assert len(glcm_vec) == 12, f"Expected 12 GLCM features, got {len(glcm_vec)}"
    assert not np.isnan(glcm_vec).any(), "GLCM vector contains NaN"
    assert not np.isinf(glcm_vec).any(), "GLCM vector contains Inf"

    # Determinism
    glcm_vec_2 = extractor._extract_glcm(gray)
    assert np.array_equal(glcm_vec, glcm_vec_2), "GLCM extraction is not deterministic"


# ============================================================
# 2. LBP Extraction Contract (18-D)
# ============================================================

def test_lbp_extraction_contract():
    config = get_config()
    extractor = HandcraftedFeatureExtractor(config)
    img = _make_synthetic_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp_vec = extractor._extract_lbp(gray)
    assert isinstance(lbp_vec, np.ndarray)
    assert lbp_vec.dtype == np.float32
    expected_dim = config.lbp_n_points + 2  # uniform method: P + 2 = 18
    assert len(lbp_vec) == expected_dim, f"Expected {expected_dim} LBP features, got {len(lbp_vec)}"
    assert not np.isnan(lbp_vec).any(), "LBP vector contains NaN"
    assert not np.isinf(lbp_vec).any(), "LBP vector contains Inf"
    # Density histogram should sum to ~1.0
    assert np.isclose(lbp_vec.sum(), 1.0, atol=1e-3), f"LBP histogram does not sum to 1.0: {lbp_vec.sum()}"


# ============================================================
# 3. HOG Raw Extraction Contract (1296-D)
# ============================================================

def test_hog_raw_extraction_contract():
    config = get_config()
    extractor = HandcraftedFeatureExtractor(config)
    img = _make_synthetic_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_vec = extractor._extract_hog(gray)
    assert isinstance(hog_vec, np.ndarray)
    assert hog_vec.dtype == np.float32
    assert len(hog_vec) == 1296, f"Expected 1296 HOG features, got {len(hog_vec)}"
    assert not np.isnan(hog_vec).any(), "HOG vector contains NaN"
    assert not np.isinf(hog_vec).any(), "HOG vector contains Inf"


# ============================================================
# 4. Color LAB Extraction Contract (6-D)
# ============================================================

def test_color_lab_extraction_contract():
    config = get_config()
    extractor = HandcraftedFeatureExtractor(config)
    img = _make_synthetic_image(color_bgr=(30, 80, 200))

    lab_vec = extractor._extract_color_lab(img)
    assert isinstance(lab_vec, np.ndarray)
    assert lab_vec.dtype == np.float32
    assert len(lab_vec) == 6, f"Expected 6 LAB features, got {len(lab_vec)}"
    assert not np.isnan(lab_vec).any(), "LAB vector contains NaN"
    assert not np.isinf(lab_vec).any(), "LAB vector contains Inf"

    # [L_mean, L_std, a_mean, a_std, b_mean, b_std]
    # In OpenCV 8-bit LAB: L in [0, 255], a in [0, 255], b in [0, 255]
    for val in lab_vec:
        assert 0.0 <= val <= 255.0, f"LAB feature out of valid 8-bit range: {val}"


def test_color_lab_discrimination_power():
    config = get_config()
    extractor = HandcraftedFeatureExtractor(config)
    size = config.image_size

    # Pure red vs pure green vs pure blue
    red_img = np.full((size, size, 3), (20, 20, 230), dtype=np.uint8)    # BGR high R
    green_img = np.full((size, size, 3), (20, 230, 20), dtype=np.uint8)  # BGR high G
    blue_img = np.full((size, size, 3), (230, 20, 20), dtype=np.uint8)   # BGR high B

    lab_red = extractor._extract_color_lab(red_img)
    lab_green = extractor._extract_color_lab(green_img)
    lab_blue = extractor._extract_color_lab(blue_img)

    # a* channel is index 2 (mean) -- differentiates red from green
    assert abs(lab_red[2] - lab_green[2]) > 30.0, "LAB a* channel failed to separate red and green"
    # b* channel is index 4 (mean) -- differentiates yellow/red from blue
    assert abs(lab_red[4] - lab_blue[4]) > 30.0, "LAB b* channel failed to separate red and blue"


# ============================================================
# 5. Combined Representation Dimensions (68-D vs 62-D)
# ============================================================

def test_combined_representation_68d_when_color_enabled():
    config = dataclasses.replace(get_config(), color_feature_enabled=True)
    extractor = HandcraftedFeatureExtractor(config)
    img = _make_synthetic_image()

    fv = extractor.extract(img)
    assert len(fv.glcm) == 12
    assert len(fv.lbp) == 18
    assert len(fv.hog) == 1296
    assert fv.color_lab is not None and len(fv.color_lab) == 6
    assert len(fv.combined) == 1332  # 12 + 18 + 1296 + 6

    # Reduced evaluation representation: 12 + 18 + 32 + 6 = 68-D
    eval_dim = len(fv.glcm) + len(fv.lbp) + config.hog_pca_components + len(fv.color_lab)
    assert eval_dim == 68


def test_combined_representation_62d_when_color_disabled(tmp_path):
    config = dataclasses.replace(get_config(), color_feature_enabled=False, aef_crc_phase4_artifacts_dir=tmp_path / "phase4")
    extractor = HandcraftedFeatureExtractor(config)
    img = _make_synthetic_image()

    fv = extractor.extract(img)
    assert len(fv.glcm) == 12
    assert len(fv.lbp) == 18
    assert len(fv.hog) == 1296
    assert fv.color_lab is None
    assert len(fv.combined) == 1326  # 12 + 18 + 1296

    # Reduced evaluation representation: 12 + 18 + 32 = 62-D
    eval_dim = len(fv.glcm) + len(fv.lbp) + config.hog_pca_components
    assert eval_dim == 62


# ============================================================
# 6. Deterministic Feature Names Contract
# ============================================================

def test_deterministic_feature_names_contract():
    config_68 = dataclasses.replace(get_config(), color_feature_enabled=True)
    names_68 = get_handcrafted_feature_names(config_68, reduced_hog=True)
    assert len(names_68) == 68
    assert len(set(names_68)) == 68  # strictly unique

    # Check order: GLCM (12) -> LBP (18) -> HOG_PCA (32) -> LAB (6)
    assert all(n.startswith("glcm_") for n in names_68[0:12])
    assert all(n.startswith("lbp_bin_") for n in names_68[12:30])
    assert all(n.startswith("hog_pca_") for n in names_68[30:62])
    assert names_68[62:] == [
        "lab_L_mean", "lab_L_std",
        "lab_a_mean", "lab_a_std",
        "lab_b_mean", "lab_b_std",
    ]

    # With color disabled: 62 names
    config_62 = dataclasses.replace(get_config(), color_feature_enabled=False)
    names_62 = get_handcrafted_feature_names(config_62, reduced_hog=True)
    assert len(names_62) == 62
    assert len(set(names_62)) == 62
    assert not any("lab_" in n for n in names_62)


# ============================================================
# 7. Fold-Safe HOG PCA Reducer Contract
# ============================================================

def test_fold_safe_hog_pca_reducer_contract():
    rng = np.random.default_rng(123)
    train_hogs = [rng.normal(size=1296).astype(np.float32) for _ in range(45)]
    val_hogs = [rng.normal(size=1296).astype(np.float32) for _ in range(15)]

    reducer = FoldSafeFeatureReducer(n_components=32)

    # 1. Transform before fit MUST fail
    with pytest.raises(RuntimeError, match="before fit"):
        reducer.transform(val_hogs)

    # 2. Fit on train
    reducer.fit(train_hogs)

    # 3. Double fit MUST fail
    with pytest.raises(RuntimeError, match="twice"):
        reducer.fit(val_hogs)

    # 4. Transform train & val
    red_train = reducer.transform(train_hogs)
    red_val = reducer.transform(val_hogs)
    assert red_train.shape == (45, 32)
    assert red_val.shape == (15, 32)


# ============================================================
# 8. Fold-Safe Feature Normalizer Contract
# ============================================================

def test_feature_normalizer_contract():
    rng = np.random.default_rng(456)
    train_vecs = [rng.normal(loc=10.0, scale=4.0, size=68).astype(np.float32) for _ in range(30)]
    val_vecs = [rng.normal(loc=10.0, scale=4.0, size=68).astype(np.float32) for _ in range(10)]

    normalizer = FeatureNormalizer()

    # 1. Transform before fit MUST fail
    with pytest.raises(RuntimeError, match="before fit"):
        normalizer.transform(val_vecs)

    # 2. Fit on train
    normalizer.fit(train_vecs)

    # 3. Double fit MUST fail
    with pytest.raises(RuntimeError, match="twice"):
        normalizer.fit(val_vecs)

    # 4. Output statistics on train
    norm_train = normalizer.transform(train_vecs)
    assert np.allclose(norm_train.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(norm_train.std(axis=0), 1.0, atol=1e-5)


# ============================================================
# 9. Cache Parameter Hash Tracks Color Setting
# ============================================================

def test_cache_parameter_hash_tracks_color_toggle(tmp_path):
    base_cfg = dataclasses.replace(get_config(), aef_crc_phase4_artifacts_dir=tmp_path / "phase4_on")
    cfg_color_on = dataclasses.replace(base_cfg, color_feature_enabled=True)
    cfg_color_off = dataclasses.replace(base_cfg, color_feature_enabled=False, aef_crc_phase4_artifacts_dir=tmp_path / "phase4_off")

    ex_on = HandcraftedFeatureExtractor(cfg_color_on)
    ex_off = HandcraftedFeatureExtractor(cfg_color_off)

    hash_on = ex_on._param_hash()
    hash_off = ex_off._param_hash()
    assert hash_on != hash_off, "Param hash failed to change when color_feature_enabled was toggled"

    # Also test that loading color_off into the color_on cache directory triggers stale cache rejection
    cfg_stale = dataclasses.replace(cfg_color_off, aef_crc_phase4_artifacts_dir=tmp_path / "phase4_on")
    with pytest.raises(ValueError, match="parameter hash mismatch"):
        HandcraftedFeatureExtractor(cfg_stale)


# ============================================================
# 10. Framework Validation CLI Method
# ============================================================

def test_validate_framework_components_executes_successfully():
    config = get_config()
    result = validate_framework_components(config)
    assert result is True, "validate_framework_components failed"
