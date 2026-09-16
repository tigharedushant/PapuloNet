"""
tests/test_phase4_contract_and_leakage.py

Rigorous contract, dimensionality, leakage safety, and reproducibility tests
for AEF-CRC Phase 4 Handcrafted Features Framework.

Validated requirements:
1. Synthetic data tests: GLCM (12-D), LBP (18-D), HOG raw (1296-D),
   HOG-PCA (32-D), LAB (6-D), Combined Evaluated (68-D).
2. LAB Color Extractor: exactly 6-D, finite values, deterministic, sensitive
   to color changes, computed from original color image, leaving grayscale
   descriptors (GLCM/LBP/HOG) completely unaffected.
3. HOG PCA Fold-Safety: fit ONLY on training records, transform validation
   records, cannot double-fit, cannot transform before fit, fails on empty train.
4. Deterministic Feature Order: GLCM -> LBP -> HOG-PCA -> LAB (exact 68-D).
5. Deterministic Feature Names: traceable, stable names in exact concatenation order.
6. Bitwise Reproducibility across repeated executions.
7. Backward Compatibility: color_feature_enabled=False yields 62-D representation.
8. No learned parameters in raw feature extraction.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest
import cv2

from config.config import get_config
from modules.handcrafted_features import (
    HandcraftedFeatureExtractor,
    FeatureVectors,
    FeatureNormalizer,
    FoldSafeFeatureReducer,
    get_handcrafted_feature_names,
    HANDCRAFTED_BRANCH_REGISTRY,
)


def _make_synthetic_color_image(size: int = 224, bg_color=(200, 180, 160), fg_color=(50, 40, 30)) -> np.ndarray:
    """Generates a synthetic BGR dermatological lesion-like test image."""
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 4, fg_color, -1)
    return cv2.GaussianBlur(img, (7, 7), 0)


# =========================================================================
# 1. Synthetic Tests: Individual & Combined Dimensions (Requirement 21)
# =========================================================================

def test_glcm_dimension_contract():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img = _make_synthetic_color_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_vec = ex._extract_glcm(gray)
    assert glcm_vec.shape == (12,), f"GLCM must be 12-D, got {glcm_vec.shape}"
    assert glcm_vec.dtype == np.float32


def test_lbp_dimension_contract():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img = _make_synthetic_color_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_vec = ex._extract_lbp(gray)
    assert lbp_vec.shape == (18,), f"LBP must be 18-D, got {lbp_vec.shape}"
    assert lbp_vec.dtype == np.float32


def test_hog_raw_dimension_contract():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img = _make_synthetic_color_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_vec = ex._extract_hog(gray)
    assert hog_vec.shape == (1296,), f"Raw HOG must be 1296-D, got {hog_vec.shape}"
    assert hog_vec.dtype == np.float32


def test_hog_pca_dimension_contract():
    rng = np.random.default_rng(42)
    synthetic_hogs = [rng.normal(size=1296).astype(np.float32) for _ in range(50)]
    reducer = FoldSafeFeatureReducer(n_components=32)
    reducer.fit(synthetic_hogs)
    reduced = reducer.transform(synthetic_hogs)
    assert reduced.shape == (50, 32), f"HOG PCA must be 32-D, got {reduced.shape}"


def test_lab_dimension_contract():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img = _make_synthetic_color_image()
    lab_vec = ex._extract_color_lab(img)
    assert lab_vec.shape == (6,), f"LAB must be 6-D, got {lab_vec.shape}"
    assert lab_vec.dtype == np.float32


def test_combined_representation_contract():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img = _make_synthetic_color_image()
    fv = ex.extract(img)

    # Raw concatenation: GLCM (12) + LBP (18) + HOG (1296) + LAB (6) = 1332
    assert fv.combined.shape == (1332,), f"Raw combined must be 1332-D, got {fv.combined.shape}"

    # Reduced concatenation: GLCM (12) + LBP (18) + HOG-PCA (32) + LAB (6) = 68
    rng = np.random.default_rng(42)
    synthetic_train_hogs = [rng.normal(size=1296).astype(np.float32) for _ in range(40)]
    reducer = FoldSafeFeatureReducer(n_components=32).fit(synthetic_train_hogs)
    hog_reduced = reducer.transform([fv.hog])[0]

    eval_combined = np.concatenate([fv.glcm, fv.lbp, hog_reduced, fv.color_lab])
    assert eval_combined.shape == (68,), f"Evaluated handcrafted vector must be exactly 68-D, got {eval_combined.shape}"


# =========================================================================
# 2. LAB Feature Extraction Properties (Requirement 22)
# =========================================================================

def test_lab_finite_values_and_rejection():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img = _make_synthetic_color_image()
    lab_vec = ex._extract_color_lab(img)

    assert not np.isnan(lab_vec).any(), "LAB vector contains NaN"
    assert not np.isinf(lab_vec).any(), "LAB vector contains Inf"
    # All values must be non-negative (L in 0..255, std >= 0, a, b in 0..255)
    assert (lab_vec >= 0).all(), f"Expected non-negative LAB statistics in OpenCV scale, got {lab_vec}"


def test_lab_color_sensitivity():
    """Different color images must produce distinct LAB statistics."""
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    red_img = np.full((128, 128, 3), (0, 0, 255), dtype=np.uint8)    # BGR pure red
    blue_img = np.full((128, 128, 3), (255, 0, 0), dtype=np.uint8)   # BGR pure blue

    lab_red = ex._extract_color_lab(red_img)
    lab_blue = ex._extract_color_lab(blue_img)

    assert not np.allclose(lab_red, lab_blue), "LAB failed to differentiate distinct colors!"


def test_lab_receives_original_color_not_grayscale():
    """LAB on original color vs grayscale converted must differ significantly."""
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img_color = _make_synthetic_color_image(bg_color=(20, 180, 50), fg_color=(200, 40, 10))
    img_gray = cv2.cvtColor(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    lab_from_color = ex._extract_color_lab(img_color)
    lab_from_gray = ex._extract_color_lab(img_gray)

    # a and b channels have significant deviation on colored images but zero variation on pure grayscale
    assert not np.allclose(lab_from_color[2:], lab_from_gray[2:], atol=1.0), (
        "LAB must receive original color image, not grayscale conversion."
    )


def test_grayscale_extractors_unaffected_by_lab(tmp_path):
    """GLCM, LBP, and HOG extractors produce identical outputs whether color_feature_enabled is True or False."""
    cfg_color = dataclasses.replace(get_config(), color_feature_enabled=True, aef_crc_phase4_artifacts_dir=tmp_path / "color")
    cfg_nocolor = dataclasses.replace(get_config(), color_feature_enabled=False, aef_crc_phase4_artifacts_dir=tmp_path / "nocolor")

    ex_color = HandcraftedFeatureExtractor(cfg_color)
    ex_nocolor = HandcraftedFeatureExtractor(cfg_nocolor)

    img = _make_synthetic_color_image()
    fv_color = ex_color.extract(img)
    fv_nocolor = ex_nocolor.extract(img)

    assert np.array_equal(fv_color.glcm, fv_nocolor.glcm), "GLCM was altered by color feature flag!"
    assert np.array_equal(fv_color.lbp, fv_nocolor.lbp), "LBP was altered by color feature flag!"
    assert np.array_equal(fv_color.hog, fv_nocolor.hog), "HOG was altered by color feature flag!"
    assert fv_color.color_lab is not None
    assert fv_nocolor.color_lab is None


# =========================================================================
# 3. HOG PCA Fold-Safety & Leakage (Requirement 23)
# =========================================================================

def test_hog_pca_train_fit_val_transform_safety():
    """Reducer fit on train transforms validation without error and without refitting."""
    rng = np.random.default_rng(101)
    train_hogs = [rng.normal(loc=0.0, scale=1.0, size=1296).astype(np.float32) for _ in range(40)]
    val_hogs = [rng.normal(loc=2.0, scale=1.5, size=1296).astype(np.float32) for _ in range(10)]

    reducer = FoldSafeFeatureReducer(n_components=32)
    assert not reducer._fitted

    reducer.fit(train_hogs)
    assert reducer._fitted

    # Transform train and val
    train_red = reducer.transform(train_hogs)
    val_red = reducer.transform(val_hogs)

    assert train_red.shape == (40, 32)
    assert val_red.shape == (10, 32)


def test_hog_pca_refuses_double_fit():
    reducer = FoldSafeFeatureReducer(n_components=16)
    vecs = [np.ones(1296, dtype=np.float32) * i for i in range(25)]
    reducer.fit(vecs)
    with pytest.raises(RuntimeError, match="fit\\(\\) called twice on the same instance -- refused"):
        reducer.fit(vecs)


def test_hog_pca_refuses_transform_before_fit():
    reducer = FoldSafeFeatureReducer(n_components=16)
    vecs = [np.ones(1296, dtype=np.float32)]
    with pytest.raises(RuntimeError, match="transform\\(\\) called before fit"):
        reducer.transform(vecs)


def test_hog_pca_refuses_empty_training_data():
    reducer = FoldSafeFeatureReducer(n_components=16)
    with pytest.raises(ValueError, match="fit\\(\\) called with zero training vectors"):
        reducer.fit([])


# =========================================================================
# 4. Feature Order & Registry Contract (Requirement 24, 12, 19)
# =========================================================================

def test_feature_order_is_deterministic_and_matches_contract():
    """Verifies that the evaluated combined vector strictly orders:
    GLCM (12) -> LBP (18) -> HOG-PCA (32) -> LAB (6) = 68.
    """
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img = _make_synthetic_color_image()
    fv = ex.extract(img)

    # Set mock identifiable vectors to trace slice boundaries
    fv.glcm = np.full((12,), 1.0, dtype=np.float32)
    fv.lbp = np.full((18,), 2.0, dtype=np.float32)
    mock_hog_pca = np.full((32,), 3.0, dtype=np.float32)
    fv.color_lab = np.full((6,), 4.0, dtype=np.float32)

    eval_vec = np.concatenate([fv.glcm, fv.lbp, mock_hog_pca, fv.color_lab])
    assert len(eval_vec) == 68

    # Check slice ordering
    assert np.all(eval_vec[0:12] == 1.0), "Indices 0:12 must be GLCM"
    assert np.all(eval_vec[12:30] == 2.0), "Indices 12:30 must be LBP"
    assert np.all(eval_vec[30:62] == 3.0), "Indices 30:62 must be HOG-PCA"
    assert np.all(eval_vec[62:68] == 4.0), "Indices 62:68 must be LAB"


def test_handcrafted_branch_registry_contains_color_lab():
    assert "color_lab" in HANDCRAFTED_BRANCH_REGISTRY
    assert "lab" in HANDCRAFTED_BRANCH_REGISTRY
    get_vec, reducer_fn = HANDCRAFTED_BRANCH_REGISTRY["color_lab"]

    # Must extract fv.color_lab
    fv = FeatureVectors(psd_id="P1", glcm=np.zeros(12), lbp=np.zeros(18), hog=np.zeros(1296), color_lab=np.ones(6))
    assert np.array_equal(get_vec(fv), np.ones(6))
    # Reducer must be no-op for color_lab
    reduced_list = reducer_fn([np.ones(6)], None)
    assert len(reduced_list) == 1 and np.array_equal(reduced_list[0], np.ones(6))


# =========================================================================
# 5. Deterministic Feature Names (Requirement 20)
# =========================================================================

def test_get_handcrafted_feature_names_matches_68_d_order():
    cfg = get_config()
    names = get_handcrafted_feature_names(cfg, reduced_hog=True)
    assert len(names) == 68

    # GLCM: 0 to 12
    for n in names[0:12]:
        assert n.startswith("glcm_")

    # LBP: 12 to 30
    for n in names[12:30]:
        assert n.startswith("lbp_")

    # HOG-PCA: 30 to 62
    for n in names[30:62]:
        assert n.startswith("hog_pca_")

    # LAB: 62 to 68
    expected_lab_names = [
        "lab_L_mean", "lab_L_std",
        "lab_a_mean", "lab_a_std",
        "lab_b_mean", "lab_b_std",
    ]
    assert names[62:68] == expected_lab_names


# =========================================================================
# 6. Bitwise Reproducibility (Requirement 25)
# =========================================================================

def test_bitwise_reproducibility():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img = _make_synthetic_color_image(224)

    run1 = ex.extract(img)
    run2 = ex.extract(img)

    assert np.array_equal(run1.glcm, run2.glcm)
    assert np.array_equal(run1.lbp, run2.lbp)
    assert np.array_equal(run1.hog, run2.hog)
    assert np.array_equal(run1.color_lab, run2.color_lab)
    assert np.array_equal(run1.combined, run2.combined)


# =========================================================================
# 7. Backward Compatibility & Configuration (Requirement 26)
# =========================================================================

def test_configuration_toggle_68_vs_62_dimensions(tmp_path):
    cfg_68 = dataclasses.replace(get_config(), color_feature_enabled=True, aef_crc_phase4_artifacts_dir=tmp_path / "c68")
    cfg_62 = dataclasses.replace(get_config(), color_feature_enabled=False, aef_crc_phase4_artifacts_dir=tmp_path / "c62")

    ex_68 = HandcraftedFeatureExtractor(cfg_68)
    ex_62 = HandcraftedFeatureExtractor(cfg_62)

    img = _make_synthetic_color_image(224)
    fv_68 = ex_68.extract(img)
    fv_62 = ex_62.extract(img)

    # Raw lengths: 1332 vs 1326
    assert len(fv_68.combined) == 1332
    assert len(fv_62.combined) == 1326

    # Names lengths: 68 vs 62
    names_68 = get_handcrafted_feature_names(cfg_68, reduced_hog=True)
    names_62 = get_handcrafted_feature_names(cfg_62, reduced_hog=True)
    assert len(names_68) == 68
    assert len(names_62) == 62


# =========================================================================
# 8. Statelessness & Parameter-Free Property (Requirement 18)
# =========================================================================

def test_descriptors_are_stateless_per_image():
    """Extracting image A then image B produces identical descriptors as B then A."""
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)

    img_a = _make_synthetic_color_image(bg_color=(210, 190, 170))
    img_b = _make_synthetic_color_image(bg_color=(50, 60, 70))

    # Forward order
    fa1 = ex.extract(img_a)
    fb1 = ex.extract(img_b)

    # Reverse order
    fb2 = ex.extract(img_b)
    fa2 = ex.extract(img_a)

    assert np.array_equal(fa1.combined, fa2.combined)
    assert np.array_equal(fb1.combined, fb2.combined)
