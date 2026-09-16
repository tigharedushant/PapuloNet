"""
tests/test_phase4_handcrafted_features.py

Phase 4 regression tests -- GLCM/LBP/HOG extraction needs no deep
learning framework, so unlike Phase 3's TF-dependent modules, these
run for real in this sandbox.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from config.config import get_config
from modules.handcrafted_features import HandcraftedFeatureExtractor, FeatureNormalizer


def _sample_image():
    img = np.full((128, 128, 3), 180, dtype=np.uint8)
    cv2.circle(img, (64, 64), 30, (100, 80, 70), -1)
    return cv2.GaussianBlur(img, (7, 7), 0)


def test_feature_dimensions_are_nonzero_and_fixed():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    result = ex.extract(_sample_image())
    assert len(result.glcm) == len(cfg.glcm_distances) * 6  # 6 GLCM props == 12
    assert len(result.lbp) == cfg.lbp_n_points + 2  # 18
    assert len(result.hog) > 0  # 1296
    if cfg.color_feature_enabled:
        assert len(result.color_lab) == 6
        assert len(result.combined) == len(result.glcm) + len(result.lbp) + len(result.hog) + len(result.color_lab)
    else:
        assert len(result.combined) == len(result.glcm) + len(result.lbp) + len(result.hog)


def test_no_nan_or_inf_in_any_descriptor():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    result = ex.extract(_sample_image())
    descriptors = [("glcm", result.glcm), ("lbp", result.lbp), ("hog", result.hog)]
    if result.color_lab is not None:
        descriptors.append(("color_lab", result.color_lab))
    for name, vec in descriptors:
        assert not np.isnan(vec).any(), f"{name} contains NaN"
        assert not np.isinf(vec).any(), f"{name} contains Inf"


def test_extraction_is_deterministic():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img = _sample_image()
    r1 = ex.extract(img)
    r2 = ex.extract(img)
    assert np.array_equal(r1.glcm, r2.glcm)
    assert np.array_equal(r1.lbp, r2.lbp)
    assert np.array_equal(r1.hog, r2.hog)
    if r1.color_lab is not None:
        assert np.array_equal(r1.color_lab, r2.color_lab)


def test_different_images_give_different_features():
    cfg = get_config()
    ex = HandcraftedFeatureExtractor(cfg)
    img_a = _sample_image()
    img_b = np.full((128, 128, 3), 60, dtype=np.uint8)  # visibly different image
    cv2.circle(img_b, (64, 64), 50, (200, 190, 180), -1)
    r_a = ex.extract(img_a)
    r_b = ex.extract(img_b)
    assert not np.array_equal(r_a.combined, r_b.combined)


def test_cache_roundtrip_preserves_psd_id_alignment(tmp_path):
    cfg = dataclasses.replace(get_config(), aef_crc_phase4_artifacts_dir=tmp_path / "phase4")
    ex = HandcraftedFeatureExtractor(cfg)
    img_a, img_b = _sample_image(), _sample_image() + 10  # slightly different

    result_a = ex.extract_and_cache(img_a, "PSD_AAAA")
    result_b = ex.extract_and_cache(img_b, "PSD_BBBB")

    # Re-extracting the SAME psd_id must return the CACHED result, not
    # recompute -- verified by checking the cache file's mtime doesn't change.
    glcm_path = cfg.aef_crc_phase4_artifacts_dir / cfg.preprocessing_mode / "glcm" / "PSD_AAAA.npy"
    mtime_before = glcm_path.stat().st_mtime_ns
    result_a_again = ex.extract_and_cache(img_a, "PSD_AAAA")
    mtime_after = glcm_path.stat().st_mtime_ns
    assert mtime_before == mtime_after, "Cache was recomputed instead of reused"
    assert np.array_equal(result_a.combined, result_a_again.combined)

    # Cross-ID contamination check: A and B must never share a cache slot.
    assert not np.array_equal(result_a.combined, result_b.combined)
    assert (cfg.aef_crc_phase4_artifacts_dir / cfg.preprocessing_mode / "glcm" / "PSD_BBBB.npy").exists()


def test_cache_is_namespaced_by_preprocessing_mode(tmp_path):
    """Item 2 fix's own regression test: switching preprocessing_mode
    between two extractions of the SAME psd_id must NOT return a stale
    cached result from the other mode."""
    base_cfg = dataclasses.replace(get_config(), aef_crc_phase4_artifacts_dir=tmp_path / "phase4")
    img = _sample_image()

    standard_cfg = dataclasses.replace(base_cfg, preprocessing_mode="standard")
    conditional_cfg = dataclasses.replace(base_cfg, preprocessing_mode="conditional")

    result_standard = HandcraftedFeatureExtractor(standard_cfg).extract_and_cache(img, "PSD_SAME_ID")
    result_conditional = HandcraftedFeatureExtractor(conditional_cfg).extract_and_cache(img, "PSD_SAME_ID")

    # Both must exist independently on disk -- neither extraction may
    # have silently reused the other mode's cache file.
    assert (tmp_path / "phase4" / "standard" / "glcm" / "PSD_SAME_ID.npy").exists()
    assert (tmp_path / "phase4" / "conditional" / "glcm" / "PSD_SAME_ID.npy").exists()


def test_normalizer_refuses_double_fit():
    norm = FeatureNormalizer()
    norm.fit([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    with pytest.raises(RuntimeError):
        norm.fit([np.array([5.0, 6.0])])


def test_normalizer_transform_before_fit_is_refused():
    norm = FeatureNormalizer()
    with pytest.raises(RuntimeError):
        norm.transform([np.array([1.0, 2.0])])


def test_normalizer_output_has_zero_mean_unit_std_on_training_data():
    norm = FeatureNormalizer()
    train_vecs = [np.array([1.0, 10.0]), np.array([3.0, 20.0]), np.array([5.0, 30.0])]
    norm.fit(train_vecs)
    transformed = norm.transform(train_vecs)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-8)


# ============================================================
# FoldSafeFeatureReducer (item 1: HOG dimension control)
# ============================================================

def test_reducer_output_dimension_matches_n_components():
    from modules.handcrafted_features import FoldSafeFeatureReducer
    rng = np.random.default_rng(0)
    train_vecs = [rng.normal(size=50) for _ in range(20)]
    reducer = FoldSafeFeatureReducer(n_components=5).fit(train_vecs)
    reduced = reducer.transform(train_vecs)
    assert reduced.shape == (20, 5)


def test_reducer_refuses_double_fit():
    from modules.handcrafted_features import FoldSafeFeatureReducer
    reducer = FoldSafeFeatureReducer(n_components=2)
    reducer.fit([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])])
    with pytest.raises(RuntimeError):
        reducer.fit([np.array([7.0, 8.0, 9.0])])


def test_reducer_transform_before_fit_is_refused():
    from modules.handcrafted_features import FoldSafeFeatureReducer
    reducer = FoldSafeFeatureReducer(n_components=2)
    with pytest.raises(RuntimeError):
        reducer.transform([np.array([1.0, 2.0])])
