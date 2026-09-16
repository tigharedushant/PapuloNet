# SYNTHETIC_TEST_FIXTURE = True
# This test uses strictly synthetic, generated, or mock fixtures.
# Zero real dataset images or private research artifacts are required or accessed.
"""
tests/test_xai_contract.py

Targeted contract tests for Explainable AI (Phase 10):
  1. Grad-CAM:
     - Spatial explanation of the EfficientNet-B0 disease-classification representation.
     - Targets the PRE-SOFTMAX logits of the Dense-4 classification head.
     - Strictly separates CNN spatial attribution from downstream Random Forest prediction.
     - Never backpropagates through Random Forest, Platt scaling, or conformal prediction.
      2. TreeSHAP:
         - Feature-level attribution of the final Random Forest prediction on the BDA-selected vector.
         - Uses TreeSHAP with feature_perturbation="tree_path_dependent" and model_output="raw".
         - Explains Random Forest's uncalibrated raw output space before Platt calibration.
         - Maps BDA-selected features back to canonical 1348-D indices and branch families.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.xai_gradcam import GradCAMExplainer, GradCAMResult, GradCAMError
from modules.xai_shap import (
    ProductionTreeSHAPExplainer, SHAPExplanationResult, SHAPError, get_canonical_feature_names,
)


CLASSES = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]


# ============================================================
# Canonical 1348-D Feature Contract
# ============================================================

def test_canonical_feature_names_matches_1348d_contract():
    names = get_canonical_feature_names()
    assert len(names) == 1348
    
    # 0:1280 EfficientNet-B0
    assert all(b == "deep" for b, _ in names[0:1280])
    assert names[0] == ("deep", "deep_feat_0000")
    assert names[1279] == ("deep", "deep_feat_1279")

    # 1280:1292 GLCM (12)
    assert all(b == "glcm" for b, _ in names[1280:1292])
    assert len(names[1280:1292]) == 12

    # 1292:1310 LBP (18)
    assert all(b == "lbp" for b, _ in names[1292:1310])
    assert len(names[1292:1310]) == 18

    # 1310:1342 HOG-PCA (32)
    assert all(b == "hog" for b, _ in names[1310:1342])
    assert len(names[1310:1342]) == 32

    # 1342:1348 LAB (6)
    assert all(b == "color_lab" for b, _ in names[1342:1348])
    assert len(names[1342:1348]) == 6


# ============================================================
# TreeSHAP Contract
# ============================================================

def test_shap_explainer_rf_initialization_and_contract():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(42)
    
    # Synthetic training data for RF (K=50 features out of 1348)
    n_samples = 60
    k_features = 50
    X = rng.normal(size=(n_samples, k_features))
    y = rng.integers(0, 4, size=n_samples)
    
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    # Selected mask into 1348-D
    mask = np.zeros(1348, dtype=bool)
    mask[:k_features] = True
    
    explainer = ProductionTreeSHAPExplainer(
        classifier=rf,
        selected_feature_mask=mask,
        classes=CLASSES,
    )
    
    assert explainer.output_space == "raw"
    assert explainer.perturbation_mode == "tree_path_dependent"
    assert explainer.k_features == k_features
    assert explainer.mask.shape == (1348,)
    assert len(explainer.classes) == 4

    # Exercise explain on a 1348-D sample
    sample_1348 = rng.normal(size=1348)
    result = explainer.explain(sample_1348)
    assert isinstance(result, SHAPExplanationResult)
    assert result.shap_output_space == "raw"
    assert result.shap_perturbation_mode == "tree_path_dependent"
    assert len(result.feature_contributions) == k_features
    assert len(result.block_contributions) > 0


def test_shap_explainer_rejects_dimension_mismatch():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(42)
    
    X = rng.normal(size=(30, 20))
    y = rng.integers(0, 4, size=30)
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    
    # Mask selects 25 features, but RF was fit on 20
    mask = np.zeros(1348, dtype=bool)
    mask[:25] = True
    
    with pytest.raises(SHAPError, match="Classifier expects 20 features, but BDA mask selected 25"):
        ProductionTreeSHAPExplainer(classifier=rf, selected_feature_mask=mask, classes=CLASSES)

    # Mask length is not 1348
    bad_length_mask = np.zeros(100, dtype=bool)
    with pytest.raises(SHAPError, match="BDA mask must have length 1348"):
        ProductionTreeSHAPExplainer(classifier=rf, selected_feature_mask=bad_length_mask, classes=CLASSES)


# ============================================================
# Grad-CAM Pre-Softmax Contract
# ============================================================

def test_gradcam_result_contract():
    """GradCAMResult must hold pre-softmax logit, normalized heatmaps,
    and preserve original image dimensions."""
    h, w = 224, 224
    raw_h, raw_w = 7, 7
    raw_hm = np.ones((raw_h, raw_w), dtype=np.float32)
    resized_hm = np.ones((h, w), dtype=np.float32)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    res = GradCAMResult(
        target_class_idx=0,
        target_class_name="Psoriasis",
        target_layer_name="top_conv",
        pre_softmax_logit=2.45,
        raw_heatmap=raw_hm,
        resized_heatmap=resized_hm,
        overlay_rgb=overlay,
        model_name="EfficientNet-B0",
        target_layer_shape=[1, 7, 7, 1280],
    )
    
    assert res.target_class_name == "Psoriasis"
    assert res.pre_softmax_logit == 2.45
    assert res.resized_heatmap.shape == (224, 224)
    assert res.overlay_rgb.shape == (224, 224, 3)
    assert np.all(res.resized_heatmap >= 0.0) and np.all(res.resized_heatmap <= 1.0)
