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
# Canonical 1316-D Production A7 Feature Contract
# ============================================================

def test_canonical_feature_names_matches_1316d_production_a7_contract():
    """
    Verifies that the production canonical feature definitions strictly match the
    1316-D A7 fusion layout (deep 1280, glcm 12, lbp 18, color_lab 6) with no HOG block.
    Also verifies isolated legacy A6 compatibility (1348-D with HOG).
    """
    # 1. Production A7 layout (default)
    names = get_canonical_feature_names()
    assert len(names) == 1316
    
    # 0:1280 EfficientNet-B0 (1280)
    assert all(b == "deep" for b, _ in names[0:1280])
    assert names[0] == ("deep", "deep_feat_0000")
    assert names[1279] == ("deep", "deep_feat_1279")

    # 1280:1292 GLCM (12)
    assert all(b == "glcm" for b, _ in names[1280:1292])
    assert len(names[1280:1292]) == 12

    # 1292:1310 LBP (18)
    assert all(b == "lbp" for b, _ in names[1292:1310])
    assert len(names[1292:1310]) == 18

    # 1310:1316 LAB Color Statistics (6)
    assert all(b == "color_lab" for b, _ in names[1310:1316])
    assert len(names[1310:1316]) == 6
    assert names[1310] == ("color_lab", "color_lab_L_mean")
    assert names[1315] == ("color_lab", "color_lab_b_std")

    # Strictly no HOG block in production A7
    assert not any(b == "hog" for b, _ in names)

    # 2. Legacy A6 compatibility mode (1348-D)
    names_legacy = get_canonical_feature_names(layout="A6")
    assert len(names_legacy) == 1348
    assert any(b == "hog" for b, _ in names_legacy)
    assert len([b for b, _ in names_legacy if b == "hog"]) == 32


# ============================================================
# TreeSHAP Contract (Production A7 1316-D / 194-D Subspace)
# ============================================================

def test_shap_explainer_rf_initialization_and_contract():
    """
    Verifies that ProductionTreeSHAPExplainer initializes against the
    production 194-feature Random Forest and A7 1316-D BDA mask contract,
    and explains both full 1316-D and already-masked 194-D inputs.
    """
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(42)
    
    # Synthetic training data matching the production active count (K=194 features out of 1316)
    n_samples = 60
    k_features = 194
    X = rng.normal(size=(n_samples, k_features))
    y = rng.integers(0, 4, size=n_samples)
    
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    # Selected mask into 1316-D
    mask = np.zeros(1316, dtype=bool)
    mask[:k_features] = True
    
    explainer = ProductionTreeSHAPExplainer(
        classifier=rf,
        selected_feature_mask=mask,
        classes=CLASSES,
    )
    
    assert explainer.output_space == "raw"
    assert explainer.perturbation_mode == "tree_path_dependent"
    assert explainer.k_features == k_features
    assert explainer.mask.shape == (1316,)
    assert explainer.full_dim == 1316
    assert explainer.layout == "A7"
    assert len(explainer.classes) == 4

    # Exercise explain on a 1316-D full fused vector
    sample_1316 = rng.normal(size=1316)
    result = explainer.explain(sample_1316)
    assert isinstance(result, SHAPExplanationResult)
    assert result.shap_output_space == "raw"
    assert result.shap_perturbation_mode == "tree_path_dependent"
    assert len(result.feature_contributions) == k_features
    assert len(result.block_contributions) == 4  # deep, glcm, lbp, color_lab
    assert all(b.branch != "hog" for b in result.block_contributions)

    # Exercise explain on an already-masked 194-D vector
    sample_194 = rng.normal(size=194)
    result_masked = explainer.explain(sample_194)
    assert isinstance(result_masked, SHAPExplanationResult)
    assert len(result_masked.feature_contributions) == k_features


def test_shap_explainer_rejects_dimension_mismatch():
    """
    Verifies that ProductionTreeSHAPExplainer strictly rejects:
    1. Mask count mismatch with classifier n_features_in_
    2. Mask dimension mismatch (neither 1316 nor legacy 1348)
    3. Input feature vector dimension mismatch
    """
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(42)
    
    X = rng.normal(size=(30, 194))
    y = rng.integers(0, 4, size=30)
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    
    # Mask selects 200 features, but RF was fit on 194
    mask = np.zeros(1316, dtype=bool)
    mask[:200] = True
    
    with pytest.raises(SHAPError, match="Classifier expects 194 features, but BDA mask selected 200"):
        ProductionTreeSHAPExplainer(classifier=rf, selected_feature_mask=mask, classes=CLASSES)

    # Mask length is not 1316 (e.g. 100)
    bad_length_mask = np.zeros(100, dtype=bool)
    with pytest.raises(SHAPError, match="BDA mask dimension mismatch"):
        ProductionTreeSHAPExplainer(classifier=rf, selected_feature_mask=bad_length_mask, classes=CLASSES)

    # Input features dimension mismatch during explain
    valid_mask = np.zeros(1316, dtype=bool)
    valid_mask[:194] = True
    explainer = ProductionTreeSHAPExplainer(classifier=rf, selected_feature_mask=valid_mask, classes=CLASSES)
    with pytest.raises(SHAPError, match="matches neither full dimension 1316 nor K=194"):
        explainer.explain(np.zeros(500))


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


def test_gradcam_mixed_precision_dtype_safety():
    """
    Verifies that GradCAMExplainer executes without InvalidArgumentError when:
    - conv_output / pooled feature tensor is float16
    - Dense head kernel and bias are float32
    and produces valid float32 logits with the expected (1, 4) shape.
    """
    import tensorflow as tf

    # Minimal model where conv layer produces float16 output and Dense head is float32
    inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
    # Conv layer explicitly producing float16
    conv = tf.keras.layers.Conv2D(16, (3, 3), padding="same", name="target_conv", dtype=tf.float16)(inputs)
    # Flatten/pool to feed dense head
    gap = tf.keras.layers.GlobalAveragePooling2D(dtype=tf.float32)(conv)
    dense = tf.keras.layers.Dense(4, name="disease_dense", dtype=tf.float32)(gap)
    model = tf.keras.Model(inputs=inputs, outputs=dense)

    explainer = GradCAMExplainer(
        model=model,
        classes=CLASSES,
        target_layer_name="target_conv",
        device="auto",
    )

    assert explainer.dense_kernel.dtype == tf.float32
    assert explainer.dense_bias.dtype == tf.float32

    # Verify that conv_submodel produces float16
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    sub_out = explainer.conv_submodel(dummy_input)
    assert sub_out.dtype == tf.float16

    # Execute explain targeting 'Psoriasis'
    result = explainer.explain(dummy_input, target_class="Psoriasis")

    assert isinstance(result, GradCAMResult)
    assert result.target_class_name == "Psoriasis"
    assert isinstance(result.pre_softmax_logit, float)
    assert not np.isnan(result.pre_softmax_logit)
    assert result.raw_heatmap.ndim == 2
    assert result.raw_heatmap.dtype == np.float32
    assert result.resized_heatmap.shape == (224, 224)
    assert np.all(result.resized_heatmap >= 0.0) and np.all(result.resized_heatmap <= 1.0)


def test_shap_explain_api_contract_and_rejection_of_unsupported_keywords():
    """
    Verifies that ProductionTreeSHAPExplainer.explain():
    1. Successfully executes with keyword features=masked_194 and positionally (masked_194).
    2. Successfully executes with full 1316-D vector (features=full_1316).
    3. Rejects unsupported keyword arguments (e.g. masked_features=...) with TypeError.
    4. Guarantees that masking is not redundantly re-applied when passing 194-D.
    """
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(42)

    k_features = 194
    X = rng.normal(size=(20, k_features))
    y = rng.integers(0, 4, size=20)
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    mask = np.zeros(1316, dtype=bool)
    mask[:k_features] = True

    explainer = ProductionTreeSHAPExplainer(
        classifier=rf,
        selected_feature_mask=mask,
        classes=CLASSES,
    )

    masked_194 = rng.normal(size=k_features)

    # 1. Calling with features=masked_194 (keyword) must succeed
    res_kw = explainer.explain(features=masked_194, target_class="Psoriasis")
    assert isinstance(res_kw, SHAPExplanationResult)
    assert len(res_kw.feature_contributions) == k_features

    # 2. Calling with masked_194 positionally must succeed
    res_pos = explainer.explain(masked_194, target_class="Psoriasis")
    assert isinstance(res_pos, SHAPExplanationResult)
    assert len(res_pos.feature_contributions) == k_features

    # 3. Calling with full 1316-D vector must succeed
    full_1316 = np.zeros(1316)
    full_1316[:k_features] = masked_194
    res_full = explainer.explain(features=full_1316, target_class="Psoriasis")
    assert isinstance(res_full, SHAPExplanationResult)

    # Values must match identically (confirming masking is identical and occurs exactly once)
    np.testing.assert_allclose(
        [f.shap_value for f in res_kw.feature_contributions],
        [f.shap_value for f in res_full.feature_contributions],
        atol=1e-6,
    )

    # 4. Calling with unsupported keyword 'masked_features' must raise TypeError
    with pytest.raises(TypeError, match="unexpected keyword argument 'masked_features'"):
        explainer.explain(masked_features=masked_194, target_class="Psoriasis")  # type: ignore


def test_shap_export_csvs_api_contract(tmp_path):
    """
    Verifies the export_csvs API contract:
    1. export_csvs() successfully writes feature CSV and block CSV.
    2. export_csvs() returns None (conforming to its type annotation -> None).
    3. Callers must invoke without tuple-unpacking (preventing 'cannot unpack non-iterable NoneType object').
    4. CSV contents accurately correspond to the SHAPExplanationResult data.
    """
    from sklearn.ensemble import RandomForestClassifier
    import csv

    rng = np.random.default_rng(42)
    k_features = 194
    X = rng.normal(size=(20, k_features))
    y = rng.integers(0, 4, size=20)
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    mask = np.zeros(1316, dtype=bool)
    mask[:k_features] = True

    explainer = ProductionTreeSHAPExplainer(
        classifier=rf,
        selected_feature_mask=mask,
        classes=CLASSES,
    )

    masked_194 = rng.normal(size=k_features)
    result = explainer.explain(features=masked_194, target_class="Psoriasis")

    feat_path = tmp_path / "shap_feature_importance.csv"
    block_path = tmp_path / "shap_block_importance.csv"

    # 1. export_csvs returns None
    ret = result.export_csvs(feature_csv_path=feat_path, block_csv_path=block_path)
    assert ret is None

    # 2. Files exist and are non-empty
    assert feat_path.exists() and feat_path.stat().st_size > 0
    assert block_path.exists() and block_path.stat().st_size > 0

    # 3. Tuple unpacking on return value raises TypeError
    with pytest.raises(TypeError, match="cannot unpack non-iterable NoneType object"):
        _f, _b = result.export_csvs(feature_csv_path=feat_path, block_csv_path=block_path)  # type: ignore

    # 4. Verify CSV contents match explanation result
    with open(feat_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == k_features
        assert reader[0]["branch"] in ("deep", "glcm", "lbp", "color_lab")
        assert reader[0]["branch"] != "hog"
        assert float(reader[0]["shap_value"]) == pytest.approx(result.feature_contributions[0].shap_value, abs=1e-5)

    with open(block_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 4
        branches = [row["branch"] for row in reader]
        assert set(branches) == {"deep", "glcm", "lbp", "color_lab"}
        assert "hog" not in branches

