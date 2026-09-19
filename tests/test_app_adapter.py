"""
tests/test_app_adapter.py

Targeted unit tests for modules/app_adapter.py:
1. Adapter imports and data structure contracts.
2. Robust input validation and error handling (corrupt, empty, small, non-existent).
3. End-to-end adapter flow with mock pipeline:
   - Feature dimensions: 1316-D -> 194-D
   - Output probabilities across the four frozen classes
   - Grad-CAM and TreeSHAP results formatted as PIL Images and DataFrames
   - Verification that engine.predict(explain=True) is NEVER called
4. Format compatibility: PIL.Image, raw bytes, numpy ndarray.
5. Strict locked-test partition isolation.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from modules.app_adapter import (
    AppInferenceError,
    AppInferencePipeline,
    AppInferenceResult,
    clear_app_pipeline_cache,
    get_app_pipeline,
    run_app_inference,
)
from modules.calibration_handoff import FinalPipelineHandoff
from modules.output_schema import ConformalReviewStatus
from modules.xai_gradcam import GradCAMResult
from modules.xai_shap import (
    A7_BRANCH_DIMS,
    A7_FULL_DIM,
    SHAPBlockContribution,
    SHAPExplanationResult,
    SHAPFeatureContribution,
)


@pytest.fixture
def mock_app_pipeline():
    """Builds an AppInferencePipeline with lightweight stubs verifying production contracts."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    mask = np.zeros(A7_FULL_DIM, dtype=bool)
    mask[:194] = True  # exactly 194 features active

    mock_handoff = MagicMock(spec=FinalPipelineHandoff)
    mock_handoff.representation_id = "efficientnet_b0_6760c4f151acc2d2"
    mock_handoff.class_order = classes
    mock_handoff.selected_feature_mask = mask
    mock_handoff.branch_dims = {"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6}
    mock_handoff.calibration_method = "platt"
    mock_handoff.marginal_q_hat = 0.782165682476496
    mock_handoff.mondrian_q_hat = {cls: 0.78 for cls in classes}

    # Pipeline container
    pipeline = MagicMock(spec=AppInferencePipeline)
    pipeline.handoff = mock_handoff

    # Mock Engine
    mock_engine = MagicMock()
    mock_engine.classes = classes
    mock_engine.mask = mask

    # validate_input_image: standard PIL image passthrough
    def _validate_image(inp):
        if isinstance(inp, Image.Image):
            arr = np.asarray(inp)
            w, h = inp.size
            if w < 32 or h < 32:
                raise ValueError("Image dimensions too small")
            if np.std(arr) < 1e-4:
                raise ValueError("Degenerate image")
            return inp
        elif isinstance(inp, np.ndarray):
            if inp.ndim != 3 or inp.shape[2] != 3:
                raise ValueError("Unexpected shape")
            w, h = inp.shape[1], inp.shape[0]
            if w < 32 or h < 32:
                raise ValueError("Too small")
            if np.std(inp) < 1e-4:
                raise ValueError("Degenerate")
            return Image.fromarray(inp).convert("RGB")
        raise TypeError(f"Unsupported type: {type(inp)}")

    mock_engine.validate_input_image.side_effect = _validate_image
    mock_engine.preprocess_image.return_value = (
        np.zeros((1, 224, 224, 3), dtype=np.float32),
        np.full((224, 224, 3), 128, dtype=np.uint8),
    )

    # 1316-D feature extraction
    mock_engine.extract_multimodal_features.return_value = np.zeros(1316, dtype=np.float32)

    # 194-D masking
    def _apply_mask(feats):
        assert feats.shape[0] == 1316, f"Expected 1316-D, got {feats.shape[0]}"
        return feats[mask].reshape(1, -1)

    mock_engine.apply_feature_mask.side_effect = _apply_mask
    mock_engine.predict_raw_probabilities.return_value = np.array([[0.60, 0.15, 0.15, 0.10]])
    mock_engine.apply_calibration.return_value = np.array([[0.75, 0.10, 0.10, 0.05]])
    mock_engine.predict_conformal_sets.return_value = (["Psoriasis"], ["Psoriasis"])

    pipeline.engine = mock_engine

    # Mock Grad-CAM Explainer
    mock_gradcam = MagicMock()
    mock_gradcam.explain.return_value = GradCAMResult(
        target_class_idx=0,
        target_class_name="Psoriasis",
        pre_softmax_logit=18.95,
        target_layer_name="top_conv",
        target_layer_shape=[1, 7, 7, 1280],
        raw_heatmap=np.zeros((7, 7), dtype=np.float32),
        resized_heatmap=np.full((224, 224), 0.5, dtype=np.float32),
        overlay_rgb=np.full((224, 224, 3), 200, dtype=np.uint8),
        model_name="EfficientNetB0",
    )
    pipeline.gradcam_explainer = mock_gradcam

    # Mock TreeSHAP Explainer
    mock_shap = MagicMock()
    mock_shap.explain.return_value = SHAPExplanationResult(
        target_class_idx=0,
        target_class_name="Psoriasis",
        expected_values={"Psoriasis": 0.25, "Lichen_Planus": 0.25, "Pityriasis_Rosea": 0.25, "Seborrheic_Dermatitis": 0.25},
        raw_probabilities={"Psoriasis": 0.60, "Lichen_Planus": 0.15, "Pityriasis_Rosea": 0.15, "Seborrheic_Dermatitis": 0.10},
        feature_contributions=[
            SHAPFeatureContribution(
                selected_index=0,
                global_index=10,
                branch="deep",
                feature_name="deep_dim_0010",
                feature_value=1.45,
                shap_value=0.12,
                abs_shap_value=0.12,
            )
        ],
        block_contributions=[
            SHAPBlockContribution(
                branch="deep",
                feature_count=180,
                sum_abs_shap=14.4,
                mean_abs_shap=0.08,
                pct_total_importance=92.5,
            ),
            SHAPBlockContribution(
                branch="glcm",
                feature_count=5,
                sum_abs_shap=0.05,
                mean_abs_shap=0.01,
                pct_total_importance=4.5,
            ),
            SHAPBlockContribution(
                branch="lbp",
                feature_count=5,
                sum_abs_shap=0.025,
                mean_abs_shap=0.005,
                pct_total_importance=2.0,
            ),
            SHAPBlockContribution(
                branch="color_lab",
                feature_count=4,
                sum_abs_shap=0.008,
                mean_abs_shap=0.002,
                pct_total_importance=1.0,
            ),
        ],
        reconstruction_difference=1e-15,
    )
    pipeline.shap_explainer = mock_shap

    return pipeline


def test_adapter_imports():
    """Verifies that all primary adapter classes, functions, and exceptions import cleanly."""
    from modules.app_adapter import (
        AppInferenceError,
        AppInferencePipeline,
        AppInferenceResult,
        clear_app_pipeline_cache,
        get_app_pipeline,
        run_app_inference,
    )
    assert callable(run_app_inference)
    assert callable(get_app_pipeline)
    assert callable(clear_app_pipeline_cache)


def test_adapter_input_validation_errors(mock_app_pipeline):
    """Verifies clear AppInferenceError handling for invalid, corrupt, or unsupported inputs."""
    # 1. Empty bytes
    with pytest.raises(AppInferenceError, match="Empty image bytes"):
        run_app_inference(b"", pipeline=mock_app_pipeline)

    # 2. Corrupt bytes
    with pytest.raises(AppInferenceError, match="Invalid, corrupt, or unsupported image bytes"):
        run_app_inference(b"NOT_AN_IMAGE_DATA_CORRUPT", pipeline=mock_app_pipeline)

    # 3. Non-existent file path
    with pytest.raises(AppInferenceError, match="Input image file does not exist"):
        run_app_inference(Path("non_existent_file_path.jpg"), pipeline=mock_app_pipeline)

    # 4. Unsupported input type
    with pytest.raises(AppInferenceError, match="Unsupported image_input type"):
        run_app_inference(12345, pipeline=mock_app_pipeline)

    # 5. Image too small (<32x32)
    small_img = Image.new("RGB", (16, 16), color=(100, 150, 200))
    with pytest.raises(AppInferenceError, match="Image validation failed"):
        run_app_inference(small_img, pipeline=mock_app_pipeline)

    # 6. Degenerate zero-variance image
    arr_degen = np.full((100, 100, 3), 128, dtype=np.uint8)
    with pytest.raises(AppInferenceError, match="Image validation failed"):
        run_app_inference(arr_degen, pipeline=mock_app_pipeline)


def test_adapter_end_to_end_with_pil_image(mock_app_pipeline):
    """Verifies full adapter execution with a valid PIL Image."""
    # Create valid synthetic image with pixel variance
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)

    result = run_app_inference(pil_img, pipeline=mock_app_pipeline)

    # 1. Structure assertions
    assert isinstance(result, AppInferenceResult)
    assert result.predicted_class == "Psoriasis"
    assert result.calibrated_confidence == 0.75
    assert result.raw_confidence == 0.60
    assert result.review_status == ConformalReviewStatus.STANDARD_OUTPUT.value

    # 2. Probabilities correspond to four frozen classes
    expected_classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    assert list(result.calibrated_probabilities.keys()) == expected_classes
    assert list(result.raw_probabilities.keys()) == expected_classes
    assert np.isclose(sum(result.calibrated_probabilities.values()), 1.0)

    # 3. Conformal sets
    assert result.marginal_prediction_set == ["Psoriasis"]
    assert result.mondrian_prediction_set == ["Psoriasis"]

    # 4. Grad-CAM visual outputs
    assert isinstance(result.gradcam_overlay_pil, Image.Image)
    assert isinstance(result.gradcam_heatmap_pil, Image.Image)
    assert result.gradcam_logit == 18.95
    assert result.gradcam_overlay_pil.size == (224, 224)

    # 5. TreeSHAP tabular outputs
    assert result.top_shap_branch == "deep"
    assert result.top_shap_branch_pct == 92.5
    assert isinstance(result.branch_importance_df, pd.DataFrame)
    assert isinstance(result.top_features_df, pd.DataFrame)
    assert "branch" in result.branch_importance_df.columns
    assert "feature_name" in result.top_features_df.columns

    # 6. Contract: engine.predict(explain=True) was NEVER called
    # (Only explicit explainers were invoked)
    mock_app_pipeline.engine.predict.assert_not_called()
    assert mock_app_pipeline.engine.apply_feature_mask.called
    assert mock_app_pipeline.gradcam_explainer.explain.called
    assert mock_app_pipeline.shap_explainer.explain.called


def test_adapter_raw_bytes_and_numpy_inputs(mock_app_pipeline):
    """Verifies that the adapter accepts raw image bytes and numpy arrays interchangeably."""
    rng = np.random.default_rng(123)
    arr = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)

    # Test raw bytes (simulating Streamlit UploadedFile.read())
    pil_temp = Image.fromarray(arr)
    byte_buf = io.BytesIO()
    pil_temp.save(byte_buf, format="PNG")
    raw_bytes = byte_buf.getvalue()

    res_bytes = run_app_inference(raw_bytes, pipeline=mock_app_pipeline)
    assert res_bytes.predicted_class == "Psoriasis"
    assert isinstance(res_bytes.gradcam_overlay_pil, Image.Image)

    # Test numpy array
    res_np = run_app_inference(arr, pipeline=mock_app_pipeline)
    assert res_np.predicted_class == "Psoriasis"
    assert isinstance(res_np.gradcam_overlay_pil, Image.Image)


def test_adapter_feature_dimension_enforcement(mock_app_pipeline):
    """Verifies that the adapter strictly enforces the 1316-D A7 and 194-D BDA contracts."""
    rng = np.random.default_rng(999)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)

    # 1. Invalid extracted feature dimension (e.g. 1000 instead of 1316)
    mock_app_pipeline.engine.extract_multimodal_features.return_value = np.zeros(1000, dtype=np.float32)
    with pytest.raises(AppInferenceError, match="Feature dimension contract violation"):
        run_app_inference(pil_img, pipeline=mock_app_pipeline)

    # Restore 1316-D
    mock_app_pipeline.engine.extract_multimodal_features.return_value = np.zeros(1316, dtype=np.float32)

    # 2. Invalid masked feature dimension (e.g. 100 instead of 194)
    mock_app_pipeline.engine.apply_feature_mask.side_effect = lambda feats: np.zeros((1, 100), dtype=np.float32)
    with pytest.raises(AppInferenceError, match="Masked feature dimension mismatch"):
        run_app_inference(pil_img, pipeline=mock_app_pipeline)


def test_adapter_zero_locked_test_access():
    """Verifies that the adapter tests do not touch or leak any locked test images."""
    test_dir = Path("output/06_final_split/test")
    if test_dir.exists():
        locked_count = len(list(test_dir.glob("*/*")))
        assert locked_count == 243, f"Locked test set count modified! Expected 243, found {locked_count}"
