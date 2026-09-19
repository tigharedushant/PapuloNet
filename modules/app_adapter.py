"""
modules/app_adapter.py

Application-Facing Inference and Explainability Adapter for AEF-CRC.
Provides a clean, unified, in-memory inference entry point for new user images
without training, refitting, recalibrating, or touching the locked test partition.

SCIENTIFIC AND ARCHITECTURAL CONTRACT:
--------------------------------------
1. Input Handling: Accepts Path, str, PIL.Image, np.ndarray, or raw uploaded image bytes.
2. Inference Stages:
   - Validates image dimensions (>=32x32), RGB variance, and format.
   - Applies standardized preprocessing (ConditionalPreprocessor + Keras preprocess_input).
   - Extracts 1316-D multimodal representation (EfficientNet-B0 1280, GLCM 12, LBP 18, Color LAB 6).
   - Applies frozen Phase 6 BDA 194-D boolean mask.
   - Runs production Random Forest (194-D) -> raw uncalibrated probabilities.
   - Applies Phase 8 Platt sigmoid probability calibration.
   - Computes Phase 9 Split-Conformal prediction sets (Marginal 90% + Mondrian Class-Conditional).
   - Determines clinical review status (STANDARD_OUTPUT, SPECIALIST_REVIEW_REQUIRED, etc.).
3. Explainable AI:
   - Grad-CAM: Explicitly targets the pre-softmax logit of the EfficientNet-B0 Dense head.
   - TreeSHAP: Explicitly computes raw prediction attributions across the 194 active features.
   - Explanations are packaged as PIL images and Pandas DataFrames for direct UI consumption.
4. Boundaries:
   - Does NOT call engine.predict(..., explain=True).
   - Operates in-memory without saving user uploads into the training or validation datasets.
   - Does NOT touch the locked 243-image test partition.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from modules.calibration_handoff import (
    FinalPipelineHandoff,
    load_final_pipeline_handoff,
)
from modules.inference import AEFCRCInferenceEngine
from modules.output_schema import ConformalReviewStatus, determine_conformal_review_status
from modules.xai_gradcam import GradCAMExplainer, GradCAMResult
from modules.xai_shap import (
    A7_BRANCH_DIMS,
    A7_FULL_DIM,
    ProductionTreeSHAPExplainer,
    SHAPExplanationResult,
)


class AppInferenceError(Exception):
    """Raised when application-facing inference fails or input is invalid."""
    pass


@dataclass
class AppInferenceResult:
    """Structured, UI-ready result from the end-to-end AEF-CRC inference adapter."""
    predicted_class: str
    calibrated_confidence: float
    raw_confidence: float
    calibrated_probabilities: Dict[str, float]
    raw_probabilities: Dict[str, float]
    marginal_prediction_set: List[str]
    mondrian_prediction_set: List[str]
    review_status: str
    review_status_explanation: str
    gradcam_overlay_pil: Optional[Image.Image]
    gradcam_heatmap_pil: Optional[Image.Image]
    gradcam_logit: float
    top_shap_branch: str
    top_shap_branch_pct: float
    branch_importance_df: pd.DataFrame
    top_features_df: pd.DataFrame
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class AppInferencePipeline:
    """
    Preloads and caches the frozen handoff, inference engine, and XAI explainers
    to avoid redundant disk I/O and model reloading across multiple user requests.
    """

    def __init__(
        self,
        handoff: FinalPipelineHandoff,
        device: str = "auto",
    ) -> None:
        self.handoff = handoff
        self.device = device
        self.engine = AEFCRCInferenceEngine(artifact=handoff, device=device)
        self.gradcam_explainer = GradCAMExplainer(
            model=self.engine._get_backbone(),
            classes=handoff.class_order,
            device=device,
        )
        self.shap_explainer = ProductionTreeSHAPExplainer(
            classifier=handoff.final_classifier,
            selected_feature_mask=handoff.selected_feature_mask,
            classes=handoff.class_order,
            branch_dims=handoff.branch_dims,
        )


_PIPELINE_CACHE: Dict[Tuple[str, str], AppInferencePipeline] = {}


def get_app_pipeline(
    handoff_or_path: Union[FinalPipelineHandoff, Path, str] = Path("artifacts/phase9/final_pipeline_handoff.joblib"),
    device: str = "auto",
    use_cache: bool = True,
) -> AppInferencePipeline:
    """Retrieves or instantiates an AppInferencePipeline instance."""
    if isinstance(handoff_or_path, FinalPipelineHandoff):
        cache_key = (handoff_or_path.representation_id, device)
        if use_cache and cache_key in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[cache_key]
        pipeline = AppInferencePipeline(handoff=handoff_or_path, device=device)
        if use_cache:
            _PIPELINE_CACHE[cache_key] = pipeline
        return pipeline

    p = Path(handoff_or_path)
    if not p.exists():
        raise AppInferenceError(f"Handoff artifact file not found at: {p}")

    cache_key = (str(p.resolve()), device)
    if use_cache and cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    try:
        handoff = load_final_pipeline_handoff(p)
    except Exception as exc:
        raise AppInferenceError(f"Failed to load pipeline handoff from '{p}': {exc}") from exc

    pipeline = AppInferencePipeline(handoff=handoff, device=device)
    if use_cache:
        _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def clear_app_pipeline_cache() -> None:
    """Clears in-memory pipeline cache."""
    _PIPELINE_CACHE.clear()


def run_app_inference(
    image_input: Union[Path, str, Image.Image, np.ndarray, bytes],
    handoff_path: Union[FinalPipelineHandoff, Path, str] = Path("artifacts/phase9/final_pipeline_handoff.joblib"),
    device: str = "auto",
    pipeline: Optional[AppInferencePipeline] = None,
) -> AppInferenceResult:
    """
    Executes end-to-end prediction and explainability for a new user image.

    Parameters
    ----------
    image_input : Union[Path, str, Image.Image, np.ndarray, bytes]
        Input image in any supported format (file path, PIL Image, numpy array, or raw bytes).
    handoff_path : Union[FinalPipelineHandoff, Path, str]
        Path to the Phase 9 FinalPipelineHandoff joblib file (or a preloaded handoff).
    device : str
        TensorFlow device target ("auto", "gpu", or "cpu").
    pipeline : Optional[AppInferencePipeline]
        Optional pre-instantiated pipeline. If None, resolves via get_app_pipeline().

    Returns
    -------
    AppInferenceResult
        Structured dataclass with predictions, conformal sets, review status, and XAI outputs.
    """
    # 1. Resolve pipeline
    if pipeline is None:
        pipeline = get_app_pipeline(handoff_or_path=handoff_path, device=device)

    engine = pipeline.engine
    gradcam_explainer = pipeline.gradcam_explainer
    shap_explainer = pipeline.shap_explainer

    # 2. Input image resolution and decoding
    if isinstance(image_input, bytes):
        if len(image_input) == 0:
            raise AppInferenceError("Empty image bytes received.")
        try:
            buf = io.BytesIO(image_input)
            raw_pil = Image.open(buf)
            raw_pil.load()
            img_candidate = raw_pil.convert("RGB")
        except Exception as exc:
            raise AppInferenceError(f"Invalid, corrupt, or unsupported image bytes: {exc}") from exc
    elif isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists() or not p.is_file():
            raise AppInferenceError(f"Input image file does not exist: {p}")
        try:
            raw_pil = Image.open(p)
            raw_pil.load()
            img_candidate = raw_pil.convert("RGB")
        except Exception as exc:
            raise AppInferenceError(f"Failed to read image from path '{p}': {exc}") from exc
    elif isinstance(image_input, Image.Image):
        img_candidate = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img_candidate = image_input
    else:
        raise AppInferenceError(
            f"Unsupported image_input type: {type(image_input)}. "
            f"Expected Path, str, PIL.Image.Image, np.ndarray, or bytes."
        )

    # 3. Stage 1: Validation
    try:
        validated_img = engine.validate_input_image(img_candidate)
    except Exception as exc:
        raise AppInferenceError(f"Image validation failed: {exc}") from exc

    # 4. Stage 2: Standardized Preprocessing
    try:
        tensor, rgb_np = engine.preprocess_image(validated_img)
    except Exception as exc:
        raise AppInferenceError(f"Image preprocessing failed: {exc}") from exc

    # 5. Stage 3: Multimodal Feature Extraction (1316-D A7 layout)
    try:
        fused_1316 = engine.extract_multimodal_features(tensor, rgb_np)
    except Exception as exc:
        raise AppInferenceError(f"Multimodal feature extraction failed: {exc}") from exc

    if fused_1316.shape[0] != A7_FULL_DIM:
        raise AppInferenceError(
            f"Feature dimension contract violation: expected {A7_FULL_DIM}-D for A7 layout, "
            f"got {fused_1316.shape[0]}-D."
        )

    # 6. Stage 4: Feature Selection Masking (194-D production BDA mask)
    try:
        masked_194 = engine.apply_feature_mask(fused_1316)
    except Exception as exc:
        raise AppInferenceError(f"Feature selection masking failed: {exc}") from exc

    k_active = int(engine.mask.sum())
    if masked_194.shape[1] != k_active:
        raise AppInferenceError(
            f"Masked feature dimension mismatch: expected {k_active}-D, got {masked_194.shape[1]}-D."
        )

    # 7. Stage 5: Base Random Forest Classifier Inference (raw prediction)
    try:
        raw_probs = engine.predict_raw_probabilities(masked_194)
    except Exception as exc:
        raise AppInferenceError(f"Random Forest classification failed: {exc}") from exc

    # 8. Stage 6: Probability Calibration (Platt Sigmoid Scaling)
    try:
        calib_probs = engine.apply_calibration(raw_probs)
    except Exception as exc:
        raise AppInferenceError(f"Probability calibration failed: {exc}") from exc

    # 9. Stage 7: Conformal Prediction Sets (Marginal + Mondrian)
    try:
        marginal_set, mondrian_set = engine.predict_conformal_sets(calib_probs)
    except Exception as exc:
        raise AppInferenceError(f"Conformal prediction set generation failed: {exc}") from exc

    # 10. Point Prediction & Clinical Review Status
    pred_idx = int(np.argmax(calib_probs[0]))
    pred_cls = engine.classes[pred_idx]
    calib_conf = float(calib_probs[0][pred_idx])
    raw_conf = float(raw_probs[0][pred_idx])

    review_status_enum, review_explanation = determine_conformal_review_status(True, marginal_set)
    review_status_str = review_status_enum.value

    raw_dict = {cls: float(raw_probs[0][i]) for i, cls in enumerate(engine.classes)}
    calib_dict = {cls: float(calib_probs[0][i]) for i, cls in enumerate(engine.classes)}

    # 11. Explicit Grad-CAM Explanation (Backbone CNN pre-softmax logit)
    try:
        g_result = gradcam_explainer.explain(
            preprocessed_tensor=tensor,
            original_image_rgb=rgb_np,
            target_class=pred_cls,
        )
        overlay_arr = getattr(g_result, "overlay_rgb", None)
        if overlay_arr is None:
            overlay_arr = getattr(g_result, "overlay_image", None)
        overlay_pil = Image.fromarray(overlay_arr) if overlay_arr is not None else None
        heatmap_uint8 = np.clip(g_result.resized_heatmap * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        heatmap_pil = Image.fromarray(heatmap_rgb)
        gradcam_logit = float(g_result.pre_softmax_logit)
    except Exception as exc:
        raise AppInferenceError(f"Grad-CAM explanation failed: {exc}") from exc

    # 12. Explicit TreeSHAP Explanation (Random Forest 194-D tree_path_dependent)
    try:
        shap_result = shap_explainer.explain(
            features=masked_194,
            target_class=pred_cls,
        )
        branch_rows = [b.to_dict() for b in shap_result.block_contributions]
        branch_df = pd.DataFrame(branch_rows)

        feat_rows = [f.to_dict() for f in shap_result.feature_contributions]
        top_features_df = pd.DataFrame(feat_rows)

        top_branch = shap_result.block_contributions[0].branch if shap_result.block_contributions else "N/A"
        top_branch_pct = float(shap_result.block_contributions[0].pct_total_importance) if shap_result.block_contributions else 0.0
    except Exception as exc:
        raise AppInferenceError(f"TreeSHAP explanation failed: {exc}") from exc

    return AppInferenceResult(
        predicted_class=pred_cls,
        calibrated_confidence=calib_conf,
        raw_confidence=raw_conf,
        calibrated_probabilities=calib_dict,
        raw_probabilities=raw_dict,
        marginal_prediction_set=marginal_set,
        mondrian_prediction_set=mondrian_set,
        review_status=review_status_str,
        review_status_explanation=review_explanation,
        gradcam_overlay_pil=overlay_pil,
        gradcam_heatmap_pil=heatmap_pil,
        gradcam_logit=gradcam_logit,
        top_shap_branch=top_branch,
        top_shap_branch_pct=top_branch_pct,
        branch_importance_df=branch_df,
        top_features_df=top_features_df,
        diagnostics={
            "fused_dimension": int(fused_1316.shape[0]),
            "masked_dimension": int(masked_194.shape[1]),
            "target_layer_name": g_result.target_layer_name,
            "shap_reconstruction_diff": shap_result.reconstruction_difference,
        },
    )
