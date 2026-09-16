"""
modules/inference.py

AEF-CRC End-to-End Inference Engine with Integrated XAI & Conformal Review Logic.

Implements the authoritative 9-stage inference contract:
1. Input Image Validation: Format, channel, dimension, and non-degeneracy checks.
2. Standardized Preprocessing: Conditional or standard preprocessing respecting config.preprocessing_mode
   and ImageNet normalization via TensorFlow/Keras EfficientNet preprocess_input.
3. Multimodal Feature Extraction: 1348-D fused vector:
   EfficientNet-B0 (1280) + GLCM (12) + LBP (18) + HOG-PCA (32) + LAB (6).
   Applies fitted FoldSafeFeatureReducer (HOG PCA) and FeatureNormalizers across all branches.
4. Feature Selection Masking: Application of the single frozen Phase 6 BDA production mask.
5. Base Classifier Inference: Frozen Random Forest predict_proba() generating raw probabilities.
6. Probability Calibration: Phase 8 Platt (sigmoid primary) / Isotonic transform generating calibrated probabilities.
7. Conformal Prediction: Phase 9 split-conformal thresholding generating guaranteed prediction sets
   (Primary: 90% Marginal Split-Conformal; Secondary: Mondrian Class-Conditional).
8. Clinical Review Status Determination:
   - STANDARD_OUTPUT (singleton set)
   - SPECIALIST_REVIEW_REQUIRED (multi-class set)
   - NO_CLASS_MEETS_CONFORMAL_THRESHOLD (empty set; NOT an OOD detector)
   - INPUT_QUALITY_FAILURE (rejected image)
9. Explainable AI (Optional / On-Demand):
   - Grad-CAM: Pre-softmax logit targeting on Keras CNN backbone (spatial representation).
   - TreeSHAP: tree_path_dependent attributions on raw RF outputs across BDA-selected features,
     with canonical 1348-D mapping and per-class expected values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from modules.calibration import apply_platt_scaling, apply_isotonic_scaling, clip_probabilities
from modules.calibration_handoff import (
    ConformalArtifact,
    FinalPipelineHandoff,
    load_conformal_artifact,
    load_final_pipeline_handoff,
)
from modules.output_schema import (
    ConformalReviewStatus,
    SingleImageInferenceResponse,
    XAIExplanationResult,
    determine_conformal_review_status,
)


@dataclass
class InferenceResult:
    """Complete structured output from the AEF-CRC end-to-end inference pipeline."""
    predicted_class: str
    calibrated_confidence: float
    raw_confidence: float
    marginal_prediction_set: List[str]
    mondrian_prediction_set: List[str]
    marginal_set_size: int
    mondrian_set_size: int
    is_singleton: bool
    is_empty: bool
    raw_probabilities: Dict[str, float]
    calibrated_probabilities: Dict[str, float]
    marginal_threshold: float
    calibration_method: str
    review_status: str = ConformalReviewStatus.STANDARD_OUTPUT.value
    review_status_explanation: str = ""
    xai_explanation: Optional[XAIExplanationResult] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    response_bundle: Optional[SingleImageInferenceResponse] = None


class AEFCRCInferenceEngine:
    """
    Production-ready inference engine packaging the frozen AEF-CRC pipeline.
    Chains all stages from raw pixels to conformal prediction sets and XAI explanations.
    """

    def __init__(
        self,
        artifact: Union[FinalPipelineHandoff, ConformalArtifact, Path, str],
        device: str = "auto",
    ) -> None:
        if isinstance(artifact, (str, Path)):
            path = Path(artifact)
            try:
                self.artifact = load_final_pipeline_handoff(path)
            except Exception:
                self.artifact = load_conformal_artifact(path)
        else:
            self.artifact = artifact

        self.device = device
        self.classes = self.artifact.class_order
        self.mask = self.artifact.selected_feature_mask
        self.classifier = self.artifact.final_classifier
        self.calib_method = self.artifact.calibration_method
        self.marginal_q_hat = self.artifact.marginal_q_hat
        self.mondrian_q_hat = self.artifact.mondrian_q_hat
        self._backbone_model = None
        self._shap_explainer = None

    # -------------------------------------------------------------------------
    # Stage 1: Input Validation
    # -------------------------------------------------------------------------
    def validate_input_image(self, image_input: Union[Path, str, Image.Image, np.ndarray]) -> Image.Image:
        """Validates input type, accessibility, dimensions, and visual variance."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Input image file not found: {p}")
            try:
                img = Image.open(p).convert("RGB")
            except Exception as exc:
                raise ValueError(f"Failed to open image file {p}: {exc}") from exc
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                if image_input.shape[2] == 1:
                    img = Image.fromarray(image_input.squeeze(-1)).convert("RGB")
                elif image_input.shape[2] in (3, 4):
                    img = Image.fromarray(image_input[:, :, :3]).convert("RGB")
                else:
                    raise ValueError(f"Unexpected image array channel count: {image_input.shape[2]}")
            else:
                raise ValueError(f"Unexpected image array dimensions: {image_input.ndim}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        w, h = img.size
        if w < 32 or h < 32:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small (minimum 32x32 required)")

        arr = np.asarray(img)
        if np.std(arr) < 1e-4:
            raise ValueError("Degenerate image detected: image has zero or near-zero pixel variance")

        return img

    # -------------------------------------------------------------------------
    # Stage 2: Standardized Preprocessing
    # -------------------------------------------------------------------------
    def preprocess_image(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardized preprocessing respecting config.preprocessing_mode:
        1. Converts PIL image to BGR for ConditionalPreprocessor.
        2. Applies ConditionalPreprocessor (hair removal / CLAHE if triggered, else no-op).
        3. Prepares:
           - TensorFlow/Keras float32 array normalized with EfficientNet preprocess_input.
           - Resized RGB uint8 numpy array for handcrafted feature extraction.
        """
        from config.config import get_config
        from modules.preprocessing import ConditionalPreprocessor

        cfg = get_config()
        preprocessor = ConditionalPreprocessor(cfg)

        # Convert to BGR uint8
        rgb_raw = np.asarray(img, dtype=np.uint8)
        bgr_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)

        # Apply conditional or standard preprocessing
        preprocessed_bgr = preprocessor.process(bgr_raw, "inference_sample").image
        img_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(img_rgb, (cfg.image_size, cfg.image_size))

        # Keras EfficientNet preprocess_input
        from tensorflow.keras.applications.efficientnet import preprocess_input
        arr_float = np.expand_dims(resized_rgb.astype(np.float32), axis=0)
        preprocessed_keras = preprocess_input(arr_float)

        return preprocessed_keras, resized_rgb

    # -------------------------------------------------------------------------
    # Stage 3: Multimodal Feature Extraction (1348-D)
    # -------------------------------------------------------------------------
    def _get_backbone(self):
        if self._backbone_model is None:
            import tensorflow as tf
            from tensorflow.keras.applications import EfficientNetB0

            # Device selection: auto, gpu, cpu
            target_device = "/CPU:0"
            if self.device in ("auto", "gpu", "cuda"):
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    target_device = "/GPU:0"
                elif self.device in ("gpu", "cuda"):
                    print("[WARNING] GPU requested for inference, but no physical GPU detected by TensorFlow. Falling back to /CPU:0.")

            with tf.device(target_device):
                ckpt_path = getattr(self.artifact, "backbone_checkpoint_path", None)
                if ckpt_path and Path(ckpt_path).exists():
                    try:
                        from modules.training import _safe_load_model
                        loaded_model = _safe_load_model(Path(ckpt_path))
                        self._backbone_model = loaded_model
                    except Exception:
                        self._backbone_model = EfficientNetB0(
                            include_top=False,
                            weights="imagenet",
                            input_shape=(224, 224, 3),
                            pooling="avg",
                        )
                else:
                    self._backbone_model = EfficientNetB0(
                        include_top=False,
                        weights="imagenet",
                        input_shape=(224, 224, 3),
                        pooling="avg",
                    )
        return self._backbone_model

    def extract_multimodal_features(self, tensor: np.ndarray, rgb_np: np.ndarray) -> np.ndarray:
        """
        Extracts the 1348-D multimodal representation vector in exact canonical order:
          - 0:1280     EfficientNet-B0 deep features (1280)
          - 1280:1292  GLCM texture features (12)
          - 1292:1310  LBP texture features (18)
          - 1310:1342  HOG-PCA shape features (32)
          - 1342:1348  LAB color statistics (6)
        Total: 1280 + 12 + 18 + 32 + 6 = 1348.
        """
        import tensorflow as tf
        from config.config import get_config
        from modules.handcrafted_features import HandcraftedFeatureExtractor

        # 1. Deep features (1280-D: 0:1280) using TensorFlow/Keras EfficientNet-B0
        model = self._get_backbone()
        target_device = "/GPU:0" if tf.config.list_physical_devices("GPU") and self.device in ("auto", "gpu", "cuda") else "/CPU:0"
        with tf.device(target_device):
            # If model has nested backbone or pooling
            if "efficientnetb0" in [l.name for l in getattr(model, "layers", [])]:
                backbone_layer = model.get_layer("efficientnetb0")
                raw_deep = backbone_layer(tensor, training=False)
            else:
                raw_deep = model(tensor, training=False)

            deep_arr = np.asarray(raw_deep)
            if deep_arr.ndim == 4:
                deep_arr = deep_arr.mean(axis=(1, 2))
            elif deep_arr.ndim == 2 and deep_arr.shape[1] != 1280:
                # If loaded model is full classification model, extract from penultimate layer
                if hasattr(model, "layers") and len(model.layers) >= 2:
                    penult_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
                    raw_deep = penult_model(tensor, training=False)
                    deep_arr = np.asarray(raw_deep)
            deep_feats = deep_arr.squeeze(0).flatten()

        # 2. Handcrafted features from Phase 4 extractor
        cfg = get_config()
        extractor = HandcraftedFeatureExtractor(cfg)
        bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        fv = extractor.extract(bgr)

        glcm_feats = fv.glcm.flatten()
        lbp_feats = fv.lbp.flatten()

        # HOG-PCA (32-D: 1310:1342) - strictly consume fitted reducer from training handoff
        hog_raw = fv.hog.flatten()
        hog_reducer = getattr(self.artifact, "hog_reducer", None)
        if hog_reducer is None:
            raise RuntimeError(
                "Inference artifact is missing 'hog_reducer' (FoldSafeFeatureReducer). "
                "HOG-PCA (32-D) requires the fitted reducer from the training pipeline handoff."
            )
        hog_feats = hog_reducer.transform([hog_raw])[0]

        # LAB color statistics (6-D: 1342:1348)
        lab_feats = fv.color_lab.flatten() if fv.color_lab is not None else np.zeros(6, dtype=np.float32)

        # Assert exact raw branch dimensions
        assert deep_feats.shape[0] == 1280, f"Expected 1280 deep features, got {deep_feats.shape[0]}"
        assert glcm_feats.shape[0] == 12, f"Expected 12 GLCM features, got {glcm_feats.shape[0]}"
        assert lbp_feats.shape[0] == 18, f"Expected 18 LBP features, got {lbp_feats.shape[0]}"
        assert hog_feats.shape[0] == 32, f"Expected 32 HOG-PCA features, got {hog_feats.shape[0]}"
        assert lab_feats.shape[0] == 6, f"Expected 6 LAB features, got {lab_feats.shape[0]}"

        # 3. Apply fitted FeatureNormalizers per branch (from training pipeline handoff)
        normalizers = getattr(self.artifact, "feature_normalizers", None)
        if normalizers is None:
            raise RuntimeError(
                "Inference artifact is missing 'feature_normalizers'. "
                "Feature branches must be normalized using parameters fitted on the training cohort."
            )

        norm_deep = normalizers["deep"].transform([deep_feats])[0] if "deep" in normalizers else deep_feats
        norm_glcm = normalizers["glcm"].transform([glcm_feats])[0] if "glcm" in normalizers else glcm_feats
        norm_lbp = normalizers["lbp"].transform([lbp_feats])[0] if "lbp" in normalizers else lbp_feats
        norm_hog = normalizers["hog"].transform([hog_feats])[0] if "hog" in normalizers else hog_feats
        norm_lab = normalizers["color_lab"].transform([lab_feats])[0] if "color_lab" in normalizers else (
            normalizers["lab"].transform([lab_feats])[0] if "lab" in normalizers else lab_feats
        )

        # Fused vector in canonical concatenation order
        fused = np.concatenate([norm_deep, norm_glcm, norm_lbp, norm_hog, norm_lab])
        assert fused.shape[0] == 1348, f"Expected 1348 fused features, got {fused.shape[0]}"
        return fused

    # -------------------------------------------------------------------------
    # Stage 4: Feature Selection Masking
    # -------------------------------------------------------------------------
    def apply_feature_mask(self, fused_features: np.ndarray) -> np.ndarray:
        """Applies the single frozen Phase 6 BDA production mask to the 1348-D vector."""
        if self.mask is None:
            raise RuntimeError("Inference artifact does not contain a selected_feature_mask.")
        if fused_features.ndim == 1:
            return fused_features[self.mask].reshape(1, -1)
        return fused_features[:, self.mask]

    # -------------------------------------------------------------------------
    # Stage 5: Base Classifier Inference
    # -------------------------------------------------------------------------
    def predict_raw_probabilities(self, masked_features: np.ndarray) -> np.ndarray:
        """Runs predict_proba() using the fitted base Random Forest classifier."""
        if self.classifier is None:
            raise RuntimeError("Inference artifact does not contain a fitted final_classifier.")
        return self.classifier.predict_proba(masked_features)

    # -------------------------------------------------------------------------
    # Stage 6: Probability Calibration
    # -------------------------------------------------------------------------
    def apply_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        """Applies the Phase 8 calibration transform (Primary: Platt; Secondary: Isotonic)."""
        if self.calib_method == "platt" and self.artifact.platt_models is not None:
            fit_obj = type("PlattFit", (), {"models": self.artifact.platt_models})()
            return apply_platt_scaling(raw_probs, fit_obj, self.classes)
        elif self.calib_method == "isotonic" and self.artifact.isotonic_models is not None:
            fit_obj = type("IsoFit", (), {"models": self.artifact.isotonic_models})()
            return apply_isotonic_scaling(raw_probs, fit_obj, self.classes)
        else:
            return clip_probabilities(raw_probs)

    # -------------------------------------------------------------------------
    # Stage 7: Conformal Prediction Set Generation
    # -------------------------------------------------------------------------
    def predict_conformal_sets(self, calib_probs: np.ndarray) -> Tuple[List[str], List[str]]:
        """
        Generates conformal prediction sets:
        1. Primary: Marginal split-conformal prediction set.
        2. Secondary: Class-conditional Mondrian prediction set.
        """
        probs_row = calib_probs[0]

        # Marginal threshold
        threshold = 1.0 - self.marginal_q_hat
        marginal_set = [self.classes[c] for c in range(len(self.classes)) if probs_row[c] >= threshold]

        # Mondrian threshold
        mondrian_set: List[str] = []
        for c, cls_name in enumerate(self.classes):
            q_c = self.mondrian_q_hat.get(cls_name, 1.0) if self.mondrian_q_hat else 1.0
            threshold_c = 1.0 - q_c
            if probs_row[c] >= threshold_c:
                mondrian_set.append(cls_name)

        return marginal_set, mondrian_set

    # -------------------------------------------------------------------------
    # Stage 8 & 9: Full Pipeline Execution with Optional XAI
    # -------------------------------------------------------------------------
    def predict(
        self,
        image_input: Union[Path, str, Image.Image, np.ndarray],
        sample_id: str = "sample_001",
        explain: bool = False,
        xai_output_dir: Optional[Path] = None,
    ) -> InferenceResult:
        """
        Executes the authoritative 9-stage inference contract.
        If explain=True, generates Grad-CAM and TreeSHAP artifacts.
        """
        # 1. Validate
        try:
            img = self.validate_input_image(image_input)
            is_valid = True
        except Exception as exc:
            status, reason = determine_conformal_review_status(False, None)
            resp = SingleImageInferenceResponse(
                sample_id=sample_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                is_valid_input=False,
                review_status=status.value,
                review_status_explanation=f"{reason} (Validation error: {exc})",
            )
            return InferenceResult(
                predicted_class="",
                calibrated_confidence=0.0,
                raw_confidence=0.0,
                marginal_prediction_set=[],
                mondrian_prediction_set=[],
                marginal_set_size=0,
                mondrian_set_size=0,
                is_singleton=False,
                is_empty=True,
                raw_probabilities={},
                calibrated_probabilities={},
                marginal_threshold=0.0,
                calibration_method=self.calib_method,
                review_status=status.value,
                review_status_explanation=reason,
                diagnostics={"error": str(exc)},
                response_bundle=resp,
            )

        # 2. Preprocess
        tensor, rgb_np = self.preprocess_image(img)

        # 3. Extract 1348-D features
        fused_1348 = self.extract_multimodal_features(tensor, rgb_np)

        # 4. Apply frozen BDA mask
        masked_feats = self.apply_feature_mask(fused_1348)

        # 5. Base classifier
        raw_probs = self.predict_raw_probabilities(masked_feats)

        # 6. Probability calibration
        calib_probs = self.apply_calibration(raw_probs)

        # 7. Conformal prediction sets
        marginal_set, mondrian_set = self.predict_conformal_sets(calib_probs)

        # 8. Point prediction and confidence
        pred_idx = int(np.argmax(calib_probs[0]))
        pred_class = self.classes[pred_idx]
        calib_conf = float(calib_probs[0][pred_idx])
        raw_conf = float(raw_probs[0][pred_idx])

        # 9. Conformal review status
        review_status, review_explanation = determine_conformal_review_status(True, marginal_set)

        raw_dict = {cls: float(raw_probs[0][i]) for i, cls in enumerate(self.classes)}
        calib_dict = {cls: float(calib_probs[0][i]) for i, cls in enumerate(self.classes)}

        # 10. Optional Explainable AI (Grad-CAM + TreeSHAP)
        xai_result = None
        if explain:
            xai_result = self._generate_explanations(
                sample_id=sample_id,
                img_rgb=rgb_np,
                tensor=tensor,
                masked_feats=masked_feats,
                target_class=pred_class,
                output_dir=xai_output_dir,
            )

        # Structured response contract
        response_bundle = SingleImageInferenceResponse(
            sample_id=sample_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            is_valid_input=True,
            review_status=review_status.value,
            review_status_explanation=review_explanation,
            predicted_class=pred_class,
            calibrated_confidence=calib_conf,
            raw_confidence=raw_conf,
            raw_probabilities=raw_dict,
            calibrated_probabilities=calib_dict,
            conformal_prediction_set=marginal_set,
            prediction_set_size=len(marginal_set),
            marginal_threshold=float(1.0 - self.marginal_q_hat),
            nominal_coverage=1.0 - self.artifact.alpha,
            alpha=self.artifact.alpha,
            backbone_checkpoint=getattr(self.artifact, "backbone_checkpoint_path", None),
            production_bda_mask_hash=getattr(self.artifact, "production_bda_mask_hash", None),
            classifier_name=self.artifact.classifier_name,
            calibration_method=self.calib_method,
            xai_explanation=xai_result,
            device_metadata={"device_configured": self.device},
            diagnostics={
                "input_dimensions": img.size,
                "fused_dimensions": 1348,
                "selected_dimensions": int(self.mask.sum()),
            },
        )

        if explain and xai_output_dir is not None:
            response_bundle.save_explanation_bundle(xai_output_dir)

        return InferenceResult(
            predicted_class=pred_class,
            calibrated_confidence=calib_conf,
            raw_confidence=raw_conf,
            marginal_prediction_set=marginal_set,
            mondrian_prediction_set=mondrian_set,
            marginal_set_size=len(marginal_set),
            mondrian_set_size=len(mondrian_set),
            is_singleton=len(marginal_set) == 1,
            is_empty=len(marginal_set) == 0,
            raw_probabilities=raw_dict,
            calibrated_probabilities=calib_dict,
            marginal_threshold=float(1.0 - self.marginal_q_hat),
            calibration_method=self.calib_method,
            review_status=review_status.value,
            review_status_explanation=review_explanation,
            xai_explanation=xai_result,
            diagnostics=response_bundle.diagnostics,
            response_bundle=response_bundle,
        )

    def _generate_explanations(
        self,
        sample_id: str,
        img_rgb: np.ndarray,
        tensor: np.ndarray,
        masked_feats: np.ndarray,
        target_class: str,
        output_dir: Optional[Path],
    ) -> XAIExplanationResult:
        """Coordinates Grad-CAM and TreeSHAP generation with exact error isolation."""
        gradcam_layer = "unknown"
        gradcam_shape = []
        heatmap_p = None
        overlay_p = None

        # 1. Grad-CAM execution
        try:
            from modules.xai_gradcam import GradCAMExplainer
            backbone_model = self._get_backbone()
            g_explainer = GradCAMExplainer(backbone_model, self.classes, device=self.device)
            gradcam_layer = g_explainer.target_layer_name
            heatmap, _, _, gradcam_shape = g_explainer.explain(tensor, target_class=target_class)

            if output_dir:
                sample_xai_dir = output_dir / sample_id if (output_dir.name != sample_id) else output_dir
                _, h_path, o_path = g_explainer.save_artifacts(sample_xai_dir, img_rgb, heatmap)
                heatmap_p = str(h_path)
                overlay_p = str(o_path)
        except Exception as exc:
            gradcam_layer = f"Grad-CAM Error: {exc}"

        # 2. TreeSHAP execution
        shap_evs: Dict[str, float] = {}
        top_contribs: List[Dict[str, Any]] = []
        block_imps: List[Dict[str, Any]] = []
        try:
            from modules.xai_shap import RFSHAPExplainer
            if self._shap_explainer is None:
                self._shap_explainer = RFSHAPExplainer(self.classifier, self.mask, self.classes)

            shap_data = self._shap_explainer.explain(masked_feats, target_class=target_class)
            shap_evs = shap_data.per_class_expected_values
            top_contribs = shap_data.feature_rankings[:10]
            block_imps = shap_data.block_importances

            if output_dir:
                sample_xai_dir = output_dir / sample_id if (output_dir.name != sample_id) else output_dir
                self._shap_explainer.save_artifacts(sample_xai_dir, shap_data)
        except Exception as exc:
            shap_evs = {"error": f"SHAP Error: {exc}"}

        return XAIExplanationResult(
            gradcam_target_class=target_class,
            gradcam_target_layer=gradcam_layer,
            gradcam_target_layer_shape=gradcam_shape,
            gradcam_heatmap_path=heatmap_p,
            shap_expected_values=shap_evs,
            shap_output_space="raw",
            shap_perturbation_mode="tree_path_dependent",
            shap_class_mapping=getattr(shap_data, "shap_class_mapping", list(self.classes)) if "shap_data" in locals() else list(self.classes),
            shap_feature_order=getattr(shap_data, "shap_feature_order", []) if "shap_data" in locals() else [],
            top_feature_contributors=top_contribs,
            branch_block_importances=block_imps,
        )

    def predict_single_image(
        self,
        image_input: Union[Path, str, Image.Image, np.ndarray],
        sample_id: str = "sample_001",
        explain: bool = False,
        xai_output_dir: Optional[Path] = None,
    ) -> SingleImageInferenceResponse:
        """Dedicated convenience method returning the complete SingleImageInferenceResponse."""
        result = self.predict(
            image_input=image_input,
            sample_id=sample_id,
            explain=explain,
            xai_output_dir=xai_output_dir,
        )
        return result.response_bundle
