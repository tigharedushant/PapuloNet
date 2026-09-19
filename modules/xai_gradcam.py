"""
modules/xai_gradcam.py

Authoritative Grad-CAM implementation for the AEF-CRC framework.

SCIENTIFIC CONTRACT:
- Operates exclusively on the TensorFlow/Keras EfficientNet-B0 backbone.
- Target: THE PRE-SOFTMAX LOGIT of the Phase 3 disease-specific Dense-4 classification head.
- Does NOT explain the Random Forest classifier.
- Does NOT backpropagate through Platt calibration or conformal prediction.
- Does NOT claim to explain the complete multimodal AEF-CRC decision.
- Strictly validates that the target convolutional layer produces a 4D feature map preceding pooling.
- Completely free of PyTorch dependencies.

"Grad-CAM provides a spatial explanation of the EfficientNet disease-classification
representation, while SHAP explains feature contributions to the downstream Random Forest prediction."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


class GradCAMError(RuntimeError):
    """Raised when Grad-CAM generation encounters an unrecoverable structural issue."""
    pass


@dataclass
class GradCAMResult:
    """Artifacts and metadata produced by GradCAMExplainer."""
    target_class_idx: int
    target_class_name: str
    target_layer_name: str
    pre_softmax_logit: float
    raw_heatmap: np.ndarray          # 2D float32 [0.0, 1.0] at conv resolution (e.g. 7x7)
    resized_heatmap: np.ndarray      # 2D float32 [0.0, 1.0] at original image resolution
    overlay_rgb: np.ndarray          # 3D uint8 (H, W, 3) RGB overlay on original image
    model_name: str
    target_layer_shape: List[int]

    def __iter__(self):
        """Allows unpacking: heatmap, raw_heatmap, overlay_rgb, target_layer_shape."""
        yield self.resized_heatmap
        yield self.raw_heatmap
        yield self.overlay_rgb
        yield self.target_layer_shape


class GradCAMExplainer:
    """
    Computes class-discriminative spatial activation maps (Grad-CAM)
    using the pre-softmax logits of the Phase 3 Keras EfficientNet-B0 classification head.
    """

    def __init__(
        self,
        model: Any,
        classes: List[str],
        target_layer_name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        """
        Parameters
        ----------
        model : tf.keras.Model
            Loaded Phase 3 EfficientNet-B0 model containing the Dense-4 disease head.
        classes : List[str]
            Canonical list of disease class names (length 4).
        target_layer_name : Optional[str]
            Name of target convolutional layer. If None, dynamically resolved with 4D validation.
        device : str
            Device specification ("auto", "gpu", "cpu").
        """
        import tensorflow as tf

        self.model = model
        self.classes = list(classes)
        self.device = device

        # 1. Locate Phase 3 disease-specific Dense classification head
        self.dense_layer = self._find_dense_layer(model)
        dense_weights = self.dense_layer.get_weights()
        if len(dense_weights) < 2:
            raise GradCAMError("Phase 3 Dense head must contain kernel and bias weights.")
        self.dense_kernel = tf.constant(dense_weights[0], dtype=tf.float32)
        self.dense_bias = tf.constant(dense_weights[1], dtype=tf.float32)

        # 2. Discover and validate target 4D convolutional layer preceding pooling
        self.target_layer = self._find_target_conv_layer(model, target_layer_name)
        self.target_layer_name = self.target_layer.name
        self.target_layer_shape = [int(s) if s is not None else -1 for s in self.target_layer.output.shape]

        # 3. Construct convolutional submodel mapping inputs to the 4D feature map
        self.conv_submodel = self._build_conv_submodel(model, self.target_layer)

    def _find_dense_layer(self, model: Any) -> Any:
        import tensorflow as tf

        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.units == len(self.classes):
                return layer
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Dense) and sub.units == len(self.classes):
                        return sub
        raise GradCAMError(
            f"Could not locate Phase 3 Dense head with {len(self.classes)} units in model {model.name}."
        )

    def _is_valid_4d_feature_map(self, layer: Any) -> bool:
        """
        Validates that the layer produces an actual 4D convolutional feature map (batch, H, W, C)
        strictly preceding global pooling (H > 1, W > 1).
        """
        try:
            shape = getattr(layer, "output_shape", None) or getattr(layer.output, "shape", None)
            if shape is not None and len(shape) == 4:
                # Shape: (batch, H, W, C). Spatial dimensions must exceed 1x1 (e.g. 7x7)
                h, w = shape[1], shape[2]
                if (h is None or h > 1) and (w is None or w > 1):
                    return True
        except Exception:
            pass
        return False

    def _find_target_conv_layer(self, model: Any, target_name: Optional[str]) -> Any:
        import tensorflow as tf

        # If explicit layer name requested, validate that it is an actual 4D feature map
        if target_name:
            cand = None
            try:
                cand = model.get_layer(target_name)
            except (ValueError, AttributeError):
                for layer in model.layers:
                    if hasattr(layer, "get_layer"):
                        try:
                            cand = layer.get_layer(target_name)
                            break
                        except (ValueError, AttributeError):
                            pass
            if cand is not None:
                if self._is_valid_4d_feature_map(cand):
                    return cand
                raise GradCAMError(
                    f"Requested layer '{target_name}' is not a 4D feature map preceding pooling "
                    f"(shape: {getattr(cand, 'output_shape', None)})."
                )
            raise GradCAMError(f"Requested target layer '{target_name}' not found in model.")

        # Preferred named layers in EfficientNet-B0
        candidates = ["top_conv", "top_activation", "block7a_project_conv"]
        for cand_name in candidates:
            cand = None
            try:
                cand = model.get_layer(cand_name)
            except (ValueError, AttributeError):
                for layer in model.layers:
                    if hasattr(layer, "get_layer"):
                        try:
                            cand = layer.get_layer(cand_name)
                            break
                        except (ValueError, AttributeError):
                            pass
            if cand is not None and self._is_valid_4d_feature_map(cand):
                return cand

        # Dynamic validated discovery: search backwards for the last layer producing a 4D feature map
        for layer in reversed(model.layers):
            if self._is_valid_4d_feature_map(layer):
                return layer
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if self._is_valid_4d_feature_map(sub):
                        return sub

        raise GradCAMError("Could not dynamically identify a valid 4D convolutional feature layer preceding pooling.")

    def _build_conv_submodel(self, model: Any, target_layer: Any) -> Any:
        import tensorflow as tf

        try:
            return tf.keras.Model(inputs=model.input, outputs=target_layer.output)
        except Exception:
            for layer in model.layers:
                if hasattr(layer, "layers") and target_layer in layer.layers:
                    return tf.keras.Model(inputs=layer.input, outputs=target_layer.output)
            raise GradCAMError(f"Failed to bind convolutional submodel for layer {target_layer.name}.")

    def explain(
        self,
        preprocessed_tensor: np.ndarray,
        original_image_rgb: Optional[np.ndarray] = None,
        target_class: Optional[Union[str, int]] = None,
    ) -> GradCAMResult:
        """
        Generates Grad-CAM heatmap and overlay targeting the true pre-softmax logit.

        Parameters
        ----------
        preprocessed_tensor : np.ndarray
            4D tensor (1, 224, 224, 3) normalized via Keras EfficientNet preprocess_input.
        original_image_rgb : Optional[np.ndarray]
            Original uint8 RGB image (H, W, 3). If None, defaults to 224x224 canvas.
        target_class : Optional[Union[str, int]]
            Target disease class name or index. If None, uses argmax of CNN head logits.
        """
        import tensorflow as tf

        # Resolve target device
        target_dev = "/CPU:0"
        if self.device in ("auto", "gpu", "cuda") and tf.config.list_physical_devices("GPU"):
            target_dev = "/GPU:0"

        x_tensor = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)

        with tf.device(target_dev):
            with tf.GradientTape() as tape:
                # 1. Forward pass to target 4D conv layer (preceding pooling)
                conv_output = self.conv_submodel(x_tensor, training=False)
                tape.watch(conv_output)

                # 2. Global Average Pooling (matching Phase 3 pooling='avg')
                pooled_features = tf.reduce_mean(conv_output, axis=[1, 2])
                pooled_features_f32 = tf.cast(pooled_features, tf.float32)
                dense_kernel_f32 = tf.cast(self.dense_kernel, tf.float32)
                dense_bias_f32 = tf.cast(self.dense_bias, tf.float32)

                # 3. Linear projection to pre-softmax logit (EXACT Phase 3 Dense weights, STRICTLY NO SOFTMAX)
                logits = tf.matmul(pooled_features_f32, dense_kernel_f32) + dense_bias_f32

                # 4. Resolve target class index
                if target_class is None:
                    target_idx = int(tf.argmax(logits[0]).numpy())
                elif isinstance(target_class, str):
                    if target_class not in self.classes:
                        raise ValueError(f"Unknown target class '{target_class}'. Options: {self.classes}")
                    target_idx = self.classes.index(target_class)
                else:
                    target_idx = int(target_class)

                # 5. Extract pre-softmax logit
                class_logit = logits[0, target_idx]

            # 6. Compute exact gradients d(logit) / d(conv_output)
            grads = tape.gradient(class_logit, conv_output)
            if grads is None:
                raise GradCAMError("Gradient calculation returned None; gradient tape was broken.")

            # 7. Channel-wise importance weighting (Global Average Pooling of gradients)
            weights = tf.cast(tf.reduce_mean(grads, axis=[1, 2], keepdims=True), tf.float32)
            conv_f32 = tf.cast(conv_output, tf.float32)

            # 8. Weighted linear combination of feature maps
            cam = tf.reduce_sum(weights * conv_f32, axis=-1)[0]

            # 9. Apply ReLU: keep features with positive influence on target logit
            cam = tf.maximum(cam, 0.0)

            # 10. Min-Max Normalization to [0.0, 1.0]
            cam_min = tf.reduce_min(cam)
            cam_max = tf.reduce_max(cam)
            denom = cam_max - cam_min
            if denom > 1e-8:
                norm_cam = (cam - cam_min) / denom
            else:
                norm_cam = tf.zeros_like(cam)

            raw_cam_np = norm_cam.numpy().astype(np.float32)
            logit_val = float(class_logit.numpy())

        # 11. Resize heatmap to original image spatial dimensions
        if original_image_rgb is not None:
            orig_h, orig_w = original_image_rgb.shape[:2]
        else:
            orig_h, orig_w = 224, 224
            original_image_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        resized_cam = cv2.resize(raw_cam_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        resized_cam = np.clip(resized_cam, 0.0, 1.0)

        # 12. Create color overlay on original RGB image
        heatmap_uint8 = np.uint8(255 * resized_cam)
        colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend: 60% original image + 40% colored heatmap
        overlay_rgb = cv2.addWeighted(original_image_rgb, 0.6, colored_rgb, 0.4, 0)

        return GradCAMResult(
            target_class_idx=target_idx,
            target_class_name=self.classes[target_idx],
            target_layer_name=self.target_layer_name,
            pre_softmax_logit=logit_val,
            raw_heatmap=raw_cam_np,
            resized_heatmap=resized_cam,
            overlay_rgb=overlay_rgb,
            model_name=getattr(self.model, "name", "EfficientNet-B0"),
            target_layer_shape=self.target_layer_shape,
        )

    def save_artifacts(
        self,
        output_dir: Path,
        original_image_rgb: np.ndarray,
        resized_heatmap: np.ndarray,
    ) -> Tuple[Path, Path, Path]:
        """
        Saves original.png, gradcam_heatmap.png, and gradcam_overlay.png to output_dir.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        orig_p = output_dir / "original.png"
        heat_p = output_dir / "gradcam_heatmap.png"
        over_p = output_dir / "gradcam_overlay.png"

        # 1. Original RGB image
        Image.fromarray(original_image_rgb).save(orig_p)

        # 2. Normalized heatmap (grayscale PNG)
        heat_uint8 = np.uint8(255 * np.clip(resized_heatmap, 0.0, 1.0))
        Image.fromarray(heat_uint8).save(heat_p)

        # 3. Blended color overlay
        colored_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.addWeighted(original_image_rgb, 0.6, colored_rgb, 0.4, 0)
        Image.fromarray(overlay_rgb).save(over_p)

        return orig_p, heat_p, over_p
