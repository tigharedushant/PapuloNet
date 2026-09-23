"""
modules/losses.py

Loss functions for AEF-CRC / PapuloNet Phase 3 V2:
- Categorical Cross-Entropy (weighted via sample_weights)
- Focal Loss (Lin et al., gamma=2.0)
- Weighted Focal Loss (gamma=2.0, weighted via fold-local sample_weights)

Mathematical formulation:
  FL(p_t) = -(1 - p_t)^gamma * log(p_t)
where p_t is the model's estimated probability for the ground-truth class.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="papulonet_v2")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Categorical Focal Loss for multi-class classification.
    
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Supports sample_weight weighting seamlessly through Keras loss reduction.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        from_logits: bool = False,
        name: str = "categorical_focal_loss",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.gamma = float(gamma)
        self.from_logits = bool(from_logits)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred = tf.cast(y_pred, tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Cross entropy: -sum(y_true * log(y_pred))
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Modulating factor: (1 - y_pred)^gamma
        modulating_factor = tf.math.pow(1.0 - y_pred, self.gamma)

        focal_loss = modulating_factor * cross_entropy
        # Return per-sample loss vector: shape (batch_size,)
        return tf.reduce_sum(focal_loss, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "from_logits": self.from_logits,
        })
        return config


def get_loss(loss_name: str, gamma: float = 2.0, from_logits: bool = False) -> tf.keras.losses.Loss:
    """Factory returning the appropriate compiled loss function.
    
    Supported:
    - 'categorical_crossentropy' or 'ce': tf.keras.losses.CategoricalCrossentropy
    - 'categorical_focal_loss' or 'focal': CategoricalFocalLoss(gamma=gamma)
    """
    clean_name = loss_name.lower().strip()
    if clean_name in ("categorical_crossentropy", "ce", "weighted_ce"):
        return tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits, name="categorical_crossentropy")
    elif clean_name in ("categorical_focal_loss", "focal", "weighted_focal"):
        return CategoricalFocalLoss(gamma=gamma, from_logits=from_logits, name="categorical_focal_loss")
    else:
        raise ValueError(
            f"Unsupported loss name '{loss_name}'. Expected 'categorical_crossentropy' or 'categorical_focal_loss'."
        )
