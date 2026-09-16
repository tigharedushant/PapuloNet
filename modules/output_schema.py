"""
modules/output_schema.py

Authoritative output schemas, data transfer objects, and clinical review status definitions
for the AEF-CRC end-to-end inference, Explainable AI (XAI), and locked outer-test evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ReviewStatus(str, Enum):
    """
    Authoritative clinical review status derived from split-conformal prediction sets.
    
    Rules:
    - STANDARD_OUTPUT: Exactly one class satisfies conformal inclusion criterion (singleton set).
      NOTE: Does not automatically guarantee high probability or medical certainty; indicates
      unambiguous conformal set membership at the pre-specified confidence level (1 - alpha = 0.90).
    - SPECIALIST_REVIEW_REQUIRED: Multiple candidate classes satisfy conformal criterion (>1 class).
      Represents clinical differential ambiguity requiring dermatologist consultation.
    - NO_CLASS_MEETS_CONFORMAL_THRESHOLD: Zero classes satisfy conformal criterion (empty set).
      Represents high model nonconformity/uncertainty.
      CRITICAL: An empty set is NOT an out-of-distribution (OOD) detector and must NEVER be
      labeled as OUT_OF_DISTRIBUTION_FLAG.
    - INPUT_QUALITY_FAILURE: Input image failed validation checks (corruption, degenerate variance,
      or dimensions < 32x32). No disease prediction is fabricated.
    """
    STANDARD_OUTPUT = "STANDARD_OUTPUT"
    SPECIALIST_REVIEW_REQUIRED = "SPECIALIST_REVIEW_REQUIRED"
    NO_CLASS_MEETS_CONFORMAL_THRESHOLD = "NO_CLASS_MEETS_CONFORMAL_THRESHOLD"
    INPUT_QUALITY_FAILURE = "INPUT_QUALITY_FAILURE"


# Backward-compatible alias
ConformalReviewStatus = ReviewStatus


def determine_conformal_review_status(
    is_valid_input: bool,
    marginal_set: Optional[List[str]],
) -> Tuple[ReviewStatus, str]:
    """
    Maps conformal prediction set membership to pre-registered clinical triage status.
    
    Returns
    -------
    Tuple[ReviewStatus, str]
        (Status Enum, Human-readable descriptive explanation).
    """
    if not is_valid_input:
        return (
            ReviewStatus.INPUT_QUALITY_FAILURE,
            "Input image failed quality validation checks (corrupt, degenerate variance, or dimensions < 32x32). "
            "No disease prediction is fabricated for invalid inputs.",
        )

    if marginal_set is None or len(marginal_set) == 0:
        return (
            ReviewStatus.NO_CLASS_MEETS_CONFORMAL_THRESHOLD,
            "No disease class satisfied the conformal inclusion threshold at 90% nominal confidence. "
            "Indicates high prediction nonconformity (NOT an out-of-distribution detector).",
        )
    elif len(marginal_set) == 1:
        return (
            ReviewStatus.STANDARD_OUTPUT,
            f"Singleton prediction set containing candidate class '{marginal_set[0]}' at 90% confidence.",
        )
    else:
        return (
            ReviewStatus.SPECIALIST_REVIEW_REQUIRED,
            f"Prediction set contains {len(marginal_set)} candidate classes: {marginal_set}. "
            "Differential diagnostic ambiguity requires specialist dermatologist evaluation.",
        )


@dataclass
class InputValidationResult:
    """Outcome of input image validation prior to feature extraction."""
    is_valid: bool
    status: ReviewStatus
    error_message: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    pixel_variance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class SHAPFeatureContribution:
    """Individual feature attribution from TreeSHAP on the production Random Forest."""
    selected_index: int       # Index in the K-dimensional BDA-masked feature vector
    global_index: int         # Index in the canonical 1348-D fused vector
    branch: str               # Branch family: 'deep', 'glcm', 'lbp', 'hog', 'color_lab'
    feature_name: str         # Readable semantic name (e.g. 'glcm_contrast_d1_0deg')
    feature_value: float      # Actual standardized numeric value passed to the RF
    shap_value: float         # Attribution in Random Forest's uncalibrated raw output space before Platt calibration
    abs_shap_value: float     # Absolute magnitude of attribution

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SHAPBlockContribution:
    """Branch-level aggregate attribution summarizing feature family contributions."""
    branch: str
    feature_count: int
    sum_abs_shap: float
    mean_abs_shap: float
    pct_total_importance: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class XAIExplanationResult:
    """
    Complete Explainable AI output bundle containing spatial CNN activations
    and tabular Random Forest feature attributions.
    """
    gradcam_target_class: str
    gradcam_target_layer: str
    gradcam_target_layer_shape: List[int] = field(default_factory=list)
    gradcam_heatmap_path: Optional[str] = None
    shap_expected_values: Dict[str, float] = field(default_factory=dict)
    shap_output_space: str = "raw"  # Strictly frozen raw Random Forest prediction space
    shap_perturbation_mode: str = "tree_path_dependent"  # Strictly frozen tree path-dependent perturbation
    shap_class_mapping: List[str] = field(default_factory=list)  # Explicit class ordering used by SHAP explainer
    shap_feature_order: List[str] = field(default_factory=list)  # K-dim ordered feature names corresponding to SHAP values
    shap_reconstruction_formula: str = (
        "expected_value[target_class] + sum(shap_values) == raw_target[target_class] "
        "(strictly pre-Platt-calibration raw output space)"
    )
    top_feature_contributors: List[Dict[str, Any]] = field(default_factory=list)
    branch_block_importances: List[Dict[str, Any]] = field(default_factory=list)
    scientific_disclaimer: str = (
        "Grad-CAM provides a spatial explanation of the EfficientNet disease-classification "
        "representation, while SHAP explains feature contributions to the downstream Random Forest "
        "prediction. SHAP explains the Random Forest raw prediction before Platt calibration; it does "
        "not explain calibrated probabilities, Platt scaling, or conformal set membership."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backward-compatible alias
XAIExplanationBundle = XAIExplanationResult


@dataclass
class SingleImageInferenceResponse:
    """
    Structured end-to-end inference prediction, calibration, and conformal output
    for a single unseen image.
    """
    sample_id: str
    timestamp_utc: str
    is_valid_input: bool
    review_status: str
    review_status_explanation: str
    predicted_class: str = ""
    calibrated_confidence: float = 0.0
    raw_confidence: float = 0.0
    raw_probabilities: Dict[str, float] = field(default_factory=dict)
    calibrated_probabilities: Dict[str, float] = field(default_factory=dict)
    conformal_prediction_set: List[str] = field(default_factory=list)
    prediction_set_size: int = 0
    marginal_threshold: float = 0.0
    nominal_coverage: float = 0.90
    alpha: float = 0.10
    backbone_checkpoint: Optional[str] = None
    production_bda_mask_hash: Optional[str] = None
    classifier_name: str = "RandomForestClassifier"
    calibration_method: str = "platt"
    xai_explanation: Optional[XAIExplanationResult] = None
    device_metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_explanation_bundle(self, output_dir: Path) -> Path:
        """Persists the structured explanation.json and associated metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "explanation.json"
        self.save_json(json_path)
        return json_path


# Backward-compatible alias
SingleImagePrediction = SingleImageInferenceResponse


@dataclass
class TestEvaluationReport:
    """Authoritative metric structure for the locked outer test set (N=243)."""
    dataset_name: str
    total_test_samples: int
    evaluation_timestamp: str
    classification_metrics: Dict[str, Any]
    calibration_metrics: Dict[str, Any]
    conformal_metrics: Dict[str, Any]
    leakage_guard_verified: bool
    device_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
