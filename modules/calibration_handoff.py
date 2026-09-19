"""
modules/calibration_handoff.py

AEF-CRC: the interfaces between Phase 7 -> Phase 8 -> Phase 9.

Deliberately minimal -- these are interface/artifact definitions, not
implementations. Calibration (Platt scaling as primary, Isotonic as
secondary) and conformal prediction (marginal primary, Mondrian secondary)
are implemented in their respective phases.

CalibrationHandoff (Phase 7 -> Phase 8) fields are either:
  - identity (representation_id, experiment_id, classifier_name,
    feature_selection_method, random_seed, source_experiment,
    feature_arm, class_order)
  - the final fitted model (final_classifier) + selected_feature_mask +
    branch_dims
  - the held-out calibration set (calibration_psd_ids,
    calibration_true_labels, calibration_predicted_labels,
    calibration_raw_probabilities)
  - provenance tracking (run_id, dataset_freeze_hash, fold_plan_hash)
  - fitted transformation state (hog_reducer, feature_normalizers,
    backbone_checkpoint_path)

ConformalHandoff (Phase 8 -> Phase 9) carries the same identity +
model/feature fields forward unchanged, plus the CALIBRATED
probabilities (frozen primary: Platt scaling; secondary comparator:
Isotonic regression) and fitted transforms (platt_models, isotonic_models)
(class-conditional) conformal prediction specifically needs per-class
nonconformity thresholds, which is why calibration_true_labels stays
alongside calibration_calibrated_probabilities here rather than only a
summary statistic.
Persistence uses joblib (a direct dependency -- see requirements.txt),
since a fitted sklearn/XGBoost classifier and numpy arrays are exactly
what joblib is for; hand-rolling a JSON+.npy split for either artifact
would be more code for no real benefit at this size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CalibrationHandoff:
    # --- Experiment identity (verifiable against modules.experiment_config) ---
    representation_id: str
    experiment_id: str
    classifier_name: str
    feature_selection_method: str
    random_seed: int
    source_experiment: str   # Phase-3 experiment name whose deep-feature cache this used
    feature_arm: str         # key into modules.fusion.ARMS
    class_order: List[str]
    # --- The final fitted model, and what's needed to feed it new data ---
    final_classifier: Any                  # fitted, predict_proba-capable classifier (Task 2's final retrain)
    selected_feature_mask: np.ndarray      # boolean mask into the fused vector (1348-D) -- apply before calling final_classifier
    branch_dims: Dict[str, int]            # maps mask indices back to branches (deep: 1280, glcm: 12, lbp: 18, hog: 32, color_lab: 6)

    # --- The calibration set: splitter.py's held-out val/, NOT a CV fold ---
    calibration_psd_ids: List[str]
    calibration_true_labels: List[str]
    calibration_predicted_labels: List[str]
    calibration_raw_probabilities: np.ndarray   # (n_calibration, n_classes), columns in class_order order

    # --- Provenance & Freeze tracking ---
    run_id: str = ""
    dataset_freeze_hash: Optional[str] = None
    fold_plan_hash: Optional[str] = None

    # --- Phase 8/9 Stratified 50/50 Partition (123 calib / 123 conf) ---
    val_calib_psd_ids: List[str] = field(default_factory=list)
    val_calib_true_labels: List[str] = field(default_factory=list)
    val_calib_raw_probabilities: Optional[np.ndarray] = None
    val_conf_psd_ids: List[str] = field(default_factory=list)
    val_conf_true_labels: List[str] = field(default_factory=list)
    val_conf_raw_probabilities: Optional[np.ndarray] = None

    # --- Fitted Transformation State (persisted for exact inference reproduction) ---
    hog_reducer: Optional[Any] = None           # FoldSafeFeatureReducer fit on final_train HOG
    feature_normalizers: Optional[Dict[str, Any]] = None  # FeatureNormalizer per branch fit on final_train
    backbone_checkpoint_path: Optional[str] = None  # Path to trained Keras backbone checkpoint

    @property
    def branch_normalizers(self) -> Optional[Dict[str, Any]]:
        return self.feature_normalizers

    note: str = (
        "calibration_raw_probabilities are RAW classifier predict_proba() output, "
        "NOT calibrated confidence. Calibration (Platt scaling, isotonic regression, "
        "or otherwise) is the responsibility of Phase 8 and has NOT been applied "
        "to any value in this artifact."
    )


def validate_bda_mask_compatibility(
    mask: np.ndarray,
    classifier: Any,
    expected_dim: Optional[int] = None,
) -> None:
    """
    Strict validation of the persisted production BDA mask:
      - mask must be 1-D boolean array
      - mask length must equal expected_dim (or in (1316, 1348) if expected_dim is None)
      - selected count must be > 0 and <= mask.shape[0]
      - if classifier exposes n_features_in_, mask.sum() must equal classifier.n_features_in_
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"BDA mask must be a numpy ndarray, got {type(mask)}")
    if mask.ndim != 1:
        raise ValueError(f"BDA mask must be 1-dimensional, got {mask.ndim}-D array of shape {mask.shape}")
    if expected_dim is not None:
        if mask.shape[0] != expected_dim:
            raise ValueError(
                f"BDA mask dimension mismatch: expected fused feature dimension {expected_dim}, "
                f"got {mask.shape[0]}"
            )
    else:
        if mask.shape[0] not in (1316, 1348):
            raise ValueError(
                f"BDA mask dimension mismatch: expected fused feature dimension in (1316, 1348), "
                f"got {mask.shape[0]}"
            )
    if mask.dtype != bool:
        raise TypeError(f"BDA mask must have boolean dtype, got {mask.dtype}")
    selected_count = int(mask.sum())
    if selected_count == 0:
        raise ValueError("BDA mask has 0 features selected; cannot feed classifier.")

    # Check compatibility with fitted classifier if attributes exist
    if hasattr(classifier, "n_features_in_"):
        clf_features = getattr(classifier, "n_features_in_")
        if selected_count != clf_features:
            raise ValueError(
                f"BDA mask / classifier incompatibility: mask selects {selected_count} features, "
                f"but classifier was trained on {clf_features} features (n_features_in_)."
            )


def get_active_run_id(config: Any) -> Optional[str]:
    """Retrieves the authoritative run_id established by Phase 3 from reports/phase3/winner.json."""
    phase3_dir = getattr(config, "aef_crc_phase3_reports_dir", None)
    if phase3_dir is not None:
        p = phase3_dir / "winner.json"
        if p.exists():
            try:
                import json
                data = json.loads(p.read_text(encoding="utf-8"))
                return data.get("run_id")
            except Exception:
                pass
    return None


def validate_calibration_handoff_provenance(
    handoff: CalibrationHandoff,
    config: Any,
    expected_run_id: Optional[str] = None,
) -> None:
    """Strictly validates cross-run provenance to prevent mixing artifacts across runs."""
    from modules.experiment_config import representation_id
    expected_repr = representation_id(config)
    if handoff.representation_id != expected_repr:
        raise RuntimeError(
            f"Cross-run artifact provenance mismatch: CalibrationHandoff representation_id "
            f"'{handoff.representation_id}' != expected '{expected_repr}'. "
            "Artifacts from different configurations cannot be mixed."
        )
    if handoff.random_seed != config.random_seed:
        raise RuntimeError(
            f"Cross-run artifact provenance mismatch: CalibrationHandoff random_seed "
            f"{handoff.random_seed} != expected {config.random_seed}."
        )
    if expected_run_id is None:
        expected_run_id = get_active_run_id(config)
    if expected_run_id:
        artifact_run_id = getattr(handoff, "run_id", None)
        if artifact_run_id != expected_run_id:
            raise RuntimeError(
                f"Cross-run artifact provenance mismatch: CalibrationHandoff run_id='{artifact_run_id}' "
                f"!= active experiment run_id='{expected_run_id}'. Artifacts from separate runs cannot be combined!"
            )


def validate_conformal_handoff_provenance(
    handoff: ConformalHandoff,
    config: Any,
    expected_run_id: Optional[str] = None,
) -> None:
    """Strictly validates cross-run provenance for Phase 9 conformal calibration."""
    from modules.experiment_config import representation_id
    expected_repr = representation_id(config)
    if handoff.representation_id != expected_repr:
        raise RuntimeError(
            f"Cross-run artifact provenance mismatch: ConformalHandoff representation_id "
            f"'{handoff.representation_id}' != expected '{expected_repr}'. "
            "Artifacts from different configurations cannot be mixed."
        )
    if handoff.random_seed != config.random_seed:
        raise RuntimeError(
            f"Cross-run artifact provenance mismatch: ConformalHandoff random_seed "
            f"{handoff.random_seed} != expected {config.random_seed}."
        )
    if expected_run_id is None:
        expected_run_id = get_active_run_id(config)
    if expected_run_id:
        artifact_run_id = getattr(handoff, "run_id", None)
        if artifact_run_id != expected_run_id:
            raise RuntimeError(
                f"Cross-run artifact provenance mismatch: ConformalHandoff run_id='{artifact_run_id}' "
                f"!= active experiment run_id='{expected_run_id}'. Artifacts from separate runs cannot be combined!"
            )


def validate_pipeline_handoff_provenance(
    handoff: ConformalArtifact,
    config: Any,
    expected_run_id: Optional[str] = None,
) -> None:
    """Strictly validates cross-run provenance for deployment/inference pipeline."""
    from modules.experiment_config import representation_id
    expected_repr = representation_id(config)
    if handoff.representation_id != expected_repr:
        raise RuntimeError(
            f"Cross-run artifact provenance mismatch: FinalPipelineHandoff representation_id "
            f"'{handoff.representation_id}' != expected '{expected_repr}'. "
            "Artifacts from different configurations cannot be mixed."
        )
    if handoff.random_seed != config.random_seed:
        raise RuntimeError(
            f"Cross-run artifact provenance mismatch: FinalPipelineHandoff random_seed "
            f"{handoff.random_seed} != expected {config.random_seed}."
        )
    if expected_run_id is None:
        expected_run_id = get_active_run_id(config)
    if expected_run_id:
        artifact_run_id = getattr(handoff, "run_id", None)
        if artifact_run_id != expected_run_id:
            raise RuntimeError(
                f"Cross-run artifact provenance mismatch: FinalPipelineHandoff run_id='{artifact_run_id}' "
                f"!= active experiment run_id='{expected_run_id}'. Artifacts from separate runs cannot be combined!"
            )


@dataclass
class ConformalHandoff:
    """Everything Phase 9 (conformal prediction) needs, without rebuilding
    Phase 7 or Phase 8."""
    # --- Identity, inherited unchanged from the Phase-7/8 chain ---
    representation_id: str
    experiment_id: str
    classifier_name: str
    feature_selection_method: str
    random_seed: int
    class_order: List[str]

    # --- The final fitted model + feature contract (same as CalibrationHandoff) ---
    final_classifier: Any
    selected_feature_mask: np.ndarray
    branch_dims: Dict[str, int]

    # --- Provenance & Freeze tracking ---
    run_id: str = ""
    dataset_freeze_hash: Optional[str] = None
    fold_plan_hash: Optional[str] = None

    # --- Calibration outcome: frozen primary method (Platt / Sigmoid) ---
    calibration_method: str = "platt"  # Frozen primary: "platt"; Secondary comparator: "isotonic"
    calibration_method_reason: str = "Frozen protocol: Sigmoid/Platt scaling is the pre-registered primary method"
    platt_models: Optional[Dict[str, Any]] = None     # fitted LogisticRegression per class
    isotonic_models: Optional[Dict[str, Any]] = None  # fitted IsotonicRegression per class (comparator)

    # --- The LOCKED calibration set: calibrated probabilities + labels ---
    calibration_psd_ids: List[str] = field(default_factory=list)
    calibration_true_labels: List[str] = field(default_factory=list)
    calibration_calibrated_probabilities: Optional[np.ndarray] = None

    # --- Phase 9 Conformal Calibration Partition (N_conf = 123) ---
    val_conf_psd_ids: List[str] = field(default_factory=list)
    val_conf_true_labels: List[str] = field(default_factory=list)
    val_conf_calibrated_probabilities: Optional[np.ndarray] = None
    val_calib_metrics: Optional[Dict[str, float]] = None
    alpha: float = 0.10

    # --- Fitted Transformation State (persisted for exact inference reproduction) ---
    hog_reducer: Optional[Any] = None
    feature_normalizers: Optional[Dict[str, Any]] = None
    backbone_checkpoint_path: Optional[str] = None

    @property
    def branch_normalizers(self) -> Optional[Dict[str, Any]]:
        return self.feature_normalizers

    note: str = (
        "val_conf_calibrated_probabilities have ALREADY had the primary Platt calibration "
        "method applied on the separate 123-image probability calibration set (val_calib). "
        "Phase 9 strictly uses these primary Platt-calibrated probabilities for conformal nonconformity scoring."
    )


@dataclass
class ConformalArtifact:
    """The final Phase 9 deployed conformal prediction artifact."""
    representation_id: str
    experiment_id: str
    classifier_name: str
    feature_selection_method: str
    random_seed: int
    class_order: List[str]
    final_classifier: Any
    selected_feature_mask: np.ndarray
    branch_dims: Dict[str, int]
    calibration_method: str
    marginal_q_hat: float
    mondrian_q_hat: Dict[str, float]
    run_id: str = ""
    dataset_freeze_hash: Optional[str] = None
    fold_plan_hash: Optional[str] = None
    platt_models: Optional[Dict[str, Any]] = None
    isotonic_models: Optional[Dict[str, Any]] = None
    alpha: float = 0.10
    n_conf_samples: int = 123
    per_class_conf_counts: Dict[str, int] = field(default_factory=dict)
    small_sample_warnings: Dict[str, str] = field(default_factory=dict)
    hog_reducer: Optional[Any] = None
    feature_normalizers: Optional[Dict[str, Any]] = None
    backbone_checkpoint_path: Optional[str] = None

    @property
    def branch_normalizers(self) -> Optional[Dict[str, Any]]:
        return self.feature_normalizers
    exchangeability_disclaimer: str = (
        "Exchangeability between conformal calibration samples and future test samples "
        "is a foundational theoretical assumption, not an empirical guarantee. "
        "Finite-sample validity holds under exchangeability; class-conditional small-sample "
        "variance applies to minority classes."
    )


@dataclass
class FinalPipelineHandoff(ConformalArtifact):
    """Alias / full container for the complete end-to-end frozen pipeline handoff."""
    pass


def _save_artifact(artifact: object, path: Path) -> Path:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def _load_artifact(path: Path, expected_type: type, produced_by: str) -> object:
    import joblib
    if not path.exists():
        raise FileNotFoundError(f"No {expected_type.__name__} artifact at {path} -- run {produced_by} first.")
    artifact = joblib.load(path)
    if not isinstance(artifact, expected_type):
        raise TypeError(f"{path} does not contain a {expected_type.__name__} (got {type(artifact)})")
    return artifact


def save_calibration_handoff(handoff: CalibrationHandoff, path: Path) -> Path:
    return _save_artifact(handoff, path)


def load_calibration_handoff(path: Path) -> CalibrationHandoff:
    return _load_artifact(path, CalibrationHandoff, "run_aef_crc_phase7.py")


def save_conformal_handoff(handoff: ConformalHandoff, path: Path) -> Path:
    return _save_artifact(handoff, path)


def load_conformal_handoff(path: Path) -> ConformalHandoff:
    return _load_artifact(path, ConformalHandoff, "run_aef_crc_phase8.py")


def save_conformal_artifact(artifact: ConformalArtifact, path: Path) -> Path:
    return _save_artifact(artifact, path)


def load_conformal_artifact(path: Path) -> ConformalArtifact:
    return _load_artifact(path, ConformalArtifact, "run_aef_crc_phase9.py")


def save_final_pipeline_handoff(handoff: FinalPipelineHandoff, path: Path) -> Path:
    return _save_artifact(handoff, path)


def load_final_pipeline_handoff(path: Path) -> FinalPipelineHandoff:
    return _load_artifact(path, FinalPipelineHandoff, "run_aef_crc_phase9.py")
