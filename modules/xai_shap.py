"""
modules/xai_shap.py

Authoritative TreeSHAP implementation for the AEF-CRC framework.

SCIENTIFIC CONTRACT:
- Explains the actual production Random Forest classifier.
- Operates on the exact BDA-selected K-dimensional feature vector (K = 194 of 1316 in production A7).
- Uses TreeSHAP with feature_perturbation="tree_path_dependent".
  Note: 'tree_path_dependent' traverses tree decision paths using leaf sample counts
  to compute conditional expectations along the tree structure; it does NOT learn
  or estimate the true joint feature distribution across correlated features.
- Explains RAW Random Forest predictions BEFORE Platt calibration:
  For scikit-learn's RandomForestClassifier, model_output="raw" is in probability space,
  representing the uncalibrated ensemble class vote proportions (summing to 1.0 across classes).
  It does NOT represent margin votes or log-odds.
- Explicitly distinct from:
  1) Platt-calibrated probabilities (per-class sigmoid logistic regression fitting log-loss on val_calib; ECE is an evaluation metric).
  2) Conformal prediction sets (finite-sample 1-alpha marginal/mondrian coverage on val_conf).
  TreeSHAP decomposes the raw RF ensemble vote proportions, NOT the calibrated probabilities
  or conformal set boundaries.
- Persists per-class baseline expected values (shap_expected_values).
- Persists shap_output_space ("raw"), SHAP class mapping, and K-dimensional feature ordering.
- Maps all selected features back to global indices (1316-D in production A7), branch families, and readable names.
- Production A7 layout: deep (1280), glcm (12), lbp (18), color_lab (6). Total = 1316. No HOG block.
- Legacy 1348-D layout (A6 with 32 HOG features) is isolated as an explicit backward-compatibility path.

"SHAP quantifies feature contributions to the Random Forest's raw prediction."
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from modules.output_schema import SHAPBlockContribution, SHAPFeatureContribution


class SHAPError(RuntimeError):
    """Raised when SHAP explanation fails or encounters invalid dimensions."""
    pass


A7_BRANCH_DIMS: Dict[str, int] = {
    "deep": 1280,
    "glcm": 12,
    "lbp": 18,
    "color_lab": 6,
}
A7_FULL_DIM: int = 1316

A6_BRANCH_DIMS: Dict[str, int] = {
    "deep": 1280,
    "glcm": 12,
    "lbp": 18,
    "hog": 32,
    "color_lab": 6,
}
A6_FULL_DIM: int = 1348


def get_canonical_feature_names(layout: str = "A7") -> List[Tuple[str, str]]:
    """
    Returns the canonical feature definitions as (branch, feature_name).
    By default (layout="A7"), strictly adheres to the production A7 1316-D schema:
      - 0:1280      EfficientNet-B0 deep features (1280)
      - 1280:1292  GLCM Texture properties (12)
      - 1292:1310  LBP Texture histogram bins (18)
      - 1310:1316  LAB Color Statistics (6)
      Total = 1316. No HOG features.

    If layout="A6" (legacy compatibility mode), adheres to the 1348-D schema:
      - 0:1280      EfficientNet-B0 (1280)
      - 1280:1292  GLCM (12)
      - 1292:1310  LBP (18)
      - 1310:1342  HOG-PCA (32)
      - 1342:1348  LAB (6)
      Total = 1348.
    """
    layout_upper = str(layout).upper()
    is_legacy_a6 = layout_upper == "A6" or layout in ("1348", 1348)
    names: List[Tuple[str, str]] = []

    # 1. EfficientNet-B0 deep features (1280)
    for i in range(1280):
        names.append(("deep", f"deep_feat_{i:04d}"))

    # 2. GLCM texture properties across distances 1 and 2 (12)
    glcm_props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    for dist in (1, 2):
        for prop in glcm_props:
            names.append(("glcm", f"glcm_d{dist}_{prop}"))

    # 3. LBP uniform histogram bins (18)
    for bin_idx in range(18):
        names.append(("lbp", f"lbp_bin_{bin_idx:02d}"))

    # 4. HOG-PCA components (32) -- ONLY present in legacy A6 layout
    if is_legacy_a6:
        for comp in range(32):
            names.append(("hog", f"hog_pca_comp_{comp:02d}"))

    # 5. LAB color statistics (6)
    for stat in ("L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std"):
        names.append(("color_lab", f"color_lab_{stat}"))

    expected = A6_FULL_DIM if is_legacy_a6 else A7_FULL_DIM
    assert len(names) == expected, f"Feature registry expected {expected} dimensions, got {len(names)}"
    return names


def validate_production_shap_contract(
    classifier: Any,
    selected_feature_mask: np.ndarray,
    expected_full_dim: int = A7_FULL_DIM,
    expected_k_features: Optional[int] = None,
    branch_dims: Optional[Dict[str, int]] = None,
) -> None:
    """
    Strict validation of the Phase 10 / SHAP production contract:
      - Full feature dimension must equal expected_full_dim (default 1316 for A7).
      - BDA mask must be 1-D boolean array of length expected_full_dim.
      - Number of selected active features must equal expected_k_features when provided.
        If expected_k_features is None the count is derived from the mask itself (no strict check).
        V2 pipeline selects 642 features; V1 legacy selected 194 features.
      - Classifier n_features_in_ must match expected_k_features.
      - No HOG block in production branch dimensions.
    """
    mask = np.asarray(selected_feature_mask, dtype=bool)
    if mask.ndim != 1:
        raise SHAPError(f"BDA mask must be 1-D, got shape {mask.shape}")
    if mask.shape[0] != expected_full_dim:
        raise SHAPError(
            f"Production BDA mask dimension mismatch: expected {expected_full_dim}, got {mask.shape[0]}"
        )
    k = int(np.sum(mask))
    # Resolve effective k: use provided value for strict enforcement, or derive from mask
    effective_k = expected_k_features if expected_k_features is not None else k
    if expected_k_features is not None and k != expected_k_features:
        raise SHAPError(
            f"Production selected features mismatch: expected {expected_k_features}, got {k}"
        )
    n_in = getattr(classifier, "n_features_in_", None)
    if n_in is not None and n_in != effective_k:
        raise SHAPError(
            f"Production classifier expects {n_in} features, but mask selects {effective_k}"
        )
    if branch_dims is not None:
        if "hog" in branch_dims and branch_dims["hog"] > 0:
            raise SHAPError("Production A7 layout strictly forbids HOG features.")
        b_sum = sum(branch_dims.values())
        if b_sum != expected_full_dim:
            raise SHAPError(f"branch_dims sum ({b_sum}) != expected full dim ({expected_full_dim})")


@dataclass
class SHAPExplanationResult:
    """Structured container holding all SHAP attribution artifacts for a sample."""
    target_class_idx: int
    target_class_name: str
    expected_values: Dict[str, float]               # Per-class baseline expected value
    raw_probabilities: Dict[str, float]             # Raw uncalibrated RF probabilities
    feature_contributions: List[SHAPFeatureContribution]
    block_contributions: List[SHAPBlockContribution]
    shap_output_space: str = "raw"                  # Strictly frozen raw Random Forest prediction space
    shap_perturbation_mode: str = "tree_path_dependent"  # Strictly frozen tree path-dependent perturbation
    shap_class_mapping: List[str] = field(default_factory=list)  # Class ordering used by the SHAP explainer
    shap_feature_order: List[str] = field(default_factory=list)  # K-dim ordered feature names corresponding to SHAP values
    reconstruction_difference: Optional[float] = None  # |base_value + sum(shap) - raw_target| in the explainer output space
    explainer_mode: str = "tree_path_dependent"
    classifier_name: str = "RandomForestClassifier"

    @property
    def per_class_expected_values(self) -> Dict[str, float]:
        return self.expected_values

    @property
    def feature_rankings(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.feature_contributions]

    @property
    def block_importances(self) -> List[Dict[str, Any]]:
        return [b.to_dict() for b in self.block_contributions]

    def export_csvs(self, feature_csv_path: Path, block_csv_path: Path) -> None:
        """Exports ranked feature-level and branch-level importance CSVs."""
        feature_csv_path.parent.mkdir(parents=True, exist_ok=True)
        block_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Feature-level CSV
        with open(feature_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "selected_index", "global_index", "branch",
                "feature_name", "feature_value", "shap_value", "abs_shap_value",
            ])
            for rank, feat in enumerate(self.feature_contributions, 1):
                writer.writerow([
                    rank,
                    feat.selected_index,
                    feat.global_index,
                    feat.branch,
                    feat.feature_name,
                    f"{feat.feature_value:.6f}",
                    f"{feat.shap_value:.6f}",
                    f"{feat.abs_shap_value:.6f}",
                ])

        # 2. Block-level CSV
        with open(block_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "branch", "feature_count", "sum_abs_shap", "mean_abs_shap", "pct_total_importance",
            ])
            for blk in self.block_contributions:
                writer.writerow([
                    blk.branch,
                    blk.feature_count,
                    f"{blk.sum_abs_shap:.6f}",
                    f"{blk.mean_abs_shap:.6f}",
                    f"{blk.pct_total_importance:.2f}%",
                ])


class ProductionTreeSHAPExplainer:
    """
    Computes TreeSHAP feature attributions on the production Random Forest.
    Strictly conforms to pre-calibration raw prediction attribution.
    """

    def __init__(
        self,
        classifier: Any,
        selected_feature_mask: np.ndarray,
        classes: List[str],
        branch_dims: Optional[Dict[str, int]] = None,
        expected_full_dim: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        classifier : Any
            Fitted production Random Forest classifier.
        selected_feature_mask : np.ndarray
            Boolean mask of shape (1316,) from Phase 6 BDA (or 1348 for legacy A6).
        classes : List[str]
            Canonical disease class names (length 4).
        branch_dims : Optional[Dict[str, int]]
            Feature branch dimensions (e.g. {'deep': 1280, 'glcm': 12, 'lbp': 18, 'color_lab': 6}).
        expected_full_dim : Optional[int]
            Expected full fused dimension (default: 1316 for production A7).
        """
        import shap

        self.classifier = classifier
        self.mask = np.asarray(selected_feature_mask, dtype=bool)
        if self.mask.ndim != 1:
            raise SHAPError(f"BDA mask must be 1-dimensional, got {self.mask.ndim}-D array of shape {self.mask.shape}")

        mask_len = self.mask.shape[0]

        if expected_full_dim is not None:
            if mask_len != expected_full_dim:
                raise SHAPError(f"BDA mask dimension mismatch: expected {expected_full_dim}, got {mask_len}")
            self.full_dim = expected_full_dim
        elif branch_dims is not None:
            sum_dim = sum(branch_dims.values())
            if mask_len != sum_dim:
                raise SHAPError(f"BDA mask dimension ({mask_len}) does not match branch_dims sum ({sum_dim})")
            self.full_dim = sum_dim
        elif mask_len == A7_FULL_DIM:
            # Authoritative production A7
            self.full_dim = A7_FULL_DIM
        elif mask_len == A6_FULL_DIM:
            # Legacy A6 compatibility
            self.full_dim = A6_FULL_DIM
        else:
            raise SHAPError(
                f"BDA mask dimension mismatch: expected production A7 dimension {A7_FULL_DIM} "
                f"(or legacy {A6_FULL_DIM}), got {mask_len}."
            )

        self.layout = "A7" if self.full_dim == A7_FULL_DIM else "A6"
        self.branch_dims = branch_dims or (A7_BRANCH_DIMS if self.layout == "A7" else A6_BRANCH_DIMS)

        # Enforce production contract: if layout is A7, verify no HOG
        if self.layout == "A7":
            if "hog" in self.branch_dims and self.branch_dims["hog"] > 0:
                raise SHAPError("Production A7 layout strictly forbids HOG features.")

        self.classes = list(classes)
        self.k_features = int(np.sum(self.mask))
        if self.k_features == 0:
            raise SHAPError("BDA mask has 0 features selected; cannot explain classifier.")

        self.global_indices = np.where(self.mask)[0]

        # Verify classifier input dimension matches selected features
        n_features_in = getattr(classifier, "n_features_in_", None)
        if n_features_in is not None and n_features_in != self.k_features:
            raise SHAPError(
                f"Classifier expects {n_features_in} features, but BDA mask selected {self.k_features}."
            )

        # Build feature registry mapping
        self.canonical_names = get_canonical_feature_names(layout=self.layout)
        self.selected_feature_names = [self.canonical_names[idx][1] for idx in self.global_indices]

        # Initialize TreeExplainer with frozen path-dependent perturbation and raw prediction output
        try:
            self.explainer = shap.TreeExplainer(
                self.classifier,
                model_output="raw",
                feature_perturbation="tree_path_dependent",
            )
        except TypeError as exc:
            raise SHAPError(
                f"SHAP TreeExplainer initialization failed: requested frozen configuration "
                f"(model_output='raw', feature_perturbation='tree_path_dependent') is unsupported "
                f"by the installed SHAP environment: {exc}. Silently switching modes is strictly forbidden."
            ) from exc
        except Exception as exc:
            raise SHAPError(
                f"Failed to initialize TreeExplainer with model_output='raw' and "
                f"feature_perturbation='tree_path_dependent': {exc}. "
                "Incompatible SHAP environment detected; failing loudly."
            ) from exc

        # Strictly validate that explainer accepted and configured model_output == 'raw'
        configured_output = getattr(self.explainer, "model_output", None)
        if configured_output is not None and str(configured_output).lower() != "raw":
            raise SHAPError(
                f"TreeExplainer output convention mismatch: expected 'raw', got '{configured_output}'. "
                "Silent fallback to alternative output spaces (probability, log_loss, margin) is prohibited."
            )
        self.output_space = "raw"
        self.perturbation_mode = "tree_path_dependent"
        active_output = getattr(self.explainer, "model_output", None)
        if active_output is not None and active_output != "raw":
            raise SHAPError(
                f"SHAP configuration error: expected model_output='raw', but explainer resolved to "
                f"'{active_output}'. Silently switching to a non-raw output space is forbidden."
            )
        active_perturbation = getattr(self.explainer, "feature_perturbation", None)
        if active_perturbation is not None and active_perturbation != "tree_path_dependent":
            raise SHAPError(
                f"SHAP configuration error: expected feature_perturbation='tree_path_dependent', "
                f"but explainer resolved to '{active_perturbation}'. Silently switching perturbation mode is forbidden."
            )

        self.shap_output_space = "raw"
        self.shap_perturbation_mode = "tree_path_dependent"

        # Extract baseline expected values per class
        raw_ev = self.explainer.expected_value
        if isinstance(raw_ev, (list, np.ndarray)):
            self.expected_values = [float(v) for v in raw_ev]
        else:
            self.expected_values = [float(raw_ev)]

    def explain(
        self,
        features: np.ndarray,
        raw_probabilities: Optional[np.ndarray] = None,
        target_class: Optional[Union[str, int]] = None,
    ) -> SHAPExplanationResult:
        """
        Computes SHAP attributions for the specified target class.

        Parameters
        ----------
        features : np.ndarray
            Input feature vector. Accepts either:
            - Full fused vector (1316-D for production A7, or 1348-D for legacy A6).
              Masking to the active K features is performed internally using self.mask.
            - Already-masked K-dimensional vector (K = 194 in production A7).
              Used directly without re-masking (masked_input = arr.reshape(1, -1)).
        raw_probabilities : Optional[np.ndarray]
            Raw uncalibrated classifier prediction probabilities. If None, derived via predict_proba.
        target_class : Optional[Union[str, int]]
            Target disease class name or index to explain.
        """
        arr = np.asarray(features).flatten()

        # 1. Format into exact K-dimensional input
        if arr.shape[0] == self.full_dim:
            masked_input = arr[self.mask].reshape(1, -1)
        elif arr.shape[0] == self.k_features:
            masked_input = arr.reshape(1, -1)
        else:
            raise SHAPError(
                f"Input features dimension ({arr.shape[0]}) matches neither full dimension "
                f"{self.full_dim} nor K={self.k_features}."
            )

        # 2. Derive or validate raw probabilities
        if raw_probabilities is None:
            raw_probs_mat = self.classifier.predict_proba(masked_input)
            raw_probs_flat = raw_probs_mat[0]
        else:
            raw_probs_flat = np.asarray(raw_probabilities).flatten()

        if raw_probs_flat.shape[0] != len(self.classes):
            raise SHAPError(
                f"Raw probabilities dimension ({raw_probs_flat.shape[0]}) mismatch with classes ({len(self.classes)})."
            )

        # 3. Resolve target class index
        if target_class is None:
            target_idx = int(np.argmax(raw_probs_flat))
        elif isinstance(target_class, str):
            if target_class not in self.classes:
                raise ValueError(f"Unknown target class '{target_class}'. Options: {self.classes}")
            target_idx = self.classes.index(target_class)
        else:
            target_idx = int(target_class)

        # 4. Compute TreeSHAP values
        raw_shap = self.explainer.shap_values(masked_input)

        if isinstance(raw_shap, list):
            class_shap = np.asarray(raw_shap[target_idx]).flatten()
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 3:
                class_shap = raw_shap[0, :, target_idx]
            elif raw_shap.ndim == 2:
                class_shap = raw_shap[0]
            else:
                raise SHAPError(f"Unexpected SHAP values ndarray shape: {raw_shap.shape}")
        else:
            raise SHAPError(f"Unexpected SHAP values type: {type(raw_shap)}")

        assert class_shap.shape[0] == self.k_features, (
            f"SHAP output length ({class_shap.shape[0]}) does not match K selected features ({self.k_features})."
        )

        # 5. Reconstruction difference check (records discrepancy without silently assuming invariant identity)
        base_val = self.expected_values[target_idx] if target_idx < len(self.expected_values) else 0.0
        sum_shap = float(np.sum(class_shap))
        reconstructed_val = base_val + sum_shap
        actual_raw_target = float(raw_probs_flat[target_idx])
        reconstruction_diff = float(abs(reconstructed_val - actual_raw_target))

        # 6. Build individual feature contributions
        feature_contributions: List[SHAPFeatureContribution] = []
        for sel_idx in range(self.k_features):
            glob_idx = int(self.global_indices[sel_idx])
            branch, feat_name = self.canonical_names[glob_idx]
            feat_val = float(masked_input[0, sel_idx])
            s_val = float(class_shap[sel_idx])
            abs_s_val = abs(s_val)

            feature_contributions.append(
                SHAPFeatureContribution(
                    selected_index=sel_idx,
                    global_index=glob_idx,
                    branch=branch,
                    feature_name=feat_name,
                    feature_value=feat_val,
                    shap_value=s_val,
                    abs_shap_value=abs_s_val,
                )
            )

        # Rank features descending by absolute attribution magnitude
        feature_contributions.sort(key=lambda x: x.abs_shap_value, reverse=True)

        # 7. Compute branch-level aggregate contributions
        branches = list(self.branch_dims.keys())
        branch_sums: Dict[str, float] = {b: 0.0 for b in branches}
        branch_counts: Dict[str, int] = {b: 0 for b in branches}

        for feat in feature_contributions:
            if feat.branch in branch_sums:
                branch_sums[feat.branch] += feat.abs_shap_value
                branch_counts[feat.branch] += 1

        total_abs = sum(branch_sums.values())
        block_contributions: List[SHAPBlockContribution] = []
        for b in branches:
            cnt = branch_counts[b]
            b_sum = branch_sums[b]
            b_mean = (b_sum / cnt) if cnt > 0 else 0.0
            pct = (100.0 * b_sum / total_abs) if total_abs > 1e-12 else 0.0
            block_contributions.append(
                SHAPBlockContribution(
                    branch=b,
                    feature_count=cnt,
                    sum_abs_shap=b_sum,
                    mean_abs_shap=b_mean,
                    pct_total_importance=pct,
                )
            )

        # Rank blocks descending by sum of absolute SHAP values
        block_contributions.sort(key=lambda x: x.sum_abs_shap, reverse=True)

        ev_dict = {
            cls_name: self.expected_values[c]
            for c, cls_name in enumerate(self.classes)
            if c < len(self.expected_values)
        }
        raw_prob_dict = {cls_name: float(raw_probs_flat[c]) for c, cls_name in enumerate(self.classes)}

        return SHAPExplanationResult(
            target_class_idx=target_idx,
            target_class_name=self.classes[target_idx],
            expected_values=ev_dict,
            raw_probabilities=raw_prob_dict,
            feature_contributions=feature_contributions,
            block_contributions=block_contributions,
            shap_output_space="raw",
            shap_perturbation_mode="tree_path_dependent",
            shap_class_mapping=list(self.classes),
            shap_feature_order=list(self.selected_feature_names),
            reconstruction_difference=reconstruction_diff,
            explainer_mode="tree_path_dependent",
            classifier_name=getattr(self.classifier, "__class__", type("")).__name__,
        )

    def save_artifacts(self, output_dir: Path, shap_result: SHAPExplanationResult) -> Tuple[Path, Path]:
        """Saves shap_feature_importance.csv and shap_block_importance.csv to output_dir."""
        feat_path = output_dir / "shap_feature_importance.csv"
        block_path = output_dir / "shap_block_importance.csv"
        shap_result.export_csvs(feat_path, block_path)
        return feat_path, block_path


# Backward-compatible alias matching inference.py import
RFSHAPExplainer = ProductionTreeSHAPExplainer
