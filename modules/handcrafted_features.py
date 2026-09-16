"""
modules/handcrafted_features.py

AEF-CRC Phase 4: Handcrafted Feature Extraction (GLCM, LBP, HOG).

Research question this module exists to answer (not "can we add
these," per the brief): do texture/shape descriptors carry information
about the four target classes at all, complementary to deep features?
Phase 4 does not decide that question -- it produces the features and
a lightweight evaluation so the question can be answered from evidence.

--- Leakage rule, followed exactly ---
Feature EXTRACTION (GLCM/LBP/HOG values themselves) is stateless per
image -- computing it for every image regardless of split is not a
leak, since no image's descriptor depends on any other image. What
WOULD leak: fitting a StandardScaler (or any normalizer) globally.
FeatureNormalizer below is fit() on a fold's train records only, and
never re-fit or peeked at using val/test -- enforced by taking a
plain list of already-selected training feature vectors, never the
whole dataset.

--- Caching ---
Every descriptor is cached to disk keyed by psd_id, exactly once,
regardless of how many later experiments read it -- re-running the
Phase 4 evaluation script does not recompute GLCM/LBP/HOG.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

from config.config import PSDConfig
from utils.logger import get_module_logger


@dataclass
class FeatureVectors:
    psd_id: str
    glcm: np.ndarray
    lbp: np.ndarray
    hog: np.ndarray
    color_lab: Optional[np.ndarray] = None

    @property
    def lab(self) -> Optional[np.ndarray]:
        """Convenience alias for color_lab."""
        return self.color_lab

    @property
    def combined(self) -> np.ndarray:
        """Deterministic raw feature concatenation in approved contract order:
        GLCM (12) -> LBP (18) -> HOG (1296) -> LAB (6, if enabled).
        """
        parts = [self.glcm, self.lbp, self.hog]
        if self.color_lab is not None:
            parts.append(self.color_lab)
        return np.concatenate(parts)


@dataclass
class ExtractionFailure:
    psd_id: str
    file_path: str
    reason: str


def _no_reduce(vecs: List[np.ndarray], reducer) -> List[np.ndarray]:
    return vecs


def _optional_reduce(vecs: List[np.ndarray], reducer) -> List[np.ndarray]:
    """Used by any branch that MAY have a fold-fitted dimensionality
    reducer passed in (currently only HOG does, via
    modules.fusion.build_fusion_fold's pre-fit step -- see that
    function's docstring for why HOG specifically needs one). If none
    was passed (reducer is None), returns vecs unchanged -- identical
    to _no_reduce in that case."""
    if reducer is not None:
        return list(reducer.transform(vecs))
    return vecs


# Resolves a handcrafted branch name to (1) how to read that branch's
# raw vector off an already-extracted FeatureVectors result, and (2)
# whether/how a fold-fitted reducer applies to it. This is the contract
# modules.fusion._load_branch_vectors resolves branches through instead
# of an `if branch == "glcm": ... if branch == "lbp": ...` chain.
HANDCRAFTED_BRANCH_REGISTRY: Dict[str, tuple] = {
    "glcm": (lambda fv: fv.glcm, _no_reduce),
    "lbp": (lambda fv: fv.lbp, _no_reduce),
    "hog": (lambda fv: fv.hog, _optional_reduce),
    "color_lab": (lambda fv: fv.color_lab, _no_reduce),
    "lab": (lambda fv: fv.color_lab, _no_reduce),
}


# GLCM texture properties -- the standard Haralick-style set, not an
# arbitrarily expanded list (Part: "do not unnecessarily increase
# feature dimensionality").
_GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


class HandcraftedFeatureExtractor:
    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("handcrafted_features", config.aef_crc_phase3_logs_dir, config.log_level)
        self._cache_root = config.aef_crc_phase4_artifacts_dir / config.preprocessing_mode
        subdirs = ["glcm", "lbp", "hog", "combined"]
        if getattr(config, "color_feature_enabled", True):
            subdirs.append("color_lab")
        for sub in subdirs:
            (self._cache_root / sub).mkdir(parents=True, exist_ok=True)
        self._init_or_verify_cache_manifest()

    def _param_hash(self) -> str:
        h = hashlib.sha256()
        params = {
            "preprocessing_mode": self.config.preprocessing_mode,
            "image_size": self.config.image_size,
            "glcm_levels": self.config.glcm_levels,
            "glcm_distances": list(self.config.glcm_distances),
            "glcm_angles": list(self.config.glcm_angles),
            "lbp_radius": self.config.lbp_radius,
            "lbp_n_points": self.config.lbp_n_points,
            "lbp_method": self.config.lbp_method,
            "hog_orientations": self.config.hog_orientations,
            "hog_pixels_per_cell": self.config.hog_pixels_per_cell,
            "hog_cells_per_block": self.config.hog_cells_per_block,
            "color_feature_enabled": getattr(self.config, "color_feature_enabled", True),
        }
        h.update(json.dumps(params, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def _expected_dims(self) -> Dict[str, int]:
        glcm_dim = len(self.config.glcm_distances) * 6
        lbp_dim = self.config.lbp_n_points + 2
        n_cells = self.config.image_size // self.config.hog_pixels_per_cell
        n_blocks = n_cells - self.config.hog_cells_per_block + 1
        hog_dim = n_blocks * n_blocks * (self.config.hog_cells_per_block ** 2) * self.config.hog_orientations
        dims = {"glcm": glcm_dim, "lbp": lbp_dim, "hog": hog_dim}
        if getattr(self.config, "color_feature_enabled", True):
            dims["color_lab"] = 6
            combined_dim = glcm_dim + lbp_dim + hog_dim + 6
        else:
            combined_dim = glcm_dim + lbp_dim + hog_dim
        dims["combined"] = combined_dim
        return dims

    def _init_or_verify_cache_manifest(self) -> None:
        manifest_path = self._cache_root / "cache_manifest.json"
        current_hash = self._param_hash()
        dims = self._expected_dims()

        freeze_path = self.config.aef_crc_reports_dir / "dataset_freeze.json"
        freeze_hash = None
        if freeze_path.exists():
            try:
                freeze_data = json.loads(freeze_path.read_text(encoding="utf-8"))
                freeze_hash = freeze_data.get("freeze_hash") or freeze_data.get("dataset_hash")
            except Exception:
                pass

        params = {
            "preprocessing_mode": self.config.preprocessing_mode,
            "image_size": self.config.image_size,
            "glcm_levels": self.config.glcm_levels,
            "glcm_distances": list(self.config.glcm_distances),
            "glcm_angles": list(self.config.glcm_angles),
            "lbp_radius": self.config.lbp_radius,
            "lbp_n_points": self.config.lbp_n_points,
            "lbp_method": self.config.lbp_method,
            "hog_orientations": self.config.hog_orientations,
            "hog_pixels_per_cell": self.config.hog_pixels_per_cell,
            "hog_cells_per_block": self.config.hog_cells_per_block,
            "color_feature_enabled": getattr(self.config, "color_feature_enabled", True),
            "param_hash": current_hash,
            "feature_dims": dims,
            "dataset_freeze_hash": freeze_hash,
        }

        if manifest_path.exists():
            try:
                cached_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"Corrupt cache manifest at {manifest_path}: {exc}") from exc

            if cached_data.get("param_hash") != current_hash:
                raise ValueError(
                    f"Handcrafted feature cache parameter hash mismatch at {manifest_path}! "
                    f"Expected {current_hash}, got {cached_data.get('param_hash')}. "
                    f"Existing cache is stale for current configuration."
                )
            if cached_data.get("preprocessing_mode") != self.config.preprocessing_mode:
                raise ValueError(
                    f"Handcrafted feature cache preprocessing_mode mismatch at {manifest_path}! "
                    f"Expected {self.config.preprocessing_mode}, got {cached_data.get('preprocessing_mode')}."
                )
            if "feature_dims" in cached_data and cached_data["feature_dims"] != dims:
                raise ValueError(
                    f"Handcrafted feature cache dimension mismatch at {manifest_path}! "
                    f"Expected {dims}, got {cached_data['feature_dims']}."
                )
            if freeze_hash and cached_data.get("dataset_freeze_hash") and cached_data["dataset_freeze_hash"] != freeze_hash:
                raise ValueError(
                    f"Handcrafted feature cache dataset freeze mismatch at {manifest_path}! "
                    f"Expected {freeze_hash}, got {cached_data['dataset_freeze_hash']}."
                )
        else:
            manifest_path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")

    # ---- Public API ----

    def extract(self, image_bgr: np.ndarray) -> FeatureVectors:
        """Stateless, deterministic given the same image and config --
        no reliance on ambient RNG state anywhere in this function.
        Extracts grayscale texture/gradient features (GLCM, LBP, HOG)
        and color features (LAB) directly from the original image.
        """
        color_lab_vec = None
        if getattr(self.config, "color_feature_enabled", True):
            color_lab_vec = self._extract_color_lab(image_bgr)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        glcm_vec = self._extract_glcm(gray)
        lbp_vec = self._extract_lbp(gray)
        hog_vec = self._extract_hog(gray)

        return FeatureVectors(psd_id="", glcm=glcm_vec, lbp=lbp_vec, hog=hog_vec, color_lab=color_lab_vec)

    def extract_and_cache(self, image_bgr: np.ndarray, psd_id: str) -> Optional[FeatureVectors]:
        """Extracts, caches each descriptor to its own .npy file under
        artifacts/phase4/<mode>/<descriptor>/<psd_id>.npy, and returns the
        result. Returns None (does NOT raise) on a corrupt/unreadable
        image -- caller is responsible for logging via
        record_extraction_failure() and continuing, per Part
        'handle corrupt/unreadable images gracefully.'"""
        cached = self._load_cached(psd_id)
        if cached is not None:
            return cached

        result = self.extract(image_bgr)
        result.psd_id = psd_id

        np.save(self._cache_root / "glcm" / f"{psd_id}.npy", result.glcm)
        np.save(self._cache_root / "lbp" / f"{psd_id}.npy", result.lbp)
        np.save(self._cache_root / "hog" / f"{psd_id}.npy", result.hog)
        if result.color_lab is not None:
            (self._cache_root / "color_lab").mkdir(parents=True, exist_ok=True)
            np.save(self._cache_root / "color_lab" / f"{psd_id}.npy", result.color_lab)
        np.save(self._cache_root / "combined" / f"{psd_id}.npy", result.combined)
        return result

    def _load_cached(self, psd_id: str) -> Optional[FeatureVectors]:
        glcm_path = self._cache_root / "glcm" / f"{psd_id}.npy"
        lbp_path = self._cache_root / "lbp" / f"{psd_id}.npy"
        hog_path = self._cache_root / "hog" / f"{psd_id}.npy"
        color_enabled = getattr(self.config, "color_feature_enabled", True)
        color_path = self._cache_root / "color_lab" / f"{psd_id}.npy"

        if glcm_path.exists() and lbp_path.exists() and hog_path.exists():
            if color_enabled and not color_path.exists():
                return None
            glcm = np.load(glcm_path)
            lbp = np.load(lbp_path)
            hog_vec = np.load(hog_path)
            color_vec = np.load(color_path) if (color_enabled and color_path.exists()) else None

            dims = self._expected_dims()
            if len(glcm) != dims["glcm"]:
                raise ValueError(f"Cached GLCM dimension mismatch for {psd_id}: got {len(glcm)}, expected {dims['glcm']}")
            if len(lbp) != dims["lbp"]:
                raise ValueError(f"Cached LBP dimension mismatch for {psd_id}: got {len(lbp)}, expected {dims['lbp']}")
            if len(hog_vec) != dims["hog"]:
                raise ValueError(f"Cached HOG dimension mismatch for {psd_id}: got {len(hog_vec)}, expected {dims['hog']}")
            if color_enabled and color_vec is not None and len(color_vec) != dims.get("color_lab", 6):
                raise ValueError(f"Cached LAB dimension mismatch for {psd_id}: got {len(color_vec)}, expected {dims.get('color_lab', 6)}")

            if np.isnan(glcm).any() or np.isinf(glcm).any():
                raise ValueError(f"NaN/Inf in cached GLCM features for {psd_id}")
            if np.isnan(lbp).any() or np.isinf(lbp).any():
                raise ValueError(f"NaN/Inf in cached LBP features for {psd_id}")
            if np.isnan(hog_vec).any() or np.isinf(hog_vec).any():
                raise ValueError(f"NaN/Inf in cached HOG features for {psd_id}")
            if color_vec is not None and (np.isnan(color_vec).any() or np.isinf(color_vec).any()):
                raise ValueError(f"NaN/Inf in cached LAB features for {psd_id}")

            return FeatureVectors(
                psd_id=psd_id,
                glcm=glcm.astype(np.float32),
                lbp=lbp.astype(np.float32),
                hog=hog_vec.astype(np.float32),
                color_lab=color_vec.astype(np.float32) if color_vec is not None else None,
            )
        return None

    def _extract_color_lab(self, image_bgr: np.ndarray) -> np.ndarray:
        """Extract 6-D LAB color statistics (mean and std of L, a, b channels)
        directly from the original color image before grayscale conversion.
        Order: [L_mean, L_std, a_mean, a_std, b_mean, b_std].
        Deterministic, stateless, parameter-free, no dataset-level fitting.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty or None image passed to _extract_color_lab")
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image for LAB extraction, got shape {image_bgr.shape}")

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float64)
        a = lab[:, :, 1].astype(np.float64)
        b = lab[:, :, 2].astype(np.float64)

        stats = np.array([
            float(np.mean(L)), float(np.std(L)),
            float(np.mean(a)), float(np.std(a)),
            float(np.mean(b)), float(np.std(b)),
        ], dtype=np.float32)

        if np.isnan(stats).any() or np.isinf(stats).any():
            raise ValueError("NaN/Inf computed in LAB color statistics")
        return stats

    # ---- Descriptors ----

    def _extract_glcm(self, gray: np.ndarray) -> np.ndarray:
        # Quantize to config.glcm_levels gray levels -- keeps the
        # co-occurrence matrix a tractable size (levels x levels) rather
        # than the full 256x256, standard GLCM practice.
        quantized = (gray.astype(np.float64) / 256.0 * self.config.glcm_levels).astype(np.uint8)
        quantized = np.clip(quantized, 0, self.config.glcm_levels - 1)

        glcm = graycomatrix(
            quantized, distances=list(self.config.glcm_distances), angles=list(self.config.glcm_angles),
            levels=self.config.glcm_levels, symmetric=True, normed=True,
        )
        # Average over angles (rotation-invariant summary) -- keeps
        # dimensionality at len(distances) * len(props) rather than
        # len(distances) * len(angles) * len(props), which would grow
        # the vector 4x for no clear benefit at this feature-extraction
        # stage (Part: don't unnecessarily inflate dimensionality).
        features = []
        for prop in _GLCM_PROPS:
            vals = graycoprops(glcm, prop)  # shape (n_distances, n_angles)
            features.extend(vals.mean(axis=1))
        return np.array(features, dtype=np.float32)

    def _extract_lbp(self, gray: np.ndarray) -> np.ndarray:
        lbp = local_binary_pattern(gray, P=self.config.lbp_n_points, R=self.config.lbp_radius, method=self.config.lbp_method)
        n_bins = self.config.lbp_n_points + 2  # 'uniform' method: n_points+2 distinct pattern bins
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        return hist.astype(np.float32)

    def _extract_hog(self, gray: np.ndarray) -> np.ndarray:
        # HOG needs a fixed input size to produce a fixed-length vector --
        # resize to config.image_size (same resolution EfficientNet uses,
        # so the two branches see the same effective image content).
        resized = cv2.resize(gray, (self.config.image_size, self.config.image_size))
        features = hog(
            resized, orientations=self.config.hog_orientations,
            pixels_per_cell=(self.config.hog_pixels_per_cell, self.config.hog_pixels_per_cell),
            cells_per_block=(self.config.hog_cells_per_block, self.config.hog_cells_per_block),
            feature_vector=True,
        )
        return features.astype(np.float32)


# ============================================================
# Fold-safe normalization (fit on train fold only)
# ============================================================

@dataclass
class FeatureNormalizer:
    """StandardScaler-equivalent, fit on ONE fold's training feature
    vectors only. transform() is safe to call on val/test after fit()
    -- fit() itself refuses to be called twice, so there is no way to
    accidentally re-fit on a different (e.g. validation-containing)
    set later in the same object's lifetime."""
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    _fitted: bool = False

    def fit(self, train_feature_vectors: List[np.ndarray]) -> "FeatureNormalizer":
        if self._fitted:
            raise RuntimeError("FeatureNormalizer.fit() called twice on the same instance -- refused. "
                                "Create a new FeatureNormalizer per fold instead.")
        if len(train_feature_vectors) == 0:  # works for both a list and a 2D ndarray -- "not x" is
            # ambiguous for a >1-element ndarray and was a real bug caught on this review
            raise ValueError("fit() called with zero training vectors.")
        matrix = np.stack(train_feature_vectors)
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # avoid divide-by-zero for a constant feature dimension
        self._fitted = True
        return self

    def transform(self, feature_vectors: List[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer.transform() called before fit().")
        matrix = np.stack(feature_vectors)
        return (matrix - self.mean_) / self.std_


@dataclass
class FoldSafeFeatureReducer:
    """PCA-based dimensionality control for HOG specifically (item 1,
    Phase-4 code review). Why this exists: even after doubling
    hog_pixels_per_cell (6084-D -> 1296-D, the simplest fix), HOG still
    outweighs GLCM+LBP combined (30-D) by ~43x -- naive concatenation
    would let HOG dominate any downstream classifier purely by
    dimension count, unrelated to its actual information content.

    Deliberately NOT applied by default inside FeatureVectors.combined
    (which stays a raw, unreduced concatenation) -- this is an
    explicit, opt-in step the evaluation harness applies per fold,
    fit on that fold's training vectors only, exactly mirroring
    FeatureNormalizer's fit-once-per-fold, train-only discipline
    (same double-fit refusal, same reasoning)."""
    n_components: int
    _pca: object = None
    _fitted: bool = False

    def fit(self, train_feature_vectors: List[np.ndarray]) -> "FoldSafeFeatureReducer":
        if self._fitted:
            raise RuntimeError("FoldSafeFeatureReducer.fit() called twice on the same instance -- refused. "
                                "Create a new instance per fold instead.")
        if len(train_feature_vectors) == 0:
            raise ValueError("fit() called with zero training vectors.")
        from sklearn.decomposition import PCA
        matrix = np.stack(train_feature_vectors)
        n_components = min(self.n_components, matrix.shape[0], matrix.shape[1])
        self._pca = PCA(n_components=n_components, random_state=0)
        self._pca.fit(matrix)
        self._fitted = True
        return self

    def transform(self, feature_vectors: List[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FoldSafeFeatureReducer.transform() called before fit().")
        matrix = np.stack(feature_vectors)
        return self._pca.transform(matrix)


# ============================================================
# Batch extraction over a list of (image_path, psd_id) with logging
# ============================================================

@dataclass
class ExtractionSummary:
    n_success: int = 0
    n_failed: int = 0
    failures: List[ExtractionFailure] = field(default_factory=list)
    feature_dims: Dict[str, int] = field(default_factory=dict)


def load_and_preprocess_image(config: PSDConfig, file_path, psd_id: str):
    """Shared by extract_batch() below and run_aef_crc_phase4.py's
    evaluation loop -- item 2 (Phase-4 review): GLCM/LBP/HOG must see
    the SAME conditional-preprocessing decision (hair removal/CLAHE)
    that image_loader.py applies before EfficientNet, not a raw image.
    Reuses modules/preprocessing.py's ConditionalPreprocessor directly
    rather than reimplementing any trigger logic here."""
    from modules.preprocessing import ConditionalPreprocessor
    img = cv2.imread(str(file_path))
    if img is None:
        return None
    preprocessor = ConditionalPreprocessor(config)
    return preprocessor.process(img, psd_id).image


def extract_batch(config: PSDConfig, records) -> ExtractionSummary:
    """records: iterable of objects with .file_path and .psd_id
    (modules.aef_input_validator.ImageRecord matches this shape)."""
    extractor = HandcraftedFeatureExtractor(config)
    summary = ExtractionSummary()

    for rec in records:
        img = load_and_preprocess_image(config, rec.file_path, rec.psd_id)
        if img is None:
            summary.n_failed += 1
            summary.failures.append(ExtractionFailure(rec.psd_id, str(rec.file_path), "cv2.imread returned None"))
            continue
        try:
            result = extractor.extract_and_cache(img, rec.psd_id)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any descriptor failure must be
            # logged and skipped, never crash the whole batch (Part: "handle corrupt/unreadable images gracefully")
            summary.n_failed += 1
            summary.failures.append(ExtractionFailure(rec.psd_id, str(rec.file_path), f"{type(exc).__name__}: {exc}"))
            continue

        if np.isnan(result.combined).any() or np.isinf(result.combined).any():
            summary.n_failed += 1
            summary.failures.append(ExtractionFailure(rec.psd_id, str(rec.file_path), "NaN/Inf in extracted features"))
            continue

        summary.n_success += 1
        dims = {"glcm": len(result.glcm), "lbp": len(result.lbp), "hog": len(result.hog)}
        if result.color_lab is not None:
            dims["color_lab"] = len(result.color_lab)
        dims["combined"] = len(result.combined)
        summary.feature_dims = dims

    return summary


def write_extraction_report(config: PSDConfig, summary: ExtractionSummary) -> Path:
    config.aef_crc_phase4_reports_dir.mkdir(parents=True, exist_ok=True)
    path = config.aef_crc_phase4_reports_dir / "extraction_report.json"
    payload = {
        "n_success": summary.n_success,
        "n_failed": summary.n_failed,
        "feature_dims": summary.feature_dims,
        "config": {
            "glcm_distances": config.glcm_distances, "glcm_levels": config.glcm_levels,
            "lbp_radius": config.lbp_radius, "lbp_n_points": config.lbp_n_points, "lbp_method": config.lbp_method,
            "hog_pixels_per_cell": config.hog_pixels_per_cell, "hog_cells_per_block": config.hog_cells_per_block,
            "hog_orientations": config.hog_orientations, "image_size": config.image_size,
            "hog_pca_components": config.hog_pca_components,
            "color_feature_enabled": getattr(config, "color_feature_enabled", True),
        },
        "failures": [{"psd_id": f.psd_id, "file_path": f.file_path, "reason": f.reason} for f in summary.failures],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def get_handcrafted_feature_names(config: PSDConfig, reduced_hog: bool = True) -> List[str]:
    """Returns stable, deterministic feature names in exact concatenation order:
    GLCM (12) -> LBP (18) -> HOG/HOG-PCA (32 or 1296) -> LAB (6, if enabled).
    Total: 68-D when reduced_hog=True and color_feature_enabled=True.
    """
    names = []
    # 1. GLCM (12-D)
    for prop in _GLCM_PROPS:
        for dist in config.glcm_distances:
            names.append(f"glcm_{prop}_d{dist}")

    # 2. LBP (18-D)
    n_bins = config.lbp_n_points + 2
    for i in range(n_bins):
        names.append(f"lbp_bin_{i}")

    # 3. HOG / HOG-PCA (32-D or 1296-D)
    if reduced_hog:
        for i in range(config.hog_pca_components):
            names.append(f"hog_pca_{i}")
    else:
        n_cells = config.image_size // config.hog_pixels_per_cell
        n_blocks = n_cells - config.hog_cells_per_block + 1
        raw_dim = n_blocks * n_blocks * (config.hog_cells_per_block ** 2) * config.hog_orientations
        for i in range(raw_dim):
            names.append(f"hog_raw_{i}")

    # 4. LAB Color (6-D)
    if getattr(config, "color_feature_enabled", True):
        names.extend([
            "lab_L_mean", "lab_L_std",
            "lab_a_mean", "lab_a_std",
            "lab_b_mean", "lab_b_std",
        ])
    return names
