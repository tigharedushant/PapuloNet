"""
modules/fusion.py

AEF-CRC Phase 5: Evidence-based fusion of EfficientNet-B0 deep features
with GLCM/LBP/HOG handcrafted features.

Reuses, does not reimplement:
- modules.efficientnet_model.load_deep_feature (deep feature cache read)
- modules.handcrafted_features.HandcraftedFeatureExtractor /
  FeatureNormalizer / FoldSafeFeatureReducer / load_and_preprocess_image
- modules.fold_loader.Fold (fold assignments -- no second splitter here)
- fold.class_weights (already fold-local, from
  modules.aef_input_validator.compute_train_fold_class_weights -- the
  formula is not recomputed here, only converted to a per-sample array
  for classifiers that want sample_weight rather than a class_weight dict)

Six arms (ARMS below), identical classifier and identical evaluation
code in every arm -- only the branch tuple changes. Arm "EfficientNet"
uses the SAME downstream classifier as every other arm (never Phase 3's
own softmax head) so the feature-set effect is isolated from the
classifier effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from config.config import PSDConfig
from modules.fold_loader import Fold
from modules.efficientnet_model import load_deep_feature
from modules.handcrafted_features import (
    HandcraftedFeatureExtractor, FeatureNormalizer, FoldSafeFeatureReducer, load_and_preprocess_image,
    HANDCRAFTED_BRANCH_REGISTRY,
)


CANONICAL_BRANCH_ORDER: Tuple[str, ...] = ("deep", "glcm", "lbp", "hog", "color_lab")
BRANCH_ORDER: Tuple[str, ...] = CANONICAL_BRANCH_ORDER
BRANCH_ALIASES: Dict[str, str] = {
    "lab": "color_lab",
}

EQUIVALENCE_MARGIN: float = 0.005


@dataclass(frozen=True)
class FusionArm:
    arm_id: str
    name: str
    representation_id: str
    branches: Tuple[str, ...]
    expected_dim: int
    purpose: str


FUSION_ARMS: List[FusionArm] = [
    FusionArm(
        arm_id="A0",
        name="EfficientNet",
        representation_id="efficientnet",
        branches=("deep",),
        expected_dim=1280,
        purpose="Deep backbone baseline",
    ),
    FusionArm(
        arm_id="A1",
        name="EfficientNet+GLCM",
        representation_id="efficientnet_glcm",
        branches=("deep", "glcm"),
        expected_dim=1292,
        purpose="Complementary macro-texture ablation",
    ),
    FusionArm(
        arm_id="A2",
        name="EfficientNet+LBP",
        representation_id="efficientnet_lbp",
        branches=("deep", "lbp"),
        expected_dim=1298,
        purpose="Complementary micro-texture ablation",
    ),
    FusionArm(
        arm_id="A3",
        name="EfficientNet+HOG-PCA",
        representation_id="efficientnet_hog",
        branches=("deep", "hog"),
        expected_dim=1312,
        purpose="Complementary gradient shape ablation",
    ),
    FusionArm(
        arm_id="A4",
        name="EfficientNet+LAB",
        representation_id="efficientnet_lab",
        branches=("deep", "color_lab"),
        expected_dim=1286,
        purpose="Complementary color distribution ablation",
    ),
    FusionArm(
        arm_id="A5",
        name="EfficientNet+GLCM+LBP",
        representation_id="efficientnet_glcm_lbp",
        branches=("deep", "glcm", "lbp"),
        expected_dim=1310,
        purpose="Macro + micro texture combination",
    ),
    FusionArm(
        arm_id="A6",
        name="EfficientNet+GLCM+LBP+HOG-PCA+LAB",
        representation_id="efficientnet_glcm_lbp_hog_lab",
        branches=("deep", "glcm", "lbp", "hog", "color_lab"),
        expected_dim=1348,
        purpose="Full handcrafted fusion",
    ),
    FusionArm(
        arm_id="A7",
        name="EfficientNet+GLCM+LBP+LAB",
        representation_id="efficientnet_glcm_lbp_lab",
        branches=("deep", "glcm", "lbp", "color_lab"),
        expected_dim=1316,
        purpose="Compact interpretable descriptors without HOG",
    ),
]

FUSION_ARMS_BY_ID: Dict[str, FusionArm] = {arm.arm_id: arm for arm in FUSION_ARMS}
FUSION_ARMS_BY_NAME: Dict[str, FusionArm] = {arm.name: arm for arm in FUSION_ARMS}
FUSION_ARMS_BY_REPR_ID: Dict[str, FusionArm] = {arm.representation_id: arm for arm in FUSION_ARMS}

CONTROLLED_ARMS: Dict[str, Tuple[str, ...]] = {arm.name: arm.branches for arm in FUSION_ARMS}

ARM_REPRESENTATION_IDS: Dict[str, str] = {
    "EfficientNet": "efficientnet",
    "EfficientNet+GLCM": "efficientnet_glcm",
    "EfficientNet+LBP": "efficientnet_lbp",
    "EfficientNet+HOG": "efficientnet_hog",
    "EfficientNet+HOG-PCA": "efficientnet_hog",
    "EfficientNet+LAB": "efficientnet_lab",
    "EfficientNet+GLCM+LBP": "efficientnet_glcm_lbp",
    "EfficientNet+GLCM+LBP+LAB": "efficientnet_glcm_lbp_lab",
    "EfficientNet+GLCM+LBP+HOG+LAB": "efficientnet_glcm_lbp_hog_lab",
    "EfficientNet+GLCM+LBP+HOG-PCA+LAB": "efficientnet_glcm_lbp_hog_lab",
    "EfficientNet+Handcrafted": "efficientnet_glcm_lbp_hog_lab",
    "EfficientNet+Full_Handcrafted": "efficientnet_glcm_lbp_hog_lab",
}

EXPECTED_ARM_DIMS: Dict[str, int] = {
    "EfficientNet": 1280,
    "EfficientNet+GLCM": 1292,
    "EfficientNet+LBP": 1298,
    "EfficientNet+HOG": 1312,
    "EfficientNet+HOG-PCA": 1312,
    "EfficientNet+LAB": 1286,
    "EfficientNet+GLCM+LBP": 1310,
    "EfficientNet+GLCM+LBP+LAB": 1316,
    "EfficientNet+GLCM+LBP+HOG+LAB": 1348,
    "EfficientNet+GLCM+LBP+HOG-PCA+LAB": 1348,
    "EfficientNet+Handcrafted": 1348,
    "EfficientNet+Full_Handcrafted": 1348,
    # Backward compatibility for Phase 7/8 intermediate arm:
    "EfficientNet+GLCM+LBP+HOG": 1342,
}

EXPECTED_BRANCH_DIMS: Dict[str, int] = {
    "deep": 1280,
    "glcm": 12,
    "lbp": 18,
    "hog": 32,
    "color_lab": 6,
    "lab": 6,
}


class _ArmsDict(dict):
    """Dictionary holding the 8 controlled fusion arms, with transparent support
    for legacy aliases, space-formatted variants, and backward-compatible lookups."""
    _ALIASES: Dict[str, str] = {
        "EfficientNet+HOG": "EfficientNet+HOG-PCA",
        "EfficientNet+GLCM+LBP+HOG+LAB": "EfficientNet+GLCM+LBP+HOG-PCA+LAB",
        "EfficientNet+Handcrafted": "EfficientNet+GLCM+LBP+HOG-PCA+LAB",
        "EfficientNet+Full_Handcrafted": "EfficientNet+GLCM+LBP+HOG-PCA+LAB",
        "EfficientNet + GLCM": "EfficientNet+GLCM",
        "EfficientNet + LBP": "EfficientNet+LBP",
        "EfficientNet + HOG": "EfficientNet+HOG-PCA",
        "EfficientNet + HOG-PCA": "EfficientNet+HOG-PCA",
        "EfficientNet + LAB": "EfficientNet+LAB",
        "EfficientNet + GLCM + LBP": "EfficientNet+GLCM+LBP",
        "EfficientNet + GLCM + LBP + LAB": "EfficientNet+GLCM+LBP+LAB",
        "EfficientNet + GLCM + LBP + HOG + LAB": "EfficientNet+GLCM+LBP+HOG-PCA+LAB",
        "EfficientNet + GLCM + LBP + HOG-PCA + LAB": "EfficientNet+GLCM+LBP+HOG-PCA+LAB",
        "EfficientNet + Handcrafted": "EfficientNet+GLCM+LBP+HOG-PCA+LAB",
    }
    _LEGACY: Dict[str, Tuple[str, ...]] = {
        "EfficientNet+GLCM+LBP+HOG": ("deep", "glcm", "lbp", "hog"),
    }

    def __getitem__(self, key: str) -> Tuple[str, ...]:
        if super().__contains__(key):
            return super().__getitem__(key)
        clean_key = key.replace(" ", "") if isinstance(key, str) else key
        if super().__contains__(clean_key):
            return super().__getitem__(clean_key)
        if clean_key in self._ALIASES:
            target = self._ALIASES[clean_key]
            if super().__contains__(target):
                return super().__getitem__(target)
        if key in self._ALIASES:
            target = self._ALIASES[key]
            if super().__contains__(target):
                return super().__getitem__(target)
        if key in self._LEGACY:
            return self._LEGACY[key]
        if clean_key in self._LEGACY:
            return self._LEGACY[clean_key]
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            clean_key = key.replace(" ", "")
            return (clean_key in self._ALIASES or key in self._ALIASES or
                    key in self._LEGACY or clean_key in self._LEGACY or
                    super().__contains__(clean_key))
        return False

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


ARMS: Dict[str, Tuple[str, ...]] = _ArmsDict(CONTROLLED_ARMS)


def _canonicalize_branches(branches: Tuple[str, ...]) -> Tuple[str, ...]:
    """Ensures deterministic branch ordering in canonical sequence:
    ('deep', 'glcm', 'lbp', 'hog', 'color_lab').
    Resolves branch aliases (e.g. 'lab' -> 'color_lab')."""
    normalized = []
    for b in branches:
        nb = BRANCH_ALIASES.get(b, b)
        if nb not in CANONICAL_BRANCH_ORDER:
            raise ValueError(f"Unknown branch {b!r}. Registered: {CANONICAL_BRANCH_ORDER}")
        if nb not in normalized:
            normalized.append(nb)
    return tuple(sorted(normalized, key=lambda b: CANONICAL_BRANCH_ORDER.index(b)))


def get_branch_feature_names(branch: str, config: Optional[PSDConfig] = None) -> List[str]:
    """Returns stable, deterministic feature names for a single branch."""
    canonical_b = BRANCH_ALIASES.get(branch, branch)
    if canonical_b == "deep":
        return [f"efficientnet_{i}" for i in range(1280)]
    elif canonical_b == "glcm":
        from modules.handcrafted_features import _GLCM_PROPS
        distances = config.glcm_distances if config else (1, 2)
        names = []
        for prop in _GLCM_PROPS:
            for dist in distances:
                names.append(f"glcm_{prop}_d{dist}")
        return names
    elif canonical_b == "lbp":
        n_points = config.lbp_n_points if config else 16
        n_bins = n_points + 2
        return [f"lbp_bin_{i}" for i in range(n_bins)]
    elif canonical_b == "hog":
        n_comp = config.hog_pca_components if config else 32
        return [f"hog_pca_{i}" for i in range(n_comp)]
    elif canonical_b == "color_lab":
        return [
            "lab_L_mean", "lab_L_std",
            "lab_a_mean", "lab_a_std",
            "lab_b_mean", "lab_b_std",
        ]
    else:
        raise ValueError(f"Unknown branch for feature naming: {branch!r}")


def get_fused_feature_names(branches: Tuple[str, ...], config: Optional[PSDConfig] = None) -> List[str]:
    """Returns stable, deterministic feature names for the fused vector in canonical order."""
    canonical = _canonicalize_branches(branches)
    names = []
    for b in canonical:
        names.extend(get_branch_feature_names(b, config))
    return names


def get_fused_slice_map(branches: Tuple[str, ...], config: Optional[PSDConfig] = None) -> Dict[str, Tuple[int, int]]:
    """Returns contiguous [start, end) column index ranges for each branch in the fused vector."""
    canonical = _canonicalize_branches(branches)
    slice_map = {}
    current = 0
    for b in canonical:
        fnames = get_branch_feature_names(b, config)
        dim = len(fnames)
        slice_map[b] = (current, current + dim)
        current += dim
    return slice_map


def get_representation_id(arm_or_branches: Union[str, Tuple[str, ...]]) -> str:
    """Returns canonical Phase 5 representation ID for cache and provenance tracking."""
    if isinstance(arm_or_branches, str):
        if arm_or_branches in ARM_REPRESENTATION_IDS:
            return ARM_REPRESENTATION_IDS[arm_or_branches]
        clean = arm_or_branches.replace(" ", "")
        if clean in ARM_REPRESENTATION_IDS:
            return ARM_REPRESENTATION_IDS[clean]
        if arm_or_branches in ARMS:
            branches = ARMS[arm_or_branches]
        else:
            raise ValueError(f"Unknown arm name: {arm_or_branches!r}")
    else:
        branches = arm_or_branches
    canonical = _canonicalize_branches(branches)
    branch_token_map = {
        "deep": "efficientnet",
        "glcm": "glcm",
        "lbp": "lbp",
        "hog": "hog",
        "color_lab": "lab",
    }
    tokens = [branch_token_map[b] for b in canonical]
    return "_".join(tokens)


def validate_fusion_dimensions(
    arm_or_branches: Union[str, Tuple[str, ...]], actual_dim: int, config: Optional[PSDConfig] = None
) -> int:
    """Validates that a fused feature matrix matches its required dimensions exactly."""
    if isinstance(arm_or_branches, str) and arm_or_branches in EXPECTED_ARM_DIMS:
        expected = EXPECTED_ARM_DIMS[arm_or_branches]
    else:
        if isinstance(arm_or_branches, str):
            branches = ARMS[arm_or_branches]
        else:
            branches = arm_or_branches
        canonical = _canonicalize_branches(branches)
        expected = sum(EXPECTED_BRANCH_DIMS[b] for b in canonical)
    if actual_dim != expected:
        raise ValueError(
            f"Fusion dimension mismatch for {arm_or_branches!r}! "
            f"Expected {expected}-D, got {actual_dim}-D."
        )
    return expected


def build_fusion_manifest(
    representation_id_str: str,
    fold_index: int,
    split_name: str,
    psd_ids: List[str],
    feature_dim: int,
    branch_dims: Dict[str, int],
    config: PSDConfig,
    source_experiment: str,
) -> Dict[str, Any]:
    """Generates a complete provenance manifest for a fused feature representation."""
    import hashlib
    from pathlib import Path

    freeze_path = config.aef_crc_reports_dir / "dataset_freeze.json"
    freeze_hash = None
    if freeze_path.exists():
        try:
            fdata = json.loads(freeze_path.read_text(encoding="utf-8"))
            freeze_hash = fdata.get("freeze_hash") or fdata.get("dataset_hash")
        except Exception:
            pass

    fold_plan_path = config.aef_crc_reports_dir / "fold_plan.csv"
    fold_plan_hash = None
    if fold_plan_path.exists():
        try:
            fold_plan_hash = hashlib.sha256(fold_plan_path.read_bytes()).hexdigest()
        except Exception:
            pass

    branches = tuple(branch_dims.keys())
    canonical = _canonicalize_branches(branches)

    return {
        "representation_id": representation_id_str,
        "phase": "phase5",
        "source_experiment": source_experiment,
        "fold_index": fold_index,
        "split": split_name,
        "n_samples": len(psd_ids),
        "feature_dim": feature_dim,
        "branch_dims": branch_dims,
        "canonical_branch_order": list(canonical),
        "slice_boundaries": {b: list(s) for b, s in get_fused_slice_map(canonical, config).items()},
        "feature_names": get_fused_feature_names(canonical, config),
        "preprocessing_mode": config.preprocessing_mode,
        "dataset_freeze_hash": freeze_hash,
        "fold_plan_hash": fold_plan_hash,
        "random_seed": config.random_seed,
    }


BRANCH_DIMENSIONS: Dict[str, int] = EXPECTED_BRANCH_DIMS
sort_branches_deterministically = _canonicalize_branches
get_fusion_slice_boundaries = get_fused_slice_map
get_fusion_feature_names = get_fused_feature_names


def get_fusion_provenance(
    arm: FusionArm | str,
    config: PSDConfig,
    fold_index: int,
    split: str,
) -> Dict[str, Any]:
    """Generates structured provenance metadata for downstream traceability."""
    if isinstance(arm, str):
        arm_obj = FUSION_ARMS_BY_NAME.get(arm) or FUSION_ARMS_BY_ID.get(arm) or FUSION_ARMS_BY_REPR_ID.get(arm)
        if arm_obj is None:
            raise KeyError(f"Unknown arm identifier: {arm}")
    else:
        arm_obj = arm

    boundaries = get_fusion_slice_boundaries(arm_obj.branches, config)
    names = get_fusion_feature_names(arm_obj.branches, config)
    return {
        "arm_id": arm_obj.arm_id,
        "name": arm_obj.name,
        "representation_id": arm_obj.representation_id,
        "purpose": arm_obj.purpose,
        "branches": list(arm_obj.branches),
        "expected_dim": arm_obj.expected_dim,
        "actual_feature_count": len(names),
        "slice_boundaries": {b: list(lims) for b, lims in boundaries.items()},
        "fold_index": fold_index,
        "split": split,
        "backbone": getattr(config, "backbone", "efficientnet_b0"),
        "preprocessing_mode": getattr(config, "preprocessing_mode", "standard"),
        "hog_pca_components": getattr(config, "hog_pca_components", 32),
        "leakage_guarantees": {
            "hog_pca_fit_on": "train_split_only",
            "feature_normalizer_fit_on": "train_split_only",
            "val_transform_only": True,
            "no_second_pca": True,
        },
    }


@dataclass
class FusionFoldData:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: List[str]
    y_val: List[str]
    psd_ids_train: List[str]
    psd_ids_val: List[str]
    branch_dims: Dict[str, int]
    hog_reducer: Optional[Any] = None
    feature_normalizers: Optional[Dict[str, Any]] = None

    @property
    def branch_normalizers(self) -> Optional[Dict[str, Any]]:
        return self.feature_normalizers


def _load_branch_vectors(config, records, branch, source_experiment, fold_index, split_name, extractor, branch_reducer=None):
    if branch == "deep":
        from modules.backbones import validate_deep_feature_cache
        validate_deep_feature_cache(
            config, source_experiment, fold_index, split_name,
            expected_psd_ids=[r.psd_id for r in records], backbone_name=config.backbone,
        )
        return [load_deep_feature(config, source_experiment, fold_index, split_name, r.psd_id) for r in records]

    if branch not in HANDCRAFTED_BRANCH_REGISTRY:
        raise ValueError(f"Unknown branch: {branch!r}. Registered: 'deep', {sorted(HANDCRAFTED_BRANCH_REGISTRY)}")

    cached = []
    for r in records:
        c = None
        if hasattr(extractor, "_load_cached"):
            c = extractor._load_cached(r.psd_id)
        if c is None:
            c = extractor.extract_and_cache(load_and_preprocess_image(config, r.file_path, r.psd_id), r.psd_id)
        cached.append(c)
    get_vector, apply_reducer = HANDCRAFTED_BRANCH_REGISTRY[branch]
    return apply_reducer([get_vector(c) for c in cached], branch_reducer)


def build_fusion_fold(
    config: PSDConfig, fold: Fold, branches: Tuple[str, ...], source_experiment: str,
    extractor: Optional[HandcraftedFeatureExtractor] = None,
) -> FusionFoldData:
    """Builds one fold's fused feature matrix for the given branch tuple.
    ALL fitting (per-branch normalizer, HOG PCA) happens on
    fold.train_records only -- val is transform-only, never re-fit.

    source_experiment: the Phase-3 experiment name (e.g. "P3-PRE") whose
    cached deep features to read -- must match what
    modules.training.run_experiment(..., extract_features=True) actually
    produced. Passed explicitly rather than hardcoded so a caller can
    never silently mix features from two different trained models.
    """
    extractor = extractor or HandcraftedFeatureExtractor(config)
    train_records, val_records = fold.train_records, fold.val_records
    canonical_branches = _canonicalize_branches(branches)

    # PSD-ID alignment: explicit, not assumed. Duplicate IDs within a
    # split would silently misalign branch matrices built independently
    # per branch below (each branch is loaded in `records` order).
    train_ids = [r.psd_id for r in train_records]
    val_ids = [r.psd_id for r in val_records]
    assert len(set(train_ids)) == len(train_ids), f"Fold {fold.fold_index}: duplicate PSD ID in train_records"
    assert len(set(val_ids)) == len(val_ids), f"Fold {fold.fold_index}: duplicate PSD ID in val_records"
    assert not (set(train_ids) & set(val_ids)), f"Fold {fold.fold_index}: PSD ID appears in both train and val"

    hog_reducer = None
    if "hog" in canonical_branches:
        train_hog_raw = []
        for r in train_records:
            c = extractor._load_cached(r.psd_id)
            if c is None:
                c = extractor.extract_and_cache(load_and_preprocess_image(config, r.file_path, r.psd_id), r.psd_id)
            train_hog_raw.append(c.hog)
        hog_reducer = FoldSafeFeatureReducer(n_components=config.hog_pca_components).fit(train_hog_raw)

    train_matrices, val_matrices, branch_dims = [], [], {}
    for branch in canonical_branches:
        train_vecs = _load_branch_vectors(config, train_records, branch, source_experiment, fold.fold_index, "train", extractor, hog_reducer)
        val_vecs = _load_branch_vectors(config, val_records, branch, source_experiment, fold.fold_index, "val", extractor, hog_reducer)

        normalizer = FeatureNormalizer().fit(train_vecs)  # fold-train only -- mirrors FeatureNormalizer's own discipline
        X_train_branch = normalizer.transform(train_vecs)
        X_val_branch = normalizer.transform(val_vecs)

        train_matrices.append(X_train_branch)
        val_matrices.append(X_val_branch)
        branch_dims[branch] = X_train_branch.shape[1]

    X_train_fused = np.concatenate(train_matrices, axis=1)
    X_val_fused = np.concatenate(val_matrices, axis=1)
    validate_fusion_dimensions(canonical_branches, X_train_fused.shape[1], config)
    validate_fusion_dimensions(canonical_branches, X_val_fused.shape[1], config)

    return FusionFoldData(
        X_train=X_train_fused,
        X_val=X_val_fused,
        y_train=[r.mapped_class for r in train_records],
        y_val=[r.mapped_class for r in val_records],
        psd_ids_train=train_ids, psd_ids_val=val_ids,
        branch_dims=branch_dims,
    )


# Sentinel fold_index for the final retrain's deep-feature cache
# location -- distinct from any real CV fold index (always >= 0), so
# artifacts/phase3/deep_features/<experiment>/fold_-1/{final_train,
# calibration}/ can never collide with a real fold's cache directory.
FINAL_MODEL_FOLD_INDEX = -1


def build_fusion_final(
    config: PSDConfig, final_train_records: List, calibration_records: List,
    branches: Tuple[str, ...], source_experiment: str,
    extractor: Optional[HandcraftedFeatureExtractor] = None,
) -> FusionFoldData:
    """The final-retrain counterpart to build_fusion_fold: CV-based
    model/hyperparameter selection is already complete by the time this
    runs, so there is no CV fold here -- final_train_records is the
    COMPLETE permitted training data (all non-augmented outer-train
    images, reunified across folds; see run_aef_crc_phase7.py), and
    calibration_records is the project's ONE true held-out calibration
    set (splitter.py's val/, FoldPlan.holdout_val_records) -- NOT a
    CV-internal validation slice.

    Returns a plain FusionFoldData (X_train/y_train = final_train_records,
    X_val/y_val = calibration_records) so every EXISTING consumer --
    modules.feature_selection.run_selector, modules.fusion.
    build_classifier, modules.fusion.compute_sample_weights -- works
    completely unchanged; none of them read anything beyond X_train/
    y_train/X_val/y_val/branch_dims, and none of them care whether
    "val" came from a CV fold or the calibration set. Same fit
    discipline as build_fusion_fold: every per-branch transform
    (normalizer, HOG PCA) is fit on final_train_records ONLY;
    calibration_records is transform-only, never re-fit -- the
    calibration set must remain as untouched by fitting as any other
    held-out data, exactly like Phase 5/6's fold.val_records.

    Reads the 'deep' branch from source_experiment's cache under
    fold_index=FINAL_MODEL_FOLD_INDEX, split_name "final_train" /
    "calibration" -- a cache location Phase 3's EXISTING, unmodified
    extract_deep_features() can populate with two more calls (one per
    split) over these exact record sets, using the already-trained
    model's checkpoint. That extraction needs the real trained Keras
    model and is Phase-3 infrastructure, not performed here; if the
    cache is missing, validate_deep_feature_cache (inside
    _load_branch_vectors) raises -- same as any other missing-cache
    case, never silently skipped.
    """
    extractor = extractor or HandcraftedFeatureExtractor(config)
    train_ids = [r.psd_id for r in final_train_records]
    calib_ids = [r.psd_id for r in calibration_records]
    canonical_branches = _canonicalize_branches(branches)
    assert len(set(train_ids)) == len(train_ids), "build_fusion_final: duplicate PSD ID in final_train_records"
    assert len(set(calib_ids)) == len(calib_ids), "build_fusion_final: duplicate PSD ID in calibration_records"
    assert not (set(train_ids) & set(calib_ids)), "build_fusion_final: PSD ID appears in both final_train and calibration"

    hog_reducer = None
    if "hog" in canonical_branches:
        train_hog_raw = [
            extractor.extract_and_cache(load_and_preprocess_image(config, r.file_path, r.psd_id), r.psd_id).hog
            for r in final_train_records
        ]
        hog_reducer = FoldSafeFeatureReducer(n_components=config.hog_pca_components).fit(train_hog_raw)

    train_matrices, calib_matrices, branch_dims = [], [], {}
    normalizers = {}
    for branch in canonical_branches:
        train_vecs = _load_branch_vectors(config, final_train_records, branch, source_experiment, FINAL_MODEL_FOLD_INDEX, "final_train", extractor, hog_reducer)
        calib_vecs = _load_branch_vectors(config, calibration_records, branch, source_experiment, FINAL_MODEL_FOLD_INDEX, "calibration", extractor, hog_reducer)

        normalizer = FeatureNormalizer().fit(train_vecs)  # final_train only -- calibration is transform-only, same discipline as build_fusion_fold
        X_train_branch = normalizer.transform(train_vecs)
        X_calib_branch = normalizer.transform(calib_vecs)

        normalizers[branch] = normalizer
        train_matrices.append(X_train_branch)
        calib_matrices.append(X_calib_branch)
        branch_dims[branch] = X_train_branch.shape[1]

    X_train_fused = np.concatenate(train_matrices, axis=1)
    X_calib_fused = np.concatenate(calib_matrices, axis=1)
    validate_fusion_dimensions(canonical_branches, X_train_fused.shape[1], config)
    validate_fusion_dimensions(canonical_branches, X_calib_fused.shape[1], config)

    return FusionFoldData(
        X_train=X_train_fused,
        X_val=X_calib_fused,
        y_train=[r.mapped_class for r in final_train_records],
        y_val=[r.mapped_class for r in calibration_records],
        psd_ids_train=train_ids, psd_ids_val=calib_ids,
        branch_dims=branch_dims,
        hog_reducer=hog_reducer,
        feature_normalizers=normalizers,
    )


def compute_sample_weights(labels: List[str], class_weights: Dict[str, float]) -> np.ndarray:
    """Per-sample weight array from fold.class_weights -- sample_weight,
    not class_weight, is the version-independent path every classifier
    below actually supports (the same lesson learned and fixed for
    Keras/tf.data in Phase 3 applies here too: don't trust a
    library-specific class_weight dict path when a plain per-sample
    array works identically everywhere)."""
    return np.array([class_weights[label] for label in labels], dtype=np.float64)


class _XGBoostStringLabelAdapter:
    """Keep AEF-CRC's canonical string labels while feeding XGBoost integers.

    AEF-CRC deliberately represents disease classes by their canonical names
    (for example ``"Psoriasis"``), while recent XGBoost releases require
    classification targets to be integer-encoded.  Encoding is therefore kept
    strictly inside this classifier adapter rather than leaking numeric class
    IDs into dataset records, fold logic, metrics, calibration, or reports.

    ``classes_`` is exposed back as the original string labels so downstream
    code such as Phase 7's probability-column alignment continues to operate
    in the project's canonical class space.
    """

    def __init__(self, estimator):
        from sklearn.preprocessing import LabelEncoder
        self.estimator = estimator
        self._label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        y_array = np.asarray(y, dtype=object)
        encoded = self._label_encoder.fit_transform(y_array)
        self.classes_ = self._label_encoder.classes_.copy()
        if sample_weight is None:
            self.estimator.fit(X, encoded)
        else:
            self.estimator.fit(X, encoded, sample_weight=sample_weight)
        return self

    def predict(self, X):
        encoded_pred = np.asarray(self.estimator.predict(X)).astype(int)
        return self._label_encoder.inverse_transform(encoded_pred)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def _xgboost_or_fallback(random_seed: int):
    """XGBoost if available, with an internal string-label adapter.

    AEF-CRC keeps canonical disease names as labels throughout its data and
    evaluation layers.  The adapter is the narrow compatibility boundary for
    XGBoost's integer-target requirement; it does not alter the public label
    representation used anywhere else in the framework.
    """
    try:
        import xgboost as xgb
        estimator = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=random_seed, eval_metric="mlogloss",
        )
        return "xgboost", _XGBoostStringLabelAdapter(estimator)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(random_state=random_seed)
        return "sklearn_hgb_fallback", clf


def _random_forest(random_seed: int):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=300,
        criterion="gini",  # Gini impurity split criterion (CART node impurity minimization; Breiman et al., 1984)
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,  # Handled exclusively via fold-local sample_weight in clf.fit() to avoid double-weighting
        random_state=random_seed,
        n_jobs=-1,
    )
    return "random_forest", clf


def _logistic_regression(random_seed: int):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,  # Handled exclusively via fold-local sample_weight in clf.fit() to avoid double-weighting
        random_state=random_seed,
    )
    return "logistic_regression", clf


def _svm(random_seed: int):
    from sklearn.svm import SVC
    return "svm", SVC(probability=True, random_state=random_seed)


def _histgradientboosting(random_seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return "histgradientboosting", HistGradientBoostingClassifier(random_state=random_seed)


# Registry, same pattern as modules/adapters.py's _ADAPTER_REGISTRY --
# see modules/backbones.py's docstring for the full reasoning.
CLASSIFIER_REGISTRY = {
    "random_forest": _random_forest,
    "rf": _random_forest,
    "logistic_regression": _logistic_regression,
    "lr": _logistic_regression,
    "xgboost_or_fallback": _xgboost_or_fallback,
    "xgboost": _xgboost_or_fallback,
    "svm": _svm,
    "histgradientboosting": _histgradientboosting,
}


def build_classifier(random_seed: int, classifier_name: str = "random_forest"):
    """Build certified downstream classifier for AEF-CRC.

    Defaults to Phase 7 primary classifier 'random_forest' (RandomForestClassifier,
    n_estimators=300, class_weight=None; fold-local sample_weight handled at fit time).
    Comparator: 'logistic_regression' (L2 multinomial, class_weight=None; fold-local
    sample_weight handled at fit time).
    Legacy comparators 'xgboost_or_fallback', 'svm', 'histgradientboosting' preserved for backward compatibility.

    Returns (backend_name, classifier).
    """
    if classifier_name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier_name '{classifier_name}'. Registered: {sorted(CLASSIFIER_REGISTRY.keys())}")
    return CLASSIFIER_REGISTRY[classifier_name](random_seed)


PHASE7_ALLOWED_CLASSIFIERS = {"random_forest", "rf", "logistic_regression", "lr"}


def validate_phase7_classifier(classifier_name: str) -> None:
    """Enforces that the Phase 7 production classifier is strictly restricted to
    Random Forest or Logistic Regression.

    Raises ValueError if any legacy or non-production model (e.g. XGBoost, SVM,
    HistGradientBoosting) is configured.
    """
    if classifier_name not in PHASE7_ALLOWED_CLASSIFIERS:
        raise ValueError(
            f"Phase 7 production classifier guard violation: classifier_name='{classifier_name}' "
            f"is not permitted. Final production classifier is strictly restricted to "
            f"{sorted(PHASE7_ALLOWED_CLASSIFIERS)}. Legacy/unsupported models (e.g. XGBoost, SVM, "
            f"HistGradientBoosting) are prohibited in Phase 7 production."
        )


