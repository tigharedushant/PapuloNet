"""
config.py

Central configuration module for PSD-HP (Papulosquamous Skin Disease
Harmonization Protocol).

Every path, threshold, and tunable parameter used anywhere in the
framework is defined here and nowhere else. No other module in PSD-HP
is permitted to hardcode a filesystem path, an image extension list,
a random seed, or a split ratio — they must all be read from the
PSDConfig object produced by get_config().

This module has zero dependencies on the rest of PSD-HP (it is always
the first thing imported), and it performs the one piece of "automatic
setup" the brief requires at startup: creating output/, reports/, and
logs/ if they do not already exist, so no module downstream has to
remember to do it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# The project root is wherever this file's parent's parent lives, i.e.
# PSD_HP/. Computed once, from disk, so PSD-HP can be cloned/moved
# anywhere without editing a single path by hand.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetConfig:
    """Describes one raw, already-downloaded dataset PSD-HP will harmonize."""
    name: str
    path: Path
    description: str = ""
    # Which DatasetAdapter (modules/adapters.py) understands this dataset's
    # real on-disk structure. "generic_flat" (the default) assumes
    # dataset_root/<class_folder>/<images> — true for Curated32. DermNet
    # and SkinDisNet need their own adapter because their real structure
    # is deeper than one level.
    adapter_kind: str = "generic_flat"


@dataclass(frozen=True)
class PSDConfig:
    """
    Immutable configuration object for the entire PSD-HP framework.

    frozen=True is deliberate: once created, a PSDConfig cannot be
    mutated by a later module, which prevents an entire class of "who
    changed my config mid-run" bugs in a multi-stage pipeline.
    """

    # ---- Core project paths ----
    project_root: Path = PROJECT_ROOT
    datasets_dir: Path = PROJECT_ROOT / "datasets"
    output_dir: Path = PROJECT_ROOT / "output"
    reports_dir: Path = PROJECT_ROOT / "reports"
    logs_dir: Path = PROJECT_ROOT / "logs"
    config_dir: Path = PROJECT_ROOT / "config"
    docs_dir: Path = PROJECT_ROOT / "docs"

    # ---- Output subfolders (each pipeline stage writes to its own) ----
    extracted_dir: Path = PROJECT_ROOT / "output" / "01_extracted"
    ambiguous_review_dir: Path = PROJECT_ROOT / "output" / "02_ambiguous_review"
    duplicates_removed_dir: Path = PROJECT_ROOT / "output" / "03_duplicates_removed"
    low_quality_dir: Path = PROJECT_ROOT / "output" / "04_low_quality_excluded"
    harmonized_dir: Path = PROJECT_ROOT / "output" / "05_harmonized"
    final_dir: Path = PROJECT_ROOT / "output" / "06_final_split"

    # ---- Source datasets already downloaded locally ----
    # Edit only the `path` values below to match where you placed each
    # dataset on disk. Everything downstream reads from here.
    datasets: List[DatasetConfig] = field(default_factory=lambda: [
        DatasetConfig(
            name="DermNet",
            path=PROJECT_ROOT / "datasets" / "dermnet",
            description="Kaggle shubhamgoel27/dermnet — 23-class clinical scrape",
            adapter_kind="dermnet",
        ),
        DatasetConfig(
            name="SkinDisNet",
            path=PROJECT_ROOT / "datasets" / "skindisnet",
            description="Mendeley yj3md44hxg — 6-class clinical set, incl. Seborrheic Dermatitis",
            adapter_kind="skindisnet",
        ),
        DatasetConfig(
            name="Curated32",
            path=PROJECT_ROOT / "datasets" / "curated32",
            description="Mendeley pgd42j3h5c — 32-category clinical/dermoscopic set",
            adapter_kind="generic_flat",
        ),
        DatasetConfig(
            name="AtlasISIC31",
            path=PROJECT_ROOT / "datasets" / "atlas_isic31",
            description="ISIC-derived 31-class atlas — dataset_root/{train,test}/<class>/, "
                         "one disease per folder (no filename reclassification needed)",
            adapter_kind="atlas31",
        ),
    ])

    # ---- Target disease taxonomy for AEF-CRC ----
    target_classes: List[str] = field(default_factory=lambda: [
        "Psoriasis",
        "Lichen_Planus",
        "Pityriasis_Rosea",
        "Seborrheic_Dermatitis",
    ])

    # ---- Class mapping file (raw dataset label -> canonical target class) ----
    class_mapping_file: Path = PROJECT_ROOT / "config" / "class_mapping.json"

    # ---- Image handling ----
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
    ])
    image_size: int = 224
    min_image_dimension: int = 32  # below this, an image is flagged as low quality

    # ---- Duplicate detection ----
    # Perceptual-hash (phash) Hamming distance at or below this value is
    # considered a near-duplicate. 0 = identical hash; typical usable
    # range for phash is 0-10 on a 64-bit hash.
    duplicate_hash_threshold: int = 5

    # ---- Quality assessment ----
    # Variance of the Laplacian below this value is flagged as blurry.
    # This is a standard, widely-used OpenCV blur heuristic, not a
    # clinical sharpness standard — tune per-dataset if needed.
    blur_variance_threshold: float = 100.0

    # ---- SkinDisNet augmented-image inclusion policy ----
    # "true"  -> always extract Augmented/ images alongside Preprocessed/
    # "false" -> never extract Augmented/ images
    # "auto"  -> extract Augmented/ images only if that class's
    #            Preprocessed/-only (original) image count is below
    #            augmented_auto_threshold
    # Set to "true" for the current project: Seborrheic Dermatitis has
    # very limited original image availability.
    include_augmented: str = "true"
    augmented_auto_threshold: int = 100

    # ---- Reproducibility ----
    random_seed: int = 42

    # ---- Train / validation / test split ----
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ---- Logging ----
    log_level: str = "INFO"

    # ---- AEF-CRC: modeling-layer paths (Phase 1) ----
    # AEF-CRC never writes into output/ or reports/ directly — it gets
    # its own subtrees so nothing it produces can ever collide with or
    # overwrite a PSD-HP harmonization report.
    aef_crc_output_dir: Path = PROJECT_ROOT / "output" / "aef_crc"
    aef_crc_reports_dir: Path = PROJECT_ROOT / "reports" / "aef_crc"

    # ---- AEF-CRC: imbalance strategy (Phase 1) ----
    # "train_fold_class_weights" is the only strategy implemented so far
    # (Part 5's formula, computed fresh inside each training fold).
    # "balanced_batch_sampling" is documented as a future secondary
    # experiment in the project brief, not yet implemented — listed
    # here only so later phases have one place to register it.
    class_weight_strategy: str = "train_fold_class_weights"

    # ---- AEF-CRC Phase 3: conditional preprocessing thresholds ----
    # Deliberately conservative defaults (see modules/preprocessing.py's
    # "brutal rule" docstring): when a signal is ambiguous, the trigger
    # is designed to fire LESS often, not more -- a missed hair-removal
    # opportunity costs nothing; a false-positive one can destroy lesion
    # boundaries, which is the worse failure mode for a medical image.
    hair_coverage_threshold: float = 0.03   # fraction of pixels (0.03 = 3%) classified as hair-like before hair removal triggers
    contrast_std_threshold: float = 40.0    # grayscale intensity std-dev below which CLAHE triggers (0-255 scale)
    preprocessing_mode: str = "conditional"  # "standard" (P0, no-op) | "conditional" (P1, trigger-based)

    # ---- AEF-CRC Phase 3: EfficientNet-B0 + training ----
    image_size: int = 224
    batch_size: int = 32
    stage1_learning_rate: float = 1e-3   # frozen-backbone, head-only training
    stage2_learning_rate: float = 1e-5   # unfrozen-upper-layers fine-tuning, deliberately much smaller
    stage1_epochs: int = 15
    stage2_epochs: int = 10
    unfrozen_layers: int = 20             # number of top backbone layers unfrozen in stage 2
    early_stopping_patience: int = 5
    dropout_rate: float = 0.3
    training_time_augmentation: str = "false"  # "false" (B0) | "true" (B1) -- see Phase 3 ablation

    # ---- AEF-CRC Phase 4: handcrafted features (GLCM/LBP/HOG) ----
    # Standard, well-precedented parameter choices -- deliberately not
    # tuned/enlarged, per Part "do not unnecessarily increase feature
    # dimensionality": these are the common textbook defaults, not a
    # search result.
    glcm_distances: tuple = (1, 3)
    glcm_angles: tuple = (0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345)  # 0, 45, 90, 135 deg
    glcm_levels: int = 32          # gray levels GLCM is computed over (quantized down from 256 -- standard practice, keeps the co-occurrence matrix a tractable 32x32 rather than 256x256)
    lbp_radius: int = 2
    lbp_n_points: int = 16         # 8 * radius, standard LBP convention
    lbp_method: str = "uniform"
    hog_pixels_per_cell: int = 32   # was 16 (6084-D); 32 brings HOG to 1296-D -- still ~43x GLCM+LBP combined (30-D), so also see hog_pca_components below
    hog_cells_per_block: int = 2
    hog_orientations: int = 9
    hog_pca_components: int = 32    # fold-safe PCA target for HOG specifically, roughly matching GLCM+LBP's combined magnitude -- see handcrafted_features.py's FoldSafeFeatureReducer
    color_feature_enabled: bool = True  # 6-D LAB color statistics (mean + std for L, a, b channels)
    aef_crc_phase4_artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "phase4"
    aef_crc_phase4_reports_dir: Path = PROJECT_ROOT / "reports" / "phase4"
    aef_crc_phase5_reports_dir: Path = PROJECT_ROOT / "reports" / "phase5"

    # ---- AEF-CRC Phase 6: feature selection/refinement ----
    # top_k is the shared target count for comparative feature selection (e.g. RFE),
    # so a "did selecting fewer features help" comparison is fair --
    # both methods are asked to reach the SAME budget, not left to pick
    # different counts that would confound the method comparison with
    # a dimensionality comparison. ~15% of the ~1280-1350-D fused vector
    # -- a round, stated number, not tuned to produce a nicer result.
    feature_selection_top_k: int = 200
    rfe_step: float = 0.05  # coarse elimination schedule -- single-feature (step=1) would mean ~1300 refits/fold, indefensible
    ga_population_size: int = 20
    ga_generations: int = 20
    ga_mutation_rate: float = 0.05
    ga_crossover_rate: float = 0.7
    ga_feature_count_penalty: float = 0.0005  # subtracted from fitness per selected feature, keeps GA from just selecting everything
    ga_nested_val_fraction: float = 0.2  # carved out of fold.train_records only, for GA fitness -- fold.val_records is never touched during evolution

    # ---- AEF-CRC Phase 6: Binary Dragonfly Algorithm (BDA) ----
    bda_population_size: int = 20
    bda_iterations: int = 30
    bda_v_max: float = 6.0
    bda_tau_min: float = 1.0
    bda_tau_max: float = 4.0
    bda_w_max: float = 0.9
    bda_w_min: float = 0.4
    bda_feature_count_penalty: float = 0.0005  # identical to ga_feature_count_penalty for fair comparison
    bda_nested_val_fraction: float = 0.2  # identical to ga_nested_val_fraction
    aef_crc_phase6_reports_dir: Path = PROJECT_ROOT / "reports" / "phase6"
    aef_crc_phase6_artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "phase6"

    # ---- AEF-CRC Phase 7: final model, CV evaluation, calibration handoff ----
    aef_crc_phase7_reports_dir: Path = PROJECT_ROOT / "reports" / "phase7"
    aef_crc_phase7_artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "phase7"

    # ---- AEF-CRC Phase 8: probability calibration ----
    # calibration_min_samples: below this, Platt/Isotonic scaling
    # are not fit at all (retain uncalibrated) -- "do NOT manufacture a
    # result" when there isn't enough calibration data to fit anything
    # defensibly. 10 is a low, permissive floor (a 2-parameter-per-class
    # Platt fit is already shaky well above 10) -- deliberately not
    # tuned to make this project's own small calibration set pass.
    calibration_min_samples: int = 10
    calibration_ece_bins: int = 10  # standard choice (Guo et al. 2017); reliability-bin count for ECE and the reliability-diagram table
    aef_crc_phase8_reports_dir: Path = PROJECT_ROOT / "reports" / "phase8"
    aef_crc_phase8_artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "phase8"

    # ---- AEF-CRC Phase 9: conformal prediction ----
    conformal_alpha: float = 0.10  # error rate, nominal coverage 1 - alpha = 0.90 (90%)
    aef_crc_phase9_reports_dir: Path = PROJECT_ROOT / "reports" / "phase9"
    aef_crc_phase9_artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "phase9"

    # ---- AEF-CRC: experiment component selection (modular framework) ----
    # These four fields ARE "the experiment configuration" -- deliberately
    # added directly to PSDConfig rather than a second, parallel
    # ExperimentConfig class. Reasoning: every function in this codebase
    # already takes a single `config: PSDConfig` argument, and every
    # existing experiment variant (P3-BASE/PRE/AUG, etc.) is already created
    # via dataclasses.replace(config, some_field=...). Adding a second
    # config object that ALSO needs threading through training.py/fusion.py/
    # feature_selection.py would duplicate that mechanism for no capability
    # gain -- exactly the "over-engineered / generic / hard to debug"
    # pattern this phase was told to avoid. Creating a new experiment is
    # still just dataclasses.replace(config, backbone="convnext_tiny").
    # See modules/experiment_config.py for validation + experiment_id().
    backbone: str = "efficientnet_b0"
    classifier_name: str = "random_forest"
    feature_selection_method: str = "bda"

    # ---- AEF-CRC Phase 3: artifact/report locations ----
    aef_crc_artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "phase3"
    aef_crc_phase3_reports_dir: Path = PROJECT_ROOT / "reports" / "phase3"
    aef_crc_phase3_logs_dir: Path = PROJECT_ROOT / "logs" / "phase3"

    def required_dirs(self) -> List[Path]:
        """Directories PSD-HP must guarantee exist before any stage runs."""
        return [
            self.output_dir, self.reports_dir, self.logs_dir,
            self.extracted_dir, self.ambiguous_review_dir,
            self.duplicates_removed_dir, self.low_quality_dir,
            self.harmonized_dir, self.final_dir,
            self.aef_crc_output_dir, self.aef_crc_reports_dir,
            self.aef_crc_artifacts_dir, self.aef_crc_phase3_reports_dir,
            self.aef_crc_phase3_logs_dir,
            self.aef_crc_phase4_artifacts_dir, self.aef_crc_phase4_reports_dir,
            self.aef_crc_phase5_reports_dir,
            self.aef_crc_phase6_reports_dir, self.aef_crc_phase6_artifacts_dir,
            self.aef_crc_phase7_reports_dir, self.aef_crc_phase7_artifacts_dir,
            self.aef_crc_phase8_reports_dir, self.aef_crc_phase8_artifacts_dir,
            self.aef_crc_phase9_reports_dir, self.aef_crc_phase9_artifacts_dir,
        ]

    def validate_split_ratios(self) -> None:
        """Fail fast if the split ratios in this config don't sum to 1.0."""
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if total != 1.0:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}"
            )


def get_config() -> PSDConfig:
    """
    Factory function: build, validate, and return the PSD-HP configuration.

    This is the ONLY way the rest of PSD-HP should obtain a PSDConfig —
    never instantiate PSDConfig() directly elsewhere, so validation and
    directory creation always happen exactly once, in one place.
    """
    cfg = PSDConfig()
    cfg.validate_split_ratios()
    for directory in cfg.required_dirs():
        directory.mkdir(parents=True, exist_ok=True)
    return cfg
