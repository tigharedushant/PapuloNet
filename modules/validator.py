"""
modules/validator.py

PSD-HP Stage 1: Dataset Validation.

Before any scanning, class mapping, or extraction happens, PSD-HP must
confirm that every configured raw dataset actually exists on disk,
contains readable image files, and is not silently empty or
corrupted. This module is the framework's first line of defense
against garbage-in-garbage-out harmonization.

As of the adapter-layer revision, the per-class checks below are
grouped by the adapter's normalized labels (modules/adapters.py),
not by raw top-level subfolders — for DermNet that means grouping by
actual disease name (with train/ and test/ already merged), not by
"train"/"test" themselves, which a plain iterdir() would incorrectly
report as if they were classes.

Explicit non-responsibilities (by design, single-responsibility
principle): this module does NOT map raw labels to target classes
(that is mapper.py, Phase 2), does NOT copy or move any files (that
is extractor.py, Phase 3), and does NOT compute detailed image
quality metrics such as blur or resolution adequacy beyond "can this
file be opened as an image at all" (that is quality_assessor.py,
Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.adapters import discover_dataset, DiscoveredImage


@dataclass
class ClassFolderResult:
    """Validation outcome for a single normalized class label within a dataset."""
    folder_name: str
    image_count: int = 0
    corrupted_files: List[str] = field(default_factory=list)


@dataclass
class DatasetValidationResult:
    """Validation outcome for one entire raw dataset, e.g. DermNet."""
    dataset_name: str
    dataset_path: Path
    exists: bool
    is_directory: bool
    class_folders: List[ClassFolderResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return sum(cf.image_count for cf in self.class_folders)

    @property
    def total_corrupted(self) -> int:
        return sum(len(cf.corrupted_files) for cf in self.class_folders)

    @property
    def status(self) -> str:
        if self.errors:
            return "FAIL"
        if self.warnings:
            return "WARNING"
        return "PASS"


class DatasetValidator:
    """
    Validates every dataset configured in PSDConfig.datasets.

    A dataset PASSes if its path exists, is a directory, contains at
    least one normalized class label (per the adapter layer), and has
    at least one readable image. Corrupted individual files and empty
    class labels are recorded as WARNINGs (not FAILures) so the
    pipeline can still proceed on the remaining valid data, with full
    transparency about exactly what was excluded and why.
    """

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("validator", config.logs_dir, config.log_level)

    def validate_all(self) -> List[DatasetValidationResult]:
        """Validate every configured dataset and write the validation report."""
        self.logger.info("=== PSD-HP Stage 1: Dataset Validation started ===")
        results: List[DatasetValidationResult] = []

        for dataset_cfg in self.config.datasets:
            self.logger.info(f"Validating dataset '{dataset_cfg.name}' at {dataset_cfg.path}")
            result = self._validate_single_dataset(dataset_cfg)
            results.append(result)
            self.logger.info(
                f"{dataset_cfg.name}: status={result.status} | "
                f"class_folders={len(result.class_folders)} | "
                f"images={result.total_images} | corrupted={result.total_corrupted}"
            )

        self._write_report(results)
        self.logger.info("=== PSD-HP Stage 1: Dataset Validation finished ===")
        return results

    def _validate_single_dataset(self, dataset_cfg) -> DatasetValidationResult:
        name, path = dataset_cfg.name, dataset_cfg.path
        result = DatasetValidationResult(
            dataset_name=name,
            dataset_path=path,
            exists=path.exists(),
            is_directory=path.is_dir() if path.exists() else False,
        )

        if not result.exists:
            result.errors.append(f"Path does not exist: {path}")
            self.logger.error(f"{name}: path does not exist -> {path}")
            return result

        if not result.is_directory:
            result.errors.append(f"Path exists but is not a directory: {path}")
            self.logger.error(f"{name}: path is not a directory -> {path}")
            return result

        discovered_folders = discover_dataset(self.config, dataset_cfg)
        if not discovered_folders:
            result.errors.append(f"No class labels discovered inside {path} (adapter: {dataset_cfg.adapter_kind})")
            self.logger.error(f"{name}: no class labels discovered inside {path}")
            return result

        for folder in discovered_folders:
            cf_result = self._validate_images(folder.normalized_label, folder.images)
            result.class_folders.append(cf_result)

            if cf_result.image_count == 0:
                msg = f"Class label '{folder.normalized_label}' contains no readable images"
                result.warnings.append(msg)
                self.logger.warning(f"{name}/{folder.normalized_label}: 0 readable images")

            if cf_result.corrupted_files:
                msg = f"Class label '{folder.normalized_label}' has {len(cf_result.corrupted_files)} corrupted file(s)"
                result.warnings.append(msg)
                self.logger.warning(f"{name}/{folder.normalized_label}: {len(cf_result.corrupted_files)} corrupted file(s)")

        if result.total_images == 0:
            result.errors.append(f"Dataset '{name}' has zero readable images across all class labels")
            self.logger.error(f"{name}: zero readable images across the entire dataset")

        return result

    def _validate_images(self, label: str, images: List[DiscoveredImage]) -> ClassFolderResult:
        """Count readable images and detect corrupted files for one normalized class label."""
        cf_result = ClassFolderResult(folder_name=label)

        for img in tqdm(images, desc=f"Checking {label}", leave=False):
            try:
                with Image.open(img.path) as pil_img:
                    pil_img.verify()  # cheap integrity check; does not decode full pixel data
                cf_result.image_count += 1
            except Exception as exc:  # noqa: BLE001 — any failure here means "unreadable image"
                cf_result.corrupted_files.append(str(img.path))
                self.logger.debug(f"Corrupted or unreadable image: {img.path} ({exc})")

        return cf_result

    def _write_report(self, results: List[DatasetValidationResult]) -> None:
        """Write reports/dataset_validation_report.txt summarizing every dataset."""
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "dataset_validation_report.txt"

        lines = ["PSD-HP Dataset Validation Report", "=" * 40, ""]
        for r in results:
            lines.append(f"Dataset: {r.dataset_name}")
            lines.append(f"  Path: {r.dataset_path}")
            lines.append(f"  Status: {r.status}")
            lines.append(f"  Class labels found: {len(r.class_folders)}")
            for cf in r.class_folders:
                lines.append(
                    f"    - {cf.folder_name}: {cf.image_count} readable image(s), "
                    f"{len(cf.corrupted_files)} corrupted"
                )
            lines.append(f"  Total readable images: {r.total_images}")
            lines.append(f"  Total corrupted images: {r.total_corrupted}")
            if r.errors:
                lines.append("  Errors:")
                lines.extend(f"    - {e}" for e in r.errors)
            if r.warnings:
                lines.append("  Warnings:")
                lines.extend(f"    - {w}" for w in r.warnings)
            lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Validation report written to {report_path}")
