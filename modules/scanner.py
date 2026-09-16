"""
modules/scanner.py

PSD-HP Stage 2a: Dataset Scanning.

Where validator.py (Stage 1) asks "is this dataset readable and
non-empty", scanner.py asks a different question: "what does this
dataset's folder structure actually look like, right now, on disk."
It produces the ground-truth inventory of normalized class labels
and per-label image counts that mapper.py cross-checks
class_mapping.json against — this is what lets the framework detect
a class_mapping.json that has drifted out of sync with the real
dataset (renamed folders, folders the mapping file has never seen,
etc.) instead of silently mis-mapping or skipping them.

As of the adapter-layer revision, scanner.py no longer walks dataset
folders directly with iterdir()/rglob() — it delegates that to
modules.adapters.discover_dataset(), which understands each
dataset's real structure (DermNet's train/test split, SkinDisNet's
dual-version/Preprocessed-Augmented layout, or the plain flat layout
for everything else). scanner.py's own responsibility is unchanged:
turn whatever the adapter found into a DatasetScanResult and
scan_report.xlsx. It still does not decide what a label maps to
(mapper.py) and does not touch class_mapping.json at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.adapters import discover_dataset, DiscoveredImage


@dataclass
class ScannedFolder:
    """All images sharing one normalized label within one dataset —
    may correspond to more than one real on-disk folder (e.g. DermNet's
    merged train+test, or SkinDisNet's merged version/augmentation dirs)."""
    dataset_name: str
    folder_name: str          # normalized label, used as the class_mapping.json lookup key
    images: List[DiscoveredImage] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def folder_path(self) -> Path:
        """Best-effort single representative path, for display/logging only.
        Real per-image paths (and split/version) live in `images`."""
        return self.images[0].path.parent if self.images else Path()


@dataclass
class DatasetScanResult:
    """Full scan result for one dataset."""
    dataset_name: str
    dataset_path: Path
    folders: List[ScannedFolder] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return sum(f.image_count for f in self.folders)


class DatasetScanner:
    """
    Asks the adapter layer for every configured dataset's real,
    normalized folder structure and records it, independent of
    class_mapping.json.
    """

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("scanner", config.logs_dir, config.log_level)

    def scan_all(self) -> List[DatasetScanResult]:
        """Scan every configured dataset and write scan_report.xlsx."""
        self.logger.info("=== PSD-HP Stage 2a: Dataset Scanning started ===")
        results: List[DatasetScanResult] = []

        for dataset_cfg in self.config.datasets:
            if not dataset_cfg.path.exists():
                self.logger.warning(f"{dataset_cfg.name}: path does not exist, skipping scan -> {dataset_cfg.path}")
                results.append(DatasetScanResult(dataset_cfg.name, dataset_cfg.path))
                continue

            self.logger.info(
                f"Scanning dataset '{dataset_cfg.name}' at {dataset_cfg.path} "
                f"(adapter: {dataset_cfg.adapter_kind})"
            )
            result = self._scan_dataset(dataset_cfg.name, dataset_cfg.path)
            results.append(result)
            self.logger.info(
                f"{dataset_cfg.name}: {len(result.folders)} normalized class label(s), "
                f"{result.total_images} total image(s)"
            )

        self._write_report(results)
        self.logger.info("=== PSD-HP Stage 2a: Dataset Scanning finished ===")
        return results

    def _scan_dataset(self, name: str, path: Path) -> DatasetScanResult:
        result = DatasetScanResult(dataset_name=name, dataset_path=path)
        dataset_cfg = next(d for d in self.config.datasets if d.name == name)

        discovered_folders = discover_dataset(self.config, dataset_cfg)
        for discovered in discovered_folders:
            scanned = ScannedFolder(
                dataset_name=name,
                folder_name=discovered.normalized_label,
                images=discovered.images,
            )
            result.folders.append(scanned)

            versions = sorted(set(img.dataset_version for img in discovered.images))
            splits = sorted(set(img.original_split for img in discovered.images))
            source_types = sorted(set(img.source_type for img in discovered.images))
            if len(versions) > 1 or "train" in splits or "augmented" in source_types:
                self.logger.info(
                    f"{name}/{discovered.normalized_label}: merged from version(s)={versions}, "
                    f"split(s)={splits}, source_type(s)={source_types} "
                    f"({discovered.image_count} image(s) total)"
                )

        return result

    def _write_report(self, results: List[DatasetScanResult]) -> None:
        """Write reports/scan_report.xlsx — one row per (dataset, normalized label)."""
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "scan_report.xlsx"

        rows = []
        for result in results:
            for folder in result.folders:
                versions = sorted(set(img.dataset_version for img in folder.images))
                splits = sorted(set(img.original_split for img in folder.images))
                source_types = sorted(set(img.source_type for img in folder.images))
                rows.append({
                    "dataset": folder.dataset_name,
                    "normalized_label": folder.folder_name,
                    "image_count": folder.image_count,
                    "dataset_versions": ", ".join(versions),
                    "original_splits": ", ".join(splits),
                    "source_types": ", ".join(source_types),
                })

        df = pd.DataFrame(rows, columns=[
            "dataset", "normalized_label", "image_count",
            "dataset_versions", "original_splits", "source_types",
        ])
        df.to_excel(report_path, index=False, sheet_name="scan_results")
        self.logger.info(f"Scan report written to {report_path} ({len(rows)} row(s))")
