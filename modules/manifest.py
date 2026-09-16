"""
modules/manifest.py

PSD-HP Reproducibility Artifacts.

Writes two files that together let anyone reproduce exactly what a
given output/ folder contains without re-reading config.py's source:

  - reports/config_snapshot.json: every path, threshold, ratio, and
    seed PSDConfig held at run time.
  - reports/dataset_manifest.json: framework version, run timestamp,
    which datasets were used, the final per-class image counts, and
    a SHA-256 hash of metadata.csv — so a specific harmonized dataset
    can be identified and verified unambiguously (e.g. "does the
    dataset I trained on match the one in this manifest") without
    re-hashing every individual image.

Design note: the brief's example uses config_snapshot.yaml. This
implementation writes JSON instead. YAML and JSON carry identical
information here — the only practical difference is a PyYAML
dependency this snapshot doesn't need to justify, since nothing about
reproducibility requires YAML specifically (see README for the full
reasoning). Anyone needing YAML can convert this file with one line
of PyYAML; going the other direction would cost a dependency PSD-HP
does not otherwise require.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from config.config import PSDConfig
from utils.logger import get_module_logger

PSD_HP_VERSION = "1.0.0"


class ManifestGenerator:
    """Writes config_snapshot.json and dataset_manifest.json."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("manifest", config.logs_dir, config.log_level)

    def write_config_snapshot(self) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.config.reports_dir / "config_snapshot.json"

        snapshot: Dict = {
            "datasets": [
                {"name": d.name, "path": str(d.path), "description": d.description}
                for d in self.config.datasets
            ],
            "target_classes": self.config.target_classes,
            "supported_extensions": self.config.supported_extensions,
            "image_size": self.config.image_size,
            "min_image_dimension": self.config.min_image_dimension,
            "duplicate_hash_threshold": self.config.duplicate_hash_threshold,
            "blur_variance_threshold": self.config.blur_variance_threshold,
            "random_seed": self.config.random_seed,
            "train_ratio": self.config.train_ratio,
            "val_ratio": self.config.val_ratio,
            "test_ratio": self.config.test_ratio,
            "log_level": self.config.log_level,
        }

        snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        self.logger.info(f"Config snapshot written to {snapshot_path}")

    def write_dataset_manifest(self, per_class_final_counts: Dict[str, int]) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.reports_dir / "dataset_manifest.json"
        metadata_csv = self.config.reports_dir / "metadata.csv"

        metadata_hash = self._sha256_of_file(metadata_csv) if metadata_csv.exists() else None

        manifest: Dict = {
            "framework": "PSD-HP",
            "framework_version": PSD_HP_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "datasets_used": [d.name for d in self.config.datasets],
            "final_image_counts_per_class": per_class_final_counts,
            "final_total_images": sum(per_class_final_counts.values()),
            "metadata_csv_sha256": metadata_hash,
        }

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.logger.info(f"Dataset manifest written to {manifest_path}")

    @staticmethod
    def _sha256_of_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
