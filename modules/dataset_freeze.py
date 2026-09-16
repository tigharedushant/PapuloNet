"""
modules/dataset_freeze.py

AEF-CRC Stage: Dataset Freeze.

Why this exists: AEF-CRC is about to run many expensive, hours-long
experiments (EfficientNet training, feature extraction, GA feature
selection) against output/06_final_split/. If that directory were to
silently change between two experiments -- someone re-runs main.py
with a different random_seed, adds a dataset, or the splitter.py fix
below gets applied to some but not all of a team's machines -- every
downstream metric becomes incomparable without anyone necessarily
noticing. This module creates one signed snapshot of exactly what
"the dataset" means at the moment AEF-CRC starts using it, and gives
every later phase a cheap way to verify against that snapshot before
touching the data.

This module is read-only with respect to PSD-HP's outputs. It never
modifies output/06_final_split/ or any PSD-HP report; it only reads
them and writes its own freeze file under config.aef_crc_reports_dir.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config.config import PSDConfig
from utils.logger import get_module_logger

_SPLITS = ("train", "val", "test")
_FREEZE_FILENAME = "dataset_freeze.json"


@dataclass
class FreezeMismatch:
    kind: str    # "FILE_ADDED" | "FILE_REMOVED" | "FILE_CHANGED"
    detail: str


@dataclass
class FreezeVerifyResult:
    matches: bool
    mismatches: List[FreezeMismatch] = field(default_factory=list)
    frozen_at: Optional[str] = None
    frozen_file_count: int = 0
    current_file_count: int = 0


class DatasetFreezer:
    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("dataset_freeze", config.logs_dir, config.log_level)

    def _freeze_path(self) -> Path:
        return self.config.aef_crc_reports_dir / _FREEZE_FILENAME

    def _scan_current(self) -> Dict[str, str]:
        """Maps relative path (split/class/filename) -> SHA-256 hex digest, for
        every file currently under config.final_dir. Full-content hashing, not
        just size/mtime -- mtime is not trustworthy across a copy/clone/CI checkout."""
        files: Dict[str, str] = {}
        for split in _SPLITS:
            for target_class in self.config.target_classes:
                split_dir = self.config.final_dir / split / target_class
                if not split_dir.exists():
                    continue
                for img_path in sorted(split_dir.iterdir()):
                    if not img_path.is_file():
                        continue
                    rel = f"{split}/{target_class}/{img_path.name}"
                    files[rel] = self._sha256(img_path)
        return files

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def freeze(self, overwrite: bool = False) -> Path:
        """Write a new freeze snapshot. Refuses to silently overwrite an
        existing freeze unless overwrite=True -- freezing is meant to be a
        deliberate act, not something that happens by accident on a re-run."""
        freeze_path = self._freeze_path()
        if freeze_path.exists() and not overwrite:
            self.logger.info(
                f"Freeze already exists at {freeze_path} -- not overwriting "
                f"(pass overwrite=True to intentionally re-freeze after a real dataset change)."
            )
            return freeze_path

        files = self._scan_current()
        if not files:
            self.logger.error(
                f"No files found under {self.config.final_dir} -- nothing to freeze. "
                f"Run main.py (PSD-HP) to produce a final split first."
            )
            raise FileNotFoundError(f"No final-split files found under {self.config.final_dir}")

        # One aggregate hash over all individual hashes, sorted by relative
        # path -- a single value a later phase can compare cheaply without
        # re-reading every field, before falling back to the full per-file
        # diff in verify() if that quick check fails.
        aggregate = hashlib.sha256()
        for rel_path in sorted(files):
            aggregate.update(f"{rel_path}:{files[rel_path]}".encode("utf-8"))

        # Count classes and splits
        class_counts = {}
        split_counts = {}
        for rel_path in files:
            parts = rel_path.split("/")
            if len(parts) >= 2:
                sp, cls = parts[0], parts[1]
                split_counts[sp] = split_counts.get(sp, 0) + 1
                class_counts[cls] = class_counts.get(cls, 0) + 1

        meta_path = self.config.reports_dir / "metadata.csv"
        meta_hash = self._sha256(meta_path) if meta_path.exists() else None

        snapshot = {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "final_dir": str(self.config.final_dir),
            "random_seed": self.config.random_seed,
            "train_ratio": self.config.train_ratio,
            "val_ratio": self.config.val_ratio,
            "test_ratio": self.config.test_ratio,
            "target_classes": self.config.target_classes,
            "file_count": len(files),
            "class_counts": class_counts,
            "split_counts": split_counts,
            "metadata_csv_sha256": meta_hash,
            "aggregate_hash": aggregate.hexdigest(),
            "hashing_algorithm": "SHA-256",
            "hash_scope_ordering": "Alphabetically sorted relative paths (split/class/filename) with '{rel_path}:{file_sha256}'",
            "files": files,
        }

        self.config.aef_crc_reports_dir.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        self.logger.info(f"Dataset frozen: {len(files)} file(s), aggregate hash {snapshot['aggregate_hash'][:16]}... -> {freeze_path}")
        return freeze_path

    def verify(self) -> FreezeVerifyResult:
        """Compares the CURRENT final_split/ contents against the last freeze.
        Every later AEF-CRC phase should call this before training/extraction
        and abort (not just warn) on a mismatch -- see Phase 2 runner."""
        freeze_path = self._freeze_path()
        if not freeze_path.exists():
            self.logger.error(f"No freeze file at {freeze_path} -- call DatasetFreezer.freeze() first.")
            return FreezeVerifyResult(matches=False, mismatches=[
                FreezeMismatch("NO_FREEZE", "No freeze file exists yet.")
            ])

        snapshot = json.loads(freeze_path.read_text(encoding="utf-8"))
        frozen_files: Dict[str, str] = snapshot["files"]
        current_files = self._scan_current()

        mismatches: List[FreezeMismatch] = []
        for rel_path in sorted(set(current_files) - set(frozen_files)):
            mismatches.append(FreezeMismatch("FILE_ADDED", rel_path))
        for rel_path in sorted(set(frozen_files) - set(current_files)):
            mismatches.append(FreezeMismatch("FILE_REMOVED", rel_path))
        for rel_path in sorted(set(current_files) & set(frozen_files)):
            if current_files[rel_path] != frozen_files[rel_path]:
                mismatches.append(FreezeMismatch("FILE_CHANGED", rel_path))

        result = FreezeVerifyResult(
            matches=len(mismatches) == 0,
            mismatches=mismatches,
            frozen_at=snapshot.get("frozen_at"),
            frozen_file_count=snapshot.get("file_count", 0),
            current_file_count=len(current_files),
        )
        if result.matches:
            self.logger.info(f"Freeze verification PASSED: current final_split/ matches the {result.frozen_at} freeze exactly.")
        else:
            self.logger.error(
                f"Freeze verification FAILED: {len(mismatches)} mismatch(es) against the {result.frozen_at} freeze. "
                f"The dataset changed since it was frozen -- do not train against it without re-freezing deliberately."
            )
        return result
