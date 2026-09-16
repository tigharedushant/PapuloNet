"""
modules/duplicate_detector.py

PSD-HP Stage 3b: Duplicate Detection.

Runs on the class-organized output of extractor.py
(output/01_extracted/<TargetClass>/), computing a perceptual hash
(phash) for every image and grouping images whose hashes are within
config.duplicate_hash_threshold Hamming distance of each other. This
catches both exact duplicates and near-duplicates (recompressed,
resized, or lightly cropped copies of the same underlying photo) —
a real risk here specifically because the harmonized dataset draws
from three independently-curated sources that may themselves have
scraped overlapping public images.

Within each duplicate group, the first image (by sorted filename,
which sorts by PSD ID and is therefore extraction order) is kept in
place; every other member of the group is moved (never deleted) to
output/03_duplicates_removed/<TargetClass>/. The decision is recorded
in two places: duplicates_report.csv (this stage's own audit trail)
and, via ProvenanceTracker, the kept/removed image's permanent
provenance record — so a later report (e.g. metadata.csv) never has
to re-derive duplicate status from which folder a file currently
sits in.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import imagehash
from PIL import Image

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.provenance import ProvenanceTracker
from modules.extractor import parse_psd_id


@dataclass
class DuplicateGroup:
    """A set of images considered near-duplicates of each other."""
    target_class: str
    kept_file: Path
    removed_files: List[Path] = field(default_factory=list)


@dataclass
class CrossClassDuplicate:
    """A pair of images in DIFFERENT target classes whose perceptual
    hashes are within threshold -- reported for human review, never
    auto-resolved. A cross-class match is a different, more serious
    signal than a within-class one: it could mean genuine mislabeling
    upstream (the same photo filed under two different diseases by two
    different source datasets), which is not something this pipeline
    should ever decide on its own."""
    psd_id_a: str
    class_a: str
    file_a: str
    psd_id_b: str
    class_b: str
    file_b: str
    hamming_distance: int


@dataclass
class DuplicateDetectionResult:
    total_images_scanned: int = 0
    groups: List[DuplicateGroup] = field(default_factory=list)
    cross_class_duplicates: List[CrossClassDuplicate] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return sum(len(g.removed_files) for g in self.groups)


class DuplicateDetector:
    """Perceptual-hash duplicate detection over the extracted target-class folders."""

    def __init__(self, config: PSDConfig, tracker: ProvenanceTracker) -> None:
        self.config = config
        self.tracker = tracker
        self.logger = get_module_logger("duplicate_detector", config.logs_dir, config.log_level)

    def detect_all(self) -> DuplicateDetectionResult:
        self.logger.info("=== PSD-HP Stage 3b: Duplicate Detection started ===")
        result = DuplicateDetectionResult()

        for target_class in self.config.target_classes:
            class_dir = self.config.extracted_dir / target_class
            if not class_dir.exists():
                continue
            self._detect_within_class(target_class, class_dir, result)

        # Cross-class pass (item 3, Phase-4 review): runs AFTER within-class
        # detection so it only ever sees images that survived within-class
        # dedup -- purely additive, does not change anything about the
        # within-class behavior above, and never moves/deletes a file.
        self._detect_cross_class(result)

        # Every surviving, never-flagged image is explicitly UNIQUE, not
        # just implicitly "never mentioned" — this keeps
        # duplicate_status meaningful for every record, not just the
        # ones involved in a group.
        flagged_ids = {parse_psd_id(f.name) for g in result.groups for f in [g.kept_file, *g.removed_files]}
        for record in self.tracker.all_records():
            if record.psd_id not in flagged_ids and not record.review_status:
                self.tracker.mark_unique(record.psd_id)

        self._write_report(result)
        self.logger.info(
            f"=== PSD-HP Stage 3b: Duplicate Detection finished "
            f"({result.total_removed} duplicate(s) removed from {result.total_images_scanned} scanned) ==="
        )
        return result

    def _detect_within_class(self, target_class: str, class_dir: Path, result: DuplicateDetectionResult) -> None:
        image_paths = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.config.supported_extensions
        )
        result.total_images_scanned += len(image_paths)

        hashes: Dict[Path, imagehash.ImageHash] = {}
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    hashes[img_path] = imagehash.phash(img)
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"Could not hash {img_path}: {exc}")

        visited: set[Path] = set()
        threshold = self.config.duplicate_hash_threshold

        for path_a in image_paths:
            if path_a in visited or path_a not in hashes:
                continue

            group_members = [path_a]
            for path_b in image_paths:
                if path_b == path_a or path_b in visited or path_b not in hashes:
                    continue
                if hashes[path_a] - hashes[path_b] <= threshold:
                    group_members.append(path_b)

            if len(group_members) > 1:
                kept, *duplicates = sorted(group_members)
                kept_psd_id = parse_psd_id(kept.name)
                self.tracker.mark_unique(kept_psd_id)

                dest_dir = self.config.duplicates_removed_dir / target_class
                dest_dir.mkdir(parents=True, exist_ok=True)

                moved = []
                for dup_path in duplicates:
                    dup_psd_id = parse_psd_id(dup_path.name)
                    dest_path = dest_dir / dup_path.name
                    try:
                        shutil.move(str(dup_path), str(dest_path))
                        moved.append(dest_path)
                        self.tracker.mark_duplicate(dup_psd_id, kept_psd_id)
                    except OSError as exc:
                        self.logger.error(f"Failed to move duplicate {dup_path}: {exc}")

                result.groups.append(DuplicateGroup(target_class, kept, moved))
                self.logger.info(
                    f"{target_class}: kept '{kept.name}', removed {len(moved)} "
                    f"near-duplicate(s) (Hamming distance <= {threshold})"
                )

            visited.update(group_members)

    def _detect_cross_class(self, result: DuplicateDetectionResult) -> None:
        """Global pass across ALL target classes together. Report-only:
        never moves or deletes a file, and never touches ProvenanceTracker
        -- a cross-class match is a labeling-conflict signal for a human
        to resolve, not a duplicate-removal decision this pipeline should
        make unilaterally."""
        all_hashes: Dict[Path, tuple] = {}  # path -> (target_class, hash)
        for target_class in self.config.target_classes:
            class_dir = self.config.extracted_dir / target_class
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in self.config.supported_extensions:
                    continue
                try:
                    with Image.open(img_path) as img:
                        all_hashes[img_path] = (target_class, imagehash.phash(img))
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"Could not hash {img_path} for cross-class check: {exc}")

        threshold = self.config.duplicate_hash_threshold
        paths = list(all_hashes.keys())
        for i, path_a in enumerate(paths):
            class_a, hash_a = all_hashes[path_a]
            for path_b in paths[i + 1:]:
                class_b, hash_b = all_hashes[path_b]
                if class_a == class_b:
                    continue  # within-class already handled by _detect_within_class -- not duplicated here
                distance = hash_a - hash_b
                if distance <= threshold:
                    result.cross_class_duplicates.append(CrossClassDuplicate(
                        psd_id_a=parse_psd_id(path_a.name), class_a=class_a, file_a=path_a.name,
                        psd_id_b=parse_psd_id(path_b.name), class_b=class_b, file_b=path_b.name,
                        hamming_distance=distance,
                    ))

        if result.cross_class_duplicates:
            self.logger.warning(
                f"{len(result.cross_class_duplicates)} cross-class near-duplicate pair(s) found -- "
                f"see reports/cross_class_duplicates_report.csv. NOT auto-resolved; requires human review "
                f"(a cross-class match may indicate a genuine upstream labeling conflict)."
            )
        self._write_cross_class_report(result)

    def _write_cross_class_report(self, result: DuplicateDetectionResult) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "cross_class_duplicates_report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["psd_id_a", "class_a", "file_a", "psd_id_b", "class_b", "file_b", "hamming_distance"])
            for d in result.cross_class_duplicates:
                writer.writerow([d.psd_id_a, d.class_a, d.file_a, d.psd_id_b, d.class_b, d.file_b, d.hamming_distance])
        self.logger.info(f"Cross-class duplicates report written to {report_path}")

    def _write_report(self, result: DuplicateDetectionResult) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "duplicates_report.csv"

        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["target_class", "kept_psd_id", "kept_file", "removed_psd_id", "removed_file"])
            for group in result.groups:
                kept_psd_id = parse_psd_id(group.kept_file.name)
                for removed in group.removed_files:
                    writer.writerow([group.target_class, kept_psd_id, group.kept_file.name,
                                      parse_psd_id(removed.name), removed.name])

        self.logger.info(f"Duplicates report written to {report_path}")
