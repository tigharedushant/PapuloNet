"""
modules/extractor.py

PSD-HP Stage 3a: Target Class Extraction.

Consumes mapper.py's decisions and physically copies (never moves —
the raw datasets are left untouched) every image from a resolved
label into a class-organized staging area under
output/01_extracted/<TargetClass>/.

Labels resolved as EXCLUDE_NOT_TARGET or UNKNOWN are skipped
entirely. Labels resolved as AMBIGUOUS_MERGED are copied into
output/02_ambiguous_review/<dataset>_<label>/ instead of any target
class folder — PSD-HP does not guess how to split a merged folder
(e.g. DermNet's combined Psoriasis/Lichen Planus folder).

Each MappingDecision now carries its own List[DiscoveredImage] (from
the adapter layer via scanner.py/mapper.py), so extraction no longer
re-globs a folder path — it iterates exactly the images the adapter
already identified, each with its real dataset_version, original
split, and source_type (original vs. augmented) intact. This is
where PSD-HP's provenance tracking begins: every copied file is
assigned a globally unique PSD ID and registered with full lineage
before extraction returns.

SkinDisNet's Augmented/ images are filtered here according to
config.include_augmented ("true" / "false" / "auto"), rather than in
the adapter — the adapter's job is only to report what exists on
disk; the include/exclude *policy decision* belongs to the stage
that actually copies data forward.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.mapper import MappingDecision, EXCLUDE_NOT_TARGET, AMBIGUOUS_MERGED, UNKNOWN
from modules.provenance import ProvenanceTracker, ProvenanceRecord
from modules.adapters import DiscoveredImage

PSD_ID_SEPARATOR = "__"


@dataclass
class ExtractionResult:
    """Summary of how many files were extracted where."""
    per_class_counts: Dict[str, int] = field(default_factory=dict)
    ambiguous_count: int = 0
    skipped_count: int = 0
    augmented_excluded_count: int = 0
    extracted_files: List[Path] = field(default_factory=list)


def parse_psd_id(filename: str) -> str:
    """Recover a file's PSD ID from PSD-HP's own generated filename prefix."""
    return filename.split(PSD_ID_SEPARATOR, 1)[0]


class TargetClassExtractor:
    """Copies raw images into class-organized folders under output/01_extracted/,
    assigning each one a unique PSD ID and a provenance record as it goes."""

    def __init__(self, config: PSDConfig, tracker: ProvenanceTracker) -> None:
        self.config = config
        self.tracker = tracker
        self.logger = get_module_logger("extractor", config.logs_dir, config.log_level)

    def extract_all(self, decisions: List[MappingDecision]) -> ExtractionResult:
        self.logger.info("=== PSD-HP Stage 3a: Target Class Extraction started ===")
        result = ExtractionResult()

        for target_class in self.config.target_classes:
            (self.config.extracted_dir / target_class).mkdir(parents=True, exist_ok=True)
            result.per_class_counts[target_class] = 0

        for decision in decisions:
            if decision.resolved_class in (EXCLUDE_NOT_TARGET, UNKNOWN):
                result.skipped_count += decision.image_count
                self.logger.info(
                    f"Skipped '{decision.dataset_name}/{decision.raw_folder_name}' "
                    f"({decision.image_count} image(s), resolved={decision.resolved_class})"
                )
                continue

            images_to_copy, excluded_augmented = self._apply_augmentation_policy(decision)
            result.augmented_excluded_count += excluded_augmented

            if decision.resolved_class == AMBIGUOUS_MERGED:
                dest_dir = self.config.ambiguous_review_dir / f"{decision.dataset_name}_{decision.raw_folder_name}"
                count = self._copy_images(images_to_copy, decision, dest_dir, mapped_class="AMBIGUOUS_MERGED", review=True)
                result.ambiguous_count += count
                result.extracted_files.extend(dest_dir.glob("*"))
                self.logger.info(
                    f"Routed {count} AMBIGUOUS image(s) from "
                    f"'{decision.dataset_name}/{decision.raw_folder_name}' to ambiguous review"
                )
                continue

            dest_dir = self.config.extracted_dir / decision.resolved_class
            count = self._copy_images(images_to_copy, decision, dest_dir, mapped_class=decision.resolved_class, review=False)
            result.per_class_counts[decision.resolved_class] += count
            self.logger.info(
                f"Extracted {count} image(s) from "
                f"'{decision.dataset_name}/{decision.raw_folder_name}' -> {decision.resolved_class}"
                + (f" ({excluded_augmented} augmented image(s) excluded per policy)" if excluded_augmented else "")
            )

        self._write_report(result)
        self.logger.info("=== PSD-HP Stage 3a: Target Class Extraction finished ===")
        return result

    def _apply_augmentation_policy(self, decision: MappingDecision) -> tuple[List[DiscoveredImage], int]:
        """Filter out Augmented/ images per config.include_augmented. Non-augmented
        images (source_type == "original") are never affected by this policy."""
        originals = [img for img in decision.images if img.source_type == "original"]
        augmented = [img for img in decision.images if img.source_type == "augmented"]

        if not augmented:
            return decision.images, 0

        policy = self.config.include_augmented.lower()
        if policy == "true":
            include = True
        elif policy == "false":
            include = False
        elif policy == "auto":
            include = len(originals) < self.config.augmented_auto_threshold
            self.logger.info(
                f"'{decision.dataset_name}/{decision.raw_folder_name}': include_augmented=auto, "
                f"{len(originals)} original image(s) "
                f"({'below' if include else 'at/above'} threshold {self.config.augmented_auto_threshold}) "
                f"-> {'including' if include else 'excluding'} {len(augmented)} augmented image(s)"
            )
        else:
            self.logger.warning(f"Unrecognized include_augmented value '{self.config.include_augmented}', defaulting to 'false'")
            include = False

        if include:
            return decision.images, 0
        return originals, len(augmented)

    def _copy_images(self, images: List[DiscoveredImage], decision: MappingDecision,
                      dest_dir: Path, mapped_class: str, review: bool) -> int:
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = 0

        for img in images:
            psd_id = self.tracker.new_id()
            dest_name = f"{psd_id}{PSD_ID_SEPARATOR}{decision.dataset_name}_{img.path.stem}{img.path.suffix.lower()}"
            dest_path = dest_dir / dest_name

            try:
                shutil.copy2(img.path, dest_path)
            except OSError as exc:
                self.logger.error(f"Failed to copy {img.path} -> {dest_path}: {exc}")
                continue

            self.tracker.register(ProvenanceRecord(
                psd_id=psd_id,
                source_dataset=decision.dataset_name,
                original_class=decision.raw_folder_name,
                original_filename=img.path.name,
                original_path=str(img.path),
                mapped_class=mapped_class,
                review_status=review,
                dataset_version=img.dataset_version,
                original_split=img.original_split,
                source_type=img.source_type,
            ))
            copied += 1

        return copied

    def _write_report(self, result: ExtractionResult) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "extraction_report.txt"

        lines = ["PSD-HP Target Class Extraction Report", "=" * 40, ""]
        for target_class, count in result.per_class_counts.items():
            lines.append(f"{target_class}: {count} image(s) extracted")
        lines.append("")
        lines.append(f"Ambiguous (routed for manual review): {result.ambiguous_count} image(s)")
        lines.append(f"Skipped (EXCLUDE_NOT_TARGET or UNKNOWN): {result.skipped_count} image(s)")
        lines.append(f"Augmented images excluded per include_augmented policy: {result.augmented_excluded_count} image(s)")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Extraction report written to {report_path}")
