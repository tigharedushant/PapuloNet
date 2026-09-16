"""
modules/standardizer.py

PSD-HP Stage 4a: Image Standardization.

Runs on whatever remains in output/01_extracted/<TargetClass>/ after
duplicate removal and quality filtering — i.e. every image that
survived both prior checks. Every remaining image is:

  1. Converted to RGB (source images may be grayscale, RGBA, or CMYK
     depending on which of the three source datasets they came from).
  2. Resized to config.image_size x config.image_size using
     LANCZOS resampling, preserving aspect ratio via letterbox
     padding rather than a distorting stretch (a plain resize would
     warp lesion shape and size — a genuinely bad idea for a
     classifier meant to learn morphology).
  3. Saved as JPEG (quality=95) into
     output/05_harmonized/<TargetClass>/, named after nothing but its
     PSD ID (e.g. "PSD_00000001.jpg") — the original filename is
     preserved in the image's ProvenanceRecord (and therefore
     metadata.csv), never in the final filename itself, exactly as
     required: two source datasets could otherwise collide on an
     identical original name.

Original images in output/01_extracted/ are left untouched; this
stage only ever reads from there and writes to
output/05_harmonized/, so re-running it with a different image_size
never requires re-running extraction, deduplication, or quality
assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageOps

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.provenance import ProvenanceTracker
from modules.extractor import parse_psd_id


@dataclass
class StandardizationResult:
    per_class_counts: Dict[str, int] = field(default_factory=dict)
    failed_files: List[str] = field(default_factory=list)


class ImageStandardizer:
    """Resizes (with aspect-ratio-preserving padding) and renames images to their final PSD ID."""

    def __init__(self, config: PSDConfig, tracker: ProvenanceTracker) -> None:
        self.config = config
        self.tracker = tracker
        self.logger = get_module_logger("standardizer", config.logs_dir, config.log_level)

    def standardize_all(self) -> StandardizationResult:
        self.logger.info("=== PSD-HP Stage 4a: Image Standardization started ===")
        result = StandardizationResult()

        for target_class in self.config.target_classes:
            source_dir = self.config.extracted_dir / target_class
            dest_dir = self.config.harmonized_dir / target_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            result.per_class_counts[target_class] = 0

            if not source_dir.exists():
                continue

            image_paths = sorted(
                f for f in source_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.config.supported_extensions
            )

            for img_path in image_paths:
                if self._standardize_image(img_path, dest_dir):
                    result.per_class_counts[target_class] += 1
                else:
                    result.failed_files.append(str(img_path))

            self.logger.info(f"{target_class}: {result.per_class_counts[target_class]} image(s) standardized")

        self._write_report(result)
        self.logger.info("=== PSD-HP Stage 4a: Image Standardization finished ===")
        return result

    def _standardize_image(self, img_path: Path, dest_dir: Path) -> bool:
        psd_id = parse_psd_id(img_path.name)
        try:
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                size = self.config.image_size
                # Aspect-ratio-preserving resize + letterbox pad, rather
                # than a distorting stretch to a square — preserves true
                # lesion shape/proportion.
                padded = ImageOps.pad(
                    rgb_img, (size, size),
                    method=Image.LANCZOS,
                    color=(0, 0, 0),
                    centering=(0.5, 0.5),
                )
                final_filename = f"{psd_id}.jpg"
                dest_path = dest_dir / final_filename
                padded.save(dest_path, "JPEG", quality=95)

            self.tracker.set_final(psd_id, final_filename, size, size)
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"Failed to standardize {img_path}: {exc}")
            return False

    def _write_report(self, result: StandardizationResult) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "standardization_report.txt"

        lines = ["PSD-HP Image Standardization Report", "=" * 40, ""]
        lines.append(
            f"Target size: {self.config.image_size}x{self.config.image_size} "
            f"(aspect-ratio-preserving pad), format: JPEG"
        )
        lines.append("")
        for target_class, count in result.per_class_counts.items():
            lines.append(f"{target_class}: {count} image(s) standardized")
        if result.failed_files:
            lines.append("")
            lines.append(f"Failed ({len(result.failed_files)}):")
            lines.extend(f"  - {f}" for f in result.failed_files)

        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Standardization report written to {report_path}")
