"""
modules/quality_assessor.py

PSD-HP Stage 3c: Quality Assessment.

Runs on whatever remains in output/01_extracted/<TargetClass>/ after
duplicate_detector.py has moved out near-duplicates. Every remaining
image is checked against two objective, OpenCV-computable criteria:

  1. Resolution: both width and height must be at least
     config.min_image_dimension pixels.
  2. Blur: the variance of the image's Laplacian must be at least
     config.blur_variance_threshold — a standard, widely-used OpenCV
     heuristic where a low variance indicates a lack of sharp edges
     (i.e. a blurry photo).

Images failing either check are moved (never deleted) to
output/04_low_quality_excluded/<TargetClass>/, logged with the
specific reason, and their permanent ProvenanceRecord is updated via
mark_quality() — the same record extraction created, not a new
parallel one.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.provenance import ProvenanceTracker
from modules.extractor import parse_psd_id


@dataclass
class QualityRecord:
    target_class: str
    file_name: str
    width: int
    height: int
    blur_variance: float
    passed: bool
    reason: str = ""


@dataclass
class QualityAssessmentResult:
    records: List[QualityRecord] = field(default_factory=list)

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self.records if r.passed)

    @property
    def total_failed(self) -> int:
        return sum(1 for r in self.records if not r.passed)


class QualityAssessor:
    """Resolution + blur quality checks on the extracted, de-duplicated images."""

    def __init__(self, config: PSDConfig, tracker: ProvenanceTracker) -> None:
        self.config = config
        self.tracker = tracker
        self.logger = get_module_logger("quality", config.logs_dir, config.log_level)

    def assess_all(self) -> QualityAssessmentResult:
        self.logger.info("=== PSD-HP Stage 3c: Quality Assessment started ===")
        result = QualityAssessmentResult()

        for target_class in self.config.target_classes:
            class_dir = self.config.extracted_dir / target_class
            if not class_dir.exists():
                continue
            self._assess_class(target_class, class_dir, result)

        self._write_report(result)
        self.logger.info(
            f"=== PSD-HP Stage 3c: Quality Assessment finished "
            f"({result.total_passed} passed, {result.total_failed} excluded) ==="
        )
        return result

    def _assess_class(self, target_class: str, class_dir: Path, result: QualityAssessmentResult) -> None:
        image_paths = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.config.supported_extensions
        )

        for img_path in image_paths:
            record = self._assess_image(target_class, img_path)
            result.records.append(record)

            psd_id = parse_psd_id(img_path.name)
            self.tracker.mark_quality(psd_id, record.passed, record.reason)

            if not record.passed:
                dest_dir = self.config.low_quality_dir / target_class
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / img_path.name
                try:
                    shutil.move(str(img_path), str(dest_path))
                    self.logger.warning(f"{target_class}/{img_path.name}: excluded ({record.reason})")
                except OSError as exc:
                    self.logger.error(f"Failed to move low-quality file {img_path}: {exc}")

    def _assess_image(self, target_class: str, img_path: Path) -> QualityRecord:
        image = cv2.imread(str(img_path))
        if image is None:
            return QualityRecord(
                target_class=target_class, file_name=img_path.name,
                width=0, height=0, blur_variance=0.0, passed=False,
                reason="unreadable by OpenCV",
            )

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        reasons = []
        if width < self.config.min_image_dimension or height < self.config.min_image_dimension:
            reasons.append(f"resolution {width}x{height} below minimum {self.config.min_image_dimension}px")
        if blur_variance < self.config.blur_variance_threshold:
            reasons.append(f"blur variance {blur_variance:.1f} below threshold {self.config.blur_variance_threshold}")

        return QualityRecord(
            target_class=target_class,
            file_name=img_path.name,
            width=width,
            height=height,
            blur_variance=round(blur_variance, 2),
            passed=not reasons,
            reason="; ".join(reasons),
        )

    def _write_report(self, result: QualityAssessmentResult) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "quality_report.csv"

        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["target_class", "file_name", "width", "height", "blur_variance", "passed", "reason"])
            for r in result.records:
                writer.writerow([r.target_class, r.file_name, r.width, r.height, r.blur_variance, r.passed, r.reason])

        self.logger.info(f"Quality report written to {report_path}")
