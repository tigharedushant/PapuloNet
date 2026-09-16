"""
modules/report_generator.py

PSD-HP Stage 5b: Report Generation.

The final stage. Pulls together the results object every prior stage
already produced — validator's DatasetValidationResult list, scanner's
DatasetScanResult list, mapper's MappingDecision list, coverage's
CoverageSummary, extractor's ExtractionResult, duplicate_detector's
DuplicateDetectionResult, quality_assessor's QualityAssessmentResult,
standardizer's StandardizationResult, statistics' DatasetStatisticsSummary,
and splitter's SplitResult — into one human-readable PDF,
reports/harmonization_report.pdf.

This module does no computation of its own; it only formats results
that already exist. That is a deliberate single-responsibility
choice: if a number in the PDF is wrong, the bug is in the stage that
computed it, never in this file.
"""

from __future__ import annotations

from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.validator import DatasetValidationResult
from modules.scanner import DatasetScanResult
from modules.mapper import MappingDecision
from modules.coverage import CoverageSummary
from modules.extractor import ExtractionResult
from modules.duplicate_detector import DuplicateDetectionResult
from modules.quality_assessor import QualityAssessmentResult
from modules.standardizer import StandardizationResult
from modules.statistics import DatasetStatisticsSummary
from modules.splitter import SplitResult


class ReportGenerator:
    """Builds reports/harmonization_report.pdf from every stage's results."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("report_generator", config.logs_dir, config.log_level)
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="PSDHeading", fontSize=16, spaceAfter=12, textColor=colors.HexColor("#1F3864")))
        self.styles.add(ParagraphStyle(name="PSDSubheading", fontSize=12, spaceAfter=8, textColor=colors.HexColor("#1F3864")))

    def generate(
        self,
        validation_results: List[DatasetValidationResult],
        scan_results: List[DatasetScanResult],
        mapping_decisions: List[MappingDecision],
        coverage_summary: CoverageSummary,
        extraction_result: ExtractionResult,
        duplicate_result: DuplicateDetectionResult,
        quality_result: QualityAssessmentResult,
        standardization_result: StandardizationResult,
        statistics_summary: DatasetStatisticsSummary,
        split_result: SplitResult,
    ) -> None:
        self.logger.info("=== PSD-HP Stage 5b: Report Generation started ===")
        report_path = self.config.reports_dir / "harmonization_report.pdf"
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(str(report_path), pagesize=A4,
                                 topMargin=2 * cm, bottomMargin=2 * cm,
                                 leftMargin=2 * cm, rightMargin=2 * cm)
        story = []

        story += self._title_section()
        story += self._validation_section(validation_results)
        story += self._scan_section(scan_results)
        story += self._mapping_section(mapping_decisions)
        story += self._coverage_section(coverage_summary)
        story += self._extraction_section(extraction_result)
        story += self._duplicate_section(duplicate_result)
        story += self._quality_section(quality_result)
        story += self._standardization_section(standardization_result)
        story += self._statistics_section(statistics_summary)
        story += self._split_section(split_result)

        doc.build(story)
        self.logger.info(f"Harmonization report written to {report_path}")
        self.logger.info("=== PSD-HP Stage 5b: Report Generation finished ===")

    # ---- Section builders ----

    def _title_section(self) -> list:
        return [
            Paragraph("PSD-HP Harmonization Report", self.styles["Title"]),
            Paragraph("Papulosquamous Skin Disease Harmonization Protocol — AEF-CRC Project", self.styles["Normal"]),
            Spacer(1, 0.6 * cm),
        ]

    def _table(self, header: list, rows: list) -> Table:
        data = [header] + rows
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    def _validation_section(self, results: List[DatasetValidationResult]) -> list:
        rows = [[r.dataset_name, r.status, str(len(r.class_folders)), str(r.total_images), str(r.total_corrupted)]
                for r in results]
        return [
            Paragraph("Stage 1 — Dataset Validation", self.styles["PSDHeading"]),
            self._table(["Dataset", "Status", "Class Folders", "Total Images", "Corrupted"], rows),
            Spacer(1, 0.6 * cm),
        ]

    def _scan_section(self, results: List[DatasetScanResult]) -> list:
        rows = []
        for r in results:
            for folder in r.folders:
                versions = sorted(set(img.dataset_version for img in folder.images))
                splits = sorted(set(img.original_split for img in folder.images))
                rows.append([r.dataset_name, folder.folder_name, str(folder.image_count),
                             ", ".join(versions), ", ".join(splits)])
        return [
            Paragraph("Stage 2a — Dataset Scanning", self.styles["PSDHeading"]),
            self._table(["Dataset", "Normalized Label", "Images", "Versions", "Splits"], rows) if rows else
            Paragraph("No folders scanned.", self.styles["Normal"]),
            Spacer(1, 0.6 * cm),
        ]

    def _mapping_section(self, decisions: List[MappingDecision]) -> list:
        rows = [[d.dataset_name, d.raw_folder_name, d.resolved_class, str(d.image_count)] for d in decisions]
        unknown_count = sum(1 for d in decisions if d.resolved_class == "UNKNOWN")
        elements = [
            Paragraph("Stage 2b — Class Mapping", self.styles["PSDHeading"]),
            Paragraph(f"UNKNOWN (unmapped) folders: {unknown_count}", self.styles["Normal"]),
            Spacer(1, 0.2 * cm),
            self._table(["Dataset", "Raw Folder", "Resolved Class", "Images"], rows) if rows else
            Paragraph("No mapping decisions.", self.styles["Normal"]),
            Spacer(1, 0.6 * cm),
        ]
        return elements

    def _coverage_section(self, summary: CoverageSummary) -> list:
        rows = [[cls, str(total)] for cls, total in summary.per_class_totals.items()]
        flags = []
        if summary.zero_coverage_classes:
            flags.append(f"ZERO coverage: {', '.join(summary.zero_coverage_classes)}")
        if summary.critical_classes:
            flags.append(f"CRITICAL low coverage: {', '.join(summary.critical_classes)}")
        elements = [
            Paragraph("Stage 2c — Coverage Verification", self.styles["PSDHeading"]),
            self._table(["Target Class", "Total Images"], rows),
        ]
        for flag in flags:
            elements.append(Paragraph(f"<b>{flag}</b>", self.styles["Normal"]))
        elements.append(Spacer(1, 0.6 * cm))
        return elements

    def _extraction_section(self, result: ExtractionResult) -> list:
        rows = [[cls, str(count)] for cls, count in result.per_class_counts.items()]
        return [
            Paragraph("Stage 3a — Target Class Extraction", self.styles["PSDHeading"]),
            self._table(["Target Class", "Images Extracted"], rows),
            Paragraph(f"Ambiguous (routed for review): {result.ambiguous_count}", self.styles["Normal"]),
            Paragraph(f"Skipped (not a target class): {result.skipped_count}", self.styles["Normal"]),
            Spacer(1, 0.6 * cm),
        ]

    def _duplicate_section(self, result: DuplicateDetectionResult) -> list:
        return [
            Paragraph("Stage 3b — Duplicate Detection", self.styles["PSDHeading"]),
            Paragraph(f"Images scanned: {result.total_images_scanned}", self.styles["Normal"]),
            Paragraph(f"Duplicate groups found: {len(result.groups)}", self.styles["Normal"]),
            Paragraph(f"Duplicate images removed: {result.total_removed}", self.styles["Normal"]),
            Spacer(1, 0.6 * cm),
        ]

    def _quality_section(self, result: QualityAssessmentResult) -> list:
        return [
            Paragraph("Stage 3c — Quality Assessment", self.styles["PSDHeading"]),
            Paragraph(f"Passed: {result.total_passed}", self.styles["Normal"]),
            Paragraph(f"Excluded (low resolution or blurry): {result.total_failed}", self.styles["Normal"]),
            Spacer(1, 0.6 * cm),
        ]

    def _standardization_section(self, result: StandardizationResult) -> list:
        rows = [[cls, str(count)] for cls, count in result.per_class_counts.items()]
        return [
            Paragraph("Stage 4a — Image Standardization", self.styles["PSDHeading"]),
            self._table(["Target Class", "Images Standardized"], rows),
            Spacer(1, 0.6 * cm),
        ]

    def _statistics_section(self, summary: DatasetStatisticsSummary) -> list:
        rows = [[cls, str(count)] for cls, count in summary.per_class_counts.items()]
        return [
            Paragraph("Stage 4c — Dataset Statistics", self.styles["PSDHeading"]),
            self._table(["Target Class", "Final Count"], rows),
            Paragraph(
                f"Class imbalance ratio: {summary.imbalance_ratio}x "
                f"(largest: {summary.largest_class}, smallest: {summary.smallest_class})",
                self.styles["Normal"],
            ),
            Spacer(1, 0.6 * cm),
        ]

    def _split_section(self, result: SplitResult) -> list:
        rows = [[cls, str(c.train), str(c.val), str(c.test)] for cls, c in result.per_class_counts.items()]
        return [
            Paragraph("Stage 5a — Train / Validation / Test Split", self.styles["PSDHeading"]),
            self._table(["Target Class", "Train", "Val", "Test"], rows),
        ]
