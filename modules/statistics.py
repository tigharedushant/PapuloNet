"""
modules/statistics.py

PSD-HP Stage 4c: Dataset Statistics.

Consumes the full set of ProvenanceRecords and computes the summary
numbers that matter most for AEF-CRC's downstream design decisions:
how many images per target class actually made it into the final
harmonized set, how much each of the three source datasets
contributed to each class, and the resulting class-imbalance ratio —
the number that directly motivates whether few-shot/meta-learning is
needed for the smallest classes.

"Final" here specifically means final_filename is set — i.e. the
image survived mapping, extraction, duplicate detection, quality
assessment, AND standardization. A record can exist (and appear in
metadata.csv) without being final — that is the entire point of
keeping every record, not just the survivors.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.provenance import ProvenanceRecord


@dataclass
class DatasetStatisticsSummary:
    per_class_counts: Dict[str, int] = field(default_factory=dict)
    per_class_per_source_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    imbalance_ratio: float = 0.0
    smallest_class: str = ""
    largest_class: str = ""


class DatasetStatistics:
    """Computes and reports final class-balance and source-contribution statistics."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("statistics", config.logs_dir, config.log_level)

    def compute(self, records: List[ProvenanceRecord]) -> DatasetStatisticsSummary:
        self.logger.info("=== PSD-HP Stage 4c: Dataset Statistics started ===")
        summary = DatasetStatisticsSummary()
        final_records = [r for r in records if r.final_filename is not None]
        source_names = sorted(set(r.source_dataset for r in records))

        for target_class in self.config.target_classes:
            class_records = [r for r in final_records if r.mapped_class == target_class]
            summary.per_class_counts[target_class] = len(class_records)
            summary.per_class_per_source_counts[target_class] = {
                src: sum(1 for r in class_records if r.source_dataset == src)
                for src in source_names
            }

        nonzero_counts = {c: n for c, n in summary.per_class_counts.items() if n > 0}
        if nonzero_counts:
            summary.largest_class = max(nonzero_counts, key=nonzero_counts.get)
            summary.smallest_class = min(nonzero_counts, key=nonzero_counts.get)
            summary.imbalance_ratio = round(
                nonzero_counts[summary.largest_class] / nonzero_counts[summary.smallest_class], 2
            )
            self.logger.info(
                f"Class imbalance ratio: {summary.imbalance_ratio}x "
                f"(largest: {summary.largest_class}={nonzero_counts[summary.largest_class]}, "
                f"smallest: {summary.smallest_class}={nonzero_counts[summary.smallest_class]})"
            )
            if summary.imbalance_ratio >= 3.0:
                self.logger.warning(
                    f"Imbalance ratio {summary.imbalance_ratio}x is significant — "
                    f"consider class weighting, oversampling, or few-shot handling "
                    f"for '{summary.smallest_class}'. PSD-HP reports this imbalance "
                    f"only; AEF-CRC's training pipeline is responsible for handling it."
                )
        else:
            self.logger.error("No images reached the final harmonized set for any target class")

        self._write_report(summary, source_names)
        self._write_source_contribution_report(records, source_names)
        self._write_class_balance_report(records, source_names)
        self.logger.info("=== PSD-HP Stage 4c: Dataset Statistics finished ===")
        return summary

    def _write_report(self, summary: DatasetStatisticsSummary, source_names: List[str]) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "dataset_statistics.xlsx"

        rows = []
        for target_class, total in summary.per_class_counts.items():
            row = {"target_class": target_class, "total": total}
            row.update(summary.per_class_per_source_counts.get(target_class, {}))
            rows.append(row)

        df = pd.DataFrame(rows, columns=["target_class", "total", *source_names])

        overview = pd.DataFrame([{
            "largest_class": summary.largest_class,
            "smallest_class": summary.smallest_class,
            "imbalance_ratio": summary.imbalance_ratio,
        }])

        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="per_class_counts", index=False)
            overview.to_excel(writer, sheet_name="overview", index=False)

        self.logger.info(f"Dataset statistics written to {report_path}")

    def _write_source_contribution_report(self, records: List[ProvenanceRecord], source_names: List[str]) -> None:
        """reports/source_contribution_report.csv — per target class, how many final
        images each source dataset contributed, exactly the breakdown needed for a
        thesis methods section."""
        report_path = self.config.reports_dir / "source_contribution_report.csv"
        final_records = [r for r in records if r.final_filename is not None]

        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["target_class", "source_dataset", "final_image_count"])
            for target_class in self.config.target_classes:
                total = 0
                for src in source_names:
                    count = sum(1 for r in final_records if r.mapped_class == target_class and r.source_dataset == src)
                    if count > 0:
                        writer.writerow([target_class, src, count])
                    total += count
                writer.writerow([target_class, "TOTAL", total])

        self.logger.info(f"Source contribution report written to {report_path}")

    def _write_class_balance_report(self, records: List[ProvenanceRecord], source_names: List[str]) -> None:
        """reports/class_balance_report.csv — full accounting per target class:
        original (extracted) images, duplicates removed, quality-rejected, and
        final count. PSD-HP reports imbalance; it never corrects it by
        oversampling/undersampling/deleting valid images."""
        report_path = self.config.reports_dir / "class_balance_report.csv"

        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["target_class", "extracted", "duplicates_removed", "quality_rejected", "final"])
            for target_class in self.config.target_classes:
                class_records = [r for r in records if r.mapped_class == target_class]
                extracted = len(class_records)
                duplicates_removed = sum(1 for r in class_records if r.duplicate_status.startswith("DUPLICATE_OF"))
                quality_rejected = sum(1 for r in class_records if r.quality_status.startswith("FAIL"))
                final = sum(1 for r in class_records if r.final_filename is not None)
                writer.writerow([target_class, extracted, duplicates_removed, quality_rejected, final])

        self.logger.info(f"Class balance report written to {report_path}")
