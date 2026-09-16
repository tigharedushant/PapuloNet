"""
modules/coverage.py

PSD-HP Stage 2c: Coverage Verification.

Consumes mapper.py's decisions and answers the question every stage
before this one has been building toward: for each of the four target
diseases, how many usable images does PSD-HP actually have, and from
how many of the three datasets. This is where a critically thin class
(e.g. a target disease found in only one dataset, or with a very low
image count) is surfaced explicitly, before extraction/standardization
time is spent on a dataset composition that downstream training would
struggle with anyway.

Single responsibility: coverage.py only counts and reports. It does
not decide to exclude a class or stop the pipeline — that judgment
call is left to the person reading coverage_matrix.xlsx, consistent
with PSD-HP never silently dropping data on its own authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.mapper import MappingDecision, AMBIGUOUS_MERGED

# A coverage count below this many images for a target class, summed
# across all datasets, is flagged as a CRITICAL warning in the report.
_CRITICAL_COVERAGE_THRESHOLD = 50


@dataclass
class CoverageCell:
    """Image count for one (target class, dataset) pair."""
    target_class: str
    dataset_name: str
    image_count: int


@dataclass
class CoverageSummary:
    """Full coverage matrix plus per-class totals and flags."""
    cells: List[CoverageCell] = field(default_factory=list)
    per_class_totals: Dict[str, int] = field(default_factory=dict)
    per_class_dataset_count: Dict[str, int] = field(default_factory=dict)
    critical_classes: List[str] = field(default_factory=list)
    zero_coverage_classes: List[str] = field(default_factory=list)


class CoverageVerifier:
    """Builds the dataset x target-class coverage matrix from mapping decisions."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("coverage", config.logs_dir, config.log_level)

    def verify(self, decisions: List[MappingDecision]) -> CoverageSummary:
        self.logger.info("=== PSD-HP Stage 2c: Coverage Verification started ===")
        summary = CoverageSummary()
        dataset_names = sorted(set(d.dataset_name for d in decisions))

        for target_class in self.config.target_classes:
            total = 0
            contributing_datasets = 0
            for dataset_name in dataset_names:
                count = sum(
                    d.image_count for d in decisions
                    if d.dataset_name == dataset_name and d.resolved_class == target_class
                )
                summary.cells.append(CoverageCell(target_class, dataset_name, count))
                total += count
                if count > 0:
                    contributing_datasets += 1

            summary.per_class_totals[target_class] = total
            summary.per_class_dataset_count[target_class] = contributing_datasets

            if total == 0:
                summary.zero_coverage_classes.append(target_class)
                self.logger.error(f"ZERO coverage for target class '{target_class}' across all datasets")
            elif total < _CRITICAL_COVERAGE_THRESHOLD:
                summary.critical_classes.append(target_class)
                self.logger.warning(
                    f"CRITICAL low coverage for '{target_class}': only {total} image(s) "
                    f"across {contributing_datasets} dataset(s) (threshold: {_CRITICAL_COVERAGE_THRESHOLD})"
                )
            else:
                self.logger.info(
                    f"'{target_class}': {total} image(s) across {contributing_datasets} dataset(s) — OK"
                )

        ambiguous = [d for d in decisions if d.resolved_class == AMBIGUOUS_MERGED]
        if ambiguous:
            ambiguous_images = sum(d.image_count for d in ambiguous)
            self.logger.warning(
                f"{ambiguous_images} image(s) across {len(ambiguous)} folder(s) are "
                f"AMBIGUOUS_MERGED and excluded from the coverage totals above — "
                f"they are not counted toward any target class until disambiguated"
            )

        self._write_report(summary, dataset_names)
        self.logger.info("=== PSD-HP Stage 2c: Coverage Verification finished ===")
        return summary

    def _write_report(self, summary: CoverageSummary, dataset_names: List[str]) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "coverage_matrix.xlsx"

        matrix: Dict[str, Dict[str, int]] = {}
        for cell in summary.cells:
            matrix.setdefault(cell.target_class, {})[cell.dataset_name] = cell.image_count

        df = pd.DataFrame.from_dict(matrix, orient="index", columns=dataset_names).fillna(0).astype(int)
        df["TOTAL"] = df.sum(axis=1)
        df["datasets_contributing"] = [summary.per_class_dataset_count[c] for c in df.index]
        df["status"] = df.index.map(
            lambda c: "ZERO COVERAGE" if c in summary.zero_coverage_classes
            else ("CRITICAL" if c in summary.critical_classes else "OK")
        )

        df.to_excel(report_path, sheet_name="coverage_matrix")
        self.logger.info(f"Coverage matrix written to {report_path}")
