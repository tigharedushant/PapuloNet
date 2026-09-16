"""
modules/integrity.py

PSD-HP Integrity Verification.

Answers one question with a hard, auditable check: did any image
disappear between stages without being accounted for somewhere?
Every ProvenanceRecord created by extractor.py must end up in
exactly one of four terminal states by the end of the run:

  - FINAL:     final_filename is set — made it all the way through
  - REJECTED:  quality_status starts with "FAIL"
  - DUPLICATE: duplicate_status starts with "DUPLICATE_OF"
  - REVIEW:    review_status is True (ambiguous/unknown, never a
               candidate for final inclusion in the first place)

If any record is in none of these states, or a record satisfies more
than one in a way that doesn't add up (e.g. a rejected image that
also has a final_filename), that is treated as a data-integrity
failure and logged as an ERROR — "no image should disappear
silently" is enforced here as a hard check, not a hope.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import List

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.provenance import ProvenanceRecord


@dataclass
class IntegrityResult:
    total_records: int = 0
    final_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    review_count: int = 0
    unaccounted: List[str] = field(default_factory=list)  # psd_ids in no terminal state
    conflicting: List[str] = field(default_factory=list)   # psd_ids in >1 terminal state

    @property
    def passed(self) -> bool:
        return not self.unaccounted and not self.conflicting


class IntegrityVerifier:
    """Verifies every extracted image is accounted for in exactly one terminal state."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("integrity", config.logs_dir, config.log_level)

    def verify(self, records: List[ProvenanceRecord]) -> IntegrityResult:
        self.logger.info("=== PSD-HP Integrity Verification started ===")
        result = IntegrityResult(total_records=len(records))

        for r in records:
            states = []
            if r.final_filename is not None:
                states.append("FINAL")
            if r.quality_status.startswith("FAIL"):
                states.append("REJECTED")
            if r.duplicate_status.startswith("DUPLICATE_OF"):
                states.append("DUPLICATE")
            if r.review_status:
                states.append("REVIEW")

            if len(states) == 0:
                result.unaccounted.append(r.psd_id)
                self.logger.error(f"UNACCOUNTED image: {r.psd_id} is in no terminal state")
            elif len(states) > 1:
                result.conflicting.append(r.psd_id)
                self.logger.error(f"CONFLICTING states for {r.psd_id}: {states}")
            else:
                {"FINAL": "final_count", "REJECTED": "rejected_count",
                 "DUPLICATE": "duplicate_count", "REVIEW": "review_count"}
                if states[0] == "FINAL":
                    result.final_count += 1
                elif states[0] == "REJECTED":
                    result.rejected_count += 1
                elif states[0] == "DUPLICATE":
                    result.duplicate_count += 1
                elif states[0] == "REVIEW":
                    result.review_count += 1

        if result.passed:
            self.logger.info(
                f"Integrity check PASSED: {result.total_records} scanned = "
                f"{result.final_count} final + {result.rejected_count} rejected + "
                f"{result.duplicate_count} duplicate + {result.review_count} review"
            )
        else:
            self.logger.error(
                f"Integrity check FAILED: {len(result.unaccounted)} unaccounted, "
                f"{len(result.conflicting)} conflicting — see integrity_report.csv"
            )

        self._write_report(records, result)
        self.logger.info("=== PSD-HP Integrity Verification finished ===")
        return result

    def _write_report(self, records: List[ProvenanceRecord], result: IntegrityResult) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "integrity_report.csv"

        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_records", result.total_records])
            writer.writerow(["final", result.final_count])
            writer.writerow(["rejected", result.rejected_count])
            writer.writerow(["duplicate", result.duplicate_count])
            writer.writerow(["review", result.review_count])
            writer.writerow(["unaccounted", len(result.unaccounted)])
            writer.writerow(["conflicting", len(result.conflicting)])
            writer.writerow(["check_passed", result.passed])
            for psd_id in result.unaccounted:
                writer.writerow(["UNACCOUNTED_ID", psd_id])
            for psd_id in result.conflicting:
                writer.writerow(["CONFLICTING_ID", psd_id])

        self.logger.info(f"Integrity report written to {report_path}")
