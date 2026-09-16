"""
modules/metadata.py

PSD-HP Stage 4b: Metadata Generation.

Produces reports/metadata.csv — one row per image PSD-HP ever
extracted (including ambiguous-review and rejected images, so nothing
disappears from the record even if it never reached the final
harmonized set), with complete lineage: source dataset, original raw
folder, original filename, original path, mapped class, duplicate
status, quality status, and — for images that made it all the way
through — the final PSD-ID-based filename and dimensions.

This stage does no inference of its own. Every field it writes was
already recorded by the stage that actually knew it (extractor.py at
copy time, duplicate_detector.py and quality_assessor.py as each
image was checked, standardizer.py once resizing finished). That is
a deliberate architectural choice: a metadata-generation stage should
serialize known facts, not reconstruct them.
"""

from __future__ import annotations

import csv
from typing import List

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.provenance import ProvenanceTracker, ProvenanceRecord

_FIELDNAMES = [
    "psd_id", "source_dataset", "dataset_version", "original_split", "source_type",
    "original_class", "original_filename", "original_path",
    "mapped_class", "review_status", "duplicate_status", "quality_status",
    "final_filename", "final_width", "final_height",
]


class MetadataGenerator:
    """Serializes every ProvenanceRecord the pipeline has accumulated into metadata.csv."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("metadata", config.logs_dir, config.log_level)

    def generate(self, tracker: ProvenanceTracker) -> List[ProvenanceRecord]:
        self.logger.info("=== PSD-HP Stage 4b: Metadata Generation started ===")
        records = tracker.all_records()
        self._write_csv(records)
        self.logger.info(f"=== PSD-HP Stage 4b: Metadata Generation finished ({len(records)} record(s)) ===")
        return records

    def _write_csv(self, records: List[ProvenanceRecord]) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "metadata.csv"

        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            for r in sorted(records, key=lambda r: r.psd_id):
                writer.writerow(r.to_dict())

        self.logger.info(f"Metadata CSV written to {report_path}")
