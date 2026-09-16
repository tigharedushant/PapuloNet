"""
modules/provenance.py

Single source of truth for per-image lineage across the PSD-HP pipeline.

Architectural note (why this module exists): before this change,
metadata.py re-derived which source dataset an image came from by
parsing a filename prefix extractor.py had written earlier
(_infer_source_dataset). That is backwards — extractor.py already
knows the true source dataset, raw folder name, and original filename
at the moment it copies a file; making a later stage guess that
information back out of a string is fragile (breaks the instant a
dataset name contains an underscore) and duplicates knowledge that
should live in one place.

ProvenanceTracker fixes this: extractor.py creates one
ProvenanceRecord per copied file, at copy time, with full lineage.
Every later stage (duplicate_detector, quality_assessor,
standardizer) looks up that image's record by its PSD ID and updates
its own field — it never re-derives anything. metadata.py's job
shrinks to "serialize whatever ProvenanceTracker already knows,"
which is exactly what a metadata-generation stage should do.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProvenanceRecord:
    """Complete lineage for one image, from raw source to final harmonized file."""
    psd_id: str
    source_dataset: str
    original_class: str          # normalized label, e.g. "Seborrheic_Dermatitis"
    original_filename: str
    original_path: str
    mapped_class: str            # canonical target class, or the special review token
    review_status: bool = False  # True if routed to ambiguous/unknown review, not a target class

    # Real structural provenance, populated by the adapter layer via extractor.py.
    # Defaults keep this additive/backward-compatible with any record created
    # before this field existed.
    dataset_version: str = ""    # e.g. "SkinDisNet" vs "SkinDisNet_2"; == source_dataset if no version concept
    original_split: str = "n/a"  # "train" | "test" | "n/a"
    source_type: str = "original"  # "original" | "augmented"

    # Filled in by later stages; PENDING until that stage actually runs.
    duplicate_status: str = "PENDING"   # "UNIQUE" | "DUPLICATE_OF:<psd_id>" | "PENDING"
    quality_status: str = "PENDING"     # "PASS" | "FAIL: <reason>" | "PENDING"
    final_filename: Optional[str] = None
    final_width: Optional[int] = None
    final_height: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ProvenanceTracker:
    """
    In-memory registry of every ProvenanceRecord created during one
    pipeline run, keyed by PSD ID. Owns unique ID assignment so no
    other module needs to implement its own counter.
    """

    def __init__(self, id_prefix: str = "PSD", id_digits: int = 8) -> None:
        self._records: Dict[str, ProvenanceRecord] = {}
        self._counter = 0
        self._id_prefix = id_prefix
        self._id_digits = id_digits

    def new_id(self) -> str:
        """Assign and reserve the next globally unique PSD ID."""
        self._counter += 1
        return f"{self._id_prefix}_{self._counter:0{self._id_digits}d}"

    def register(self, record: ProvenanceRecord) -> None:
        if record.psd_id in self._records:
            raise ValueError(f"Duplicate PSD ID registered: {record.psd_id}")
        self._records[record.psd_id] = record

    def get(self, psd_id: str) -> ProvenanceRecord:
        return self._records[psd_id]

    def all_records(self) -> List[ProvenanceRecord]:
        return list(self._records.values())

    def records_for_class(self, mapped_class: str) -> List[ProvenanceRecord]:
        return [r for r in self._records.values() if r.mapped_class == mapped_class]

    def mark_duplicate(self, psd_id: str, kept_psd_id: str) -> None:
        self._records[psd_id].duplicate_status = f"DUPLICATE_OF:{kept_psd_id}"

    def mark_unique(self, psd_id: str) -> None:
        self._records[psd_id].duplicate_status = "UNIQUE"

    def mark_quality(self, psd_id: str, passed: bool, reason: str = "") -> None:
        self._records[psd_id].quality_status = "PASS" if passed else f"FAIL: {reason}"

    def set_final(self, psd_id: str, final_filename: str, width: int, height: int) -> None:
        record = self._records[psd_id]
        record.final_filename = final_filename
        record.final_width = width
        record.final_height = height
