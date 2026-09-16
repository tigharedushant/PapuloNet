"""
modules/mapper.py

PSD-HP Stage 2b: Class Mapping.

Consumes scanner.py's real, on-disk folder inventory and
config/class_mapping.json's declared raw-label -> canonical-class
mapping, and produces one MappingDecision per scanned folder. This is
the only module in PSD-HP that reads class_mapping.json — every other
module works with MappingDecision objects, not raw JSON.

Every scanned folder resolves to exactly one of four outcomes:
  - a canonical target class ("Psoriasis", "Lichen_Planus", ...)
  - EXCLUDE_NOT_TARGET   (declared in the mapping file as irrelevant)
  - AMBIGUOUS_MERGED     (declared in the mapping file as mixing
                           multiple target diseases in one folder)
  - UNKNOWN              (folder exists on disk but class_mapping.json
                           says nothing about it at all)

UNKNOWN is the important safety case: rather than guessing or silently
dropping a folder the mapping file has never seen, mapper.py logs it
explicitly as a WARNING and lists it in mapping_report.txt, so a human
can add it to class_mapping.json before the next run.

Lookup normalization (root-cause fix): datasets without an
adapter-level normalization step (e.g. Curated32, which uses
GenericFlatAdapter) hand mapper.py whatever folder name string
actually exists on disk, verbatim. A real folder named "Lichen
Planus" will never equal the JSON key "Lichen_Planus" under a plain
dict lookup, and previously resolved silently to UNKNOWN — meaning
the whole class disappeared downstream with only a WARNING log to
explain why. mapper.py now normalizes both the JSON's keys and the
incoming folder name (lowercase, whitespace/hyphens collapsed to a
single underscore) before comparing, so this class of mismatch no
longer causes silent data loss. The *raw* folder name is still what
gets logged and reported everywhere else — only the comparison itself
is normalized.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.scanner import DatasetScanResult
from modules.adapters import DiscoveredImage

EXCLUDE_NOT_TARGET = "EXCLUDE_NOT_TARGET"
AMBIGUOUS_MERGED = "AMBIGUOUS_MERGED"
UNKNOWN = "UNKNOWN"

_WHITESPACE_OR_SEPARATOR = re.compile(r"[\s_\-]+")


def normalize_key(name: str) -> str:
    """Case/whitespace/separator-insensitive form used only for matching
    class_mapping.json keys against real folder names — never used for
    display or logging, which always use the original string."""
    return _WHITESPACE_OR_SEPARATOR.sub("_", name.strip()).casefold()


@dataclass
class MappingDecision:
    """What one normalized (dataset, label) resolves to."""
    dataset_name: str
    raw_folder_name: str  # the normalized label scanner.py/adapters produced
    images: List[DiscoveredImage] = field(default_factory=list)
    resolved_class: str = UNKNOWN  # one of: target class name, EXCLUDE_NOT_TARGET, AMBIGUOUS_MERGED, UNKNOWN

    @property
    def image_count(self) -> int:
        return len(self.images)


class ClassMapper:
    """
    Applies config/class_mapping.json to scanner.py's output.

    Loads the mapping file once, validates that every value in it is
    either a declared target class or one of the two recognized
    special tokens (catches typos in the JSON itself), then resolves
    every scanned folder against it using a normalized-key lookup.
    """

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("mapper", config.logs_dir, config.log_level)
        self._raw_mapping: Dict[str, Dict[str, str]] = self._load_mapping_file()
        # dataset_name -> {normalize_key(raw_key): mapped_value}
        self._normalized_mapping: Dict[str, Dict[str, str]] = {
            dataset_name: {normalize_key(k): v for k, v in folder_map.items()}
            for dataset_name, folder_map in self._raw_mapping.items()
        }
        # normalize_key(dataset_name) -> mapped_value dict, so a config.py
        # DatasetConfig.name that differs from class_mapping.json's
        # top-level key only in case/spacing/underscores still resolves
        # correctly, instead of silently mapping the entire dataset to
        # UNKNOWN (the second, previously-unpatched half of the bug that
        # produced "01_extracted has the folders, statistics shows zero").
        self._dataset_key_lookup: Dict[str, str] = {
            normalize_key(dataset_name): dataset_name for dataset_name in self._raw_mapping
        }
        self._warn_on_dataset_name_mismatch()

    def _warn_on_dataset_name_mismatch(self) -> None:
        """Proactive check, run at startup rather than discovered after the
        fact: does every dataset configured in config.py have a matching
        entry in class_mapping.json, and vice versa? A mismatch here means
        every folder in that dataset will resolve to UNKNOWN, silently."""
        configured_names = {d.name for d in self.config.datasets}
        mapping_names = set(self._raw_mapping.keys())

        for name in configured_names:
            if normalize_key(name) not in self._dataset_key_lookup:
                self.logger.error(
                    f"STARTUP CHECK FAILED: dataset '{name}' is configured in config.py "
                    f"but has NO matching top-level entry in class_mapping.json "
                    f"(checked against: {sorted(mapping_names)}) — every folder in "
                    f"'{name}' will resolve to UNKNOWN. Fix the dataset name in one "
                    f"of the two files so they match."
                )
        for name in mapping_names:
            if normalize_key(name) not in {normalize_key(n) for n in configured_names}:
                self.logger.warning(
                    f"STARTUP CHECK: class_mapping.json has a top-level entry '{name}' "
                    f"that doesn't match any dataset configured in config.py "
                    f"(configured: {sorted(configured_names)}) — this section will never be used."
                )

    def _load_mapping_file(self) -> Dict[str, Dict[str, str]]:
        path = self.config.class_mapping_file
        if not path.exists():
            raise FileNotFoundError(
                f"class_mapping.json not found at {path}. "
                f"mapper.py cannot run without it."
            )

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        raw.pop("_meta", None)  # documentation block, not mapping data

        valid_values = set(self.config.target_classes) | {EXCLUDE_NOT_TARGET, AMBIGUOUS_MERGED}
        for dataset_name, folder_map in raw.items():
            for folder_name, mapped_value in folder_map.items():
                if mapped_value not in valid_values:
                    self.logger.error(
                        f"class_mapping.json: '{dataset_name}/{folder_name}' maps to "
                        f"'{mapped_value}', which is not a declared target class or a "
                        f"recognized special token ({EXCLUDE_NOT_TARGET}, {AMBIGUOUS_MERGED}). "
                        f"Fix this entry in class_mapping.json."
                    )

        return raw

    def map_all(self, scan_results: List[DatasetScanResult]) -> List[MappingDecision]:
        """Resolve every scanned folder to a class-mapping decision."""
        self.logger.info("=== PSD-HP Stage 2b: Class Mapping started ===")
        decisions: List[MappingDecision] = []
        unknown_count = 0

        for scan_result in scan_results:
            actual_key = self._dataset_key_lookup.get(normalize_key(scan_result.dataset_name))
            dataset_map = self._normalized_mapping.get(actual_key, {}) if actual_key else {}
            if actual_key is None:
                self.logger.warning(
                    f"'{scan_result.dataset_name}' has no entry at all in class_mapping.json "
                    f"(even after normalized matching) — every folder in it will resolve to UNKNOWN"
                )

            for folder in scan_result.folders:
                resolved = dataset_map.get(normalize_key(folder.folder_name), UNKNOWN)
                decisions.append(MappingDecision(
                    dataset_name=folder.dataset_name,
                    raw_folder_name=folder.folder_name,
                    images=folder.images,
                    resolved_class=resolved,
                ))

                if resolved == UNKNOWN:
                    unknown_count += 1
                    self.logger.warning(
                        f"UNKNOWN class mapping: '{folder.dataset_name}/{folder.folder_name}' "
                        f"({folder.image_count} images) is not declared in class_mapping.json "
                        f"— checked normalized key '{normalize_key(folder.folder_name)}' "
                        f"against {len(dataset_map)} known key(s) for this dataset"
                    )
                elif resolved == AMBIGUOUS_MERGED:
                    self.logger.warning(
                        f"AMBIGUOUS_MERGED: '{folder.dataset_name}/{folder.folder_name}' "
                        f"({folder.image_count} images) mixes multiple target diseases — "
                        f"routed to ambiguous review, not auto-assigned"
                    )
                else:
                    self.logger.info(
                        f"'{folder.dataset_name}/{folder.folder_name}' -> {resolved} "
                        f"({folder.image_count} images)"
                    )

        self._write_report(decisions, unknown_count)
        self.logger.info(
            f"=== PSD-HP Stage 2b: Class Mapping finished "
            f"({unknown_count} UNKNOWN folder(s) found) ==="
        )
        return decisions

    def _write_report(self, decisions: List[MappingDecision], unknown_count: int) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "mapping_report.txt"

        lines = ["PSD-HP Class Mapping Report", "=" * 40, ""]
        lines.append(f"Total folders resolved: {len(decisions)}")
        lines.append(f"UNKNOWN (not in class_mapping.json): {unknown_count}")
        lines.append("")

        for label in [*sorted(set(d.resolved_class for d in decisions))]:
            group = [d for d in decisions if d.resolved_class == label]
            lines.append(f"[{label}] — {len(group)} folder(s), {sum(g.image_count for g in group)} image(s)")
            for d in group:
                lines.append(f"    {d.dataset_name}/{d.raw_folder_name} ({d.image_count} images)")
            lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Mapping report written to {report_path}")
