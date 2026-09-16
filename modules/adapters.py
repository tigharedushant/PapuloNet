"""
modules/adapters.py

The PSD-HP dataset adapter layer.

Responsibility (and ONLY responsibility): understand each dataset's
real on-disk structure and present a normalized, flat list of
(image, real-original-metadata) records to the rest of the pipeline.
Adapters do NOT decide what counts as a target class — that remains
mapper.py's job, unchanged, working off class_mapping.json exactly as
before. An adapter's normalized "label" is deliberately just the
disease name as found on disk (post-normalization of naming
variants); mapper.py still has the final say on target-class
inclusion/exclusion.

Why this module exists: scanner.py and validator.py both assume
dataset_root/<class_folder>/<images> — one level deep. That is true
for Curated32, but not for DermNet (dataset_root/{train,test}/<class>/),
SkinDisNet (dataset_root/{SkinDisNet,SkinDisNet_2}/{Preprocessed,
Augmented}/<class>/, with two different class-naming conventions
across the two versions), or Atlas ISIC2019/31-Classes
(dataset_root/{train,test}/<class>/ — same two-split shape as
DermNet, but every class folder already names exactly one disease,
so no per-image filename reclassification is needed for it). Rather
than teaching scanner.py and validator.py dataset-specific structural
knowledge directly (which would violate their single responsibility),
that knowledge lives here, in one place, behind one small interface.

No dataset's raw files are ever modified, moved, or renamed by any
adapter — discovery is read-only. Copying into PSD-HP's own
output/ directories remains extractor.py's job, unchanged.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config.config import PSDConfig, DatasetConfig
from utils.logger import get_module_logger

_FILENAME_SEPARATOR = re.compile(r"[\s_\-]+")


def _filename_normalize(s: str) -> str:
    """Case/whitespace/hyphen/underscore-insensitive form, used only for
    filename-prefix classification — never for display or logging."""
    return _FILENAME_SEPARATOR.sub("-", s.strip()).lower()


def classify_filename_by_prefix(filename_stem: str, target_classes: List[str]) -> Optional[str]:
    """
    Generic filename classifier: does this filename start with a
    normalized target class name? No per-filename lookup table — any
    filename beginning with a target class's own name (case/hyphen/
    whitespace-insensitive) is recognized automatically, including
    subtypes never explicitly listed anywhere (e.g. "psoriasis-scalp",
    "psoriasis-palms-soles", or any future "psoriasis-<anything>").

    Longest-name-first ordering is defensive: if target class names
    ever overlapped as prefixes of each other, this guarantees the
    more specific one wins. Not currently triggered by the 4 target
    diseases (none is a prefix of another), but costs nothing to keep
    correct as the class list grows.
    """
    normalized = _filename_normalize(filename_stem)
    for canonical in sorted(target_classes, key=len, reverse=True):
        prefix = _filename_normalize(canonical)
        if normalized.startswith(prefix):
            return canonical
    return None


@dataclass
class DiscoveredImage:
    """One image file, with the real structural metadata an adapter found for it."""
    path: Path
    dataset_version: str    # e.g. "SkinDisNet" vs "SkinDisNet_2"; == dataset name if no version concept
    original_split: str     # "train" | "test" | "n/a"
    source_type: str        # "original" | "augmented"; "original" for every dataset except SkinDisNet
    normalized_label: str   # canonical raw label used for class_mapping.json lookup
    raw_label_on_disk: str  # the literal folder name found on disk, pre-normalization


@dataclass
class DiscoveredFolder:
    """All images belonging to one normalized label within one dataset,
    regardless of how many real subfolders (splits/versions) they came from."""
    dataset_name: str
    normalized_label: str
    images: List[DiscoveredImage] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.images)


class DatasetAdapter:
    """Base adapter: understands nothing dataset-specific. Subclasses override discover()."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("adapters", config.logs_dir, config.log_level)

    def discover(self, dataset_cfg: DatasetConfig) -> List[DiscoveredFolder]:
        raise NotImplementedError


class GenericFlatAdapter(DatasetAdapter):
    """
    Default adapter for any dataset already shaped as
    dataset_root/<class_folder>/<images> — the assumption scanner.py
    and validator.py originally hardcoded. Used by Curated32, and by
    any future dataset that doesn't need special handling.
    """

    def discover(self, dataset_cfg: DatasetConfig) -> List[DiscoveredFolder]:
        folders: Dict[str, DiscoveredFolder] = {}
        if not dataset_cfg.path.exists():
            return []

        for class_dir in sorted(p for p in dataset_cfg.path.iterdir() if p.is_dir()):
            label = class_dir.name
            folder = folders.setdefault(label, DiscoveredFolder(dataset_cfg.name, label))
            for img_path in class_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in self.config.supported_extensions:
                    folder.images.append(DiscoveredImage(
                        path=img_path, dataset_version=dataset_cfg.name,
                        original_split="n/a", source_type="original",
                        normalized_label=label, raw_label_on_disk=label,
                    ))
        return list(folders.values())


class DermNetAdapter(DatasetAdapter):
    """
    Understands DermNet's real structure: dataset_root/{train,test}/<class>/.

    Both splits are merged into one DiscoveredFolder per class — the
    original train/test split is NOT preserved as a dataset split
    (per the brief: "Do not preserve the original train/test split"),
    but it IS preserved per-image as original_split, so provenance is
    never lost even though the split itself isn't reused downstream.

    Per-image filename reclassification (redesign): DermNet's merged
    folder ("Psoriasis pictures Lichen Planus and related diseases")
    was previously treated as entirely ambiguous, discarding every
    image in it to review. In reality many filenames already encode
    the true diagnosis (e.g. "psoriasis-scalp-90.jpg",
    "lichen-planus-12.jpg"). Every discovered image — not just those
    in the known-merged folder, since this adapter has no dependency
    on class_mapping.json and therefore no way to know in advance
    which raw folder is "the ambiguous one" — is checked against
    classify_filename_by_prefix(). A confident match reassigns that
    image's normalized_label directly to the canonical target class
    name, overriding its raw folder-derived label; no match leaves
    the image under its original raw-folder label exactly as before,
    so already-correctly-labelled folders are unaffected either way.
    This needs zero changes to class_mapping.json's existing entries
    beyond adding the four canonical class names as new DermNet keys
    (identity mappings) — the raw merged-folder entry mapping to
    AMBIGUOUS_MERGED stays exactly as it was, still catching whatever
    filenames could not be confidently classified.

    Writes reports/dermnet_filename_classification_report.csv logging
    every classification decision for full auditability.
    """

    _SPLIT_DIRS = ("train", "test")

    def discover(self, dataset_cfg: DatasetConfig) -> List[DiscoveredFolder]:
        folders: Dict[str, DiscoveredFolder] = {}
        classification_rows: List[dict] = []
        if not dataset_cfg.path.exists():
            return []

        found_any_split_dir = False
        for split in self._SPLIT_DIRS:
            split_dir = dataset_cfg.path / split
            if not split_dir.is_dir():
                continue
            found_any_split_dir = True

            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                raw_label = class_dir.name
                for img_path in class_dir.rglob("*"):
                    if not (img_path.is_file() and img_path.suffix.lower() in self.config.supported_extensions):
                        continue

                    matched_class = classify_filename_by_prefix(img_path.stem, self.config.target_classes)
                    effective_label = matched_class if matched_class is not None else raw_label

                    folder = folders.setdefault(effective_label, DiscoveredFolder(dataset_cfg.name, effective_label))
                    folder.images.append(DiscoveredImage(
                        path=img_path, dataset_version=dataset_cfg.name,
                        original_split=split, source_type="original",
                        normalized_label=effective_label, raw_label_on_disk=raw_label,
                    ))

                    classification_rows.append({
                        "original_filename": img_path.name,
                        "raw_folder": raw_label,
                        "detected_disease": matched_class or "",
                        "canonical_class": effective_label,
                        "decision": "RECOVERED" if matched_class is not None else "UNMATCHED_RETAINS_FOLDER_LABEL",
                    })

        if not found_any_split_dir:
            self.logger.warning(
                f"DermNetAdapter: no 'train' or 'test' subfolder found under {dataset_cfg.path} — "
                f"falling back to flat (dataset_root/<class>/) discovery. If your DermNet copy "
                f"really has no train/test split, this is expected; otherwise check the path."
            )
            return GenericFlatAdapter(self.config).discover(dataset_cfg)

        self._write_classification_report(classification_rows)
        recovered = sum(1 for r in classification_rows if r["decision"] == "RECOVERED")
        self.logger.info(
            f"DermNetAdapter: filename classification recovered {recovered} of "
            f"{len(classification_rows)} image(s) into a canonical target class directly"
        )
        return list(folders.values())

    def _write_classification_report(self, rows: List[dict]) -> None:
        if not rows:
            return
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "dermnet_filename_classification_report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "original_filename", "raw_folder", "detected_disease", "canonical_class", "decision",
            ])
            writer.writeheader()
            writer.writerows(rows)
        self.logger.info(f"DermNet filename classification report written to {report_path}")


class AtlasISIC31Adapter(DatasetAdapter):
    """
    Understands Atlas ISIC2019 (31 Classes)'s real structure:
    dataset_root/{train,test}/<class>/ — the same two-split shape as
    DermNet, but structurally simpler in one specific way: each of the
    31 class folders here already names exactly one disease, with no
    merged/ambiguous folder mixing several target diseases together.

    Because of that, this adapter deliberately does NOT run
    classify_filename_by_prefix() the way DermNetAdapter does — there
    is nothing to recover. A folder named "Psoriasis" already means
    Psoriasis for every file inside it; per-image filename parsing
    would be pure overhead with zero label-quality benefit, and would
    risk *reducing* correctness if any filename happened to contain
    another disease's name as a substring. So: no filename parsing, no
    ambiguity detection, no recovery logic — just discovery.

    Both splits are merged into one DiscoveredFolder per class, exactly
    as DermNetAdapter does: the original train/test split is not
    reused as a dataset split downstream, but it IS preserved per-image
    via original_split, so provenance is never lost.

    One disease in this dataset needs a naming callout: this atlas's
    "Seborrheic Keratosis" folder is a *different* disease from PSD-HP's
    target "Seborrheic_Dermatitis" — the names are superficially close
    but clinically unrelated (a benign keratinocytic growth vs. an
    inflammatory dermatitis). This adapter does not special-case that
    folder at all; it discovers it like any other and hands mapper.py
    the raw label "Seborrheic Keratosis" unchanged. The actual
    exclusion decision belongs to class_mapping.json (Task 5), not
    here — the adapter layer's job is discovery, never classification.
    """

    _SPLIT_DIRS = ("train", "test")

    def discover(self, dataset_cfg: DatasetConfig) -> List[DiscoveredFolder]:
        folders: Dict[str, DiscoveredFolder] = {}
        if not dataset_cfg.path.exists():
            return []

        found_any_split_dir = False
        for split in self._SPLIT_DIRS:
            split_dir = dataset_cfg.path / split
            if not split_dir.is_dir():
                continue
            found_any_split_dir = True

            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                raw_label = class_dir.name
                folder = folders.setdefault(raw_label, DiscoveredFolder(dataset_cfg.name, raw_label))
                for img_path in class_dir.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in self.config.supported_extensions:
                        folder.images.append(DiscoveredImage(
                            path=img_path, dataset_version=dataset_cfg.name,
                            original_split=split, source_type="original",
                            normalized_label=raw_label, raw_label_on_disk=raw_label,
                        ))

        if not found_any_split_dir:
            self.logger.warning(
                f"AtlasISIC31Adapter: no 'train' or 'test' subfolder found under {dataset_cfg.path} — "
                f"falling back to flat (dataset_root/<class>/) discovery. If your Atlas ISIC31 copy "
                f"really has no train/test split, this is expected; otherwise check the path."
            )
            return GenericFlatAdapter(self.config).discover(dataset_cfg)

        self.logger.info(
            f"AtlasISIC31Adapter: discovered {len(folders)} class label(s) "
            f"({sum(f.image_count for f in folders.values())} image(s) total) across "
            f"{self._SPLIT_DIRS} — no filename reclassification applied"
        )
        return list(folders.values())


class SkinDisNetAdapter(DatasetAdapter):
    """
    Understands SkinDisNet's real structure:
    dataset_root/{SkinDisNet,SkinDisNet_2}/{Preprocessed,Augmented}/<class>/,
    where the two versions name the same six classes differently:

        SkinDisNet    (v1, abbreviated): AD, CD, EC, SC, SD, TC
        SkinDisNet_2  (v2, full name):   "Atopic Dermatitis (AD)", ...

    Both are normalized to the same canonical, underscored labels
    already used in class_mapping.json (e.g. "Seborrheic_Dermatitis"),
    so mapper.py needs zero changes to handle either naming
    convention — it only ever sees the normalized label.

    Augmented/ images are discovered and tagged source_type="augmented"
    regardless of config.include_augmented — the INCLUDE/EXCLUDE
    decision is deliberately left to extractor.py (Stage 5), not made
    here, so the adapter's output always reflects everything that
    actually exists on disk.
    """

    # Maps every known raw folder name variant -> canonical underscored label.
    _LABEL_NORMALIZATION = {
        "AD": "Atopic_Dermatitis", "Atopic Dermatitis (AD)": "Atopic_Dermatitis",
        "CD": "Contact_Dermatitis", "Contact Dermatitis (CD)": "Contact_Dermatitis",
        "EC": "Eczema", "Eczema (EC)": "Eczema",
        "SC": "Scabies", "Scabies (SC)": "Scabies",
        "SD": "Seborrheic_Dermatitis", "Seborrheic Dermatitis (SD)": "Seborrheic_Dermatitis",
        "TC": "Tinea_Corporis", "Tinea Corporis (TC)": "Tinea_Corporis",
    }
    _VERSION_DIRS = ("SkinDisNet", "SkinDisNet_2")
    _SOURCE_TYPE_DIRS = {"Preprocessed": "original", "Augmented": "augmented"}

    def discover(self, dataset_cfg: DatasetConfig) -> List[DiscoveredFolder]:
        folders: Dict[str, DiscoveredFolder] = {}
        if not dataset_cfg.path.exists():
            return []

        found_any_version_dir = False
        for version in self._VERSION_DIRS:
            version_dir = dataset_cfg.path / version
            if not version_dir.is_dir():
                continue
            found_any_version_dir = True
            self._discover_version(dataset_cfg.name, version, version_dir, folders)

        if not found_any_version_dir:
            self.logger.warning(
                f"SkinDisNetAdapter: neither 'SkinDisNet' nor 'SkinDisNet_2' found under "
                f"{dataset_cfg.path} — falling back to flat discovery. If your copy uses a "
                f"different folder name, this will miss it; check the real path."
            )
            return GenericFlatAdapter(self.config).discover(dataset_cfg)

        return list(folders.values())

    def _discover_version(self, dataset_name: str, version: str, version_dir: Path,
                           folders: Dict[str, DiscoveredFolder]) -> None:
        for subfolder_name, source_type in self._SOURCE_TYPE_DIRS.items():
            source_dir = version_dir / subfolder_name
            if not source_dir.is_dir():
                continue

            for class_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
                raw_label = class_dir.name
                normalized = self._LABEL_NORMALIZATION.get(raw_label)
                if normalized is None:
                    self.logger.warning(
                        f"SkinDisNetAdapter: unrecognized class folder name "
                        f"'{version}/{subfolder_name}/{raw_label}' — no normalization entry for it, "
                        f"skipping (add it to SkinDisNetAdapter._LABEL_NORMALIZATION if this is real data)"
                    )
                    continue

                folder = folders.setdefault(normalized, DiscoveredFolder(dataset_name, normalized))
                for img_path in class_dir.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in self.config.supported_extensions:
                        folder.images.append(DiscoveredImage(
                            path=img_path, dataset_version=version,
                            original_split="n/a", source_type=source_type,
                            normalized_label=normalized, raw_label_on_disk=raw_label,
                        ))


_ADAPTER_REGISTRY = {
    "generic_flat": GenericFlatAdapter,
    "dermnet": DermNetAdapter,
    "skindisnet": SkinDisNetAdapter,
    "atlas31": AtlasISIC31Adapter,
}


def get_adapter(config: PSDConfig, dataset_cfg: DatasetConfig) -> DatasetAdapter:
    """Look up the right adapter for a dataset by its configured adapter_kind."""
    adapter_cls = _ADAPTER_REGISTRY.get(dataset_cfg.adapter_kind, GenericFlatAdapter)
    return adapter_cls(config)


def discover_dataset(config: PSDConfig, dataset_cfg: DatasetConfig) -> List[DiscoveredFolder]:
    """Single entry point scanner.py and validator.py both call: give me this
    dataset's real images, normalized, regardless of its actual folder depth."""
    return get_adapter(config, dataset_cfg).discover(dataset_cfg)
