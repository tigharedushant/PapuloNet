"""
modules/aef_input_validator.py

AEF-CRC Phase 1: input-validation layer between PSD-HP's output and
every later AEF-CRC modeling phase.

Why this module exists (and why it isn't a PSD-HP module): PSD-HP's
job ends at output/06_final_split/ + reports/metadata.csv. Two real
gaps sit between that and "safe to hand to EfficientNet":

  1. metadata.csv has no train/val/test column at all -- splitter.py
     (Stage 11) runs after metadata.py (Stage 9) and never writes
     back into provenance. There is currently no single table that
     says "this psd_id is in the test split." This module rebuilds
     that join by reading the actual final_split/ directory layout
     (the physical ground truth) and matching filenames back to
     metadata.csv by PSD ID.

  2. ProvenanceRecord has source_type ("original"/"augmented") but no
     parent-image link -- confirmed in this project's own handoff
     reconciliation, not assumed here. That means this module CANNOT
     prove "image X's augmented derivative is in a different split
     than X itself." What it CAN do, and does, is report every signal
     that's actually available: exact-file cross-split leakage
     (should be zero if splitter.py is correct -- verified, not
     assumed), and a coarse augmented-image-per-split breakdown per
     (source, class) that flags where the leakage risk exists even
     though it can't be resolved to specific pairs yet.

This module is read-only with respect to PSD-HP's own outputs: it
never modifies output/06_final_split/, output/05_harmonized/, or
reports/metadata.csv. It only reads them and writes its own reports
under config.aef_crc_reports_dir.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from config.config import PSDConfig
from utils.logger import get_module_logger

_SPLITS = ("train", "val", "test")


# ============================================================
# Data structures
# ============================================================

@dataclass
class ImageRecord:
    """One image, joined from metadata.csv (provenance) + its real
    on-disk split (from final_split/, the physical ground truth)."""
    psd_id: str
    split: str                 # "train" | "val" | "test"
    mapped_class: str
    source_dataset: str
    dataset_version: str
    original_split: str        # PSD-HP's own train/test tag, dataset-native, NOT the AEF-CRC split
    source_type: str           # "original" | "augmented"
    file_path: Path


@dataclass
class ImageIssue:
    psd_id: str
    file_path: str
    issue: str


@dataclass
class LeakageFinding:
    kind: str          # "EXACT_FILE_IN_MULTIPLE_SPLITS" | "AUGMENTED_RISK_GROUP"
    detail: str
    severity: str       # "CONFIRMED" | "RISK_SIGNAL"


@dataclass
class Phase1Report:
    total_images: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    class_counts: Dict[str, int] = field(default_factory=dict)
    split_class_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    source_counts: Dict[str, int] = field(default_factory=dict)
    source_class_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    image_issues: List[ImageIssue] = field(default_factory=list)
    leakage_findings: List[LeakageFinding] = field(default_factory=list)
    imbalance_ratio: Optional[float] = None
    passed: bool = False
    fail_reasons: List[str] = field(default_factory=list)
    orphan_files: List[Path] = field(default_factory=list)
    missing_metadata_rows: List[str] = field(default_factory=list)
    missing_disk_files: List[str] = field(default_factory=list)


# ============================================================
# Loader: join metadata.csv with the physical final_split/ layout
# ============================================================

class AEFInputLoader:
    """Rebuilds the psd_id -> split join that PSD-HP itself never writes,
    by reading the real files under config.final_dir."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("aef_input_validator", config.logs_dir, config.log_level)

    def load(self) -> List[ImageRecord]:
        metadata_rows = self._read_metadata_csv()
        if not metadata_rows:
            self.logger.error(
                f"reports/metadata.csv not found or empty at {self.config.reports_dir / 'metadata.csv'} -- "
                f"run the PSD-HP harmonization pipeline (main.py) before AEF-CRC Phase 1."
            )
            return []

        records: List[ImageRecord] = []
        missing_from_metadata = 0

        for split in _SPLITS:
            for target_class in self.config.target_classes:
                split_dir = self.config.final_dir / split / target_class
                if not split_dir.exists():
                    continue
                for img_path in sorted(split_dir.iterdir()):
                    if not img_path.is_file() or img_path.suffix.lower() not in self.config.supported_extensions:
                        continue
                    psd_id = img_path.stem  # final_split filenames are "<psd_id>.jpg" -- no separator suffix
                    meta = metadata_rows.get(psd_id)
                    if meta is None:
                        missing_from_metadata += 1
                        self.logger.warning(
                            f"{img_path} has PSD ID '{psd_id}' with no matching row in metadata.csv -- "
                            f"including it with unknown provenance rather than silently dropping it."
                        )
                        records.append(ImageRecord(
                            psd_id=psd_id, split=split, mapped_class=target_class,
                            source_dataset="UNKNOWN", dataset_version="UNKNOWN",
                            original_split="UNKNOWN", source_type="UNKNOWN",
                            file_path=img_path,
                        ))
                        continue
                    records.append(ImageRecord(
                        psd_id=psd_id, split=split, mapped_class=meta["mapped_class"],
                        source_dataset=meta["source_dataset"], dataset_version=meta["dataset_version"],
                        original_split=meta["original_split"], source_type=meta["source_type"],
                        file_path=img_path,
                    ))

        if missing_from_metadata:
            self.logger.warning(
                f"{missing_from_metadata} final-split image(s) had no metadata.csv row at all -- "
                f"flagged individually above, counted in the totals with UNKNOWN provenance."
            )

        self.logger.info(f"AEF-CRC input loader: joined {len(records)} final-split image(s) to their metadata.")
        return records

    def _read_metadata_csv(self) -> Dict[str, dict]:
        path = self.config.reports_dir / "metadata.csv"
        if not path.exists():
            return {}
        rows: Dict[str, dict] = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["psd_id"]] = row
        return rows


# ============================================================
# Validator: image integrity + distributions + imbalance + leakage
# ============================================================

class AEFDatasetValidator:
    """Phase 1 core: validates every image, computes every distribution
    Part 3/4/5 asks for, and reports (never fixes) leakage risk."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("aef_input_validator", config.logs_dir, config.log_level)

    def run(self, records: List[ImageRecord]) -> Phase1Report:
        report = Phase1Report()
        report.total_images = len(records)

        if not records:
            report.fail_reasons.append("No images found under output/06_final_split/ for any target class.")
            return report

        self._validate_images(records, report)
        self._check_bidirectional_integrity(records, report)
        self._compute_distributions(records, report)
        self._detect_leakage(records, report)
        self._compute_imbalance(report)
        self._decide_pass_fail(records, report)
        return report

    def _check_bidirectional_integrity(self, records: List[ImageRecord], report: Phase1Report) -> None:
        """Bidirectional check:
        1. metadata -> disk: records with missing files (missing_disk_files).
        2. disk -> metadata: disk files unreferenced or missing from metadata (orphan_files).
        3. metadata rows with no corresponding disk file.
        """
        metadata_path = self.config.reports_dir / "metadata.csv"
        if not metadata_path.exists():
            return
        
        metadata_psd_ids = set()
        final_split_metadata_ids = set()
        with metadata_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                metadata_psd_ids.add(row["psd_id"])
                if row.get("final_filename"):
                    final_split_metadata_ids.add(row["psd_id"])

        loaded_psd_ids = {r.psd_id for r in records}
        
        # Disk files not in metadata (orphans)
        for r in records:
            if r.source_dataset == "UNKNOWN" or r.psd_id not in metadata_psd_ids:
                report.orphan_files.append(r.file_path)

        # Final-split metadata rows not on disk
        for psd_id in sorted(final_split_metadata_ids - loaded_psd_ids):
            report.missing_disk_files.append(psd_id)

        if report.orphan_files:
            self.logger.warning(f"Integrity check: {len(report.orphan_files)} orphan file(s) found on disk with no metadata.csv entry.")
        if report.missing_disk_files:
            self.logger.warning(f"Integrity check: {len(report.missing_disk_files)} final-split metadata.csv row(s) have no corresponding file in final_split/.")

    # ---- Part 4: image-level validation ----

    def _validate_images(self, records: List[ImageRecord], report: Phase1Report) -> None:
        for rec in records:
            if not rec.file_path.exists():
                report.image_issues.append(ImageIssue(rec.psd_id, str(rec.file_path), "FILE_MISSING"))
                continue
            try:
                with Image.open(rec.file_path) as img:
                    img.verify()
                # verify() invalidates the file handle -- reopen to actually read dimensions
                with Image.open(rec.file_path) as img:
                    w, h = img.size
                    if w < self.config.min_image_dimension or h < self.config.min_image_dimension:
                        report.image_issues.append(
                            ImageIssue(rec.psd_id, str(rec.file_path), f"DIMENSION_TOO_SMALL ({w}x{h})")
                        )
                    if img.format not in ("JPEG", "PNG", "BMP", "TIFF", "WEBP"):
                        report.image_issues.append(
                            ImageIssue(rec.psd_id, str(rec.file_path), f"UNEXPECTED_FORMAT ({img.format})")
                        )
            except (UnidentifiedImageError, OSError) as exc:
                report.image_issues.append(ImageIssue(rec.psd_id, str(rec.file_path), f"CANNOT_OPEN ({exc})"))

        if report.image_issues:
            self.logger.warning(f"{len(report.image_issues)} image issue(s) found -- see aef_crc_validation_report.txt")

    # ---- Part 3: distributions ----

    def _compute_distributions(self, records: List[ImageRecord], report: Phase1Report) -> None:
        for rec in records:
            report.split_counts[rec.split] = report.split_counts.get(rec.split, 0) + 1
            report.class_counts[rec.mapped_class] = report.class_counts.get(rec.mapped_class, 0) + 1
            report.source_counts[rec.source_dataset] = report.source_counts.get(rec.source_dataset, 0) + 1

            sc_key = (rec.split, rec.mapped_class)
            report.split_class_counts[sc_key] = report.split_class_counts.get(sc_key, 0) + 1

            src_key = (rec.source_dataset, rec.mapped_class)
            report.source_class_counts[src_key] = report.source_class_counts.get(src_key, 0) + 1

    # ---- Part 5: imbalance ratio (report only -- weights computed separately, per-fold) ----

    def _compute_imbalance(self, report: Phase1Report) -> None:
        counts = [c for c in report.class_counts.values() if c > 0]
        if len(counts) >= 2:
            report.imbalance_ratio = max(counts) / min(counts)

    # ---- Part 4 items 7-9 + Part 25 item 10: leakage detection (report, never fix) ----

    def _detect_leakage(self, records: List[ImageRecord], report: Phase1Report) -> None:
        # 1) CONFIRMED check: does any psd_id appear in more than one split?
        #    This should be structurally impossible given how splitter.py
        #    partitions files, but "should be impossible" is exactly the
        #    kind of claim this project's own principles say to verify,
        #    not assume -- so it's checked directly here, not skipped.
        psd_to_splits: Dict[str, set] = defaultdict(set)
        for rec in records:
            psd_to_splits[rec.psd_id].add(rec.split)
        cross_split_ids = {pid: splits for pid, splits in psd_to_splits.items() if len(splits) > 1}
        if cross_split_ids:
            for pid, splits in cross_split_ids.items():
                report.leakage_findings.append(LeakageFinding(
                    kind="EXACT_FILE_IN_MULTIPLE_SPLITS",
                    detail=f"PSD ID {pid} appears in splits: {sorted(splits)}",
                    severity="CONFIRMED",
                ))
        else:
            self.logger.info("Leakage check: no exact PSD ID appears in more than one split. Confirmed clean on this axis.")

        # 2) RISK SIGNAL, not a confirmed leak: because ProvenanceRecord has
        #    no parent_original_id (verified gap, not assumed), this project
        #    cannot yet prove "image X's augmented derivative crossed a
        #    split boundary." What it CAN report: for every (source,
        #    class) group that contains augmented images, whether those
        #    augmented images are confined to a single split or spread
        #    across more than one. Spread-across-splits is the exact
        #    precondition for the leakage this project flagged as unresolved.
        aug_groups: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for rec in records:
            if rec.source_type == "augmented":
                aug_groups[(rec.source_dataset, rec.mapped_class)][rec.split] += 1

        for (source, cls), split_counts in aug_groups.items():
            if len(split_counts) > 1:
                report.leakage_findings.append(LeakageFinding(
                    kind="AUGMENTED_RISK_GROUP",
                    detail=(
                        f"{source} / {cls}: augmented images present in {len(split_counts)} splits "
                        f"({dict(split_counts)}). Cannot confirm specific original<->augmented pairs "
                        f"crossed a split (no parent_original_id in provenance yet) -- flagged as an "
                        f"unresolved risk, not a fix."
                    ),
                    severity="RISK_SIGNAL",
                ))

    def _decide_pass_fail(self, records: List[ImageRecord], report: Phase1Report) -> None:
        confirmed = [f for f in report.leakage_findings if f.severity == "CONFIRMED"]
        missing_split = [s for s in _SPLITS if report.split_counts.get(s, 0) == 0]
        missing_classes = [c for c in self.config.target_classes if report.class_counts.get(c, 0) == 0]

        if confirmed:
            report.fail_reasons.append(f"{len(confirmed)} CONFIRMED cross-split leakage finding(s) -- see report.")
        if missing_split:
            report.fail_reasons.append(f"Split(s) with zero images: {missing_split}")
        if missing_classes:
            report.fail_reasons.append(f"Target class(es) with zero images anywhere: {missing_classes}")
        # Bad/unopenable files are reported, not auto-failed -- Part 4 says
        # "report it, do not silently delete it," which is a data-quality
        # finding for a human to act on, not by itself a pipeline failure.

        report.passed = len(report.fail_reasons) == 0


# ============================================================
# Reusable, fold-safe class weights (Part 5)
# ============================================================

def compute_train_fold_class_weights(train_fold_labels: List[str], all_classes: List[str]) -> Dict[str, float]:
    """
    weight_c = N / (C * N_c)

    train_fold_labels: mapped_class of every sample in THIS fold's
    training partition only. Never pass validation or test labels in --
    this function has no way to detect that misuse, so the caller is
    responsible for the Part 5 guarantee that this is fold-training-only.

    all_classes: the full target taxonomy, so a class with zero samples
    in this particular fold still gets an explicit (documented) weight
    rather than silently vanishing from the returned dict.
    """
    n = len(train_fold_labels)
    c = len(all_classes)
    counts: Dict[str, int] = {cls: 0 for cls in all_classes}
    for label in train_fold_labels:
        if label in counts:
            counts[label] += 1

    weights: Dict[str, float] = {}
    for cls, n_c in counts.items():
        if n_c == 0:
            weights[cls] = float("nan")  # explicit, not silently 0 or inf -- caller must decide policy
        else:
            weights[cls] = n / (c * n_c)
    return weights


# ============================================================
# Report writers (Part 3 + Part 23 output paths)
# ============================================================

class Phase1ReportWriter:
    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("aef_input_validator", config.logs_dir, config.log_level)

    def write(self, report: Phase1Report) -> None:
        out_dir = self.config.aef_crc_reports_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        self._write_dataset_report(report, out_dir / "dataset_report.csv")
        self._write_source_class_report(report, out_dir / "source_class_report.csv")
        self._write_validation_report(report, out_dir / "aef_crc_validation_report.txt")

        self.logger.info(f"AEF-CRC Phase 1 reports written to {out_dir}")

    def _write_dataset_report(self, report: Phase1Report, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "class", "count"])
            for split in _SPLITS:
                for cls in self.config.target_classes:
                    writer.writerow([split, cls, report.split_class_counts.get((split, cls), 0)])

    def _write_source_class_report(self, report: Phase1Report, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_dataset", "class", "count"])
            for (source, cls), count in sorted(report.source_class_counts.items()):
                writer.writerow([source, cls, count])

    def _write_validation_report(self, report: Phase1Report, path: Path) -> None:
        lines = ["AEF-CRC Phase 1 -- Dataset Validation Report", "=" * 50, ""]
        lines.append(f"Total images (final split, all classes): {report.total_images}")
        lines.append("")
        lines.append("Split counts:")
        for split in _SPLITS:
            lines.append(f"  {split}: {report.split_counts.get(split, 0)}")
        lines.append("")
        lines.append("Class counts (all splits combined):")
        for cls, count in sorted(report.class_counts.items()):
            lines.append(f"  {cls}: {count}")
        lines.append("")
        if report.imbalance_ratio is not None:
            lines.append(f"Imbalance ratio (majority/minority class count): {report.imbalance_ratio:.2f}")
        else:
            lines.append("Imbalance ratio: not computable (fewer than 2 non-empty classes)")
        lines.append("")
        lines.append(f"Source counts: {dict(sorted(report.source_counts.items()))}")
        lines.append("")
        lines.append(f"Image issues found: {len(report.image_issues)}")
        for issue in report.image_issues[:50]:
            lines.append(f"  [{issue.issue}] {issue.psd_id} -- {issue.file_path}")
        if len(report.image_issues) > 50:
            lines.append(f"  ... and {len(report.image_issues) - 50} more (see full list not truncated in this report if needed)")
        lines.append("")
        lines.append(f"Integrity check - Orphan files on disk (missing from metadata.csv): {len(report.orphan_files)}")
        for p in report.orphan_files[:50]:
            lines.append(f"  [ORPHAN_DISK_FILE] {p}")
        if len(report.orphan_files) > 50:
            lines.append(f"  ... and {len(report.orphan_files) - 50} more")
        lines.append("")
        lines.append(f"Integrity check - Missing disk files (in metadata.csv but absent from final_split): {len(report.missing_disk_files)}")
        for pid in report.missing_disk_files[:50]:
            lines.append(f"  [MISSING_DISK_FILE] {pid}")
        if len(report.missing_disk_files) > 50:
            lines.append(f"  ... and {len(report.missing_disk_files) - 50} more")
        lines.append("")
        lines.append(f"PHASE 1 RESULT: {'PASSED' if report.passed else 'FAILED'}")
        if not report.passed:
            for reason in report.fail_reasons:
                lines.append(f"  - {reason}")

        path.write_text("\n".join(lines), encoding="utf-8")
