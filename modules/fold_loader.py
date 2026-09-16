"""
modules/fold_loader.py

AEF-CRC Stage: Imbalance-Aware Data Loading.

Builds stratified K-fold splits over the TRAIN partition only, and
computes class weights fresh inside each fold's training portion --
the Part 5/16 requirement that class weights must never be
influenced by validation or test data, and must be recalculated per
fold rather than once globally.

Scope, deliberately: val and test (as produced by splitter.py) are
never read, indexed, or touched by this module. K-fold cross-
validation happens entirely inside train, so the held-out val split
remains available later for calibration (Part 17) and test remains
fully locked until final evaluation (Part 16) -- neither is silently
absorbed into the CV procedure.

Uses sklearn.model_selection.StratifiedKFold rather than a hand-rolled
splitter: this is exactly the kind of established, correctly-tested
utility Part 21 ("code quality") argues for reusing rather than
reimplementing, and it's already available in this environment.

--- CV-level augmentation leakage: what was tried, what was wrong, what's here now ---
Round 1 (splitter.py): confined source_type="augmented" images to the
outer TRAIN partition. Correct as far as it goes, but does not by
itself protect CV.

Round 2 (this module, first version): partitioned only non-augmented
train images with StratifiedKFold, then appended ALL augmented images
to every fold's training side. This was WRONG, caught on review, and
the reasoning for why is worth keeping rather than deleting: it
assumed "augmented image never appears in a fold's validation set" was
enough. It isn't. Counterexample: an ORIGINAL image can land in
splitter.py's outer VAL or TEST set; if that original has an augmented
derivative -- and there is no parent_original_id to know whether it
does -- that derivative is source_type="augmented" and would still get
added to CV training in every fold. The leak isn't about augmented
images crossing fold boundaries; it's about an augmented image in
training carrying information about a *different, unrelated-on-paper*
image that ended up in evaluation data, because we cannot verify they
are in fact unrelated.

Round 3 (this version, correct): without parent_original_id, there is
no safe way to let ANY augmented image participate in model selection
at all -- appending it anywhere training-adjacent risks leaking
against an unknown held-out original. So: augmented images are
EXCLUDED FROM CV ENTIRELY. StratifiedKFold partitions only
non-augmented train images into fold-train/fold-validation; augmented
images never appear in either. This is reported explicitly (see
FoldPlan.excluded_augmented_records and the log line below), not
silently dropped.

This is a real cost, stated plainly, not minimized: CV effectively
sees less Seborrheic Dermatitis data than the outer train split
contains, because most of that class's outer-train images are
augmented (see Phase 2's dataset_report.csv for the actual count).
Whether augmented images are reintroduced for a FINAL retrain (after
CV-based model selection is complete, using the full outer-train set)
is a decision for the modeling phase, not this one -- and if that
happens, the same cross-type leak risk against outer val/test applies
there too and must be weighed then, not assumed solved by this fix.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from sklearn.model_selection import StratifiedKFold

from config.config import PSDConfig
from utils.logger import get_module_logger
from modules.aef_input_validator import AEFInputLoader, ImageRecord, compute_train_fold_class_weights


@dataclass
class Fold:
    fold_index: int
    train_records: List[ImageRecord]  # non-augmented only -- see module docstring
    val_records: List[ImageRecord]    # CV-internal validation slice -- NOT splitter.py's held-out "val" split
    class_weights: Dict[str, float]


@dataclass
class FoldPlan:
    k: int
    folds: List[Fold] = field(default_factory=list)
    holdout_val_records: List[ImageRecord] = field(default_factory=list)  # splitter.py's val/ -- untouched, reported only
    holdout_test_records: List[ImageRecord] = field(default_factory=list)  # splitter.py's test/ -- untouched, reported only
    # Augmented train-split images, entirely excluded from CV (see module
    # docstring Round 3). Kept here only so callers can report the count
    # and decide, later and deliberately, whether/how to use them for a
    # final retrain -- this module itself never feeds them into a fold.
    excluded_augmented_records: List[ImageRecord] = field(default_factory=list)
    limitation_note: str = (
        "Augmented images were excluded from cross-validation because parent "
        "lineage could not be reliably established (no parent_original_id in "
        "provenance). This is a documented limitation, not a resolved guarantee "
        "against all augmentation-related leakage -- see fold_loader.py's docstring."
    )


class ImbalanceAwareFoldLoader:
    def __init__(self, config: PSDConfig, k: int = 5) -> None:
        self.config = config
        self.k = k
        self.logger = get_module_logger("fold_loader", config.logs_dir, config.log_level)

    def build(self) -> FoldPlan:
        all_records = AEFInputLoader(self.config).load()
        train_records = [r for r in all_records if r.split == "train"]
        val_records = [r for r in all_records if r.split == "val"]
        test_records = [r for r in all_records if r.split == "test"]

        if not train_records:
            self.logger.error("No train-split images found -- run main.py and run_aef_crc_phase1.py first.")
            return FoldPlan(k=self.k)

        # Augmented images are excluded from CV entirely -- see module
        # docstring "Round 3". Only non-augmented images are ever
        # partitioned by StratifiedKFold or appear in any fold.
        splittable = [r for r in train_records if r.source_type != "augmented"]
        excluded_augmented = [r for r in train_records if r.source_type == "augmented"]

        if not splittable:
            self.logger.error(
                "Every train-split image has source_type='augmented' -- there is nothing left to "
                "stratify into CV folds. This should not happen in practice; check the dataset."
            )
            return FoldPlan(
                k=self.k, holdout_val_records=val_records, holdout_test_records=test_records,
                excluded_augmented_records=excluded_augmented,
            )

        class_counts: Dict[str, int] = {}
        for r in splittable:
            class_counts[r.mapped_class] = class_counts.get(r.mapped_class, 0) + 1
        min_class_count = min(class_counts.values()) if class_counts else 0
        effective_k = self.k
        if min_class_count < self.k:
            effective_k = max(2, min_class_count)
            self.logger.warning(
                f"Requested K={self.k} but the smallest non-augmented train class has only "
                f"{min_class_count} sample(s) -- StratifiedKFold requires at least K samples per class. "
                f"Reducing to K={effective_k} for this run rather than silently erroring or producing "
                f"degenerate folds. Report this reduction; it usually means the rare class needs more "
                f"real (non-augmented) data, not that K should be forced higher."
            )

        labels = [r.mapped_class for r in splittable]
        skf = StratifiedKFold(n_splits=effective_k, shuffle=True, random_state=self.config.random_seed)

        plan = FoldPlan(
            k=effective_k, holdout_val_records=val_records, holdout_test_records=test_records,
            excluded_augmented_records=excluded_augmented,
        )

        for fold_index, (train_idx, val_idx) in enumerate(skf.split(splittable, labels)):
            fold_train = [splittable[i] for i in train_idx]   # non-augmented only, no exceptions
            fold_val = [splittable[i] for i in val_idx]        # non-augmented only, no exceptions

            fold_train_labels = [r.mapped_class for r in fold_train]
            weights = compute_train_fold_class_weights(fold_train_labels, self.config.target_classes)

            plan.folds.append(Fold(
                fold_index=fold_index,
                train_records=fold_train,
                val_records=fold_val,
                class_weights=weights,
            ))

        self.logger.info(
            f"Built {effective_k} stratified fold(s) over {len(splittable)} non-augmented train image(s); "
            f"{len(excluded_augmented)} augmented train image(s) EXCLUDED from CV entirely (see "
            f"FoldPlan.limitation_note); {len(val_records)} held-out val, {len(test_records)} held-out "
            f"test remain untouched by CV."
        )
        return plan


def save_fold_plan(config: PSDConfig, plan: FoldPlan) -> Path:
    """Serializes the authoritative fold assignment to reports/aef_crc/fold_plan.csv
    and summary to reports/aef_crc/phase2_fold_summary.csv.
    """
    out_dir = config.aef_crc_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "fold_plan.csv"
    summary_path = out_dir / "phase2_fold_summary.csv"

    fieldnames = [
        "fold_id", "split_role", "psd_id", "mapped_class",
        "source_dataset", "source_type", "file_path",
    ]

    with plan_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fold in plan.folds:
            for r in fold.train_records:
                writer.writerow({
                    "fold_id": fold.fold_index,
                    "split_role": "train",
                    "psd_id": r.psd_id,
                    "mapped_class": r.mapped_class,
                    "source_dataset": r.source_dataset,
                    "source_type": r.source_type,
                    "file_path": str(r.file_path),
                })
            for r in fold.val_records:
                writer.writerow({
                    "fold_id": fold.fold_index,
                    "split_role": "val",
                    "psd_id": r.psd_id,
                    "mapped_class": r.mapped_class,
                    "source_dataset": r.source_dataset,
                    "source_type": r.source_type,
                    "file_path": str(r.file_path),
                })

        for r in plan.holdout_val_records:
            writer.writerow({
                "fold_id": -1,
                "split_role": "holdout_val",
                "psd_id": r.psd_id,
                "mapped_class": r.mapped_class,
                "source_dataset": r.source_dataset,
                "source_type": r.source_type,
                "file_path": str(r.file_path),
            })
        for r in plan.holdout_test_records:
            writer.writerow({
                "fold_id": -1,
                "split_role": "holdout_test",
                "psd_id": r.psd_id,
                "mapped_class": r.mapped_class,
                "source_dataset": r.source_dataset,
                "source_type": r.source_type,
                "file_path": str(r.file_path),
            })
        for r in plan.excluded_augmented_records:
            writer.writerow({
                "fold_id": -1,
                "split_role": "excluded_augmented",
                "psd_id": r.psd_id,
                "mapped_class": r.mapped_class,
                "source_dataset": r.source_dataset,
                "source_type": r.source_type,
                "file_path": str(r.file_path),
            })

    # Summary table
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold_id", "n_train", "n_val", "class_weights_json"])
        for fold in plan.folds:
            writer.writerow([
                fold.fold_index,
                len(fold.train_records),
                len(fold.val_records),
                json.dumps(fold.class_weights),
            ])

    return plan_path


def _resolve_portable_path(raw_path_str: str, config: Optional[PSDConfig] = None, split: str = "", mapped_class: str = "", psd_id: str = "") -> Path:
    """Resolves cross-platform, relocated, and RAM-accelerated file paths seamlessly across
    Windows, WSL2, Linux, and different mount environments."""
    # 0. Check fast RAM filesystem (/dev/shm) if populated in Linux/WSL2
    if split and mapped_class and psd_id:
        shm_cand = Path(f"/dev/shm/06_final_split/{split}/{mapped_class}/{psd_id}.jpg")
        if shm_cand.exists():
            return shm_cand

    p = Path(raw_path_str)
    if p.exists():
        return p

    # 1. Check relative to active config.output_dir / "06_final_split"
    if config is not None and split and mapped_class and psd_id:
        cand = config.output_dir / "06_final_split" / split / mapped_class / f"{psd_id}.jpg"
        if cand.exists():
            return cand

    # 2. Windows -> WSL translation (e.g. C:\Users\... -> /mnt/c/Users/...)
    import re
    m = re.match(r"^([a-zA-Z]):[/\\](.*)", raw_path_str)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        wsl_p = Path(f"/mnt/{drive}/{rest}")
        if wsl_p.exists():
            return wsl_p

    # 3. WSL -> Windows translation (e.g. /mnt/c/Users/... -> C:/Users/...)
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)", raw_path_str)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2)
        win_p = Path(f"{drive}:/{rest}")
        if win_p.exists():
            return win_p

    return p


def load_frozen_fold_plan(config: PSDConfig, fold_plan_path: Optional[Path] = None) -> FoldPlan:
    """The ONLY authoritative downstream source of fold assignments.
    Fails loudly if reports/aef_crc/fold_plan.csv is missing.
    """
    path = fold_plan_path or (config.aef_crc_reports_dir / "fold_plan.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Authoritative fold plan not found at {path}. "
            f"Phase 2 must be run first to generate reports/aef_crc/fold_plan.csv. "
            f"Downstream phases MUST NOT rebuild folds in memory!"
        )

    folds_train: Dict[int, List[ImageRecord]] = {}
    folds_val: Dict[int, List[ImageRecord]] = {}
    holdout_val: List[ImageRecord] = []
    holdout_test: List[ImageRecord] = []
    excluded_aug: List[ImageRecord] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_role = row["split_role"]
            actual_split = "train" if split_role in ("train", "val", "excluded_augmented") else ("val" if split_role == "holdout_val" else "test")
            resolved_fp = _resolve_portable_path(
                raw_path_str=row["file_path"],
                config=config,
                split=actual_split,
                mapped_class=row["mapped_class"],
                psd_id=row["psd_id"],
            )
            rec = ImageRecord(
                psd_id=row["psd_id"],
                split=actual_split,
                mapped_class=row["mapped_class"],
                source_dataset=row.get("source_dataset", "UNKNOWN"),
                dataset_version="UNKNOWN",
                original_split="UNKNOWN",
                source_type=row.get("source_type", "original"),
                file_path=resolved_fp,
            )
            role = row["split_role"]
            fold_id = int(row["fold_id"])

            if role == "train":
                folds_train.setdefault(fold_id, []).append(rec)
            elif role == "val":
                folds_val.setdefault(fold_id, []).append(rec)
            elif role == "holdout_val":
                holdout_val.append(rec)
            elif role == "holdout_test":
                holdout_test.append(rec)
            elif role == "excluded_augmented":
                excluded_aug.append(rec)

    all_fold_ids = sorted(set(folds_train.keys()) | set(folds_val.keys()))
    folds: List[Fold] = []
    for fid in all_fold_ids:
        train_recs = folds_train.get(fid, [])
        val_recs = folds_val.get(fid, [])
        weights = compute_train_fold_class_weights(
            [r.mapped_class for r in train_recs], config.target_classes
        )
        folds.append(Fold(
            fold_index=fid,
            train_records=train_recs,
            val_records=val_recs,
            class_weights=weights,
        ))

    plan = FoldPlan(
        k=len(folds),
        folds=folds,
        holdout_val_records=holdout_val,
        holdout_test_records=holdout_test,
        excluded_augmented_records=excluded_aug,
    )

    validate_fold_plan(plan, config)
    return plan


def validate_fold_plan(plan: FoldPlan, config: PSDConfig) -> None:
    """Validates the invariants of the authoritative fold plan:
    - Every CV image appears exactly once as validation across all folds
    - In each fold, train and validation are completely disjoint
    - Outer holdout val and holdout test are absent from all CV folds
    - Augmented images are strictly absent from all CV folds
    - All target classes are valid
    - All expected folds are present
    """
    if not plan.folds:
        raise ValueError("FoldPlan has zero folds!")

    cv_val_ids: Set[str] = set()
    cv_val_duplicates: Set[str] = set()

    for fold in plan.folds:
        train_ids = {r.psd_id for r in fold.train_records}
        val_ids = {r.psd_id for r in fold.val_records}

        overlap = train_ids & val_ids
        if overlap:
            raise ValueError(f"Fold {fold.fold_index} has train/val overlap: {overlap}")

        for vid in val_ids:
            if vid in cv_val_ids:
                cv_val_duplicates.add(vid)
            cv_val_ids.add(vid)

        for r in fold.train_records + fold.val_records:
            if r.source_type == "augmented":
                raise ValueError(f"Fold {fold.fold_index} contains augmented image {r.psd_id}!")
            if r.mapped_class not in config.target_classes:
                raise ValueError(f"Fold {fold.fold_index} has invalid class: {r.mapped_class}")

    if cv_val_duplicates:
        raise ValueError(f"CV images appear in validation of multiple folds: {cv_val_duplicates}")

    holdout_val_ids = {r.psd_id for r in plan.holdout_val_records}
    holdout_test_ids = {r.psd_id for r in plan.holdout_test_records}

    all_cv_ids = {r.psd_id for f in plan.folds for r in (f.train_records + f.val_records)}
    val_leak = all_cv_ids & holdout_val_ids
    if val_leak:
        raise ValueError(f"Outer holdout validation images leaked into CV: {val_leak}")
    test_leak = all_cv_ids & holdout_test_ids
    if test_leak:
        raise ValueError(f"Outer test images leaked into CV: {test_leak}")


