"""
modules/splitter.py

PSD-HP Stage 5a: Train / Validation / Test Split.

ACCURATE SCOPE STATEMENT (do not describe this splitter as more than
this): the split is CLASS-stratified plus augmentation-safe. It is
NOT source-stratified -- it does not attempt to give train/val/test
matching proportions of DermNet vs. SkinDisNet vs. Curated32 vs.
AtlasISIC31 within each class. Source × class cell sizes vary too
much for that to be done safely for every class (e.g. Curated32
contributes zero images to Pityriasis_Rosea and Seborrheic_Dermatitis
entirely -- there is nothing to stratify by source for those cells).
Source distribution per split is reported (see reports/aef_crc/ and
Phase 1's source_class_report.csv) so it's visible, not hidden --
reporting an imbalance honestly is different from claiming it was
corrected.

Consumes the harmonized images in output/05_harmonized/<TargetClass>/
and performs a stratified split — meaning the split is computed
independently within each target class, so a class-imbalanced overall
dataset still gets proportionally correct train/val/test ratios per
class, rather than a class-blind random split that could leave (say)
the test set with zero Pityriasis Rosea images by chance.

The split is fully reproducible: shuffling uses Python's random
module seeded with config.random_seed, so re-running splitter.py on
an unchanged harmonized set always produces an identical split — a
requirement for any result in this project to be reproducible, which
this project's own reviewer summary lists as a core AEF-CRC strength.

Images are copied (not moved) into output/06_final_split/, so
output/05_harmonized/ remains a stable, re-splittable source if
config.random_seed or the ratios ever change.

--- Augmentation-leakage fix (AEF-CRC handoff, confirmed defect) ---
ProvenanceRecord has source_type ("original"/"augmented") but no
parent_original_id linking an augmented image back to its source
image -- confirmed by direct code inspection, not assumed. Without
that link, an augmented derivative and its original could previously
land in different splits (e.g. original in train, its rotated/
flipped augmented copy in test), which is a real leakage vector for
SkinDisNet's Augmented/ images specifically.

The fix implemented here does NOT attempt to reconstruct that missing
link (that would mean guessing a filename convention this project has
not confirmed against real data). Instead it applies the safe,
defensible rule requested in the AEF-CRC handoff: every image with
source_type == "augmented" is placed in train and ONLY train, never
val or test. This closes the leakage vector completely without
needing to know which augmented image belongs to which original.

RESIDUAL RISK, STATED EXPLICITLY (not resolved by this fix, and not
claimed to be): this rule only prevents an augmented image from
appearing in val/test. It does NOT prevent the reverse relationship --
an ORIGINAL image can land in val or test via the normal stratified
split above, while an augmented derivative of that same original
(unknowable which one, with no parent_original_id) sits in train.
That augmented image could still carry information about the
val/test original it was derived from. This was caught on review via
a concrete counterexample, not discovered independently here, and it
is why modules/fold_loader.py now excludes source_type="augmented"
images from cross-validation ENTIRELY rather than merely keeping them
out of the CV-validation side (see fold_loader.py's own docstring,
"Round 3"). Whether augmented images are safe to reintroduce for a
final retrain after model selection -- accepting this same residual
risk against the outer val/test sets -- is a decision explicitly left
open for the modeling phase, not resolved or assumed here.

NOTE: this outer-split rule is necessary but was found (on review,
before Phase 3) to not be sufficient by itself -- modules/fold_loader.py
had to apply the same principle a second time, one level deeper, for
CV folds. See fold_loader.py's own docstring for that fix; it is not
duplicated here.

This is a per-class, per-source_type change to _split_class(); every
other dataset (DermNet, Curated32, AtlasISIC31) has source_type
"original" for 100% of its images, so this fix is a no-op for them --
their split behavior is unchanged from before. Train ratio therefore
increases slightly for classes with augmented images (Seborrheic
Dermatitis, via SkinDisNet) -- this is reported explicitly in
split_report.txt rather than silently changing the advertised ratio.
"""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from config.config import PSDConfig
from utils.logger import get_module_logger


@dataclass
class SplitCounts:
    train: int = 0
    val: int = 0
    test: int = 0


@dataclass
class SplitResult:
    per_class_counts: Dict[str, SplitCounts] = field(default_factory=dict)
    # Count of images forced into train per class because source_type
    # was "augmented" -- reported so the ratio deviation is visible,
    # never silent.
    forced_train_augmented: Dict[str, int] = field(default_factory=dict)


class DatasetSplitter:
    """Stratified, reproducible train/val/test split of the harmonized dataset.
    Augmented-source images are always confined to train (see module docstring)."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("splitter", config.logs_dir, config.log_level)
        self._source_type_by_psd_id = self._load_source_types()

    def _load_source_types(self) -> Dict[str, str]:
        """Reads reports/metadata.csv (written by Stage 9, before this
        stage runs) to recover each image's source_type. This is a
        read of an existing, already-generated PSD-HP report -- not a
        new dependency and not a change to metadata.py itself."""
        path = self.config.reports_dir / "metadata.csv"
        mapping: Dict[str, str] = {}
        if not path.exists():
            self.logger.warning(
                f"{path} not found -- cannot determine which images are augmented. "
                f"Proceeding as if every image is source_type='original' (old behavior). "
                f"Run the full pipeline in order so metadata.csv exists before splitting."
            )
            return mapping
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mapping[row["psd_id"]] = row.get("source_type", "original")
        return mapping

    def split_all(self) -> SplitResult:
        self.logger.info(
            f"=== PSD-HP Stage 5a: Train/Val/Test Split started "
            f"(ratios {self.config.train_ratio}/{self.config.val_ratio}/{self.config.test_ratio}, "
            f"seed={self.config.random_seed}, augmented images confined to train) ==="
        )
        result = SplitResult()

        for split_name in ("train", "val", "test"):
            for target_class in self.config.target_classes:
                (self.config.final_dir / split_name / target_class).mkdir(parents=True, exist_ok=True)

        for target_class in self.config.target_classes:
            source_dir = self.config.harmonized_dir / target_class
            if not source_dir.exists():
                result.per_class_counts[target_class] = SplitCounts()
                continue

            counts, forced = self._split_class(target_class, source_dir)
            result.per_class_counts[target_class] = counts
            result.forced_train_augmented[target_class] = forced
            self.logger.info(
                f"{target_class}: train={counts.train}, val={counts.val}, test={counts.test}"
                + (f" (of which {forced} forced into train as augmented-source)" if forced else "")
            )

        self._write_report(result)
        self.logger.info("=== PSD-HP Stage 5a: Train/Val/Test Split finished ===")
        return result

    def _split_class(self, target_class: str, source_dir: Path):
        image_paths = sorted(
            f for f in source_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.config.supported_extensions
        )

        # Partition by source_type BEFORE any split logic runs. Augmented
        # images never enter the eligible-for-val/test pool at all --
        # this is the leakage fix, applied at the earliest possible point.
        augmented_paths = [
            p for p in image_paths if self._source_type_by_psd_id.get(p.stem) == "augmented"
        ]
        splittable_paths = [
            p for p in image_paths if self._source_type_by_psd_id.get(p.stem) != "augmented"
        ]

        rng = random.Random(self.config.random_seed)
        rng.shuffle(splittable_paths)

        n = len(splittable_paths)
        n_train = round(n * self.config.train_ratio)
        n_val = round(n * self.config.val_ratio)
        n_test = n - n_train - n_val  # remainder, so rounding never drops or duplicates a file

        splits = {
            "train": splittable_paths[:n_train] + augmented_paths,  # augmented always appended to train
            "val": splittable_paths[n_train:n_train + n_val],
            "test": splittable_paths[n_train + n_val:],
        }

        for split_name, files in splits.items():
            dest_dir = self.config.final_dir / split_name / target_class
            for img_path in files:
                try:
                    shutil.copy2(img_path, dest_dir / img_path.name)
                except OSError as exc:
                    self.logger.error(f"Failed to copy {img_path} into {split_name} split: {exc}")

        counts = SplitCounts(
            train=len(splits["train"]), val=len(splits["val"]), test=len(splits["test"]),
        )
        return counts, len(augmented_paths)

    def _write_report(self, result: SplitResult) -> None:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.reports_dir / "split_report.txt"

        lines = ["PSD-HP Train/Val/Test Split Report", "=" * 40, ""]
        lines.append(
            f"Ratios: train={self.config.train_ratio}, val={self.config.val_ratio}, "
            f"test={self.config.test_ratio} | seed={self.config.random_seed}"
        )
        lines.append(
            "Augmentation-leakage fix: every source_type='augmented' image is confined to "
            "train (never val/test) because no parent_original_id exists to pair it safely. "
            "Classes with augmented images will show a train ratio ABOVE the nominal ratio -- "
            "this is intentional, not a bug; see 'forced into train' counts below."
        )
        lines.append("")
        for target_class, counts in result.per_class_counts.items():
            total = counts.train + counts.val + counts.test
            forced = result.forced_train_augmented.get(target_class, 0)
            extra = f", {forced} forced into train as augmented-source" if forced else ""
            lines.append(f"{target_class}: train={counts.train}, val={counts.val}, test={counts.test}, total={total}{extra}")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Split report written to {report_path}")
