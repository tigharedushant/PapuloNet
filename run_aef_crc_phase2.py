"""
run_aef_crc_phase2.py

AEF-CRC Phase 2: dataset freeze + imbalance-aware fold loading.

Prerequisite: run_aef_crc_phase1.py must have reported PASSED against
the current output/06_final_split/. This script does not re-run
Phase 1's checks -- it assumes you already did, and its own freeze
step exists specifically to catch it if the dataset changes AFTER
that Phase 1 pass without anyone re-validating.

Usage:
    python main.py                  # PSD-HP harmonization
    python run_aef_crc_phase1.py    # dataset validation -- must PASS
    python run_aef_crc_phase2.py    # freeze + imbalance-aware folds (this script)
"""

from __future__ import annotations

import sys

from config.config import get_config
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import ImbalanceAwareFoldLoader


def main() -> int:
    config = get_config()

    print("=== AEF-CRC Phase 2: Dataset Freeze + Imbalance-Aware Fold Loading ===")
    print()

    freezer = DatasetFreezer(config)
    freeze_path = freezer.freeze(overwrite=False)
    verify_result = freezer.verify()

    print(f"Freeze file: {freeze_path}")
    print(f"Freeze verification: {'PASSED' if verify_result.matches else 'FAILED'}")
    if not verify_result.matches:
        for m in verify_result.mismatches[:20]:
            print(f"  [{m.kind}] {m.detail}")
        print()
        print("Dataset changed since freeze -- refusing to build folds against an unverified dataset.")
        return 1
    print()

    loader = ImbalanceAwareFoldLoader(config, k=5)
    plan = loader.build()

    if not plan.folds:
        print("No folds were built -- see log for the reason (likely: no train-split images found).")
        return 1

    print(f"K={plan.k} stratified fold(s) built over non-augmented TRAIN images only.")
    print(f"Held-out val (untouched by CV): {len(plan.holdout_val_records)} images")
    print(f"Held-out test (untouched, locked): {len(plan.holdout_test_records)} images")
    print(f"Augmented train images EXCLUDED from CV entirely: {len(plan.excluded_augmented_records)}")
    print(f"  LIMITATION: {plan.limitation_note}")
    print()

    for fold in plan.folds:
        print(f"--- Fold {fold.fold_index} ---")
        print(f"  fold-train: {len(fold.train_records)} images | fold-val: {len(fold.val_records)} images")
        print(f"  class weights (computed from THIS fold's training portion only):")
        for cls, w in fold.class_weights.items():
            w_str = f"{w:.4f}" if w == w else "NaN (zero samples of this class in this fold's training portion)"
            print(f"    {cls}: {w_str}")

    print()
    # PHASE 2 PASS CRITERIA (all must hold, not just "code ran without error"):
    #   - dataset freeze verifies (checked above, already returned 1 if not)
    #   - CV pool contains zero augmented images (structural guarantee of
    #     this module's design, verified here as a runtime assertion too --
    #     not just trusted from the code path that built it)
    #   - test/holdout-val were never read into any fold (structural: this
    #     module never receives them as CV input at all)
    from modules.fold_loader import save_fold_plan, validate_fold_plan
    validate_fold_plan(plan, config)
    plan_file = save_fold_plan(config, plan)
    print(f"Authoritative fold plan persisted to:    {plan_file}")
    print(f"Authoritative fold summary persisted to: {config.aef_crc_reports_dir / 'phase2_fold_summary.csv'}")

    cv_augmented_leak = any(
        r.source_type == "augmented"
        for fold in plan.folds
        for r in (fold.train_records + fold.val_records)
    )
    if cv_augmented_leak:
        print("PHASE 2 RESULT: FAILED -- an augmented image was found inside a CV fold. This should be structurally impossible; do not proceed.")
        return 1

    print("\nPHASE 2 RESULT: PASSED (with the documented residual limitation above -- not a claim of zero augmentation-related risk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
