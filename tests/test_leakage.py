"""
tests/test_leakage.py

Leakage/reproducibility test suite, revised after a second review round
that found the first version's CV fix ("append augmented to every
fold-train") was itself insufficient -- see fold_loader.py's docstring,
"Round 3", for the full counterexample that motivated this rewrite.

Runs the REAL pipeline modules (extractor, duplicate_detector,
quality_assessor, standardizer, metadata, splitter, fold_loader)
against a small synthetic dataset built inside a pytest tmp_path --
not a hand-simulated/mocked version of them. If these tests pass,
the actual shipped code is what was exercised.

Numbered to match the second review's required test list exactly:
  1. An augmentation group cannot cross outer train/val/test.
  2. An augmentation group cannot cross CV train/validation IF lineage
     is known. (N/A in this codebase -- no lineage-known code path
     exists at all, so this is documented as not-applicable rather
     than faked. See test_02's docstring.)
  3. If lineage is unknown, augmented images are excluded from CV.
  4. Non-augmented data behaves normally.
  5. Test is never used in CV.
  6. Held-out validation is never used in CV.
  7. Class weights are computed only from each fold's training data.
  8. Same seed reproduces identical folds.

Plus two retained from the first round that are still independently
useful and don't duplicate the above: every image in exactly one
outer split, and a different seed producing a different assignment
(proves the seed is actually wired through, not decorative).

Run with: PYTHONPATH=<repo>:<shims> python -m pytest tests/test_leakage.py -v
"""

from __future__ import annotations

import pytest

from modules.aef_input_validator import AEFInputLoader, compute_train_fold_class_weights
from modules.fold_loader import ImbalanceAwareFoldLoader
from modules.synthetic_fixtures import (
    make_test_config as _make_test_config,
    make_img as _make_img,
    build_synthetic_dataset as _build_synthetic_dataset,
    run_pipeline_through_split as _run_pipeline_through_split,
)





@pytest.fixture()
def pipeline_config(tmp_path):
    config = _make_test_config(tmp_path, seed=42)
    _build_synthetic_dataset(config)
    _run_pipeline_through_split(config)
    return config


# ============================================================
# Tests
# ============================================================

def test_01_augmented_never_in_val_or_test(pipeline_config):
    """No source_type='augmented' image appears anywhere except train/."""
    records = AEFInputLoader(pipeline_config).load()
    augmented = [r for r in records if r.source_type == "augmented"]
    assert augmented, "Test setup problem: no augmented images were found at all."
    assert all(r.split == "train" for r in augmented), (
        f"Found augmented image(s) outside train: "
        f"{[(r.psd_id, r.split) for r in augmented if r.split != 'train']}"
    )


def test_02_lineage_known_case_not_applicable(pipeline_config):
    """Required test 2: 'An augmentation group cannot cross CV
    train/validation IF lineage is known.' NOT APPLICABLE in this
    codebase: there is no parent_original_id anywhere in provenance,
    and no code path that branches on 'lineage is known' -- every
    augmented image is currently treated identically (excluded from
    CV, see test_03). This test exists to make that fact explicit and
    checkable, rather than silently omitting required-test #2: it
    asserts the N/A precondition (no lineage field exists) so that if
    parent_original_id is ever added, this test starts failing loudly
    and must be replaced with a real group-aware CV test instead of
    staying a stale pass."""
    records = AEFInputLoader(pipeline_config).load()
    assert not any(hasattr(r, "parent_original_id") for r in records), (
        "parent_original_id now exists on ImageRecord -- lineage IS knowable. "
        "This test is now stale: replace it with a real 'augmentation group "
        "stays together across CV folds' assertion instead of this N/A marker."
    )


def test_03_unknown_lineage_augmented_excluded_from_cv(pipeline_config):
    """Required test 3: with lineage unknown, augmented images must be
    excluded from CV entirely -- not merely kept out of the
    validation side (that was the first, insufficient fix). This is
    the actual correction from the second review round: augmented
    images must not appear in fold.train_records OR fold.val_records,
    for any fold."""
    plan = ImbalanceAwareFoldLoader(pipeline_config, k=3).build()
    assert plan.excluded_augmented_records, "Test setup problem: no augmented images were found to exclude."
    assert plan.limitation_note, "FoldPlan must carry an explicit, non-empty limitation note."

    for fold in plan.folds:
        train_aug = [r for r in fold.train_records if r.source_type == "augmented"]
        val_aug = [r for r in fold.val_records if r.source_type == "augmented"]
        assert not train_aug, f"Fold {fold.fold_index}: augmented image(s) found in fold-train (should be fully excluded): {[r.psd_id for r in train_aug]}"
        assert not val_aug, f"Fold {fold.fold_index}: augmented image(s) found in fold-val: {[r.psd_id for r in val_aug]}"


def test_04_non_augmented_dataset_behavior_unchanged(pipeline_config):
    """DermNet (source_type='original' for 100% of its images) must
    show normal ~70/15/15 proportions, unaffected by the augmented-image
    rule -- proving the fix is a genuine no-op for non-augmented sources."""
    records = AEFInputLoader(pipeline_config).load()
    dermnet = [r for r in records if r.source_dataset == "DermNet"]
    assert dermnet, "Test setup problem: no DermNet images found."
    by_split = {s: len([r for r in dermnet if r.split == s]) for s in ("train", "val", "test")}
    total = sum(by_split.values())
    train_fraction = by_split["train"] / total
    # Should be close to the nominal 0.70 -- allow slack for small-N rounding only.
    assert 0.55 <= train_fraction <= 0.85, f"DermNet train fraction {train_fraction:.2f} is far from nominal 0.70: {by_split}"


def test_05_same_seed_identical_folds(pipeline_config):
    """Re-running the fold loader with the same config/seed must produce
    byte-identical fold membership -- required for any result in this
    project to be reproducible."""
    plan_a = ImbalanceAwareFoldLoader(pipeline_config, k=3).build()
    plan_b = ImbalanceAwareFoldLoader(pipeline_config, k=3).build()
    for fold_a, fold_b in zip(plan_a.folds, plan_b.folds):
        ids_a = sorted(r.psd_id for r in fold_a.train_records)
        ids_b = sorted(r.psd_id for r in fold_b.train_records)
        assert ids_a == ids_b, f"Fold {fold_a.fold_index}: train membership differs between two runs with the same seed."


def test_06_different_seed_different_assignment(tmp_path):
    """A different random_seed must produce a genuinely different (but
    still internally valid) fold assignment -- proves the seed is
    actually wired through to StratifiedKFold, not silently ignored."""
    config_a = _make_test_config(tmp_path / "a", seed=42)
    _build_synthetic_dataset(config_a, seed_offset=0)
    _run_pipeline_through_split(config_a)
    plan_a = ImbalanceAwareFoldLoader(config_a, k=3).build()

    config_b = _make_test_config(tmp_path / "b", seed=999)
    _build_synthetic_dataset(config_b, seed_offset=0)
    _run_pipeline_through_split(config_b)
    plan_b = ImbalanceAwareFoldLoader(config_b, k=3).build()

    ids_a_fold0 = sorted(r.psd_id for r in plan_a.folds[0].val_records)
    ids_b_fold0 = sorted(r.psd_id for r in plan_b.folds[0].val_records)
    assert ids_a_fold0 != ids_b_fold0, "Different seeds produced identical fold-0 validation membership -- seed is not actually affecting the split."


def test_07_every_image_in_exactly_one_outer_split(pipeline_config):
    records = AEFInputLoader(pipeline_config).load()
    seen = {}
    for r in records:
        assert r.psd_id not in seen, f"PSD ID {r.psd_id} appears twice in the loaded records (splits: {seen[r.psd_id]}, {r.split})"
        seen[r.psd_id] = r.split
    assert all(s in ("train", "val", "test") for s in seen.values())


def test_08_augmentation_pool_confined_to_one_outer_split(pipeline_config):
    """NOTE: with no real per-group identity, 'every augmentation group
    belongs to exactly one outer split' reduces exactly to test 1 (every
    augmented image, individually, is in train) -- there are no
    multi-image groups to check independently of that. Kept as a
    separate test for direct traceability to the review's Task 7 list,
    not because it exercises different code than test 1."""
    records = AEFInputLoader(pipeline_config).load()
    augmented_splits = {r.split for r in records if r.source_type == "augmented"}
    assert augmented_splits == {"train"}, f"Augmented images span splits: {augmented_splits}"


def test_09_class_weights_computed_only_from_fold_training_data(pipeline_config):
    """Required test 7: class weights must be computed only from each
    fold's own training data -- never validation, test, or the excluded
    augmented pool. Verified by independently recomputing the formula
    from fold.train_records ONLY and checking it matches what the loader
    actually stored, not by re-reading the loader's internal logic."""
    plan = ImbalanceAwareFoldLoader(pipeline_config, k=3).build()
    target_classes = pipeline_config.target_classes

    for fold in plan.folds:
        expected = compute_train_fold_class_weights(
            [r.mapped_class for r in fold.train_records], target_classes
        )
        for cls in target_classes:
            e, a = expected[cls], fold.class_weights[cls]
            if e != e:  # NaN
                assert a != a, f"Fold {fold.fold_index}, {cls}: expected NaN, got {a}"
            else:
                assert abs(e - a) < 1e-9, f"Fold {fold.fold_index}, {cls}: expected {e}, got {a}"

        # And explicitly: nothing from val_records, holdout val/test, or
        # the excluded augmented pool could have influenced this fold's
        # weights, because none of those were passed into the formula at all.
        fold_train_ids = {r.psd_id for r in fold.train_records}
        contaminating_ids = (
            {r.psd_id for r in fold.val_records} |
            {r.psd_id for r in plan.holdout_val_records} |
            {r.psd_id for r in plan.holdout_test_records} |
            {r.psd_id for r in plan.excluded_augmented_records}
        )
        assert not (fold_train_ids & contaminating_ids), (
            f"Fold {fold.fold_index}: training IDs overlap with val/test/excluded-augmented IDs -- "
            f"class weights would be contaminated."
        )


def test_10_holdout_val_and_test_never_enter_cv(pipeline_config):
    records = AEFInputLoader(pipeline_config).load()
    holdout_val_ids = {r.psd_id for r in records if r.split == "val"}
    holdout_test_ids = {r.psd_id for r in records if r.split == "test"}

    plan = ImbalanceAwareFoldLoader(pipeline_config, k=3).build()
    for fold in plan.folds:
        cv_ids = {r.psd_id for r in fold.train_records} | {r.psd_id for r in fold.val_records}
        assert not (cv_ids & holdout_val_ids), f"Fold {fold.fold_index}: holdout val image(s) leaked into CV: {cv_ids & holdout_val_ids}"
        assert not (cv_ids & holdout_test_ids), f"Fold {fold.fold_index}: holdout test image(s) leaked into CV: {cv_ids & holdout_test_ids}"
