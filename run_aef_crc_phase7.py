"""
run_aef_crc_phase7.py

AEF-CRC Phase 7: Final Model + Evaluation Pipeline.

Continues after Phase 6's candidate comparison. Everything here is
config-driven (Task 4): config.backbone, the Phase-5 winning fused arm
(reused via run_aef_crc_phase6.parse_phase5_winner, not reimplemented),
config.classifier_name, config.feature_selection_method. None of
EfficientNet/RandomForest/BDA/RFE/GA/"no selection" is hardcoded as the only
possible choice anywhere below -- change any of those four config
values and this script's behavior changes with it, with no edit to
this file (see tests/test_phase7_final_model.py for a swap proof, same
style as Phase 6/7's existing registry-swap tests).

  1. run_final_cv()      -- Task 1: full Stratified K-Fold CV of the
     ONE configured (selector, classifier) pipeline -- not a Phase-6-
     style comparison of four candidates. Comprehensive metrics,
     never a "best fold" (see modules.evaluation.AggregatedMetrics).
  2. run_final_retrain()  -- Task 2: retrains that same configured
     pipeline once on the COMPLETE permitted training data (every
     non-augmented outer-train image, all folds reunified). Augmented
     images stay EXCLUDED for this retrain too -- see this function's
     docstring for why that is a deliberate decision, not an
     oversight. Never touches FoldPlan.holdout_test_records.
  3. Produces + persists a CalibrationHandoff (Task 3) built from
     predictions on FoldPlan.holdout_val_records -- the project's ONE
     true held-out calibration set (fold_loader.py's own docstring:
     "the held-out val split remains available later for
     calibration"), NOT a CV-internal validation slice and NOT the
     test set.

Step 0 gate check: reuses run_aef_crc_phase6.parse_phase5_winner
directly (not reimplemented). If no real Phase-5 winner / Phase-3
deep-feature cache exists, falls through to a clearly-labeled
PLUMBING-ONLY path -- same discipline as run_aef_crc_phase5.py /
run_aef_crc_phase6.py.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from config.config import get_config
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import load_frozen_fold_plan
from modules.aef_input_validator import compute_train_fold_class_weights
from modules.fusion import (
    ARMS, build_fusion_fold, build_fusion_final, build_classifier, compute_sample_weights,
    PHASE7_ALLOWED_CLASSIFIERS, validate_phase7_classifier,
)
from modules.feature_selection import run_selector, load_production_bda_mask
from modules.evaluation import compute_fold_metrics, aggregate_fold_metrics
from modules.experiment_config import representation_id, experiment_id
from modules.calibration_handoff import CalibrationHandoff, save_calibration_handoff
from run_aef_crc_phase6 import parse_phase5_winner, align_phase3_winner_config  # reuse, do not duplicate


MINORITY_CLASSES = ("Pityriasis_Rosea", "Seborrheic_Dermatitis")


def run_final_cv(config, plan, branches, source_experiment, label_prefix=""):
    """Task 1. ONE configured pipeline (config.feature_selection_method
    + config.classifier_name), evaluated across every CV fold -- never
    a subset, never a 'best fold' pick."""
    validate_phase7_classifier(config.classifier_name)
    from modules.handcrafted_features import HandcraftedFeatureExtractor
    extractor = HandcraftedFeatureExtractor(config)
    classes = config.target_classes

    fold_metrics = []
    for fold in plan.folds:
        data = build_fusion_fold(config, fold, branches, source_experiment, extractor)
        selection = run_selector(config.feature_selection_method, data, config, config.random_seed)
        mask = selection.selected_mask
        sw_train = compute_sample_weights(data.y_train, fold.class_weights)

        _, clf = build_classifier(config.random_seed, config.classifier_name)
        t0 = time.perf_counter()
        clf.fit(data.X_train[:, mask], data.y_train, sample_weight=sw_train)
        fit_seconds = time.perf_counter() - t0
        y_pred = clf.predict(data.X_val[:, mask])

        metrics = compute_fold_metrics(data.y_val, list(y_pred), classes, fold.fold_index)
        fold_metrics.append(metrics)
        print(f"{label_prefix}  fold {fold.fold_index}: {selection.selected_count} features | "
              f"macro_f1={metrics.macro_f1:.4f} | classifier_fit_time={fit_seconds:.2f}s")

    agg = aggregate_fold_metrics(fold_metrics)

    print(f"\n{label_prefix}--- Final CV results ({config.feature_selection_method} + {config.classifier_name}) ---")
    print(f"{label_prefix}Macro-F1:          {agg.macro_f1_mean:.4f} +/- {agg.macro_f1_std:.4f}")
    print(f"{label_prefix}Balanced Accuracy: {agg.balanced_accuracy_mean:.4f} +/- {agg.balanced_accuracy_std:.4f}")
    print(f"{label_prefix}MCC:               {agg.mcc_mean:.4f} +/- {agg.mcc_std:.4f}")
    print(f"{label_prefix}Accuracy:          {agg.accuracy_mean:.4f} +/- {agg.accuracy_std:.4f}")
    print(f"{label_prefix}Weighted F1:       {agg.weighted_f1_mean:.4f} +/- {agg.weighted_f1_std:.4f}")
    print(f"{label_prefix}Per-class (mean +/- std across folds; support = total across folds):")
    for cls in classes:
        tag = " (minority)" if cls in MINORITY_CLASSES else ""
        print(f"{label_prefix}  {cls}{tag}: "
              f"P={agg.per_class_precision_mean[cls]:.4f}+/-{agg.per_class_precision_std[cls]:.4f} "
              f"R={agg.per_class_recall_mean[cls]:.4f}+/-{agg.per_class_recall_std[cls]:.4f} "
              f"F1={agg.per_class_f1_mean[cls]:.4f}+/-{agg.per_class_f1_std[cls]:.4f} "
              f"support={agg.per_class_support_total[cls]}")
    print(f"{label_prefix}Confusion matrix (summed across {agg.n_folds} folds; rows=true, cols=predicted, order={agg.class_order}):")
    for cls, row in zip(agg.class_order, agg.confusion_sum):
        print(f"{label_prefix}  {cls:24s}: {list(int(v) for v in row)}")

    return agg


def run_final_retrain(config, plan, branches, source_experiment, label_prefix=""):
    """Task 2 + 3.

    Reunifies every CV fold's train+val to reconstruct the complete
    non-augmented outer-train set: fold 0's train_records+val_records
    already covers it exactly once (StratifiedKFold partitions
    `splittable` completely; any single fold's train+val reconstructs
    the same set regardless of which fold is picked -- verified by an
    assertion below, not just assumed).

    Augmented images (FoldPlan.excluded_augmented_records) are, by
    construction, never in any fold.train_records/val_records --
    reunification does NOT reintroduce them. This is a DELIBERATE
    decision for this retrain, not an accident of reuse:
    fold_loader.py's own docstring explicitly leaves "whether augmented
    images are reintroduced for a final retrain" as "a decision for the
    modeling phase... if that happens, the same cross-type leak risk
    against outer val/test applies there too and must be weighed then."
    Weighed here: an augmented image's parent-original lineage is
    unknown (no parent_original_id), so there is still no way to verify
    a given augmented image isn't a near-duplicate of something in
    FoldPlan.holdout_val_records or holdout_test_records. That risk is
    unchanged for a final retrain -- augmented images stay excluded.

    Evaluates on FoldPlan.holdout_val_records -- the project's ONE true
    held-out calibration set (splitter.py's val/, distinct from any
    CV-internal validation slice) -- and NEVER reads
    FoldPlan.holdout_test_records at all.
    """
    validate_phase7_classifier(config.classifier_name)
    from modules.handcrafted_features import HandcraftedFeatureExtractor
    extractor = HandcraftedFeatureExtractor(config)
    classes = config.target_classes

    final_train_records = plan.folds[0].train_records + plan.folds[0].val_records
    train_ids = [r.psd_id for r in final_train_records]
    assert len(set(train_ids)) == len(train_ids), "final retrain: duplicate PSD ID after fold reunification"
    splittable_count = sum(len(f.val_records) for f in plan.folds)  # folds' val_records partition `splittable` exactly once each
    assert len(final_train_records) == splittable_count, (
        f"final retrain: reunified {len(final_train_records)} records but folds' val_records sum to "
        f"{splittable_count} -- reunification did not reconstruct the full splittable set as expected."
    )
    print(f"{label_prefix}Final retrain set: {len(final_train_records)} non-augmented outer-train images "
          f"({len(plan.excluded_augmented_records)} augmented images EXCLUDED -- same leak-risk reasoning "
          f"as CV; see this function's docstring).")

    if not plan.holdout_val_records:
        raise ValueError(
            "FoldPlan.holdout_val_records is empty -- there is no calibration set to evaluate the final "
            "model against, so no calibration handoff can be produced."
        )

    class_weights = compute_train_fold_class_weights(
        [r.mapped_class for r in final_train_records], config.target_classes,
    )

    data = build_fusion_final(config, final_train_records, plan.holdout_val_records, branches, source_experiment, extractor)

    # Contract: Load persisted production BDA mask without recomputing
    if config.feature_selection_method in ("bda", "dragonfly", "dfa"):
        print(f"{label_prefix}Loading authoritative frozen production BDA mask from Phase 6 artifacts...")
        mask = load_production_bda_mask(config)
        assert mask.shape[0] == data.X_train.shape[1], (
            f"Production BDA mask dimension mismatch: {mask.shape[0]} != {data.X_train.shape[1]}"
        )
        selected_count = int(mask.sum())
        branch_retained = {}
        offset = 0
        for b_name, b_dim in data.branch_dims.items():
            branch_retained[b_name] = int(mask[offset : offset + b_dim].sum())
            offset += b_dim
    elif config.feature_selection_method == "none":
        mask = np.ones(data.X_train.shape[1], dtype=bool)
        selected_count = len(mask)
        branch_retained = dict(data.branch_dims)
    else:
        selection = run_selector(config.feature_selection_method, data, config, config.random_seed)
        mask = selection.selected_mask
        selected_count = selection.selected_count
        branch_retained = selection.branch_retained

    sw_train = compute_sample_weights(data.y_train, class_weights)

    _, clf = build_classifier(config.random_seed, config.classifier_name)
    t0 = time.perf_counter()
    clf.fit(data.X_train[:, mask], data.y_train, sample_weight=sw_train)
    fit_seconds = time.perf_counter() - t0

    y_pred = list(clf.predict(data.X_val[:, mask]))
    raw_probabilities = clf.predict_proba(data.X_val[:, mask])
    # predict_proba's column order is clf.classes_ -- not necessarily
    # config.target_classes, and not guaranteed to even COVER every
    # target class (a class absent from the training data never
    # appears in clf.classes_; this project's synthetic plumbing
    # fixture hits exactly this case). Reorder explicitly and fill any
    # missing class's column with 0.0 -- so CalibrationHandoff.
    # class_order is trustworthy without the next phase needing to
    # know this particular classifier's internal class ordering, and
    # without this crashing on a class the model never saw.
    prob_class_order = list(clf.classes_)
    probabilities = np.zeros((raw_probabilities.shape[0], len(classes)), dtype=raw_probabilities.dtype)
    for j, cls in enumerate(classes):
        if cls in prob_class_order:
            probabilities[:, j] = raw_probabilities[:, prob_class_order.index(cls)]

    calib_metrics = compute_fold_metrics(data.y_val, y_pred, classes, fold_index=-1)
    print(f"{label_prefix}Final model fit in {fit_seconds:.2f}s on {len(final_train_records)} images, "
          f"{selected_count} features selected ({branch_retained}).")
    print(f"{label_prefix}Calibration-set performance (splitter.py's held-out val/, NOT the test set): "
          f"macro_f1={calib_metrics.macro_f1:.4f}, balanced_accuracy={calib_metrics.balanced_accuracy:.4f}, "
          f"mcc={calib_metrics.mcc:.4f}")
    print(f"{label_prefix}FoldPlan.holdout_test_records ({len(plan.holdout_test_records)} images): "
          f"untouched -- not read, not indexed, not evaluated against.")

    feature_arm = next((name for name, b in ARMS.items() if b == branches), str(branches))
    backbone_ckpt = config.aef_crc_artifacts_dir / source_experiment / "fold_00" / "best_model.keras"
    from modules.calibration_handoff import get_active_run_id
    active_run = get_active_run_id(config) or ""
    v_res = DatasetFreezer(config).verify()
    freeze_hash = getattr(v_res, "freeze_hash", None)
    fold_hash = getattr(plan, "fold_plan_hash", None)
    return CalibrationHandoff(
        run_id=active_run,
        dataset_freeze_hash=freeze_hash,
        fold_plan_hash=fold_hash,
        representation_id=representation_id(config), experiment_id=experiment_id(config),
        classifier_name=config.classifier_name, feature_selection_method=config.feature_selection_method,
        random_seed=config.random_seed, source_experiment=source_experiment, feature_arm=feature_arm,
        class_order=classes,
        final_classifier=clf, selected_feature_mask=mask, branch_dims=data.branch_dims,
        calibration_psd_ids=data.psd_ids_val, calibration_true_labels=data.y_val,
        calibration_predicted_labels=y_pred, calibration_raw_probabilities=probabilities,
        hog_reducer=data.hog_reducer,
        feature_normalizers=data.feature_normalizers,
        backbone_checkpoint_path=str(backbone_ckpt) if backbone_ckpt.exists() else None,
    )


def _write_synthetic_deep_feature_cache(config, experiment_name, fold_index, split_name, records, backbone_dim, repr_id):
    """Local plumbing helper for THIS script's synthetic path only --
    same manifest schema run_aef_crc_phase5.py/phase6.py's own plumbing
    paths write (full schema, not a stripped-down stand-in), so this
    genuinely exercises modules.backbones.validate_deep_feature_cache,
    not something exempt from it. Not shared with phase5/6's own
    (separately maintained, untouched) inline copies of this same
    pattern -- see module docstring: Phase 4/5/6 are not being redone
    this phase."""
    import json
    rng = np.random.default_rng(0)
    out_dir = config.aef_crc_artifacts_dir / "deep_features" / experiment_name / f"fold_{fold_index:02d}" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    psd_ids = [r.psd_id for r in records]
    for r in records:
        np.save(out_dir / f"{r.psd_id}.npy", rng.normal(size=backbone_dim).astype(np.float32))
    manifest = {
        "experiment_name": experiment_name, "fold_index": fold_index, "split": split_name,
        "checkpoint_path": "SYNTHETIC PLUMBING DATA -- no real checkpoint", "n_features": len(psd_ids),
        "feature_dim": backbone_dim, "psd_ids": psd_ids, "backbone_name": config.backbone,
        "representation_id": repr_id,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    config, p3_winner = align_phase3_winner_config(get_config())
    validate_phase7_classifier(config.classifier_name)
    print("=== AEF-CRC Phase 7: Final Model + Evaluation Pipeline ===\n")
    print("--- Step 0: Gate check (reuses run_aef_crc_phase6.parse_phase5_winner) ---")
    winner, detail = parse_phase5_winner(config)
    print(detail)
    print(f"classifier_name={config.classifier_name!r}, feature_selection_method={config.feature_selection_method!r}, "
          f"source_experiment={p3_winner!r} (Task 4: config-driven, not re-derived, not hardcoded)\n")

    if winner:
        print(f"Phase-5 winning arm: {winner} (branches: {ARMS[winner]})")
        freeze_ok = DatasetFreezer(config).verify().matches
        print(f"DATASET FREEZE: {'PASS' if freeze_ok else 'FAIL'}")
        if not freeze_ok:
            return 1

        try:
            plan = load_frozen_fold_plan(config)
        except FileNotFoundError as exc:
            print(f"\n[PHASE 7 BLOCKER] Authoritative fold plan missing:\n  Expected: {config.aef_crc_reports_dir / 'fold_plan.csv'}")
            print(f"Error: {exc}")
            print("Phase 7 strictly requires the authoritative Phase 2 fold plan (reports/aef_crc/fold_plan.csv).")
            print("Run run_aef_crc_phase2.py first -- dynamic fold regeneration is strictly prohibited.")
            return 1

        from modules.fold_loader import validate_fold_plan
        validate_fold_plan(plan, config)
        if not plan.folds:
            print("ERROR: Loaded authoritative fold plan contains no folds.")
            return 1

        print("\n--- Task 1: Final CV evaluation ---")
        run_final_cv(config, plan, ARMS[winner], p3_winner)

        print("\n--- Task 2/3: Final retrain + calibration handoff ---")
        handoff = run_final_retrain(config, plan, ARMS[winner], p3_winner)

        out_path = config.aef_crc_phase7_artifacts_dir / "calibration_handoff.joblib"
        save_calibration_handoff(handoff, out_path)
        print(f"\nCalibration handoff artifact written to {out_path}")
        print("\nPHASE 7 RESULT: REAL EXECUTION COMPLETE.")
        return 0

    print("\nNo real Phase-5 winner available -- running PLUMBING-ONLY verification.")
    print("*** EVERYTHING BELOW IS A SYNTHETIC PLUMBING CHECK, NOT A RESEARCH RESULT ***\n")

    import shutil
    import dataclasses
    from modules.synthetic_fixtures import make_test_config, build_synthetic_dataset, run_pipeline_through_split
    from modules.backbones import get_backbone

    tmp_root = config.project_root / "_phase7_plumbing_tmp"
    test_config = make_test_config(tmp_root, seed=42)
    test_config = dataclasses.replace(test_config, feature_selection_method="none")
    build_synthetic_dataset(test_config)
    run_pipeline_through_split(test_config)

    plan = ImbalanceAwareFoldLoader(test_config, k=3).build()
    if not plan.folds:
        print("PLUMBING CHECK FAILED: no folds built from synthetic data.")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return 1
    if not plan.holdout_val_records:
        print("PLUMBING CHECK FAILED: synthetic dataset produced no held-out val records -- "
              "cannot exercise the calibration-handoff path.")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return 1

    backbone_dim = get_backbone(test_config.backbone).output_dim
    repr_id = representation_id(test_config)
    fake_experiment = "PLUMBING-FAKE"
    branches = ARMS["EfficientNet+GLCM+LBP+HOG"]

    for fold in plan.folds:
        for records, split_name in ((fold.train_records, "train"), (fold.val_records, "val")):
            _write_synthetic_deep_feature_cache(test_config, fake_experiment, fold.fold_index, split_name, records, backbone_dim, repr_id)
    # Final-retrain cache: fold_index=-1 sentinel, "final_train"/"calibration" splits (see build_fusion_final's docstring)
    final_train_records = plan.folds[0].train_records + plan.folds[0].val_records
    _write_synthetic_deep_feature_cache(test_config, fake_experiment, -1, "final_train", final_train_records, backbone_dim, repr_id)
    _write_synthetic_deep_feature_cache(test_config, fake_experiment, -1, "calibration", plan.holdout_val_records, backbone_dim, repr_id)

    print(f"Synthetic deep-feature cache written ({len(plan.folds)} CV folds + final_train/calibration). "
          f"Using the full combined arm (deep+glcm+lbp+hog) for plumbing.\n")

    try:
        print("--- Task 1: Final CV evaluation (plumbing) ---")
        run_final_cv(test_config, plan, branches, fake_experiment, label_prefix="[PLUMBING] ")
        print("\n--- Task 2/3: Final retrain + calibration handoff (plumbing) ---")
        handoff = run_final_retrain(test_config, plan, branches, fake_experiment, label_prefix="[PLUMBING] ")
        assert handoff.calibration_raw_probabilities.shape == (len(plan.holdout_val_records), len(config.target_classes))
        assert set(handoff.calibration_psd_ids) == {r.psd_id for r in plan.holdout_val_records}
        out_path = test_config.aef_crc_phase7_artifacts_dir / "calibration_handoff.joblib"
        save_calibration_handoff(handoff, out_path)
        from modules.calibration_handoff import load_calibration_handoff
        reloaded = load_calibration_handoff(out_path)
        assert reloaded.representation_id == handoff.representation_id
        assert np.array_equal(reloaded.calibration_raw_probabilities, handoff.calibration_raw_probabilities)
        print(f"\n[PLUMBING] Calibration handoff round-tripped through disk correctly ({out_path}).")
        plumbing_ok = True
    except Exception as exc:
        print(f"PLUMBING CHECK FAILED: {type(exc).__name__}: {exc}")
        plumbing_ok = False

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\nPLUMBING CHECK: {'PASSED' if plumbing_ok else 'FAILED'}")
    print("\nPhase-7 implementation complete; final model/evaluation pending real Phase-3/Phase-5 features.")
    return 0 if plumbing_ok else 1


if __name__ == "__main__":
    sys.exit(main())
