"""
run_aef_crc_phase8.py

AEF-CRC Phase 8: Probability Calibration.

Consumes the Phase-7 CalibrationHandoff (modules.calibration_handoff)
without rebuilding any of Phase 3-7's pipeline.

Frozen Calibration Protocol:
----------------------------
1. Calibration Hierarchy:
   - PRIMARY: Sigmoid / Platt scaling (per-class 1D logistic regression).
   - SECONDARY COMPARATOR: Isotonic regression (per-class non-parametric monotonic fit).
   - No adaptive selection, ECE threshold (e.g. >= 0.01), or data-dependent cascade
     is permitted to replace Platt as the primary calibration method.
   - Temperature scaling is NOT in the protocol.

2. Data Partitioning & Usage:
   - 123 val_calib: Used to fit the primary Platt calibrator and secondary Isotonic comparator.
   - 123 val_conf: Transformed using the PRIMARY Platt calibrator; handed to Phase 9.
   - 243 outer test: Locked final evaluation set (never used for fitting or selection).

3. Primary Final Calibration Metrics:
   - Expected Calibration Error (ECE, 10 bins) and Multiclass Brier score are evaluation metrics used to assess calibration quality.
   - Platt/sigmoid calibration fits a logistic sigmoid mapping using logistic regression/log-loss; ECE is an evaluation metric, not the optimization objective.
   - Reliability diagram tables reported as supporting diagnostic analysis.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.config import get_config
from run_aef_crc_phase6 import align_phase3_winner_config
from modules.calibration_handoff import (
    CalibrationHandoff, ConformalHandoff,
    load_calibration_handoff, save_conformal_handoff,
    validate_bda_mask_compatibility,
    validate_calibration_handoff_provenance,
)
from modules.calibration import (
    clip_probabilities, evaluate_calibration,
    fit_platt_scaling, apply_platt_scaling,
    fit_isotonic_scaling, apply_isotonic_scaling,
    partition_outer_validation,
)

MINORITY_CLASSES = ("Pityriasis_Rosea", "Seborrheic_Dermatitis")


def _print_metrics_row(m, label_prefix=""):
    print(f"{label_prefix}  {m.method:14s}: NLL={m.nll:.4f}  Brier={m.brier:.4f}  ECE={m.ece:.4f}  "
          f"macro_f1(diagnostic)={m.macro_f1:.4f}")


def run_calibration(
    config,
    handoff: CalibrationHandoff,
    label_prefix: str = "",
) -> ConformalHandoff:
    """
    Executes Phase 8 probability calibration:
    1. Validates BDA mask / classifier compatibility.
    2. Partitions outer validation into D_prob (123) and D_conf (123).
    3. Fits PRIMARY Platt scaling and SECONDARY Isotonic regression on D_prob.
    4. Evaluates ECE and Brier scores.
    5. Transforms D_conf using PRIMARY Platt-calibrated probabilities for Phase 9.
    """
    classes = handoff.class_order
    n_total = len(handoff.calibration_true_labels)
    print(f"{label_prefix}Total outer validation records: {n_total}")

    # Validate BDA mask
    validate_bda_mask_compatibility(handoff.selected_feature_mask, handoff.final_classifier)
    print(f"{label_prefix}BDA feature mask verified: selects {int(handoff.selected_feature_mask.sum())} features.")

    # 1. Partition Outer Validation into D_prob (Phase 8) and D_conf (Phase 9)
    partition = partition_outer_validation(
        psd_ids=handoff.calibration_psd_ids,
        labels=handoff.calibration_true_labels,
        class_order=classes,
        random_seed=config.random_seed,
        calib_fraction=0.5,
    )

    n_prob = len(partition.val_calib_indices)
    n_conf = len(partition.val_conf_indices)
    print(f"{label_prefix}Stratified 50/50 partition:")
    print(f"{label_prefix}  val_calib (fit probability calibrators): {n_prob} samples")
    print(f"{label_prefix}  val_conf  (fit conformal prediction):     {n_conf} samples")
    for cls in classes:
        print(f"{label_prefix}    {cls:22s}: val_calib={partition.per_class_calib_counts[cls]:2d}, "
              f"val_conf={partition.per_class_conf_counts[cls]:2d}")

    prob_idx = partition.val_calib_indices
    conf_idx = partition.val_conf_indices

    raw_prob = handoff.calibration_raw_probabilities[prob_idx]
    labels_prob = [handoff.calibration_true_labels[i] for i in prob_idx]
    psd_prob = [handoff.calibration_psd_ids[i] for i in prob_idx]

    raw_conf = handoff.calibration_raw_probabilities[conf_idx]
    labels_conf = [handoff.calibration_true_labels[i] for i in conf_idx]
    psd_conf = [handoff.calibration_psd_ids[i] for i in conf_idx]

    label_to_idx = {c: i for i, c in enumerate(classes)}
    true_indices_prob = np.array([label_to_idx[y] for y in labels_prob])

    if n_prob < config.calibration_min_samples:
        reason = f"Sample count N={n_prob} below the minimum threshold ({config.calibration_min_samples}) for reliable calibration; retaining uncalibrated probabilities."
        print(f"\n{label_prefix}MIN SAMPLES FALLBACK: uncalibrated")
        print(f"{label_prefix}  {reason}")
        return ConformalHandoff(
            representation_id=handoff.representation_id,
            experiment_id=handoff.experiment_id,
            classifier_name=handoff.classifier_name,
            feature_selection_method=handoff.feature_selection_method,
            random_seed=handoff.random_seed,
            class_order=classes,
            final_classifier=handoff.final_classifier,
            selected_feature_mask=handoff.selected_feature_mask,
            branch_dims=handoff.branch_dims,
            calibration_method="uncalibrated",
            calibration_method_reason=reason,
            platt_models=None,
            isotonic_models=None,
            calibration_psd_ids=psd_prob,
            calibration_true_labels=labels_prob,
            calibration_calibrated_probabilities=raw_prob,
            val_conf_psd_ids=psd_conf,
            val_conf_true_labels=labels_conf,
            val_conf_calibrated_probabilities=raw_conf,
            val_calib_metrics={},
            alpha=0.10,
            hog_reducer=getattr(handoff, "hog_reducer", None),
            feature_normalizers=getattr(handoff, "feature_normalizers", None),
            backbone_checkpoint_path=getattr(handoff, "backbone_checkpoint_path", None),
        )

    # 2. Fit Calibrators on val_calib
    uncal_metrics = evaluate_calibration(raw_prob, labels_prob, classes, "uncalibrated", config.calibration_ece_bins)

    # Primary: Platt Scaling
    platt_fit = fit_platt_scaling(raw_prob, true_indices_prob, classes, config.random_seed)
    platt_probs = apply_platt_scaling(raw_prob, platt_fit, classes)
    platt_metrics = evaluate_calibration(platt_probs, labels_prob, classes, "platt", config.calibration_ece_bins)

    # Secondary: Isotonic Regression Comparator
    iso_fit = fit_isotonic_scaling(raw_prob, true_indices_prob, classes)
    iso_probs = apply_isotonic_scaling(raw_prob, iso_fit, classes)
    iso_metrics = evaluate_calibration(iso_probs, labels_prob, classes, "isotonic", config.calibration_ece_bins)

    print(f"\n{label_prefix}--- val_calib Internal Development Metrics (N={n_prob}) ---")
    print(f"{label_prefix}(Primary metrics: ECE, Multiclass Brier score; Reliability diagrams are supporting analysis)")
    for m in (uncal_metrics, platt_metrics, iso_metrics):
        _print_metrics_row(m, label_prefix)

    # 3. Enforce Frozen Protocol: PRIMARY = Platt
    chosen_method = "platt"
    reason = "Frozen protocol: Platt (Sigmoid) scaling is the pre-registered primary method; Isotonic is secondary comparator."
    print(f"\n{label_prefix}FROZEN PROTOCOL DEPLOYMENT: {chosen_method}")
    print(f"{label_prefix}  {reason}")

    # 4. Transform val_conf using PRIMARY Platt Calibrator
    calibrated_prob_val_calib = platt_probs
    calibrated_prob_val_conf = apply_platt_scaling(raw_conf, platt_fit, classes)

    # Write Phase 8 reports
    out_dir = config.aef_crc_phase8_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "calibration_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "role", "nll", "brier", "ece", "macro_f1_diagnostic"])
        writer.writerow(["uncalibrated", "baseline", uncal_metrics.nll, uncal_metrics.brier, uncal_metrics.ece, uncal_metrics.macro_f1])
        writer.writerow(["platt", "PRIMARY", platt_metrics.nll, platt_metrics.brier, platt_metrics.ece, platt_metrics.macro_f1])
        writer.writerow(["isotonic", "SECONDARY", iso_metrics.nll, iso_metrics.brier, iso_metrics.ece, iso_metrics.macro_f1])

    with (out_dir / "reliability_diagram_uncalibrated.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_low", "bin_high", "count", "mean_confidence", "accuracy"])
        for b in uncal_metrics.reliability_bins:
            writer.writerow([b.bin_low, b.bin_high, b.count, b.mean_confidence, b.accuracy])

    with (out_dir / "reliability_diagram_calibrated_platt.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_low", "bin_high", "count", "mean_confidence", "accuracy"])
        for b in platt_metrics.reliability_bins:
            writer.writerow([b.bin_low, b.bin_high, b.count, b.mean_confidence, b.accuracy])

    with (out_dir / "validation_partition_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "val_calib_count", "val_conf_count", "total_outer_val_count"])
        for cls in classes:
            writer.writerow([
                cls,
                partition.per_class_calib_counts[cls],
                partition.per_class_conf_counts[cls],
                partition.per_class_calib_counts[cls] + partition.per_class_conf_counts[cls],
            ])

    print(f"{label_prefix}Reports written to {out_dir}")

    # 5. Assemble ConformalHandoff for Phase 9 using PRIMARY Platt probabilities
    return ConformalHandoff(
        run_id=getattr(handoff, "run_id", ""),
        dataset_freeze_hash=getattr(handoff, "dataset_freeze_hash", None),
        fold_plan_hash=getattr(handoff, "fold_plan_hash", None),
        representation_id=handoff.representation_id,
        experiment_id=handoff.experiment_id,
        classifier_name=handoff.classifier_name,
        feature_selection_method=handoff.feature_selection_method,
        random_seed=handoff.random_seed,
        class_order=classes,
        final_classifier=handoff.final_classifier,
        selected_feature_mask=handoff.selected_feature_mask,
        branch_dims=handoff.branch_dims,
        calibration_method="platt",
        calibration_method_reason=reason,
        platt_models=platt_fit.models,
        isotonic_models=iso_fit.models,
        calibration_psd_ids=psd_prob,
        calibration_true_labels=labels_prob,
        calibration_calibrated_probabilities=calibrated_prob_val_calib,
        val_conf_psd_ids=psd_conf,
        val_conf_true_labels=labels_conf,
        val_conf_calibrated_probabilities=calibrated_prob_val_conf,
        val_calib_metrics={
            "uncalibrated_ece": uncal_metrics.ece,
            "calibrated_ece": platt_metrics.ece,
            "calibrated_brier": platt_metrics.brier,
            "calibrated_nll": platt_metrics.nll,
            "secondary_isotonic_ece": iso_metrics.ece,
            "secondary_isotonic_brier": iso_metrics.brier,
        },
        alpha=0.10,
        hog_reducer=getattr(handoff, "hog_reducer", None),
        feature_normalizers=getattr(handoff, "feature_normalizers", None),
        backbone_checkpoint_path=getattr(handoff, "backbone_checkpoint_path", None),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AEF-CRC Phase 8: Probability Calibration")
    parser.add_argument(
        "--validate-framework",
        action="store_true",
        help="Run synthetic plumbing verification only (does not produce research results)",
    )
    args = parser.parse_args()

    config, _ = align_phase3_winner_config(get_config())
    print("=== AEF-CRC Phase 8: Probability Calibration ===\n")
    print("--- Step 0: Check Phase-7 CalibrationHandoff ---")

    handoff_path = config.aef_crc_phase7_artifacts_dir / "calibration_handoff.joblib"
    if handoff_path.exists():
        handoff = load_calibration_handoff(handoff_path)
        validate_calibration_handoff_provenance(handoff, config)
        print(f"Loaded real Phase-7 handoff from {handoff_path}")
        print(f"  representation_id={handoff.representation_id}")
        print(f"  classifier_name={handoff.classifier_name}, feature_selection_method={handoff.feature_selection_method}\n")

        conformal_handoff = run_calibration(config, handoff)

        out_path = config.aef_crc_phase8_artifacts_dir / "conformal_handoff.joblib"
        save_conformal_handoff(conformal_handoff, out_path)
        print(f"\nConformal handoff artifact written to {out_path}")
        print("\nPHASE 8 RESULT: REAL EXECUTION COMPLETE.")
        return 0

    if not args.validate_framework:
        print(
            f"\n[PHASE 8 BLOCKER] Prerequisite artifact missing:\n"
            f"  Expected: {handoff_path}\n\n"
            f"Reason: Real Phase 7 final training has not yet been executed by user command.\n"
            f"In accordance with pre-registered methodology:\n"
            f"  - The final production BDA mask and matching final Random Forest classifier\n"
            f"    must be loaded from the authorized Phase 7 handoff.\n"
            f"  - BDA feature selection must NEVER be recomputed dynamically.\n"
            f"  - To run a purely synthetic plumbing check without real models, pass --validate-framework.\n"
            f"\nExecution halted cleanly."
        )
        return 2

    # Plumbing verification path (strictly gated behind --validate-framework)
    print(f"No real Phase-7 handoff at {handoff_path} -- running PLUMBING-ONLY verification (--validate-framework).")
    print("*** EVERYTHING BELOW IS A SYNTHETIC PLUMBING CHECK, NOT A RESEARCH RESULT ***\n")

    import shutil
    from modules.synthetic_fixtures import make_test_config, build_synthetic_dataset, run_pipeline_through_split
    from modules.fold_loader import ImbalanceAwareFoldLoader
    from modules.backbones import get_backbone
    from modules.experiment_config import representation_id
    from modules.fusion import ARMS
    import run_aef_crc_phase7 as phase7

    tmp_root = config.project_root / "_phase8_plumbing_tmp"
    test_config = make_test_config(tmp_root, seed=42)
    build_synthetic_dataset(test_config)
    run_pipeline_through_split(test_config)
    plan = ImbalanceAwareFoldLoader(test_config, k=3).build()
    if not plan.folds or not plan.holdout_val_records:
        print("PLUMBING CHECK FAILED: no folds or no calibration records from synthetic data.")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return 1

    backbone_dim = get_backbone(test_config.backbone).output_dim
    repr_id = representation_id(test_config)
    experiment_name = "PLUMBING-FAKE"
    branches = ARMS["EfficientNet+GLCM+LBP+HOG"]
    for fold in plan.folds:
        for records, split_name in ((fold.train_records, "train"), (fold.val_records, "val")):
            phase7._write_synthetic_deep_feature_cache(test_config, experiment_name, fold.fold_index, split_name, records, backbone_dim, repr_id)
    final_train_records = plan.folds[0].train_records + plan.folds[0].val_records
    phase7._write_synthetic_deep_feature_cache(test_config, experiment_name, -1, "final_train", final_train_records, backbone_dim, repr_id)
    phase7._write_synthetic_deep_feature_cache(test_config, experiment_name, -1, "calibration", plan.holdout_val_records, backbone_dim, repr_id)

    try:
        print("[PLUMBING] Building a real CalibrationHandoff via run_aef_crc_phase7.run_final_retrain over synthetic data...\n")
        handoff = phase7.run_final_retrain(test_config, plan, branches, experiment_name, label_prefix="[PLUMBING] ")

        print("\n[PLUMBING] --- Calibrating (Platt Primary, Isotonic Secondary) ---")
        conformal_handoff = run_calibration(test_config, handoff, label_prefix="[PLUMBING] ")

        out_path = test_config.aef_crc_phase8_artifacts_dir / "conformal_handoff.joblib"
        save_conformal_handoff(conformal_handoff, out_path)
        from modules.calibration_handoff import load_conformal_handoff
        reloaded = load_conformal_handoff(out_path)
        assert reloaded.calibration_method == "platt"
        assert np.array_equal(reloaded.val_conf_calibrated_probabilities, conformal_handoff.val_conf_calibrated_probabilities)
        print(f"\n[PLUMBING] Conformal handoff round-tripped through disk correctly ({out_path}).")
        plumbing_ok = True
    except Exception as exc:
        print(f"PLUMBING CHECK FAILED: {type(exc).__name__}: {exc}")
        plumbing_ok = False

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\nPLUMBING CHECK: {'PASSED' if plumbing_ok else 'FAILED'}")
    return 0 if plumbing_ok else 1


if __name__ == "__main__":
    sys.exit(main())
