"""
run_aef_crc_phase8_v2.py

PapuloNet V2 Phase 8: Probability Calibration Pipeline.
Certified Winner: A7-BDA (1316-D: EfficientNet-B0 + GLCM + LBP + LAB)
Primary Calibration: Multiclass Temperature Scaling (Guo et al., 2017)

Scientific Invariants & Protocol:
---------------------------------
1. Rejection of V1 Independent Platt Scaling:
   - V1 independently fit 4 binary logistic regressions and normalized row sums.
   - In multiclass settings with class imbalance, independent binary Platt scaling
     scrambles the logit order, causing catastrophic argmax collapse (238/246 Psoriasis in V1).
   - V2 strictly rejects per-class Platt sigmoid calibration and independent binary calibrators.

2. Multiclass Temperature Scaling Guarantees:
   - Single scalar temperature T > 0 applied to pseudo-logits z = log(clip(P, eps, 1-eps)).
   - Scaled probabilities: P_hat(T) = softmax(z / T).
   - Since g(z) = z / T is strictly monotonic for any T > 0 and softmax preserves ordering,
     argmax_k P_hat_{i, k}(T) == argmax_k P_{i, k} for all samples i.
   - The number of changed class predictions is strictly IDENTICALLY ZERO.
   - Discrete classification performance (Accuracy, Balanced Accuracy, Macro-F1, MCC,
     and Confusion Matrix) is mathematically identical before and after calibration.
   - Calibration exclusively refines confidence estimates: minimizing Multiclass NLL,
     Expected Calibration Error (ECE), and Brier Score.

3. Objective & Optimization:
   - Fit scalar T > 0 strictly on the 123-sample held-out val_calib subset by minimizing
     Multiclass Negative Log Likelihood (NLL).
   - Optimization is completely UNWEIGHTED: calibration is probability post-processing,
     NOT classifier retraining (no class weighting, focal loss, or sampling).

4. Dual-Cohort Evaluation:
   - Fit cohort: val_calib (123 samples from Phase 7 V2 handoff).
   - Generalization cohort: val_conf (123 held-out samples from Phase 7 V2 handoff).
   - Pooled validation: 246 samples.
   - Locked test set (243 samples) remains completely untouched and unread.

5. Artifacts Produced:
   - artifacts/phase8_v2/conformal_handoff.joblib (for Phase 9 Conformal Prediction)
   - artifacts/phase8_v2/temperature_scaler.joblib
   - reports/phase8_v2/phase8_manifest.json
   - reports/phase8_v2/phase8_report.md
   - reports/phase8_v2/calibration_comparison.csv
   - reports/phase8_v2/reliability_diagram_uncalibrated.csv
   - reports/phase8_v2/reliability_diagram_calibrated_temperature.csv
   - reports/phase8_v2/validation_partition_summary.csv
   - reports/phase8_v2/per_class_calibration.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np

from config.config import get_config, PSDConfig
from modules.calibration import (
    clip_probabilities,
    negative_log_likelihood,
    brier_score,
    expected_calibration_error,
    per_class_ece,
    evaluate_calibration,
    MulticlassTemperatureScaler,
    TemperatureScaler,
    fit_temperature_scaling,
    apply_temperature_scaling,
    CalibrationMetrics,
    ReliabilityBin,
)
from modules.calibration_handoff import (
    CalibrationHandoff,
    ConformalHandoff,
    load_calibration_handoff,
    save_conformal_handoff,
    validate_bda_mask_compatibility,
    validate_calibration_handoff_provenance,
)
from modules.dataset_freeze import DatasetFreezer
from modules.evaluation import compute_fold_metrics
from modules.experiment_config import representation_id, experiment_id
from modules.fold_loader import load_frozen_fold_plan

logger = logging.getLogger("papulonet.phase8_v2")

# ======================================================================
# CONSTANTS & PROTOCOL INVARIANTS
# ======================================================================

P3_V2_REPR_ID = "efficientnet_b0_43d581b96f8ec368"
P3_V2_FOCAL_EXP = "P3-V2-Focal"
A7_ARM_ID = "A7"
A7_EXPECTED_DIM = 1316
CLASSIFIER_NAME = "random_forest"
FEATURE_SELECTOR_NAME = "bda"
CALIBRATION_METHOD = "temperature_scaling"
TEMPERATURE_BOUNDS = (0.05, 20.0)


def configure_phase8_v2(config: PSDConfig) -> PSDConfig:
    """Binds authoritative Phase 8 V2 configuration and paths."""
    return dataclasses.replace(
        config,
        aef_crc_phase8_reports_dir=config.aef_crc_phase8_v2_reports_dir,
        aef_crc_phase8_artifacts_dir=config.aef_crc_phase8_v2_artifacts_dir,
        preprocessing_mode="standard",
        training_time_augmentation="false",
        loss_name="categorical_focal_loss",
        focal_gamma=2.0,
        use_class_weights=False,
        adam_clipnorm=1.0,
        classifier_name=CLASSIFIER_NAME,
        feature_selection_method=FEATURE_SELECTOR_NAME,
        bda_feature_count_penalty=0.0,
    )


# ======================================================================
# STEP 0: GATE CHECK & UPSTREAM VERIFICATION
# ======================================================================

def gate_check_phase8_v2(config: PSDConfig) -> Tuple[bool, str, Dict[str, Any]]:
    """Strictly verifies upstream Phase 7 V2 certification manifest and artifacts."""
    p7_manifest_path = config.aef_crc_phase7_v2_reports_dir / "phase7_manifest.json"
    if not p7_manifest_path.exists():
        return False, f"Missing Phase 7 V2 manifest at {p7_manifest_path}. Phase 7 V2 must be completed first.", {}

    try:
        manifest = json.loads(p7_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse Phase 7 V2 manifest: {exc}", {}

    status = manifest.get("gate_evaluation", {}).get("decision") or manifest.get("status")
    if status != "CERTIFIED_WINNER":
        return False, f"Phase 7 V2 status is '{status}' (expected 'CERTIFIED_WINNER').", manifest

    repr_id = manifest.get("representation_id")
    if repr_id != P3_V2_REPR_ID:
        return False, f"Representation ID mismatch in Phase 7 manifest: '{repr_id}' != '{P3_V2_REPR_ID}'", manifest

    pipe_def = manifest.get("pipeline_definition", {})
    arm = pipe_def.get("arm_id") or manifest.get("arm_id")
    if arm != A7_ARM_ID:
        return False, f"Arm mismatch in Phase 7 manifest: '{arm}' != '{A7_ARM_ID}'", manifest

    clf = pipe_def.get("primary_classifier", {}).get("name") or manifest.get("classifier")
    if clf != CLASSIFIER_NAME:
        return False, f"Classifier mismatch in Phase 7 manifest: '{clf}' != '{CLASSIFIER_NAME}'", manifest

    sel = pipe_def.get("feature_selector", {}).get("name") or manifest.get("feature_selection_method")
    if sel != FEATURE_SELECTOR_NAME:
        return False, f"Feature selector mismatch in Phase 7 manifest: '{sel}' != '{FEATURE_SELECTOR_NAME}'", manifest

    handoff_path = config.aef_crc_phase7_v2_artifacts_dir / "calibration_handoff.joblib"
    if not handoff_path.exists():
        return False, f"Missing Phase 7 V2 CalibrationHandoff artifact at {handoff_path}", manifest

    # Hash verification
    freeze_record = config.aef_crc_reports_dir / "dataset_freeze.json"
    if freeze_record.exists():
        expected_freeze = hashlib.sha256(freeze_record.read_bytes()).hexdigest()
        actual_freeze = manifest.get("dataset_freeze_hash")
        if actual_freeze != expected_freeze:
            return False, f"Dataset freeze hash mismatch: manifest='{actual_freeze}' != current='{expected_freeze}'", manifest

    fold_record = config.reports_dir / "aef_crc" / "fold_plan.csv"
    if fold_record.exists():
        expected_fold = hashlib.sha256(fold_record.read_bytes()).hexdigest()
        actual_fold = manifest.get("fold_plan_hash")
        if actual_fold != expected_fold:
            return False, f"Fold plan hash mismatch: manifest='{actual_fold}' != current='{expected_fold}'", manifest

    return True, "Phase 7 V2 certified winner manifest and handoff verified successfully.", manifest


# ======================================================================
# TASK 1: PARTITION & PREFLIGHT VERIFICATION
# ======================================================================

def preflight_phase8_v2_partitions(
    config: PSDConfig,
    handoff: CalibrationHandoff,
) -> Tuple[bool, str]:
    """Verifies sample counts, partition disjointness, and isolation from dev/test cohorts."""
    classes = handoff.class_order
    n_calib = len(handoff.val_calib_psd_ids)
    n_conf = len(handoff.val_conf_psd_ids)
    n_total = len(handoff.calibration_psd_ids)

    if n_total != 246:
        return False, f"Expected 246 total outer validation samples, got {n_total}"
    if n_calib != 123:
        return False, f"Expected 123 val_calib samples, got {n_calib}"
    if n_conf != 123:
        return False, f"Expected 123 val_conf samples, got {n_conf}"

    calib_set = set(handoff.val_calib_psd_ids)
    conf_set = set(handoff.val_conf_psd_ids)
    if not calib_set.isdisjoint(conf_set):
        overlap = calib_set.intersection(conf_set)
        return False, f"Partition leakage: val_calib and val_conf overlap by {len(overlap)} samples!"

    # Verify isolation against development (1146), test (243), and quarantined (264)
    plan = load_frozen_fold_plan(config)
    dev_ids = {r.psd_id for f in plan.folds for r in f.train_records + f.val_records}
    test_ids = {r.psd_id for r in plan.holdout_test_records}
    quar_ids = {r.psd_id for r in plan.excluded_augmented_records}

    if not calib_set.isdisjoint(dev_ids):
        return False, "Data leakage: val_calib overlaps with 1146 development cohort!"
    if not conf_set.isdisjoint(dev_ids):
        return False, "Data leakage: val_conf overlaps with 1146 development cohort!"
    if not calib_set.isdisjoint(test_ids):
        return False, "Data leakage: val_calib overlaps with 243 locked test set!"
    if not conf_set.isdisjoint(test_ids):
        return False, "Data leakage: val_conf overlaps with 243 locked test set!"
    if not calib_set.isdisjoint(quar_ids):
        return False, "Data leakage: val_calib overlaps with 264 quarantined cohort!"
    if not conf_set.isdisjoint(quar_ids):
        return False, "Data leakage: val_conf overlaps with 264 quarantined cohort!"

    return True, f"Partition verification passed: val_calib=123, val_conf=123 (total=246); strictly disjoint from dev (1146), test (243), quarantined (264)."


# ======================================================================
# TASK 2 & 3: FIT & EVALUATE MULTICLASS TEMPERATURE SCALING
# ======================================================================

def run_phase8_v2_calibration(
    config: PSDConfig,
    handoff: CalibrationHandoff,
    label_prefix: str = "",
) -> Tuple[MulticlassTemperatureScaler, Dict[str, Any], ConformalHandoff]:
    """Fits Multiclass Temperature Scaling on val_calib (123) and evaluates on val_calib, val_conf, and pooled."""
    classes = handoff.class_order
    label_to_idx = {c: i for i, c in enumerate(classes)}

    val_calib_raw = handoff.val_calib_raw_probabilities
    val_calib_labels = handoff.val_calib_true_labels
    val_calib_idx = np.array([label_to_idx[y] for y in val_calib_labels])

    val_conf_raw = handoff.val_conf_raw_probabilities
    val_conf_labels = handoff.val_conf_true_labels
    val_conf_idx = np.array([label_to_idx[y] for y in val_conf_labels])

    pooled_raw = handoff.calibration_raw_probabilities
    pooled_labels = handoff.calibration_true_labels
    pooled_idx = np.array([label_to_idx[y] for y in pooled_labels])

    print(f"\n{label_prefix}======================================================================")
    print(f"{label_prefix}TASK 2: FIT MULTICLASS TEMPERATURE SCALING (val_calib, N=123)")
    print(f"{label_prefix}======================================================================")

    t0_fit = time.perf_counter()
    scaler = fit_temperature_scaling(val_calib_raw, val_calib_idx, bounds=TEMPERATURE_BOUNDS)
    fit_time = time.perf_counter() - t0_fit

    T_opt = scaler.temperature
    print(f"{label_prefix}Optimal Temperature T:    {T_opt:.4f}")
    print(f"{label_prefix}Optimization Method:      scipy.optimize.minimize_scalar (bounded in {TEMPERATURE_BOUNDS})")
    print(f"{label_prefix}Loss Objective:           Unweighted Multiclass Negative Log Likelihood (NLL)")
    print(f"{label_prefix}Fit Time:                 {fit_time:.4f}s")
    print(f"{label_prefix}val_calib NLL (before):   {scaler.nll_before_:.4f}")
    print(f"{label_prefix}val_calib NLL (after):    {scaler.nll_after_:.4f}")
    print(f"{label_prefix}Fit Status:               {scaler.fit_status_}")

    assert T_opt > 0, f"Temperature must be positive, got {T_opt}"
    assert np.isfinite(T_opt), f"Temperature must be finite, got {T_opt}"

    # Apply scaling to all partitions
    val_calib_cal = scaler.predict_proba(val_calib_raw)
    val_conf_cal = scaler.predict_proba(val_conf_raw)
    pooled_cal = scaler.predict_proba(pooled_raw)

    print(f"\n{label_prefix}======================================================================")
    print(f"{label_prefix}TASK 3: DUAL-COHORT EVALUATION & CLASS DECISION INVARIANCE")
    print(f"{label_prefix}======================================================================")

    # Invariance check: verify argmax is strictly unchanged
    mismatches_calib = int(np.sum(np.argmax(val_calib_raw, axis=1) != np.argmax(val_calib_cal, axis=1)))
    mismatches_conf = int(np.sum(np.argmax(val_conf_raw, axis=1) != np.argmax(val_conf_cal, axis=1)))
    mismatches_pooled = int(np.sum(np.argmax(pooled_raw, axis=1) != np.argmax(pooled_cal, axis=1)))

    print(f"{label_prefix}Argmax Invariance Checks (Target: Changed Predictions == 0):")
    print(f"{label_prefix}  val_calib (N=123): {mismatches_calib} changed decisions")
    print(f"{label_prefix}  val_conf  (N=123): {mismatches_conf} changed decisions")
    print(f"{label_prefix}  pooled    (N=246): {mismatches_pooled} changed decisions")

    if mismatches_calib != 0 or mismatches_conf != 0 or mismatches_pooled != 0:
        raise RuntimeError("FATAL: Temperature scaling violated strict class decision invariance!")

    # Evaluation metrics
    n_bins = config.calibration_ece_bins
    eval_calib_uncal = evaluate_calibration(val_calib_raw, val_calib_labels, classes, "uncalibrated", n_bins, val_calib_raw)
    eval_calib_cal = evaluate_calibration(val_calib_cal, val_calib_labels, classes, "temperature_scaling", n_bins, val_calib_raw)

    eval_conf_uncal = evaluate_calibration(val_conf_raw, val_conf_labels, classes, "uncalibrated", n_bins, val_conf_raw)
    eval_conf_cal = evaluate_calibration(val_conf_cal, val_conf_labels, classes, "temperature_scaling", n_bins, val_conf_raw)

    eval_pooled_uncal = evaluate_calibration(pooled_raw, pooled_labels, classes, "uncalibrated", n_bins, pooled_raw)
    eval_pooled_cal = evaluate_calibration(pooled_cal, pooled_labels, classes, "temperature_scaling", n_bins, pooled_raw)

    print(f"\n{label_prefix}--- val_calib Fit Set Performance (N=123) ---")
    print(f"{label_prefix}  Uncalibrated:        NLL={eval_calib_uncal.nll:.4f}  Brier={eval_calib_uncal.brier:.4f}  ECE={eval_calib_uncal.ece:.4f}  Macro-F1={eval_calib_uncal.macro_f1:.4f}")
    print(f"{label_prefix}  Temperature Scaling: NLL={eval_calib_cal.nll:.4f}  Brier={eval_calib_cal.brier:.4f}  ECE={eval_calib_cal.ece:.4f}  Macro-F1={eval_calib_cal.macro_f1:.4f}")

    print(f"\n{label_prefix}--- val_conf Held-Out Generalization Set Performance (N=123) ---")
    print(f"{label_prefix}  Uncalibrated:        NLL={eval_conf_uncal.nll:.4f}  Brier={eval_conf_uncal.brier:.4f}  ECE={eval_conf_uncal.ece:.4f}  Macro-F1={eval_conf_uncal.macro_f1:.4f}")
    print(f"{label_prefix}  Temperature Scaling: NLL={eval_conf_cal.nll:.4f}  Brier={eval_conf_cal.brier:.4f}  ECE={eval_conf_cal.ece:.4f}  Macro-F1={eval_conf_cal.macro_f1:.4f}")

    print(f"\n{label_prefix}--- Pooled Outer Validation Performance (N=246) ---")
    print(f"{label_prefix}  Uncalibrated:        NLL={eval_pooled_uncal.nll:.4f}  Brier={eval_pooled_uncal.brier:.4f}  ECE={eval_pooled_uncal.ece:.4f}  Macro-F1={eval_pooled_uncal.macro_f1:.4f}")
    print(f"{label_prefix}  Temperature Scaling: NLL={eval_pooled_cal.nll:.4f}  Brier={eval_pooled_cal.brier:.4f}  ECE={eval_pooled_cal.ece:.4f}  Macro-F1={eval_pooled_cal.macro_f1:.4f}")

    # Build ConformalHandoff for Phase 9
    conformal_handoff = ConformalHandoff(
        representation_id=handoff.representation_id,
        experiment_id=handoff.experiment_id,
        classifier_name=handoff.classifier_name,
        feature_selection_method=handoff.feature_selection_method,
        random_seed=handoff.random_seed,
        class_order=classes,
        final_classifier=handoff.final_classifier,
        selected_feature_mask=handoff.selected_feature_mask,
        branch_dims=handoff.branch_dims,
        run_id=getattr(handoff, "run_id", ""),
        dataset_freeze_hash=getattr(handoff, "dataset_freeze_hash", None),
        fold_plan_hash=getattr(handoff, "fold_plan_hash", None),
        calibration_method="temperature_scaling",
        calibration_method_reason="Multiclass Temperature Scaling: preserves base RF class predictions exactly (argmax invariant) while optimizing probability confidence.",
        platt_models=None,
        isotonic_models=None,
        temperature_scaler=scaler,
        temperature=T_opt,
        calibration_psd_ids=handoff.val_calib_psd_ids,
        calibration_true_labels=handoff.val_calib_true_labels,
        calibration_calibrated_probabilities=val_calib_cal,
        val_conf_psd_ids=handoff.val_conf_psd_ids,
        val_conf_true_labels=handoff.val_conf_true_labels,
        val_conf_calibrated_probabilities=val_conf_cal,
        val_calib_metrics={
            "calib_uncalibrated_nll": eval_calib_uncal.nll,
            "calib_calibrated_nll": eval_calib_cal.nll,
            "calib_uncalibrated_brier": eval_calib_uncal.brier,
            "calib_calibrated_brier": eval_calib_cal.brier,
            "calib_uncalibrated_ece": eval_calib_uncal.ece,
            "calib_calibrated_ece": eval_calib_cal.ece,
            "conf_uncalibrated_nll": eval_conf_uncal.nll,
            "conf_calibrated_nll": eval_conf_cal.nll,
            "conf_uncalibrated_brier": eval_conf_uncal.brier,
            "conf_calibrated_brier": eval_conf_cal.brier,
            "conf_uncalibrated_ece": eval_conf_uncal.ece,
            "conf_calibrated_ece": eval_conf_cal.ece,
            "temperature": T_opt,
            "changed_predictions_calib": mismatches_calib,
            "changed_predictions_conf": mismatches_conf,
            "changed_predictions_pooled": mismatches_pooled,
        },
        alpha=config.conformal_alpha,
        hog_reducer=getattr(handoff, "hog_reducer", None),
        feature_normalizers=getattr(handoff, "feature_normalizers", None),
        backbone_checkpoint_path=getattr(handoff, "backbone_checkpoint_path", None),
    )

    results = {
        "temperature": T_opt,
        "fit_time_sec": fit_time,
        "scaler": scaler,
        "eval_calib_uncal": eval_calib_uncal,
        "eval_calib_cal": eval_calib_cal,
        "eval_conf_uncal": eval_conf_uncal,
        "eval_conf_cal": eval_conf_cal,
        "eval_pooled_uncal": eval_pooled_uncal,
        "eval_pooled_cal": eval_pooled_cal,
        "val_calib_cal": val_calib_cal,
        "val_conf_cal": val_conf_cal,
        "pooled_cal": pooled_cal,
        "mismatches": {
            "val_calib": mismatches_calib,
            "val_conf": mismatches_conf,
            "pooled": mismatches_pooled,
        },
    }

    return scaler, results, conformal_handoff


# ======================================================================
# TASK 5: REPORT GENERATION
# ======================================================================

def write_phase8_v2_reports(
    config: PSDConfig,
    handoff: CalibrationHandoff,
    results: Dict[str, Any],
    conformal_handoff: ConformalHandoff,
    p7_manifest: Dict[str, Any],
    label_prefix: str = "",
) -> Dict[str, Path]:
    """Generates all comprehensive reports and manifests in reports/phase8_v2/."""
    out_dir = config.aef_crc_phase8_v2_reports_dir
    art_dir = config.aef_crc_phase8_v2_artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    classes = handoff.class_order
    T_opt = results["temperature"]
    c_uncal = results["eval_calib_uncal"]
    c_cal = results["eval_calib_cal"]
    cf_uncal = results["eval_conf_uncal"]
    cf_cal = results["eval_conf_cal"]
    p_uncal = results["eval_pooled_uncal"]
    p_cal = results["eval_pooled_cal"]

    generated_files: Dict[str, Path] = {}

    # 1. calibration_comparison.csv
    comp_csv = out_dir / "calibration_comparison.csv"
    with comp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["partition", "method", "n_samples", "nll", "brier", "ece", "macro_f1", "changed_predictions", "temperature"])
        writer.writerow(["val_calib", "uncalibrated", 123, f"{c_uncal.nll:.6f}", f"{c_uncal.brier:.6f}", f"{c_uncal.ece:.6f}", f"{c_uncal.macro_f1:.6f}", 0, 1.0])
        writer.writerow(["val_calib", "temperature_scaling", 123, f"{c_cal.nll:.6f}", f"{c_cal.brier:.6f}", f"{c_cal.ece:.6f}", f"{c_cal.macro_f1:.6f}", results["mismatches"]["val_calib"], f"{T_opt:.6f}"])
        writer.writerow(["val_conf", "uncalibrated", 123, f"{cf_uncal.nll:.6f}", f"{cf_uncal.brier:.6f}", f"{cf_uncal.ece:.6f}", f"{cf_uncal.macro_f1:.6f}", 0, 1.0])
        writer.writerow(["val_conf", "temperature_scaling", 123, f"{cf_cal.nll:.6f}", f"{cf_cal.brier:.6f}", f"{cf_cal.ece:.6f}", f"{cf_cal.macro_f1:.6f}", results["mismatches"]["val_conf"], f"{T_opt:.6f}"])
        writer.writerow(["val_pooled", "uncalibrated", 246, f"{p_uncal.nll:.6f}", f"{p_uncal.brier:.6f}", f"{p_uncal.ece:.6f}", f"{p_uncal.macro_f1:.6f}", 0, 1.0])
        writer.writerow(["val_pooled", "temperature_scaling", 246, f"{p_cal.nll:.6f}", f"{p_cal.brier:.6f}", f"{p_cal.ece:.6f}", f"{p_cal.macro_f1:.6f}", results["mismatches"]["pooled"], f"{T_opt:.6f}"])
    generated_files["calibration_comparison"] = comp_csv

    # 2. reliability_diagram_uncalibrated.csv (on val_conf held-out)
    rel_uncal_csv = out_dir / "reliability_diagram_uncalibrated.csv"
    with rel_uncal_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_idx", "bin_low", "bin_high", "count", "mean_confidence", "accuracy"])
        for idx, b in enumerate(cf_uncal.reliability_bins):
            writer.writerow([idx, f"{b.bin_low:.2f}", f"{b.bin_high:.2f}", b.count, f"{b.mean_confidence:.6f}", f"{b.accuracy:.6f}"])
    generated_files["reliability_diagram_uncalibrated"] = rel_uncal_csv

    # 3. reliability_diagram_calibrated_temperature.csv (on val_conf held-out)
    rel_cal_csv = out_dir / "reliability_diagram_calibrated_temperature.csv"
    with rel_cal_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_idx", "bin_low", "bin_high", "count", "mean_confidence", "accuracy"])
        for idx, b in enumerate(cf_cal.reliability_bins):
            writer.writerow([idx, f"{b.bin_low:.2f}", f"{b.bin_high:.2f}", b.count, f"{b.mean_confidence:.6f}", f"{b.accuracy:.6f}"])
    generated_files["reliability_diagram_calibrated_temperature"] = rel_cal_csv

    # 4. validation_partition_summary.csv
    part_csv = out_dir / "validation_partition_summary.csv"
    with part_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "val_calib_count", "val_conf_count", "total_count", "class_ratio_percent"])
        n_tot = len(handoff.calibration_true_labels)
        for cls in classes:
            c_cnt = sum(1 for y in handoff.val_calib_true_labels if y == cls)
            cf_cnt = sum(1 for y in handoff.val_conf_true_labels if y == cls)
            tot = c_cnt + cf_cnt
            ratio = (tot / n_tot) * 100.0
            writer.writerow([cls, c_cnt, cf_cnt, tot, f"{ratio:.2f}"])
    generated_files["validation_partition_summary"] = part_csv

    # 5. per_class_calibration.csv
    per_cls_csv = out_dir / "per_class_calibration.csv"
    with per_cls_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "val_calib_uncal_ece", "val_calib_cal_ece", "val_conf_uncal_ece", "val_conf_cal_ece"])
        for cls in classes:
            writer.writerow([
                cls,
                f"{c_uncal.per_class_ece.get(cls, 0.0):.6f}",
                f"{c_cal.per_class_ece.get(cls, 0.0):.6f}",
                f"{cf_uncal.per_class_ece.get(cls, 0.0):.6f}",
                f"{cf_cal.per_class_ece.get(cls, 0.0):.6f}",
            ])
    generated_files["per_class_calibration"] = per_cls_csv

    # 6. phase8_manifest.json
    now_utc = datetime.now(timezone.utc).isoformat()
    p8_run_id = f"AEFCRC_P8_V2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(T_opt).encode()).hexdigest()[:8]}"

    manifest_data = {
        "phase": "phase8_v2",
        "run_id": p8_run_id,
        "timestamp_utc": now_utc,
        "status": "CALIBRATION_CERTIFIED",
        "pipeline": {
            "arm_id": A7_ARM_ID,
            "classifier": CLASSIFIER_NAME,
            "feature_selection_method": FEATURE_SELECTOR_NAME,
            "representation_id": P3_V2_REPR_ID,
            "source_experiment": P3_V2_FOCAL_EXP,
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "temperature": T_opt,
            "bounds": list(TEMPERATURE_BOUNDS),
            "optimization_objective": "unweighted_multiclass_negative_log_likelihood",
            "fit_cohort_size": 123,
            "generalization_cohort_size": 123,
            "total_outer_validation_size": 246,
            "fit_time_seconds": results["fit_time_sec"],
            "class_decision_invariance": {
                "val_calib_changed_decisions": results["mismatches"]["val_calib"],
                "val_conf_changed_decisions": results["mismatches"]["val_conf"],
                "pooled_changed_decisions": results["mismatches"]["pooled"],
                "invariant_preserved": True,
            },
        },
        "metrics": {
            "val_calib": {
                "uncalibrated": {"nll": c_uncal.nll, "brier": c_uncal.brier, "ece": c_uncal.ece, "macro_f1": c_uncal.macro_f1},
                "calibrated": {"nll": c_cal.nll, "brier": c_cal.brier, "ece": c_cal.ece, "macro_f1": c_cal.macro_f1},
                "delta_nll": c_cal.nll - c_uncal.nll,
                "delta_ece": c_cal.ece - c_uncal.ece,
            },
            "val_conf": {
                "uncalibrated": {"nll": cf_uncal.nll, "brier": cf_uncal.brier, "ece": cf_uncal.ece, "macro_f1": cf_uncal.macro_f1},
                "calibrated": {"nll": cf_cal.nll, "brier": cf_cal.brier, "ece": cf_cal.ece, "macro_f1": cf_cal.macro_f1},
                "delta_nll": cf_cal.nll - cf_uncal.nll,
                "delta_ece": cf_cal.ece - cf_uncal.ece,
            },
            "val_pooled": {
                "uncalibrated": {"nll": p_uncal.nll, "brier": p_uncal.brier, "ece": p_uncal.ece, "macro_f1": p_uncal.macro_f1},
                "calibrated": {"nll": p_cal.nll, "brier": p_cal.brier, "ece": p_cal.ece, "macro_f1": p_cal.macro_f1},
                "delta_nll": p_cal.nll - p_uncal.nll,
                "delta_ece": p_cal.ece - p_uncal.ece,
            },
        },
        "provenance": {
            "upstream_phase7_run_id": p7_manifest.get("run_id"),
            "dataset_freeze_hash": p7_manifest.get("dataset_freeze_hash"),
            "fold_plan_hash": p7_manifest.get("fold_plan_hash"),
            "locked_test_untouched": True,
            "locked_test_records_count": 243,
        },
        "artifacts": {
            "conformal_handoff_joblib": str(art_dir / "conformal_handoff.joblib"),
            "temperature_scaler_joblib": str(art_dir / "temperature_scaler.joblib"),
        },
    }

    manifest_json = out_dir / "phase8_manifest.json"
    manifest_json.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    generated_files["phase8_manifest"] = manifest_json

    # 7. phase8_report.md
    report_md = out_dir / "phase8_report.md"
    report_text = f"""# PapuloNet V2 Phase 8: Probability Calibration Report
**Pipeline**: A7-BDA (1316-D: EfficientNet-B0 + GLCM + LBP + LAB, Random Forest 300 Trees)  
**Primary Calibration Method**: Multiclass Temperature Scaling (Guo et al., 2017)  
**Run ID**: `{p8_run_id}`  
**Date**: {now_utc}  

---

## 1. Executive Summary & Calibration Outcome

Phase 8 executes **Multiclass Temperature Scaling** to calibrate predicted class probabilities from the frozen Phase 7 V2 winner pipeline (A7-BDA Random Forest).

### Core Scientific Findings:
1. **Strict Class Decision Invariance**:
   - The number of changed class predictions is **exactly 0** across all 246 validation images (0 on `val_calib`, 0 on `val_conf`).
   - Macro-F1 ({cf_cal.macro_f1:.4f}), Balanced Accuracy, MCC, and the confusion matrix are mathematically preserved.
2. **Rejection of V1 Platt Scaling**:
   - V1 independently fit 4 binary logistic regressions, which scrambled multiclass logit rankings and collapsed 238/246 predictions to Psoriasis.
   - Temperature scaling applies a single positive scalar temperature $T > 0$ over pseudo-logits, mathematically guaranteeing that $\\operatorname{{argmax}} \\hat{{p}} \\equiv \\operatorname{{argmax}} p$.
3. **Optimal Temperature**:
   - $T = {T_opt:.4f}$ (fitted on 123 `val_calib` images by unweighted NLL minimization).
   - Fit Time: {results['fit_time_sec']:.4f}s.
4. **Generalization on Held-Out `val_conf` (N=123)**:
   - **NLL**: {cf_uncal.nll:.4f} $\\rightarrow$ **{cf_cal.nll:.4f}** ({cf_cal.nll - cf_uncal.nll:+.4f})
   - **ECE (10 bins)**: {cf_uncal.ece:.4f} $\\rightarrow$ **{cf_cal.ece:.4f}** ({cf_cal.ece - cf_uncal.ece:+.4f})
   - **Brier Score**: {cf_uncal.brier:.4f} $\\rightarrow$ **{cf_cal.brier:.4f}** ({cf_cal.brier - cf_uncal.brier:+.4f})

---

## 2. Calibration Comparison Table

| Partition | Set Role | Method | N | NLL | Brier Score | ECE (10 bins) | Macro-F1 | Changed Decisions |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `val_calib` | Fit Set | Uncalibrated | 123 | {c_uncal.nll:.4f} | {c_uncal.brier:.4f} | {c_uncal.ece:.4f} | {c_uncal.macro_f1:.4f} | 0 |
| `val_calib` | Fit Set | Temperature Scaling ($T={T_opt:.4f}$) | 123 | **{c_cal.nll:.4f}** | **{c_cal.brier:.4f}** | **{c_cal.ece:.4f}** | **{c_cal.macro_f1:.4f}** | **0** |
| `val_conf` | Generalization | Uncalibrated | 123 | {cf_uncal.nll:.4f} | {cf_uncal.brier:.4f} | {cf_uncal.ece:.4f} | {cf_uncal.macro_f1:.4f} | 0 |
| `val_conf` | Generalization | Temperature Scaling ($T={T_opt:.4f}$) | 123 | **{cf_cal.nll:.4f}** | **{cf_cal.brier:.4f}** | **{cf_cal.ece:.4f}** | **{cf_cal.macro_f1:.4f}** | **0** |
| `val_pooled`| Pooled Cohort | Uncalibrated | 246 | {p_uncal.nll:.4f} | {p_uncal.brier:.4f} | {p_uncal.ece:.4f} | {p_uncal.macro_f1:.4f} | 0 |
| `val_pooled`| Pooled Cohort | Temperature Scaling ($T={T_opt:.4f}$) | 246 | **{p_cal.nll:.4f}** | **{p_cal.brier:.4f}** | **{p_cal.ece:.4f}** | **{p_cal.macro_f1:.4f}** | **0** |

---

## 3. Reliability Diagram Analysis (`val_conf`, N=123)

### Calibrated 10-Bin Breakdown:
| Bin Index | Range | Count | Mean Confidence | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
"""
    for idx, b in enumerate(cf_cal.reliability_bins):
        report_text += f"| {idx} | [{b.bin_low:.2f}, {b.bin_high:.2f}) | {b.count} | {b.mean_confidence:.4f} | {b.accuracy:.4f} |\n"

    report_text += f"""
---

## 4. Verification of Invariants & Data Isolation

1. **Class Decision Invariance**:
   - $\\operatorname{{argmax}}_{{k}} \\hat{{p}}_{{i, k}} \\equiv \\operatorname{{argmax}}_{{k}} p_{{i, k}}$ verified across all 246 validation instances.
   - Total changed predictions: **0**.
2. **Strict Data Partitioning**:
   - Fit set `val_calib`: 123 images.
   - Held-out evaluation set `val_conf`: 123 images.
   - Leakage: strictly 0 images overlapping between `val_calib` and `val_conf`.
   - Development cohort (1146 images): untouched.
   - Locked test set (243 images): strictly untouched and unread.
3. **Upstream Provenance**:
   - Phase 7 V2 Run ID: `{p7_manifest.get('run_id')}`
   - Dataset Freeze Hash: `{p7_manifest.get('dataset_freeze_hash')}`
   - Fold Plan Hash: `{p7_manifest.get('fold_plan_hash')}`
   - Backbone Representation ID: `{P3_V2_REPR_ID}`

---

## 5. Phase 9 Conformal Handoff

The calibrated probabilities for `val_conf` (123 images) and the fitted `TemperatureScaler` have been packaged into `artifacts/phase8_v2/conformal_handoff.joblib`.  
Phase 9 will use these calibrated probabilities directly for conformal prediction set generation.
"""
    report_md.write_text(report_text, encoding="utf-8")
    generated_files["phase8_report"] = report_md

    print(f"\n{label_prefix}All Phase 8 V2 reports successfully generated in {out_dir}")
    return generated_files


# ======================================================================
# FRAMEWORK VALIDATION (--validate-framework / --validate-only)
# ======================================================================

def run_phase8_v2_framework_validation(config: PSDConfig) -> int:
    """Non-destructive synthetic plumbing check verifying all Phase 8 V2 operations."""
    print("======================================================================")
    print("PHASE 8 V2 FRAMEWORK VALIDATION: SYNTHETIC PLUMBING VERIFICATION")
    print("======================================================================")
    import tempfile
    import shutil

    tmp_dir = Path(tempfile.mkdtemp(prefix="papulonet_phase8_v2_test_"))
    try:
        classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
        rng = np.random.default_rng(42)

        # 1. Synthesize 246 probability vectors
        N = 246
        logits = rng.standard_normal((N, 4))
        # Add artificial temperature miscalibration
        logits = logits * 2.5
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        raw_probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        y_true_indices = rng.integers(0, 4, size=N)
        true_labels = [classes[i] for i in y_true_indices]
        psd_ids = [f"SYNTH_{i:04d}" for i in range(N)]

        # 2. Partition 123 / 123
        cal_idx = np.arange(123)
        conf_idx = np.arange(123, 246)

        print("[VALIDATION 1/6] Testing MulticlassTemperatureScaler fit on synthetic data...")
        scaler = fit_temperature_scaling(raw_probs[cal_idx], y_true_indices[cal_idx], bounds=TEMPERATURE_BOUNDS)
        assert scaler.temperature > 0, f"Expected T > 0, got {scaler.temperature}"
        assert scaler.nll_after_ <= scaler.nll_before_ + 1e-6, "Expected NLL not to degrade"
        print(f"  Passed: T={scaler.temperature:.4f}, NLL: {scaler.nll_before_:.4f} -> {scaler.nll_after_:.4f}")

        print("[VALIDATION 2/6] Testing predict_proba and strict argmax invariance...")
        cal_probs = scaler.predict_proba(raw_probs)
        raw_argmax = np.argmax(raw_probs, axis=1)
        cal_argmax = np.argmax(cal_probs, axis=1)
        assert np.array_equal(raw_argmax, cal_argmax), "Argmax invariance violated!"
        print("  Passed: 0 changed decisions across all 246 samples.")

        print("[VALIDATION 3/6] Testing sum to 1.0 constraint...")
        row_sums = np.sum(cal_probs, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Calibrated probabilities do not sum to 1!"
        print("  Passed: all rows sum to 1.0 exactly.")

        print("[VALIDATION 4/6] Testing evaluation metrics computation...")
        m = evaluate_calibration(cal_probs, true_labels, classes, "temperature_scaling", n_bins=10, raw_probs=raw_probs)
        assert m.n_changed_predictions == 0
        assert m.nll > 0
        assert 0 <= m.ece <= 1.0
        assert 0 <= m.brier <= 2.0
        print(f"  Passed: NLL={m.nll:.4f}, Brier={m.brier:.4f}, ECE={m.ece:.4f}, changed={m.n_changed_predictions}")

        print("[VALIDATION 5/6] Testing ConformalHandoff serialization & roundtrip...")
        test_handoff = ConformalHandoff(
            representation_id=P3_V2_REPR_ID,
            experiment_id=P3_V2_FOCAL_EXP,
            classifier_name=CLASSIFIER_NAME,
            feature_selection_method=FEATURE_SELECTOR_NAME,
            random_seed=42,
            class_order=classes,
            final_classifier=None,
            selected_feature_mask=np.ones(1316, dtype=bool),
            branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
            calibration_method="temperature_scaling",
            temperature_scaler=scaler,
            temperature=scaler.temperature,
            calibration_psd_ids=psd_ids[:123],
            calibration_true_labels=true_labels[:123],
            calibration_calibrated_probabilities=cal_probs[:123],
            val_conf_psd_ids=psd_ids[123:],
            val_conf_true_labels=true_labels[123:],
            val_conf_calibrated_probabilities=cal_probs[123:],
        )
        handoff_file = tmp_dir / "conformal_handoff.joblib"
        save_conformal_handoff(test_handoff, handoff_file)
        reloaded = joblib.load(handoff_file)
        assert reloaded.calibration_method == "temperature_scaling"
        assert reloaded.temperature == scaler.temperature
        assert np.array_equal(reloaded.val_conf_calibrated_probabilities, cal_probs[123:])
        print("  Passed: ConformalHandoff joblib artifact serialized and round-tripped cleanly.")

        print("[VALIDATION 6/6] Testing report generation pipeline in temporary directory...")
        test_cfg = dataclasses.replace(
            config,
            aef_crc_phase8_v2_reports_dir=tmp_dir / "reports",
            aef_crc_phase8_v2_artifacts_dir=tmp_dir / "artifacts",
        )
        fake_handoff = CalibrationHandoff(
            representation_id=P3_V2_REPR_ID,
            experiment_id=P3_V2_FOCAL_EXP,
            classifier_name=CLASSIFIER_NAME,
            feature_selection_method=FEATURE_SELECTOR_NAME,
            random_seed=42,
            source_experiment=P3_V2_FOCAL_EXP,
            feature_arm=A7_ARM_ID,
            class_order=classes,
            final_classifier=None,
            selected_feature_mask=np.ones(1316, dtype=bool),
            branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
            calibration_psd_ids=psd_ids,
            calibration_true_labels=true_labels,
            calibration_predicted_labels=true_labels,
            calibration_raw_probabilities=raw_probs,
            val_calib_psd_ids=psd_ids[:123],
            val_calib_true_labels=true_labels[:123],
            val_calib_raw_probabilities=raw_probs[:123],
            val_conf_psd_ids=psd_ids[123:],
            val_conf_true_labels=true_labels[123:],
            val_conf_raw_probabilities=raw_probs[123:],
        )
        _, fake_results, fake_conf_handoff = run_phase8_v2_calibration(test_cfg, fake_handoff, label_prefix="[VALIDATION] ")
        files = write_phase8_v2_reports(
            test_cfg, fake_handoff, fake_results, fake_conf_handoff,
            {"run_id": "TEST_P7", "dataset_freeze_hash": "abc", "fold_plan_hash": "def"},
            label_prefix="[VALIDATION] "
        )
        for name, p in files.items():
            assert p.exists() and p.stat().st_size > 0, f"Report {name} missing or empty at {p}"
        print("  Passed: All 7 reports and manifests generated successfully.")

        print("\n======================================================================")
        print("ALL 6 FRAMEWORK VALIDATION CHECKS PASSED PERFECTLY!")
        print("======================================================================")
        return 0

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ======================================================================
# MAIN RUNNER
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PapuloNet V2 Phase 8: Probability Calibration (Multiclass Temperature Scaling)"
    )
    parser.add_argument(
        "--validate-framework",
        "--validate-only",
        action="store_true",
        dest="validate_framework",
        help="Run non-destructive synthetic verification only (does not overwrite research results)",
    )
    args = parser.parse_args()

    config = configure_phase8_v2(get_config())

    if args.validate_framework:
        return run_phase8_v2_framework_validation(config)

    print("======================================================================")
    print("PAPULONET V2 PHASE 8: PROBABILITY CALIBRATION")
    print("CERTIFIED WINNER: A7-BDA (1316-D)")
    print("PRIMARY METHOD: MULTICLASS TEMPERATURE SCALING (Guo et al., 2017)")
    print("======================================================================")

    # Step 0: Gate check
    print("\n--- STEP 0: GATE CHECK & UPSTREAM VERIFICATION ---")
    gate_ok, gate_msg, p7_manifest = gate_check_phase8_v2(config)
    print(f"Gate check status: {'PASSED' if gate_ok else 'FAILED'}")
    print(f"Details: {gate_msg}")
    if not gate_ok:
        print("\n[PHASE 8 V2 BLOCKER] Upstream verification failed. Halting.")
        return 2

    # Load CalibrationHandoff
    print("\n--- STEP 1: LOAD CALIBRATION HANDOFF ---")
    handoff_path = config.aef_crc_phase7_v2_artifacts_dir / "calibration_handoff.joblib"
    handoff = load_calibration_handoff(handoff_path)
    validate_calibration_handoff_provenance(handoff, config, expected_run_id=p7_manifest.get("run_id"))
    validate_bda_mask_compatibility(handoff.selected_feature_mask, handoff.final_classifier, expected_dim=A7_EXPECTED_DIM)
    print(f"Loaded Phase 7 V2 CalibrationHandoff from {handoff_path}")
    print(f"  Representation ID: {handoff.representation_id}")
    print(f"  Arm ID:            {handoff.feature_arm}")
    print(f"  Selected features: {int(handoff.selected_feature_mask.sum())} / {A7_EXPECTED_DIM}")

    # Partition preflight
    part_ok, part_msg = preflight_phase8_v2_partitions(config, handoff)
    print(f"Partition preflight: {'PASSED' if part_ok else 'FAILED'}")
    print(f"  {part_msg}")
    if not part_ok:
        print("\n[PHASE 8 V2 BLOCKER] Partition preflight failed. Halting.")
        return 3

    # Tasks 2 & 3: Fit & Evaluate Temperature Scaling
    scaler, results, conformal_handoff = run_phase8_v2_calibration(config, handoff)

    # Task 4: Serialize Artifacts
    print("\n======================================================================")
    print("TASK 4: PERSIST PHASE 8 V2 ARTIFACTS")
    print("======================================================================")
    art_dir = config.aef_crc_phase8_v2_artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)

    conformal_path = art_dir / "conformal_handoff.joblib"
    save_conformal_handoff(conformal_handoff, conformal_path)
    print(f"Saved ConformalHandoff artifact: {conformal_path}")

    scaler_path = art_dir / "temperature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved TemperatureScaler artifact: {scaler_path}")

    # Task 5: Write Reports
    print("\n======================================================================")
    print("TASK 5: WRITE PHASE 8 V2 REPORTS & MANIFEST")
    print("======================================================================")
    report_files = write_phase8_v2_reports(config, handoff, results, conformal_handoff, p7_manifest)
    for name, p in report_files.items():
        print(f"  {name:35s} -> {p.name}")

    print("\n======================================================================")
    print("PHASE 8 V2 PROBABILITY CALIBRATION COMPLETE.")
    print(f"Optimal Temperature: T = {scaler.temperature:.4f}")
    print("Strict Argmax Invariance: 0 changed class decisions across all 246 samples.")
    print("Conformal Handoff ready for Phase 9.")
    print("======================================================================")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
