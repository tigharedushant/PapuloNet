"""
run_aef_crc_phase9_v2.py

PapuloNet V2 Phase 9: Conformal Prediction.
Consumes the Phase 8 V2 ConformalHandoff (artifacts/phase8_v2/conformal_handoff.joblib)
carrying the dedicated 123-image conformal calibration partition (val_conf)
with multiclass temperature-scaled probabilities.

Scientific & Methodological Foundations:
----------------------------------------
1. Objective:
   Construct finite-sample valid prediction sets C_hat(x) at nominal 90% coverage
   (alpha = 0.10) over the 4 target classes:
   ['Psoriasis', 'Lichen_Planus', 'Pityriasis_Rosea', 'Seborrheic_Dermatitis'].

2. Primary Method: Pooled Marginal Split-Conformal Prediction:
   - Conformal calibration set: val_conf (N_conf = 123 samples).
   - Calibrated probability vector: p = [p_1, ..., p_K] from Phase 8 Multiclass Temperature Scaling.
   - Nonconformity score:
       s_i = 1.0 - p_i(y_i)
     where p_i(y_i) is the temperature-scaled probability assigned to the true label y_i.
   - Exact finite-sample quantile order statistic (Vovk et al., Angelopoulos & Bates):
       k = ceil((n + 1) * (1 - alpha))
     For n = 123, alpha = 0.10:
       k = ceil(124 * 0.90) = ceil(111.6) = 112.
       q_hat = s_{(112)}
   - Prediction set:
       C_hat(x) = { c in Y : 1 - p_c <= q_hat } = { c in Y : p_c >= 1 - q_hat }

3. Secondary / Diagnostic Method: Class-Conditional Mondrian Conformal Prediction:
   - For each class c, compute nonconformity scores restricted to samples where true label is c.
   - Exact per-class quantile index:
       k_c = ceil((n_c + 1) * (1 - alpha))
       q_hat_c = s_{c, (min(k_c, n_c))}
   - Inclusion rule:
       include class c if p_c >= 1 - q_hat_c
   - Diagnostic small-sample reporting:
       Minority classes with n_c < 15 (e.g. Seborrheic Dermatitis with n_c = 9) exhibit coarse
       empirical quantiles and elevated finite-sample variance. Explicitly documented.

4. Data Discipline:
   - NO classifier retraining.
   - NO BDA mask alteration.
   - NO temperature refitting.
   - NO class weighting, SMOTE, oversampling, or threshold tuning.
   - Locked test set (243 images) remains STRICTLY UNTOUCHED & UNREAD.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from config.config import PSDConfig, get_config
from modules.calibration_handoff import (
    ConformalHandoff,
    FinalPipelineHandoff,
    load_conformal_handoff,
    save_final_pipeline_handoff,
    validate_conformal_handoff_provenance,
)
from modules.conformal import (
    ConformalMetrics,
    MarginalConformalFit,
    MondrianConformalFit,
    compute_conformal_quantile,
    compute_nonconformity_scores,
    evaluate_conformal_sets,
    fit_class_conditional_mondrian,
    fit_marginal_conformal,
    predict_conformal_sets,
    predict_conformal_sets_mondrian,
)
from modules.dataset_freeze import DatasetFreezer
from modules.experiment_config import representation_id
from modules.fold_loader import load_frozen_fold_plan

logger = logging.getLogger("papulonet.phase9_v2")

# ======================================================================
# CONSTANTS & PROTOCOL INVARIANTS
# ======================================================================

P3_V2_REPR_ID = "efficientnet_b0_43d581b96f8ec368"
P3_V2_FOCAL_EXP = "P3-V2-Focal"
A7_ARM_ID = "A7"
A7_EXPECTED_DIM = 1316
A7_SELECTED_FEATURES = 642
CLASSIFIER_NAME = "random_forest"
FEATURE_SELECTOR_NAME = "bda"
CALIBRATION_METHOD = "temperature_scaling"
NOMINAL_ALPHA = 0.10
NOMINAL_COVERAGE = 0.90


def configure_phase9_v2(config: PSDConfig) -> PSDConfig:
    """Binds authoritative Phase 9 V2 configuration and paths:
      - Phase 3 V2 Focal representation fields:
          * preprocessing_mode = 'standard'
          * training_time_augmentation = 'false'
          * loss_name = 'categorical_focal_loss'
          * focal_gamma = 2.0
          * use_class_weights = False
          * adam_clipnorm = 1.0
        Guarantees representation_id(config) == 'efficientnet_b0_43d581b96f8ec368'.
      - Downstream pipeline parameters:
          * classifier_name = 'random_forest'
          * feature_selection_method = 'bda'
          * bda_feature_count_penalty = 0.0
          * conformal_alpha = 0.10
      - Output directories:
          * aef_crc_phase9_reports_dir = reports/phase9_v2
          * aef_crc_phase9_artifacts_dir = artifacts/phase9_v2
          * aef_crc_phase9_logs_dir = logs/phase9_v2
    """
    return dataclasses.replace(
        config,
        aef_crc_phase9_reports_dir=config.aef_crc_phase9_v2_reports_dir,
        aef_crc_phase9_artifacts_dir=config.aef_crc_phase9_v2_artifacts_dir,
        conformal_alpha=NOMINAL_ALPHA,
        classifier_name=CLASSIFIER_NAME,
        feature_selection_method=FEATURE_SELECTOR_NAME,
        preprocessing_mode="standard",
        training_time_augmentation="false",
        loss_name="categorical_focal_loss",
        focal_gamma=2.0,
        use_class_weights=False,
        adam_clipnorm=1.0,
        bda_feature_count_penalty=0.0,
    )


# ======================================================================
# STEP 0: GATE CHECK & UPSTREAM VERIFICATION
# ======================================================================

def gate_check_phase9_v2(config: PSDConfig) -> Tuple[bool, str, Dict[str, Any]]:
    """Strictly verifies upstream Phase 8 V2 certification manifest and artifacts."""
    p8_manifest_path = config.aef_crc_phase8_v2_reports_dir / "phase8_manifest.json"
    if not p8_manifest_path.exists():
        return False, f"Missing Phase 8 V2 manifest at {p8_manifest_path}. Phase 8 V2 must be certified first.", {}

    try:
        manifest = json.loads(p8_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse Phase 8 V2 manifest JSON: {exc}", {}

    status = manifest.get("status")
    if status != "CALIBRATION_CERTIFIED":
        return False, f"Phase 8 V2 status is '{status}', expected 'CALIBRATION_CERTIFIED'", manifest

    pipe = manifest.get("pipeline", {})
    if pipe.get("arm_id") != A7_ARM_ID:
        return False, f"Arm mismatch in Phase 8 manifest: '{pipe.get('arm_id')}' != '{A7_ARM_ID}'", manifest

    if pipe.get("classifier") != CLASSIFIER_NAME:
        return False, f"Classifier mismatch in Phase 8 manifest: '{pipe.get('classifier')}' != '{CLASSIFIER_NAME}'", manifest

    if pipe.get("feature_selection_method") != FEATURE_SELECTOR_NAME:
        return False, f"Feature selector mismatch in Phase 8 manifest: '{pipe.get('feature_selection_method')}' != '{FEATURE_SELECTOR_NAME}'", manifest

    if pipe.get("representation_id") != P3_V2_REPR_ID:
        return False, f"Representation ID mismatch in Phase 8 manifest: '{pipe.get('representation_id')}' != '{P3_V2_REPR_ID}'", manifest

    cal = manifest.get("calibration", {})
    if cal.get("method") != CALIBRATION_METHOD:
        return False, f"Calibration method mismatch: '{cal.get('method')}' != '{CALIBRATION_METHOD}'", manifest

    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    if not handoff_path.exists():
        return False, f"Missing Phase 8 V2 ConformalHandoff artifact at {handoff_path}", manifest

    scaler_path = config.aef_crc_phase8_v2_artifacts_dir / "temperature_scaler.joblib"
    if not scaler_path.exists():
        return False, f"Missing Phase 8 V2 TemperatureScaler artifact at {scaler_path}", manifest

    # Hashes verification
    freeze_record = config.aef_crc_reports_dir / "dataset_freeze.json"
    if freeze_record.exists():
        expected_freeze = hashlib.sha256(freeze_record.read_bytes()).hexdigest()
        actual_freeze = manifest.get("provenance", {}).get("dataset_freeze_hash")
        if actual_freeze != expected_freeze:
            return False, f"Dataset freeze hash mismatch: manifest='{actual_freeze}' != current='{expected_freeze}'", manifest

    fold_record = config.reports_dir / "aef_crc" / "fold_plan.csv"
    if fold_record.exists():
        expected_fold = hashlib.sha256(fold_record.read_bytes()).hexdigest()
        actual_fold = manifest.get("provenance", {}).get("fold_plan_hash")
        if actual_fold != expected_fold:
            return False, f"Fold plan hash mismatch: manifest='{actual_fold}' != current='{expected_fold}'", manifest

    return True, "Phase 8 V2 certified calibration manifest and handoff verified successfully.", manifest


# ======================================================================
# STEP 1: PREFLIGHT DATA & PARTITION ISOLATION
# ======================================================================

def preflight_phase9_v2_data(
    config: PSDConfig,
    handoff: ConformalHandoff,
    p8_manifest: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Verifies that the conformal calibration partition is valid, strictly isolated,
    and adheres to the certified upstream provenance contract.
    """
    # 1. Dynamic Representation ID validation:
    actual_repr_id = representation_id(config)
    if actual_repr_id != handoff.representation_id:
        return False, (
            f"Representation ID mismatch: representation_id(config)='{actual_repr_id}' "
            f"!= handoff.representation_id='{handoff.representation_id}'. "
            "Phase 9 configuration must match Phase 8 handoff representation!"
        )
    if handoff.representation_id != P3_V2_REPR_ID:
        return False, (
            f"Handoff representation ID does not match certified Phase 3 V2 Focal representation: "
            f"'{handoff.representation_id}' != '{P3_V2_REPR_ID}'"
        )

    # 2. Upstream Run IDs and Arm verification:
    if p8_manifest is not None:
        p8_run_id = p8_manifest.get("run_id")
        p7_run_id = p8_manifest.get("provenance", {}).get("upstream_phase7_run_id")
        if not p8_run_id:
            return False, "Missing Phase 8 run_id in Phase 8 manifest"
        if not p7_run_id:
            return False, "Missing upstream Phase 7 run_id in Phase 8 manifest"
        if handoff.run_id != p7_run_id:
            return False, f"Handoff run_id '{handoff.run_id}' does not match upstream Phase 7 run_id '{p7_run_id}'"
        arm_id = p8_manifest.get("pipeline", {}).get("arm_id")
        if arm_id != A7_ARM_ID:
            return False, f"Arm ID mismatch in Phase 8 manifest: '{arm_id}' != '{A7_ARM_ID}'"

    # 3. Model & Feature Mask dimensions:
    if handoff.selected_feature_mask is None:
        return False, "Missing selected_feature_mask in ConformalHandoff"
    mask = handoff.selected_feature_mask
    if mask.shape != (A7_EXPECTED_DIM,):
        return False, f"BDA mask shape mismatch: expected ({A7_EXPECTED_DIM},), got {mask.shape}"
    if int(mask.sum()) != A7_SELECTED_FEATURES:
        return False, f"BDA mask selected feature count mismatch: expected {A7_SELECTED_FEATURES}, got {int(mask.sum())}"

    # 4. Temperature calibration validation:
    if handoff.temperature is None or handoff.temperature <= 0.0:
        return False, f"Invalid fitted temperature in handoff: {handoff.temperature}"
    if p8_manifest is not None:
        manifest_t = p8_manifest.get("calibration", {}).get("temperature")
        if manifest_t is not None and not np.isclose(handoff.temperature, manifest_t, rtol=1e-5):
            return False, f"Temperature mismatch between handoff ({handoff.temperature}) and manifest ({manifest_t})"

    # 5. Class order invariant:
    expected_classes = [
        "Psoriasis",
        "Lichen_Planus",
        "Pityriasis_Rosea",
        "Seborrheic_Dermatitis",
    ]
    if list(handoff.class_order) != expected_classes:
        return False, f"Class order mismatch: expected {expected_classes}, got {handoff.class_order}"

    # 6. Sample counts & probabilities checks
    n_conf = len(handoff.val_conf_psd_ids)
    if n_conf != 123:
        return False, f"Expected exactly 123 val_conf samples, got {n_conf}"

    if len(handoff.val_conf_true_labels) != 123:
        return False, f"Expected 123 val_conf_true_labels, got {len(handoff.val_conf_true_labels)}"

    probs = handoff.val_conf_calibrated_probabilities
    if probs is None or probs.shape != (123, 4):
        return False, f"Expected val_conf_calibrated_probabilities of shape (123, 4), got {getattr(probs, 'shape', None)}"

    if np.isnan(probs).any() or np.isinf(probs).any():
        return False, "NaN or Inf detected in val_conf_calibrated_probabilities!"

    if (probs < 0.0).any():
        return False, "Negative probability values detected in val_conf_calibrated_probabilities!"

    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        return False, f"Probability row sums deviate from 1.0: max error={np.max(np.abs(row_sums - 1.0)):.6e}"

    # Partition isolation: check against fold_plan.csv
    plan = load_frozen_fold_plan(config)
    dev_ids = {r.psd_id for f in plan.folds for r in f.train_records + f.val_records}
    test_ids = {r.psd_id for r in plan.holdout_test_records}
    quar_ids = {r.psd_id for r in plan.excluded_augmented_records}
    calib_fit_ids = set(handoff.calibration_psd_ids)
    conf_ids = set(handoff.val_conf_psd_ids)

    if not conf_ids.isdisjoint(dev_ids):
        return False, "Data leakage: val_conf overlaps with 1146 development cohort!"
    if not conf_ids.isdisjoint(calib_fit_ids):
        return False, "Data leakage: val_conf overlaps with 123 val_calib cohort!"
    if not conf_ids.isdisjoint(test_ids):
        return False, "Data leakage: val_conf overlaps with 243 locked test set!"
    if not conf_ids.isdisjoint(quar_ids):
        return False, "Data leakage: val_conf overlaps with 264 quarantined cohort!"

    return True, (
        f"Preflight verification passed: N_conf=123, shape=(123, 4), "
        f"representation='{actual_repr_id}', strictly disjoint from dev (1146), "
        f"val_calib (123), test (243), and quarantined (264)."
    )


# ======================================================================
# STEP 2 & 3: CONFORMAL CALIBRATION ENGINE
# ======================================================================

def run_phase9_v2_conformal_calibration(
    config: PSDConfig,
    handoff: ConformalHandoff,
    p8_manifest: Dict[str, Any],
    label_prefix: str = "",
) -> Tuple[MarginalConformalFit, MondrianConformalFit, Dict[str, Any], FinalPipelineHandoff]:
    """Executes Phase 9 V2 conformal calibration on val_conf (N=123) with T-scaled probabilities."""
    classes = handoff.class_order
    probs_conf = handoff.val_conf_calibrated_probabilities
    labels_conf = handoff.val_conf_true_labels
    psd_ids_conf = handoff.val_conf_psd_ids
    n_conf = len(labels_conf)
    alpha = config.conformal_alpha

    print(f"\n{label_prefix}======================================================================")
    print(f"{label_prefix}TASK 2: PRIMARY POOLED MARGINAL SPLIT-CONFORMAL CALIBRATION")
    print(f"{label_prefix}======================================================================")
    marginal_fit = fit_marginal_conformal(
        probs=probs_conf,
        true_labels=labels_conf,
        class_order=classes,
        alpha=alpha,
    )
    print(f"{label_prefix}Nominal Coverage (1 - alpha): {marginal_fit.nominal_coverage * 100:.1f}% (alpha = {alpha:.2f})")
    print(f"{label_prefix}Calibration Cohort:          val_conf (N = {marginal_fit.n_samples})")
    print(f"{label_prefix}Exact Quantile Formula:      k = ceil(({marginal_fit.n_samples} + 1) * {1.0 - alpha:.2f}) = {marginal_fit.formula_k}")
    print(f"{label_prefix}Marginal Quantile Threshold: q_hat = {marginal_fit.q_hat:.6f}")
    print(f"{label_prefix}Prediction Set Condition:    include class c if p_c >= {1.0 - marginal_fit.q_hat:.6f}")

    print(f"\n{label_prefix}======================================================================")
    print(f"{label_prefix}TASK 3: SECONDARY / DIAGNOSTIC CLASS-CONDITIONAL MONDRIAN CONFORMAL")
    print(f"{label_prefix}======================================================================")
    mondrian_fit = fit_class_conditional_mondrian(
        probs=probs_conf,
        true_labels=labels_conf,
        class_order=classes,
        alpha=alpha,
        min_recommended_n=15,
    )
    for cls in classes:
        n_c = mondrian_fit.per_class_n[cls]
        q_c = mondrian_fit.q_hat_by_class[cls]
        print(f"{label_prefix}  {cls:22s}: n = {n_c:2d}, q_hat_c = {q_c:.6f} (inclusion threshold: p_c >= {1.0 - q_c:.6f})")
        if cls in mondrian_fit.small_sample_warnings:
            print(f"{label_prefix}    [DIAGNOSTIC WARNING] {mondrian_fit.small_sample_warnings[cls]}")

    print(f"\n{label_prefix}======================================================================")
    print(f"{label_prefix}TASK 4: IN-SAMPLE EVALUATION ON VAL_CONF (N=123)")
    print(f"{label_prefix}======================================================================")
    marginal_sets = predict_conformal_sets(probs_conf, marginal_fit.q_hat, classes)
    eval_marginal = evaluate_conformal_sets(marginal_sets, labels_conf, classes, nominal_coverage=1.0 - alpha)

    mondrian_sets = predict_conformal_sets_mondrian(probs_conf, mondrian_fit.q_hat_by_class, classes)
    eval_mondrian = evaluate_conformal_sets(mondrian_sets, labels_conf, classes, nominal_coverage=1.0 - alpha)

    print(f"{label_prefix}--- Primary Marginal Split-Conformal Diagnostics ---")
    print(f"{label_prefix}  Empirical Marginal Coverage:  {eval_marginal.marginal_coverage * 100:.2f}% (nominal: {NOMINAL_COVERAGE * 100:.1f}%)")
    print(f"{label_prefix}  Mean Prediction Set Size:     {eval_marginal.mean_set_size:.3f}")
    print(f"{label_prefix}  Median Prediction Set Size:   {eval_marginal.median_set_size:.1f}")
    print(f"{label_prefix}  Singleton Rate (|C| = 1):     {eval_marginal.singleton_fraction * 100:.2f}%")
    print(f"{label_prefix}  Empty Set Rate (|C| = 0):     {eval_marginal.empty_set_fraction * 100:.2f}%")
    for cls in classes:
        cov_c = eval_marginal.per_class_coverage.get(cls, float("nan"))
        print(f"{label_prefix}    Coverage for {cls:22s}: {cov_c * 100:.2f}%")

    print(f"\n{label_prefix}--- Class-Conditional Mondrian Diagnostics ---")
    print(f"{label_prefix}  Empirical Coverage:           {eval_mondrian.marginal_coverage * 100:.2f}%")
    print(f"{label_prefix}  Mean Prediction Set Size:     {eval_mondrian.mean_set_size:.3f}")
    print(f"{label_prefix}  Singleton Rate (|C| = 1):     {eval_mondrian.singleton_fraction * 100:.2f}%")
    print(f"{label_prefix}  Empty Set Rate (|C| = 0):     {eval_mondrian.empty_set_fraction * 100:.2f}%")
    for cls in classes:
        cov_c = eval_mondrian.per_class_coverage.get(cls, float("nan"))
        print(f"{label_prefix}    Coverage for {cls:22s}: {cov_c * 100:.2f}%")

    # Assemble evaluation bundle
    eval_bundle = {
        "marginal_fit": marginal_fit,
        "mondrian_fit": mondrian_fit,
        "eval_marginal": eval_marginal,
        "eval_mondrian": eval_mondrian,
        "marginal_sets": marginal_sets,
        "mondrian_sets": mondrian_sets,
        "psd_ids_conf": psd_ids_conf,
        "labels_conf": labels_conf,
        "probs_conf": probs_conf,
    }

    # Mint Phase 9 Run ID
    now_utc = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    p9_run_id = f"AEFCRC_P9_V2_{now_utc}_{hashlib.sha256(str(marginal_fit.q_hat).encode()).hexdigest()[:8]}"

    final_handoff = FinalPipelineHandoff(
        representation_id=handoff.representation_id,
        experiment_id=handoff.experiment_id,
        classifier_name=handoff.classifier_name,
        feature_selection_method=handoff.feature_selection_method,
        random_seed=handoff.random_seed,
        class_order=classes,
        final_classifier=handoff.final_classifier,
        selected_feature_mask=handoff.selected_feature_mask,
        branch_dims=handoff.branch_dims,
        calibration_method=CALIBRATION_METHOD,
        marginal_q_hat=marginal_fit.q_hat,
        mondrian_q_hat=mondrian_fit.q_hat_by_class,
        run_id=p9_run_id,
        upstream_phase8_run_id=p8_manifest.get("run_id"),
        upstream_phase7_run_id=p8_manifest.get("provenance", {}).get("upstream_phase7_run_id"),
        dataset_freeze_hash=getattr(handoff, "dataset_freeze_hash", None),
        fold_plan_hash=getattr(handoff, "fold_plan_hash", None),
        platt_models=None,
        isotonic_models=None,
        temperature_scaler=handoff.temperature_scaler,
        temperature=handoff.temperature,
        alpha=alpha,
        n_conf_samples=n_conf,
        per_class_conf_counts=mondrian_fit.per_class_n,
        small_sample_warnings=mondrian_fit.small_sample_warnings,
        hog_reducer=getattr(handoff, "hog_reducer", None),
        feature_normalizers=getattr(handoff, "feature_normalizers", None),
        backbone_checkpoint_path=getattr(handoff, "backbone_checkpoint_path", None),
    )

    return marginal_fit, mondrian_fit, eval_bundle, final_handoff


# ======================================================================
# STEP 5 & 6: SERIALIZE ARTIFACTS & WRITE REPORTS
# ======================================================================

def write_phase9_v2_reports(
    config: PSDConfig,
    handoff: ConformalHandoff,
    eval_bundle: Dict[str, Any],
    final_handoff: FinalPipelineHandoff,
    p8_manifest: Dict[str, Any],
    label_prefix: str = "",
) -> Dict[str, Path]:
    """Generates all Phase 9 V2 certified reports, CSVs, JSON manifest, and Markdown report."""
    out_dir = config.aef_crc_phase9_v2_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: Dict[str, Path] = {}

    classes = handoff.class_order
    marginal_fit: MarginalConformalFit = eval_bundle["marginal_fit"]
    mondrian_fit: MondrianConformalFit = eval_bundle["mondrian_fit"]
    eval_marginal: ConformalMetrics = eval_bundle["eval_marginal"]
    eval_mondrian: ConformalMetrics = eval_bundle["eval_mondrian"]
    marginal_sets: List[List[str]] = eval_bundle["marginal_sets"]
    psd_ids: List[str] = eval_bundle["psd_ids_conf"]
    true_labels: List[str] = eval_bundle["labels_conf"]
    probs: np.ndarray = eval_bundle["probs_conf"]

    # 1. conformal_summary.csv
    summary_csv = out_dir / "conformal_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "alpha", "nominal_coverage", "n_samples", "q_hat_or_mean",
            "inclusion_threshold", "empirical_coverage", "mean_set_size",
            "median_set_size", "singleton_fraction", "empty_set_fraction",
        ])
        writer.writerow([
            "marginal_split_conformal",
            f"{marginal_fit.alpha:.4f}",
            f"{marginal_fit.nominal_coverage:.4f}",
            marginal_fit.n_samples,
            f"{marginal_fit.q_hat:.6f}",
            f"{1.0 - marginal_fit.q_hat:.6f}",
            f"{eval_marginal.marginal_coverage:.6f}",
            f"{eval_marginal.mean_set_size:.4f}",
            f"{eval_marginal.median_set_size:.1f}",
            f"{eval_marginal.singleton_fraction:.6f}",
            f"{eval_marginal.empty_set_fraction:.6f}",
        ])
        writer.writerow([
            "mondrian_class_conditional",
            f"{mondrian_fit.alpha:.4f}",
            f"{mondrian_fit.nominal_coverage:.4f}",
            marginal_fit.n_samples,
            f"{np.mean(list(mondrian_fit.q_hat_by_class.values())):.6f}",
            "per_class",
            f"{eval_mondrian.marginal_coverage:.6f}",
            f"{eval_mondrian.mean_set_size:.4f}",
            f"{eval_mondrian.median_set_size:.1f}",
            f"{eval_mondrian.singleton_fraction:.6f}",
            f"{eval_mondrian.empty_set_fraction:.6f}",
        ])
    generated_files["conformal_summary"] = summary_csv

    # 2. mondrian_class_quantiles.csv
    mondrian_csv = out_dir / "mondrian_class_quantiles.csv"
    with mondrian_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name", "n_calib_support", "q_hat_c", "inclusion_threshold",
            "empirical_coverage", "formula_k", "small_sample_warning",
        ])
        for cls in classes:
            n_c = mondrian_fit.per_class_n[cls]
            q_c = mondrian_fit.q_hat_by_class[cls]
            thresh_c = 1.0 - q_c
            cov_c = eval_mondrian.per_class_coverage.get(cls, float("nan"))
            k_c = math.ceil((n_c + 1) * (1.0 - mondrian_fit.alpha))
            warn = mondrian_fit.small_sample_warnings.get(cls, "None")
            writer.writerow([
                cls, n_c, f"{q_c:.6f}", f"{thresh_c:.6f}",
                f"{cov_c:.6f}" if not np.isnan(cov_c) else "N/A",
                k_c, warn,
            ])
    generated_files["mondrian_class_quantiles"] = mondrian_csv

    # 3. set_size_distribution.csv
    size_csv = out_dir / "set_size_distribution.csv"
    with size_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set_size", "marginal_count", "marginal_percentage", "mondrian_count", "mondrian_percentage"])
        n_tot = len(true_labels)
        for s in range(len(classes) + 1):
            m_cnt = eval_marginal.set_size_distribution.get(s, 0)
            mon_cnt = eval_mondrian.set_size_distribution.get(s, 0)
            writer.writerow([
                s, m_cnt, f"{(m_cnt / n_tot) * 100.0:.2f}%",
                mon_cnt, f"{(mon_cnt / n_tot) * 100.0:.2f}%",
            ])
    generated_files["set_size_distribution"] = size_csv

    # 4. per_class_conformal_metrics.csv
    per_class_csv = out_dir / "per_class_conformal_metrics.csv"
    with per_class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name", "support", "marginal_coverage", "mondrian_coverage",
            "marginal_inclusion_rate", "mondrian_inclusion_rate",
        ])
        for c_idx, cls in enumerate(classes):
            supp = sum(1 for y in true_labels if y == cls)
            m_cov = eval_marginal.per_class_coverage.get(cls, float("nan"))
            mon_cov = eval_mondrian.per_class_coverage.get(cls, float("nan"))
            m_inc = float(np.mean([cls in s for s in marginal_sets]))
            mon_inc = float(np.mean([cls in s for s in eval_bundle["mondrian_sets"]]))
            writer.writerow([
                cls, supp,
                f"{m_cov:.6f}" if not np.isnan(m_cov) else "N/A",
                f"{mon_cov:.6f}" if not np.isnan(mon_cov) else "N/A",
                f"{m_inc:.6f}", f"{mon_inc:.6f}",
            ])
    generated_files["per_class_conformal_metrics"] = per_class_csv

    # 5. conformal_prediction_sets.csv (Sample-level predictions for val_conf)
    preds_csv = out_dir / "conformal_prediction_sets.csv"
    with preds_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "psd_id", "true_class", "marginal_prediction_set", "marginal_set_size",
            "marginal_covered", "mondrian_prediction_set", "mondrian_set_size", "mondrian_covered",
        ])
        for i in range(len(psd_ids)):
            p_id = psd_ids[i]
            y_true = true_labels[i]
            m_set = marginal_sets[i]
            mon_set = eval_bundle["mondrian_sets"][i]
            writer.writerow([
                p_id, y_true,
                ";".join(m_set), len(m_set), int(y_true in m_set),
                ";".join(mon_set), len(mon_set), int(y_true in mon_set),
            ])
    generated_files["conformal_prediction_sets"] = preds_csv

    # 6. phase9_manifest.json
    manifest_path = out_dir / "phase9_manifest.json"
    manifest_data = {
        "phase": "phase9_v2",
        "run_id": final_handoff.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CONFORMAL_CALIBRATION_CERTIFIED",
        "pipeline": {
            "arm_id": A7_ARM_ID,
            "classifier": CLASSIFIER_NAME,
            "feature_selection_method": FEATURE_SELECTOR_NAME,
            "representation_id": P3_V2_REPR_ID,
            "source_experiment": P3_V2_FOCAL_EXP,
        },
        "conformal_calibration": {
            "alpha": marginal_fit.alpha,
            "nominal_coverage": marginal_fit.nominal_coverage,
            "cohort": "val_conf",
            "n_samples": marginal_fit.n_samples,
            "primary_method": "pooled_marginal_split_conformal",
            "marginal_q_hat": marginal_fit.q_hat,
            "marginal_formula_k": marginal_fit.formula_k,
            "marginal_inclusion_threshold": 1.0 - marginal_fit.q_hat,
            "mondrian_q_hat_by_class": mondrian_fit.q_hat_by_class,
            "per_class_n": mondrian_fit.per_class_n,
            "small_sample_warnings": mondrian_fit.small_sample_warnings,
        },
        "val_conf_evaluation": {
            "marginal": {
                "empirical_coverage": eval_marginal.marginal_coverage,
                "mean_set_size": eval_marginal.mean_set_size,
                "median_set_size": eval_marginal.median_set_size,
                "singleton_fraction": eval_marginal.singleton_fraction,
                "empty_set_fraction": eval_marginal.empty_set_fraction,
                "per_class_coverage": eval_marginal.per_class_coverage,
            },
            "mondrian": {
                "empirical_coverage": eval_mondrian.marginal_coverage,
                "mean_set_size": eval_mondrian.mean_set_size,
                "median_set_size": eval_mondrian.median_set_size,
                "singleton_fraction": eval_mondrian.singleton_fraction,
                "empty_set_fraction": eval_mondrian.empty_set_fraction,
                "per_class_coverage": eval_mondrian.per_class_coverage,
            },
        },
        "provenance": {
            "upstream_phase8_run_id": p8_manifest.get("run_id"),
            "upstream_phase7_run_id": p8_manifest.get("provenance", {}).get("upstream_phase7_run_id"),
            "dataset_freeze_hash": p8_manifest.get("provenance", {}).get("dataset_freeze_hash"),
            "fold_plan_hash": p8_manifest.get("provenance", {}).get("fold_plan_hash"),
            "locked_test_untouched": True,
            "locked_test_records_count": 243,
        },
        "artifacts": {
            "final_pipeline_handoff_joblib": str(config.aef_crc_phase9_v2_artifacts_dir / "final_pipeline_handoff.joblib"),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    generated_files["phase9_manifest"] = manifest_path

    # 7. phase9_report.md
    report_md = out_dir / "phase9_report.md"
    md_lines = [
        "# PapuloNet V2 Phase 9: Conformal Prediction Report",
        f"**Pipeline**: A7-BDA (1316-D: EfficientNet-B0 + GLCM + LBP + LAB, Random Forest 300 Trees)  ",
        f"**Temperature Scaling**: Fitted $T = {handoff.temperature:.6f}$  ",
        f"**Primary Conformal Method**: Pooled Marginal Split-Conformal Prediction  ",
        f"**Run ID**: `{final_handoff.run_id}`  ",
        f"**Date**: {datetime.now(timezone.utc).isoformat()}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Conformal Calibration Outcome",
        "",
        "Phase 9 executes **Split-Conformal Prediction** at nominal 90% coverage ($\\alpha = 0.10$) over the temperature-calibrated probabilities produced in Phase 8.",
        "",
        "### Key Conformal Parameters:",
        f"- **Calibration Cohort**: `val_conf` ($N = {marginal_fit.n_samples}$ samples), strictly disjoint from `val_calib` and the locked test set.",
        f"- **Nominal Coverage**: {NOMINAL_COVERAGE * 100:.1f}% ($\\alpha = {marginal_fit.alpha:.2f}$)",
        f"- **Exact Finite-Sample Quantile Index**: $k = \\lceil ({marginal_fit.n_samples} + 1) \\times {1.0 - marginal_fit.alpha:.2f} \\rceil = {marginal_fit.formula_k}$",
        f"- **Prediction Set Rule**: Include class $c$ if $s(x, c) = (1.0 - P_{{cal}}(c \\mid x)) \\le {marginal_fit.q_hat:.6f}$ (nominally $P_{{cal}}(c \\mid x) \\ge {1.0 - marginal_fit.q_hat:.6f}$)",
        "",
        "---",
        "",
        "## 2. Conformal Summary Table",
        "",
        "> [!NOTE]",
        "> **Observed Calibration-Cohort Diagnostic Coverage**:",
        "> The empirical coverage reported below is an observed diagnostic evaluated on the calibration cohort `val_conf` ($N=123$) from which $\\hat{q}$ was derived. It must **not** be interpreted as an independent test result or domain generalization claim. Formal conformal coverage guarantees apply to future exchangeable observations under the stated exchangeability assumption.",
        "",
        "| Method | Nominal Cov | Observed Calib Diagnostic Cov | Mean Set Size | Median Set Size | Singleton Rate | Empty Set Rate |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Marginal Split-Conformal** | {NOMINAL_COVERAGE * 100:.1f}% | **{eval_marginal.marginal_coverage * 100:.2f}%** | **{eval_marginal.mean_set_size:.3f}** | **{eval_marginal.median_set_size:.1f}** | **{eval_marginal.singleton_fraction * 100:.2f}%** | **{eval_marginal.empty_set_fraction * 100:.2f}%** |",
        f"| **Class-Conditional Mondrian** | {NOMINAL_COVERAGE * 100:.1f}% | **{eval_mondrian.marginal_coverage * 100:.2f}%** | **{eval_mondrian.mean_set_size:.3f}** | **{eval_mondrian.median_set_size:.1f}** | **{eval_mondrian.singleton_fraction * 100:.2f}%** | **{eval_mondrian.empty_set_fraction * 100:.2f}%** |",
        "",
        "---",
        "",
        "## 3. Class-Conditional Mondrian Calibration Quantiles",
        "",
        "| Target Class | Support ($n_c$) | $\\hat{q}_c$ | Inclusion Threshold ($1 - \\hat{q}_c$) | Empirical Coverage | Diagnostic Warning |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for cls in classes:
        n_c = mondrian_fit.per_class_n[cls]
        q_c = mondrian_fit.q_hat_by_class[cls]
        thresh_c = 1.0 - q_c
        cov_c = eval_mondrian.per_class_coverage.get(cls, float("nan"))
        cov_str = f"{cov_c * 100:.2f}%" if not np.isnan(cov_c) else "N/A"
        warn = mondrian_fit.small_sample_warnings.get(cls, "None")
        md_lines.append(f"| `{cls}` | {n_c} | `{q_c:.6f}` | `{thresh_c:.6f}` | {cov_str} | {warn} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Set Size Distribution",
        "",
        "| Set Size ($|C|$) | Marginal Count | Marginal % | Mondrian Count | Mondrian % |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ])
    for s in range(len(classes) + 1):
        m_cnt = eval_marginal.set_size_distribution.get(s, 0)
        mon_cnt = eval_mondrian.set_size_distribution.get(s, 0)
        md_lines.append(
            f"| {s} | {m_cnt} | {(m_cnt / len(true_labels)) * 100.0:.2f}% | {mon_cnt} | {(mon_cnt / len(true_labels)) * 100.0:.2f}% |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 5. Scientific Interpretation & Critical Caveats",
        "",
        "1. **Exchangeability Assumption**:",
        "   - Finite-sample marginal coverage guarantees strictly depend on the assumption that calibration samples and future test samples are exchangeable.",
        "   - Conformal coverage does **not** establish clinical diagnostic validity or guarantee domain generalization.",
        "2. **Singleton Sets ($|C| = 1$)**:",
        "   - A singleton prediction set does **not** equal absolute certainty; it indicates that the nonconformity of all alternative classes exceeded the 90% empirical threshold.",
        "3. **Empty Sets ($|C| = 0$)**:",
        "   - An empty set occurs when no class achieves probability $\\ge 1 - \\hat{q}$. It signals unusual or ambiguous feature patterns but must **not** be treated as a formally calibrated out-of-distribution (OOD) detector.",
        "4. **Minority Class Variance**:",
        "   - For `Seborrheic_Dermatitis` ($n_c = 9$), the class-conditional quantile index is $k = \\lceil 10 \\times 0.90 \\rceil = 9$. The quantile is determined by the maximum score, resulting in wider prediction sets.",
        "",
        "---",
        "",
        "## 6. Phase 10 / Final Pipeline Handoff",
        "",
        f"The complete frozen bundle has been saved to `{config.aef_crc_phase9_v2_artifacts_dir / 'final_pipeline_handoff.joblib'}` for test-set evaluation.",
        "",
    ])

    report_md.write_text("\n".join(md_lines), encoding="utf-8")
    generated_files["phase9_report"] = report_md

    print(f"\n{label_prefix}All Phase 9 V2 reports and manifests successfully written to {out_dir}")
    return generated_files


# ======================================================================
# SYNTHETIC FRAMEWORK VALIDATION (--validate-framework)
# ======================================================================

def run_phase9_v2_framework_validation(config: PSDConfig) -> int:
    """Performs non-destructive synthetic verification of the entire Phase 9 V2 pipeline."""
    print("======================================================================")
    print("PHASE 9 V2 FRAMEWORK VALIDATION: SYNTHETIC PLUMBING VERIFICATION")
    print("======================================================================")

    tmp_dir = Path(tempfile.mkdtemp(prefix="papulonet_phase9_v2_test_"))
    try:
        classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
        n_synthetic = 123
        rng = np.random.default_rng(42)

        # Synthetic Dirichlet probabilities (sum to 1)
        raw_probs = rng.dirichlet(alpha=[2.0, 1.5, 1.0, 0.8], size=n_synthetic)
        synth_labels = rng.choice(classes, size=n_synthetic, p=[0.55, 0.23, 0.14, 0.08]).tolist()
        synth_ids = [f"SYNTH_CONF_{i:03d}" for i in range(n_synthetic)]

        print("\n[VALIDATION 1/5] Testing nonconformity scores computation...")
        label_to_idx = {c: i for i, c in enumerate(classes)}
        indices = np.array([label_to_idx[y] for y in synth_labels])
        scores = compute_nonconformity_scores(raw_probs, indices)
        assert len(scores) == n_synthetic
        assert (scores >= 0.0).all() and (scores <= 1.0).all()
        print("  Passed: Nonconformity scores computed correctly in [0, 1].")

        print("\n[VALIDATION 2/5] Testing exact finite-sample quantile order statistic...")
        # For n=123, alpha=0.10: k = ceil(124 * 0.90) = 112
        k_expected = int(math.ceil((n_synthetic + 1) * 0.90))
        assert k_expected == 112, f"Expected k=112 for n=123, alpha=0.10, got {k_expected}"
        q_hat = compute_conformal_quantile(scores, alpha=0.10)
        sorted_s = np.sort(scores)
        assert q_hat == sorted_s[111], f"Quantile mismatch: {q_hat} != {sorted_s[111]}"
        print(f"  Passed: Exact finite-sample quantile k={k_expected} verified (q_hat={q_hat:.4f}).")

        print("\n[VALIDATION 3/5] Testing prediction set construction and invariants...")
        sets = predict_conformal_sets(raw_probs, q_hat, classes)
        assert len(sets) == n_synthetic
        # Verify inclusion condition: p_c >= 1 - q_hat
        threshold = 1.0 - q_hat
        for i, s in enumerate(sets):
            for c_idx, c_name in enumerate(classes):
                if raw_probs[i, c_idx] >= threshold:
                    assert c_name in s
                else:
                    assert c_name not in s
        print("  Passed: Prediction set rule p_c >= 1 - q_hat strictly satisfied.")

        print("\n[VALIDATION 4/5] Testing class-conditional Mondrian conformal calibration...")
        mon_fit = fit_class_conditional_mondrian(raw_probs, synth_labels, classes, alpha=0.10, min_recommended_n=15)
        for cls in classes:
            n_c = mon_fit.per_class_n[cls]
            q_c = mon_fit.q_hat_by_class[cls]
            assert 0.0 <= q_c <= 1.0
            if n_c < 15:
                assert cls in mon_fit.small_sample_warnings
        print("  Passed: Mondrian per-class quantiles and small-sample warnings verified.")

        print("\n[VALIDATION 5/5] Testing serialization and report generation pipeline...")
        test_cfg = dataclasses.replace(
            config,
            aef_crc_phase9_v2_reports_dir=tmp_dir / "reports",
            aef_crc_phase9_v2_artifacts_dir=tmp_dir / "artifacts",
        )
        fake_handoff = ConformalHandoff(
            representation_id=P3_V2_REPR_ID,
            experiment_id=P3_V2_FOCAL_EXP,
            classifier_name=CLASSIFIER_NAME,
            feature_selection_method=FEATURE_SELECTOR_NAME,
            random_seed=42,
            class_order=classes,
            final_classifier=None,
            selected_feature_mask=(lambda: (m := np.zeros(1316, dtype=bool), m.__setitem__(slice(0, 642), True), m)[2])(),
            branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
            calibration_method="temperature_scaling",
            temperature=0.3257,
            calibration_psd_ids=synth_ids,
            calibration_true_labels=synth_labels,
            calibration_calibrated_probabilities=raw_probs,
            val_conf_psd_ids=synth_ids,
            val_conf_true_labels=synth_labels,
            val_conf_calibrated_probabilities=raw_probs,
        )
        p8_fake_manifest = {
            "run_id": "TEST_P8_RUN",
            "pipeline": {
                "arm_id": A7_ARM_ID,
                "representation_id": P3_V2_REPR_ID,
                "classifier": CLASSIFIER_NAME,
                "feature_selection_method": FEATURE_SELECTOR_NAME,
            },
            "calibration": {
                "method": CALIBRATION_METHOD,
                "temperature": 0.3257,
            },
            "provenance": {
                "upstream_phase7_run_id": "TEST_P7_RUN",
                "dataset_freeze_hash": "test_hash_1",
                "fold_plan_hash": "test_hash_2",
            },
        }

        m_fit, mon_fit, eval_b, final_h = run_phase9_v2_conformal_calibration(
            test_cfg, fake_handoff, p8_fake_manifest, label_prefix="[VALIDATION] ",
        )
        test_cfg.aef_crc_phase9_v2_artifacts_dir.mkdir(parents=True, exist_ok=True)
        final_art_path = test_cfg.aef_crc_phase9_v2_artifacts_dir / "final_pipeline_handoff.joblib"
        save_final_pipeline_handoff(final_h, final_art_path)
        assert final_art_path.exists() and final_art_path.stat().st_size > 0

        reloaded_h = joblib.load(final_art_path)
        assert reloaded_h.marginal_q_hat == m_fit.q_hat
        assert reloaded_h.temperature == 0.3257

        report_files = write_phase9_v2_reports(
            test_cfg, fake_handoff, eval_b, final_h, p8_fake_manifest, label_prefix="[VALIDATION] ",
        )
        for name, p in report_files.items():
            assert p.exists() and p.stat().st_size > 0, f"Report {name} missing or empty at {p}"

        print("  Passed: All reports, CSVs, manifest, and FinalPipelineHandoff artifact generated successfully.")
        print("\n======================================================================")
        print("ALL 5 FRAMEWORK VALIDATION CHECKS PASSED PERFECTLY!")
        print("======================================================================")
        return 0

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ======================================================================
# MAIN RUNNER
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PapuloNet V2 Phase 9: Conformal Prediction"
    )
    parser.add_argument(
        "--validate-framework",
        "--validate-only",
        action="store_true",
        dest="validate_framework",
        help="Run non-destructive synthetic verification only (does not execute on real data)",
    )
    args = parser.parse_args()

    config = configure_phase9_v2(get_config())

    if args.validate_framework:
        return run_phase9_v2_framework_validation(config)

    print("======================================================================")
    print("PAPULONET V2 PHASE 9: CONFORMAL PREDICTION")
    print("CERTIFIED PIPELINE: A7-BDA (1316-D, RF 300 Trees)")
    print("CALIBRATION INPUT: Temperature-Scaled Probabilities (Phase 8 V2)")
    print("NOMINAL COVERAGE: 90% (alpha = 0.10)")
    print("======================================================================")

    # Step 0: Gate check
    print("\n--- STEP 0: GATE CHECK & UPSTREAM VERIFICATION ---")
    gate_ok, gate_msg, p8_manifest = gate_check_phase9_v2(config)
    print(f"Gate check status: {'PASSED' if gate_ok else 'FAILED'}")
    print(f"Details: {gate_msg}")
    if not gate_ok:
        print("\n[PHASE 9 V2 BLOCKER] Upstream verification failed. Halting.")
        return 2

    # Step 1: Load ConformalHandoff
    print("\n--- STEP 1: LOAD CONFORMAL HANDOFF & PREFLIGHT ---")
    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    handoff = load_conformal_handoff(handoff_path)

    # Validate provenance against certified Phase 7 / Phase 8 run IDs
    validate_conformal_handoff_provenance(handoff, config, expected_run_id=handoff.run_id)

    print(f"Loaded Phase 8 V2 ConformalHandoff from {handoff_path}")
    print(f"  Run ID in handoff: {handoff.run_id}")
    print(f"  Representation ID: {handoff.representation_id}")
    print(f"  Classifier:        {handoff.classifier_name}")
    print(f"  Selected features: {int(handoff.selected_feature_mask.sum())} / {A7_EXPECTED_DIM}")
    print(f"  Fitted temperature: {handoff.temperature:.6f}")

    # Data preflight & partition isolation
    preflight_ok, preflight_msg = preflight_phase9_v2_data(config, handoff, p8_manifest)
    print(f"Preflight status: {'PASSED' if preflight_ok else 'FAILED'}")
    print(f"  {preflight_msg}")
    if not preflight_ok:
        print("\n[PHASE 9 V2 BLOCKER] Preflight data verification failed. Halting.")
        return 3

    # Steps 2-4: Conformal Calibration Engine
    marginal_fit, mondrian_fit, eval_bundle, final_handoff = run_phase9_v2_conformal_calibration(
        config, handoff, p8_manifest,
    )

    # Step 5: Serialize FinalPipelineHandoff
    print("\n======================================================================")
    print("STEP 5: PERSIST FINAL PIPELINE HANDOFF ARTIFACT")
    print("======================================================================")
    art_dir = config.aef_crc_phase9_v2_artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    final_path = art_dir / "final_pipeline_handoff.joblib"
    save_final_pipeline_handoff(final_handoff, final_path)
    print(f"Saved FinalPipelineHandoff artifact: {final_path}")

    # Step 6: Write Reports & Manifest
    print("\n======================================================================")
    print("STEP 6: WRITE PHASE 9 V2 REPORTS & MANIFEST")
    print("======================================================================")
    report_files = write_phase9_v2_reports(
        config, handoff, eval_bundle, final_handoff, p8_manifest,
    )
    for name, p in report_files.items():
        print(f"  [{name}] {p}")

    print("\n" + "=" * 70)
    print("PAPULONET V2 PHASE 9 EXECUTION COMPLETE.")
    print(f"MARGINAL QUANTILE THRESHOLD (q_hat): {marginal_fit.q_hat:.6f}")
    print(f"EMPIRICAL COVERAGE ON VAL_CONF:      {eval_bundle['eval_marginal'].marginal_coverage * 100:.2f}%")
    print(f"MEAN SET SIZE ON VAL_CONF:           {eval_bundle['eval_marginal'].mean_set_size:.3f}")
    print(f"FINAL DEPLOYED ARTIFACT:             {final_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
