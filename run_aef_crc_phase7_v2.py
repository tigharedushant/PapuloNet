"""
run_aef_crc_phase7_v2.py

PapuloNet V2 Phase 7: Final Model Selection, Evaluation & Calibration Handoff Pipeline.

Target Pipeline:
  - Fused Representation: A7 (1316-D: EfficientNet deep 1280 + GLCM 12 + LBP 18 + LAB 6)
  - Upstream Deep Model: P3-V2-Focal (representation_id: efficientnet_b0_43d581b96f8ec368)
  - Feature Refinement: Binary Dragonfly Algorithm (BDA) on A7
  - Primary Classifier: Random Forest (300 trees, gini, max_features='sqrt', bootstrap=True)

Execution Structure:
  1. Step 0: Gate check strictly verifying upstream Phase 6 V2-A7 manifest,
     dataset freeze hash, fold plan hash, and representation IDs.
  2. Preflight checks for deep feature caches (all 5 outer folds + fold -1 final_train and calibration)
     and handcrafted caches (GLCM 12, LBP 18, LAB 6 for all 1146 dev + 246 calib images).
  3. Task 1: Full 5-Fold Stratified Cross-Validation Benchmark:
     - Fold-local BDA with inner-train balanced sample weights (inner-val Macro-F1 fitness, lambda=0.0).
     - Fold-local Random Forest trained with fold-local balanced class weights.
     - Unweighted validation evaluation across disjoint validation slices.
     - Per-fold mask serialization to artifacts/phase7_v2/masks/.
     - Aggregated metrics (Macro-F1, BalAcc, MCC, Acc, Weighted F1, per-class P/R/F1/support,
       pooled 4x4 confusion matrix, dominance diagnostics).
  4. Task 2: Final Retrain on Unified 1146 Development Cohort:
     - Reconstructs all 1146 non-augmented images (zero quarantined 264 images, zero outer val 246, zero test 243).
     - Executes production BDA on unified 1146 cohort to produce authoritative production mask.
     - Serializes production mask to artifacts/phase7_v2/production_bda_mask.joblib.
     - Trains final Random Forest on selected features with cohort balanced class weights.
     - Evaluates on 246 held-out validation images (FoldPlan.holdout_val_records) to obtain calibration predictions.
  5. Task 3: Calibration Handoff Generation & Persistence:
     - Stratified 50/50 partition of the 246 calibration records into 123 calibration / 123 conformal sets.
     - Populates CalibrationHandoff with all fitted classifiers, masks, probabilities, and provenance.
     - Saves to artifacts/phase7_v2/calibration_handoff.joblib.
  6. Task 4: Predefined Gating Evaluation:
     - Hardcoded criteria: Macro-F1 >= 0.7000, minority classes viable, Psoriasis pred ratio <= 1.15,
       held-out val Macro-F1 >= 0.6500. Predefined fallback: Full A7.
  7. Task 5: Report Generation:
     - Comprehensive CSVs, JSON manifest, and Markdown report to reports/phase7_v2/.
  8. Framework Validation (--validate-framework / --validate-only):
     - 10 non-destructive verification steps on synthetic data.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from config.config import get_config, PSDConfig
from modules.aef_input_validator import compute_train_fold_class_weights
from modules.calibration_handoff import (
    CalibrationHandoff,
    save_calibration_handoff,
    validate_bda_mask_compatibility,
    validate_calibration_handoff_provenance,
)
from modules.dataset_freeze import DatasetFreezer
from modules.evaluation import (
    FoldMetrics,
    AggregatedMetrics,
    compute_fold_metrics,
    aggregate_fold_metrics,
)
from modules.experiment_config import representation_id, experiment_id
from modules.feature_selection import (
    run_selector,
    save_production_bda_mask,
    compute_feature_family_breakdown,
    SelectionResult,
)
from modules.fold_loader import load_frozen_fold_plan, validate_fold_plan
from modules.fusion import (
    build_fusion_fold,
    build_fusion_final,
    compute_sample_weights,
    FusionFoldData,
)
from modules.handcrafted_features import HandcraftedFeatureExtractor

logger = logging.getLogger("papulonet.phase7_v2")

# ======================================================================
# CONSTANTS & PROTOCOL INVARIANTS
# ======================================================================

A7_ARM_ID = "A7"
A7_ARM_NAME = "EfficientNet+GLCM+LBP+LAB"
A7_EXPECTED_DIM = 1316
A7_BRANCHES = ("deep", "glcm", "lbp", "color_lab")
A7_BRANCH_DIMS = {
    "deep": 1280,
    "glcm": 12,
    "lbp": 18,
    "color_lab": 6,
}
P3_V2_FOCAL_EXP = "P3-V2-Focal"
P3_V2_REPR_ID = "efficientnet_b0_43d581b96f8ec368"
CLASSIFIER_NAME = "random_forest"
FEATURE_SELECTOR_NAME = "bda"

# Predefined Phase 7 Acceptance Criteria
GATE_CRITERIA = {
    "min_macro_f1": 0.7000,
    "min_per_class_f1": {
        "Psoriasis": 0.750,
        "Lichen_Planus": 0.550,
        "Pityriasis_Rosea": 0.600,
        "Seborrheic_Dermatitis": 0.600,
    },
    "max_psoriasis_pred_ratio": 1.150,
    "min_calibration_macro_f1": 0.6500,
    "fallback_candidate": "full_a7",
}


def configure_phase7_v2(config: PSDConfig) -> PSDConfig:
    """Binds authoritative Phase 3 V2 Focal winner configuration and Phase 7 V2 paths:
      - Phase 3 V2 Focal representation fields:
          * preprocessing_mode = 'standard'
          * training_time_augmentation = 'false'
          * loss_name = 'categorical_focal_loss'
          * focal_gamma = 2.0
          * use_class_weights = False
          * adam_clipnorm = 1.0
        Guarantees representation_id(config) == 'efficientnet_b0_43d581b96f8ec368'.
      - Output directories:
          * aef_crc_phase7_reports_dir = reports/phase7_v2
          * aef_crc_phase7_artifacts_dir = artifacts/phase7_v2
          * aef_crc_phase7_logs_dir = logs/phase7_v2
      - Downstream pipeline parameters:
          * classifier_name = 'random_forest'
          * feature_selection_method = 'bda'
          * bda_feature_count_penalty = 0.0
    """
    return dataclasses.replace(
        config,
        aef_crc_artifacts_dir=config.aef_crc_phase3_v2_artifacts_dir,
        aef_crc_phase3_reports_dir=config.aef_crc_phase3_v2_reports_dir,
        aef_crc_phase5_reports_dir=config.aef_crc_phase5_v2_reports_dir,
        aef_crc_phase6_reports_dir=config.aef_crc_phase6_v2_a7_reports_dir,
        aef_crc_phase6_artifacts_dir=config.aef_crc_phase6_v2_a7_artifacts_dir,
        aef_crc_phase7_reports_dir=config.aef_crc_phase7_v2_reports_dir,
        aef_crc_phase7_artifacts_dir=config.aef_crc_phase7_v2_artifacts_dir,
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

def gate_check_phase7_v2(config: PSDConfig) -> Tuple[bool, str, Dict[str, Any]]:
    """Strictly verifies upstream Phase 6 V2-A7 exploratory results and frozen foundations:
    1. reports/phase6_v2_a7/phase6_a7_manifest.json exists and is valid.
    2. Arm is A7 (1316-D, branches deep+glcm+lbp+color_lab).
    3. Representation ID matches efficientnet_b0_43d581b96f8ec368.
    4. Dataset freeze hash and fold plan hash match authoritative records.
    """
    p6_manifest_path = config.aef_crc_phase6_v2_a7_reports_dir / "phase6_a7_manifest.json"
    if not p6_manifest_path.exists():
        msg = (
            f"Phase 6 V2-A7 manifest missing at {p6_manifest_path}. "
            "Phase 7 V2 requires certified Phase 6 V2-A7 results before model selection. "
            "Execute run_aef_crc_phase6_v2_a7.py first."
        )
        return False, msg, {}

    try:
        manifest_data = json.loads(p6_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse Phase 6 V2-A7 manifest: {exc}", {}

    # Verify A7 definition
    a7_def = manifest_data.get("a7_definition", {})
    if a7_def.get("arm_id") != A7_ARM_ID:
        return False, f"Arm mismatch: expected '{A7_ARM_ID}', got '{a7_def.get('arm_id')}'", {}
    if a7_def.get("expected_dimension") != A7_EXPECTED_DIM:
        return False, f"Dimension mismatch: expected {A7_EXPECTED_DIM}, got {a7_def.get('expected_dimension')}", {}
    if tuple(a7_def.get("branches", [])) != A7_BRANCHES:
        return False, f"Branches mismatch: expected {A7_BRANCHES}, got {a7_def.get('branches')}", {}

    # Verify representation ID
    repr_id = manifest_data.get("representation_id")
    if repr_id != P3_V2_REPR_ID:
        return False, f"Representation ID mismatch: expected '{P3_V2_REPR_ID}', got '{repr_id}'", {}

    # Verify dataset freeze
    freeze_verify = DatasetFreezer(config).verify()
    if not freeze_verify.matches:
        return False, f"Dataset freeze verification failed: {len(freeze_verify.mismatches)} mismatch(es)", {}

    freeze_p = config.aef_crc_reports_dir / "dataset_freeze.json"
    current_freeze_hash = hashlib.sha256(freeze_p.read_bytes()).hexdigest() if freeze_p.exists() else None
    manifest_freeze_hash = manifest_data.get("dataset_freeze_hash")
    if manifest_freeze_hash != current_freeze_hash:
        return False, f"Freeze hash mismatch: manifest='{manifest_freeze_hash}', current='{current_freeze_hash}'", {}

    # Verify fold plan
    plan_p = config.reports_dir / "aef_crc" / "fold_plan.csv"
    current_plan_hash = hashlib.sha256(plan_p.read_bytes()).hexdigest() if plan_p.exists() else None
    manifest_plan_hash = manifest_data.get("fold_plan_hash")
    if manifest_plan_hash != current_plan_hash:
        return False, f"Fold plan hash mismatch: manifest='{manifest_plan_hash}', current='{current_plan_hash}'", {}

    detail = (
        f"Verified upstream Phase 6 V2-A7 candidate: [A7-BDA] {A7_ARM_NAME} "
        f"({A7_EXPECTED_DIM}-D, branches={A7_BRANCHES}) | Upstream run_id: {manifest_data.get('run_id')} | "
        f"Dataset freeze hash: {current_freeze_hash[:16]}... | Fold plan hash: {current_plan_hash[:16]}..."
    )
    return True, detail, manifest_data


# ======================================================================
# PREFLIGHT CHECKS
# ======================================================================

def preflight_deep_feature_artifacts(
    config: PSDConfig, experiment_name: str, folds: List[Any],
) -> Tuple[bool, str]:
    """Preflight checks all required deep-feature artifacts under artifacts/phase3_v2/:
    1. Current config representation_id strictly matches P3_V2_REPR_ID.
    2. All 10 manifests for folds 0..4 (train and val) pass validate_deep_feature_cache.
    3. The 2 manifests for fold -1 (final_train and calibration) exist and match.
    4. Exact feature dimensionality (1280) and representation ID.
    """
    from modules.backbones import validate_deep_feature_cache

    p3_artifacts = getattr(config, "aef_crc_phase3_v2_artifacts_dir", config.aef_crc_artifacts_dir)
    deep_dir = p3_artifacts / "deep_features" / experiment_name

    if not deep_dir.exists():
        return False, f"Deep feature cache directory does not exist: {deep_dir}"

    current_repr_id = representation_id(config)
    if current_repr_id != P3_V2_REPR_ID:
        return (
            False,
            f"Current config representation_id '{current_repr_id}' does not match "
            f"authoritative Phase 3 V2 Focal ID '{P3_V2_REPR_ID}'. "
            "Configure Phase 7 V2 using configure_phase7_v2(config) before running preflight."
        )

    checked_manifests = 0
    # 1. Folds 0..4: invoke authoritative validate_deep_feature_cache
    for fold in folds:
        f_idx = fold.fold_index
        for split_name, records in [("train", fold.train_records), ("val", fold.val_records)]:
            expected_ids = [r.psd_id for r in records]
            try:
                validate_deep_feature_cache(
                    config, experiment_name, f_idx, split_name,
                    expected_psd_ids=expected_ids, backbone_name=config.backbone,
                )
            except Exception as exc:
                return False, f"Deep feature cache validation failed for fold {f_idx} {split_name}: {exc}"
            checked_manifests += 1

    # 2. Fold -1 (final_train & calibration)
    fold_neg1_dir = deep_dir / "fold_-1"
    for split_name, exp_count in [("final_train", 1146), ("calibration", 246)]:
        split_dir = fold_neg1_dir / split_name
        m_path = split_dir / "manifest.json"
        if not m_path.exists():
            return False, f"Missing deep feature manifest for fold -1 {split_name}: {m_path}"
        try:
            m_data = json.loads(m_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"Failed to read manifest {m_path}: {exc}"
        if m_data.get("feature_dim") != 1280:
            return False, f"Manifest {m_path} dimension mismatch: {m_data.get('feature_dim')} != 1280"
        if m_data.get("representation_id") != current_repr_id:
            return False, f"Manifest {m_path} representation mismatch: {m_data.get('representation_id')} != {current_repr_id}"
        if m_data.get("n_features") != exp_count:
            return False, f"Manifest {m_path} record count mismatch: {m_data.get('n_features')} != {exp_count}"
        checked_manifests += 1

    return True, f"Verified all {checked_manifests} deep-feature manifests (folds 0..4 + fold -1) under '{deep_dir}' with representation_id '{current_repr_id}'"


def preflight_handcrafted_features(config: PSDConfig, plan: Any) -> Tuple[bool, str]:
    """Preflight checks that GLCM (12), LBP (18), and LAB (6) caches exist for:
    - All 1,146 development cohort images.
    - All 246 outer calibration images.
    """
    mode = getattr(config, "preprocessing_mode", "standard")
    if mode != "standard":
        mode = "standard"  # V2 uses standard handcrafted features
    base_dir = config.aef_crc_phase4_artifacts_dir / mode

    all_records = []
    for f in plan.folds:
        all_records.extend(f.train_records)
        all_records.extend(f.val_records)
    all_records.extend(plan.holdout_val_records)

    unique_records = {r.psd_id: r for r in all_records}
    exp_total = 1146 + 246
    if len(unique_records) != exp_total:
        return False, f"Expected {exp_total} unique development + calibration records, found {len(unique_records)}"

    branches_to_check = [("glcm", 12), ("lbp", 18), ("color_lab", 6)]
    for branch, exp_dim in branches_to_check:
        b_dir = base_dir / branch
        if not b_dir.exists():
            return False, f"Missing handcrafted cache directory: {b_dir}"
        for psd_id in unique_records:
            f_path = b_dir / f"{psd_id}.npy"
            if not f_path.exists():
                return False, f"Missing {branch} cache file for {psd_id}: {f_path}"

    return True, f"Verified GLCM (12-D), LBP (18-D), and LAB (6-D) caches for all {exp_total} records under '{base_dir}'"


def preflight_partition_isolation(plan: Any) -> Tuple[bool, str]:
    """Verifies that development (1146), calibration (246), locked test (243),
    and quarantined (264) are completely disjoint.
    """
    dev_ids = set()
    for fold in plan.folds:
        tr_ids = {r.psd_id for r in fold.train_records}
        va_ids = {r.psd_id for r in fold.val_records}
        if tr_ids & va_ids:
            return False, f"Fold {fold.fold_index}: leakage between train and val!"
        dev_ids |= (tr_ids | va_ids)

    calib_ids = {r.psd_id for r in plan.holdout_val_records}
    test_ids = {r.psd_id for r in plan.holdout_test_records}
    quar_ids = {r.psd_id for r in plan.excluded_augmented_records}

    if len(dev_ids) != 1146:
        return False, f"Development cohort size mismatch: expected 1146, got {len(dev_ids)}"
    if len(calib_ids) != 246:
        return False, f"Calibration partition size mismatch: expected 246, got {len(calib_ids)}"
    if len(test_ids) != 243:
        return False, f"Locked test partition size mismatch: expected 243, got {len(test_ids)}"
    if len(quar_ids) != 264:
        return False, f"Quarantined augmented size mismatch: expected 264, got {len(quar_ids)}"

    if dev_ids & calib_ids:
        return False, "Leakage between development cohort and outer calibration!"
    if dev_ids & test_ids:
        return False, "Leakage between development cohort and locked test set!"
    if calib_ids & test_ids:
        return False, "Leakage between outer calibration and locked test set!"
    if quar_ids & (dev_ids | calib_ids | test_ids):
        return False, "Quarantined images contaminated active experiment partitions!"

    return True, "Verified partition isolation: dev=1146, calib=246, test=243, quarantined=264 (all disjoint)."


# ======================================================================
# DOMINANCE & DIAGNOSTIC UTILITIES
# ======================================================================

def compute_dominance_diagnostics(
    predictions_map: Dict[str, Tuple[str, str]], classes: Sequence[str],
) -> Dict[str, Any]:
    """Computes prediction-distribution and dominance diagnostics across all evaluation samples."""
    total = len(predictions_map)
    true_counts = {c: 0 for c in classes}
    pred_counts = {c: 0 for c in classes}

    for true_lbl, pred_lbl in predictions_map.values():
        true_counts[true_lbl] = true_counts.get(true_lbl, 0) + 1
        pred_counts[pred_lbl] = pred_counts.get(pred_lbl, 0) + 1

    diag: Dict[str, Any] = {"total_samples": total}
    for c in classes:
        t_c = true_counts[c]
        p_c = pred_counts[c]
        t_pct = round(t_c / total * 100, 2) if total > 0 else 0.0
        p_pct = round(p_c / total * 100, 2) if total > 0 else 0.0
        ratio = round(p_c / max(1, t_c), 4)
        diag[f"{c}_true_count"] = t_c
        diag[f"{c}_true_pct"] = t_pct
        diag[f"{c}_pred_count"] = p_c
        diag[f"{c}_pred_pct"] = p_pct
        diag[f"{c}_pred_ratio"] = ratio

    return diag


# ======================================================================
# TASK 1: FULL 5-FOLD CV BENCHMARK OF A7-BDA
# ======================================================================

def run_phase7_v2_cv(
    config: PSDConfig,
    plan: Any,
    source_experiment: str,
    label_prefix: str = "",
) -> Dict[str, Any]:
    """Task 1: Executes full 5-Fold Stratified Cross-Validation of the A7-BDA pipeline:
      - Fused vector: A7 (1316-D).
      - Feature selection: BDA on inner-train with inner-train balanced sample weights.
      - Outer classifier: RandomForestClassifier(300 trees, gini, max_features='sqrt', bootstrap=True).
      - Sample weights on outer train: fold-local balanced class weights.
      - Validation: unweighted.
      - Persists fold-local masks to artifacts/phase7_v2/masks/.
    """
    classes = config.target_classes
    extractor = HandcraftedFeatureExtractor(config)
    masks_dir = config.aef_crc_phase7_v2_artifacts_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics_list: List[FoldMetrics] = []
    pooled_predictions: Dict[str, Tuple[str, str]] = {}
    selected_masks: List[np.ndarray] = []
    fold_selected_counts: List[int] = []
    family_breakdowns: List[Dict[str, int]] = []

    print(f"{label_prefix}Starting Phase 7 V2 5-Fold Stratified Cross-Validation on A7-BDA ({A7_EXPECTED_DIM}-D)...")
    print(f"{label_prefix}Pipeline: A7 (1316-D) -> BDA (inner-train weighted, lambda=0.0) -> RandomForest(300 trees, balanced)")

    for fold in plan.folds:
        fold_idx = fold.fold_index
        fold_seed = config.random_seed + fold_idx

        # 1. Build fused A7 fold data
        data = build_fusion_fold(config, fold, A7_BRANCHES, source_experiment, extractor)
        assert data.X_train.shape[1] == A7_EXPECTED_DIM, f"Expected {A7_EXPECTED_DIM} features, got {data.X_train.shape[1]}"

        # 2. Execute BDA on inner-train (nested validation fold remains untouched)
        t_sel_0 = time.perf_counter()
        sel_bda = run_selector("bda", data, config, fold_seed)
        sel_time = time.perf_counter() - t_sel_0
        mask = sel_bda.selected_mask
        assert mask.shape == (A7_EXPECTED_DIM,)
        assert mask.dtype == bool

        selected_masks.append(mask)
        fold_selected_counts.append(sel_bda.selected_count)
        fb = compute_feature_family_breakdown(mask, A7_BRANCH_DIMS)
        family_breakdowns.append(fb)

        # 3. Persist fold-local mask artifact
        mask_joblib_path = masks_dir / f"fold_{fold_idx:02d}_bda_mask.joblib"
        mask_npy_path = masks_dir / f"fold_{fold_idx:02d}_bda_mask.npy"
        np.save(mask_npy_path, mask)
        joblib.dump({
            "fold_index": fold_idx,
            "mask": mask,
            "selected_count": sel_bda.selected_count,
            "total_dim": A7_EXPECTED_DIM,
            "branch_retained": fb,
            "fold_seed": fold_seed,
            "fit_time_sec": sel_time,
        }, mask_joblib_path)

        # 4. Train outer Random Forest with fold-local balanced class weights
        sw_train = compute_sample_weights(data.y_train, fold.class_weights)
        clf = RandomForestClassifier(
            n_estimators=300,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=config.random_seed,
            n_jobs=-1,
        )
        t_fit_0 = time.perf_counter()
        clf.fit(data.X_train[:, mask], data.y_train, sample_weight=sw_train)
        fit_time = time.perf_counter() - t_fit_0

        # 5. Evaluate on unweighted outer validation fold
        y_pred = list(clf.predict(data.X_val[:, mask]))
        metrics = compute_fold_metrics(data.y_val, y_pred, classes, fold_idx)
        fold_metrics_list.append(metrics)

        for pid, true_l, pred_l in zip(data.psd_ids_val, data.y_val, y_pred):
            pooled_predictions[pid] = (true_l, pred_l)

        print(
            f"{label_prefix}  [Fold {fold_idx}] selected={sel_bda.selected_count}/{A7_EXPECTED_DIM} "
            f"({fb}) | Macro-F1={metrics.macro_f1:.4f} | BalAcc={metrics.balanced_accuracy:.4f} | "
            f"MCC={metrics.mcc:.4f} | BDA_time={sel_time:.1f}s | RF_time={fit_time:.2f}s"
        )

    # Aggregate across 5 folds
    agg = aggregate_fold_metrics(fold_metrics_list)
    dim_mean = float(np.mean(fold_selected_counts))
    dim_std = float(np.std(fold_selected_counts))
    dominance = compute_dominance_diagnostics(pooled_predictions, classes)

    print(f"\n{label_prefix}--- Final 5-Fold Cross-Validation Aggregate Metrics (A7-BDA) ---")
    print(f"{label_prefix}Dimensions:        {dim_mean:.1f} +/- {dim_std:.1f} (out of {A7_EXPECTED_DIM})")
    print(f"{label_prefix}Macro-F1:          {agg.macro_f1_mean:.4f} +/- {agg.macro_f1_std:.4f}")
    print(f"{label_prefix}Balanced Accuracy: {agg.balanced_accuracy_mean:.4f} +/- {agg.balanced_accuracy_std:.4f}")
    print(f"{label_prefix}MCC:               {agg.mcc_mean:.4f} +/- {agg.mcc_std:.4f}")
    print(f"{label_prefix}Accuracy:          {agg.accuracy_mean:.4f} +/- {agg.accuracy_std:.4f}")
    print(f"{label_prefix}Weighted F1:       {agg.weighted_f1_mean:.4f} +/- {agg.weighted_f1_std:.4f}")
    print(f"{label_prefix}Psoriasis Pred Ratio: {dominance['Psoriasis_pred_ratio']:.4f} "
          f"({dominance['Psoriasis_pred_pct']}% pred vs {dominance['Psoriasis_true_pct']}% true)")

    return {
        "fold_metrics": fold_metrics_list,
        "aggregates": agg,
        "pooled_predictions": pooled_predictions,
        "selected_masks": selected_masks,
        "dim_stats": {
            "mean": dim_mean,
            "std": dim_std,
            "per_fold": fold_selected_counts,
        },
        "family_breakdowns": family_breakdowns,
        "dominance": dominance,
    }


# ======================================================================
# TASK 2: FINAL RETRAIN ON UNIFIED 1146 COHORT
# ======================================================================

def run_phase7_v2_final_retrain(
    config: PSDConfig,
    plan: Any,
    source_experiment: str,
    label_prefix: str = "",
) -> Dict[str, Any]:
    """Task 2: Reunified final model retrain on the complete 1,146 non-augmented outer-train images:
      - Reunified outer-train images = 1,146 (all 5 folds' train+val).
      - Zero quarantined 264 images.
      - Zero outer validation 246 images in training.
      - Zero locked test 243 images.
      - Production BDA on unified 1,146 images -> single production mask.
      - Final Random Forest fit with cohort balanced class weights.
      - Evaluates on the 246 held-out validation images (FoldPlan.holdout_val_records).
    """
    classes = config.target_classes
    extractor = HandcraftedFeatureExtractor(config)

    # Reconstruct the 1,146 development records
    final_train_records = plan.folds[0].train_records + plan.folds[0].val_records
    train_ids = [r.psd_id for r in final_train_records]
    assert len(set(train_ids)) == len(train_ids), "Duplicate PSD ID in reunified training set!"
    assert len(final_train_records) == 1146, f"Expected 1146 development records, got {len(final_train_records)}"

    print(f"{label_prefix}Executing final retrain on {len(final_train_records)} unified non-augmented outer-train images...")
    print(f"{label_prefix}Held-out calibration partition: {len(plan.holdout_val_records)} images (splitter.py's val/).")
    print(f"{label_prefix}Locked test partition: {len(plan.holdout_test_records)} images (strictly untouched).")

    # Build unified fusion data (final_train=1146, calib=246)
    data = build_fusion_final(config, final_train_records, plan.holdout_val_records, A7_BRANCHES, source_experiment, extractor)
    assert data.X_train.shape == (1146, A7_EXPECTED_DIM), f"Expected shape (1146, {A7_EXPECTED_DIM}), got {data.X_train.shape}"
    assert data.X_val.shape == (246, A7_EXPECTED_DIM), f"Expected shape (246, {A7_EXPECTED_DIM}), got {data.X_val.shape}"

    # Production BDA feature selection on unified 1146 images
    print(f"{label_prefix}Running authoritative production BDA on unified 1146 cohort (seed={config.random_seed})...")
    t_bda_0 = time.perf_counter()
    prod_sel = run_selector("bda", data, config, config.random_seed)
    prod_bda_time = time.perf_counter() - t_bda_0
    prod_mask = prod_sel.selected_mask

    fb_prod = compute_feature_family_breakdown(prod_mask, A7_BRANCH_DIMS)
    print(
        f"{label_prefix}Production BDA complete in {prod_bda_time:.1f}s: selected {prod_sel.selected_count}/{A7_EXPECTED_DIM} "
        f"features ({fb_prod})."
    )

    # Save production BDA mask artifact
    prod_mask_path = save_production_bda_mask(
        prod_mask,
        config,
        metadata={
            "selected_count": prod_sel.selected_count,
            "branch_retained": fb_prod,
            "fit_time_sec": prod_bda_time,
            "source_experiment": source_experiment,
            "cohort_size": 1146,
            "arm_id": A7_ARM_ID,
        },
        artifacts_dir=config.aef_crc_phase7_v2_artifacts_dir,
    )
    print(f"{label_prefix}Saved production BDA mask artifact: {prod_mask_path}")

    # Compute cohort balanced class weights on 1146 images
    cohort_class_weights = compute_train_fold_class_weights(
        [r.mapped_class for r in final_train_records], classes,
    )
    sw_cohort = compute_sample_weights(data.y_train, cohort_class_weights)

    # Fit final Random Forest on selected features
    final_clf = RandomForestClassifier(
        n_estimators=300,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=config.random_seed,
        n_jobs=-1,
    )
    t_rf_0 = time.perf_counter()
    final_clf.fit(data.X_train[:, prod_mask], data.y_train, sample_weight=sw_cohort)
    final_rf_time = time.perf_counter() - t_rf_0

    # Evaluate on the 246 held-out calibration records
    y_calib_pred = list(final_clf.predict(data.X_val[:, prod_mask]))
    raw_probs = final_clf.predict_proba(data.X_val[:, prod_mask])

    # Ensure probability columns match classes exactly
    prob_class_order = list(final_clf.classes_)
    calib_probabilities = np.zeros((raw_probs.shape[0], len(classes)), dtype=raw_probs.dtype)
    for j, cls in enumerate(classes):
        if cls in prob_class_order:
            calib_probabilities[:, j] = raw_probs[:, prob_class_order.index(cls)]

    calib_metrics = compute_fold_metrics(data.y_val, y_calib_pred, classes, fold_index=-1)
    print(f"\n{label_prefix}--- Final Model Calibration-Set Performance (Holdout Val, N=246) ---")
    print(f"{label_prefix}Fit Time:          {final_rf_time:.2f}s on {len(final_train_records)} images")
    print(f"{label_prefix}Macro-F1:          {calib_metrics.macro_f1:.4f}")
    print(f"{label_prefix}Balanced Accuracy: {calib_metrics.balanced_accuracy:.4f}")
    print(f"{label_prefix}MCC:               {calib_metrics.mcc:.4f}")
    print(f"{label_prefix}Accuracy:          {calib_metrics.accuracy:.4f}")
    print(f"{label_prefix}Weighted F1:       {calib_metrics.weighted_f1:.4f}")

    return {
        "final_classifier": final_clf,
        "selected_mask": prod_mask,
        "selected_count": prod_sel.selected_count,
        "branch_retained": fb_prod,
        "cohort_class_weights": cohort_class_weights,
        "calibration_records": plan.holdout_val_records,
        "calibration_psd_ids": data.psd_ids_val,
        "calibration_true_labels": data.y_val,
        "calibration_pred_labels": y_calib_pred,
        "calibration_probabilities": calib_probabilities,
        "calibration_metrics": calib_metrics,
        "final_train_records": final_train_records,
        "data": data,
        "fit_times": {
            "bda_time_sec": prod_bda_time,
            "rf_time_sec": final_rf_time,
        },
    }


# ======================================================================
# TASK 3: CALIBRATION HANDOFF GENERATION
# ======================================================================

def build_phase7_v2_calibration_handoff(
    config: PSDConfig,
    plan: Any,
    retrain_results: Dict[str, Any],
    p6_manifest: Dict[str, Any],
    source_experiment: str,
    phase7_run_id: Optional[str] = None,
) -> CalibrationHandoff:
    """Task 3: Constructs the authoritative CalibrationHandoff artifact:
      - Validates BDA mask compatibility.
      - Partitions the 246 calibration records into a stratified 50/50 split (123 / 123)
        for Phase 8 calibration and Phase 9 conformal prediction.
      - Populates exact provenance hashes and fitted classifiers.
      - Persists to artifacts/phase7_v2/calibration_handoff.joblib.
    """
    classes = config.target_classes
    final_clf = retrain_results["final_classifier"]
    mask = retrain_results["selected_mask"]
    calib_records = retrain_results["calibration_records"]
    calib_psd_ids = retrain_results["calibration_psd_ids"]
    calib_true = retrain_results["calibration_true_labels"]
    calib_pred = retrain_results["calibration_pred_labels"]
    calib_probs = retrain_results["calibration_probabilities"]

    # Validate mask compatibility with classifier
    validate_bda_mask_compatibility(mask, final_clf, expected_dim=A7_EXPECTED_DIM)

    # Stratified 50/50 partition of the 246 calibration records (123 for calib, 123 for conformal)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=config.random_seed)
    y_calib_arr = np.array(calib_true)
    calib_sub_idx, conf_sub_idx = next(sss.split(np.zeros(len(calib_true)), y_calib_arr))

    assert len(calib_sub_idx) == 123, f"Expected 123 calibration split samples, got {len(calib_sub_idx)}"
    assert len(conf_sub_idx) == 123, f"Expected 123 conformal split samples, got {len(conf_sub_idx)}"

    val_calib_psd_ids = [calib_psd_ids[i] for i in calib_sub_idx]
    val_calib_true = [calib_true[i] for i in calib_sub_idx]
    val_calib_probs = calib_probs[calib_sub_idx]

    val_conf_psd_ids = [calib_psd_ids[i] for i in conf_sub_idx]
    val_conf_true = [calib_true[i] for i in conf_sub_idx]
    val_conf_probs = calib_probs[conf_sub_idx]

    # Provenance hashes
    freeze_hash = p6_manifest.get("dataset_freeze_hash") or (
        hashlib.sha256((config.aef_crc_reports_dir / "dataset_freeze.json").read_bytes()).hexdigest()
        if (config.aef_crc_reports_dir / "dataset_freeze.json").exists() else None
    )
    fold_hash = p6_manifest.get("fold_plan_hash") or (
        hashlib.sha256((config.reports_dir / "aef_crc" / "fold_plan.csv").read_bytes()).hexdigest()
        if (config.reports_dir / "aef_crc" / "fold_plan.csv").exists() else None
    )

    backbone_ckpt = config.aef_crc_artifacts_dir / source_experiment / "fold_01" / "best_model.keras"
    if not backbone_ckpt.exists():
        backbone_ckpt = config.aef_crc_artifacts_dir / source_experiment / "fold_00" / "best_model.keras"

    if phase7_run_id is None:
        phase7_run_id = f"AEFCRC_P7_V2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(config.aef_crc_phase7_v2_reports_dir).encode()).hexdigest()[:8]}"

    handoff = CalibrationHandoff(
        representation_id=P3_V2_REPR_ID,
        experiment_id=source_experiment,
        classifier_name=CLASSIFIER_NAME,
        feature_selection_method=FEATURE_SELECTOR_NAME,
        random_seed=config.random_seed,
        source_experiment=source_experiment,
        feature_arm=A7_ARM_ID,
        class_order=list(classes),
        final_classifier=final_clf,
        selected_feature_mask=mask,
        branch_dims=A7_BRANCH_DIMS,
        calibration_psd_ids=calib_psd_ids,
        calibration_true_labels=calib_true,
        calibration_predicted_labels=calib_pred,
        calibration_raw_probabilities=calib_probs,
        run_id=phase7_run_id,
        upstream_phase6_run_id=p6_manifest.get("run_id"),
        dataset_freeze_hash=freeze_hash,
        fold_plan_hash=fold_hash,
        val_calib_psd_ids=val_calib_psd_ids,
        val_calib_true_labels=val_calib_true,
        val_calib_raw_probabilities=val_calib_probs,
        val_conf_psd_ids=val_conf_psd_ids,
        val_conf_true_labels=val_conf_true,
        val_conf_raw_probabilities=val_conf_probs,
        hog_reducer=None,
        feature_normalizers=retrain_results["data"].feature_normalizers,
        backbone_checkpoint_path=str(backbone_ckpt) if backbone_ckpt.exists() else None,
    )

    validate_calibration_handoff_provenance(handoff, config, expected_run_id=phase7_run_id)
    out_path = config.aef_crc_phase7_v2_artifacts_dir / "calibration_handoff.joblib"
    save_calibration_handoff(handoff, out_path)
    print(f"\nCalibration handoff artifact successfully persisted to {out_path}")

    return handoff


# ======================================================================
# TASK 4: PREDEFINED GATING EVALUATION
# ======================================================================

def evaluate_phase7_v2_gate(
    agg: AggregatedMetrics,
    calib_metrics: FoldMetrics,
    dominance: Dict[str, Any],
) -> Dict[str, Any]:
    """Task 4: Evaluates the predefined Phase 7 criteria for A7-BDA:
      1. Macro-F1 >= 0.7000
      2. Minority classes viable (SebDerm >= 0.600, Lichen >= 0.550, Pityriasis >= 0.600)
      3. Psoriasis prediction ratio <= 1.150
      4. Calibration-set Macro-F1 >= 0.6500
    """
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}

    # Check 1: Macro-F1
    c1 = agg.macro_f1_mean >= GATE_CRITERIA["min_macro_f1"]
    checks["macro_f1_generalization"] = c1
    details["macro_f1"] = {
        "observed": round(agg.macro_f1_mean, 4),
        "threshold": GATE_CRITERIA["min_macro_f1"],
        "passed": c1,
    }

    # Check 2: Minority classes
    min_pass = True
    min_details = {}
    for cls, min_th in GATE_CRITERIA["min_per_class_f1"].items():
        obs = agg.per_class_f1_mean.get(cls, 0.0)
        p = obs >= min_th
        min_details[cls] = {"observed": round(obs, 4), "threshold": min_th, "passed": p}
        if not p:
            min_pass = False
    checks["minority_class_viability"] = min_pass
    details["minority_classes"] = min_details

    # Check 3: Psoriasis dominance
    pso_ratio = dominance.get("Psoriasis_pred_ratio", 999.0)
    c3 = pso_ratio <= GATE_CRITERIA["max_psoriasis_pred_ratio"]
    checks["psoriasis_dominance_mitigated"] = c3
    details["psoriasis_dominance"] = {
        "observed_ratio": pso_ratio,
        "threshold": GATE_CRITERIA["max_psoriasis_pred_ratio"],
        "passed": c3,
    }

    # Check 4: Calibration generalization
    c4 = calib_metrics.macro_f1 >= GATE_CRITERIA["min_calibration_macro_f1"]
    checks["calibration_generalization"] = c4
    details["calibration_set"] = {
        "observed_macro_f1": round(calib_metrics.macro_f1, 4),
        "threshold": GATE_CRITERIA["min_calibration_macro_f1"],
        "passed": c4,
    }

    all_passed = all(checks.values())
    decision = "CERTIFIED_WINNER" if all_passed else "FALLBACK_TRIGGERED"

    return {
        "passed_all": all_passed,
        "decision": decision,
        "pipeline_selected": "A7-BDA" if all_passed else GATE_CRITERIA["fallback_candidate"],
        "checks": checks,
        "details": details,
        "criteria_frozen": GATE_CRITERIA,
    }


# ======================================================================
# TASK 5: REPORT & MANIFEST GENERATION
# ======================================================================

def write_phase7_v2_reports(
    config: PSDConfig,
    cv_results: Dict[str, Any],
    retrain_results: Dict[str, Any],
    gate_results: Dict[str, Any],
    p6_manifest: Dict[str, Any],
    source_experiment: str,
    phase7_run_id: Optional[str] = None,
) -> Dict[str, Path]:
    """Task 5: Generates all certified Phase 7 V2 reports, CSVs, JSON manifest, and Markdown report:
      1. final_cv_results.csv
      2. final_cv_class_performance.csv
      3. final_cv_confusion_matrix.csv
      4. psoriasis_dominance_diagnostic.csv
      5. calibration_set_performance.csv
      6. phase7_manifest.json
      7. phase7_report.md
    """
    out_dir = config.aef_crc_phase7_v2_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = config.target_classes

    agg: AggregatedMetrics = cv_results["aggregates"]
    fold_metrics: List[FoldMetrics] = cv_results["fold_metrics"]
    dim_stats = cv_results["dim_stats"]
    dominance = cv_results["dominance"]
    calib_m: FoldMetrics = retrain_results["calibration_metrics"]

    # 1. final_cv_results.csv
    cv_csv_path = out_dir / "final_cv_results.csv"
    with cv_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "dimension", "macro_f1", "balanced_accuracy", "mcc", "accuracy", "weighted_f1",
        ])
        for m, dim_cnt in zip(fold_metrics, dim_stats["per_fold"]):
            writer.writerow([
                f"fold_{m.fold_index}", dim_cnt,
                round(m.macro_f1, 6), round(m.balanced_accuracy, 6),
                round(m.mcc, 6), round(m.accuracy, 6), round(m.weighted_f1, 6),
            ])
        writer.writerow([
            "mean", round(dim_stats["mean"], 2),
            round(agg.macro_f1_mean, 6), round(agg.balanced_accuracy_mean, 6),
            round(agg.mcc_mean, 6), round(agg.accuracy_mean, 6), round(agg.weighted_f1_mean, 6),
        ])
        writer.writerow([
            "std", round(dim_stats["std"], 2),
            round(agg.macro_f1_std, 6), round(agg.balanced_accuracy_std, 6),
            round(agg.mcc_std, 6), round(agg.accuracy_std, 6), round(agg.weighted_f1_std, 6),
        ])

    # 2. final_cv_class_performance.csv
    class_perf_path = out_dir / "final_cv_class_performance.csv"
    with class_perf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name", "precision_mean", "precision_std", "recall_mean", "recall_std",
            "f1_mean", "f1_std", "support_total",
        ])
        for c in classes:
            writer.writerow([
                c,
                round(agg.per_class_precision_mean.get(c, 0.0), 4),
                round(agg.per_class_precision_std.get(c, 0.0), 4),
                round(agg.per_class_recall_mean.get(c, 0.0), 4),
                round(agg.per_class_recall_std.get(c, 0.0), 4),
                round(agg.per_class_f1_mean.get(c, 0.0), 4),
                round(agg.per_class_f1_std.get(c, 0.0), 4),
                agg.per_class_support_total.get(c, 0),
            ])

    # 3. final_cv_confusion_matrix.csv
    cm_path = out_dir / "final_cv_confusion_matrix.csv"
    with cm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class"] + list(classes))
        for cls, row in zip(agg.class_order, agg.confusion_sum):
            writer.writerow([cls] + [int(v) for v in row])

    # 4. psoriasis_dominance_diagnostic.csv
    dom_path = out_dir / "psoriasis_dominance_diagnostic.csv"
    with dom_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dominance.keys()))
        writer.writeheader()
        writer.writerow(dominance)

    # 5. calibration_set_performance.csv
    calib_perf_path = out_dir / "calibration_set_performance.csv"
    with calib_perf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric", "value",
        ])
        writer.writerow(["dataset_role", "held_out_validation (splitter.py val/)"])
        writer.writerow(["n_samples", 246])
        writer.writerow(["macro_f1", round(calib_m.macro_f1, 6)])
        writer.writerow(["balanced_accuracy", round(calib_m.balanced_accuracy, 6)])
        writer.writerow(["mcc", round(calib_m.mcc, 6)])
        writer.writerow(["accuracy", round(calib_m.accuracy, 6)])
        writer.writerow(["weighted_f1", round(calib_m.weighted_f1, 6)])
        for c in classes:
            cm_entry = calib_m.per_class.get(c)
            if cm_entry:
                writer.writerow([f"{c}_f1", round(cm_entry.f1, 4)])
                writer.writerow([f"{c}_precision", round(cm_entry.precision, 4)])
                writer.writerow([f"{c}_recall", round(cm_entry.recall, 4)])
                writer.writerow([f"{c}_support", cm_entry.support])

    # 6. phase7_manifest.json
    manifest_path = out_dir / "phase7_manifest.json"
    if phase7_run_id is None:
        phase7_run_id = f"AEFCRC_P7_V2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(out_dir).encode()).hexdigest()[:8]}"
    manifest_data = {
        "phase": "phase7_v2",
        "experiment_type": "model_selection_and_evaluation",
        "pipeline_name": "A7-BDA-RandomForest",
        "run_id": phase7_run_id,
        "upstream_phase6_run_id": p6_manifest.get("run_id"),
        "representation_id": P3_V2_REPR_ID,
        "source_experiment": source_experiment,
        "random_seed": config.random_seed,
        "pipeline_definition": {
            "arm_id": A7_ARM_ID,
            "arm_name": A7_ARM_NAME,
            "total_dimension": A7_EXPECTED_DIM,
            "branches": list(A7_BRANCHES),
            "branch_dimensions": A7_BRANCH_DIMS,
            "feature_selector": {
                "name": FEATURE_SELECTOR_NAME,
                "objective": "inner_validation_macro_f1",
                "penalty": 0.0,
                "proxy_classifier": "LogisticRegression(C=1.0, lbfgs, max_iter=300)",
                "proxy_weighting": "inner_train_balanced",
            },
            "primary_classifier": {
                "name": CLASSIFIER_NAME,
                "n_estimators": 300,
                "criterion": "gini",
                "max_features": "sqrt",
                "bootstrap": True,
                "sample_weight": "fold_local_balanced",
            },
        },
        "dataset_freeze_hash": p6_manifest.get("dataset_freeze_hash"),
        "fold_plan_hash": p6_manifest.get("fold_plan_hash"),
        "final_cv_summary": {
            "n_folds": 5,
            "dimension_mean": dim_stats["mean"],
            "dimension_std": dim_stats["std"],
            "macro_f1_mean": agg.macro_f1_mean,
            "macro_f1_std": agg.macro_f1_std,
            "balanced_accuracy_mean": agg.balanced_accuracy_mean,
            "balanced_accuracy_std": agg.balanced_accuracy_std,
            "mcc_mean": agg.mcc_mean,
            "mcc_std": agg.mcc_std,
            "accuracy_mean": agg.accuracy_mean,
            "accuracy_std": agg.accuracy_std,
            "weighted_f1_mean": agg.weighted_f1_mean,
            "weighted_f1_std": agg.weighted_f1_std,
            "per_class_f1": {c: agg.per_class_f1_mean.get(c, 0.0) for c in classes},
            "confusion_matrix_sum": agg.confusion_sum.tolist() if agg.confusion_sum is not None else None,
        },
        "final_retrain_summary": {
            "training_samples": 1146,
            "calibration_samples": 246,
            "production_bda_selected_count": retrain_results["selected_count"],
            "production_bda_branch_retained": retrain_results["branch_retained"],
            "calibration_performance": {
                "macro_f1": calib_m.macro_f1,
                "balanced_accuracy": calib_m.balanced_accuracy,
                "mcc": calib_m.mcc,
                "accuracy": calib_m.accuracy,
                "weighted_f1": calib_m.weighted_f1,
                "per_class_f1": {c: calib_m.per_class[c].f1 for c in classes if c in calib_m.per_class},
            },
        },
        "dominance_diagnostics": dominance,
        "gate_evaluation": gate_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reports_generated": [
            str(cv_csv_path),
            str(class_perf_path),
            str(cm_path),
            str(dom_path),
            str(calib_perf_path),
            str(manifest_path),
        ],
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 7. phase7_report.md
    report_md_path = out_dir / "phase7_report.md"
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write("# PapuloNet V2 Phase 7: Model Selection & Evaluation Report\n\n")
        f.write(f"**Generated:** {manifest_data['timestamp']}  \n")
        f.write(f"**Pipeline Candidate:** A7-BDA (`{A7_ARM_NAME}` with Binary Dragonfly Algorithm)  \n")
        f.write(f"**Classifier:** Random Forest (300 trees, balanced weights)  \n")
        f.write(f"**Gate Decision:** `{gate_results['decision']}` (Pipeline Selected: `{gate_results['pipeline_selected']}`)  \n\n")

        f.write("## 1. 5-Fold Stratified Cross-Validation Benchmark\n\n")
        f.write("| Metric | Mean +/- Std |\n|---|---|\n")
        f.write(f"| **Macro-F1** | **{agg.macro_f1_mean:.4f} +/- {agg.macro_f1_std:.4f}** |\n")
        f.write(f"| Balanced Accuracy | {agg.balanced_accuracy_mean:.4f} +/- {agg.balanced_accuracy_std:.4f} |\n")
        f.write(f"| MCC | {agg.mcc_mean:.4f} +/- {agg.mcc_std:.4f} |\n")
        f.write(f"| Accuracy | {agg.accuracy_mean:.4f} +/- {agg.accuracy_std:.4f} |\n")
        f.write(f"| Weighted F1 | {agg.weighted_f1_mean:.4f} +/- {agg.weighted_f1_std:.4f} |\n")
        f.write(f"| Selected Dimensions | {dim_stats['mean']:.1f} +/- {dim_stats['std']:.1f} / {A7_EXPECTED_DIM} |\n\n")

        f.write("## 2. Per-Class Generalization\n\n")
        f.write("| Class | Precision | Recall | F1 | Total Support |\n|---|---|---|---|---|\n")
        for c in classes:
            f.write(f"| {c} | {agg.per_class_precision_mean[c]:.4f} +/- {agg.per_class_precision_std[c]:.4f} | "
                    f"{agg.per_class_recall_mean[c]:.4f} +/- {agg.per_class_recall_std[c]:.4f} | "
                    f"**{agg.per_class_f1_mean[c]:.4f} +/- {agg.per_class_f1_std[c]:.4f}** | "
                    f"{agg.per_class_support_total[c]} |\n")
        f.write("\n")

        f.write("## 3. Final Retrain & Calibration-Set Performance (Holdout Val, N=246)\n\n")
        f.write(f"- **Unified Training Images:** 1,146 (all outer folds reunified, zero quarantined)\n")
        f.write(f"- **Authoritative Production Mask:** {retrain_results['selected_count']} / {A7_EXPECTED_DIM} features ({retrain_results['branch_retained']})\n")
        f.write(f"- **Calibration Partition:** 246 images (splitter.py held-out val/)\n")
        f.write(f"- **Calibration Macro-F1:** {calib_m.macro_f1:.4f}\n")
        f.write(f"- **Calibration Balanced Accuracy:** {calib_m.balanced_accuracy:.4f}\n")
        f.write(f"- **Calibration MCC:** {calib_m.mcc:.4f}\n\n")

        f.write("## 4. Dominance Diagnostics\n\n")
        f.write(f"- **Psoriasis True Ratio:** {dominance['Psoriasis_true_pct']}% ({dominance['Psoriasis_true_count']} / {dominance['total_samples']})\n")
        f.write(f"- **Psoriasis Pred Ratio:** {dominance['Psoriasis_pred_pct']}% ({dominance['Psoriasis_pred_count']} / {dominance['total_samples']})\n")
        f.write(f"- **Dominance Ratio:** **{dominance['Psoriasis_pred_ratio']:.4f}** (Gating threshold: <= {GATE_CRITERIA['max_psoriasis_pred_ratio']})\n\n")

        f.write("## 5. Predefined Gating Evaluation\n\n")
        for chk, res in gate_results["checks"].items():
            f.write(f"- **{chk}:** {'PASS' if res else 'FAIL'}\n")

    return {
        "final_cv_results_csv": cv_csv_path,
        "final_cv_class_performance_csv": class_perf_path,
        "final_cv_confusion_matrix_csv": cm_path,
        "psoriasis_dominance_diagnostic_csv": dom_path,
        "calibration_set_performance_csv": calib_perf_path,
        "phase7_manifest_json": manifest_path,
        "phase7_report_md": report_md_path,
    }


# ======================================================================
# FRAMEWORK VALIDATION
# ======================================================================

def validate_phase7_v2_framework(config: PSDConfig) -> bool:
    """Non-destructive validation of Phase 7 V2 contracts and invariants on synthetic data:
    1. Upstream Phase 6 V2-A7 manifest presence and provenance.
    2. A7-BDA pipeline definition and dimension sanity.
    3. Deep feature preflight (folds 0..4 + fold -1 final_train and calibration).
    4. Handcrafted feature preflight for development (1146) and calibration (246).
    5. Partition isolation (1146 dev, 246 calib, 243 test, 264 quarantined).
    6. Non-destructive micro-benchmark of A7-BDA on synthetic 1316-D data.
    7. Inner proxy balanced sample weights and outer RF balanced class weights.
    8. Production BDA mask format and compatibility with classifier.
    9. Calibration handoff stratification (123 / 123) and persistence validation.
    10. Output directory isolation (phase7_v2 paths only, zero collision).
    """
    print("=" * 70)
    print("PHASE 7 V2 (A7-BDA) FRAMEWORK & CONTRACT VALIDATION")
    print("=" * 70)

    config = configure_phase7_v2(config)

    # 0. Representation ID binding
    print("\n[CHECK 0/10] Verifying Phase 7 V2 representation binding...")
    repr_id = representation_id(config)
    assert repr_id == P3_V2_REPR_ID, f"Config representation_id={repr_id} != expected {P3_V2_REPR_ID}"
    print(f"  PASS: Representation ID strictly bound: {repr_id}")

    # 1. Gate check on real upstream manifest
    print("\n[CHECK 1/10] Verifying upstream Phase 6 V2-A7 manifest...")
    gate_ok, gate_msg, p6_manifest = gate_check_phase7_v2(config)
    assert gate_ok, f"Gate check failed: {gate_msg}"
    print(f"  PASS: {gate_msg}")

    # 2. Pipeline definition
    print("\n[CHECK 2/10] Verifying A7-BDA pipeline definition...")
    assert A7_EXPECTED_DIM == 1316
    assert A7_BRANCHES == ("deep", "glcm", "lbp", "color_lab")
    assert sum(A7_BRANCH_DIMS.values()) == 1316
    print(f"  PASS: A7-BDA definition valid: 1316-D, branches={A7_BRANCHES}, classifier=RandomForest(300)")

    # 3. Load plan and check partition isolation
    print("\n[CHECK 3/10] Verifying fold plan and partition isolation...")
    plan = load_frozen_fold_plan(config)
    iso_ok, iso_msg = preflight_partition_isolation(plan)
    assert iso_ok, f"Partition isolation check failed: {iso_msg}"
    print(f"  PASS: {iso_msg}")

    # 4. Deep feature artifacts check
    print("\n[CHECK 4/10] Preflight checking deep-feature artifacts (folds 0..4 + fold -1)...")
    deep_ok, deep_msg = preflight_deep_feature_artifacts(config, P3_V2_FOCAL_EXP, plan.folds)
    assert deep_ok, f"Deep feature preflight failed: {deep_msg}"
    print(f"  PASS: {deep_msg}")

    # 5. Handcrafted feature cache check
    print("\n[CHECK 5/10] Preflight checking handcrafted caches for dev (1146) and calib (246)...")
    hc_ok, hc_msg = preflight_handcrafted_features(config, plan)
    assert hc_ok, f"Handcrafted preflight failed: {hc_msg}"
    print(f"  PASS: {hc_msg}")

    # 6. Micro-benchmark on synthetic 1316-D data
    print("\n[CHECK 6/10] Running non-destructive micro-benchmark on synthetic 1316-D data...")
    rng = np.random.default_rng(42)
    n_tr, n_va, n_dim = 80, 20, A7_EXPECTED_DIM
    X_tr = rng.standard_normal((n_tr, n_dim)).astype(np.float32)
    y_tr = ["Psoriasis"] * 45 + ["Lichen_Planus"] * 18 + ["Pityriasis_Rosea"] * 11 + ["Seborrheic_Dermatitis"] * 6
    X_va = rng.standard_normal((n_va, n_dim)).astype(np.float32)
    y_va = ["Psoriasis"] * 11 + ["Lichen_Planus"] * 5 + ["Pityriasis_Rosea"] * 3 + ["Seborrheic_Dermatitis"] * 1

    synth_data = FusionFoldData(
        X_train=X_tr, y_train=y_tr, psd_ids_train=[f"TR-{i:03d}" for i in range(n_tr)],
        X_val=X_va, y_val=y_va, psd_ids_val=[f"VA-{i:03d}" for i in range(n_va)],
        branch_dims=A7_BRANCH_DIMS,
    )

    micro_cfg = dataclasses.replace(
        config,
        bda_population_size=4,
        bda_iterations=3,
        bda_feature_count_penalty=0.0,
    )
    sel_res = run_selector("bda", synth_data, micro_cfg, random_seed=42)
    assert sel_res.selected_mask.shape == (A7_EXPECTED_DIM,)
    assert sel_res.selected_count > 0
    fb_synth = compute_feature_family_breakdown(sel_res.selected_mask, A7_BRANCH_DIMS)
    print(f"  PASS: Micro-benchmark selected {sel_res.selected_count}/1316 features ({fb_synth})")

    # 7. Inner sample weights & outer classifier fit
    print("\n[CHECK 7/10] Verifying inner sample weights & outer classifier fit...")
    classes = config.target_classes
    class_weights = compute_train_fold_class_weights(y_tr, classes)
    sw_tr = compute_sample_weights(y_tr, class_weights)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_tr[:, sel_res.selected_mask], y_tr, sample_weight=sw_tr)
    y_pred = list(clf.predict(X_va[:, sel_res.selected_mask]))
    metrics = compute_fold_metrics(y_va, y_pred, classes, fold_index=0)
    print(f"  PASS: Synthetic fold evaluation succeeded: Macro-F1={metrics.macro_f1:.4f}")

    # 8. Production BDA mask validation
    print("\n[CHECK 8/10] Verifying production BDA mask contract & compatibility...")
    validate_bda_mask_compatibility(sel_res.selected_mask, clf, expected_dim=A7_EXPECTED_DIM)
    print("  PASS: Production BDA mask contract and classifier compatibility verified.")

    # 9. Gating logic verification
    print("\n[CHECK 9/10] Verifying predefined gating evaluation logic...")
    fake_agg = AggregatedMetrics(
        n_folds=5, macro_f1_mean=0.7143, macro_f1_std=0.0361,
        accuracy_mean=0.7574, accuracy_std=0.0245,
        balanced_accuracy_mean=0.7106, balanced_accuracy_std=0.0346,
        weighted_f1_mean=0.7526, weighted_f1_std=0.0247,
        mcc_mean=0.5991, mcc_std=0.0420,
        per_class_f1_mean={"Psoriasis": 0.8228, "Lichen_Planus": 0.6263, "Pityriasis_Rosea": 0.7017, "Seborrheic_Dermatitis": 0.7063},
        per_class_f1_std={"Psoriasis": 0.023, "Lichen_Planus": 0.037, "Pityriasis_Rosea": 0.065, "Seborrheic_Dermatitis": 0.088},
        per_class_precision_mean={"Psoriasis": 0.7946, "Lichen_Planus": 0.7205, "Pityriasis_Rosea": 0.6958, "Seborrheic_Dermatitis": 0.7102},
        per_class_precision_std={"Psoriasis": 0.035, "Lichen_Planus": 0.036, "Pityriasis_Rosea": 0.076, "Seborrheic_Dermatitis": 0.101},
        per_class_recall_mean={"Psoriasis": 0.8542, "Lichen_Planus": 0.5623, "Pityriasis_Rosea": 0.7100, "Seborrheic_Dermatitis": 0.7157},
        per_class_recall_std={"Psoriasis": 0.024, "Lichen_Planus": 0.081, "Pityriasis_Rosea": 0.066, "Seborrheic_Dermatitis": 0.119},
        per_class_support_total={"Psoriasis": 638, "Lichen_Planus": 258, "Pityriasis_Rosea": 162, "Seborrheic_Dermatitis": 88},
        confusion_sum=np.zeros((len(classes), len(classes)), dtype=int),
        class_order=list(classes),
    )
    fake_calib = compute_fold_metrics(y_va, y_pred, classes, fold_index=-1)
    fake_dom = {
        "Psoriasis_pred_ratio": 1.0768,
        "Psoriasis_pred_pct": 59.95,
        "Psoriasis_true_pct": 55.67,
    }
    gate_res = evaluate_phase7_v2_gate(fake_agg, fake_calib, fake_dom)
    print(f"  PASS: Predefined gate evaluation evaluated cleanly: decision='{gate_res['decision']}'")

    # 10. Output directory isolation
    print("\n[CHECK 10/10] Verifying output directory isolation (phase7_v2 paths only)...")
    v2_rep = config.aef_crc_phase7_v2_reports_dir
    v2_art = config.aef_crc_phase7_v2_artifacts_dir
    v2_log = config.aef_crc_phase7_v2_logs_dir
    v1_rep = config.project_root / "reports" / "phase7"
    v1_art = config.project_root / "artifacts" / "phase7"
    p6_rep = config.project_root / "reports" / "phase6_v2_a7"

    assert "phase7_v2" in str(v2_rep), f"Report path {v2_rep} not in phase7_v2"
    assert "phase7_v2" in str(v2_art), f"Artifact path {v2_art} not in phase7_v2"
    assert "phase7_v2" in str(v2_log), f"Log path {v2_log} not in phase7_v2"
    assert str(v2_rep) != str(v1_rep), "Phase 7 V2 reports collision with V1"
    assert str(v2_art) != str(v1_art), "Phase 7 V2 artifacts collision with V1"
    assert str(v2_rep) != str(p6_rep), "Phase 7 V2 reports collision with Phase 6"

    print(f"  Reports:   {v2_rep}")
    print(f"  Artifacts: {v2_art}")
    print(f"  Logs:      {v2_log}")
    print("  PASS: All Phase 7 V2 paths strictly isolated from certified Phase 6 and V1 directories.")

    print("\n" + "=" * 70)
    print("PHASE 7 V2 (A7-BDA) FRAMEWORK VALIDATION COMPLETE: ALL 10 CHECKS PASSED.")
    print("=" * 70)
    return True


# ======================================================================
# MAIN CLI ENTRYPOINT
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="PapuloNet V2 Phase 7: Model Selection & Evaluation Pipeline (A7-BDA)")
    parser.add_argument("--validate-framework", action="store_true", help="Run non-destructive framework validation checks")
    parser.add_argument("--validate-only", action="store_true", help="Alias for --validate-framework")
    args = parser.parse_args()

    config = configure_phase7_v2(get_config())

    if args.validate_framework or args.validate_only:
        ok = validate_phase7_v2_framework(config)
        return 0 if ok else 1

    print("=" * 70)
    print("PAPULONET V2 PHASE 7: MODEL SELECTION & EVALUATION (A7-BDA PIPELINE)")
    print("=" * 70)

    # Step 0: Gate check
    print("\n--- Step 0: Phase 7 V2 Gate Check ---")
    gate_ok, gate_msg, p6_manifest = gate_check_phase7_v2(config)
    print(gate_msg)
    if not gate_ok:
        print("\n[PHASE 7 V2 BLOCKER] Gate check failed. Aborting execution.")
        return 1

    # Load frozen fold plan
    plan = load_frozen_fold_plan(config)
    validate_fold_plan(plan, config)

    # Preflight checks
    print("\n--- Preflight Checks ---")
    deep_ok, deep_msg = preflight_deep_feature_artifacts(config, P3_V2_FOCAL_EXP, plan.folds)
    print(f"Deep features:       {'PASS' if deep_ok else 'FAIL'} - {deep_msg}")
    if not deep_ok:
        return 1

    hc_ok, hc_msg = preflight_handcrafted_features(config, plan)
    print(f"Handcrafted features: {'PASS' if hc_ok else 'FAIL'} - {hc_msg}")
    if not hc_ok:
        return 1

    iso_ok, iso_msg = preflight_partition_isolation(plan)
    print(f"Partition isolation: {'PASS' if iso_ok else 'FAIL'} - {iso_msg}")
    if not iso_ok:
        return 1

    # Task 1: 5-Fold Stratified Cross-Validation Benchmark
    print("\n--- Task 1: Full 5-Fold Stratified Cross-Validation Benchmark ---")
    cv_results = run_phase7_v2_cv(config, plan, P3_V2_FOCAL_EXP, label_prefix="  ")

    # Task 2: Final Retrain on Unified 1146 Development Cohort
    print("\n--- Task 2: Final Model Retrain on Unified 1146 Cohort ---")
    retrain_results = run_phase7_v2_final_retrain(config, plan, P3_V2_FOCAL_EXP, label_prefix="  ")

    # Authoritative Phase 7 Run ID
    phase7_run_id = f"AEFCRC_P7_V2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(config.aef_crc_phase7_v2_reports_dir).encode()).hexdigest()[:8]}"

    # Task 3: Calibration Handoff Generation
    print("\n--- Task 3: Calibration Handoff Generation & Persistence ---")
    handoff = build_phase7_v2_calibration_handoff(
        config, plan, retrain_results, p6_manifest, P3_V2_FOCAL_EXP, phase7_run_id=phase7_run_id,
    )

    # Task 4: Predefined Gating Evaluation
    print("\n--- Task 4: Predefined Phase 7 Gating Evaluation ---")
    gate_results = evaluate_phase7_v2_gate(
        cv_results["aggregates"],
        retrain_results["calibration_metrics"],
        cv_results["dominance"],
    )
    print(f"  Decision:          {gate_results['decision']}")
    print(f"  Selected Pipeline: {gate_results['pipeline_selected']}")
    for chk, status in gate_results["checks"].items():
        print(f"    - {chk}: {'PASS' if status else 'FAIL'}")

    # Task 5: Report Generation
    print("\n--- Task 5: Generating Certified Phase 7 V2 Reports & Manifest ---")
    reports = write_phase7_v2_reports(
        config, cv_results, retrain_results, gate_results, p6_manifest, P3_V2_FOCAL_EXP, phase7_run_id=phase7_run_id,
    )
    for k, p in reports.items():
        print(f"  [{k}] {p}")

    print("\n" + "=" * 70)
    print("PAPULONET V2 PHASE 7 EXECUTION COMPLETE.")
    print(f"FINAL DECISION: {gate_results['decision']} -> PIPELINE: {gate_results['pipeline_selected']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
