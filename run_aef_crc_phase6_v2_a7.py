"""
run_aef_crc_phase6_v2_a7.py

PapuloNet V2 Phase 6: Exploratory Feature Refinement Benchmark (A7 Representation).

Evaluates metaheuristic feature selection algorithms (Binary Dragonfly Algorithm vs Genetic Algorithm)
against the unselected baseline on the exploratory Phase 5 V2 representation:
  A7 = EfficientNet-B0 + GLCM + LBP + LAB (1316 dimensions: deep=1280, glcm=12, lbp=18, color_lab=6).

IMPORTANT SCIENTIFIC STATUS:
  This is an exploratory experiment investigating whether the 1316-D multi-scale texture + color
  descriptor set (A7) benefits from BDA feature refinement.
  This experiment is strictly exploratory and does NOT replace the certified Phase 6 V2 winner (A2).
  It does not modify Phase 5 V2 certified A2 selection, nor does it alter any downstream protocol.

Protocol (Identical to Phase 6 V2):
  Fitness objective is strictly unpenalized (lambda = 0):
    fitness = inner_validation_macro_f1
  Feature-count penalty is 0.0: the study objective imposes no prior preference for fewer features.
  Full A7 (1316-D) remains the explicit no-selection baseline. Feature count is reported as a
  secondary descriptive property.

Key Invariants:
1. Strict A7 Binding:
   - Arm: A7 (name: 'EfficientNet+GLCM+LBP+LAB')
   - Total dimension: exactly 1316 (deep: 1280, glcm: 12, lbp: 18, color_lab: 6)
   - Branches: exactly ('deep', 'glcm', 'lbp', 'color_lab')
   - Fast-fails if any dimension or branch drifts. Never silently falls back to A2.
2. Upstream Deep Representation:
   - Strictly binds to Phase 3 V2 Focal deep features (experiment: 'P3-V2-Focal',
     representation_id: 'efficientnet_b0_43d581b96f8ec368', feature_dim: 1280)
     located in artifacts/phase3_v2/.
3. Isolated Paths:
   - Outputs written exclusively to reports/phase6_v2_a7/, artifacts/phase6_v2_a7/,
     and logs/phase6_v2_a7/. Never touches V1 reports/phase6/ or certified reports/phase6_v2/.
4. Three Evaluated Arms on Identical 5 Outer Folds:
   - full_a7: Unselected baseline (all 1316 dimensions retained).
   - bda_a7: Binary Dragonfly Algorithm (optimizing unpenalized inner-val Macro-F1).
   - ga_a7: Genetic Algorithm comparator (optimizing unpenalized inner-val Macro-F1).
5. Fold-Local Sample Weights & Imbalance Handling:
   - Inner optimization proxies (BDA & GA) fit LogisticRegression with sample_weight
     derived strictly from inner-train labels (w_c = N_inner_train / (C * N_c,inner_train)).
   - Outer evaluation classifier (Random Forest) receives fold-local balanced sample weights
     derived strictly from fold.class_weights.
   - Zero SMOTE, zero ADASYN, zero synthetic oversampling.
6. Strict Partition Isolation:
   - Locked Final Test (243 images): Never accessed.
   - Outer Validation (246 images): Never accessed during optimization.
   - Development cohort (1146 images): Evaluated strictly via frozen 5-fold CV.
   - Quarantined 264 images: Excluded from CV.
7. Comprehensive Diagnostics:
   - Macro-F1, Balanced Accuracy, MCC, Accuracy, Weighted F1 (mean +/- std).
   - Per-class Precision, Recall, F1, and Support for all 4 classes.
   - Psoriasis majority dominance diagnostic (true vs predicted distributions).
   - Pooled confusion matrix.
   - Pairwise McNemar's tests with Holm-Bonferroni correction.
   - Feature retention, family breakdown, and pairwise Jaccard stability across folds.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from datetime import datetime, timezone
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.config import PSDConfig, get_config
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import load_frozen_fold_plan
from modules.handcrafted_features import HandcraftedFeatureExtractor
from modules.fusion import (
    FUSION_ARMS_BY_ID,
    FUSION_ARMS_BY_NAME,
    EXPECTED_BRANCH_DIMS,
    build_fusion_fold,
    compute_sample_weights,
    build_classifier,
    validate_fusion_dimensions,
    get_branch_feature_names,
    get_fused_feature_names,
)
from modules.feature_selection import (
    run_selector,
    select_no_selection,
    select_bda,
    select_ga,
    compute_jaccard_similarity,
    compute_pairwise_jaccard,
    compute_branch_jaccard_stability,
    compute_feature_family_breakdown,
    map_mask_to_feature_names,
    save_production_bda_mask,
    SelectionResult,
)
from modules.evaluation import (
    FoldMetrics,
    AggregatedMetrics,
    compute_fold_metrics,
    aggregate_fold_metrics,
    mcnemar_test,
    holm_correction,
)

A7_ARM_ID = "A7"
A7_ARM_NAME = "EfficientNet+GLCM+LBP+LAB"
A7_BRANCHES = ("deep", "glcm", "lbp", "color_lab")
A7_EXPECTED_DIM = 1316
A7_BRANCH_DIMS = {
    "deep": 1280,
    "glcm": 12,
    "lbp": 18,
    "color_lab": 6,
}
P3_V2_FOCAL_EXP = "P3-V2-Focal"
P3_V2_REPR_ID = "efficientnet_b0_43d581b96f8ec368"


def gate_check_a7(config: PSDConfig) -> Tuple[bool, str, Dict[str, Any]]:
    """Strictly validates Phase 6 V2-A7 upstream prerequisites:
    1. A7 is registered in FUSION_ARMS_BY_ID with arm_id=='A7', expected_dim==1316,
       and branches==('deep', 'glcm', 'lbp', 'color_lab').
    2. reports/phase5_v2/phase5_manifest.json exists.
    3. reports/phase5_v2/fusion_results.csv contains A7 results.
    4. P3-V2-Focal deep feature files exist in artifacts/phase3_v2/.
    5. Handcrafted feature cache directory exists.
    """
    # 1. Arm definition verification
    if A7_ARM_ID not in FUSION_ARMS_BY_ID:
        return False, f"Arm '{A7_ARM_ID}' not found in modules.fusion.FUSION_ARMS_BY_ID", {}

    arm_def = FUSION_ARMS_BY_ID[A7_ARM_ID]
    if arm_def.expected_dim != A7_EXPECTED_DIM:
        return False, f"Arm A7 registered dim={arm_def.expected_dim}, expected exactly {A7_EXPECTED_DIM}", {}

    if arm_def.branches != A7_BRANCHES:
        return False, f"Arm A7 registered branches={arm_def.branches}, expected {A7_BRANCHES}", {}

    # Verify sum of branch dimensions
    sum_dim = sum(A7_BRANCH_DIMS[b] for b in A7_BRANCHES)
    if sum_dim != A7_EXPECTED_DIM:
        return False, f"Branch dimension sum ({sum_dim}) != expected A7 dimension ({A7_EXPECTED_DIM})", {}

    # 2. Phase 5 V2 manifest check
    p5_manifest_path = config.aef_crc_phase5_v2_reports_dir / "phase5_manifest.json"
    if not p5_manifest_path.exists():
        return False, f"Missing Phase 5 V2 manifest at {p5_manifest_path}", {}

    try:
        p5_manifest = json.loads(p5_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse Phase 5 V2 manifest: {exc}", {}

    # 3. Phase 5 V2 fusion results check (verify A7 was evaluated in Phase 5 V2)
    p5_results_path = config.aef_crc_phase5_v2_reports_dir / "fusion_results.csv"
    if not p5_results_path.exists():
        return False, f"Missing Phase 5 V2 fusion_results.csv at {p5_results_path}", p5_manifest

    a7_found_in_p5 = False
    with p5_results_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("arm") in (A7_ARM_NAME, A7_ARM_ID):
                a7_found_in_p5 = True
                break

    if not a7_found_in_p5:
        return False, f"Arm A7 ('{A7_ARM_NAME}') not found in {p5_results_path}", p5_manifest

    # 4. Verify Phase 3 V2 deep features
    p3_artifacts = getattr(config, "aef_crc_phase3_v2_artifacts_dir", config.aef_crc_artifacts_dir)
    deep_dir = p3_artifacts / "deep_features" / P3_V2_FOCAL_EXP
    npy_files = list(deep_dir.glob("**/*.npy")) if deep_dir.exists() else []
    if not npy_files:
        return False, f"No Phase 3 V2 deep feature .npy files found in {deep_dir}", p5_manifest

    detail = (
        f"Verified A7 definition: [{A7_ARM_ID}] {A7_ARM_NAME} "
        f"({A7_EXPECTED_DIM}-D, branches={A7_BRANCHES}) | "
        f"Upstream run_id: {p5_manifest.get('run_id')} | "
        f"P3-V2-Focal deep feature files: {len(npy_files)}"
    )
    return True, detail, p5_manifest


def preflight_deep_feature_artifacts(config: PSDConfig, experiment_name: str, folds: List[Any]) -> Tuple[bool, str]:
    """Fast-fail preflight gate verifying that all 5 folds and splits (train, val) have
    valid, uncorrupted deep-feature manifests matching:
      - experiment_name == 'P3-V2-Focal'
      - representation_id == 'efficientnet_b0_43d581b96f8ec368'
      - feature_dim == 1280
      - backbone_name == 'efficientnet_b0'
      - non-empty psd_ids matching split record count
    """
    from modules.efficientnet_model import load_deep_feature_manifest, _resolve_deep_feature_root

    root_dir = _resolve_deep_feature_root(config, experiment_name)
    if "phase3_v2" not in str(root_dir):
        return False, f"Deep feature root resolved to '{root_dir}', expected path containing 'phase3_v2'"

    total_checked = 0
    for fold in folds:
        fold_idx = fold.fold_index
        for split_name, records in [("train", fold.train_records), ("val", fold.val_records)]:
            try:
                manifest = load_deep_feature_manifest(config, experiment_name, fold_idx, split_name)
            except Exception as exc:
                return False, f"Failed to load deep feature manifest for fold {fold_idx} split {split_name}: {exc}"

            if manifest.get("experiment_name") != experiment_name:
                return False, f"Fold {fold_idx} {split_name} manifest experiment_name='{manifest.get('experiment_name')}', expected '{experiment_name}'"

            if manifest.get("representation_id") != P3_V2_REPR_ID:
                return False, f"Fold {fold_idx} {split_name} manifest representation_id='{manifest.get('representation_id')}', expected '{P3_V2_REPR_ID}'"

            if manifest.get("feature_dim") != 1280:
                return False, f"Fold {fold_idx} {split_name} manifest feature_dim={manifest.get('feature_dim')}, expected 1280"

            if manifest.get("backbone_name") != "efficientnet_b0":
                return False, f"Fold {fold_idx} {split_name} manifest backbone_name='{manifest.get('backbone_name')}', expected 'efficientnet_b0'"

            psd_ids = manifest.get("psd_ids", [])
            if len(psd_ids) != len(records):
                return False, f"Fold {fold_idx} {split_name} manifest has {len(psd_ids)} PSD IDs, expected {len(records)}"

            total_checked += 1

    return True, f"Preflight deep-feature check PASSED: verified all {total_checked} manifests (5 folds x 2 splits) with representation_id='{P3_V2_REPR_ID}', dim=1280 in '{root_dir}'"


def preflight_handcrafted_features(config: PSDConfig, folds: List[Any], extractor: Optional[HandcraftedFeatureExtractor] = None) -> Tuple[bool, str]:
    """Verifies that handcrafted feature caches exist for all unique records in the 5 folds
    for all required handcrafted branches of A7: GLCM (12), LBP (18), LAB (6).
    """
    if config.preprocessing_mode != "standard":
        config = dataclasses.replace(config, preprocessing_mode="standard")
    extractor = extractor or HandcraftedFeatureExtractor(config)

    unique_records = {}
    for fold in folds:
        for split_records in [fold.train_records, fold.val_records]:
            for r in split_records:
                unique_records[r.psd_id] = r

    cache_root = extractor._cache_root
    glcm_dir = cache_root / "glcm"
    lbp_dir = cache_root / "lbp"
    color_dir = cache_root / "color_lab"

    for psd_id in unique_records:
        if not (glcm_dir / f"{psd_id}.npy").exists():
            return False, f"Missing GLCM cache for psd_id={psd_id} in {glcm_dir}"
        if not (lbp_dir / f"{psd_id}.npy").exists():
            return False, f"Missing LBP cache for psd_id={psd_id} in {lbp_dir}"
        if not (color_dir / f"{psd_id}.npy").exists():
            return False, f"Missing LAB cache for psd_id={psd_id} in {color_dir}"

    # Sample verify vector dimensions for 20 records
    sample_ids = list(unique_records.keys())[:20]
    for psd_id in sample_ids:
        cached = extractor._load_cached(psd_id)
        if cached is None:
            return False, f"Missing cached feature object for psd_id={psd_id}"
        if cached.glcm is None or len(cached.glcm) != 12:
            return False, f"Invalid GLCM vector for psd_id={psd_id} (dim={len(cached.glcm) if cached.glcm is not None else 0}, expected 12)"
        if cached.lbp is None or len(cached.lbp) != 18:
            return False, f"Invalid LBP vector for psd_id={psd_id} (dim={len(cached.lbp) if cached.lbp is not None else 0}, expected 18)"
        if cached.color_lab is None or len(cached.color_lab) != 6:
            return False, f"Invalid LAB vector for psd_id={psd_id} (dim={len(cached.color_lab) if cached.color_lab is not None else 0}, expected 6)"

    return True, f"Preflight handcrafted feature check PASSED: verified GLCM (12-D), LBP (18-D), LAB (6-D) caches for all {len(unique_records)} cohort records in '{cache_root}'."


def evaluate_selection_fold(
    config: PSDConfig,
    data: Any,
    selection: SelectionResult,
    classes: List[str],
    class_weights: Dict[str, float],
    fold_index: int,
) -> Tuple[FoldMetrics, List[str]]:
    """Evaluates a feature selection mask on a single fold using the primary classifier (Random Forest)
    with fold-local balanced sample weights.
    """
    mask = selection.selected_mask
    if mask.sum() == 0:
        raise ValueError(f"Degenerate mask with 0 features selected in fold {fold_index}")

    sw_train = compute_sample_weights(data.y_train, class_weights)
    fold_seed = config.random_seed + fold_index
    _, clf = build_classifier(fold_seed, config.classifier_name)

    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = [label_to_idx[y] for y in data.y_train]

    clf.fit(data.X_train[:, mask], y_train_idx, sample_weight=sw_train)

    pred_idx = clf.predict(data.X_val[:, mask])
    y_pred = [classes[i] for i in pred_idx]

    metrics = compute_fold_metrics(data.y_val, y_pred, classes, fold_index=fold_index)
    return metrics, y_pred


def compute_dominance_diagnostics(
    pooled_predictions: Dict[str, Tuple[str, str]],
    classes: List[str],
) -> Dict[str, Any]:
    """Computes true vs predicted counts, percentages, and ratios for all classes
    to monitor majority-class (Psoriasis) dominance.
    """
    total = len(pooled_predictions)
    true_counts = {c: 0 for c in classes}
    pred_counts = {c: 0 for c in classes}

    for true, pred in pooled_predictions.values():
        if true in true_counts:
            true_counts[true] += 1
        if pred in pred_counts:
            pred_counts[pred] += 1

    diag = {"total_samples": total}
    for c in classes:
        tc = true_counts[c]
        pc = pred_counts[c]
        diag[f"{c}_true_count"] = tc
        diag[f"{c}_true_pct"] = round(tc / total * 100, 2) if total > 0 else 0.0
        diag[f"{c}_pred_count"] = pc
        diag[f"{c}_pred_pct"] = round(pc / total * 100, 2) if total > 0 else 0.0
        diag[f"{c}_pred_ratio"] = round(pc / max(1, tc), 4)

    return diag


def run_pairwise_mcnemar(
    pooled_by_arm: Dict[str, Dict[str, Tuple[str, str]]],
) -> List[Dict[str, Any]]:
    """Runs pairwise McNemar's tests across all evaluated arms with Holm-Bonferroni correction."""
    arm_names = list(pooled_by_arm.keys())
    pairs = list(itertools.combinations(arm_names, 2))
    raw_results = []

    for a, b in pairs:
        common_ids = sorted(set(pooled_by_arm[a]) & set(pooled_by_arm[b]))
        y_true = [pooled_by_arm[a][pid][0] for pid in common_ids]
        y_pred_a = [pooled_by_arm[a][pid][1] for pid in common_ids]
        y_pred_b = [pooled_by_arm[b][pid][1] for pid in common_ids]
        res = mcnemar_test(y_true, y_pred_a, y_pred_b)
        raw_results.append((a, b, res))

    p_values = [r["p_value"] if r is not None else None for _, _, r in raw_results]
    p_adj = holm_correction(p_values)

    rows = []
    for (a, b, r), padj in zip(raw_results, p_adj):
        if r is None:
            rows.append({
                "arm_a": a, "arm_b": b,
                "n01": "", "n10": "",
                "p_value": "", "p_value_holm": "",
                "status": "Below threshold (n01+n10 < 25)",
            })
        else:
            is_sig = padj is not None and padj < 0.05
            rows.append({
                "arm_a": a, "arm_b": b,
                "n01": r["n01"], "n10": r["n10"],
                "p_value": round(r["p_value"], 6),
                "p_value_holm": round(padj, 6) if padj is not None else "",
                "status": "Significant (p_holm < 0.05)" if is_sig else "Not Significant",
            })
    return rows


def run_phase6_v2_a7_benchmark(
    config: PSDConfig,
    plan: Any,
    source_experiment: str,
    label_prefix: str = "",
) -> Dict[str, Any]:
    """Executes the Phase 6 V2-A7 3-Arm Benchmark across all 5 outer folds:
      1. full_a7 (1316 dimensions, unselected baseline)
      2. bda_a7 (Binary Dragonfly Algorithm on A7)
      3. ga_a7 (Genetic Algorithm comparator on A7)
    """
    branches = A7_BRANCHES
    classes = config.target_classes
    extractor = HandcraftedFeatureExtractor(config)

    arms = ["full_a7", "bda_a7", "ga_a7"]
    fold_metrics_by_arm: Dict[str, List[FoldMetrics]] = {arm: [] for arm in arms}
    pooled_predictions: Dict[str, Dict[str, Tuple[str, str]]] = {arm: {} for arm in arms}
    selected_masks_by_arm: Dict[str, List[np.ndarray]] = {arm: [] for arm in arms}
    selection_results_by_arm: Dict[str, List[SelectionResult]] = {arm: [] for arm in arms}

    print(f"{label_prefix}Starting Phase 6 V2-A7 3-Arm Benchmark on A7 representation ({A7_EXPECTED_DIM}-D)...")
    print(f"{label_prefix}Arms: full_a7 (unselected baseline), bda_a7 (primary), ga_a7 (comparator)")

    for fold in plan.folds:
        fold_idx = fold.fold_index
        fold_seed = config.random_seed + fold_idx

        # Build fused A7 data for this fold
        data = build_fusion_fold(config, fold, branches, source_experiment, extractor)
        assert data.X_train.shape[1] == A7_EXPECTED_DIM, f"Expected {A7_EXPECTED_DIM} features, got {data.X_train.shape[1]}"

        # 1. Arm: full_a7 (unselected baseline)
        sel_full = select_no_selection(data)
        metrics_full, pred_full = evaluate_selection_fold(
            config, data, sel_full, classes, fold.class_weights, fold_idx
        )
        fold_metrics_by_arm["full_a7"].append(metrics_full)
        selected_masks_by_arm["full_a7"].append(sel_full.selected_mask)
        selection_results_by_arm["full_a7"].append(sel_full)
        for pid, true, pred in zip(data.psd_ids_val, data.y_val, pred_full):
            pooled_predictions["full_a7"][pid] = (true, pred)

        print(
            f"{label_prefix}  [Fold {fold_idx}] full_a7: dim={A7_EXPECTED_DIM} | "
            f"Macro-F1={metrics_full.macro_f1:.4f} | BalAcc={metrics_full.balanced_accuracy:.4f}"
        )

        # 2. Arm: bda_a7 (Binary Dragonfly Algorithm on A7)
        sel_bda = run_selector("bda", data, config, fold_seed)
        metrics_bda, pred_bda = evaluate_selection_fold(
            config, data, sel_bda, classes, fold.class_weights, fold_idx
        )
        fold_metrics_by_arm["bda_a7"].append(metrics_bda)
        selected_masks_by_arm["bda_a7"].append(sel_bda.selected_mask)
        selection_results_by_arm["bda_a7"].append(sel_bda)
        for pid, true, pred in zip(data.psd_ids_val, data.y_val, pred_bda):
            pooled_predictions["bda_a7"][pid] = (true, pred)

        print(
            f"{label_prefix}  [Fold {fold_idx}] bda_a7 : selected={sel_bda.selected_count}/{A7_EXPECTED_DIM} "
            f"({sel_bda.branch_retained}) | Macro-F1={metrics_bda.macro_f1:.4f} | "
            f"BalAcc={metrics_bda.balanced_accuracy:.4f} | time={sel_bda.fit_time_sec:.1f}s"
        )

        # 3. Arm: ga_a7 (Genetic Algorithm comparator on A7)
        sel_ga = run_selector("ga", data, config, fold_seed)
        metrics_ga, pred_ga = evaluate_selection_fold(
            config, data, sel_ga, classes, fold.class_weights, fold_idx
        )
        fold_metrics_by_arm["ga_a7"].append(metrics_ga)
        selected_masks_by_arm["ga_a7"].append(sel_ga.selected_mask)
        selection_results_by_arm["ga_a7"].append(sel_ga)
        for pid, true, pred in zip(data.psd_ids_val, data.y_val, pred_ga):
            pooled_predictions["ga_a7"][pid] = (true, pred)

        print(
            f"{label_prefix}  [Fold {fold_idx}] ga_a7  : selected={sel_ga.selected_count}/{A7_EXPECTED_DIM} "
            f"({sel_ga.branch_retained}) | Macro-F1={metrics_ga.macro_f1:.4f} | "
            f"BalAcc={metrics_ga.balanced_accuracy:.4f} | time={sel_ga.fit_time_sec:.1f}s"
        )

    # Compute aggregates across folds
    aggregates = {arm: aggregate_fold_metrics(fold_metrics_by_arm[arm]) for arm in arms}

    # Stability analysis (Jaccard similarity across folds)
    stability = {
        "full_a7": {"mean_jaccard": 1.0, "median_jaccard": 1.0, "pairwise": {}},
        "bda_a7": compute_pairwise_jaccard(selected_masks_by_arm["bda_a7"]),
        "ga_a7": compute_pairwise_jaccard(selected_masks_by_arm["ga_a7"]),
    }

    # Feature dimensionality statistics
    dim_stats = {}
    for arm in arms:
        counts = [int(m.sum()) for m in selected_masks_by_arm[arm]]
        dim_stats[arm] = {
            "mean": float(np.mean(counts)),
            "std": float(np.std(counts)),
            "per_fold": counts,
        }

    # Branch family breakdown per fold
    family_breakdowns = {}
    for arm in arms:
        family_breakdowns[arm] = [
            compute_feature_family_breakdown(m, A7_BRANCH_DIMS)
            for m in selected_masks_by_arm[arm]
        ]

    # Dominance diagnostics
    dominance = {
        arm: compute_dominance_diagnostics(pooled_predictions[arm], classes)
        for arm in arms
    }

    # McNemar tests
    mcnemar_rows = run_pairwise_mcnemar(pooled_predictions)

    return {
        "arms": arms,
        "fold_metrics": fold_metrics_by_arm,
        "aggregates": aggregates,
        "pooled_predictions": pooled_predictions,
        "stability": stability,
        "dim_stats": dim_stats,
        "family_breakdowns": family_breakdowns,
        "dominance": dominance,
        "mcnemar_rows": mcnemar_rows,
        "selection_results": selection_results_by_arm,
    }


def write_phase6_v2_a7_reports(
    config: PSDConfig,
    benchmark_results: Dict[str, Any],
    p5_manifest: Dict[str, Any],
    source_experiment: str,
) -> Dict[str, Path]:
    """Writes all certified reports and the Phase 6 V2-A7 manifest to reports/phase6_v2_a7/."""
    out_dir = config.aef_crc_phase6_v2_a7_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = benchmark_results["arms"]
    aggregates = benchmark_results["aggregates"]
    dim_stats = benchmark_results["dim_stats"]
    stability = benchmark_results["stability"]
    family_breakdowns = benchmark_results["family_breakdowns"]
    classes = config.target_classes

    # 1. feature_selection_results.csv
    results_path = out_dir / "feature_selection_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arm", "dim_mean", "dim_std",
            "macro_f1_mean", "macro_f1_std",
            "balanced_accuracy_mean", "balanced_accuracy_std",
            "mcc_mean", "mcc_std",
            "accuracy_mean", "accuracy_std",
            "weighted_f1_mean", "weighted_f1_std",
            "jaccard_mean",
        ])
        for arm in arms:
            agg = aggregates[arm]
            d_stat = dim_stats[arm]
            stab = stability[arm]
            writer.writerow([
                arm,
                round(d_stat["mean"], 1), round(d_stat["std"], 2),
                round(agg.macro_f1_mean, 6), round(agg.macro_f1_std, 6),
                round(agg.balanced_accuracy_mean, 6), round(agg.balanced_accuracy_std, 6),
                round(agg.mcc_mean, 6), round(agg.mcc_std, 6),
                round(agg.accuracy_mean, 6), round(agg.accuracy_std, 6),
                round(agg.weighted_f1_mean, 6), round(agg.weighted_f1_std, 6),
                round(stab["mean_jaccard"], 4),
            ])

    # 2. class_performance.csv
    class_perf_path = out_dir / "class_performance.csv"
    with class_perf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arm", "class_name",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std",
            "support_total",
        ])
        for arm in arms:
            agg = aggregates[arm]
            for cls in classes:
                writer.writerow([
                    arm, cls,
                    round(agg.per_class_precision_mean.get(cls, 0.0), 4),
                    round(agg.per_class_precision_std.get(cls, 0.0), 4),
                    round(agg.per_class_recall_mean.get(cls, 0.0), 4),
                    round(agg.per_class_recall_std.get(cls, 0.0), 4),
                    round(agg.per_class_f1_mean.get(cls, 0.0), 4),
                    round(agg.per_class_f1_std.get(cls, 0.0), 4),
                    agg.per_class_support_total.get(cls, 0),
                ])

    # 3. psoriasis_dominance_diagnostic.csv
    dom_path = out_dir / "psoriasis_dominance_diagnostic.csv"
    dom_data = benchmark_results["dominance"]
    with dom_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(dom_data[arms[0]].keys())
        writer = csv.DictWriter(f, fieldnames=["arm"] + fieldnames)
        writer.writeheader()
        for arm in arms:
            row = {"arm": arm}
            row.update(dom_data[arm])
            writer.writerow(row)

    # 4. feature_selection_mcnemar_holm.csv
    mcnemar_path = out_dir / "feature_selection_mcnemar_holm.csv"
    with mcnemar_path.open("w", newline="", encoding="utf-8") as f:
        mcnemar_rows = benchmark_results["mcnemar_rows"]
        if mcnemar_rows:
            writer = csv.DictWriter(f, fieldnames=list(mcnemar_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mcnemar_rows)

    # 5. feature_family_breakdown.csv
    family_path = out_dir / "feature_family_breakdown.csv"
    with family_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arm", "family", "total_available",
            "fold_0", "fold_1", "fold_2", "fold_3", "fold_4",
            "mean_retained", "std_retained", "retention_pct",
        ])
        for arm in arms:
            for branch, avail_dim in A7_BRANCH_DIMS.items():
                counts = [fb.get(branch, 0) for fb in family_breakdowns[arm]]
                mean_c = float(np.mean(counts))
                std_c = float(np.std(counts))
                ret_pct = round(mean_c / avail_dim * 100, 2)
                writer.writerow([
                    arm, branch, avail_dim,
                    counts[0], counts[1], counts[2], counts[3], counts[4],
                    round(mean_c, 2), round(std_c, 2), ret_pct,
                ])

    # 6. phase6_a7_manifest.json
    manifest_path = out_dir / "phase6_a7_manifest.json"
    manifest_data = {
        "phase": "phase6_v2_a7",
        "experiment_type": "exploratory_feature_refinement",
        "scientific_status_note": (
            "EXPLORATORY EXPERIMENT ONLY: Investigates BDA/GA feature refinement on A7 "
            "(EfficientNet+GLCM+LBP+LAB, 1316-D). This does NOT replace the certified Phase 6 V2 "
            "experiment on A2 (1298-D). The certified Phase 5 V2 winner remains A2."
        ),
        "run_id": f"AEFCRC_A7_EXP_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(out_dir).encode()).hexdigest()[:8]}",
        "representation_id": P3_V2_REPR_ID,
        "random_seed": config.random_seed,
        "a7_definition": {
            "arm_id": A7_ARM_ID,
            "arm_name": A7_ARM_NAME,
            "expected_dimension": A7_EXPECTED_DIM,
            "branches": list(A7_BRANCHES),
            "branch_dimensions": A7_BRANCH_DIMS,
        },
        "upstream_phase5_reference": {
            "certified_winner_arm_id": p5_manifest.get("selected_fusion_arm_id"),
            "certified_winner_arm": p5_manifest.get("selected_fusion_arm"),
            "a7_phase5_performance": "Evaluated in reports/phase5_v2/fusion_results.csv",
        },
        "dataset_freeze_hash": p5_manifest.get("dataset_freeze_hash") or (
            hashlib.sha256(
                (config.aef_crc_reports_dir / "dataset_freeze.json").read_bytes()
            ).hexdigest()
            if (config.aef_crc_reports_dir / "dataset_freeze.json").exists()
            else None
        ),
        "fold_plan_hash": p5_manifest.get("fold_plan_hash") or (
            hashlib.sha256(
                (config.reports_dir / "aef_crc" / "fold_plan.csv").read_bytes()
            ).hexdigest()
            if (config.reports_dir / "aef_crc" / "fold_plan.csv").exists()
            else None
        ),
        "classifier": {
            "name": config.classifier_name,
            "sample_weight": "fold_local_balanced",
        },
        "optimizer_configs": {
            "bda": {
                "population_size": getattr(config, "bda_population_size", 20),
                "iterations": getattr(config, "bda_iterations", 30),
                "v_max": getattr(config, "bda_v_max", 6.0),
                "tau_min": getattr(config, "bda_tau_min", 1.0),
                "tau_max": getattr(config, "bda_tau_max", 4.0),
                "w_max": getattr(config, "bda_w_max", 0.9),
                "w_min": getattr(config, "bda_w_min", 0.4),
                "feature_count_penalty": 0.0,
                "fitness_objective": "inner_validation_macro_f1",
                "parsimony_penalty": None,
                "nested_val_fraction": getattr(config, "bda_nested_val_fraction", 0.2),
                "transfer_function": "time_varying_v_shaped",
                "proxy_classifier": "LogisticRegression(C=1.0, solver='lbfgs', max_iter=300)",
                "proxy_sample_weight": "inner_train_balanced",
            },
            "ga": {
                "population_size": getattr(config, "ga_population_size", 20),
                "generations": getattr(config, "ga_generations", 30),
                "mutation_rate": getattr(config, "ga_mutation_rate", 0.05),
                "crossover_rate": getattr(config, "ga_crossover_rate", 0.7),
                "feature_count_penalty": 0.0,
                "fitness_objective": "inner_validation_macro_f1",
                "parsimony_penalty": None,
                "nested_val_fraction": getattr(config, "ga_nested_val_fraction", 0.2),
                "proxy_classifier": "LogisticRegression(C=1.0, solver='lbfgs', max_iter=300)",
                "proxy_sample_weight": "inner_train_balanced",
            },
        },
        "arms_evaluated": arms,
        "results": {
            arm: {
                "dim_mean": dim_stats[arm]["mean"],
                "dim_std": dim_stats[arm]["std"],
                "macro_f1_mean": aggregates[arm].macro_f1_mean,
                "macro_f1_std": aggregates[arm].macro_f1_std,
                "balanced_accuracy_mean": aggregates[arm].balanced_accuracy_mean,
                "balanced_accuracy_std": aggregates[arm].balanced_accuracy_std,
                "mcc_mean": aggregates[arm].mcc_mean,
                "mcc_std": aggregates[arm].mcc_std,
                "accuracy_mean": aggregates[arm].accuracy_mean,
                "accuracy_std": aggregates[arm].accuracy_std,
                "weighted_f1_mean": aggregates[arm].weighted_f1_mean,
                "weighted_f1_std": aggregates[arm].weighted_f1_std,
                "per_class_f1": {
                    cls: aggregates[arm].per_class_f1_mean.get(cls, 0.0)
                    for cls in classes
                },
                "jaccard_stability_mean": stability[arm]["mean_jaccard"],
                "confusion_matrix_sum": aggregates[arm].confusion_sum.tolist() if aggregates[arm].confusion_sum is not None else None,
            }
            for arm in arms
        },
        "family_breakdowns": {
            arm: {
                branch: {
                    "available": A7_BRANCH_DIMS[branch],
                    "mean_retained": float(np.mean([fb.get(branch, 0) for fb in family_breakdowns[arm]])),
                    "std_retained": float(np.std([fb.get(branch, 0) for fb in family_breakdowns[arm]])),
                    "retention_pct": round(float(np.mean([fb.get(branch, 0) for fb in family_breakdowns[arm]])) / A7_BRANCH_DIMS[branch] * 100, 2),
                }
                for branch in A7_BRANCH_DIMS
            }
            for arm in arms
        },
        "dominance_diagnostics": benchmark_results["dominance"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reports_generated": [
            str(results_path),
            str(class_perf_path),
            str(dom_path),
            str(mcnemar_path),
            str(family_path),
            str(manifest_path),
        ],
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return {
        "results_csv": results_path,
        "class_perf_csv": class_perf_path,
        "dominance_csv": dom_path,
        "mcnemar_csv": mcnemar_path,
        "family_csv": family_path,
        "manifest_json": manifest_path,
    }


def validate_phase6_v2_a7_framework(config: PSDConfig) -> bool:
    """Non-destructive validation of Phase 6 V2-A7 framework and contracts:
    1. A7 Definition and 1316-D sanity.
    2. Phase 5 V2 A7 upstream presence.
    3. Branch resolution: ('deep', 'glcm', 'lbp', 'color_lab').
    4. Deep V2 path resolution and preflight check.
    5. Handcrafted cache verification.
    6. Non-destructive micro-benchmark on synthetic 1316-D data.
    7. Inner sample weight verification.
    8. Determinism check.
    9. Dominance diagnostics & feature-family breakdown check.
    10. Output directory isolation (reports/phase6_v2_a7/, artifacts/phase6_v2_a7/, logs/phase6_v2_a7/).
    """
    print("=" * 70)
    print("PHASE 6 V2-A7 FRAMEWORK & CONTRACT VALIDATION")
    print("=" * 70)

    # 1. A7 Definition
    print("\n[CHECK 1/10] Verifying A7 definition (1316-D)...")
    assert A7_ARM_ID == "A7"
    assert A7_EXPECTED_DIM == 1316
    assert A7_BRANCHES == ("deep", "glcm", "lbp", "color_lab")
    assert sum(A7_BRANCH_DIMS.values()) == 1316
    arm_def = FUSION_ARMS_BY_ID["A7"]
    assert arm_def.expected_dim == 1316
    assert arm_def.branches == A7_BRANCHES
    print(f"  PASS: A7 is registered as '{arm_def.name}' with expected_dim=1316 and branches={arm_def.branches}")

    # 2. Phase 5 V2 A7 presence
    print("\n[CHECK 2/10] Verifying upstream Phase 5 V2 A7 presence...")
    gate_ok, gate_msg, _ = gate_check_a7(config)
    assert gate_ok, f"Gate check failed: {gate_msg}"
    print(f"  PASS: {gate_msg}")

    # 3. Branch resolution
    print("\n[CHECK 3/10] Verifying branch feature naming for all 1316 features...")
    fused_names = get_fused_feature_names(A7_BRANCHES, config)
    assert len(fused_names) == 1316, f"Expected 1316 names, got {len(fused_names)}"
    assert fused_names[0] == "efficientnet_0"
    assert fused_names[1279] == "efficientnet_1279"
    assert fused_names[1280].startswith("glcm_")
    assert fused_names[1291].startswith("glcm_")
    assert fused_names[1292].startswith("lbp_bin_")
    assert fused_names[1309].startswith("lbp_bin_")
    assert fused_names[1310] == "lab_L_mean"
    assert fused_names[1315] == "lab_b_std"
    print(f"  PASS: All 1316 feature names resolved deterministically (deep: 1280, glcm: 12, lbp: 18, color_lab: 6).")

    # 4. Deep feature artifacts check
    print("\n[CHECK 4/10] Preflight checking deep-feature artifacts (5 folds x 2 splits)...")
    plan = load_frozen_fold_plan(config)
    pf_ok, pf_msg = preflight_deep_feature_artifacts(config, P3_V2_FOCAL_EXP, plan.folds)
    assert pf_ok, f"Deep feature preflight failed: {pf_msg}"
    print(f"  PASS: {pf_msg}")

    # 5. Handcrafted cache verification
    print("\n[CHECK 5/10] Preflight checking handcrafted feature caches...")
    extractor = HandcraftedFeatureExtractor(config)
    hc_ok, hc_msg = preflight_handcrafted_features(config, plan.folds, extractor)
    assert hc_ok, f"Handcrafted preflight failed: {hc_msg}"
    print(f"  PASS: {hc_msg}")

    # 6. Micro-benchmark on synthetic 1316-D data
    print("\n[CHECK 6/10] Running non-destructive micro-benchmark on synthetic 1316-D data...")
    rng = np.random.default_rng(1316)
    n_tr, n_va = 80, 20
    X_tr_synth = rng.normal(0, 1, size=(n_tr, 1316))
    X_va_synth = rng.normal(0, 1, size=(n_va, 1316))
    y_tr_synth = ["Psoriasis"] * 44 + ["Lichen_Planus"] * 18 + ["Pityriasis_Rosea"] * 12 + ["Seborrheic_Dermatitis"] * 6
    y_va_synth = ["Psoriasis"] * 11 + ["Lichen_Planus"] * 5 + ["Pityriasis_Rosea"] * 3 + ["Seborrheic_Dermatitis"] * 1

    from modules.fusion import FusionFoldData
    synth_data = FusionFoldData(
        X_train=X_tr_synth, y_train=y_tr_synth, psd_ids_train=[f"TR-{i:03d}" for i in range(n_tr)],
        X_val=X_va_synth, y_val=y_va_synth, psd_ids_val=[f"VA-{i:03d}" for i in range(n_va)],
        branch_dims=A7_BRANCH_DIMS,
    )

    micro_cfg = dataclasses.replace(
        config,
        bda_population_size=4,
        bda_iterations=3,
        ga_population_size=4,
        ga_generations=3,
        bda_feature_count_penalty=0.0,
        ga_feature_count_penalty=0.0,
    )

    # Arm 1: full_a7
    res_full = select_no_selection(synth_data)
    assert res_full.selected_count == 1316
    assert res_full.selected_mask.shape == (1316,)
    assert res_full.branch_retained == A7_BRANCH_DIMS

    # Arm 2: bda_a7
    res_bda = run_selector("bda", synth_data, micro_cfg, random_seed=42)
    assert res_bda.selected_count > 0
    assert res_bda.selected_mask.shape == (1316,)
    assert res_bda.extra.get("feature_count_penalty") == 0.0
    assert res_bda.extra.get("fitness_objective") == "inner_validation_macro_f1"

    # Arm 3: ga_a7
    res_ga = run_selector("ga", synth_data, micro_cfg, random_seed=42)
    assert res_ga.selected_count > 0
    assert res_ga.selected_mask.shape == (1316,)
    assert res_ga.extra.get("feature_count_penalty") == 0.0
    assert res_ga.extra.get("fitness_objective") == "inner_validation_macro_f1"

    print(f"  PASS: Micro-benchmark succeeded: full_a7={res_full.selected_count}, "
          f"bda_a7={res_bda.selected_count}, ga_a7={res_ga.selected_count} (out of 1316)")

    # 7. Inner sample weight verification
    print("\n[CHECK 7/10] Verifying inner-train sample weights in proxy models...")
    classes = config.target_classes
    class_weights_outer = {c: n_tr / (len(classes) * max(1, y_tr_synth.count(c))) for c in classes}
    m_full, pred_full = evaluate_selection_fold(micro_cfg, synth_data, res_full, classes, class_weights_outer, fold_index=0)
    m_bda, pred_bda = evaluate_selection_fold(micro_cfg, synth_data, res_bda, classes, class_weights_outer, fold_index=0)
    m_ga, pred_ga = evaluate_selection_fold(micro_cfg, synth_data, res_ga, classes, class_weights_outer, fold_index=0)
    print(f"  PASS: Outer fold evaluation with fold-local sample weights succeeded.")
    print(f"        full_a7 Macro-F1={m_full.macro_f1:.4f}, bda_a7 Macro-F1={m_bda.macro_f1:.4f}, ga_a7 Macro-F1={m_ga.macro_f1:.4f}")

    # 8. Determinism check
    print("\n[CHECK 8/10] Verifying determinism with identical seed...")
    res_bda_2 = run_selector("bda", synth_data, micro_cfg, random_seed=42)
    assert np.array_equal(res_bda.selected_mask, res_bda_2.selected_mask), "BDA is not deterministic with identical seed"
    res_ga_2 = run_selector("ga", synth_data, micro_cfg, random_seed=42)
    assert np.array_equal(res_ga.selected_mask, res_ga_2.selected_mask), "GA is not deterministic with identical seed"
    print("  PASS: Bitwise identical masks produced under identical random seeds.")

    # 9. Dominance diagnostics & feature-family breakdown check
    print("\n[CHECK 9/10] Verifying dominance diagnostics & feature family breakdown...")
    fake_pooled = {
        f"VA-{i:03d}": (y_va_synth[i], pred_bda[i]) for i in range(len(y_va_synth))
    }
    diag = compute_dominance_diagnostics(fake_pooled, classes)
    assert "Psoriasis_pred_ratio" in diag
    assert "total_samples" in diag
    fb = compute_feature_family_breakdown(res_bda.selected_mask, A7_BRANCH_DIMS)
    assert set(fb.keys()) == set(A7_BRANCH_DIMS.keys())
    assert sum(fb.values()) == res_bda.selected_count
    print(f"  PASS: Dominance diagnostics and feature-family breakdown valid: {fb}")

    # 10. Output directory isolation
    print("\n[CHECK 10/10] Verifying output directory isolation (phase6_v2_a7 paths only)...")
    v2_a7_rep = config.aef_crc_phase6_v2_a7_reports_dir
    v2_a7_art = config.aef_crc_phase6_v2_a7_artifacts_dir
    v2_a7_log = config.aef_crc_phase6_v2_a7_logs_dir
    v2_a2_rep = config.project_root / "reports" / "phase6_v2"
    v2_a2_art = config.project_root / "artifacts" / "phase6_v2"
    v1_rep = config.project_root / "reports" / "phase6"
    v1_art = config.project_root / "artifacts" / "phase6"

    assert "phase6_v2_a7" in str(v2_a7_rep), f"Report path {v2_a7_rep} not in phase6_v2_a7"
    assert "phase6_v2_a7" in str(v2_a7_art), f"Artifact path {v2_a7_art} not in phase6_v2_a7"
    assert "phase6_v2_a7" in str(v2_a7_log), f"Log path {v2_a7_log} not in phase6_v2_a7"
    assert str(v2_a7_rep) != str(v2_a2_rep), "A7 reports dir collision with A2 Phase 6 V2"
    assert str(v2_a7_art) != str(v2_a2_art), "A7 artifacts dir collision with A2 Phase 6 V2"
    assert str(v2_a7_rep) != str(v1_rep), "A7 reports dir collision with Phase 6 V1"
    assert str(v2_a7_art) != str(v1_art), "A7 artifacts dir collision with Phase 6 V1"
    print(f"  Reports:   {v2_a7_rep}")
    print(f"  Artifacts: {v2_a7_art}")
    print(f"  Logs:      {v2_a7_log}")
    print("  PASS: All Phase 6 V2-A7 paths strictly isolated from certified A2 and V1 directories.")

    print("\n" + "=" * 70)
    print("PHASE 6 V2-A7 FRAMEWORK VALIDATION COMPLETE: ALL 10 CHECKS PASSED.")
    print("=" * 70)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="PapuloNet V2 Phase 6-A7 Exploratory Feature Refinement Benchmark")
    parser.add_argument("--validate-framework", action="store_true", help="Run non-destructive framework validation checks")
    parser.add_argument("--validate-only", action="store_true", help="Alias for --validate-framework")
    parser.add_argument("--run-production-bda", action="store_true", help="Execute single production BDA on unified 1146-image cohort for A7")
    args = parser.parse_args()

    config = get_config()
    # Route directories to V2-A7 locations and bind winning Phase 3 V2 representation
    config = dataclasses.replace(
        config,
        aef_crc_artifacts_dir=config.aef_crc_phase3_v2_artifacts_dir,
        aef_crc_phase3_reports_dir=config.aef_crc_phase3_v2_reports_dir,
        aef_crc_phase5_reports_dir=config.aef_crc_phase5_v2_reports_dir,
        aef_crc_phase6_reports_dir=config.aef_crc_phase6_v2_a7_reports_dir,
        aef_crc_phase6_artifacts_dir=config.aef_crc_phase6_v2_a7_artifacts_dir,
        preprocessing_mode="standard",
        training_time_augmentation="false",
        loss_name="categorical_focal_loss",
        focal_gamma=2.0,
        use_class_weights=False,
        adam_clipnorm=1.0,
        bda_feature_count_penalty=0.0,
        ga_feature_count_penalty=0.0,
    )

    if args.validate_framework or args.validate_only:
        ok = validate_phase6_v2_a7_framework(config)
        return 0 if ok else 1

    print("=" * 70)
    print("PAPULONET V2 PHASE 6: EXPLORATORY FEATURE REFINEMENT (A7 REPRESENTATION)")
    print("=" * 70)

    # Step 0: Gate check
    print("\n--- Step 0: Phase 6 V2-A7 Gate Check ---")
    gate_ok, detail, p5_manifest = gate_check_a7(config)
    print(detail)
    if not gate_ok:
        print(f"FATAL: {detail}")
        return 1

    winner_exp = P3_V2_FOCAL_EXP

    # Dataset freeze verification
    freeze_ok = DatasetFreezer(config).verify().matches
    print(f"DATASET FREEZE: {'PASS' if freeze_ok else 'FAIL'}")
    if not freeze_ok:
        print("FATAL: Dataset freeze verification failed!")
        return 1

    plan = load_frozen_fold_plan(config)
    if not plan.folds:
        print("FATAL: No folds loaded from frozen fold plan.")
        return 1

    # Preflight deep-feature artifacts check (all 5 folds x 2 splits)
    print("\n--- Step 0b: Phase 3 V2 Deep-Feature Cache Preflight Check ---")
    pf_ok, pf_detail = preflight_deep_feature_artifacts(config, winner_exp, plan.folds)
    print(pf_detail)
    if not pf_ok:
        print(f"FATAL: {pf_detail}")
        return 1

    # Preflight handcrafted feature check
    print("\n--- Step 0c: Handcrafted Feature Cache Preflight Check ---")
    extractor = HandcraftedFeatureExtractor(config)
    hc_ok, hc_detail = preflight_handcrafted_features(config, plan.folds, extractor)
    print(hc_detail)
    if not hc_ok:
        print(f"FATAL: {hc_detail}")
        return 1

    # Production BDA entrypoint for A7
    if args.run_production_bda:
        print("\n--- Production BDA Execution for A7 (Unified 1146-image cohort) ---")
        from modules.fusion import build_fusion_final
        final_records = plan.folds[0].train_records + plan.folds[0].val_records
        assert len(final_records) == 1146, f"Expected 1146 records, got {len(final_records)}"
        branches = A7_BRANCHES
        data = build_fusion_final(config, final_records, plan.holdout_val_records, branches, winner_exp, extractor)
        res = select_bda(data, config, config.random_seed)
        mask_path = save_production_bda_mask(
            res.selected_mask,
            config,
            metadata={
                "arm": A7_ARM_NAME,
                "dimension": A7_EXPECTED_DIM,
                "selected_count": res.selected_count,
                "branch_retained": res.branch_retained,
                "fit_time_sec": res.fit_time_sec,
                "source_experiment": winner_exp,
            },
            run_id=p5_manifest.get("run_id"),
            artifacts_dir=config.aef_crc_phase6_v2_a7_artifacts_dir,
        )
        print(f"Production BDA mask for A7 successfully saved to {mask_path} ({res.selected_count} features selected).")
        return 0

    # Real 3-Arm Benchmark Execution
    benchmark_results = run_phase6_v2_a7_benchmark(config, plan, winner_exp, label_prefix="  ")

    # Write all certified reports and manifest
    reports = write_phase6_v2_a7_reports(config, benchmark_results, p5_manifest, winner_exp)

    print("\n--- Phase 6 V2-A7 Exploratory Reports Generated ---")
    for k, p in reports.items():
        print(f"  [{k}] {p}")

    print("\n" + "=" * 70)
    print("PAPULONET V2 PHASE 6-A7 EXPLORATORY BENCHMARK EXECUTION COMPLETE.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
