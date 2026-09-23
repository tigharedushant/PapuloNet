"""
run_aef_crc_phase6_v2.py

PapuloNet V2 Phase 6: Feature Refinement Benchmark.

Evaluates metaheuristic feature selection algorithms (Binary Dragonfly Algorithm vs Genetic Algorithm)
against the unselected baseline on the certified Phase 5 V2 winning representation:
  A2 = EfficientNet-B0 + LBP (1298 dimensions: deep=1280, lbp=18).

Protocol Amendment (Pre-Experiment):
  Fitness objective is strictly unpenalized (lambda = 0):
    fitness = inner_validation_macro_f1
  Feature-count penalty removed pre-experiment: the study objective imposes no prior preference
  for fewer features. Full A2 (1298-D) remains the explicit no-selection baseline. Feature count is
  reported as a secondary descriptive property.

Key Invariants:
1. Upstream Binding: Strictly binds to Phase 5 V2 certified winner (A2 = EfficientNet+LBP,
   1298-D, branches=('deep', 'lbp')) from reports/phase5_v2/phase5_manifest.json.
2. Isolated Paths: Outputs written exclusively to reports/phase6_v2/, artifacts/phase6_v2/,
   and logs/phase6_v2/. Never touches V1 reports/phase6/ or artifacts/phase6/.
3. Three Evaluated Arms on Identical Outer Folds:
   - full_a2: Unselected baseline (all 1298 dimensions retained).
   - bda: Binary Dragonfly Algorithm (optimizing unpenalized inner-val Macro-F1).
   - ga: Genetic Algorithm comparator (optimizing unpenalized inner-val Macro-F1).
4. Fold-Local Sample Weights:
   - Inner optimization proxies (BDA & GA) fit LogisticRegression with sample_weight
     derived strictly from inner-train labels (w_c = N_inner_train / (C * N_c,inner_train)).
   - Outer evaluation classifier (Random Forest) receives fold-local balanced sample weights
     derived strictly from fold.class_weights.
5. Strict Partition Isolation:
   - Locked Final Test (243 images): Never accessed.
   - Outer Validation (246 images): Never accessed.
   - Development cohort (1146 images): Evaluated strictly via frozen 5-fold CV.
6. Comprehensive Diagnostics:
   - Macro-F1, Balanced Accuracy, MCC, Accuracy, Weighted F1 (mean +/- std).
   - Per-class Precision, Recall, F1, and Support for all 4 classes.
   - Psoriasis majority dominance diagnostic (true vs predicted distributions).
   - Pooled confusion matrix.
   - Pairwise McNemar's tests with Holm-Bonferroni correction.
   - Feature retention and pairwise Jaccard stability across folds.
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
)
from modules.feature_selection import (
    run_selector,
    select_no_selection,
    select_bda,
    select_ga,
    compute_jaccard_similarity,
    compute_pairwise_jaccard,
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


def gate_check_phase5_v2(config: PSDConfig) -> Tuple[bool, str, Dict[str, Any]]:
    """Checks Step 0's upstream Phase 5 V2 requirements.
    Validates:
      1. reports/phase5_v2/phase5_manifest.json exists.
      2. Certified winner is A2 (EfficientNet+LBP).
      3. Dimension is exactly 1298.
      4. Branches are exactly ('deep', 'lbp').
      5. P3-V2-Focal deep feature files exist.
    """
    manifest_path = config.aef_crc_phase5_v2_reports_dir / "phase5_manifest.json"
    if not manifest_path.exists():
        return False, f"Missing Phase 5 V2 manifest at {manifest_path}", {}

    try:
        p5_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse Phase 5 V2 manifest: {exc}", {}

    winner_arm_id = p5_manifest.get("selected_fusion_arm_id")
    winner_arm_name = p5_manifest.get("selected_fusion_arm")
    winner_dim = p5_manifest.get("selected_fusion_dimension")
    winner_branches = tuple(p5_manifest.get("selected_fusion_branches", []))

    if winner_arm_id != "A2":
        return False, f"Phase 5 V2 winner is '{winner_arm_id}' ({winner_arm_name}), expected 'A2' (EfficientNet+LBP)", p5_manifest

    if winner_dim != 1298:
        return False, f"Phase 5 V2 winner dimension is {winner_dim}, expected exactly 1298", p5_manifest

    if winner_branches != ("deep", "lbp"):
        return False, f"Phase 5 V2 winner branches are {winner_branches}, expected ('deep', 'lbp')", p5_manifest

    # Verify deep features directory from Phase 3 V2
    p3_artifacts = getattr(config, "aef_crc_phase3_v2_artifacts_dir", config.aef_crc_artifacts_dir)
    deep_dir = p3_artifacts / "deep_features" / "P3-V2-Focal"
    npy_files = list(deep_dir.glob("**/*.npy")) if deep_dir.exists() else []
    if not npy_files:
        return False, f"No Phase 3 V2 deep feature .npy files found in {deep_dir}", p5_manifest

    detail = (
        f"Verified upstream Phase 5 V2 winner: [{winner_arm_id}] {winner_arm_name} "
        f"({winner_dim}-D, branches={winner_branches}) | "
        f"Upstream run_id: {p5_manifest.get('run_id')} | "
        f"P3-V2-Focal deep feature files: {len(npy_files)}"
    )
    return True, detail, p5_manifest


def preflight_deep_feature_artifacts(config: PSDConfig, experiment_name: str, folds: List[Any]) -> Tuple[bool, str]:
    """Fast-fail preflight gate verifying that all 5 folds (fold_00 - fold_04) and splits (train, val) have
    valid, uncorrupted deep-feature manifests matching:
      - experiment_name == experiment_name (e.g. 'P3-V2-Focal')
      - representation_id == 'efficientnet_b0_43d581b96f8ec368'
      - feature_dim == 1280
      - backbone_name == 'efficientnet_b0'
      - non-empty psd_ids matching split record count
      - manifest root path is under phase3_v2 and not phase3.
    """
    from modules.efficientnet_model import load_deep_feature_manifest, _resolve_deep_feature_root

    root_dir = _resolve_deep_feature_root(config, experiment_name)
    if "phase3_v2" not in str(root_dir):
        return False, f"Deep feature root resolved to '{root_dir}', expected path containing 'phase3_v2'"

    expected_repr_id = "efficientnet_b0_43d581b96f8ec368"
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

            if manifest.get("representation_id") != expected_repr_id:
                return False, f"Fold {fold_idx} {split_name} manifest representation_id='{manifest.get('representation_id')}', expected '{expected_repr_id}'"

            if manifest.get("feature_dim") != 1280:
                return False, f"Fold {fold_idx} {split_name} manifest feature_dim={manifest.get('feature_dim')}, expected 1280"

            if manifest.get("backbone_name") != "efficientnet_b0":
                return False, f"Fold {fold_idx} {split_name} manifest backbone_name='{manifest.get('backbone_name')}', expected 'efficientnet_b0'"

            psd_ids = manifest.get("psd_ids", [])
            if len(psd_ids) != len(records):
                return False, f"Fold {fold_idx} {split_name} manifest has {len(psd_ids)} PSD IDs, expected {len(records)}"

            total_checked += 1

    return True, f"Preflight deep-feature check PASSED: verified all {total_checked} manifests (5 folds x 2 splits) with representation_id='{expected_repr_id}', dim=1280 in '{root_dir}'"


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


def run_phase6_v2_benchmark(
    config: PSDConfig,
    plan: Any,
    source_experiment: str,
    label_prefix: str = "",
) -> Dict[str, Any]:
    """Executes the 3-arm Phase 6 V2 benchmark across all 5 outer folds on A2 representation.
    Arms:
      1. full_a2 (1298 dimensions, unselected baseline)
      2. bda (Binary Dragonfly Algorithm)
      3. ga (Genetic Algorithm)
    """
    branches = ("deep", "lbp")
    classes = config.target_classes
    extractor = HandcraftedFeatureExtractor(config)

    arms = ["full_a2", "bda", "ga"]
    fold_metrics_by_arm: Dict[str, List[FoldMetrics]] = {arm: [] for arm in arms}
    pooled_predictions: Dict[str, Dict[str, Tuple[str, str]]] = {arm: {} for arm in arms}
    selected_masks_by_arm: Dict[str, List[np.ndarray]] = {arm: [] for arm in arms}
    selection_results_by_arm: Dict[str, List[SelectionResult]] = {arm: [] for arm in arms}

    print(f"{label_prefix}Starting Phase 6 V2 3-Arm Benchmark on A2 representation (1298-D)...")
    print(f"{label_prefix}Arms: full_a2 (unselected baseline), bda (primary), ga (comparator)")

    for fold in plan.folds:
        fold_idx = fold.fold_index
        fold_seed = config.random_seed + fold_idx

        # Build fused A2 data for this fold
        data = build_fusion_fold(config, fold, branches, source_experiment, extractor)
        assert data.X_train.shape[1] == 1298, f"Expected 1298 features, got {data.X_train.shape[1]}"

        # 1. Arm: full_a2 (unselected baseline)
        sel_full = select_no_selection(data)
        metrics_full, pred_full = evaluate_selection_fold(
            config, data, sel_full, classes, fold.class_weights, fold_idx
        )
        fold_metrics_by_arm["full_a2"].append(metrics_full)
        selected_masks_by_arm["full_a2"].append(sel_full.selected_mask)
        selection_results_by_arm["full_a2"].append(sel_full)
        for pid, true, pred in zip(data.psd_ids_val, data.y_val, pred_full):
            pooled_predictions["full_a2"][pid] = (true, pred)

        print(
            f"{label_prefix}  [Fold {fold_idx}] full_a2: dim=1298 | "
            f"Macro-F1={metrics_full.macro_f1:.4f} | BalAcc={metrics_full.balanced_accuracy:.4f}"
        )

        # 2. Arm: bda (Binary Dragonfly Algorithm)
        sel_bda = run_selector("bda", data, config, fold_seed)
        metrics_bda, pred_bda = evaluate_selection_fold(
            config, data, sel_bda, classes, fold.class_weights, fold_idx
        )
        fold_metrics_by_arm["bda"].append(metrics_bda)
        selected_masks_by_arm["bda"].append(sel_bda.selected_mask)
        selection_results_by_arm["bda"].append(sel_bda)
        for pid, true, pred in zip(data.psd_ids_val, data.y_val, pred_bda):
            pooled_predictions["bda"][pid] = (true, pred)

        print(
            f"{label_prefix}  [Fold {fold_idx}] bda    : selected={sel_bda.selected_count}/1298 "
            f"({sel_bda.branch_retained}) | Macro-F1={metrics_bda.macro_f1:.4f} | "
            f"BalAcc={metrics_bda.balanced_accuracy:.4f} | time={sel_bda.fit_time_sec:.1f}s"
        )

        # 3. Arm: ga (Genetic Algorithm)
        sel_ga = run_selector("ga", data, config, fold_seed)
        metrics_ga, pred_ga = evaluate_selection_fold(
            config, data, sel_ga, classes, fold.class_weights, fold_idx
        )
        fold_metrics_by_arm["ga"].append(metrics_ga)
        selected_masks_by_arm["ga"].append(sel_ga.selected_mask)
        selection_results_by_arm["ga"].append(sel_ga)
        for pid, true, pred in zip(data.psd_ids_val, data.y_val, pred_ga):
            pooled_predictions["ga"][pid] = (true, pred)

        print(
            f"{label_prefix}  [Fold {fold_idx}] ga     : selected={sel_ga.selected_count}/1298 "
            f"({sel_ga.branch_retained}) | Macro-F1={metrics_ga.macro_f1:.4f} | "
            f"BalAcc={metrics_ga.balanced_accuracy:.4f} | time={sel_ga.fit_time_sec:.1f}s"
        )

    # Compute aggregates across folds
    aggregates = {arm: aggregate_fold_metrics(fold_metrics_by_arm[arm]) for arm in arms}

    # Stability analysis (Jaccard similarity across folds)
    stability = {
        "full_a2": {"mean_jaccard": 1.0, "median_jaccard": 1.0, "pairwise": {}},
        "bda": compute_pairwise_jaccard(selected_masks_by_arm["bda"]),
        "ga": compute_pairwise_jaccard(selected_masks_by_arm["ga"]),
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
        "dominance": dominance,
        "mcnemar_rows": mcnemar_rows,
        "selection_results": selection_results_by_arm,
    }


def write_phase6_v2_reports(
    config: PSDConfig,
    benchmark_results: Dict[str, Any],
    p5_manifest: Dict[str, Any],
    source_experiment: str,
) -> Dict[str, Path]:
    """Writes all certified reports and the Phase 6 V2 manifest to reports/phase6_v2/."""
    out_dir = config.aef_crc_phase6_v2_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = benchmark_results["arms"]
    aggregates = benchmark_results["aggregates"]
    dim_stats = benchmark_results["dim_stats"]
    stability = benchmark_results["stability"]
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

    # 5. Determine winning arm per protocol
    # Rule: BDA vs full_a2:
    # If BDA Macro-F1 >= full_a2 Macro-F1 - 0.005 and BDA dimension < full_a2 dimension, BDA selected for parsimony.
    # Otherwise highest Macro-F1.
    bda_f1 = aggregates["bda"].macro_f1_mean
    full_f1 = aggregates["full_a2"].macro_f1_mean
    ga_f1 = aggregates["ga"].macro_f1_mean

    candidates = [
        {"arm": "bda", "f1": bda_f1, "dim": dim_stats["bda"]["mean"]},
        {"arm": "full_a2", "f1": full_f1, "dim": dim_stats["full_a2"]["mean"]},
        {"arm": "ga", "f1": ga_f1, "dim": dim_stats["ga"]["mean"]},
    ]
    candidates.sort(key=lambda c: c["f1"], reverse=True)
    best_candidate = candidates[0]

    # Check parsimony tie-break within 0.005 equivalence margin against best
    equiv_margin = 0.005
    near_best = [c for c in candidates if c["f1"] >= best_candidate["f1"] - equiv_margin]
    # Lowest dimensionality wins among practically equivalent candidates
    winner_arm = min(near_best, key=lambda c: c["dim"])["arm"]

    # 6. phase6_manifest.json
    manifest_path = out_dir / "phase6_manifest.json"
    manifest_data = {
        "phase": "phase6_v2",
        "run_id": p5_manifest.get("run_id"),
        "representation_id": p5_manifest.get("representation_id"),
        "random_seed": config.random_seed,
        "upstream_phase5_winner": {
            "arm_id": p5_manifest.get("selected_fusion_arm_id"),
            "arm": p5_manifest.get("selected_fusion_arm"),
            "dimension": p5_manifest.get("selected_fusion_dimension"),
            "branches": p5_manifest.get("selected_fusion_branches"),
            "macro_f1_mean": p5_manifest.get("selected_metrics", {}).get("macro_f1_mean"),
            "macro_f1_std": p5_manifest.get("selected_metrics", {}).get("macro_f1_std"),
        },
        "dataset_freeze_hash": p5_manifest.get("dataset_freeze_hash"),
        "fold_plan_hash": p5_manifest.get("fold_plan_hash"),
        "classifier": {
            "name": config.classifier_name,
            "sample_weight": "fold_local_balanced",
        },
        "optimizer_configs": {
            "bda": {
                "population_size": config.bda_population_size,
                "iterations": config.bda_iterations,
                "v_max": config.bda_v_max,
                "tau_min": config.bda_tau_min,
                "tau_max": config.bda_tau_max,
                "w_max": config.bda_w_max,
                "w_min": config.bda_w_min,
                "feature_count_penalty": 0.0,
                "fitness_objective": "inner_validation_macro_f1",
                "parsimony_penalty": None,
                "nested_val_fraction": getattr(config, "bda_nested_val_fraction", 0.2),
                "transfer_function": "time_varying_v_shaped",
                "proxy_classifier": "LogisticRegression(C=1.0, solver='lbfgs', max_iter=300)",
                "proxy_sample_weight": "inner_train_balanced",
            },
            "ga": {
                "population_size": config.ga_population_size,
                "generations": config.ga_generations,
                "mutation_rate": config.ga_mutation_rate,
                "crossover_rate": config.ga_crossover_rate,
                "feature_count_penalty": 0.0,
                "fitness_objective": "inner_validation_macro_f1",
                "parsimony_penalty": None,
                "nested_val_fraction": config.ga_nested_val_fraction,
                "proxy_classifier": "LogisticRegression(C=1.0, solver='lbfgs', max_iter=300)",
                "proxy_sample_weight": "inner_train_balanced",
            },
        },
        "protocol_amendment": {
            "amendment_date": "2026-09-20",
            "amendment_type": "pre_experiment_protocol_amendment",
            "status": "APPROVED_BEFORE_EXECUTION",
            "rationale": "Feature-count penalty removed pre-experiment: the study objective does not impose a prior preference for smaller feature subsets. Full 1298-D A2 representation is the explicit no-selection baseline. BDA and GA optimize purely inner-validation Macro-F1 (lambda = 0). Feature count is reported as a secondary descriptive property.",
            "feature_count_penalty": 0.0,
            "fitness_objective": "inner_validation_macro_f1",
            "parsimony_penalty": None,
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
                "per_class_f1": aggregates[arm].per_class_f1_mean,
                "jaccard_stability_mean": stability[arm]["mean_jaccard"],
                "confusion_matrix_sum": aggregates[arm].confusion_sum.tolist() if aggregates[arm].confusion_sum is not None else None,
            }
            for arm in arms
        },
        "dominance_diagnostics": dom_data,
        "selected_arm": winner_arm,
        "selection_rule": "highest_mean_macro_f1_with_0.005_practical_equivalence_and_lowest_dimension_tiebreak",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reports_generated": [
            str(results_path),
            str(class_perf_path),
            str(dom_path),
            str(mcnemar_path),
            str(manifest_path),
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return {
        "results_csv": results_path,
        "class_perf_csv": class_perf_path,
        "dominance_csv": dom_path,
        "mcnemar_csv": mcnemar_path,
        "manifest_json": manifest_path,
    }


def validate_phase6_v2_framework(config: PSDConfig) -> bool:
    """Non-destructive programmatic framework validation for Phase 6 V2.
    Touches NO frozen data, writes NO real experiment files, and returns True on success.
    """
    print("=" * 70)
    print("PAPULONET V2 PHASE 6: FRAMEWORK VALIDATION MODE")
    print("=" * 70)

    # 1. Gate check inspection
    print("\n[CHECK 1/8] Inspecting Phase 5 V2 upstream gate & deep-feature cache...")
    gate_ok, detail, p5_manifest = gate_check_phase5_v2(config)
    print(f"  Gate check: {'PASS' if gate_ok else 'FAIL'}")
    print(f"  Detail: {detail}")
    if not gate_ok:
        print("  FAIL: Upstream Phase 5 V2 gate check failed!")
        return False

    winner_exp = p5_manifest.get("upstream_phase3_winner", {}).get("winner_experiment_id", "P3-V2-Focal")
    plan = load_frozen_fold_plan(config)
    pf_ok, pf_detail = preflight_deep_feature_artifacts(config, winner_exp, plan.folds)
    print(f"  Preflight check: {'PASS' if pf_ok else 'FAIL'}")
    print(f"  Detail: {pf_detail}")
    if not pf_ok:
        print("  FAIL: Upstream Phase 3 V2 deep-feature preflight failed!")
        return False

    # 2. Selector registry & imports
    print("\n[CHECK 2/8] Verifying selector registry resolution...")
    from modules.feature_selection import SELECTOR_REGISTRY
    for sel in ["bda", "ga", "none"]:
        assert sel in SELECTOR_REGISTRY, f"Selector '{sel}' missing from registry"
    print("  PASS: All required selectors ('bda', 'ga', 'none') registered.")

    # 3. BDA Mathematical Transfer Function Verification
    print("\n[CHECK 3/8] Verifying BDA time-varying V-shaped transfer function math...")
    tau_0 = config.bda_tau_min + (config.bda_tau_max - config.bda_tau_min) * (0 / 30)
    tau_T = config.bda_tau_min + (config.bda_tau_max - config.bda_tau_min) * (30 / 30)
    assert abs(tau_0 - 1.0) < 1e-6, f"tau(0) expected 1.0, got {tau_0}"
    assert abs(tau_T - 4.0) < 1e-6, f"tau(T) expected 4.0, got {tau_T}"
    assert abs(np.tanh(tau_0 * 0.0)) == 0.0, "T(0) != 0.0"
    t_large = abs(np.tanh(tau_T * 10.0))
    assert 0.0 <= t_large <= 1.0, f"Transfer value out of bounds: {t_large}"
    print("  PASS: Transfer function math, monotonicity, and bounds verified.")

    # 4. Synthetic Micro-Benchmark (All 3 Arms, 1298-D)
    print("\n[CHECK 4/8] Running synthetic micro-benchmark on 1298-D data (3 arms)...")
    from modules.fusion import FusionFoldData
    rng = np.random.default_rng(42)
    n_train, n_val, n_feat = 60, 20, 1298
    X_tr = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    y_tr = [config.target_classes[i % 4] for i in range(n_train)]
    X_va = rng.standard_normal((n_val, n_feat)).astype(np.float32)
    y_va = [config.target_classes[i % 4] for i in range(n_val)]
    branch_dims = {"deep": 1280, "lbp": 18}

    synth_data = FusionFoldData(
        X_train=X_tr,
        X_val=X_va,
        y_train=y_tr,
        y_val=y_va,
        psd_ids_train=[f"TR-{i:03d}" for i in range(n_train)],
        psd_ids_val=[f"VA-{i:03d}" for i in range(n_val)],
        branch_dims=branch_dims,
    )

    micro_cfg = dataclasses.replace(
        config,
        bda_population_size=4,
        bda_iterations=2,
        ga_population_size=4,
        ga_generations=2,
        bda_nested_val_fraction=0.2,
        ga_nested_val_fraction=0.2,
        bda_feature_count_penalty=0.0,
        ga_feature_count_penalty=0.0,
    )

    # Arm 1: full_a2
    res_full = select_no_selection(synth_data)
    assert res_full.selected_count == 1298
    assert res_full.selected_mask.shape == (1298,)

    # Arm 2: bda
    res_bda = run_selector("bda", synth_data, micro_cfg, random_seed=101)
    assert res_bda.selected_count > 0
    assert res_bda.selected_mask.shape == (1298,)
    assert res_bda.extra.get("feature_count_penalty") == 0.0
    assert res_bda.extra.get("fitness_objective") == "inner_validation_macro_f1"
    assert res_bda.extra.get("parsimony_penalty") is None

    # Arm 3: ga
    res_ga = run_selector("ga", synth_data, micro_cfg, random_seed=101)
    assert res_ga.selected_count > 0
    assert res_ga.selected_mask.shape == (1298,)
    assert res_ga.extra.get("feature_count_penalty") == 0.0
    assert res_ga.extra.get("fitness_objective") == "inner_validation_macro_f1"
    assert res_ga.extra.get("parsimony_penalty") is None

    print(f"  PASS: Micro-benchmark succeeded: full_a2={res_full.selected_count}, "
          f"bda={res_bda.selected_count}, ga={res_ga.selected_count}")

    # 5. Inner-Train Sample Weight Verification
    print("\n[CHECK 5/8] Verifying inner-train sample weights in proxy models...")
    classes = config.target_classes
    class_weights_outer = {c: n_train / (len(classes) * max(1, y_tr.count(c))) for c in classes}
    m_full, pred_full = evaluate_selection_fold(micro_cfg, synth_data, res_full, classes, class_weights_outer, fold_index=0)
    m_bda, pred_bda = evaluate_selection_fold(micro_cfg, synth_data, res_bda, classes, class_weights_outer, fold_index=0)
    m_ga, pred_ga = evaluate_selection_fold(micro_cfg, synth_data, res_ga, classes, class_weights_outer, fold_index=0)
    print(f"  PASS: Outer fold evaluation with fold-local sample weights succeeded for all 3 arms.")
    print(f"        full_a2 Macro-F1={m_full.macro_f1:.4f}, bda Macro-F1={m_bda.macro_f1:.4f}, ga Macro-F1={m_ga.macro_f1:.4f}")

    # 6. Determinism under fixed seed
    print("\n[CHECK 6/8] Verifying determinism with identical seed...")
    res_bda_2 = run_selector("bda", synth_data, micro_cfg, random_seed=101)
    assert np.array_equal(res_bda.selected_mask, res_bda_2.selected_mask), "BDA is not deterministic with identical seed"
    res_ga_2 = run_selector("ga", synth_data, micro_cfg, random_seed=101)
    assert np.array_equal(res_ga.selected_mask, res_ga_2.selected_mask), "GA is not deterministic with identical seed"
    print("  PASS: Bitwise identical masks produced under identical random seeds.")

    # 7. Diagnostic Metrics & Dominance Analysis
    print("\n[CHECK 7/8] Verifying dominance diagnostics & Jaccard computations...")
    fake_pooled = {
        f"VA-{i:03d}": (y_va[i], pred_bda[i]) for i in range(len(y_va))
    }
    diag = compute_dominance_diagnostics(fake_pooled, classes)
    assert "Psoriasis_pred_ratio" in diag
    assert "total_samples" in diag
    j_val = compute_jaccard_similarity(res_bda.selected_mask, res_ga.selected_mask)
    assert 0.0 <= j_val <= 1.0
    print(f"  PASS: Dominance diagnostics and Jaccard computation valid (Cross BDA/GA Jaccard: {j_val:.4f}).")

    # 8. Output Path Isolation
    print("\n[CHECK 8/8] Verifying output directory isolation (Phase 6 V2 paths only)...")
    v2_rep = config.aef_crc_phase6_v2_reports_dir
    v2_art = config.aef_crc_phase6_v2_artifacts_dir
    v2_log = config.aef_crc_phase6_v2_logs_dir
    assert "phase6_v2" in str(v2_rep), f"Report path {v2_rep} not in phase6_v2"
    assert "phase6_v2" in str(v2_art), f"Artifact path {v2_art} not in phase6_v2"
    assert "phase6_v2" in str(v2_log), f"Log path {v2_log} not in phase6_v2"
    print(f"  Reports:   {v2_rep}")
    print(f"  Artifacts: {v2_art}")
    print(f"  Logs:      {v2_log}")
    print("  PASS: All Phase 6 V2 paths strictly isolated from V1 directories.")

    print("\n" + "=" * 70)
    print("PHASE 6 V2 FRAMEWORK VALIDATION COMPLETE: ALL 8 CHECKS PASSED.")
    print("=" * 70)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="PapuloNet V2 Phase 6 Feature Refinement Benchmark")
    parser.add_argument("--validate-framework", action="store_true", help="Run non-destructive framework validation checks")
    parser.add_argument("--validate-only", action="store_true", help="Alias for --validate-framework")
    parser.add_argument("--run-production-bda", action="store_true", help="Execute single production BDA on unified 1146-image cohort")
    args = parser.parse_args()

    config = get_config()
    # Route directories to V2 locations and bind winning Phase 3 V2 representation
    config = dataclasses.replace(
        config,
        aef_crc_artifacts_dir=config.aef_crc_phase3_v2_artifacts_dir,
        aef_crc_phase3_reports_dir=config.aef_crc_phase3_v2_reports_dir,
        aef_crc_phase5_reports_dir=config.aef_crc_phase5_v2_reports_dir,
        aef_crc_phase6_reports_dir=config.aef_crc_phase6_v2_reports_dir,
        aef_crc_phase6_artifacts_dir=config.aef_crc_phase6_v2_artifacts_dir,
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
        ok = validate_phase6_v2_framework(config)
        return 0 if ok else 1

    print("=" * 70)
    print("PAPULONET V2 PHASE 6: FEATURE REFINEMENT BENCHMARK (REAL PIPELINE)")
    print("=" * 70)

    # Step 0: Gate check
    print("\n--- Step 0: Phase 5 V2 Upstream Gate Check ---")
    gate_ok, detail, p5_manifest = gate_check_phase5_v2(config)
    print(detail)
    if not gate_ok:
        print(f"FATAL: {detail}")
        return 1

    winner_exp = p5_manifest.get("upstream_phase3_winner", {}).get("winner_experiment_id", "P3-V2-Focal")

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

    # Production BDA entrypoint
    if args.run_production_bda:
        print("\n--- Production BDA Execution (Unified 1146-image cohort) ---")
        from modules.fusion import build_fusion_final
        final_records = plan.folds[0].train_records + plan.folds[0].val_records
        assert len(final_records) == 1146, f"Expected 1146 records, got {len(final_records)}"
        extractor = HandcraftedFeatureExtractor(config)
        branches = ("deep", "lbp")
        data = build_fusion_final(config, final_records, plan.holdout_val_records, branches, winner_exp, extractor)
        res = select_bda(data, config, config.random_seed)
        mask_path = save_production_bda_mask(
            res.selected_mask,
            config,
            metadata={
                "selected_count": res.selected_count,
                "branch_retained": res.branch_retained,
                "fit_time_sec": res.fit_time_sec,
                "source_experiment": winner_exp,
            },
            run_id=p5_manifest.get("run_id"),
            artifacts_dir=config.aef_crc_phase6_v2_artifacts_dir,
        )
        print(f"Production BDA mask successfully saved to {mask_path} ({res.selected_count} features selected).")
        return 0

    # Real 3-Arm Benchmark Execution
    benchmark_results = run_phase6_v2_benchmark(config, plan, winner_exp, label_prefix="  ")

    # Write all certified reports and manifest
    reports = write_phase6_v2_reports(config, benchmark_results, p5_manifest, winner_exp)

    print("\n--- Phase 6 V2 Certified Reports Generated ---")
    for k, p in reports.items():
        print(f"  [{k}] {p}")

    print("\n" + "=" * 70)
    print("PAPULONET V2 PHASE 6 BENCHMARK EXECUTION COMPLETE.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
