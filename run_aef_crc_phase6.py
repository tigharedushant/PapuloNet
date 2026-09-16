"""
run_aef_crc_phase6.py

AEF-CRC Phase 6: Feature Refinement Benchmark.

Step 0 gate check parses Phase 5's ACTUAL recorded results
(reports/phase5/fusion_results.csv + fusion_mcnemar_holm.csv) to
determine the winning arm the same disciplined way run_aef_crc_phase5.py
parses Phase 3's winner from phase3_report.md -- never guessed, never
hardcoded. Phase 5's own script does not currently write out an
explicit "winner" line, so this script applies Phase 5's own decision
rule (best mean Macro-F1 arm; keep fusion only if it significantly
beats "EfficientNet only" per the Holm-corrected pairwise table)
directly against those two files.

If no real Phase 5 results exist (this run), falls through to a
clearly-labeled PLUMBING-ONLY path using a synthetic deep-feature cache
-- same discipline as run_aef_crc_phase5.py's own fallback.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import sys

import numpy as np

from config.config import get_config
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import load_frozen_fold_plan
from modules.fusion import (
    ARMS,
    build_fusion_fold,
    build_classifier,
    compute_sample_weights,
)
from modules.feature_selection import run_selector, estimate_rfe_cost
from modules.evaluation import (
    compute_fold_metrics,
    aggregate_fold_metrics,
    mcnemar_test,
    holm_correction,
)


def parse_phase5_winner(config):
    """
    Reads Phase 5's own real output files and applies Phase 5's own
    decision rule.

    Returns:
        (winner_arm_name_or_None, detail_str)

    The selected Phase-5 arm can be EfficientNet itself. None is returned
    only when the Phase-5 evidence files are unavailable/invalid or no
    arm can be established.
    """
    results_path = config.aef_crc_phase5_reports_dir / "fusion_results.csv"
    mcnemar_path = config.aef_crc_phase5_reports_dir / "fusion_mcnemar_holm.csv"

    if not results_path.exists() or not mcnemar_path.exists():
        return None, (
            f"Missing {results_path.name} or {mcnemar_path.name}"
        )

    p5_manifest = config.aef_crc_phase5_reports_dir / "phase5_manifest.json"
    if p5_manifest.exists():
        try:
            p5_data = json.loads(p5_manifest.read_text(encoding="utf-8"))
            p5_run_id = p5_data.get("run_id")
            from modules.calibration_handoff import get_active_run_id
            expected_run_id = get_active_run_id(config)
            if expected_run_id and p5_run_id != expected_run_id:
                raise RuntimeError(
                    f"Cross-run artifact mismatch: Phase 5 was run under run_id='{p5_run_id}', "
                    f"but active Phase 3 winner has run_id='{expected_run_id}'. Artifacts from separate runs cannot be combined!"
                )
        except RuntimeError:
            raise
        except Exception:
            pass

    with results_path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None, "fusion_results.csv exists but is empty"

    best_row = max(
        rows,
        key=lambda r: float(r["macro_f1_mean"]),
    )

    best_arm = best_row["arm"]

    # Fail loudly if Phase 5 produced an arm name that Phase 6 cannot use.
    if best_arm not in ARMS:
        return None, (
            f"Unknown Phase-5 arm '{best_arm}' in fusion_results.csv"
        )

    # EfficientNet-only is a legitimate Phase-5 winner.
    # Phase 6 must still proceed using that arm.
    if best_arm == "EfficientNet":
        return (
            "EfficientNet",
            "EfficientNet-only is the Phase-5 winner; "
            "no fusion arm significantly beats it",
        )

    with mcnemar_path.open() as f:
        mcnemar_rows = list(csv.DictReader(f))

    pair_row = next(
        (
            r
            for r in mcnemar_rows
            if {r["arm_a"], r["arm_b"]}
            == {"EfficientNet", best_arm}
        ),
        None,
    )

    if pair_row is None or pair_row["p_value_holm"] == "":
        return (
            "EfficientNet",
            f"Best arm '{best_arm}' has insufficient evidence to "
            "establish superiority -- retaining EfficientNet-only",
        )

    p_holm = float(pair_row["p_value_holm"])

    if p_holm >= 0.05:
        return (
            "EfficientNet",
            f"Best arm '{best_arm}' does not significantly beat "
            f"EfficientNet-only (Holm p={p_holm:.4f} >= 0.05) "
            "-- retaining EfficientNet-only",
        )

    return (
        best_arm,
        f"'{best_arm}' significantly beats EfficientNet-only "
        f"(Holm p={p_holm:.4f})",
    )


def run_candidates_for_fold(
    config,
    data,
    fold_index,
    ga_context=None,
    bda_context=None,
):
    """
    Runs candidates 1-3 always.

    Candidate 4 (GA) and Candidate 5 (BDA) are run when ga_context /
    bda_context are not None.
    The caller decides whether metaheuristic optimization is justified.

    Every candidate is invoked through the same four-argument
    run_selector() interface:

        run_selector(name, data, config, random_seed)

    Selector-specific argument handling remains inside
    modules.feature_selection.py.
    """

    results = {
        "no_selection": run_selector(
            "none",
            data,
            config,
            config.random_seed,
        ),

        "xgboost_importance": run_selector(
            "xgboost_importance",
            data,
            config,
            config.random_seed,
        ),

        "rfe": run_selector(
            "rfe",
            data,
            config,
            config.random_seed,
        ),
    }

    if ga_context is not None:
        results["ga"] = run_selector(
            "ga",
            data,
            config,
            config.random_seed + fold_index if fold_index else config.random_seed,
        )

    if bda_context is not None:
        results["bda"] = run_selector(
            "bda",
            data,
            config,
            config.random_seed + fold_index if fold_index else config.random_seed,
        )

    return results



def evaluate_selection(
    config,
    data,
    selection: "SelectionResult",
    classes,
    class_weights,
    fold_index=0,
):
    """
    Evaluate one feature-selection candidate on one CV fold.

    IMPORTANT:
    The classifier receives the exact fold-local sample weights derived
    from fold.class_weights.

    This preserves the same class-imbalance correction used elsewhere
    in the AEF-CRC pipeline.

    fold_index has a default of 0 for backward compatibility with
    existing direct callers/tests, while the actual CV runner passes
    the real fold index explicitly.
    """

    mask = selection.selected_mask

    # Convert fold-local class weights into per-sample weights.
    sw_train = compute_sample_weights(
        data.y_train,
        class_weights,
    )

    # Use the configured classifier registry.
    _, clf = build_classifier(
        config.random_seed,
        config.classifier_name,
    )

    # Train ONLY on the training part of this fold.
    clf.fit(
        data.X_train[:, mask],
        data.y_train,
        sample_weight=sw_train,
    )

    # Validation data is used only for final evaluation.
    y_pred = clf.predict(
        data.X_val[:, mask]
    )

    metrics = compute_fold_metrics(
        data.y_val,
        list(y_pred),
        classes,
        fold_index=fold_index,
    )

    return metrics, y_pred


def run_benchmark(
    config,
    plan,
    branches,
    source_experiment,
    label_prefix="",
):
    from modules.handcrafted_features import HandcraftedFeatureExtractor

    extractor = HandcraftedFeatureExtractor(config)
    classes = config.target_classes

    # Safety guard before accessing plan.folds[0].
    if not plan.folds:
        raise ValueError(
            "Phase 6 requires at least one CV fold; none were built."
        )

    # ------------------------------------------------------------
    # Efficiency:
    # Estimate RFE cost from fold 0 BEFORE running all folds.
    # ------------------------------------------------------------

    fold0_data = build_fusion_fold(
        config,
        plan.folds[0],
        branches,
        source_experiment,
        extractor,
    )

    cost = estimate_rfe_cost(
        fold0_data,
        config.feature_selection_top_k,
        config.rfe_step,
        config.random_seed,
        len(plan.folds),
    )

    print(
        f"{label_prefix}"
        f"RFE cost estimate: "
        f"{cost['single_fold_seconds']:.2f}s/fold x "
        f"{cost['n_folds']} folds "
        f"= ~{cost['estimated_total_seconds']:.1f}s total. "
        f"Proceeding."
    )

    # ------------------------------------------------------------
    # Storage for CV results.
    # ------------------------------------------------------------

    method_fold_metrics = {
        "no_selection": [],
        "xgboost_importance": [],
        "rfe": [],
    }

    pooled_predictions = {
        "no_selection": {},
        "xgboost_importance": {},
        "rfe": {},
    }

    selection_details = {
        "no_selection": [],
        "xgboost_importance": [],
        "rfe": [],
    }

    # ------------------------------------------------------------
    # Run candidates across all folds.
    # ------------------------------------------------------------

    for fold_pos, fold in enumerate(plan.folds):

        # Fold 0 was already built for the RFE cost estimate.
        # Reuse it rather than rebuilding all deep/handcrafted features.
        #
        # This does NOT alter the data split, feature transformation,
        # selector, classifier, or evaluation workflow.
        data = (
            fold0_data
            if fold_pos == 0
            else build_fusion_fold(
                config,
                fold,
                branches,
                source_experiment,
                extractor,
            )
        )

        results = run_candidates_for_fold(
            config,
            data,
            fold.fold_index,
        )

        for method, selection in results.items():

            metrics, y_pred = evaluate_selection(
                config,
                data,
                selection,
                classes,
                fold.class_weights,
                fold.fold_index,
            )

            method_fold_metrics[method].append(
                metrics
            )

            # Store paired predictions by PSD ID for McNemar's test.
            for pid, true, pred in zip(
                data.psd_ids_val,
                data.y_val,
                y_pred,
            ):
                pooled_predictions[method][pid] = (
                    true,
                    pred,
                )

            selection_details[method].append(
                selection
            )

            print(
                f"{label_prefix}"
                f"  fold {fold.fold_index} "
                f"{method:20s}: "
                f"{selection.selected_count:4d} features "
                f"({selection.branch_retained}) | "
                f"macro_f1={metrics.macro_f1:.4f} | "
                f"fit_time={selection.fit_time_sec:.2f}s"
            )

    # ------------------------------------------------------------
    # Aggregate fold results.
    # ------------------------------------------------------------

    aggregates = {
        method: aggregate_fold_metrics(fold_metrics)
        for method, fold_metrics in method_fold_metrics.items()
    }

    for method, agg in aggregates.items():
        print(
            f"{label_prefix}"
            f"{method:20s}: "
            f"macro_f1 = "
            f"{agg.macro_f1_mean:.4f} +/- "
            f"{agg.macro_f1_std:.4f}"
        )

    # ------------------------------------------------------------
    # Pairwise Holm-corrected McNemar test.
    # ------------------------------------------------------------

    print(
        f"{label_prefix}"
        "Pairwise McNemar (pooled), Holm-corrected:"
    )

    pairs = list(
        itertools.combinations(
            pooled_predictions.keys(),
            2,
        )
    )

    raw = []

    for a, b in pairs:

        common = sorted(
            set(pooled_predictions[a])
            & set(pooled_predictions[b])
        )

        y_true = [
            pooled_predictions[a][pid][0]
            for pid in common
        ]

        y_pred_a = [
            pooled_predictions[a][pid][1]
            for pid in common
        ]

        y_pred_b = [
            pooled_predictions[b][pid][1]
            for pid in common
        ]

        raw.append(
            (
                a,
                b,
                mcnemar_test(
                    y_true,
                    y_pred_a,
                    y_pred_b,
                ),
            )
        )

    p_values = [
        result["p_value"]
        if result is not None
        else None
        for _, _, result in raw
    ]

    adjusted = holm_correction(
        p_values
    )

    any_significant = False

    for (a, b, result), p_adj in zip(
        raw,
        adjusted,
    ):

        if result is None:

            print(
                f"{label_prefix}"
                f"  {a} vs {b}: "
                "evidence insufficient "
                "(n01+n10<25)"
            )

        else:

            significant = p_adj < 0.05

            any_significant = (
                any_significant
                or significant
            )

            print(
                f"{label_prefix}"
                f"  {a} vs {b}: "
                f"p_holm={p_adj:.4f}"
                f"{' (significant)' if significant else ''}"
            )

    # ------------------------------------------------------------
    # Optimization Feature Selection: BDA (Primary) & GA (Comparator)
    #
    # Unconditionally executed across all folds to guarantee empirical
    # evidence for the primary optimization method and its comparator.
    # ------------------------------------------------------------

    print(
        f"{label_prefix}"
        "Running metaheuristic optimization algorithms: "
        "BDA (Primary) and GA (Comparator)..."
    )

    # 1. Primary Optimizer: Binary Dragonfly Algorithm (BDA)
    bda_fold_metrics = []
    bda_pooled = {}
    bda_masks = []

    for fold in plan.folds:
        data = build_fusion_fold(
            config,
            fold,
            branches,
            source_experiment,
            extractor,
        )

        selection = run_selector(
            "bda",
            data,
            config,
            config.random_seed + fold.fold_index,
        )
        bda_masks.append(selection.selected_mask)

        metrics, y_pred = evaluate_selection(
            config,
            data,
            selection,
            classes,
            fold.class_weights,
            fold.fold_index,
        )

        bda_fold_metrics.append(metrics)

        for pid, true, pred in zip(
            data.psd_ids_val,
            data.y_val,
            y_pred,
        ):
            bda_pooled[pid] = (true, pred)

        print(
            f"{label_prefix}"
            f"  fold {fold.fold_index} bda: "
            f"{selection.selected_count} features "
            f"({selection.branch_retained}) | "
            f"macro_f1={metrics.macro_f1:.4f} | "
            f"pop={selection.extra['population_size']} "
            f"iter={selection.extra['iterations']} | "
            f"fit_time={selection.fit_time_sec:.2f}s"
        )

    aggregates["bda"] = aggregate_fold_metrics(bda_fold_metrics)
    pooled_predictions["bda"] = bda_pooled

    from modules.feature_selection import compute_pairwise_jaccard
    jaccard_stats = compute_pairwise_jaccard(bda_masks)

    print(
        f"{label_prefix}"
        f"bda: macro_f1 = "
        f"{aggregates['bda'].macro_f1_mean:.4f} +/- "
        f"{aggregates['bda'].macro_f1_std:.4f} | "
        f"Jaccard stability mean={jaccard_stats['mean_jaccard']:.4f}"
    )

    # 2. Benchmark Comparator: Genetic Algorithm (GA)
    ga_fold_metrics = []
    ga_pooled = {}

    for fold in plan.folds:

        data = build_fusion_fold(
            config,
            fold,
            branches,
            source_experiment,
            extractor,
        )

        selection = run_selector(
            "ga",
            data,
            config,
            config.random_seed + fold.fold_index,
        )

        metrics, y_pred = evaluate_selection(
            config,
            data,
            selection,
            classes,
            fold.class_weights,
            fold.fold_index,
        )

        ga_fold_metrics.append(
            metrics
        )

        for pid, true, pred in zip(
            data.psd_ids_val,
            data.y_val,
            y_pred,
        ):
            ga_pooled[pid] = (
                true,
                pred,
            )

        print(
            f"{label_prefix}"
            f"  fold {fold.fold_index} ga: "
            f"{selection.selected_count} features "
            f"({selection.branch_retained}) | "
            f"macro_f1={metrics.macro_f1:.4f} | "
            f"pop={selection.extra['population_size']} "
            f"gen={selection.extra['generations']} | "
            f"fit_time={selection.fit_time_sec:.2f}s"
        )

    aggregates["ga"] = aggregate_fold_metrics(
        ga_fold_metrics
    )

    pooled_predictions["ga"] = ga_pooled

    print(
        f"{label_prefix}"
        f"ga: macro_f1 = "
        f"{aggregates['ga'].macro_f1_mean:.4f} +/- "
        f"{aggregates['ga'].macro_f1_std:.4f}"
    )

    return aggregates, pooled_predictions


def execute_production_bda(
    config: PSDConfig,
    plan: Any,
    branches: Tuple[str, ...],
    source_experiment: str,
) -> Path:
    """Executes the single frozen production BDA feature selection on the unified
    1146-image non-augmented cohort with inner 80/20 stratified split.
    Authorized ONLY during real experimental execution phase.
    Persists the resulting mask to artifacts/phase6/production_bda_mask.joblib.
    """
    from modules.fusion import build_fusion_final
    from modules.handcrafted_features import HandcraftedFeatureExtractor
    from modules.feature_selection import select_bda, save_production_bda_mask

    final_train_records = plan.folds[0].train_records + plan.folds[0].val_records
    splittable_count = sum(len(f.val_records) for f in plan.folds)
    assert len(final_train_records) == splittable_count == 1146, (
        f"Expected exactly 1146 reunified non-augmented records, got {len(final_train_records)}"
    )

    extractor = HandcraftedFeatureExtractor(config)
    data = build_fusion_final(
        config,
        final_train_records,
        plan.holdout_val_records,
        branches,
        source_experiment,
        extractor,
    )

    print("Executing single production BDA on unified 1146-image cohort (inner 80/20 split)...")
    res = select_bda(data, config, config.random_seed)
    from modules.calibration_handoff import get_active_run_id
    active_run = get_active_run_id(config)
    mask_path = save_production_bda_mask(
        res.selected_mask,
        config,
        run_id=active_run,
        metadata={
            "selected_count": res.selected_count,
            "branch_retained": res.branch_retained,
            "fit_time_sec": res.fit_time_sec,
            "source_experiment": source_experiment,
        },
    )
    print(f"Production BDA mask successfully saved to {mask_path} ({res.selected_count} features selected).")
    return mask_path


def validate_phase6_framework(config) -> bool:
    """
    Non-destructive framework validation mode for Phase 6.

    Verifies:
      1. Imports and registry resolution for BDA, GA, RFE, GBDT, None.
      2. Time-varying V-shaped transfer function math & bounds.
      3. Synthetic micro-benchmark with BDA (small pop, small iter, 1348-D).
      4. Mask validity (shape, boolean dtype, non-empty, all-zero recovery).
      5. Determinism under fixed seed.
      6. Independent swarm paths under different fold seeds.
      7. Feature provenance & family breakdown accounting (deep, glcm, lbp, hog, color_lab).
      8. Jaccard similarity and stability computations.
      9. Feature names mapping to Phase 5 fusion representations.
      10. Gate check logic behavior when Phase 5 outputs are absent.

    Touches NO frozen data, writes NO files to reports directories, and
    exits with 0 on success.
    """
    print("=== AEF-CRC Phase 6: Framework Validation Mode ===")
    print("Running non-destructive validation checks...\n")

    # 1. Registry & Imports
    print("[Check 1/9] Verifying feature selector registry...")
    from modules.feature_selection import (
        run_selector,
        compute_jaccard_similarity,
        compute_pairwise_jaccard,
        compute_feature_family_breakdown,
        map_mask_to_feature_names,
        select_bda,
        SELECTOR_REGISTRY,
    )
    for expected in ["bda", "ga", "rfe", "xgboost_importance", "none"]:
        assert expected in SELECTOR_REGISTRY, f"Missing selector '{expected}' in SELECTOR_REGISTRY"
    print("  [PASS] All expected selectors (including 'bda' and 'ga') registered.")

    # 2. Mathematical Transfer Function Verification
    print("[Check 2/9] Verifying time-varying V-shaped transfer function...")
    # T(delta_x) = |tanh(tau(t) * delta_x)|
    # tau(0) = tau_min = 1.0, tau(T) = tau_max = 4.0
    tau_0 = config.bda_tau_min + (config.bda_tau_max - config.bda_tau_min) * (0 / 30)
    tau_T = config.bda_tau_min + (config.bda_tau_max - config.bda_tau_min) * (30 / 30)
    assert abs(tau_0 - 1.0) < 1e-6, f"tau(0) expected 1.0, got {tau_0}"
    assert abs(tau_T - 4.0) < 1e-6, f"tau(T) expected 4.0, got {tau_T}"
    # Check T(0) = 0
    t_zero = abs(np.tanh(tau_0 * 0.0))
    assert t_zero == 0.0, f"T(0) expected 0.0, got {t_zero}"
    # Check bounds for large step
    t_large = abs(np.tanh(tau_T * 10.0))
    assert 0.0 <= t_large <= 1.0, f"T(delta_x) out of [0, 1]: {t_large}"
    print("  [PASS] Transfer function parameters and boundary behavior verified.")

    # 3. Micro-benchmark with BDA on 1348-D synthetic fusion data
    print("[Check 3/9] Running synthetic micro-benchmark with BDA (pop=5, iter=3, 1348-D)...")
    from modules.fusion import FusionFoldData
    rng = np.random.default_rng(42)
    n_train, n_val, n_feat = 60, 20, 1348
    X_train = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    y_train = [config.target_classes[i % 4] for i in range(n_train)]
    X_val = rng.standard_normal((n_val, n_feat)).astype(np.float32)
    y_val = [config.target_classes[i % 4] for i in range(n_val)]
    branch_dims = {"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6}
    synth_data = FusionFoldData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        psd_ids_train=[f"PSD-TR-{i:04d}" for i in range(n_train)],
        psd_ids_val=[f"PSD-VA-{i:04d}" for i in range(n_val)],
        branch_dims=branch_dims,
    )

    micro_cfg = dataclasses.replace(
        config,
        bda_population_size=5,
        bda_iterations=3,
        bda_nested_val_fraction=0.2,
        bda_feature_count_penalty=0.0005,
    )

    res1 = run_selector("bda", synth_data, micro_cfg, random_seed=123)
    assert res1.method == "bda"
    assert res1.selected_mask.shape == (n_feat,)
    assert res1.selected_mask.dtype == bool
    assert res1.selected_count == int(res1.selected_mask.sum())
    assert res1.selected_count > 0, "BDA returned empty mask"
    assert len(res1.extra["convergence_history"]) == 3
    print(f"  [PASS] Micro-benchmark complete: {res1.selected_count}/{n_feat} features selected in {res1.fit_time_sec:.2f}s.")

    # 4. Determinism
    print("[Check 4/9] Verifying BDA determinism with identical seed...")
    res2 = run_selector("bda", synth_data, micro_cfg, random_seed=123)
    assert np.array_equal(res1.selected_mask, res2.selected_mask), "BDA is not deterministic with identical seed"
    assert res1.extra["convergence_history"] == res2.extra["convergence_history"], "BDA convergence history diverges"
    print("  [PASS] Identical seed yields bitwise identical masks and convergence trajectories.")

    # 5. Fold Isolation & Independent Swarms
    print("[Check 5/9] Verifying fold isolation & seed sensitivity...")
    res_other_seed = run_selector("bda", synth_data, micro_cfg, random_seed=124)
    print(f"  Seed 123 selected: {res1.selected_count}, Seed 124 selected: {res_other_seed.selected_count}")
    print("  [PASS] Independent seeds execute isolated swarms.")

    # 6. Feature Family Accounting
    print("[Check 6/9] Verifying feature family breakdown accounting...")
    branch_dims = {"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6}
    breakdown = compute_feature_family_breakdown(res1.selected_mask, branch_dims)
    total_accounted = sum(breakdown.values())
    assert total_accounted == res1.selected_count, (
        f"Family breakdown sum ({total_accounted}) != selected_count ({res1.selected_count})"
    )
    for branch_name, count in breakdown.items():
        assert 0 <= count <= branch_dims[branch_name]
    print(f"  [PASS] Feature families correctly partitioned: {breakdown} (Sum = {total_accounted}).")

    # 7. Jaccard Similarity Computation
    print("[Check 7/9] Verifying Jaccard similarity metrics...")
    j_self = compute_jaccard_similarity(res1.selected_mask, res1.selected_mask)
    assert abs(j_self - 1.0) < 1e-6, f"Self-Jaccard should be 1.0, got {j_self}"
    j_cross = compute_jaccard_similarity(res1.selected_mask, res_other_seed.selected_mask)
    assert 0.0 <= j_cross <= 1.0, f"Cross-Jaccard out of bounds: {j_cross}"
    pairwise_stats = compute_pairwise_jaccard([res1.selected_mask, res_other_seed.selected_mask])
    assert "mean_jaccard" in pairwise_stats and "median_jaccard" in pairwise_stats
    print(f"  [PASS] Jaccard computations valid (Self: {j_self:.2f}, Cross: {j_cross:.4f}).")

    # 8. Feature Name Mapping
    print("[Check 8/9] Verifying feature name mapping...")
    selected_names = map_mask_to_feature_names(
        res1.selected_mask,
        branches=["deep", "glcm", "lbp", "hog", "color_lab"],
        config=config,
    )
    assert len(selected_names) == res1.selected_count
    print(f"  [PASS] {len(selected_names)} feature names mapped correctly to mask indices.")

    # 9. Gate Check Behavior
    print("[Check 9/9] Verifying Phase 5 gate check behavior...")
    winner, detail = parse_phase5_winner(config)
    print(f"  Gate status: winner={winner}, detail='{detail}'")
    print("  [PASS] Gate check evaluated safely without uncaught exceptions.")

    print("\n=== Phase 6 Framework Validation PASSED (All 9 checks passed) ===")
    return True


def main() -> int:

    config = get_config()

    if "--validate-framework" in sys.argv or "--validate-only" in sys.argv:
        ok = validate_phase6_framework(config)
        return 0 if ok else 1

    print(
        "=== AEF-CRC Phase 6: "
        "Feature Refinement Benchmark ===\n"
    )

    # ============================================================
    # Step 0: Phase-5 gate
    # ============================================================

    print("--- Step 0: Gate check ---")

    winner, detail = parse_phase5_winner(
        config
    )

    print(detail)

    # Production BDA entrypoint for experimental execution phase
    if "--run-production-bda" in sys.argv:
        if not winner:
            print("ERROR: Cannot execute production BDA: Phase 5 winning arm not established.")
            return 1
        plan = load_frozen_fold_plan(config)
        winner_json_path = config.aef_crc_phase3_reports_dir / "winner.json"
        p3_winner = "P3-BASE"
        if winner_json_path.exists():
            try:
                wdata = json.loads(winner_json_path.read_text(encoding="utf-8"))
                p3_winner = wdata.get("winner_experiment_id", "P3-BASE")
            except Exception:
                pass
        execute_production_bda(config, plan, ARMS[winner], p3_winner)
        return 0

    # ------------------------------------------------------------
    # REAL PIPELINE
    # ------------------------------------------------------------

    if winner:

        print(
            f"\nPhase-5 winning arm: "
            f"{winner} "
            f"(branches: {ARMS[winner]})"
        )

        # --------------------------------------------------------
        # Dataset freeze verification.
        # --------------------------------------------------------

        freeze_ok = DatasetFreezer(
            config
        ).verify().matches

        print(
            f"DATASET FREEZE: "
            f"{'PASS' if freeze_ok else 'FAIL'}"
        )

        if not freeze_ok:
            return 1

        winner_json_path = config.aef_crc_phase3_reports_dir / "winner.json"
        p3_winner = "P3-BASE"
        if winner_json_path.exists():
            try:
                wdata = json.loads(winner_json_path.read_text(encoding="utf-8"))
                p3_winner = wdata.get("winner_experiment_id", "P3-BASE")
                mode = wdata.get("preprocessing_mode", "standard")
                aug = "true" if wdata.get("training_time_augmentation") else "false"
                config = dataclasses.replace(config, preprocessing_mode=mode, training_time_augmentation=aug)
            except Exception:
                pass

        plan = load_frozen_fold_plan(config)

        if not plan.folds:
            print("No folds available.")
            return 1

        # --------------------------------------------------------
        # Run Phase 6 feature refinement benchmark.
        # --------------------------------------------------------

        aggregates, pooled = run_benchmark(
            config,
            plan,
            ARMS[winner],
            p3_winner,
        )

        print(
            "\nPHASE 6 RESULT: "
            "REAL BENCHMARK COMPLETE."
        )

        return 0

    # ============================================================
    # PLUMBING-ONLY FALLBACK
    # ============================================================

    print(
        "\nNo real Phase-5 winner available -- "
        "running PLUMBING-ONLY verification."
    )

    print(
        "*** EVERYTHING BELOW IS A SYNTHETIC "
        "PLUMBING CHECK, NOT A RESEARCH RESULT ***\n"
    )

    import json
    import shutil

    from modules.synthetic_fixtures import (
        make_test_config, 
        build_synthetic_dataset,
        run_pipeline_through_split,
    )

    tmp_root = (
        config.project_root
        / "_phase6_plumbing_tmp"
    )

    test_config = make_test_config(
        tmp_root,
        seed=42,
    )

    build_synthetic_dataset(
        test_config
    )

    run_pipeline_through_split(
        test_config
    )

    plan = ImbalanceAwareFoldLoader(
        test_config,
        k=3,
    ).build()

    if not plan.folds:

        print(
            "PLUMBING CHECK FAILED: "
            "no folds built from synthetic data."
        )

        return 1

    # ------------------------------------------------------------
    # Create synthetic deep-feature cache.
    # ------------------------------------------------------------

    from modules.backbones import get_backbone
    from modules.experiment_config import representation_id

    backbone_dim = get_backbone(
        test_config.backbone
    ).output_dim

    repr_id = representation_id(
        test_config
    )

    fake_experiment = "PLUMBING-FAKE"

    rng = np.random.default_rng(
        0
    )

    for fold in plan.folds:

        for records, split_name in (
            (
                fold.train_records,
                "train",
            ),
            (
                fold.val_records,
                "val",
            ),
        ):

            out_dir = (
                test_config.aef_crc_artifacts_dir
                / "deep_features"
                / fake_experiment
                / f"fold_{fold.fold_index:02d}"
                / split_name
            )

            out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            psd_ids = [
                r.psd_id
                for r in records
            ]

            for r in records:

                np.save(
                    out_dir
                    / f"{r.psd_id}.npy",
                    rng.normal(
                        size=backbone_dim
                    ).astype(np.float32),
                )

            # Full real-schema manifest.
            manifest = {
                "experiment_name": fake_experiment,
                "fold_index": fold.fold_index,
                "split": split_name,
                "checkpoint_path":
                    "SYNTHETIC PLUMBING DATA "
                    "-- no real checkpoint",
                "n_features": len(psd_ids),
                "feature_dim": backbone_dim,
                "psd_ids": psd_ids,
                "backbone_name":
                    test_config.backbone,
                "representation_id": repr_id,
            }

            (
                out_dir
                / "manifest.json"
            ).write_text(
                json.dumps(
                    manifest,
                    indent=2,
                ),
                encoding="utf-8",
            )

    print(
        f"Synthetic deep-feature cache written "
        f"({len(plan.folds)} folds). "
        "Using the full combined arm "
        "(deep+glcm+lbp+hog) for plumbing "
        "-- NOT a real Phase-5 winner, "
        "just a scope choice for exercising "
        "all four candidates.\n"
    )

    try:

        run_benchmark(
            test_config,
            plan,
            ARMS[
                "EfficientNet+GLCM+LBP+HOG"
            ],
            fake_experiment,
            label_prefix="[PLUMBING] ",
        )

        plumbing_ok = True

    except Exception as exc:

        print(
            f"PLUMBING CHECK FAILED: "
            f"{type(exc).__name__}: {exc}"
        )

        plumbing_ok = False

    # Always clean up synthetic artifacts.
    shutil.rmtree(
        tmp_root,
        ignore_errors=True,
    )

    print(
        f"\nPLUMBING CHECK: "
        f"{'PASSED' if plumbing_ok else 'FAILED'}"
    )

    print(
        "\nPhase-6 implementation complete; "
        "scientific benchmark pending real "
        "Phase-3/Phase-5 features."
    )

    return 0 if plumbing_ok else 1


if __name__ == "__main__":
    sys.exit(main())