"""
run_aef_crc_phase9.py

AEF-CRC Phase 9: Conformal Prediction.

Consumes the Phase-8 ConformalHandoff (modules.calibration_handoff)
carrying the dedicated 123-image conformal calibration partition (val_conf).

Frozen Protocol & Scientific Rigor:
-----------------------------------
1. Primary Calibration Input:
   Phase 9 conformal calibration MUST consume probabilities calibrated by
   the PRIMARY Platt (Sigmoid) calibrator fitted on val_calib.

2. Primary Conformal Method: Pooled Marginal Split-Conformal Prediction:
   - Target significance: alpha = 0.10 (90% nominal coverage).
   - Calibration set: val_conf (N_conf = 123 samples), strictly disjoint from val_calib.
   - Exact finite-sample quantile index:
       k = ceil((n + 1) * (1 - alpha))
     For n = 123, alpha = 0.10:
       k = ceil(124 * 0.90) = ceil(111.6) = 112.
       q_hat = s_{(112)}.

3. Secondary / Diagnostic Method: Class-Conditional Mondrian Conformal Prediction:
   - Evaluated across the 4 disease classes independently:
     Psoriasis, Lichen Planus, Pityriasis Rosea, Seborrheic Dermatitis.
   - Source x class Mondrian grouping is strictly rejected due to sparse/empty cells.
   - Small-sample diagnostic caveat: Rare classes (especially Seborrheic Dermatitis with n ~ 9-10)
     exhibit coarser empirical quantiles (k = ceil(11*0.9) = 10 for n=10) and higher finite-sample variance.

4. Data Architecture & Final Evaluation Role:
   - 123 val_calib: fit probability calibrators (Platt primary, Isotonic secondary).
   - 123 val_conf: fit conformal prediction sets using primary Platt-calibrated probabilities.
   - 243 outer test: locked final evaluation set (MUST NOT be used for fitting or selection).
   - Primary final calibration metrics: ECE, multiclass Brier score (Reliability diagrams are supporting analysis).

5. Final Deployed Artifact:
   - Saves ConformalArtifact / FinalPipelineHandoff carrying the complete frozen inference bundle:
     frozen feature mask, base Random Forest, primary Platt calibration models, and conformal quantiles.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config.config import get_config
from modules.calibration_handoff import (
    ConformalHandoff, ConformalArtifact, FinalPipelineHandoff,
    load_conformal_handoff, save_conformal_artifact, save_final_pipeline_handoff,
    validate_conformal_handoff_provenance,
)
from modules.conformal import (
    compute_nonconformity_scores,
    compute_conformal_quantile,
    fit_marginal_conformal,
    fit_class_conditional_mondrian,
    predict_conformal_sets,
    predict_conformal_sets_mondrian,
    evaluate_conformal_sets,
)


def run_conformal_calibration(
    config,
    handoff: ConformalHandoff,
    label_prefix: str = "",
) -> FinalPipelineHandoff:
    """
    Executes Phase 9 conformal calibration on the dedicated val_conf partition
    using probabilities produced by the primary Platt calibrator.
    """
    classes = handoff.class_order
    n_conf = len(handoff.val_conf_true_labels)
    print(f"{label_prefix}Conformal calibration set (val_conf) size: {n_conf}")
    print(f"{label_prefix}Calibration input: Primary Platt (Sigmoid) probabilities")

    if n_conf < 10:
        raise ValueError(f"Insufficient conformal calibration samples: {n_conf} (minimum 10 required)")

    probs_conf = handoff.val_conf_calibrated_probabilities
    labels_conf = handoff.val_conf_true_labels
    alpha = handoff.alpha if hasattr(handoff, "alpha") and handoff.alpha else 0.10

    # 1. Fit Primary: Pooled Marginal Split-Conformal Prediction
    marginal_fit = fit_marginal_conformal(
        probs=probs_conf,
        true_labels=labels_conf,
        class_order=classes,
        alpha=alpha,
    )
    print(f"\n{label_prefix}--- Primary: Pooled Marginal Split-Conformal ---")
    print(f"{label_prefix}  Nominal coverage: {marginal_fit.nominal_coverage * 100:.1f}% (alpha={alpha:.2f})")
    print(f"{label_prefix}  Samples (N_conf): {marginal_fit.n_samples}")
    print(f"{label_prefix}  Exact quantile index: k = ceil(({marginal_fit.n_samples} + 1) * {1.0 - alpha:.2f}) = {marginal_fit.formula_k}")
    print(f"{label_prefix}  Marginal quantile threshold (q_hat): {marginal_fit.q_hat:.4f}")
    print(f"{label_prefix}  Inclusion condition: P_hat(c | x) >= {1.0 - marginal_fit.q_hat:.4f}")

    # 2. Fit Secondary / Diagnostic: Class-Conditional Mondrian Conformal Prediction
    mondrian_fit = fit_class_conditional_mondrian(
        probs=probs_conf,
        true_labels=labels_conf,
        class_order=classes,
        alpha=alpha,
        min_recommended_n=15,
    )
    print(f"\n{label_prefix}--- Secondary / Diagnostic: Class-Conditional Mondrian ---")
    for cls in classes:
        n_c = mondrian_fit.per_class_n[cls]
        q_c = mondrian_fit.q_hat_by_class[cls]
        print(f"{label_prefix}  {cls:22s}: n={n_c:2d}, q_hat={q_c:.4f} (inclusion: P>={1.0 - q_c:.4f})")
        if cls in mondrian_fit.small_sample_warnings:
            print(f"{label_prefix}    [DIAGNOSTIC WARNING] {mondrian_fit.small_sample_warnings[cls]}")

    # 3. In-sample calibration diagnostics on val_conf
    marginal_sets = predict_conformal_sets(probs_conf, marginal_fit.q_hat, classes)
    eval_marginal = evaluate_conformal_sets(marginal_sets, labels_conf, classes, nominal_coverage=1.0 - alpha)

    mondrian_sets = predict_conformal_sets_mondrian(probs_conf, mondrian_fit.q_hat_by_class, classes)
    eval_mondrian = evaluate_conformal_sets(mondrian_sets, labels_conf, classes, nominal_coverage=1.0 - alpha)

    print(f"\n{label_prefix}--- val_conf In-Sample Verification (Diagnostic) ---")
    print(f"{label_prefix}  Marginal empirical coverage: {eval_marginal.marginal_coverage * 100:.1f}%")
    print(f"{label_prefix}  Marginal mean set size:       {eval_marginal.mean_set_size:.2f}")
    print(f"{label_prefix}  Marginal singleton fraction:  {eval_marginal.singleton_fraction * 100:.1f}%")
    print(f"{label_prefix}  Marginal empty set fraction:  {eval_marginal.empty_set_fraction * 100:.1f}%")
    print(f"{label_prefix}  Mondrian empirical coverage: {eval_mondrian.marginal_coverage * 100:.1f}%")
    print(f"{label_prefix}  Mondrian mean set size:       {eval_mondrian.mean_set_size:.2f}")

    # 4. Write Phase 9 reports
    out_dir = config.aef_crc_phase9_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "conformal_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "alpha", "nominal_coverage", "n_samples", "q_hat", "mean_set_size", "singleton_frac"])
        writer.writerow([
            "marginal", alpha, 1.0 - alpha, marginal_fit.n_samples, marginal_fit.q_hat,
            eval_marginal.mean_set_size, eval_marginal.singleton_fraction,
        ])

    with (out_dir / "mondrian_class_quantiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "n_conf", "q_hat_c", "inclusion_threshold", "warning"])
        for cls in classes:
            writer.writerow([
                cls,
                mondrian_fit.per_class_n[cls],
                mondrian_fit.q_hat_by_class[cls],
                1.0 - mondrian_fit.q_hat_by_class[cls],
                mondrian_fit.small_sample_warnings.get(cls, "None"),
            ])

    print(f"{label_prefix}Conformal calibration reports written to {out_dir}")

    # 5. Build Final Deployed Pipeline Handoff
    return FinalPipelineHandoff(
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
        marginal_q_hat=marginal_fit.q_hat,
        mondrian_q_hat=mondrian_fit.q_hat_by_class,
        platt_models=handoff.platt_models,
        isotonic_models=handoff.isotonic_models,
        alpha=alpha,
        n_conf_samples=n_conf,
        per_class_conf_counts=mondrian_fit.per_class_n,
        small_sample_warnings=mondrian_fit.small_sample_warnings,
        hog_reducer=getattr(handoff, "hog_reducer", None),
        feature_normalizers=getattr(handoff, "feature_normalizers", None),
        backbone_checkpoint_path=getattr(handoff, "backbone_checkpoint_path", None),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AEF-CRC Phase 9: Conformal Prediction")
    parser.add_argument(
        "--validate-framework",
        action="store_true",
        help="Run synthetic plumbing verification only (does not produce research results)",
    )
    args = parser.parse_args()

    config = get_config()
    print("=== AEF-CRC Phase 9: Conformal Prediction ===\n")
    print("--- Step 0: Check Phase-8 ConformalHandoff ---")

    handoff_path = config.aef_crc_phase8_artifacts_dir / "conformal_handoff.joblib"
    if handoff_path.exists():
        handoff = load_conformal_handoff(handoff_path)
        validate_conformal_handoff_provenance(handoff, config)
        print(f"Loaded real Phase-8 handoff from {handoff_path}")
        print(f"  representation_id={handoff.representation_id}")
        print(f"  calibration_method={handoff.calibration_method} (Primary: Platt)\n")

        final_artifact = run_conformal_calibration(config, handoff)

        out_path = config.aef_crc_phase9_artifacts_dir / "final_pipeline_handoff.joblib"
        save_final_pipeline_handoff(final_artifact, out_path)
        save_conformal_artifact(final_artifact, config.aef_crc_phase9_artifacts_dir / "conformal_artifact.joblib")
        print(f"\nFinal deployed pipeline handoff written to {out_path}")
        print("\nPHASE 9 RESULT: REAL EXECUTION COMPLETE.")
        return 0

    if not args.validate_framework:
        print(
            f"\n[PHASE 9 BLOCKER] Prerequisite artifact missing:\n"
            f"  Expected: {handoff_path}\n\n"
            f"Reason: Real Phase 8 probability calibration has not yet been executed by user command.\n"
            f"In accordance with pre-registered methodology:\n"
            f"  - Conformal calibration must consume primary Platt-calibrated probabilities from val_conf (123 samples).\n"
            f"  - Outer test set (243 samples) remains locked until final evaluation.\n"
            f"  - To run a purely synthetic plumbing check without real models, pass --validate-framework.\n"
            f"\nExecution halted cleanly."
        )
        return 2

    # Plumbing verification path (strictly gated behind --validate-framework)
    print(f"No real Phase-8 handoff at {handoff_path} -- running PLUMBING-ONLY verification (--validate-framework).")
    print("*** EVERYTHING BELOW IS A SYNTHETIC PLUMBING CHECK, NOT A RESEARCH RESULT ***\n")

    # Mock synthetic handoff
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    rng = np.random.default_rng(42)
    n = 123
    mock_probs = rng.dirichlet(np.ones(4), size=n)
    mock_labels = [classes[i] for i in rng.integers(0, 4, size=n)]

    mock_handoff = ConformalHandoff(
        representation_id="PLUMBING-REPR",
        experiment_id="PLUMBING-EXP",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=42,
        class_order=classes,
        final_classifier=object(),
        selected_feature_mask=np.ones(50, dtype=bool),
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6},
        calibration_method="platt",
        calibration_method_reason="Frozen primary method: Platt scaling",
        val_conf_psd_ids=[f"CONF_{i:04d}" for i in range(n)],
        val_conf_true_labels=mock_labels,
        val_conf_calibrated_probabilities=mock_probs,
        alpha=0.10,
    )

    try:
        final_artifact = run_conformal_calibration(config, mock_handoff, label_prefix="[PLUMBING] ")
        out_path = config.project_root / "_phase9_plumbing_tmp" / "final_pipeline_handoff.joblib"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_final_pipeline_handoff(final_artifact, out_path)
        reloaded = load_final_pipeline_handoff(out_path)
        assert reloaded.marginal_q_hat == final_artifact.marginal_q_hat
        import shutil
        shutil.rmtree(out_path.parent, ignore_errors=True)
        print(f"\n[PLUMBING] Final pipeline handoff round-tripped correctly.")
        plumbing_ok = True
    except Exception as exc:
        print(f"PLUMBING CHECK FAILED: {type(exc).__name__}: {exc}")
        plumbing_ok = False

    print(f"\nPLUMBING CHECK: {'PASSED' if plumbing_ok else 'FAILED'}")
    return 0 if plumbing_ok else 1


if __name__ == "__main__":
    sys.exit(main())
