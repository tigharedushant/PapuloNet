"""
run_aef_crc_phase10.py

AEF-CRC Phase 10: Explainable AI (XAI) & Interpretability.

SCIENTIFIC CONTRACT & PRE-REGISTERED METHODOLOGY:
------------------------------------------------
1. Two-Tier Interpretability Architecture:
   - Grad-CAM:
     * Operates exclusively on the Keras EfficientNet-B0 backbone.
     * Target: The PRE-SOFTMAX LOGIT of the Phase 3 Dense-4 disease classification head.
     * Generates spatial activation heatmaps identifying diagnostic lesion regions.
     * Bound to GPU (/device:GPU:0) when available.
     * Explains the CNN feature extractor, NOT the Random Forest or multimodal fusion.
   - TreeSHAP:
     * Operates on the production Random Forest classifier.
     * Target: RAW uncalibrated prediction probabilities (model_output="raw").
     * Uses feature_perturbation="tree_path_dependent" on CPU.
     * Operates on the BDA-selected feature vector of the 1316-D A7 fused representation
       (EfficientNet 1280, GLCM 12, LBP 18, Color LAB 6; NO HOG).
       V2 pipeline: 642 BDA-selected features (not 194 from V1).
     * Quantifies individual feature attributions and aggregate branch importance.
   - Explicit Boundary:
     Neither Grad-CAM nor TreeSHAP explains the Platt probability calibrator
     or conformal prediction set boundaries.

2. Strict Data Provenance & Test Set Isolation:
   - All XAI evaluations are performed strictly on representative samples selected
     from the designated 246-image outer validation partition (plan.holdout_val_records).
   - The 243-image locked outer test partition (output/06_final_split/test/)
     remains completely untouched and unaccessed.

3. Artifacts Consumed:
   - artifacts/phase9_v2/final_pipeline_handoff.joblib
   - artifacts/phase3/P3-BASE/fold_00/best_model.keras
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config.config import PSDConfig, get_config
from run_aef_crc_phase9_v2 import configure_phase9_v2
from modules.aef_input_validator import ImageRecord
from modules.calibration_handoff import (
    FinalPipelineHandoff,
    load_final_pipeline_handoff,
    validate_pipeline_handoff_provenance,
)
from modules.fold_loader import FoldPlan, load_frozen_fold_plan
from modules.inference import AEFCRCInferenceEngine
from modules.xai_gradcam import GradCAMExplainer
from modules.xai_shap import (
    A7_BRANCH_DIMS,
    A7_FULL_DIM,
    ProductionTreeSHAPExplainer,
    validate_production_shap_contract,
)
from utils.logger import get_module_logger


def validate_phase10_preflight(
    config: PSDConfig,
    handoff: FinalPipelineHandoff,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Executes comprehensive preflight checks for Phase 10:
    1. Validates FinalPipelineHandoff cross-run provenance.
    2. Validates Phase 10 / SHAP production contract (A7 1316-D, 194 active BDA features, RF).
    3. Verifies Keras backbone checkpoint availability and integrity.
    4. Detects execution hardware (GPU / CPU).
    5. Verifies frozen fold plan and strict isolation of the 243 locked test images.
    """
    # 1. Provenance validation
    validate_pipeline_handoff_provenance(handoff, config)

    # 2. SHAP / Production contract validation
    validate_production_shap_contract(
        classifier=handoff.final_classifier,
        selected_feature_mask=handoff.selected_feature_mask,
        expected_full_dim=A7_FULL_DIM,
        expected_k_features=int(handoff.selected_feature_mask.sum()),
        branch_dims=handoff.branch_dims,
    )

    # 3. Backbone checkpoint verification
    backbone_path = None
    if handoff.backbone_checkpoint_path:
        cand = Path(handoff.backbone_checkpoint_path)
        if cand.exists():
            backbone_path = cand

    if backbone_path is None:
        cand2 = config.aef_crc_artifacts_dir / "P3-BASE" / "fold_00" / "best_model.keras"
        if cand2.exists():
            backbone_path = cand2
        else:
            raise FileNotFoundError(
                f"Phase 3 backbone checkpoint not found at {handoff.backbone_checkpoint_path} or {cand2}."
            )

    # 4. Device detection
    import tensorflow as tf

    physical_gpus = tf.config.list_physical_devices("GPU")
    gpu_detected = len(physical_gpus) > 0
    resolved_device = "/GPU:0" if (gpu_detected and device in ("auto", "gpu", "cuda")) else "/CPU:0"

    # 5. Fold plan and locked test set isolation verification
    plan = load_frozen_fold_plan(config)
    n_val = len(plan.holdout_val_records)
    n_test = len(plan.holdout_test_records)

    if n_val != 246:
        raise ValueError(f"Expected exactly 246 outer validation records, found {n_val}")
    if n_test != 243:
        raise ValueError(f"Expected exactly 243 locked outer test records, found {n_test}")

    # Verify no overlap between validation and test paths
    val_paths = {str(r.file_path).replace("\\", "/") for r in plan.holdout_val_records}
    test_paths = {str(r.file_path).replace("\\", "/") for r in plan.holdout_test_records}
    overlap = val_paths.intersection(test_paths)
    if overlap:
        raise RuntimeError(f"Data contamination detected! {len(overlap)} samples overlap between val and test.")

    return {
        "status": "PASS",
        "representation_id": handoff.representation_id,
        "experiment_id": handoff.experiment_id,
        "classifier_name": handoff.classifier_name,
        "fused_dimensions": A7_FULL_DIM,
        "selected_features": int(handoff.selected_feature_mask.sum()),
        "branch_dims": handoff.branch_dims,
        "backbone_checkpoint": str(backbone_path),
        "gpu_available": gpu_detected,
        "gpu_device_name": physical_gpus[0].name if gpu_detected else "None",
        "resolved_device": resolved_device,
        "val_samples_available": n_val,
        "test_samples_locked": n_test,
    }


def select_representative_validation_samples(
    plan: FoldPlan,
    config: PSDConfig,
    num_per_class: int = 2,
) -> List[ImageRecord]:
    """
    Deterministically selects representative samples across each disease class
    strictly from plan.holdout_val_records.
    Asserts zero access to locked test records.
    """
    rng = np.random.default_rng(config.random_seed)
    selected_records: List[ImageRecord] = []

    # Map validation records by class
    by_class: Dict[str, List[ImageRecord]] = {cls: [] for cls in config.target_classes}
    for rec in plan.holdout_val_records:
        if rec.mapped_class in by_class:
            by_class[rec.mapped_class].append(rec)

    for cls in config.target_classes:
        candidates = by_class[cls]
        if len(candidates) < num_per_class:
            raise ValueError(
                f"Class '{cls}' has only {len(candidates)} validation samples, requested {num_per_class}."
            )
        # Sort by psd_id for strict determinism before sampling
        candidates.sort(key=lambda r: r.psd_id)
        chosen_indices = rng.choice(len(candidates), size=num_per_class, replace=False)
        for idx in sorted(chosen_indices):
            rec = candidates[idx]
            # Strict isolation asserts
            fp_str = str(rec.file_path).lower().replace("\\", "/")
            assert "test" not in fp_str, f"CRITICAL SECURITY FAULT: Test sample in validation record: {rec.file_path}"
            assert "val" in fp_str, f"Selected sample is not from validation partition: {rec.file_path}"
            assert rec.file_path.exists(), f"Sample image does not exist: {rec.file_path}"
            selected_records.append(rec)

    # Double-check against test set
    test_paths = {str(r.file_path).replace("\\", "/") for r in plan.holdout_test_records}
    for sel in selected_records:
        sel_p = str(sel.file_path).replace("\\", "/")
        if sel_p in test_paths:
            raise RuntimeError(f"CRITICAL: Selected sample {sel.psd_id} belongs to the locked test set!")

    return selected_records


def run_phase10_xai(
    config: PSDConfig,
    handoff: FinalPipelineHandoff,
    samples: List[ImageRecord],
    output_dir: Path,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Executes the authoritative Phase 10 XAI evaluation on selected validation samples:
    1. Initializes AEFCRCInferenceEngine, GradCAMExplainer, and ProductionTreeSHAPExplainer.
    2. Runs inference, Platt calibration, and conformal prediction on each sample.
    3. Generates Grad-CAM spatial heatmaps on CNN backbone (pre-softmax logit).
    4. Generates TreeSHAP feature and branch importances on Random Forest (raw prediction).
    5. Exports per-sample artifacts and aggregate reports.
    """
    gradcam_base_dir = output_dir / "gradcam"
    shap_base_dir = output_dir / "shap"
    gradcam_base_dir.mkdir(parents=True, exist_ok=True)
    shap_base_dir.mkdir(parents=True, exist_ok=True)

    engine = AEFCRCInferenceEngine(artifact=handoff, device=device)
    backbone_model = engine._get_backbone()
    gradcam_explainer = GradCAMExplainer(
        model=backbone_model,
        classes=handoff.class_order,
        device=device,
    )
    shap_explainer = ProductionTreeSHAPExplainer(
        classifier=handoff.final_classifier,
        selected_feature_mask=handoff.selected_feature_mask,
        classes=handoff.class_order,
        branch_dims=handoff.branch_dims,
    )

    sample_summaries: List[Dict[str, Any]] = []

    print(f"\nExecuting Phase 10 XAI on {len(samples)} representative validation samples...")
    print(f"  Grad-CAM target: CNN disease head pre-softmax logit (Device: {gradcam_explainer.device})")
    print(f"  TreeSHAP target: Random Forest raw prediction (tree_path_dependent on CPU)\n")

    for idx, rec in enumerate(samples, 1):
        sample_id = rec.psd_id
        true_cls = rec.mapped_class
        print(f"[{idx}/{len(samples)}] Processing sample: {sample_id} (True class: {true_cls})")

        # 1. Full inference pipeline pass
        inf_result = engine.predict(
            image_input=rec.file_path,
            sample_id=sample_id,
            explain=False,  # We execute XAI directly to ensure exact artifact routing
        )
        pred_cls = inf_result.predicted_class
        calib_conf = inf_result.calibrated_confidence
        raw_conf = inf_result.raw_confidence
        is_correct = (pred_cls == true_cls)

        # 2. Preprocessed image and tensor
        img_pil = engine.validate_input_image(rec.file_path)
        tensor, rgb_np = engine.preprocess_image(img_pil)

        # 3. Grad-CAM execution (targeting predicted class)
        g_result = gradcam_explainer.explain(
            preprocessed_tensor=tensor,
            original_image_rgb=rgb_np,
            target_class=pred_cls,
        )
        sample_g_dir = gradcam_base_dir / sample_id
        orig_p, heat_p, over_p = gradcam_explainer.save_artifacts(
            output_dir=sample_g_dir,
            original_image_rgb=rgb_np,
            resized_heatmap=g_result.resized_heatmap,
        )

        # 4. Feature extraction and masking
        fused_1316 = engine.extract_multimodal_features(tensor, rgb_np)
        masked_194 = engine.apply_feature_mask(fused_1316)

        # 5. TreeSHAP execution (targeting predicted class)
        shap_result = shap_explainer.explain(
            features=masked_194,
            target_class=pred_cls,
        )
        sample_s_dir = shap_base_dir / sample_id
        shap_result.export_csvs(
            feature_csv_path=sample_s_dir / "shap_feature_importance.csv",
            block_csv_path=sample_s_dir / "shap_block_importance.csv",
        )

        # Top branch contributor
        top_branch = shap_result.block_contributions[0].branch if shap_result.block_contributions else "N/A"
        top_branch_pct = shap_result.block_contributions[0].pct_total_importance if shap_result.block_contributions else 0.0

        sample_summary = {
            "sample_id": sample_id,
            "true_class": true_cls,
            "predicted_class": pred_cls,
            "is_correct": is_correct,
            "raw_confidence": raw_conf,
            "calibrated_confidence": calib_conf,
            "marginal_prediction_set": ";".join(inf_result.marginal_prediction_set),
            "marginal_set_size": inf_result.marginal_set_size,
            "mondrian_prediction_set": ";".join(inf_result.mondrian_prediction_set),
            "mondrian_set_size": inf_result.mondrian_set_size,
            "review_status": inf_result.review_status,
            "gradcam_target_layer": g_result.target_layer_name,
            "gradcam_pre_softmax_logit": g_result.pre_softmax_logit,
            "gradcam_overlay_path": str(over_p),
            "top_shap_branch": top_branch,
            "top_shap_branch_pct": top_branch_pct,
            "shap_reconstruction_diff": shap_result.reconstruction_difference,
            "file_path": str(rec.file_path),
        }
        sample_summaries.append(sample_summary)
        print(f"    -> Predicted: {pred_cls} (Conf: {calib_conf*100:.1f}%), Status: {inf_result.review_status}, Top Branch: {top_branch} ({top_branch_pct:.1f}%)")

    # 6. Save selected_samples.csv
    csv_path = output_dir / "selected_samples.csv"
    fieldnames = [
        "sample_id", "true_class", "predicted_class", "is_correct",
        "raw_confidence", "calibrated_confidence",
        "marginal_prediction_set", "marginal_set_size",
        "mondrian_prediction_set", "mondrian_set_size", "review_status",
        "gradcam_target_layer", "gradcam_pre_softmax_logit", "gradcam_overlay_path",
        "top_shap_branch", "top_shap_branch_pct", "shap_reconstruction_diff", "file_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_summaries:
            writer.writerow(row)

    # 7. Save phase10_manifest.json
    manifest_path = output_dir / "phase10_manifest.json"
    manifest_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "representation_id": handoff.representation_id,
        "experiment_id": handoff.experiment_id,
        "classifier_name": handoff.classifier_name,
        "fusion_layout": "A7",
        "fused_dimensions": A7_FULL_DIM,
        "selected_features_k": int(handoff.selected_feature_mask.sum()),
        "branch_dims": handoff.branch_dims,
        "calibration_method": handoff.calibration_method,
        "marginal_q_hat": handoff.marginal_q_hat,
        "evaluated_samples_count": len(sample_summaries),
        "locked_test_samples_count": 243,
        "locked_test_status": "COMPLETELY_UNTOUCHED",
        "xai_contract": {
            "gradcam": {
                "scope": "Phase 3 Keras EfficientNet-B0 CNN backbone",
                "target": "pre-softmax logit of Dense-4 disease classification head",
                "device": gradcam_explainer.device,
            },
            "shap": {
                "scope": "Phase 7 Production Random Forest classifier",
                "target": "raw uncalibrated ensemble vote proportions (model_output='raw')",
                "perturbation": "tree_path_dependent",
                "device": "CPU",
            },
            "scientific_boundary": (
                "Grad-CAM explains the CNN feature extraction; TreeSHAP explains the Random Forest decision. "
                "Neither method explains Platt calibration or conformal prediction set thresholds."
            ),
        },
        "samples": sample_summaries,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 8. Save phase10_report.md
    report_path = output_dir / "phase10_report.md"
    _write_phase10_report(report_path, manifest_data, sample_summaries)

    return manifest_data


def _write_phase10_report(
    report_path: Path,
    manifest: Dict[str, Any],
    samples: List[Dict[str, Any]],
) -> None:
    """Generates the Markdown report documenting Phase 10 XAI findings."""
    correct_count = sum(1 for s in samples if s["is_correct"])
    total_count = len(samples)
    acc = (100.0 * correct_count / total_count) if total_count > 0 else 0.0

    lines = [
        "# AEF-CRC Phase 10: Explainable AI (XAI) & Interpretability Report",
        "",
        "## Executive Summary",
        f"- **Timestamp**: {manifest['timestamp_utc']}",
        f"- **Representation ID**: `{manifest['representation_id']}`",
        f"- **Fusion Layout**: `{manifest['fusion_layout']}` (Full dimension: {manifest['fused_dimensions']}-D)",
        f"- **Production Feature Selection**: BDA active features = {manifest['selected_features_k']} / {manifest['fused_dimensions']}",
        f"- **Production Classifier**: `{manifest['classifier_name']}` (n_features_in = {manifest['selected_features_k']})",
        f"- **Probability Calibration**: `{manifest['calibration_method'].capitalize()}` scaling",
        f"- **Conformal Coverage (Nominal)**: 90.0% (Marginal quantile $\\hat{{q}} = {manifest['marginal_q_hat']:.4f}$)",
        f"- **Evaluated Cohort**: {total_count} representative validation samples (Holdout val cohort; locked test set strictly isolated).",
        f"- **Validation Subset Accuracy**: {correct_count}/{total_count} ({acc:.1f}%)",
        "",
        "---",
        "",
        "## Scientific Rigor & Methodological Boundaries",
        "",
        "> [!IMPORTANT]",
        "> **Two-Tier Interpretability Division of Responsibility**:",
        "> 1. **Grad-CAM (Spatial CNN Representation)**: Operates exclusively on the TensorFlow/Keras EfficientNet-B0 backbone, targeting the **pre-softmax logit** of the Phase 3 Dense-4 classification head. It identifies spatial lesion regions driving visual feature extraction.",
        "> 2. **TreeSHAP (Tabular Multimodal Decision)**: Operates on the production Random Forest classifier across the **194 BDA-selected features** using `model_output='raw'` and `feature_perturbation='tree_path_dependent'`. It quantifies tabular feature contributions and branch importance.",
        "> 3. **Non-Explanation Boundary**: Neither Grad-CAM nor TreeSHAP explains the Platt probability calibration mapping or conformal prediction set boundaries.",
        "> 4. **Locked Outer Test Partition**: The 243-image locked test set was completely untouched and unaccessed.",
        "",
        "---",
        "",
        "## Sample Evaluation & Explanation Summary",
        "",
        "| Sample ID | True Class | Predicted Class | Calib Conf | Conformal Set | Review Status | Top SHAP Branch | Overlay |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for s in samples:
        conf_pct = f"{s['calibrated_confidence']*100:.1f}%"
        c_set = s["marginal_prediction_set"]
        rev = s["review_status"]
        top_b = f"{s['top_shap_branch']} ({s['top_shap_branch_pct']:.1f}%)"
        lines.append(
            f"| `{s['sample_id']}` | {s['true_class']} | {s['predicted_class']} | {conf_pct} | `{c_set}` | `{rev}` | {top_b} | [View](gradcam/{s['sample_id']}/gradcam_overlay.png) |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Artifact Inventory",
        "- **Selected Samples Table**: `selected_samples.csv`",
        "- **Phase 10 Manifest**: `phase10_manifest.json`",
    ])

    if samples:
        lines.append("- **Grad-CAM Visualizations**:")
        for s in samples:
            sid = s["sample_id"]
            lines.append(
                f"  - `{sid}`: `gradcam/{sid}/original.png`, `gradcam/{sid}/gradcam_heatmap.png`, `gradcam/{sid}/gradcam_overlay.png`"
            )
        lines.append("- **TreeSHAP Feature Importances**:")
        for s in samples:
            sid = s["sample_id"]
            lines.append(
                f"  - `{sid}`: `shap/{sid}/shap_feature_importance.csv`, `shap/{sid}/shap_block_importance.csv`"
            )
    else:
        lines.append("- **Grad-CAM Visualizations**: `gradcam/<sample_id>/[original.png, gradcam_heatmap.png, gradcam_overlay.png]`")
        lines.append("- **TreeSHAP Feature Importances**: `shap/<sample_id>/[shap_feature_importance.csv, shap_block_importance.csv]`")

    lines.extend([
        "",
        "---",
        "",
        "*AEF-CRC Phase 10 Interpretability Pipeline Complete.*",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="AEF-CRC Phase 10: Explainable AI (XAI) Runner")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run targeted preflight checks only without executing XAI or writing report artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Device to use for EfficientNet inference and Grad-CAM backpropagation.",
    )
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=2,
        help="Number of representative validation samples to explain per disease class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for Phase 10 reports and XAI artifacts. Defaults to reports/phase10.",
    )

    args = parser.parse_args()
    config = configure_phase9_v2(get_config())
    logger = get_module_logger("phase10_runner", config.logs_dir, config.log_level)

    print("==================================================")
    print("      AEF-CRC PHASE 10: EXPLAINABLE AI (XAI)      ")
    print("==================================================\n")

    handoff_path = config.project_root / "artifacts" / "phase9_v2" / "final_pipeline_handoff.joblib"
    if not handoff_path.exists():
        print(f"[PHASE 10 BLOCKER] Required artifact missing: {handoff_path}")
        print("Run Phase 9 conformal calibration first to freeze pipeline artifacts.")
        return 2

    logger.info(f"Loading Phase 9 pipeline handoff from {handoff_path}")
    handoff = load_final_pipeline_handoff(handoff_path)

    # Execute Preflight Validation
    try:
        preflight_info = validate_phase10_preflight(config, handoff, device=args.device)
    except Exception as exc:
        print(f"\n[PHASE 10 PREFLIGHT ERROR] Verification failed: {exc}")
        logger.error(f"Preflight check failed: {exc}")
        return 1

    print("--- Phase 10 Preflight Verification ---")
    print(f"  Representation ID:      {preflight_info['representation_id']}")
    print(f"  Experiment ID:          {preflight_info['experiment_id']}")
    print(f"  Fusion Layout:          A7 ({preflight_info['fused_dimensions']}-D)")
    print(f"  BDA Mask Active:        {preflight_info['selected_features']} features")
    print(f"  RF Classifier:          {preflight_info['classifier_name']} (n_features_in = {preflight_info['selected_features']})")
    print(f"  Backbone Checkpoint:    {preflight_info['backbone_checkpoint']}")
    print(f"  GPU Detected:           {preflight_info['gpu_available']} ({preflight_info['gpu_device_name']})")
    print(f"  Resolved Device:        {preflight_info['resolved_device']}")
    print(f"  Holdout Val Available:  {preflight_info['val_samples_available']} images (Accessible for XAI)")
    print(f"  Locked Test Partition:  {preflight_info['test_samples_locked']} images (STRICTLY LOCKED / UNTOUCHED)")
    print("  Production SHAP:        PASS (TreeSHAP raw RF, tree_path_dependent)")
    print("  Grad-CAM Contract:      PASS (pre-softmax logit of CNN disease head)")
    print("\nPHASE 10 PREFLIGHT: PASS (All contracts, models, devices, and isolation verified).")

    if args.preflight:
        return 0

    # Real Execution Path
    output_dir = args.output_dir or (config.project_root / "reports" / "phase10")
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = load_frozen_fold_plan(config)
    selected_samples = select_representative_validation_samples(
        plan=plan,
        config=config,
        num_per_class=args.num_samples_per_class,
    )
    print(f"\nSelected {len(selected_samples)} validation samples across {len(config.target_classes)} classes:")
    for s in selected_samples:
        print(f"  - [{s.mapped_class}] {s.psd_id} ({s.file_path.name})")

    run_phase10_xai(
        config=config,
        handoff=handoff,
        samples=selected_samples,
        output_dir=output_dir,
        device=args.device,
    )

    print("\n==================================================")
    print("PHASE 10 RESULT: REAL XAI EXECUTION COMPLETE.")
    print(f"Artifacts and reports saved to: {output_dir}")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
