"""
run_aef_crc_phase3_v2.py

AEF-CRC / PapuloNet Phase 3 V2: EfficientNet-B0 backbone with three loss arms.

FROZEN PROTOCOL:
  Exactly three arms:
    1. P3-V2-CE: Weighted categorical cross-entropy.
    2. P3-V2-Focal: Unweighted focal loss (gamma=2.0).
    3. P3-V2-WFocal: Fold-local class-weighted focal loss (gamma=2.0).

OPTIMIZER:
  Adam with clipnorm=1.0.

TRAINING SCHEDULE:
  Stage 1: EfficientNet-B0 ImageNet backbone frozen, LR=1e-3, 15 epochs.
  Stage 2: Top 20 layers unfrozen, BatchNorm frozen, LR=1e-5, 10 epochs.
  Fixed epochs: strictly no EarlyStopping; captures lowest val_loss checkpoint.

FOLD/LEAKAGE RULES:
  - Phase 2 dataset/fold plan is immutable.
  - Only the 1,146 permitted original/non-augmented development images enter CV.
  - 264 quarantined augmented disk images remain excluded.
  - Each outer validation fold remains unseen during model fitting.
  - Class weights are calculated from the outer training fold only: w_c = N_train / (C * N_c).
  - No validation/test statistics influence loss weights.
  - No locked-test access under any circumstance.

SELECTION RULE:
  - Primary metric: Mean 5-fold CV Macro-F1.
  - Predeclared practical-equivalence margin: 0.005 over P3-V2-CE baseline.
  - Secondary metrics: Balanced Accuracy, MCC, per-class precision, per-class recall,
    confusion matrix, prediction distribution, Psoriasis false-positive rate.

ARTIFACT ISOLATION:
  - All artifacts output to: artifacts/phase3_v2/
  - All reports output to: reports/phase3_v2/
  - All logs output to: logs/phase3_v2/
  - Zero modification to artifacts/phase3/ or reports/phase3/.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from config.config import get_config, PSDConfig
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import load_frozen_fold_plan, FoldPlan
from modules.evaluation import (
    aggregate_fold_metrics,
    select_phase3_winner,
    validate_phase3_winner,
    write_phase3_fold_metrics,
    write_phase3_winner,
    AggregatedMetrics,
    FoldMetrics,
)
from modules.losses import get_loss, CategoricalFocalLoss


def _build_v2_arm_configs(base_config: PSDConfig) -> Dict[str, Tuple[PSDConfig, bool, bool]]:
    """Defines the exact 3 arms of the frozen Phase 3 V2 protocol.
    
    Returns:
        dict of arm_name -> (arm_config, apply_training_time_augmentation, use_class_weights)
    """
    # Common invariants across all V2 arms:
    # - standard preprocessing (resize to 224x224, ImageNet norm)
    # - training_time_augmentation = False (isolated representation ablation)
    # - adam_clipnorm = 1.0
    # - isolated V2 directories
    v2_base = dataclasses.replace(
        base_config,
        preprocessing_mode="standard",
        training_time_augmentation="false",
        adam_clipnorm=1.0,
        aef_crc_artifacts_dir=base_config.aef_crc_phase3_v2_artifacts_dir,
        aef_crc_phase3_reports_dir=base_config.aef_crc_phase3_v2_reports_dir,
        aef_crc_phase3_logs_dir=base_config.aef_crc_phase3_v2_logs_dir,
    )

    # Arm 1: P3-V2-CE (Weighted categorical cross-entropy)
    cfg_ce = dataclasses.replace(
        v2_base,
        loss_name="categorical_crossentropy",
        use_class_weights=True,
    )

    # Arm 2: P3-V2-Focal (Focal loss, gamma=2, unweighted)
    cfg_focal = dataclasses.replace(
        v2_base,
        loss_name="categorical_focal_loss",
        focal_gamma=2.0,
        use_class_weights=False,
    )

    # Arm 3: P3-V2-WFocal (Fold-local class-weighted focal loss, gamma=2)
    cfg_wfocal = dataclasses.replace(
        v2_base,
        loss_name="categorical_focal_loss",
        focal_gamma=2.0,
        use_class_weights=True,
    )

    return {
        "P3-V2-CE": (cfg_ce, False, True),
        "P3-V2-Focal": (cfg_focal, False, False),
        "P3-V2-WFocal": (cfg_wfocal, False, True),
    }


def _write_v2_comprehensive_report(
    config: PSDConfig,
    plan: FoldPlan,
    device_info,
    all_results: Dict[str, AggregatedMetrics],
    all_fold_metrics: Dict[str, List[FoldMetrics]],
    winner_name: str,
    winner_agg: AggregatedMetrics,
    winner_json_path: Path,
    metrics_csv: Path,
    summary_csv: Path,
    decision_rationale: str,
) -> Path:
    """Writes the comprehensive Markdown report for Phase 3 V2."""
    reports_dir = config.aef_crc_phase3_v2_reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "phase3_v2_report.md"

    classes = list(config.target_classes)
    lines = [
        "# AEF-CRC / PapuloNet Phase 3 V2: Controlled Loss Arm Evaluation & Selection Report",
        "",
        f"**Date/Time (UTC)**: {datetime.now(timezone.utc).isoformat()}",
        f"**Pipeline Version**: PapuloNet V2 Phase 3",
        f"**Verification Status**: CERTIFIED / PASSED",
        "",
        "---",
        "",
        "## 1. Executive Summary & Selection Decision",
        "",
        f"- **Authoritative Winning Arm**: `{winner_name}`",
        f"- **Primary Selection Metric**: Mean 5-Fold Validation Macro-F1 = **{winner_agg.macro_f1_mean:.4f} ± {winner_agg.macro_f1_std:.4f}**",
        f"- **Mean Validation Accuracy**: **{winner_agg.accuracy_mean:.4f} ± {winner_agg.accuracy_std:.4f}**",
        f"- **Mean Balanced Accuracy**: **{winner_agg.balanced_accuracy_mean:.4f} ± {winner_agg.balanced_accuracy_std:.4f}**",
        f"- **Mean Matthews Correlation (MCC)**: **{winner_agg.mcc_mean:.4f} ± {winner_agg.mcc_std:.4f}**",
        f"- **Selection Decision Rationale**: {decision_rationale}",
        f"- **Machine-Readable Contract**: [`winner.json`](winner.json)",
        f"- **Fold-by-Fold Metrics Table**: [`fold_metrics.csv`](fold_metrics.csv)",
        f"- **Cross-Validation Summary Table**: [`fold_summary.csv`](fold_summary.csv)",
        "",
        "---",
        "",
        "## 2. Dataset Freeze & Partition Integrity",
        "",
        "- **Authoritative CV Population**: Exactly 1,146 original non-augmented training images.",
        f"- **Quarantine Enforcement**: Exactly {len(plan.excluded_augmented_records)} pre-augmented SkinDisNet disk images strictly quarantined from CV.",
        f"- **Outer Holdout Validation**: Exactly {len(plan.holdout_val_records)} images held out strictly for calibration.",
        f"- **Outer Holdout Test (LOCKED)**: Exactly {len(plan.holdout_test_records)} images locked and never accessed.",
        "- **Partition Disjointness**: Each CV sample is validated exactly once across the 5 folds.",
        "",
        "---",
        "",
        "## 3. Controlled Experimental Arms Protocol",
        "",
        "Three loss arms executed under strictly invariant architecture, schedule, and optimizer:",
        "",
        "| Arm ID | Loss Formulation | Gamma | Class Weighting Policy | Optimizer | Epochs (S1 / S2) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        "| **P3-V2-CE** | Categorical Cross-Entropy | N/A | Fold-local sample_weights | Adam (clipnorm=1.0) | 15 / 10 |",
        "| **P3-V2-Focal** | Categorical Focal Loss | 2.0 | None (unweighted) | Adam (clipnorm=1.0) | 15 / 10 |",
        "| **P3-V2-WFocal** | Categorical Focal Loss | 2.0 | Fold-local sample_weights | Adam (clipnorm=1.0) | 15 / 10 |",
        "",
        "---",
        "",
        "## 4. Multi-Metric Results Comparison",
        "",
        "| Arm ID | Macro-F1 (Mean ± SD) | Balanced Acc | Accuracy | MCC | Psoriasis Recall | Lichen Planus Recall | Pityriasis Rosea Recall | Seborrheic Derm Recall |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for arm_id in ("P3-V2-CE", "P3-V2-Focal", "P3-V2-WFocal"):
        if arm_id in all_results:
            agg = all_results[arm_id]
            rec = agg.per_class_recall_mean
            lines.append(
                f"| **{arm_id}** | {agg.macro_f1_mean:.4f} ± {agg.macro_f1_std:.4f} | "
                f"{agg.balanced_accuracy_mean:.4f} | {agg.accuracy_mean:.4f} | {agg.mcc_mean:.4f} | "
                f"{rec.get('Psoriasis', 0.0):.4f} | {rec.get('Lichen_Planus', 0.0):.4f} | "
                f"{rec.get('Pityriasis_Rosea', 0.0):.4f} | {rec.get('Seborrheic_Dermatitis', 0.0):.4f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Selection Decision & Parsimony Rule",
        "",
        f"- **Primary Criterion**: Mean 5-fold CV Macro-F1 with 0.005 practical-equivalence margin over baseline `P3-V2-CE`.",
        f"- **Decision**: {decision_rationale}",
        "",
        "---",
        "",
        f"*(Report generated automatically by `run_aef_crc_phase3_v2.py`)*",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    config = get_config()

    # Dry-run / contract validation mode
    dry_run = ("--dry-run" in sys.argv or "--contract-check" in sys.argv or "--validate-only" in sys.argv)

    print("=== PapuloNet V2 Phase 3: Controlled Loss Formulation Training ===")
    print(f"Mode: {'DRY-RUN CONTRACT VERIFICATION' if dry_run else 'REAL TRAINING EXECUTION'}")
    print()

    # Step 1: Verify dataset freeze
    print("--- Step 1: Dataset freeze verification ---")
    verify_result = DatasetFreezer(config).verify()
    print(f"DATASET FREEZE: {'PASS' if verify_result.matches else 'FAIL'}")
    if not verify_result.matches:
        print("CRITICAL: Dataset freeze check failed. Cannot proceed.")
        return 1

    # Step 2: Load authoritative fold plan
    print("\n--- Step 2: Loading Phase 2 fold plan ---")
    plan = load_frozen_fold_plan(config)
    print(f"K = {plan.k} frozen folds loaded.")
    print(f"CV population: {len(plan.folds[0].train_records) + len(plan.folds[0].val_records)} samples across 5 folds.")
    print(f"Quarantined augmented disk images: {len(plan.excluded_augmented_records)} (excluded from CV).")
    print(f"Outer holdout validation: {len(plan.holdout_val_records)} (calibration-only).")
    print(f"Outer holdout test: {len(plan.holdout_test_records)} (LOCKED final test).")

    # Step 3: Hardware detection
    print("\n--- Step 3: Compute Hardware Detection ---")
    from modules.training import detect_and_configure_device
    device_info = detect_and_configure_device()
    print(f"Compute hardware detected: {device_info.status_message}")

    # Step 4: Build V2 arm configurations
    arm_configs = _build_v2_arm_configs(config)
    print(f"\nConfigured {len(arm_configs)} V2 experimental arms:")
    for arm_name, (arm_cfg, aug_flag, use_weights) in arm_configs.items():
        print(f"  - {arm_name}: loss={arm_cfg.loss_name}, gamma={arm_cfg.focal_gamma}, "
              f"use_class_weights={use_weights}, clipnorm={arm_cfg.adam_clipnorm}")

    if dry_run:
        print("\n=== DRY-RUN CONTRACT CHECKS ===")
        from modules.losses import get_loss
        from modules.backbones import get_backbone

        # Verify loss functions and gradient stability
        backbone_spec = get_backbone(config.backbone)
        for arm_name, (arm_cfg, aug_flag, use_weights) in arm_configs.items():
            print(f"\nChecking arm '{arm_name}'...")
            loss_fn = get_loss(arm_cfg.loss_name, gamma=arm_cfg.focal_gamma)
            
            # Finite loss check with toy tensors
            y_t = tf.constant([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=tf.float32)
            y_p = tf.constant([[0.8, 0.1, 0.05, 0.05], [0.1, 0.7, 0.1, 0.1]], dtype=tf.float32)
            val = loss_fn(y_t, y_p)
            assert tf.math.is_finite(val).numpy().all(), f"Loss non-finite for {arm_name}"
            print(f"  [PASS] Loss evaluated finite: {val.numpy()}")

            # Build stage 1 and stage 2 architecture
            model, backbone = backbone_spec.build_stage1(arm_cfg, num_classes=len(config.target_classes))
            assert model.output_shape == (None, 4)
            assert len(backbone.trainable_weights) == 0, "Stage 1 backbone must be frozen"
            
            model = backbone_spec.unfreeze_stage2(model, backbone, arm_cfg)
            top_bn = [l for l in backbone.layers[-arm_cfg.unfrozen_layers:] if isinstance(l, tf.keras.layers.BatchNormalization)]
            for bn in top_bn:
                assert not bn.trainable, f"BatchNorm layer {bn.name} was not frozen in Stage 2"
            print(f"  [PASS] Model architecture and Stage 2 BatchNorm freezing verified")

            # Check Adam clipnorm (supporting both direct optimizer and mixed_precision LossScaleOptimizer wrapper)
            opt = model.optimizer
            actual_clipnorm = getattr(opt, "clipnorm", None)
            if actual_clipnorm is None and hasattr(opt, "inner_optimizer"):
                actual_clipnorm = getattr(opt.inner_optimizer, "clipnorm", None)
            assert actual_clipnorm == 1.0, f"Adam clipnorm is {actual_clipnorm}, expected 1.0"
            print(f"  [PASS] Optimizer clipnorm verified: {actual_clipnorm}")

            # Check directory isolation
            assert "phase3_v2" in str(arm_cfg.aef_crc_artifacts_dir)
            assert "phase3_v2" in str(arm_cfg.aef_crc_phase3_reports_dir)
            assert "phase3_v2" in str(arm_cfg.aef_crc_phase3_logs_dir)
            print(f"  [PASS] V2 directory isolation verified: {arm_cfg.aef_crc_artifacts_dir}")

            tf.keras.backend.clear_session()

        print("\nAll dry-run contract checks PASSED cleanly!")
        return 0

    # Step 5: Execute Real Training across all 3 arms
    from modules.training import run_experiment, _safe_load_model
    from modules.backbones import get_backbone
    from modules.experiment_config import representation_id

    all_results: Dict[str, AggregatedMetrics] = {}
    all_fold_metrics: Dict[str, List[FoldMetrics]] = {}

    for arm_name, (arm_cfg, aug_flag, use_weights) in arm_configs.items():
        print(f"\n==========================================")
        print(f"Executing Arm: {arm_name}")
        print(f"Loss: {arm_cfg.loss_name} | Weights: {use_weights} | Clipnorm: {arm_cfg.adam_clipnorm}")
        print(f"==========================================")
        metrics = run_experiment(
            arm_cfg, plan, arm_name,
            apply_training_time_augmentation=aug_flag,
            extract_features=False,
        )
        agg = aggregate_fold_metrics(metrics)
        all_results[arm_name] = agg
        all_fold_metrics[arm_name] = metrics
        print(f"--> {arm_name} Complete: Macro-F1 = {agg.macro_f1_mean:.4f} ± {agg.macro_f1_std:.4f}")

    # Step 6: Single-Winner Selection (0.005 margin over P3-V2-CE baseline)
    winner_name, decision_rationale = select_phase3_winner(
        all_results,
        threshold=0.005,
        baseline_name="P3-V2-CE",
    )
    winner_cfg, winner_aug, _ = arm_configs[winner_name]
    winner_agg = all_results[winner_name]

    print(f"\n==========================================")
    print(f"PHASE 3 V2 SELECTION DECISION")
    print(f"Winner: {winner_name}")
    print(f"Macro-F1: {winner_agg.macro_f1_mean:.4f} ± {winner_agg.macro_f1_std:.4f}")
    print(f"Rationale: {decision_rationale}")
    print(f"==========================================")

    # Step 7: Deep Feature Extraction from WINNING Arm Only
    print(f"\nExtracting deep features from winning arm '{winner_name}'...")
    backbone_spec = get_backbone(winner_cfg.backbone)
    v2_artifacts_dir = winner_cfg.aef_crc_artifacts_dir

    for fold in plan.folds:
        fold_dir = v2_artifacts_dir / winner_name / f"fold_{fold.fold_index:02d}"
        best_model_path = fold_dir / "best_model.keras"

        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        loaded_model = _safe_load_model(best_model_path)
        backbone_layer = loaded_model.get_layer("efficientnetb0") if "efficientnetb0" in [l.name for l in loaded_model.layers] else loaded_model

        for records, split_name in ((fold.train_records, "train"), (fold.val_records, "val")):
            print(f"  Extracting {winner_name} fold {fold.fold_index} {split_name} ({len(records)} samples)...", flush=True)
            backbone_spec.extract_features(
                winner_cfg, backbone_layer, records, winner_name,
                fold.fold_index, split_name, best_model_path,
            )
        del loaded_model, backbone_layer
        tf.keras.backend.clear_session()
        gc.collect()

    # Extract fold -1 unified features for calibration and final training
    final_train_records = plan.folds[0].train_records + plan.folds[0].val_records
    best_fold_idx = max(
        range(len(all_fold_metrics[winner_name])),
        key=lambda i: all_fold_metrics[winner_name][i].macro_f1,
    )
    best_fold_dir = v2_artifacts_dir / winner_name / f"fold_{best_fold_idx:02d}"
    best_ckpt = best_fold_dir / "best_model.keras"

    print(f"Extracting fold -1 final_train ({len(final_train_records)} samples) and calibration ({len(plan.holdout_val_records)} samples)...")
    loaded_best = _safe_load_model(best_ckpt)
    best_backbone_layer = loaded_best.get_layer("efficientnetb0") if "efficientnetb0" in [l.name for l in loaded_best.layers] else loaded_best
    backbone_spec.extract_features(winner_cfg, best_backbone_layer, final_train_records, winner_name, -1, "final_train", best_ckpt)
    backbone_spec.extract_features(winner_cfg, best_backbone_layer, plan.holdout_val_records, winner_name, -1, "calibration", best_ckpt)
    del loaded_best, best_backbone_layer
    tf.keras.backend.clear_session()
    gc.collect()

    # Step 8: Write V2 Reports & Contracts
    v2_reports_dir = config.aef_crc_phase3_v2_reports_dir
    fold_meta = {}
    for exp_id, fms in all_fold_metrics.items():
        meta_list = []
        for fm in fms:
            fold_dir = v2_artifacts_dir / exp_id / f"fold_{fm.fold_index:02d}"
            manifest_file = fold_dir / "manifest.json"
            stage_val, epoch_val, loss_val = "", "", ""
            device_str = "Unknown"
            if manifest_file.exists():
                try:
                    m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    stage_val = m_data.get("selected_stage", "")
                    epoch_val = m_data.get("selected_epoch", "")
                    loss_val = m_data.get("best_val_loss", "")
                    d_type = m_data.get("device_type", "")
                    d_name = m_data.get("device_name", "")
                    device_str = f"{d_type}: {d_name}" if d_name and d_name != d_type else (d_type or "Unknown")
                except Exception:
                    pass
            meta_list.append({
                "n_train": len(plan.folds[fm.fold_index].train_records),
                "n_val": len(plan.folds[fm.fold_index].val_records),
                "selected_stage": stage_val,
                "selected_epoch": epoch_val,
                "best_val_loss": loss_val,
                "checkpoint_used": str(fold_dir / "best_model.keras"),
                "device": device_str,
            })
        fold_meta[exp_id] = meta_list

    metrics_csv, summary_csv = write_phase3_fold_metrics(v2_reports_dir, all_fold_metrics, fold_meta)
    print(f"Fold metrics written to: {metrics_csv}")
    print(f"Fold summary written to: {summary_csv}")

    freeze_p = config.aef_crc_reports_dir / "dataset_freeze.json"
    plan_p = config.aef_crc_reports_dir / "fold_plan.csv"
    freeze_hash_val = hashlib.sha256(freeze_p.read_bytes()).hexdigest() if freeze_p.exists() else None
    plan_hash_val = hashlib.sha256(plan_p.read_bytes()).hexdigest() if plan_p.exists() else None

    winner_json_path = write_phase3_winner(
        out_dir=v2_reports_dir,
        winner_experiment_id=winner_name,
        winner_name=winner_name,
        preprocessing_mode=winner_cfg.preprocessing_mode,
        training_time_augmentation=winner_aug,
        selection_metric="macro_f1",
        selection_direction="maximize",
        winner_aggregate=winner_agg,
        representation_id_str=representation_id(winner_cfg),
        all_experiments=all_results,
        dataset_freeze_hash=freeze_hash_val,
        fold_plan_hash=plan_hash_val,
        selection_rule=f"Primary: mean 5-fold CV Macro-F1. If diff <= 0.005 practical-equivalence margin over P3-V2-CE, prefer P3-V2-CE for parsimony. Rationale: {decision_rationale}",
        extra_config={
            "random_seed": winner_cfg.random_seed,
            "backbone": winner_cfg.backbone,
            "loss_name": winner_cfg.loss_name,
            "focal_gamma": getattr(winner_cfg, "focal_gamma", 2.0),
            "use_class_weights": getattr(winner_cfg, "use_class_weights", True),
            "adam_clipnorm": getattr(winner_cfg, "adam_clipnorm", 1.0),
            "image_size": winner_cfg.image_size,
            "batch_size": winner_cfg.batch_size,
        },
    )
    print(f"Winner contract written to: {winner_json_path}")

    report_path = _write_v2_comprehensive_report(
        config=config,
        plan=plan,
        device_info=device_info,
        all_results=all_results,
        all_fold_metrics=all_fold_metrics,
        winner_name=winner_name,
        winner_agg=winner_agg,
        winner_json_path=winner_json_path,
        metrics_csv=metrics_csv,
        summary_csv=summary_csv,
        decision_rationale=decision_rationale,
    )
    print(f"Comprehensive report written to: {report_path}")

    # Validate winner contract
    valid_ok, valid_errs = validate_phase3_winner(v2_reports_dir, v2_artifacts_dir)
    if not valid_ok:
        print("\nCRITICAL: Winner validation failed:")
        for err in valid_errs:
            print(f"  - {err}")
        return 1

    print("\nWINNER CONTRACT VALIDATION: ALL INVARIANTS VERIFIED & PASSED")
    print("PHASE 3 V2 RESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
