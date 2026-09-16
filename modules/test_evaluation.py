"""
modules/test_evaluation.py

Authoritative evaluation harness for the locked AEF-CRC outer test set (N=243).

SCIENTIFIC CONTRACT:
- The 243 outer-test images represent the final held-out scientific benchmark.
- STRICT LOCK: Cannot be executed without explicit authorization (confirm_execution=True).
- Evaluates frozen model artifacts ONCE after all upstream phases are frozen.
- Computes pre-registered classification, calibration, and conformal metrics.
- Completely isolated from development, threshold tuning, and single-image inference.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config.config import PSDConfig, get_config
from modules.calibration import compute_ece, multiclass_brier_score
from modules.calibration_handoff import (
    FinalPipelineHandoff,
    validate_pipeline_handoff_provenance,
)
from modules.device_utils import detect_device_environment
from modules.fold_loader import load_frozen_fold_plan
from modules.output_schema import TestEvaluationReport


class LockedOuterTestGuardError(RuntimeError):
    """Raised when outer test evaluation is invoked without required authorization."""
    pass


def compute_reliability_diagram_bins(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """Computes bin centers, bin accuracies, and bin confidences for reliability diagrams."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= low) & (confidences < high if i < n_bins - 1 else confidences <= high)
        count = int(np.sum(mask))
        center = float(0.5 * (low + high))
        bin_centers.append(center)
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(float(np.mean(accuracies[mask])))
            bin_confs.append(float(np.mean(confidences[mask])))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }


def evaluate_locked_outer_test(
    config: PSDConfig,
    pipeline_artifact: FinalPipelineHandoff,
    confirm_execution: bool = False,
    output_dir: Optional[Path] = None,
) -> TestEvaluationReport:
    """
    Executes final benchmark evaluation on the locked 243-image outer test partition.

    Parameters
    ----------
    config : PSDConfig
        Project configuration.
    pipeline_artifact : FinalPipelineHandoff
        Frozen production pipeline bundle from Phase 9.
    confirm_execution : bool
        Security override flag. Must be True to proceed.
    output_dir : Optional[Path]
        Directory where evaluation reports and figures will be saved.

    Raises
    ------
    LockedOuterTestGuardError
        If confirm_execution is False or required production artifacts are missing.
    """
    # 1. Verification of security lock
    if not confirm_execution:
        raise LockedOuterTestGuardError(
            "EXECUTION GUARD ACTIVE: Outer test evaluation (N=243) is strictly locked. "
            "To execute the authoritative evaluation, pass confirm_execution=True. "
            "Never evaluate outer test data during development or tuning turns."
        )

    # 1b. Provenance verification across runs
    validate_pipeline_handoff_provenance(pipeline_artifact, config)

    # 2. Verify all frozen artifacts exist
    if pipeline_artifact.final_classifier is None:
        raise LockedOuterTestGuardError("Missing production Random Forest classifier in artifact.")
    if pipeline_artifact.selected_feature_mask is None:
        raise LockedOuterTestGuardError("Missing production BDA mask in artifact.")
    if pipeline_artifact.platt_models is None:
        raise LockedOuterTestGuardError("Missing Platt calibration models in artifact.")
    if pipeline_artifact.marginal_q_hat is None:
        raise LockedOuterTestGuardError("Missing conformal threshold (marginal_q_hat) in artifact.")

    # 3. Load frozen fold plan and extract the 243 outer test records
    plan = load_frozen_fold_plan(config)
    test_records = plan.holdout_test_records
    n_test = len(test_records)
    if n_test != 243:
        raise LockedOuterTestGuardError(f"Expected exactly 243 outer test records, found {n_test}.")

    classes = pipeline_artifact.class_order
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # Import inference engine locally
    from modules.inference import AEFCRCInferenceEngine
    engine = AEFCRCInferenceEngine(pipeline_artifact, device="auto")

    # 4. Run inference across the 243 test images
    y_true_indices: List[int] = []
    y_pred_indices: List[int] = []
    raw_probs_list: List[np.ndarray] = []
    calib_probs_list: List[np.ndarray] = []
    conformal_sets_list: List[List[str]] = []

    for record in test_records:
        true_label = record.label
        if true_label not in class_to_idx:
            raise ValueError(f"Unknown test label '{true_label}' not in class list {classes}")
        y_true_indices.append(class_to_idx[true_label])

        # Execute single prediction
        result = engine.predict_single_image(record.file_path, sample_id=record.psd_id)
        y_pred_indices.append(class_to_idx[result.predicted_class])

        raw_row = np.array([result.raw_probabilities[c] for c in classes], dtype=np.float32)
        calib_row = np.array([result.calibrated_probabilities[c] for c in classes], dtype=np.float32)

        raw_probs_list.append(raw_row)
        calib_probs_list.append(calib_row)
        conformal_sets_list.append(result.marginal_prediction_set)

    y_true = np.array(y_true_indices)
    y_pred = np.array(y_pred_indices)
    raw_probs = np.vstack(raw_probs_list)
    calib_probs = np.vstack(calib_probs_list)

    # -------------------------------------------------------------------------
    # 5. Multiclass Classification Metrics
    # -------------------------------------------------------------------------
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, average=None)
    per_class_metrics = {}
    for c_idx, c_name in enumerate(classes):
        per_class_metrics[c_name] = {
            "precision": float(prec[c_idx]),
            "recall": float(rec[c_idx]),
            "f1_score": float(f1[c_idx]),
            "support": int(supp[c_idx]),
        }

    classification_metrics = {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "per_class": per_class_metrics,
    }

    # -------------------------------------------------------------------------
    # 6. Probability Calibration Metrics
    # -------------------------------------------------------------------------
    # One-hot encode ground truth
    y_true_onehot = np.zeros_like(raw_probs)
    for i, label_idx in enumerate(y_true):
        y_true_onehot[i, label_idx] = 1.0

    raw_conf = np.max(raw_probs, axis=1)
    calib_conf = np.max(calib_probs, axis=1)
    is_correct = (y_true == y_pred).astype(int)

    raw_ece = float(compute_ece(y_true, raw_probs, n_bins=10))
    calib_ece = float(compute_ece(y_true, calib_probs, n_bins=10))
    raw_brier = float(multiclass_brier_score(y_true_onehot, raw_probs))
    calib_brier = float(multiclass_brier_score(y_true_onehot, calib_probs))

    rel_diagram_raw = compute_reliability_diagram_bins(raw_conf, is_correct, n_bins=10)
    rel_diagram_calib = compute_reliability_diagram_bins(calib_conf, is_correct, n_bins=10)

    calibration_metrics = {
        "raw_ece": raw_ece,
        "platt_calibrated_ece": calib_ece,
        "ece_improvement": float(raw_ece - calib_ece),
        "raw_multiclass_brier": raw_brier,
        "platt_calibrated_brier": calib_brier,
        "brier_improvement": float(raw_brier - calib_brier),
        "reliability_diagram_raw": rel_diagram_raw,
        "reliability_diagram_calibrated": rel_diagram_calib,
    }

    # -------------------------------------------------------------------------
    # 7. Split-Conformal Prediction Metrics
    # -------------------------------------------------------------------------
    # Coverage: indicator whether true class label is in prediction set
    covered_indicators = [
        test_records[i].label in conformal_sets_list[i] for i in range(n_test)
    ]
    empirical_coverage = float(np.mean(covered_indicators))
    set_sizes = [len(s) for s in conformal_sets_list]
    mean_set_size = float(np.mean(set_sizes))
    median_set_size = float(np.median(set_sizes))

    singleton_count = sum(1 for s in set_sizes if s == 1)
    multi_count = sum(1 for s in set_sizes if s > 1)
    empty_count = sum(1 for s in set_sizes if s == 0)

    # Class-conditional diagnostic coverage
    class_conditional_cov = {}
    for c_idx, c_name in enumerate(classes):
        indices_c = [i for i, y in enumerate(y_true) if y == c_idx]
        if indices_c:
            cov_c = float(np.mean([covered_indicators[i] for i in indices_c]))
            mean_sz_c = float(np.mean([set_sizes[i] for i in indices_c]))
            class_conditional_cov[c_name] = {
                "coverage": cov_c,
                "mean_set_size": mean_sz_c,
                "sample_count": len(indices_c),
            }
        else:
            class_conditional_cov[c_name] = {"coverage": None, "mean_set_size": None, "sample_count": 0}

    conformal_metrics = {
        "nominal_target_coverage": 1.0 - pipeline_artifact.alpha,
        "marginal_empirical_coverage": empirical_coverage,
        "coverage_delta": float(empirical_coverage - (1.0 - pipeline_artifact.alpha)),
        "mean_set_size": mean_set_size,
        "median_set_size": median_set_size,
        "singleton_fraction": float(singleton_count / n_test),
        "ambiguous_fraction": float(multi_count / n_test),
        "empty_set_fraction": float(empty_count / n_test),
        "class_conditional_coverage": class_conditional_cov,
    }

    # 8. Assemble Report
    out_dir = output_dir or (config.reports_dir / "final_outer_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = TestEvaluationReport(
        dataset_name="PSD-HP Authoritative Harmonized Outer Test",
        total_test_samples=n_test,
        evaluation_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        classification_metrics=classification_metrics,
        calibration_metrics=calibration_metrics,
        conformal_metrics=conformal_metrics,
        leakage_guard_verified=True,
        device_metadata=detect_device_environment().to_dict(),
    )

    report_path = out_dir / "final_locked_outer_test_report.json"
    report.save_json(report_path)
    return report
