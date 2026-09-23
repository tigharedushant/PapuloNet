"""
modules/evaluation.py

AEF-CRC Phase 3: Metrics.

Deliberately framework-agnostic: every function here takes plain
numpy arrays of true/predicted labels (and optionally predicted
probabilities), never a TensorFlow model or tensor directly. That
means this module works identically whether the predictions came
from EfficientNet-B0, a downstream Random Forest stage, or a synthetic array in
a test -- and it means this module's correctness can be verified
completely independently of whether TensorFlow is even installed.

Primary metric per Part 17: Macro-F1. Reported first, not buried
after accuracy -- Part 17 is explicit that accuracy alone must never
be the headline metric for an imbalanced 4-class problem like this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timezone
import csv
import json
import uuid

import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    precision_recall_fscore_support, confusion_matrix,
)


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class FoldMetrics:
    fold_index: int
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    weighted_f1: float
    mcc: float
    per_class: Dict[str, ClassMetrics]
    confusion: np.ndarray
    class_order: List[str]


def compute_fold_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], class_order: List[str], fold_index: int
) -> FoldMetrics:
    """class_order fixes label ordering for both per-class metrics and
    the confusion matrix -- passed explicitly rather than inferred from
    the data, so a class with zero predictions in this fold still gets
    an explicit (zero) row/column instead of silently vanishing."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        raise ValueError("compute_fold_metrics called with zero samples -- nothing to evaluate.")

    macro_f1 = f1_score(y_true, y_pred, labels=class_order, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=class_order, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=class_order, zero_division=0
    )
    per_class = {
        cls: ClassMetrics(precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for cls, p, r, f, s in zip(class_order, precisions, recalls, f1s, supports)
    }

    cm = confusion_matrix(y_true, y_pred, labels=class_order)

    return FoldMetrics(
        fold_index=fold_index, macro_f1=float(macro_f1), accuracy=float(accuracy),
        balanced_accuracy=float(balanced_acc), weighted_f1=float(weighted_f1), mcc=float(mcc),
        per_class=per_class, confusion=cm, class_order=class_order,
    )


@dataclass
class AggregatedMetrics:
    """Mean +/- std across folds -- never a single 'best fold' result.
    Part 16/23: 'Do NOT choose a single best fold.'"""
    n_folds: int
    macro_f1_mean: float
    macro_f1_std: float
    accuracy_mean: float
    accuracy_std: float
    balanced_accuracy_mean: float
    balanced_accuracy_std: float
    weighted_f1_mean: float
    weighted_f1_std: float
    mcc_mean: float
    mcc_std: float
    per_class_f1_mean: Dict[str, float] = field(default_factory=dict)
    per_class_f1_std: Dict[str, float] = field(default_factory=dict)
    # Added (AEF-CRC final evaluation phase): precision/recall were
    # already computed per fold in ClassMetrics but never aggregated
    # across folds -- only F1 was. Task explicitly asks for per-class
    # Precision/Recall/F1/support, not F1 alone.
    per_class_precision_mean: Dict[str, float] = field(default_factory=dict)
    per_class_precision_std: Dict[str, float] = field(default_factory=dict)
    per_class_recall_mean: Dict[str, float] = field(default_factory=dict)
    per_class_recall_std: Dict[str, float] = field(default_factory=dict)
    per_class_support_total: Dict[str, int] = field(default_factory=dict)  # SUM across folds -- support is a count of a non-overlapping validation slice per fold, not a quantity to average
    # Element-wise SUM of every fold's confusion matrix -- folds'
    # validation slices are disjoint (StratifiedKFold), so summing is
    # the standard, correct way to report ONE confusion matrix for a
    # whole CV run rather than k separate ones or an arbitrary "best
    # fold" pick.
    confusion_sum: Optional[np.ndarray] = None
    class_order: List[str] = field(default_factory=list)


def aggregate_fold_metrics(fold_metrics: List[FoldMetrics]) -> AggregatedMetrics:
    if not fold_metrics:
        raise ValueError("aggregate_fold_metrics called with zero folds.")

    def mean_std(values):
        arr = np.array(values, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    macro_f1_mean, macro_f1_std = mean_std([m.macro_f1 for m in fold_metrics])
    acc_mean, acc_std = mean_std([m.accuracy for m in fold_metrics])
    bal_acc_mean, bal_acc_std = mean_std([m.balanced_accuracy for m in fold_metrics])
    wf1_mean, wf1_std = mean_std([m.weighted_f1 for m in fold_metrics])
    mcc_values = [m.mcc for m in fold_metrics if m.mcc == m.mcc]  # drop NaN folds
    mcc_mean, mcc_std = mean_std(mcc_values) if mcc_values else (float("nan"), float("nan"))

    per_class_f1_mean: Dict[str, float] = {}
    per_class_f1_std: Dict[str, float] = {}
    per_class_precision_mean: Dict[str, float] = {}
    per_class_precision_std: Dict[str, float] = {}
    per_class_recall_mean: Dict[str, float] = {}
    per_class_recall_std: Dict[str, float] = {}
    per_class_support_total: Dict[str, int] = {}
    for cls in fold_metrics[0].class_order:
        f1_vals = [m.per_class[cls].f1 for m in fold_metrics]
        prec_vals = [m.per_class[cls].precision for m in fold_metrics]
        rec_vals = [m.per_class[cls].recall for m in fold_metrics]
        per_class_f1_mean[cls], per_class_f1_std[cls] = mean_std(f1_vals)
        per_class_precision_mean[cls], per_class_precision_std[cls] = mean_std(prec_vals)
        per_class_recall_mean[cls], per_class_recall_std[cls] = mean_std(rec_vals)
        per_class_support_total[cls] = sum(m.per_class[cls].support for m in fold_metrics)

    confusion_sum = np.sum([m.confusion for m in fold_metrics], axis=0)

    return AggregatedMetrics(
        n_folds=len(fold_metrics),
        macro_f1_mean=macro_f1_mean, macro_f1_std=macro_f1_std,
        accuracy_mean=acc_mean, accuracy_std=acc_std,
        balanced_accuracy_mean=bal_acc_mean, balanced_accuracy_std=bal_acc_std,
        weighted_f1_mean=wf1_mean, weighted_f1_std=wf1_std,
        mcc_mean=mcc_mean, mcc_std=mcc_std,
        per_class_f1_mean=per_class_f1_mean, per_class_f1_std=per_class_f1_std,
        per_class_precision_mean=per_class_precision_mean, per_class_precision_std=per_class_precision_std,
        per_class_recall_mean=per_class_recall_mean, per_class_recall_std=per_class_recall_std,
        per_class_support_total=per_class_support_total,
        confusion_sum=confusion_sum, class_order=list(fold_metrics[0].class_order),
    )


def holm_correction(p_values: List[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni step-down correction for multiple comparisons
    (item 4, Phase-4 review). None entries (mcnemar_test's honest
    "below reliability threshold" result) pass through as None --
    never assigned an adjusted p-value, since there was no real
    p-value to correct in the first place."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    adjusted: Dict[int, float] = {}
    max_so_far = 0.0
    for rank, (i, p) in enumerate(indexed):
        val = min(1.0, p * (m - rank))
        max_so_far = max(max_so_far, val)  # step-down monotonicity
        adjusted[i] = max_so_far
    return [adjusted.get(i) for i in range(len(p_values))]


def mcnemar_test(y_true: Sequence[str], y_pred_a: Sequence[str], y_pred_b: Sequence[str]) -> Optional[dict]:
    """Paired comparison of two models' predictions on the SAME instances
    (Part 24). Returns None (not a fabricated p-value) if the contingency
    table's off-diagonal counts are too small for the test's chi-square
    approximation to be trustworthy -- consistent with Part 24's "do not
    perform inappropriate statistical tests merely to produce a p-value."
    """
    y_true, y_pred_a, y_pred_b = list(y_true), list(y_pred_a), list(y_pred_b)
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("y_true, y_pred_a, y_pred_b must be the same length (paired predictions).")

    a_correct = np.array([a == t for a, t in zip(y_pred_a, y_true)])
    b_correct = np.array([b == t for b, t in zip(y_pred_b, y_true)])

    n01 = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n10 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong

    if n01 + n10 < 25:
        # Standard rule of thumb for McNemar's chi-square approximation
        # (Edwards' continuity correction assumption). Below this, the
        # test result would not be trustworthy -- report that honestly
        # rather than compute a misleading p-value.
        return None

    from scipy.stats import chi2
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)  # continuity-corrected
    p_value = float(1 - chi2.cdf(statistic, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(statistic), "p_value": p_value}


def write_phase3_fold_metrics(
    out_dir: Path,
    experiment_metrics: Dict[str, List[FoldMetrics]],
    fold_metadata: Optional[Dict[str, List[dict]]] = None,
) -> Tuple[Path, Path]:
    """Persists per-fold metrics and aggregate summary as required by faculty and project specification:
    - reports/phase3/fold_metrics.csv: Every fold row individually (Fold 0-4)
    - reports/phase3/fold_summary.csv: Summary computed directly from stored fold results
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "fold_metrics.csv"
    summary_path = out_dir / "fold_summary.csv"

    fieldnames = [
        "arm", "experiment_id", "fold", "fold_id", "device", "accuracy", "macro_f1",
        "balanced_accuracy", "weighted_f1", "mcc",
        "n_train", "n_val", "validation_n",
        "selected_stage", "selected_epoch", "best_val_loss", "checkpoint_used", "checkpoint",
    ]
    # Add per-class precision, recall, f1, support columns if present
    sample_folds = [fm for folds in experiment_metrics.values() for fm in folds]
    class_order = sample_folds[0].class_order if sample_folds else []
    for cls in class_order:
        fieldnames.extend([f"{cls}_precision", f"{cls}_recall", f"{cls}_f1", f"{cls}_support"])

    metric_rows = []
    for exp_id, folds in experiment_metrics.items():
        meta_list = fold_metadata.get(exp_id, []) if fold_metadata else []
        for i, fm in enumerate(folds):
            meta = meta_list[i] if i < len(meta_list) else {}
            ckpt_val = meta.get("checkpoint_used", "")
            n_val_val = meta.get("n_val", "")
            row = {
                "arm": exp_id,
                "experiment_id": exp_id,
                "fold": fm.fold_index,
                "fold_id": fm.fold_index,
                "device": meta.get("device", "Unknown"),
                "accuracy": f"{fm.accuracy:.6f}",
                "macro_f1": f"{fm.macro_f1:.6f}",
                "balanced_accuracy": f"{fm.balanced_accuracy:.6f}",
                "weighted_f1": f"{fm.weighted_f1:.6f}",
                "mcc": f"{fm.mcc:.6f}" if fm.mcc == fm.mcc else "nan",
                "n_train": meta.get("n_train", ""),
                "n_val": n_val_val,
                "validation_n": n_val_val,
                "selected_stage": meta.get("selected_stage", ""),
                "selected_epoch": meta.get("selected_epoch", ""),
                "best_val_loss": f"{meta['best_val_loss']:.6f}" if isinstance(meta.get("best_val_loss"), (int, float)) else meta.get("best_val_loss", ""),
                "checkpoint_used": ckpt_val,
                "checkpoint": ckpt_val,
            }
            for cls in class_order:
                cm = fm.per_class.get(cls)
                if cm is not None:
                    row[f"{cls}_precision"] = f"{cm.precision:.6f}"
                    row[f"{cls}_recall"] = f"{cm.recall:.6f}"
                    row[f"{cls}_f1"] = f"{cm.f1:.6f}"
                    row[f"{cls}_support"] = cm.support
                else:
                    row[f"{cls}_precision"] = ""
                    row[f"{cls}_recall"] = ""
                    row[f"{cls}_f1"] = ""
                    row[f"{cls}_support"] = ""
            metric_rows.append(row)

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_fields = [
        "experiment_id", "n_folds",
        "macro_f1_mean", "macro_f1_std",
        "accuracy_mean", "accuracy_std",
        "balanced_accuracy_mean", "balanced_accuracy_std",
        "weighted_f1_mean", "weighted_f1_std",
        "mcc_mean", "mcc_std",
    ]

    summary_rows = []
    for exp_id, folds in experiment_metrics.items():
        agg = aggregate_fold_metrics(folds)
        summary_rows.append({
            "experiment_id": exp_id,
            "n_folds": agg.n_folds,
            "macro_f1_mean": f"{agg.macro_f1_mean:.6f}",
            "macro_f1_std": f"{agg.macro_f1_std:.6f}",
            "accuracy_mean": f"{agg.accuracy_mean:.6f}",
            "accuracy_std": f"{agg.accuracy_std:.6f}",
            "balanced_accuracy_mean": f"{agg.balanced_accuracy_mean:.6f}",
            "balanced_accuracy_std": f"{agg.balanced_accuracy_std:.6f}",
            "weighted_f1_mean": f"{agg.weighted_f1_mean:.6f}",
            "weighted_f1_std": f"{agg.weighted_f1_std:.6f}",
            "mcc_mean": f"{agg.mcc_mean:.6f}" if agg.mcc_mean == agg.mcc_mean else "nan",
            "mcc_std": f"{agg.mcc_std:.6f}" if agg.mcc_std == agg.mcc_std else "nan",
        })

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    return metrics_path, summary_path


def write_phase3_winner(
    out_dir: Path,
    winner_experiment_id: str,
    winner_name: str,
    preprocessing_mode: str,
    training_time_augmentation: bool,
    selection_metric: str,
    selection_direction: str,
    winner_aggregate: AggregatedMetrics,
    representation_id_str: str,
    all_experiments: Optional[Dict[str, AggregatedMetrics]] = None,
    dataset_freeze_hash: Optional[str] = None,
    fold_plan_hash: Optional[str] = None,
    selection_rule: Optional[str] = None,
    extra_config: Optional[dict] = None,
) -> Path:
    """Authoritative machine-readable winner contract for Phase 3.
    Persisted to reports/phase3/winner.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    winner_path = out_dir / "winner.json"

    exp_summary = {}
    if all_experiments:
        for name, agg in all_experiments.items():
            exp_summary[name] = {
                "macro_f1_mean": agg.macro_f1_mean,
                "macro_f1_std": agg.macro_f1_std,
                "accuracy_mean": agg.accuracy_mean,
                "balanced_accuracy_mean": agg.balanced_accuracy_mean,
                "mcc_mean": agg.mcc_mean if agg.mcc_mean == agg.mcc_mean else None,
                "per_class_f1_mean": agg.per_class_f1_mean,
            }

    now_utc = datetime.now(timezone.utc)
    run_id = (extra_config or {}).get("run_id") or f"AEFCRC_RUN_{now_utc.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex}"

    data = {
        "run_id": run_id,
        "created_at_utc": now_utc.isoformat(),
        "winner": winner_name,
        "selection_metric": selection_metric or "mean_validation_macro_f1",
        "winner_mean_macro_f1": winner_aggregate.macro_f1_mean,
        "equivalence_margin": 0.005,
        "folds": winner_aggregate.n_folds or 5,
        "winner_experiment_id": winner_experiment_id,
        "winner_name": winner_name,
        "selection_direction": selection_direction,
        "selection_rule": selection_rule or "Highest mean validation Macro-F1 with 0.005 practical-equivalence margin preferring P3-BASE",
        "dataset_freeze_hash": dataset_freeze_hash,
        "fold_plan_hash": fold_plan_hash,
        "preprocessing_mode": preprocessing_mode,
        "training_time_augmentation": training_time_augmentation,
        "representation_id": representation_id_str,
        "macro_f1_mean": winner_aggregate.macro_f1_mean,
        "macro_f1_std": winner_aggregate.macro_f1_std,
        "accuracy_mean": winner_aggregate.accuracy_mean,
        "accuracy_std": winner_aggregate.accuracy_std,
        "balanced_accuracy_mean": winner_aggregate.balanced_accuracy_mean,
        "balanced_accuracy_std": winner_aggregate.balanced_accuracy_std,
        "mcc_mean": winner_aggregate.mcc_mean if winner_aggregate.mcc_mean == winner_aggregate.mcc_mean else None,
        "mcc_std": winner_aggregate.mcc_std if winner_aggregate.mcc_std == winner_aggregate.mcc_std else None,
        "per_class_f1_mean": winner_aggregate.per_class_f1_mean,
        "per_class_f1_std": winner_aggregate.per_class_f1_std,
        "all_experiments": exp_summary,
        "config": extra_config or {},
    }

    winner_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return winner_path


def validate_phase3_winner(reports_dir: Path, artifacts_dir: Path) -> Tuple[bool, List[str]]:
    """Programmatically validates the Phase 3 winner contract per Item 27."""
    errors = []
    winner_path = reports_dir / "winner.json"
    if not winner_path.exists():
        return False, [f"Winner file missing: {winner_path}"]

    try:
        w_data = json.loads(winner_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Unreadable winner.json: {e}"]

    winner_name = w_data.get("winner") or w_data.get("winner_experiment_id")
    if not winner_name:
        errors.append("winner.json missing 'winner' or 'winner_experiment_id' key")

    # Check 5 fold checkpoints exist
    for fold_idx in range(5):
        ckpt = artifacts_dir / winner_name / f"fold_{fold_idx:02d}" / "best_model.keras"
        if not ckpt.exists():
            errors.append(f"Missing winner checkpoint for fold {fold_idx}: {ckpt}")

    # Check fold_metrics.csv exists and has 5 rows for winner
    metrics_csv = reports_dir / "fold_metrics.csv"
    if not metrics_csv.exists():
        errors.append(f"Missing fold_metrics.csv: {metrics_csv}")
    else:
        with metrics_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        winner_rows = [r for r in rows if r.get("experiment_id") == winner_name or r.get("arm") == winner_name]
        if len(winner_rows) != 5:
            errors.append(f"Expected 5 fold metrics rows for winner {winner_name}, got {len(winner_rows)}")
        else:
            f1_mean = sum(float(r["macro_f1"]) for r in winner_rows) / 5.0
            stored_f1 = float(w_data.get("winner_mean_macro_f1", w_data.get("macro_f1_mean", 0.0)))
            if abs(f1_mean - stored_f1) > 1e-4:
                errors.append(f"Winner mean Macro-F1 ({stored_f1:.6f}) differs from fold metrics mean ({f1_mean:.6f})")

    # Check 5 confusion matrices exist
    for fold_idx in range(5):
        cm_csv = artifacts_dir / winner_name / f"fold_{fold_idx:02d}" / "confusion_matrix.csv"
        cm_json = artifacts_dir / winner_name / f"fold_{fold_idx:02d}" / "confusion_matrix.json"
        if not (cm_csv.exists() or cm_json.exists()):
            errors.append(f"Missing confusion matrix for fold {fold_idx} in {artifacts_dir / winner_name / f'fold_{fold_idx:02d}'}")

    return len(errors) == 0, errors


def select_phase3_winner(
    experiment_aggregates: Dict[str, AggregatedMetrics],
    threshold: float = 0.005,
    baseline_name: str = "P3-BASE",
) -> Tuple[str, str]:
    """Enforces the authoritative Phase 3 winner-selection rule.

    Rule:
    - Primary metric: mean validation Macro-F1 across 5 folds.
    - Threshold: A candidate arm must exceed the baseline by
      strictly > 0.005 Macro-F1 to be declared the winner.
    - Equivalence / Parsimony: If the difference between candidate and
      baseline is <= 0.005, the candidate is deemed practically equivalent or inferior;
      prefer the baseline for parsimony.
    - SD tie-breaker: Completely removed.

    Returns:
        (winner_name, rationale)
    """
    if baseline_name not in experiment_aggregates:
        # If specified baseline not present, default to arm with highest Macro-F1
        best_arm = max(experiment_aggregates.keys(), key=lambda k: experiment_aggregates[k].macro_f1_mean)
        return best_arm, f"Baseline '{baseline_name}' not in results; selected highest Macro-F1 arm '{best_arm}'."

    base_agg = experiment_aggregates[baseline_name]
    base_f1 = base_agg.macro_f1_mean

    candidates = [k for k in experiment_aggregates if k != baseline_name]
    candidates.sort(key=lambda k: experiment_aggregates[k].macro_f1_mean, reverse=True)

    if not candidates:
        return baseline_name, f"No candidate arms provided; {baseline_name} selected as baseline."

    top_candidate = candidates[0]
    top_f1 = experiment_aggregates[top_candidate].macro_f1_mean
    delta = top_f1 - base_f1

    if delta > threshold:
        rationale = (
            f"{top_candidate} won with Macro-F1 {top_f1:.6f}, exceeding {baseline_name} "
            f"({base_f1:.6f}) by {delta:+.6f} (threshold > {threshold:.3f})."
        )
        return top_candidate, rationale
    else:
        rationale = (
            f"{baseline_name} retained for parsimony. Top candidate {top_candidate} "
            f"Macro-F1 is {top_f1:.6f} vs {baseline_name} {base_f1:.6f} (delta {delta:+.6f} <= {threshold:.3f}; "
            f"practically equivalent or inferior)."
        )
        return baseline_name, rationale


