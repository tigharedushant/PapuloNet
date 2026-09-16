"""
modules/calibration.py

AEF-CRC Phase 8: Probability Calibration.

Frozen Calibration Protocol:
----------------------------
1. Primary Method: Sigmoid / Platt Scaling
   - One 1D Logistic Regression model fit independently per target class:
       y_k in {0, 1} regressed on P_raw(Y=k | X)
   - Platt/sigmoid calibration fits a logistic sigmoid mapping using logistic regression/log-loss.
   - ECE is an evaluation metric used to assess calibration quality (ECE, 10 equal-width bins),
     along with Multiclass Brier Score and reliability diagrams; ECE is NOT the optimization objective.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

EPS = 1e-7


def clip_probabilities(probs: np.ndarray) -> np.ndarray:
    """Keeps every probability in (EPS, 1-EPS). Renormalizes each row
    back to sum to 1 after clipping, preventing log(0) and numerical instability."""
    clipped = np.clip(probs, EPS, 1.0 - EPS)
    return clipped / clipped.sum(axis=1, keepdims=True)


def negative_log_likelihood(clipped_probs: np.ndarray, true_indices: np.ndarray) -> float:
    """Mean negative log-likelihood (NLL) over the sample set."""
    n = len(true_indices)
    return float(-np.mean(np.log(clipped_probs[np.arange(n), true_indices])))


def brier_score(probs: np.ndarray, true_indices: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean squared difference between full
    probability vector and one-hot ground-truth indicator."""
    n = len(true_indices)
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), true_indices] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


@dataclass
class ReliabilityBin:
    bin_low: float
    bin_high: float
    count: int
    mean_confidence: float
    accuracy: float


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10,
) -> Tuple[float, List[ReliabilityBin]]:
    """Standard top-label Expected Calibration Error (ECE) across n_bins equal-width bins."""
    n = len(confidences)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[ReliabilityBin] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == n_bins - 1 else (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(ReliabilityBin(float(lo), float(hi), 0, 0.0, 0.0))
            continue
        mean_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        bins.append(ReliabilityBin(float(lo), float(hi), count, mean_conf, acc))
        ece += (count / n) * abs(acc - mean_conf)
    return float(ece), bins


def per_class_ece(
    probs: np.ndarray, true_indices: np.ndarray, class_order: List[str], n_bins: int = 10,
) -> Dict[str, float]:
    """One-vs-rest ECE per target class."""
    result = {}
    for k, cls in enumerate(class_order):
        ece, _ = expected_calibration_error(probs[:, k], (true_indices == k).astype(float), n_bins=n_bins)
        result[cls] = ece
    return result


def compute_ece(
    y_true: Union[np.ndarray, List[int], List[str]],
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) across n_bins equal-width bins.
    Compatibility wrapper returning scalar ECE.
    """
    confidences = np.max(probs, axis=1)
    y_pred = np.argmax(probs, axis=1)
    if isinstance(y_true, list) and len(y_true) > 0 and isinstance(y_true[0], str):
        raise ValueError("y_true must be numeric indices or one-hot vectors, not raw string labels")
    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim == 2:
        y_true_indices = np.argmax(y_true_arr, axis=1)
    else:
        y_true_indices = y_true_arr
    correct = (y_true_indices == y_pred).astype(float)
    ece, _ = expected_calibration_error(confidences, correct, n_bins=n_bins)
    return float(ece)


def multiclass_brier_score(
    y_true_onehot: np.ndarray,
    probs: np.ndarray,
) -> float:
    """Multiclass Brier score: mean squared difference between predicted
    probability vector and one-hot ground truth.
    Compatibility wrapper accepting (y_true_onehot, probs).
    """
    return float(np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1)))



# =============================================================================
# Primary Calibration: Sigmoid / Platt Scaling
# =============================================================================

@dataclass
class PlattScalingFit:
    models: Dict[str, object]  # One fitted LogisticRegression per class (or None if uninformative)


def fit_platt_scaling(
    probs: np.ndarray,
    true_indices: np.ndarray,
    class_order: List[str],
    random_seed: int,
) -> PlattScalingFit:
    """Fits one sigmoid (2-parameter Logistic Regression) per class minimizing log-loss to produce calibrated class probabilities.
    Platt/sigmoid calibration fits a logistic sigmoid mapping using logistic regression/log-loss.
    ECE is an evaluation metric used to assess calibration quality, not the optimization objective.
    PRIMARY pre-registered calibration method."""
    from sklearn.linear_model import LogisticRegression

    models: Dict[str, object] = {}
    for k, cls in enumerate(class_order):
        y = (true_indices == k).astype(int)
        if len(set(y)) < 2:
            models[cls] = None
            continue
        clf = LogisticRegression(random_state=random_seed)
        clf.fit(probs[:, k].reshape(-1, 1), y)
        models[cls] = clf
    return PlattScalingFit(models=models)


def apply_platt_scaling(
    probs: np.ndarray,
    fit: PlattScalingFit,
    class_order: List[str],
) -> np.ndarray:
    """Applies per-class fitted sigmoids and renormalizes probabilities to sum to 1, producing calibrated class probabilities."""
    calibrated = np.zeros_like(probs)
    for k, cls in enumerate(class_order):
        model = fit.models.get(cls)
        calibrated[:, k] = probs[:, k] if model is None else model.predict_proba(probs[:, k].reshape(-1, 1))[:, 1]
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return calibrated / row_sums


# =============================================================================
# Secondary Comparator: Isotonic Regression
# =============================================================================

@dataclass
class IsotonicScalingFit:
    models: Dict[str, object]  # One fitted IsotonicRegression per class (or None)


def fit_isotonic_scaling(
    probs: np.ndarray,
    true_indices: np.ndarray,
    class_order: List[str],
) -> IsotonicScalingFit:
    """One-vs-rest Isotonic Regression per class. SECONDARY comparator only."""
    from sklearn.isotonic import IsotonicRegression

    models: Dict[str, object] = {}
    for k, cls in enumerate(class_order):
        y = (true_indices == k).astype(float)
        if len(set(y)) < 2:
            models[cls] = None
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(probs[:, k], y)
        models[cls] = iso
    return IsotonicScalingFit(models=models)


def apply_isotonic_scaling(
    probs: np.ndarray,
    fit: IsotonicScalingFit,
    class_order: List[str],
) -> np.ndarray:
    """Applies per-class fitted isotonic regressions and renormalizes probabilities to sum to 1."""
    calibrated = np.zeros_like(probs)
    for k, cls in enumerate(class_order):
        model = fit.models.get(cls)
        calibrated[:, k] = probs[:, k] if model is None else model.predict(probs[:, k])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return calibrated / row_sums


# =============================================================================
# Validation Set Partitioning (123 val_calib / 123 val_conf)
# =============================================================================

@dataclass
class ValidationPartitionResult:
    val_calib_psd_ids: List[str]
    val_conf_psd_ids: List[str]
    val_calib_labels: List[str]
    val_conf_labels: List[str]
    val_calib_indices: np.ndarray
    val_conf_indices: np.ndarray
    per_class_calib_counts: Dict[str, int]
    per_class_conf_counts: Dict[str, int]
    random_seed: int
    note: str = (
        "Strictly disjoint, stratified 50/50 partition of outer validation (246 -> 123 calib + 123 conf). "
        "val_calib fits probability calibrators; val_conf fits conformal prediction sets. "
        "Outer test (243 images) remains locked and completely excluded."
    )


def partition_outer_validation(
    psd_ids: List[str],
    labels: List[str],
    class_order: List[str],
    random_seed: int = 42,
    calib_fraction: float = 0.5,
) -> ValidationPartitionResult:
    """
    Partitions outer validation into two disjoint, stratified subsets:
      - val_calib: 123 images for Phase 8 probability calibration
      - val_conf: 123 images for Phase 9 conformal prediction
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    n_total = len(psd_ids)
    if n_total < 2:
        raise ValueError(f"Cannot partition validation set with fewer than 2 samples (got {n_total})")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=calib_fraction, random_state=random_seed)
    train_idx, val_idx = next(sss.split(psd_ids, labels))

    calib_ids = [psd_ids[i] for i in train_idx]
    conf_ids = [psd_ids[i] for i in val_idx]
    calib_lbls = [labels[i] for i in train_idx]
    conf_lbls = [labels[i] for i in val_idx]

    # Leakage & integrity guards
    assert set(calib_ids).isdisjoint(set(conf_ids)), "Leakage: val_calib and val_conf subsets overlap!"
    assert len(calib_ids) + len(conf_ids) == n_total, "Sample loss: partition count mismatch!"
    if n_total == 246:
        assert len(calib_ids) == 123, f"Expected 123 calib samples from 246 outer validation, got {len(calib_ids)}"
        assert len(conf_ids) == 123, f"Expected 123 conf samples from 246 outer validation, got {len(conf_ids)}"

    calib_counts = {cls: int(sum(1 for lbl in calib_lbls if lbl == cls)) for cls in class_order}
    conf_counts = {cls: int(sum(1 for lbl in conf_lbls if lbl == cls)) for cls in class_order}

    return ValidationPartitionResult(
        val_calib_psd_ids=calib_ids,
        val_conf_psd_ids=conf_ids,
        val_calib_labels=calib_lbls,
        val_conf_labels=conf_lbls,
        val_calib_indices=train_idx,
        val_conf_indices=val_idx,
        per_class_calib_counts=calib_counts,
        per_class_conf_counts=conf_counts,
        random_seed=random_seed,
    )


def partition_validation_set(
    records: Sequence[Any],
    seed: int = 42,
    calib_fraction: float = 0.5,
) -> Tuple[List[Any], List[Any]]:
    """Partitions outer validation records into (D_prob, D_conf) subsets (123 + 123 for N=246)."""
    if not records:
        return [], []
    psd_ids = [getattr(r, "psd_id", str(i)) for i, r in enumerate(records)]
    labels = [
        getattr(r, "mapped_class", getattr(r, "label", str(r.get("mapped_class", r.get("label", "")))))
        if hasattr(r, "mapped_class") or hasattr(r, "label") or isinstance(r, dict)
        else str(r)
        for r in records
    ]
    classes = sorted(list(set(labels)))
    res = partition_outer_validation(
        psd_ids=psd_ids, labels=labels, class_order=classes, random_seed=seed, calib_fraction=calib_fraction,
    )
    calib_records = [records[i] for i in res.val_calib_indices]
    conf_records = [records[i] for i in res.val_conf_indices]
    return calib_records, conf_records


# =============================================================================
# Evaluation & Frozen Protocol Specification
# =============================================================================

@dataclass
class CalibrationMetrics:
    method: str  # "uncalibrated" | "platt" | "isotonic"
    nll: float
    brier: float
    ece: float
    reliability_bins: List[ReliabilityBin]
    per_class_ece: Dict[str, float]
    macro_f1: float  # diagnostic check ONLY -- never used to alter calibration protocol


def evaluate_calibration(
    probs: np.ndarray,
    true_labels: List[str],
    class_order: List[str],
    method: str,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Computes calibration evaluation metrics for one probability matrix."""
    label_to_idx = {c: i for i, c in enumerate(class_order)}
    true_indices = np.array([label_to_idx[y] for y in true_labels])
    clipped = clip_probabilities(probs)

    nll = negative_log_likelihood(clipped, true_indices)
    brier = brier_score(probs, true_indices, len(class_order))

    confidences = probs.max(axis=1)
    predicted_indices = probs.argmax(axis=1)
    correct = (predicted_indices == true_indices).astype(float)
    ece, bins = expected_calibration_error(confidences, correct, n_bins=n_bins)
    per_cls_ece = per_class_ece(probs, true_indices, class_order, n_bins=n_bins)

    from modules.evaluation import compute_fold_metrics
    predicted_labels = [class_order[i] for i in predicted_indices]
    fold_metrics = compute_fold_metrics(true_labels, predicted_labels, class_order, fold_index=-1)

    return CalibrationMetrics(
        method=method, nll=nll, brier=brier, ece=ece, reliability_bins=bins,
        per_class_ece=per_cls_ece, macro_f1=fold_metrics.macro_f1,
    )


def select_calibration_method(
    *args, **kwargs
) -> Tuple[str, str]:
    """
    FROZEN PROTOCOL SELECTOR:
    Sigmoid / Platt scaling is the pre-registered PRIMARY calibration method.
    No adaptive selection or data-dependent cascade is permitted to override Platt.
    """
    return "platt", "Frozen protocol: Sigmoid / Platt scaling is the pre-registered primary method"


def select_calibration_method_cv(
    *args, **kwargs
) -> Tuple[str, str, Dict[str, float]]:
    """
    FROZEN PROTOCOL SELECTOR:
    Sigmoid / Platt scaling is the pre-registered PRIMARY calibration method.
    No adaptive cross-validation cascade is permitted to override Platt.
    """
    return "platt", "Frozen protocol: Sigmoid / Platt scaling is the pre-registered primary method", {}
