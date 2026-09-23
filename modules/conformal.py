"""
modules/conformal.py

AEF-CRC Phase 9: Conformal Prediction.

Theoretical Foundations and Scientific Rigor:
---------------------------------------------
1. Role & Order:
   Phase 9 consumes calibrated probabilities produced in Phase 8 (on the 123-image
   probability-calibration partition) and calibrates conformal prediction sets
   using the dedicated 123-image conformal-calibration partition (N_conf = 123).
   Inference pipeline:
     new image -> preprocessing -> frozen feature extraction (1348-D)
     -> frozen BDA mask -> frozen production RF -> raw probabilities
     -> Phase 8 calibration (primary Platt) -> Phase 9 conformal prediction
     -> final prediction set C_hat(x)

2. Primary Method: Pooled Marginal Split-Conformal Prediction
   - Conformal calibration set size: N_conf = 123
   - Target significance level: alpha = 0.10 (nominal coverage: 1 - alpha = 0.90 / 90%)
   - Nonconformity score:
       s_i = 1 - P_hat(y_i | x_i)
     where P_hat is the calibrated probability of the true label y_i.
   - Exact conformal quantile arithmetic:
       k = ceil((n + 1) * (1 - alpha))
       q_hat = sorted(s_1, ..., s_n)[k - 1] (with k clipped to n)
     For n = 123, alpha = 0.10:
       k = ceil(124 * 0.90) = ceil(111.6) = 112
     For n = 10, alpha = 0.10:
       k = ceil(11 * 0.90) = ceil(9.9) = 10 (NOT 11)
   - Prediction set:
       C_hat(x) = { c in Y : 1 - P_hat(c | x) <= q_hat }
                = { c in Y : P_hat(c | x) >= 1 - q_hat }

3. Secondary / Diagnostic Method: Class-Conditional Mondrian Conformal Prediction
   - Computes separate nonconformity quantiles per target class:
       Psoriasis, Lichen_Planus, Pityriasis_Rosea, Seborrheic_Dermatitis.
   - Evaluated across the 4 classes independently.
   - NOTE: Source x class Mondrian grouping is strictly rejected because cross-tabulated
     source x class cell sizes contain sparse and empty cells.
   - Small-Sample Limitation:
     For minority classes (especially Seborrheic Dermatitis, where calibration support n_c
     in a 123-image split may be small):
       - Mathematical validity is not automatically violated.
       - However, empirical quantiles become coarse.
       - Finite-sample variance is elevated.
       - Prediction sets may be wide and set efficiency lower.
     This limitation is documented and reported as a diagnostic caveat.

4. Exchangeability Wording:
   Exchangeability of the conformal calibration samples and future test samples
   is a foundational theoretical assumption, NOT an empirical guarantee.
   Empirical coverage is verified on the locked 243-image outer test set.
   Unconditional real-world coverage guarantees are never claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def compute_nonconformity_scores(
    probs: np.ndarray,
    true_indices: np.ndarray,
) -> np.ndarray:
    """
    Computes standard softmax-based nonconformity scores:
        s_i = 1.0 - P_hat(y_i | x_i)

    Args:
        probs: Array of shape (N, C) containing predicted class probabilities.
        true_indices: 1D array of length N containing true class indices (0 to C-1).

    Returns:
        1D array of length N with nonconformity scores in [0, 1].
    """
    n = len(true_indices)
    if probs.shape[0] != n:
        raise ValueError(
            f"Shape mismatch: probs has {probs.shape[0]} rows, true_indices has {n} elements"
        )
    return 1.0 - probs[np.arange(n), true_indices]


def compute_conformal_quantile(
    scores: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Computes the split-conformal empirical quantile:
        k = ceil((n + 1) * (1 - alpha))
        q_hat = sorted(scores)[k - 1]

    Guarantees exact finite-sample marginal coverage under exchangeability:
        P(Y_{n+1} in C_hat(X_{n+1})) >= 1 - alpha

    Example:
        For n = 10, alpha = 0.10:
          k = ceil((10 + 1) * 0.90) = ceil(9.9) = 10.
        For n = 123, alpha = 0.10:
          k = ceil((123 + 1) * 0.90) = ceil(111.6) = 112.

    Args:
        scores: 1D array of nonconformity scores on calibration data.
        alpha: Error rate (significance level), default 0.10 for 90% coverage.

    Returns:
        The empirical quantile threshold q_hat as float.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile on empty score array.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Theoretical index: 1-based k = ceil((n + 1) * (1 - alpha))
    k = int(math.ceil((n + 1) * (1.0 - alpha)))

    # Finite sample clipping: if k > n, clip to n (takes maximum score)
    k_clipped = min(k, n)
    k_clipped = max(1, k_clipped)

    sorted_scores = np.sort(scores)
    q_hat = float(sorted_scores[k_clipped - 1])
    return q_hat


def predict_conformal_sets(
    probs: np.ndarray,
    q_hat: float,
    class_order: List[str],
) -> List[List[str]]:
    """
    Constructs pooled marginal conformal prediction sets for each sample:
        C_hat(x) = { c in Y : s(x, c) <= q_hat }
                 = { c in Y : 1.0 - P_hat(c | x) <= q_hat }

    Directly evaluates the canonical nonconformity score condition:
        (1.0 - P_hat(c | x)) <= q_hat
    to prevent asymmetric IEEE-754 floating-point roundoff errors where
    1.0 - (1.0 - p) > p causes the exact boundary sample (s == q_hat) to be dropped.

    Args:
        probs: Array of shape (N, C) containing predicted class probabilities.
        q_hat: Calibrated conformal quantile threshold.
        class_order: List of class names corresponding to columns of probs.

    Returns:
        List of length N, where each element is a list of predicted class names.
    """
    pred_sets: List[List[str]] = []
    for i in range(len(probs)):
        row = probs[i]
        included = [class_order[c] for c in range(len(class_order)) if (1.0 - row[c]) <= q_hat]
        pred_sets.append(included)
    return pred_sets


def predict_conformal_sets_mondrian(
    probs: np.ndarray,
    q_hat_by_class: Dict[str, float],
    class_order: List[str],
) -> List[List[str]]:
    """
    Constructs class-conditional Mondrian conformal prediction sets:
        C_hat(x) = { c in Y : s(x, c) <= q_hat_c }
                 = { c in Y : 1.0 - P_hat(c | x) <= q_hat_c }

    Directly evaluates the canonical nonconformity score condition:
        (1.0 - P_hat(c | x)) <= q_c
    to prevent asymmetric IEEE-754 floating-point roundoff errors.

    Args:
        probs: Array of shape (N, C) containing predicted class probabilities.
        q_hat_by_class: Mapping from class name to class-specific quantile threshold.
        class_order: List of class names corresponding to columns of probs.

    Returns:
        List of length N, where each element is a list of predicted class names.
    """
    pred_sets: List[List[str]] = []
    for i in range(len(probs)):
        row = probs[i]
        included: List[str] = []
        for c, cls_name in enumerate(class_order):
            q_c = q_hat_by_class.get(cls_name, 1.0)
            if (1.0 - row[c]) <= q_c:
                included.append(cls_name)
        pred_sets.append(included)
    return pred_sets


@dataclass
class MarginalConformalFit:
    """Artifact of pooled marginal split-conformal calibration."""
    alpha: float
    nominal_coverage: float
    n_samples: int
    q_hat: float
    scores: np.ndarray
    formula_k: int
    clipped_k: int


@dataclass
class MondrianConformalFit:
    """Artifact of class-conditional Mondrian conformal calibration."""
    alpha: float
    nominal_coverage: float
    per_class_n: Dict[str, int]
    q_hat_by_class: Dict[str, float]
    per_class_scores: Dict[str, np.ndarray]
    small_sample_warnings: Dict[str, str]


def fit_marginal_conformal(
    probs: np.ndarray,
    true_labels: List[str],
    class_order: List[str],
    alpha: float = 0.10,
) -> MarginalConformalFit:
    """
    Fits pooled marginal split-conformal prediction threshold on calibration set.
    """
    label_to_idx = {c: i for i, c in enumerate(class_order)}
    true_indices = np.array([label_to_idx[y] for y in true_labels])
    scores = compute_nonconformity_scores(probs, true_indices)

    n = len(scores)
    formula_k = int(math.ceil((n + 1) * (1.0 - alpha)))
    clipped_k = min(formula_k, n)
    q_hat = compute_conformal_quantile(scores, alpha=alpha)

    return MarginalConformalFit(
        alpha=alpha,
        nominal_coverage=1.0 - alpha,
        n_samples=n,
        q_hat=q_hat,
        scores=scores,
        formula_k=formula_k,
        clipped_k=clipped_k,
    )


def fit_class_conditional_mondrian(
    probs: np.ndarray,
    true_labels: List[str],
    class_order: List[str],
    alpha: float = 0.10,
    min_recommended_n: int = 15,
) -> MondrianConformalFit:
    """
    Fits class-conditional Mondrian conformal prediction thresholds per class.
    Logs explicit small-sample diagnostic warnings for classes with small support.
    """
    label_to_idx = {c: i for i, c in enumerate(class_order)}
    true_indices = np.array([label_to_idx[y] for y in true_labels])
    scores = compute_nonconformity_scores(probs, true_indices)

    per_class_n: Dict[str, int] = {}
    q_hat_by_class: Dict[str, float] = {}
    per_class_scores: Dict[str, np.ndarray] = {}
    small_sample_warnings: Dict[str, str] = {}

    for c, cls_name in enumerate(class_order):
        mask = (true_indices == c)
        c_scores = scores[mask]
        n_c = len(c_scores)
        per_class_n[cls_name] = n_c
        per_class_scores[cls_name] = c_scores

        if n_c == 0:
            q_hat_by_class[cls_name] = 1.0
            small_sample_warnings[cls_name] = (
                f"Class '{cls_name}' has 0 calibration samples. Threshold defaulted to 1.0."
            )
            continue

        q_c = compute_conformal_quantile(c_scores, alpha=alpha)
        q_hat_by_class[cls_name] = q_c

        if n_c < min_recommended_n:
            small_sample_warnings[cls_name] = (
                f"Class '{cls_name}' has only {n_c} calibration samples (recommended >= {min_recommended_n}). "
                f"Empirical quantile k={math.ceil((n_c+1)*(1-alpha))} is coarse, causing higher finite-sample "
                f"variance and potentially wide prediction sets."
            )

    return MondrianConformalFit(
        alpha=alpha,
        nominal_coverage=1.0 - alpha,
        per_class_n=per_class_n,
        q_hat_by_class=q_hat_by_class,
        per_class_scores=per_class_scores,
        small_sample_warnings=small_sample_warnings,
    )


@dataclass
class ConformalMetrics:
    """Summary of conformal prediction performance on an evaluation set."""
    nominal_coverage: float
    marginal_coverage: float
    per_class_coverage: Dict[str, float]
    mean_set_size: float
    median_set_size: float
    set_size_distribution: Dict[int, int]
    singleton_fraction: float
    empty_set_fraction: float
    total_samples: int


def evaluate_conformal_sets(
    pred_sets: List[List[str]],
    true_labels: List[str],
    class_order: List[str],
    nominal_coverage: float = 0.90,
) -> ConformalMetrics:
    """
    Evaluates conformal prediction sets against ground-truth labels.
    """
    n = len(true_labels)
    if len(pred_sets) != n:
        raise ValueError(
            f"Length mismatch: pred_sets has {len(pred_sets)} sets, true_labels has {n}"
        )

    # Coverage
    covered = np.array([true_labels[i] in pred_sets[i] for i in range(n)], dtype=float)
    marginal_coverage = float(np.mean(covered)) if n > 0 else 0.0

    per_class_cov: Dict[str, float] = {}
    for cls in class_order:
        cls_mask = np.array([true_labels[i] == cls for i in range(n)])
        if cls_mask.sum() > 0:
            per_class_cov[cls] = float(np.mean(covered[cls_mask]))
        else:
            per_class_cov[cls] = float("nan")

    # Set efficiency
    sizes = np.array([len(s) for s in pred_sets], dtype=int)
    mean_size = float(np.mean(sizes)) if n > 0 else 0.0
    median_size = float(np.median(sizes)) if n > 0 else 0.0

    size_dist: Dict[int, int] = {k: int((sizes == k).sum()) for k in range(len(class_order) + 1)}
    singleton_frac = float((sizes == 1).mean()) if n > 0 else 0.0
    empty_frac = float((sizes == 0).mean()) if n > 0 else 0.0

    return ConformalMetrics(
        nominal_coverage=nominal_coverage,
        marginal_coverage=marginal_coverage,
        per_class_coverage=per_class_cov,
        mean_set_size=mean_size,
        median_set_size=median_size,
        set_size_distribution=size_dist,
        singleton_fraction=singleton_frac,
        empty_set_fraction=empty_frac,
        total_samples=n,
    )
