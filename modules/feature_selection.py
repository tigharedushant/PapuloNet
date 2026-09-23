"""
modules/feature_selection.py

AEF-CRC Phase 6: Feature Refinement Benchmark.

Four candidates over modules.fusion's already-built fused feature
matrix (FusionFoldData): no-selection, XGBoost native importance, RFE,
GA. All fitting is train-fold-only; fold.val_records is transform-only
for the first three, and NEVER touched by GA's fitness evaluation
either -- GA carves its own nested train/val split out of
fold.train_records for that purpose (see select_ga's docstring).

Reuses modules.fusion.FusionFoldData.branch_dims to report per-branch
retained-feature counts, not just a bare total -- branch_dims gives
each branch's column count in the SAME order columns were concatenated
in build_fusion_fold, so cumulative offsets here map a global feature
index back to its branch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.config import PSDConfig
from modules.fusion import FusionFoldData


@dataclass
class SelectionResult:
    method: str
    selected_mask: np.ndarray          # boolean, length == total fused dimensionality
    selected_count: int
    branch_retained: Dict[str, int]    # branch -> count of its columns still selected
    fit_time_sec: float
    extra: dict = field(default_factory=dict)  # method-specific detail (e.g. GA generation history)

    @property
    def selected_indices(self) -> np.ndarray:
        """Column indices into the fused feature matrix that survived
        selection -- derived from selected_mask (never stored
        separately) so the two can never drift out of sync."""
        return np.where(self.selected_mask)[0]

    @property
    def original_dim(self) -> int:
        """Total fused dimensionality BEFORE selection -- len(selected_mask)."""
        return len(self.selected_mask)


def _branch_offsets(branch_dims: Dict[str, int]) -> List[Tuple[str, int, int]]:
    """[(branch_name, start_col, end_col), ...] in concatenation order."""
    offsets = []
    start = 0
    for branch, dim in branch_dims.items():
        offsets.append((branch, start, start + dim))
        start += dim
    return offsets


def _branch_retained_counts(mask: np.ndarray, branch_dims: Dict[str, int]) -> Dict[str, int]:
    return {
        branch: int(mask[start:end].sum())
        for branch, start, end in _branch_offsets(branch_dims)
    }


# ============================================================
# Candidate 1: no selection
# ============================================================

def select_no_selection(data: FusionFoldData) -> SelectionResult:
    mask = np.ones(data.X_train.shape[1], dtype=bool)
    return SelectionResult(
        method="no_selection", selected_mask=mask, selected_count=int(mask.sum()),
        branch_retained=_branch_retained_counts(mask, data.branch_dims), fit_time_sec=0.0,
    )


# ============================================================
# Candidate 2: XGBoost native importance (or sklearn fallback)
# ============================================================

def select_xgboost_importance(data: FusionFoldData, top_k: int, random_seed: int) -> SelectionResult:
    """Fits on train only. Selects the top_k features by native
    importance -- a fixed, configuration-driven count (config.
    feature_selection_top_k), never a threshold tuned by inspecting
    validation performance."""
    classes = sorted(set(data.y_train))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = np.array([label_to_idx[y] for y in data.y_train])

    t0 = time.perf_counter()
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=random_seed, eval_metric="mlogloss")
        clf.fit(data.X_train, y_train_idx)
        importances = clf.feature_importances_
        backend = "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.inspection import permutation_importance
        clf = HistGradientBoostingClassifier(random_state=random_seed)
        clf.fit(data.X_train, y_train_idx)
        # HistGradientBoostingClassifier has no native feature_importances_ --
        # permutation importance on TRAIN data only (never val) is the
        # closest train-only substitute, explicitly reported as such.
        perm = permutation_importance(clf, data.X_train, y_train_idx, n_repeats=5, random_state=random_seed)
        importances = perm.importances_mean
        backend = "sklearn_hgb_permutation_importance_fallback"
    fit_time = time.perf_counter() - t0

    k = min(top_k, len(importances))
    top_indices = np.argsort(importances)[::-1][:k]
    mask = np.zeros(len(importances), dtype=bool)
    mask[top_indices] = True

    return SelectionResult(
        method="xgboost_importance", selected_mask=mask, selected_count=int(mask.sum()),
        branch_retained=_branch_retained_counts(mask, data.branch_dims), fit_time_sec=fit_time,
        extra={"backend": backend, "top_k_requested": top_k},
    )


# ============================================================
# Candidate 3: RFE
# ============================================================

def select_rfe(data: FusionFoldData, n_features_to_select: int, step: float, random_seed: int) -> SelectionResult:
    """step is a FRACTION (sklearn RFE accepts float step) -- coarse
    elimination schedule. step=0.05 removes ~5% of remaining features
    per iteration, not one at a time (Part: single-feature elimination
    at ~1300-D would mean ~1300 refits/fold, indefensible)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFE

    classes = sorted(set(data.y_train))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = np.array([label_to_idx[y] for y in data.y_train])

    estimator = LogisticRegression(max_iter=500, random_state=random_seed)
    n_select = min(n_features_to_select, data.X_train.shape[1])

    t0 = time.perf_counter()
    rfe = RFE(estimator=estimator, n_features_to_select=n_select, step=step)
    rfe.fit(data.X_train, y_train_idx)
    fit_time = time.perf_counter() - t0

    mask = rfe.support_
    return SelectionResult(
        method="rfe", selected_mask=mask, selected_count=int(mask.sum()),
        branch_retained=_branch_retained_counts(mask, data.branch_dims), fit_time_sec=fit_time,
        extra={"step": step, "n_features_to_select": n_select},
    )


def estimate_rfe_cost(data: FusionFoldData, n_features_to_select: int, step: float, random_seed: int, n_folds: int) -> dict:
    """Part 'Efficiency': time ONE fold's RFE fit, extrapolate to all
    folds, BEFORE committing to running it on every fold. Returns the
    estimate; does not itself decide whether to proceed -- the caller
    reports this number and continues (or documents a change) explicitly."""
    t0 = time.perf_counter()
    select_rfe(data, n_features_to_select, step, random_seed)
    single_fold_time = time.perf_counter() - t0
    return {
        "single_fold_seconds": single_fold_time,
        "estimated_total_seconds": single_fold_time * n_folds,
        "n_folds": n_folds,
    }


# ============================================================
# Candidate 4: GA (only run if the early-stop rule says it's warranted)
# ============================================================

def select_ga(
    data: FusionFoldData, config: PSDConfig, random_seed: int,
) -> SelectionResult:
    """Training-fold fitness ONLY -- fold.val_records is never touched
    during evolution. To evaluate candidate feature subsets without
    touching the real validation split, this carves a NESTED train/val
    split out of data.X_train/data.y_train alone
    (config.ga_nested_val_fraction, seeded), used only internally here.

    Binary chromosome = boolean mask over all fused features. Fitness =
    nested-val Macro-F1 of a fast LogisticRegression trained on the
    nested-train subset. Under the amended Phase 6 V2 protocol, this objective
    is completely unpenalized (config.ga_feature_count_penalty = 0.0; lambda = 0),
    evaluating candidate subsets purely on classification performance without an
    arbitrary sparsity prior. Fixed seed throughout -- population/crossover/
    mutation/tournament selection all draw from one seeded RNG.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from modules.evaluation import compute_fold_metrics

    rng = np.random.default_rng(random_seed)
    n_features = data.X_train.shape[1]
    classes = sorted(set(data.y_train))

    # Nested split, carved from TRAIN only -- never from data.X_val/y_val.
    idx = np.arange(len(data.y_train))
    ga_train_idx, ga_val_idx = train_test_split(
        idx, test_size=config.ga_nested_val_fraction, random_state=random_seed,
        stratify=data.y_train,
    )
    X_ga_train, X_ga_val = data.X_train[ga_train_idx], data.X_train[ga_val_idx]
    y_ga_train = [data.y_train[i] for i in ga_train_idx]
    y_ga_val = [data.y_train[i] for i in ga_val_idx]

    # Fold-local class weights calculated strictly from INNER-TRAIN ONLY:
    # w_c = N_inner_train / (C * N_c_inner_train)
    n_inner_train = len(y_ga_train)
    n_classes = len(classes)
    inner_class_counts = {c: y_ga_train.count(c) for c in classes}
    inner_class_weights = {
        c: n_inner_train / (n_classes * max(1, count))
        for c, count in inner_class_counts.items()
    }
    sw_ga_train = np.array([inner_class_weights[y] for y in y_ga_train], dtype=np.float64)
    penalty = getattr(config, "ga_feature_count_penalty", 0.0)

    def fitness(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return -1.0  # degenerate individual -- worst possible fitness, never selected
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=300,
            random_state=random_seed,
        )
        clf.fit(X_ga_train[:, mask], y_ga_train, sample_weight=sw_ga_train)
        y_pred = clf.predict(X_ga_val[:, mask])
        metrics = compute_fold_metrics(y_ga_val, list(y_pred), classes, fold_index=0)
        return metrics.macro_f1 - penalty * int(mask.sum())

    # Initialize population -- each individual selects a random ~15% of
    # features initially (roughly matching feature_selection_top_k's
    # proportion), not 50%, so early individuals aren't dominated by the
    # feature-count penalty before selection pressure has a chance to act.
    pop_size = config.ga_population_size
    population = [rng.random(n_features) < 0.15 for _ in range(pop_size)]

    t0 = time.perf_counter()
    history = []
    best_mask, best_fitness = None, -np.inf

    for gen in range(config.ga_generations):
        fitnesses = np.array([fitness(ind) for ind in population])
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_mask = population[gen_best_idx].copy()
        history.append({"generation": gen, "best_fitness": float(fitnesses.max()), "mean_fitness": float(fitnesses.mean())})

        # Tournament selection (size 3), elitism (best individual survives unmutated)
        new_population = [best_mask.copy()]
        while len(new_population) < pop_size:
            i, j, k = rng.integers(0, pop_size, 3)
            parent_a = population[i] if fitnesses[i] >= max(fitnesses[j], fitnesses[k]) else (population[j] if fitnesses[j] >= fitnesses[k] else population[k])
            i, j, k = rng.integers(0, pop_size, 3)
            parent_b = population[i] if fitnesses[i] >= max(fitnesses[j], fitnesses[k]) else (population[j] if fitnesses[j] >= fitnesses[k] else population[k])

            if rng.random() < config.ga_crossover_rate:
                point = rng.integers(1, n_features)
                child = np.concatenate([parent_a[:point], parent_b[point:]])
            else:
                child = parent_a.copy()

            mutation = rng.random(n_features) < config.ga_mutation_rate
            child = np.where(mutation, ~child, child)
            new_population.append(child)

        population = new_population

    fit_time = time.perf_counter() - t0

    return SelectionResult(
        method="ga", selected_mask=best_mask, selected_count=int(best_mask.sum()),
        branch_retained=_branch_retained_counts(best_mask, data.branch_dims), fit_time_sec=fit_time,
        extra={
            "population_size": pop_size, "generations": config.ga_generations,
            "mutation_rate": config.ga_mutation_rate, "crossover_rate": config.ga_crossover_rate,
            "feature_count_penalty": penalty,
            "fitness_objective": "inner_validation_macro_f1",
            "parsimony_penalty": None if penalty == 0.0 else penalty,
            "best_nested_fitness": best_fitness,
            "history": history,
        },
    )


# ============================================================
# Candidate 5: BDA (Binary Dragonfly Algorithm with time-varying V-shaped transfer function)
# ============================================================

def select_bda(
    data: FusionFoldData, config: PSDConfig, random_seed: int,
) -> SelectionResult:
    """Binary Dragonfly Algorithm (BDA) with Time-Varying V-Shaped Transfer Function.

    Operates strictly within a nested stratified split of data.X_train / data.y_train.
    data.X_val / data.y_val is NEVER touched during optimization.

    Five behavioral components:
    - Separation (S): avoidance of local crowding with neighboring dragonflies
    - Alignment (A): velocity matching with neighbors
    - Cohesion (C): tendency toward the center of mass of neighbors
    - Attraction to Food (F): tendency toward the best solution found under the configured search budget (X+)
    - Distraction from Enemy (E): repulsion from the worst solution found under the configured search budget (X-)

    Step vectors (velocities) are bounded within [-v_max, v_max].
    Time-varying transfer function:
        tau(t) = tau_min + (tau_max - tau_min) * (t / (T - 1))  # 0-indexed: t in [0, T-1] -> [tau_min, tau_max]
        T(Delta_X) = |tanh(tau(t) * Delta_X)|
    Position update:
        if rand() < T(Delta_X):
            X_new = not X (flip bit)
        else:
            X_new = X (retain bit)

    All-zero recovery rule:
        if X_new.sum() == 0:
            d_star = argmax(|Delta_X|)
            X_new[d_star] = True

    Fitness:
        fitness = Macro-F1(nested_val)  # lambda = 0 (unpenalized under Phase 6 V2 protocol)
        identical to GA's objective formulation for direct comparability.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from modules.evaluation import compute_fold_metrics

    rng = np.random.default_rng(random_seed)
    n_features = data.X_train.shape[1]
    classes = sorted(set(data.y_train))

    # Stratified nested split carved strictly from fold training data
    nested_fraction = getattr(config, "bda_nested_val_fraction", getattr(config, "ga_nested_val_fraction", 0.2))
    idx = np.arange(len(data.y_train))
    bda_train_idx, bda_val_idx = train_test_split(
        idx, test_size=nested_fraction, random_state=random_seed,
        stratify=data.y_train,
    )
    X_bda_train, X_bda_val = data.X_train[bda_train_idx], data.X_train[bda_val_idx]
    y_bda_train = [data.y_train[i] for i in bda_train_idx]
    y_bda_val = [data.y_train[i] for i in bda_val_idx]

    penalty = getattr(config, "bda_feature_count_penalty", getattr(config, "ga_feature_count_penalty", 0.0))

    # Fold-local class weights calculated strictly from INNER-TRAIN ONLY:
    # w_c = N_inner_train / (C * N_c_inner_train)
    n_inner_train = len(y_bda_train)
    n_classes = len(classes)
    inner_class_counts = {c: y_bda_train.count(c) for c in classes}
    inner_class_weights = {
        c: n_inner_train / (n_classes * max(1, count))
        for c, count in inner_class_counts.items()
    }
    sw_bda_train = np.array([inner_class_weights[y] for y in y_bda_train], dtype=np.float64)

    def evaluate_mask(mask: np.ndarray) -> Tuple[float, float]:
        """Returns (fitness, macro_f1)."""
        k = int(mask.sum())
        if k == 0:
            return -1.0, 0.0
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=300,
            random_state=random_seed,
        )
        clf.fit(X_bda_train[:, mask], y_bda_train, sample_weight=sw_bda_train)
        y_pred = clf.predict(X_bda_val[:, mask])
        metrics = compute_fold_metrics(y_bda_val, list(y_pred), classes, fold_index=0)
        fit_val = metrics.macro_f1 - penalty * k
        return fit_val, metrics.macro_f1

    pop_size = getattr(config, "bda_population_size", 20)
    iterations = getattr(config, "bda_iterations", 30)
    v_max = getattr(config, "bda_v_max", 6.0)
    tau_min = getattr(config, "bda_tau_min", 1.0)
    tau_max = getattr(config, "bda_tau_max", 4.0)
    w_max = getattr(config, "bda_w_max", 0.9)
    w_min = getattr(config, "bda_w_min", 0.4)

    # Initialize positions (~15% active features matching GA) and step vectors in [-1, 1]
    population = [rng.random(n_features) < 0.15 for _ in range(pop_size)]
    velocities = [rng.uniform(-1.0, 1.0, size=n_features) for _ in range(pop_size)]

    t0 = time.perf_counter()
    history = []
    best_mask = None
    best_fitness = -np.inf
    best_macro_f1 = 0.0

    for it in range(iterations):
        # 1. Evaluate fitness for all dragonflies
        eval_results = [evaluate_mask(ind) for ind in population]
        fitnesses = np.array([res[0] for res in eval_results])
        macro_f1s = np.array([res[1] for res in eval_results])

        best_idx = int(np.argmax(fitnesses))
        worst_idx = int(np.argmin(fitnesses))

        food = population[best_idx].copy().astype(np.float64)
        enemy = population[worst_idx].copy().astype(np.float64)

        if fitnesses[best_idx] > best_fitness:
            best_fitness = float(fitnesses[best_idx])
            best_macro_f1 = float(macro_f1s[best_idx])
            best_mask = population[best_idx].copy()

        history.append({
            "iteration": it,
            "best_fitness": best_fitness,
            "mean_fitness": float(fitnesses.mean()),
            "best_selected_count": int(best_mask.sum()) if best_mask is not None else 0,
            "best_nested_macro_f1": best_macro_f1,
        })

        # 2. Dynamic swarming weights (exploration early -> exploitation late)
        progress = it / max(1, iterations - 1)
        w = w_max - (w_max - w_min) * progress     # inertia weight: w_max -> w_min
        s = 2.0 * rng.random() * (1.0 - progress) # separation: high early
        e = rng.random() * (1.0 - progress)       # enemy distraction: high early
        a = 2.0 * rng.random() * progress         # alignment: high late
        c = 2.0 * rng.random() * progress         # cohesion: high late
        f = 2.0 * rng.random() * progress         # food attraction: high late

        tau_t = tau_min + (tau_max - tau_min) * progress

        pop_matrix = np.array(population, dtype=np.float64)
        vel_matrix = np.array(velocities, dtype=np.float64)
        mean_pos = pop_matrix.mean(axis=0)
        mean_vel = vel_matrix.mean(axis=0)

        new_population = []
        new_velocities = []

        for i in range(pop_size):
            Xi = pop_matrix[i]
            Vi = vel_matrix[i]

            # Swarm forces
            S_i = -np.sum(pop_matrix - Xi, axis=0) / max(1, pop_size - 1)
            A_i = mean_vel
            C_i = mean_pos - Xi
            F_i = food - Xi
            E_i = enemy + Xi

            # Step vector update
            Delta_X = (s * S_i + a * A_i + c * C_i + f * F_i + e * E_i) + w * Vi
            Delta_X = np.clip(Delta_X, -v_max, v_max)
            new_velocities.append(Delta_X)

            # Time-varying V-shaped transfer function
            T_val = np.abs(np.tanh(tau_t * Delta_X))

            # Complement-based binary position update
            flip_mask = rng.random(n_features) < T_val
            Xi_bool = population[i].copy()
            Xi_new = np.where(flip_mask, ~Xi_bool, Xi_bool)

            # All-zero recovery: deterministically activate dimension with largest velocity magnitude
            if Xi_new.sum() == 0:
                d_star = int(np.argmax(np.abs(Delta_X)))
                Xi_new[d_star] = True

            new_population.append(Xi_new)

        # Elitism: best mask always survives
        if best_mask is not None:
            new_population[0] = best_mask.copy()

        population = new_population
        velocities = new_velocities

    fit_time = time.perf_counter() - t0

    return SelectionResult(
        method="bda", selected_mask=best_mask, selected_count=int(best_mask.sum()),
        branch_retained=_branch_retained_counts(best_mask, data.branch_dims), fit_time_sec=fit_time,
        extra={
            "population_size": pop_size,
            "iterations": iterations,
            "v_max": v_max,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "w_max": w_max,
            "w_min": w_min,
            "transfer_function": "time_varying_v_shaped",
            "feature_count_penalty": penalty,
            "fitness_objective": "inner_validation_macro_f1",
            "parsimony_penalty": None if penalty == 0.0 else penalty,
            "best_nested_fitness": best_fitness,
            "best_nested_macro_f1": best_macro_f1,
            "convergence_history": history,
            "history": history,
        },
    )


# ============================================================
# Stability and Interpretability Helpers
# ============================================================

def compute_jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calculates Jaccard similarity index J(A, B) = |A ∩ B| / |A ∪ B| between two binary masks."""
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_pairwise_jaccard(masks: List[np.ndarray]) -> Dict[str, Any]:
    """Computes pairwise, mean, median, min, and max Jaccard similarity across a list of fold masks."""
    if len(masks) < 2:
        return {"pairwise": {}, "mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0}
    pairwise = {}
    values = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            sim = compute_jaccard_similarity(masks[i], masks[j])
            pairwise[f"fold_{i}_vs_fold_{j}"] = round(sim, 4)
            values.append(sim)
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    return {
        "pairwise": pairwise,
        "mean": mean_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "mean_jaccard": mean_val,
        "median_jaccard": median_val,
        "min_jaccard": min_val,
        "max_jaccard": max_val,
    }


def compute_pairwise_jaccard_stability(masks: List[np.ndarray]) -> float:
    """Computes mean pairwise Jaccard similarity across a list of fold masks."""
    if len(masks) < 2:
        return 1.0
    sims = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            sims.append(compute_jaccard_similarity(masks[i], masks[j]))
    return float(np.mean(sims)) if sims else 1.0


def compute_branch_jaccard_stability(
    masks: List[np.ndarray], branch_dims: Dict[str, int]
) -> Dict[str, float]:
    """Computes pairwise Jaccard stability separately for each feature branch."""
    if len(masks) < 2:
        return {branch: 1.0 for branch in branch_dims}
    result = {}
    for branch, start, end in _branch_offsets(branch_dims):
        branch_masks = [m[start:end] for m in masks]
        result[branch] = compute_pairwise_jaccard_stability(branch_masks)
    return result


def compute_feature_family_breakdown(mask: np.ndarray, branch_dims: Dict[str, int]) -> Dict[str, int]:
    """Returns the count of selected features belonging to each branch family."""
    return _branch_retained_counts(mask, branch_dims)


def map_mask_to_feature_names(
    mask: np.ndarray, branches: Tuple[str, ...], config: Optional[PSDConfig] = None,
) -> List[str]:
    """Maps selected feature mask indices to canonical Phase 5 column names."""
    from modules.fusion import get_fusion_feature_names
    full_names = get_fusion_feature_names(branches, config)
    if len(full_names) != len(mask):
        raise ValueError(f"Mask length ({len(mask)}) does not match feature names length ({len(full_names)})")
    return [full_names[i] for i in np.where(mask)[0]]


# ============================================================
# Registry
# ============================================================

SELECTOR_REGISTRY = {
    "none": select_no_selection,
    "xgboost_importance": select_xgboost_importance,
    "rfe": select_rfe,
    "ga": select_ga,
    "bda": select_bda,
    "dragonfly": select_bda,
    "dfa": select_bda,
}

_ARG_SHAPES = {
    "none": lambda data, config, seed: (data,),
    "xgboost_importance": lambda data, config, seed: (data, config.feature_selection_top_k, seed),
    "rfe": lambda data, config, seed: (data, config.feature_selection_top_k, config.rfe_step, seed),
    "ga": lambda data, config, seed: (data, config, seed),
    "bda": lambda data, config, seed: (data, config, seed),
    "dragonfly": lambda data, config, seed: (data, config, seed),
    "dfa": lambda data, config, seed: (data, config, seed),
}


def run_selector(name: str, data: FusionFoldData, config: PSDConfig, random_seed: int) -> SelectionResult:
    """The stable Phase-6 selector contract: every caller uses this SAME
    four-argument call regardless of which method 'name' names."""
    if name not in SELECTOR_REGISTRY:
        raise ValueError(f"Unknown feature_selection method '{name}'. Registered: {sorted(SELECTOR_REGISTRY)}")
    fn = SELECTOR_REGISTRY[name]
    build_args = _ARG_SHAPES.get(name, lambda data, config, seed: (data, config, seed))
    return fn(*build_args(data, config, random_seed))


# ============================================================
# Production BDA Mask Interface (Authorized for experimental execution)
# ============================================================

def get_production_bda_mask_path(config: PSDConfig, artifacts_dir: Optional[Path] = None) -> Path:
    """Returns the canonical artifact path for the persisted production BDA mask."""
    base_dir = artifacts_dir if artifacts_dir is not None else config.aef_crc_phase6_artifacts_dir
    return base_dir / "production_bda_mask.joblib"


def save_production_bda_mask(
    mask: np.ndarray,
    config: PSDConfig,
    metadata: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
) -> Path:
    """Persists the single frozen production BDA mask artifact after execution on the
    unified 1146-image cohort.
    Enforces the exact boolean contract (1298-D for A2, 1316-D for A7, or 1348-D for A6).
    """
    import joblib
    from datetime import datetime, timezone
    from modules.calibration_handoff import get_active_run_id
    from modules.experiment_config import representation_id

    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Production BDA mask must be a numpy ndarray, got {type(mask)}")
    if mask.ndim != 1 or mask.shape[0] not in (1298, 1316, 1348):
        raise ValueError(f"Production BDA mask dimension mismatch: expected 1298-D, 1316-D, or 1348-D, got shape {mask.shape}")
    if mask.dtype != bool:
        raise TypeError(f"Production BDA mask must have boolean dtype, got {mask.dtype}")
    if mask.sum() == 0:
        raise ValueError("Production BDA mask selects 0 features; invalid mask.")

    if run_id is None:
        run_id = get_active_run_id(config) or ""

    out_path = get_production_bda_mask_path(config, artifacts_dir=artifacts_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mask": mask,
        "selected_count": int(mask.sum()),
        "total_dim": int(mask.shape[0]),
        "run_id": run_id,
        "representation_id": representation_id(config),
        "random_seed": config.random_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    joblib.dump(payload, out_path)
    return out_path


def load_production_bda_mask(
    config: PSDConfig, expected_run_id: Optional[str] = None, artifacts_dir: Optional[Path] = None,
) -> np.ndarray:
    """Loads and validates the authoritative frozen production BDA mask.
    Fails loudly if the artifact does not exist (does NOT fabricate or re-run).
    """
    import joblib
    from modules.calibration_handoff import get_active_run_id

    path = get_production_bda_mask_path(config, artifacts_dir=artifacts_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Authoritative production BDA mask artifact not found at {path}. "
            f"Production BDA must be executed on the unified 1146-image cohort during the "
            f"experimental execution phase before Phase 7 final retrain can proceed. "
            f"Downstream phases MUST NOT fabricate or silently recompute BDA masks in memory!"
        )

    payload = joblib.load(path)
    if isinstance(payload, dict) and "mask" in payload:
        if expected_run_id is None:
            expected_run_id = get_active_run_id(config)
        mask_run_id = payload.get("run_id")
        if expected_run_id and mask_run_id != expected_run_id:
            raise RuntimeError(
                f"Cross-run artifact mismatch: production BDA mask was generated with "
                f"run_id='{mask_run_id}', but current active run expects '{expected_run_id}'. "
                f"Artifacts from separate experimental runs cannot be combined!"
            )
        if "representation_id" in payload:
            from modules.experiment_config import representation_id
            expected_repr = representation_id(config)
            if payload["representation_id"] != expected_repr:
                raise RuntimeError(
                    f"Cross-run artifact mismatch: production BDA mask was generated with "
                    f"representation_id='{payload['representation_id']}', but current config expects "
                    f"'{expected_repr}'. Run Phase 6 for the current configuration."
                )
        mask = payload["mask"]
    elif isinstance(payload, np.ndarray):
        mask = payload
    else:
        raise TypeError(f"Unexpected production BDA mask payload type: {type(payload)}")

    if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.shape[0] not in (1298, 1316, 1348) or mask.dtype != bool:
        raise ValueError(
            f"Corrupt production BDA mask at {path}: expected 1298-D, 1316-D, or 1348-D boolean array, "
            f"got {type(mask)} with shape {getattr(mask, 'shape', None)} and dtype {getattr(mask, 'dtype', None)}"
        )
    return mask


def run_production_bda(data: FusionFoldData, config: PSDConfig, random_seed: int) -> SelectionResult:
    """Performs ONE final production BDA execution on the unified 1146-image cohort
    using the established inner 80/20 stratified split.
    NOTE: Authorized ONLY during later experimental execution phase, NOT during pre-experiment setup."""
    if len(data.y_train) != 1146:
        raise ValueError(f"Production BDA requires unified cohort of 1146 images, got {len(data.y_train)}")
    return select_bda(data, config, random_seed)
