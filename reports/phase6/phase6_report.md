# AEF-CRC Phase 6: Feature Refinement Benchmark Report

## 1. Executive Summary
Phase 6 evaluated metaheuristic feature selection algorithms on the winning Phase 5 representation **A7 (EfficientNet+GLCM+LBP+LAB, 1316-D)**.
The primary optimizer, **Binary Dragonfly Algorithm (BDA)**, was benchmarked against the comparator, **Genetic Algorithm (GA)**, across all 5 frozen cross-validation folds under strict nested split isolation.

- **Primary Optimizer (BDA)**: Macro-F1 = **0.6994 ± 0.0287**, Mean Features = **183.2** (86.1% compression), Jaccard stability = **0.0761**.
- **Comparator (GA)**: Macro-F1 = **0.6947 ± 0.0207**, Mean Features = **328.2** (75.1% compression).
- **Inferential Comparison**: Pooled McNemar test between BDA and GA yielded **p = 0.9247** (not significant). BDA achieved parity in classification performance while selecting **44.2% fewer features** than GA.
- **Production Mask**: BDA was executed on the unified 1146-image non-augmented cohort, selecting **194 features** (`deep`: 190, `glcm`: 1, `lbp`: 3, `color_lab`: 0) with a compression ratio of **85.26%**, persisted at `artifacts/phase6/production_bda_mask.joblib`.

## 2. 5-Fold Cross-Validation Results

| Fold | BDA Features | BDA Macro-F1 | GA Features | GA Macro-F1 |
|---|---|---|---|---|
| Fold 0 | 238 | 0.7193 | 351 | 0.6921 |
| Fold 1 | 97 | 0.6727 | 254 | 0.7168 |
| Fold 2 | 181 | 0.7017 | 467 | 0.6808 |
| Fold 3 | 212 | 0.6629 | 336 | 0.6653 |
| Fold 4 | 188 | 0.7403 | 233 | 0.7187 |
| **Mean ± Std** | **183.2 ± 52.4** | **0.6994 ± 0.0287** | **328.2 ± 91.1** | **0.6947 ± 0.0207** |

## 3. Stability & Statistical Comparison
- **Pairwise Jaccard Stability (BDA)**: `mean = 0.0761`.
- **Pooled McNemar Test (BDA vs GA)**: `p = 0.9247` (no significant performance difference).

## 4. Production BDA Feature Mask
- **Artifact**: `artifacts/phase6/production_bda_mask.joblib`
- **Total Input Dimensions**: 1316
- **Selected Dimensions**: 194 (14.74% retained, 85.26% pruned)
- **Family Breakdown**:
  - `deep` (EfficientNet): 190 / 1280 (14.8%)
  - `glcm` (Haralick Texture): 1 / 12 (8.3%)
  - `lbp` (Local Binary Patterns): 3 / 18 (16.7%)
  - `color_lab` (Color Moments): 0 / 6 (0.0%)
- **Optimization Time**: 1240.58 seconds (20.68 minutes)
- **Population**: 20 dragonflies, 30 iterations (600 fitness evaluations).
