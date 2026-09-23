# PapuloNet V2 Phase 7: Model Selection & Evaluation Report

**Generated:** 2026-09-21T07:40:18.199035+00:00  
**Pipeline Candidate:** A7-BDA (`EfficientNet+GLCM+LBP+LAB` with Binary Dragonfly Algorithm)  
**Classifier:** Random Forest (300 trees, balanced weights)  
**Gate Decision:** `CERTIFIED_WINNER` (Pipeline Selected: `A7-BDA`)  

## 1. 5-Fold Stratified Cross-Validation Benchmark

| Metric | Mean +/- Std |
|---|---|
| **Macro-F1** | **0.7089 +/- 0.0336** |
| Balanced Accuracy | 0.7065 +/- 0.0261 |
| MCC | 0.5873 +/- 0.0367 |
| Accuracy | 0.7496 +/- 0.0226 |
| Weighted F1 | 0.7457 +/- 0.0229 |
| Selected Dimensions | 639.4 +/- 140.6 / 1316 |

## 2. Per-Class Generalization

| Class | Precision | Recall | F1 | Total Support |
|---|---|---|---|---|
| Psoriasis | 0.7885 +/- 0.0264 | 0.8385 +/- 0.0262 | **0.8124 +/- 0.0202** | 638 |
| Lichen_Planus | 0.7168 +/- 0.0375 | 0.5739 +/- 0.0842 | **0.6323 +/- 0.0401** | 258 |
| Pityriasis_Rosea | 0.6718 +/- 0.0585 | 0.6979 +/- 0.0501 | **0.6831 +/- 0.0442** | 162 |
| Seborrheic_Dermatitis | 0.7131 +/- 0.1211 | 0.7157 +/- 0.1193 | **0.7078 +/- 0.1005** | 88 |

## 3. Final Retrain & Calibration-Set Performance (Holdout Val, N=246)

- **Unified Training Images:** 1,146 (all outer folds reunified, zero quarantined)
- **Authoritative Production Mask:** 642 / 1316 features ({'deep': 626, 'glcm': 7, 'lbp': 6, 'color_lab': 3})
- **Calibration Partition:** 246 images (splitter.py held-out val/)
- **Calibration Macro-F1:** 0.7643
- **Calibration Balanced Accuracy:** 0.7633
- **Calibration MCC:** 0.6573

## 4. Dominance Diagnostics

- **Psoriasis True Ratio:** 55.67% (638 / 1146)
- **Psoriasis Pred Ratio:** 59.25% (679 / 1146)
- **Dominance Ratio:** **1.0643** (Gating threshold: <= 1.15)

## 5. Predefined Gating Evaluation

- **macro_f1_generalization:** PASS
- **minority_class_viability:** PASS
- **psoriasis_dominance_mitigated:** PASS
- **calibration_generalization:** PASS
