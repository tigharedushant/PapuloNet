# PapuloNet V2 Phase 8: Probability Calibration Report
**Pipeline**: A7-BDA (1316-D: EfficientNet-B0 + GLCM + LBP + LAB, Random Forest 300 Trees)  
**Primary Calibration Method**: Multiclass Temperature Scaling (Guo et al., 2017)  
**Run ID**: `AEFCRC_P8_V2_20260921_124809_c9bee427`  
**Date**: 2026-09-21T12:48:09.496611+00:00  

---

## 1. Executive Summary & Calibration Outcome

Phase 8 executes **Multiclass Temperature Scaling** to calibrate predicted class probabilities from the frozen Phase 7 V2 winner pipeline (A7-BDA Random Forest).

### Core Scientific Findings:
1. **Strict Class Decision Invariance**:
   - The number of changed class predictions is **exactly 0** across all 246 validation images (0 on `val_calib`, 0 on `val_conf`).
   - Macro-F1 (0.8001), Balanced Accuracy, MCC, and the confusion matrix are mathematically preserved.
2. **Rejection of V1 Platt Scaling**:
   - V1 independently fit 4 binary logistic regressions, which scrambled multiclass logit rankings and collapsed 238/246 predictions to Psoriasis.
   - Temperature scaling applies a single positive scalar temperature $T > 0$ over pseudo-logits, mathematically guaranteeing that $\operatorname{argmax} \hat{p} \equiv \operatorname{argmax} p$.
3. **Optimal Temperature**:
   - $T = 0.3257$ (fitted on 123 `val_calib` images by unweighted NLL minimization).
   - Fit Time: 0.0074s.
4. **Generalization on Held-Out `val_conf` (N=123)**:
   - **NLL**: 0.7383 $\rightarrow$ **0.4967** (-0.2416)
   - **ECE (10 bins)**: 0.2849 $\rightarrow$ **0.0877** (-0.1972)
   - **Brier Score**: 0.4028 $\rightarrow$ **0.2916** (-0.1112)

---

## 2. Calibration Comparison Table

| Partition | Set Role | Method | N | NLL | Brier Score | ECE (10 bins) | Macro-F1 | Changed Decisions |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `val_calib` | Fit Set | Uncalibrated | 123 | 0.7678 | 0.4249 | 0.2553 | 0.7320 | 0 |
| `val_calib` | Fit Set | Temperature Scaling ($T=0.3257$) | 123 | **0.5441** | **0.3294** | **0.0736** | **0.7320** | **0** |
| `val_conf` | Generalization | Uncalibrated | 123 | 0.7383 | 0.4028 | 0.2849 | 0.8001 | 0 |
| `val_conf` | Generalization | Temperature Scaling ($T=0.3257$) | 123 | **0.4967** | **0.2916** | **0.0877** | **0.8001** | **0** |
| `val_pooled`| Pooled Cohort | Uncalibrated | 246 | 0.7530 | 0.4138 | 0.2701 | 0.7643 | 0 |
| `val_pooled`| Pooled Cohort | Temperature Scaling ($T=0.3257$) | 246 | **0.5204** | **0.3105** | **0.0528** | **0.7643** | **0** |

---

## 3. Reliability Diagram Analysis (`val_conf`, N=123)

### Calibrated 10-Bin Breakdown:
| Bin Index | Range | Count | Mean Confidence | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 0 | [0.00, 0.10) | 0 | 0.0000 | 0.0000 |
| 1 | [0.10, 0.20) | 0 | 0.0000 | 0.0000 |
| 2 | [0.20, 0.30) | 0 | 0.0000 | 0.0000 |
| 3 | [0.30, 0.40) | 0 | 0.0000 | 0.0000 |
| 4 | [0.40, 0.50) | 3 | 0.4697 | 1.0000 |
| 5 | [0.50, 0.60) | 16 | 0.5533 | 0.8125 |
| 6 | [0.60, 0.70) | 17 | 0.6435 | 0.4118 |
| 7 | [0.70, 0.80) | 24 | 0.7512 | 0.7917 |
| 8 | [0.80, 0.90) | 19 | 0.8475 | 0.8421 |
| 9 | [0.90, 1.00) | 44 | 0.9555 | 0.9545 |

---

## 4. Verification of Invariants & Data Isolation

1. **Class Decision Invariance**:
   - $\operatorname{argmax}_{k} \hat{p}_{i, k} \equiv \operatorname{argmax}_{k} p_{i, k}$ verified across all 246 validation instances.
   - Total changed predictions: **0**.
2. **Strict Data Partitioning**:
   - Fit set `val_calib`: 123 images.
   - Held-out evaluation set `val_conf`: 123 images.
   - Leakage: strictly 0 images overlapping between `val_calib` and `val_conf`.
   - Development cohort (1146 images): untouched.
   - Locked test set (243 images): strictly untouched and unread.
3. **Upstream Provenance**:
   - Phase 7 V2 Run ID: `AEFCRC_P7_V2_20260921_074018_936e6e73`
   - Dataset Freeze Hash: `964d91f8178d5bb7123a7cc02388cd57912bba1e836b2a3ab70b9fa98441919f`
   - Fold Plan Hash: `3c54f09f834632f8f771b46d5a98fb773aa6ee78878b00a0c58806c575b8f7b9`
   - Backbone Representation ID: `efficientnet_b0_43d581b96f8ec368`

---

## 5. Phase 9 Conformal Handoff

The calibrated probabilities for `val_conf` (123 images) and the fitted `TemperatureScaler` have been packaged into `artifacts/phase8_v2/conformal_handoff.joblib`.  
Phase 9 will use these calibrated probabilities directly for conformal prediction set generation.
