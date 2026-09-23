# PapuloNet V2 Phase 9: Conformal Prediction Report
**Pipeline**: A7-BDA (1316-D: EfficientNet-B0 + GLCM + LBP + LAB, Random Forest 300 Trees)  
**Temperature Scaling**: Fitted $T = 0.325677$  
**Primary Conformal Method**: Pooled Marginal Split-Conformal Prediction  
**Run ID**: `AEFCRC_P9_V2_20260921_150441_54fc3c7b`  
**Date**: 2026-09-21T15:04:42.744516+00:00  

---

## 1. Executive Summary & Conformal Calibration Outcome

Phase 9 executes **Split-Conformal Prediction** at nominal 90% coverage ($\alpha = 0.10$) over the temperature-calibrated probabilities produced in Phase 8.

### Key Conformal Parameters:
- **Calibration Cohort**: `val_conf` ($N = 123$ samples), strictly disjoint from `val_calib` and the locked test set.
- **Nominal Coverage**: 90.0% ($\alpha = 0.10$)
- **Exact Finite-Sample Quantile Index**: $k = \lceil (123 + 1) \times 0.90 \rceil = 112$
- **Prediction Set Rule**: Include class $c$ if $s(x, c) = (1.0 - P_{cal}(c \mid x)) \le 0.738243$ (nominally $P_{cal}(c \mid x) \ge 0.261757$)

---

## 2. Conformal Summary Table

> [!NOTE]
> **Observed Calibration-Cohort Diagnostic Coverage**:
> The empirical coverage reported below is an observed diagnostic evaluated on the calibration cohort `val_conf` ($N=123$) from which $\hat{q}$ was derived. It must **not** be interpreted as an independent test result or domain generalization claim. Formal conformal coverage guarantees apply to future exchangeable observations under the stated exchangeability assumption.

| Method | Nominal Cov | Observed Calib Diagnostic Cov | Mean Set Size | Median Set Size | Singleton Rate | Empty Set Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Marginal Split-Conformal** | 90.0% | **91.06%** | **1.260** | **1.0** | **73.98%** | **0.00%** |
| **Class-Conditional Mondrian** | 90.0% | **94.31%** | **1.553** | **1.0** | **50.41%** | **0.81%** |

---

## 3. Class-Conditional Mondrian Calibration Quantiles

| Target Class | Support ($n_c$) | $\hat{q}_c$ | Inclusion Threshold ($1 - \hat{q}_c$) | Empirical Coverage | Diagnostic Warning |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Psoriasis` | 69 | `0.564112` | `0.435888` | 91.30% | None |
| `Lichen_Planus` | 28 | `0.838085` | `0.161915` | 96.43% | None |
| `Pityriasis_Rosea` | 17 | `0.980949` | `0.019051` | 100.00% | None |
| `Seborrheic_Dermatitis` | 9 | `0.288240` | `0.711760` | 100.00% | Class 'Seborrheic_Dermatitis' has only 9 calibration samples (recommended >= 15). Empirical quantile k=9 is coarse, causing higher finite-sample variance and potentially wide prediction sets. |

---

## 4. Set Size Distribution

| Set Size ($|C|$) | Marginal Count | Marginal % | Mondrian Count | Mondrian % |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0.00% | 1 | 0.81% |
| 1 | 91 | 73.98% | 62 | 50.41% |
| 2 | 32 | 26.02% | 51 | 41.46% |
| 3 | 0 | 0.00% | 9 | 7.32% |
| 4 | 0 | 0.00% | 0 | 0.00% |

---

## 5. Scientific Interpretation & Critical Caveats

1. **Exchangeability Assumption**:
   - Finite-sample marginal coverage guarantees strictly depend on the assumption that calibration samples and future test samples are exchangeable.
   - Conformal coverage does **not** establish clinical diagnostic validity or guarantee domain generalization.
2. **Singleton Sets ($|C| = 1$)**:
   - A singleton prediction set does **not** equal absolute certainty; it indicates that the nonconformity of all alternative classes exceeded the 90% empirical threshold.
3. **Empty Sets ($|C| = 0$)**:
   - An empty set occurs when no class achieves probability $\ge 1 - \hat{q}$. It signals unusual or ambiguous feature patterns but must **not** be treated as a formally calibrated out-of-distribution (OOD) detector.
4. **Minority Class Variance**:
   - For `Seborrheic_Dermatitis` ($n_c = 9$), the class-conditional quantile index is $k = \lceil 10 \times 0.90 \rceil = 9$. The quantile is determined by the maximum score, resulting in wider prediction sets.

---

## 6. Phase 10 / Final Pipeline Handoff

The complete frozen bundle has been saved to `/mnt/c/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP/artifacts/phase9_v2/final_pipeline_handoff.joblib` for test-set evaluation.
