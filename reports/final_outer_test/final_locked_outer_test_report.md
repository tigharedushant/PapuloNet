# PapuloNet V2 — Final Locked Outer Test Benchmark Report

## 1. Executive Summary

This report documents the final independent evaluation of the certified, frozen PapuloNet V2 pipeline executed on the locked held-out outer test partition ($N = 243$).

- **Evaluation Date**: 2026-09-23T17:46:51Z
- **Evaluated Samples**: $N = 243$ (strictly held out, zero training/validation exposure)
- **Pipeline Handoff**: `artifacts/phase9_v2/final_pipeline_handoff.joblib`
- **Handoff SHA-256**: `d109b1fb798ea1ecda32d9f8723ce1f14b9f7b54b6d57322d61f30d9a5c4309f`
- **Backbone Representation**: `efficientnet_b0_43d581b96f8ec368` (Phase 3 V2 Focal)
- **Multimodal Layout**: A7 (1316-D fused: 1280 Deep + 12 GLCM + 18 LBP + 6 LAB)
- **Selected Feature Subspace**: 642 active BDA features ($K = 642$)
- **Base Classifier**: `RandomForestClassifier(n_estimators=300, random_state=42)`
- **Calibration Method**: Multiclass Temperature Scaling ($T = 0.32567700916361153$)
- **Split-Conformal Threshold**: Marginal $\hat{q} = 0.7382427827337363$

---

## 2. Benchmark Performance Metrics

### A. Global Classification Performance
- **Primary Scientific Metric (Macro-F1)**: **0.8004** (0.8004206751773794)
- **Overall Accuracy**: **81.07%** (0.8106995884773662; 197 / 243 correct)
- **Balanced Accuracy**: **77.28%** (0.7728052584670232)
- **Weighted-F1**: **0.8065** (0.8065322493880954)
- **Matthews Correlation Coefficient (MCC)**: **0.6812** (0.6812477240458646)

### B. Probability Calibration
- **Raw Ensemble ECE**: 0.2898065187803512
- **Temperature-Calibrated ECE**: **0.0291** (0.02910932739085133)
- **ECE Improvement**: +0.26069719138949987
- **Raw Multiclass Brier Score**: 0.4165247678756714
- **Calibrated Multiclass Brier Score**: **0.2942** (0.2942149043083191)
- **Brier Improvement**: +0.1223098635673523
- **Calibrated Negative Log-Likelihood (NLL)**: **0.5263** (0.5263468027114868)

### C. Split-Conformal Prediction Coverage
- **Nominal Target Coverage**: 90.0% ($\alpha = 0.10$)
- **Empirical Coverage**: The 90% nominal split-conformal procedure achieved **88.07%** (0.8806584362139918; 214 / 243) empirical coverage on the 243-image locked outer test set.
- **Coverage Delta**: -1.93% (-0.019341563786008265)
- **Mean Prediction Set Size**: **1.26** (1.2551440329218106)
- **Median Prediction Set Size**: **1.0**
- **Singleton Set Fraction**: **74.5%** (0.7448559670781894; 181 / 243)
- **Ambiguous Set Fraction**: **25.5%** (0.2551440329218107; 62 / 243)
- **Empty Set Fraction**: **0.0%** (0 / 243)

---

## 3. Confusion Matrix and Error Distribution

Canonical Class Order: `['Psoriasis', 'Lichen_Planus', 'Pityriasis_Rosea', 'Seborrheic_Dermatitis']`

```
                      Predicted Pso  Predicted LP  Predicted PR  Predicted SebDerm    Total
True Psoriasis                 123             9             3                  1      136
True Lichen Planus              19            34             2                  0       55
True Pityriasis Rosea            9             0            25                  0       34
True Seborrheic Derm             2             1             0                 15       18
Total Predicted                153            44            30                 16      243
```

### Detailed Directional Error Analysis
- **Lichen Planus**: Among the 21 misclassified Lichen Planus test cases, 19 (90.5%) were predicted as Psoriasis.
- **Pityriasis Rosea**: Among the 9 misclassified Pityriasis Rosea test cases, 9 (100.0%) were predicted as Psoriasis.
- **Seborrheic Dermatitis**: Among the 3 misclassified Seborrheic Dermatitis test cases, 2 were predicted as Psoriasis and 1 as Lichen Planus.
- **Psoriasis**: Among the 13 misclassified Psoriasis test cases, 9 were predicted as Lichen Planus, 3 as Pityriasis Rosea, and 1 as Seborrheic Dermatitis.

---

## 4. Per-Class Benchmark Metrics

| Class Name | Precision | Recall | F1-Score | Support ($N_{\text{true}}$) | Conformal Coverage | Mean Set Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Psoriasis** | 0.8039 | 0.9044 | 0.8512 | 136 | 96.32% | 1.26 |
| **Lichen Planus** | 0.7727 | 0.6182 | 0.6869 | 55 | 76.36% | 1.33 |
| **Pityriasis Rosea** | 0.8333 | 0.7353 | 0.7812 | 34 | 76.47% | 1.18 |
| **Seborrheic Dermatitis** | 0.9375 | 0.8333 | 0.8824 | 18 | 83.33% | 1.17 |
| **Macro Average** | **0.8369** | **0.7728** | **0.8004** | 243 | **88.07%** | **1.26** |

---

## 5. Development CV vs. Locked Test Comparison

| Metric | Phase 7 Development 5-Fold CV ($N = 1,146$) | Locked Outer Test ($N = 243$) | Numerical Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Macro-F1 (Primary)** | $0.7089 \pm 0.0336$ | **0.8004** | +0.0915 |
| **Balanced Accuracy** | $0.7065 \pm 0.0261$ | **0.7728** | +0.0663 |
| **Overall Accuracy** | $0.7496 \pm 0.0245$ | **0.8107** | +0.0611 |
| **Weighted-F1** | $0.7457 \pm 0.0247$ | **0.8065** | +0.0608 |
| **MCC** | $0.5873 \pm 0.0420$ | **0.6812** | +0.0939 |

---

## 6. Scientific Integrity & Data Governance Statement

1. **Number of test samples evaluated**: Exactly 243.
2. **Were any test samples used for model selection, parameter tuning, or calibration?**: NO.
3. **Did any model artifact or handoff change during or after evaluation?**: NO.
4. **Handoff immutability**: Verified bitwise identical via SHA-256 (`d109b1fb798ea1ecda32d9f8723ce1f14b9f7b54b6d57322d61f30d9a5c4309f`).
5. **Execution mode**: Single forward evaluation pass using the frozen production pipeline.
