# AEF-CRC V1 — INDEPENDENT VERIFICATION OF PLATT ARGMAX INVERSION

**Authoritative Forensic Verification Report**  
**Date**: September 19, 2026  
**Target Repository**: `C:\Users\ASUS\Downloads\PSD_HP_AEF_CRC_PopuloNet\PSD_HP`  
**Status**: INDEPENDENT VERIFICATION COMPLETE (ZERO CODE MODIFICATIONS; ZERO RETRAINING)

---

## 1. Executive Verdict on Major Audit Claims

| Claim from Forensic Audit | Claimed Value | Independently Verified Value | Status |
| :--- | :---: | :---: | :---: |
| **Total prediction flips from raw RF to calibrated argmax** | 97 / 246 (39.43%) | **97 / 246 (39.43%)** | **VERIFIED** |
| **Non-Psoriasis $\to$ Psoriasis flips** | 97 | **97** | **VERIFIED** |
| **Psoriasis $\to$ Non-Psoriasis reverse flips** | 0 | **0** | **VERIFIED** |
| **Other-class $\to$ Other-class flips** | 0 | **0** | **VERIFIED** |
| **Total Psoriasis predictions under calibrated argmax** | 238 / 246 (96.75%) | **238 / 246 (96.75%)** | **VERIFIED** |
| **Total Psoriasis predictions under raw RF argmax** | 141 / 246 (57.32%) | **141 / 246 (57.32%)** | **VERIFIED** |
| **True Psoriasis count in calibration set** | 137 / 246 (55.69%) | **137 / 246 (55.69%)** | **VERIFIED** |
| **Macro-F1 collapse under calibrated argmax** | Severe drop | **0.7299 (Raw) $\to$ 0.2892 (Calibrated)** | **VERIFIED** |
| **Balanced Accuracy collapse under calibrated argmax** | Severe drop | **0.7468 (Raw) $\to$ 0.3094 (Calibrated)** | **VERIFIED** |

---

## 2. Exact Implementation in Production Code

We traced the exact execution path across the frozen codebase:

### A. Production Base Classifier Inference
- **File**: [`modules/inference.py`](file:///c:/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP/modules/inference.py)
- **Method**: `AEFCRCInferenceEngine.predict_raw_probabilities(masked_features)` (Lines 334–338)
- **Code**:
  ```python
  return self.classifier.predict_proba(masked_features)
  ```
- **Classifier**: `RandomForestClassifier(n_estimators=300, criterion='gini', max_features='sqrt', bootstrap=True, class_weight=None, random_state=42)` (`artifacts/phase9/final_pipeline_handoff.joblib`).
- **Output**: 4-dimensional raw ensemble vote proportions vector `raw_probs` $[p_0, p_1, p_2, p_3]$ summing strictly to $1.0$.

### B. Platt Scaling Transformation
- **File**: [`modules/calibration.py`](file:///c:/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP/modules/calibration.py)
- **Function**: `apply_platt_scaling(probs, fit, class_order)` (Lines 158–171)
- **Code**:
  ```python
  calibrated = np.zeros_like(probs)
  for k, cls in enumerate(class_order):
      model = fit.models.get(cls)
      calibrated[:, k] = probs[:, k] if model is None else model.predict_proba(probs[:, k].reshape(-1, 1))[:, 1]
  row_sums = calibrated.sum(axis=1, keepdims=True)
  row_sums = np.where(row_sums <= 0, 1.0, row_sums)
  return calibrated / row_sums
  ```
- **Mechanism**: Evaluates each class's raw probability through an independent 1D Logistic Regression model:
  $$s_k = \sigma(w_k \cdot p_k + b_k) = \frac{1}{1 + e^{-(w_k \cdot p_k + b_k)}}$$
- **Normalization**: Renormalizes the four uncoupled sigmoid outputs by dividing by their row sum:
  $$p_k^{\text{calib}} = \frac{s_k}{\sum_{j=1}^{4} s_j}$$

### C. Where `pred_class` Is Selected (The Critical Anomaly)
- **File**: [`modules/inference.py`](file:///c:/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP/modules/inference.py)
- **Method**: `AEFCRCInferenceEngine.predict` (Lines 444–449)
- **Code**:
  ```python
  # 8. Point prediction and confidence
  pred_idx = int(np.argmax(calib_probs[0]))
  pred_class = self.classes[pred_idx]
  calib_conf = float(calib_probs[0][pred_idx])
  raw_conf = float(raw_probs[0][pred_idx])
  ```
- **Finding**: **Production inference explicitly takes the argmax of `calib_probs`, NOT `raw_probs`.**
- Furthermore, Line 448 indexes `raw_probs[0][pred_idx]` at the calibrated argmax index, which explains why Phase 10 reported low "raw confidence" ($17.0\% - 24.7\%$) for samples whose headline prediction was Psoriasis!

---

## 3. Frozen Platt Model Parameters & Mathematical Proof

The Platt calibrators stored in `artifacts/phase9/final_pipeline_handoff.joblib` were fitted on the 123-sample `val_calib` split during Phase 8.

### Exact Parameter Values

| Target Class | Slope ($w_k$) | Intercept ($b_k$) | Class Base Rate in `val_calib` | Calibration Split Prevalence |
| :--- | :---: | :---: | :---: | :---: |
| **Psoriasis** | **+3.5610** | **-1.2045** | **68 / 123** | **55.28%** |
| **Lichen Planus** | +2.4140 | -1.9706 | 28 / 123 | 22.76% |
| **Pityriasis Rosea** | +2.9705 | -2.4044 | 18 / 123 | 14.63% |
| **Seborrheic Dermatitis** | +3.1697 | **-2.9611** | **9 / 123** | **7.32%** |

### Mathematical Mechanism: Log-Odds Intercept Skew
The 1D logistic regressions learn intercepts that directly encode the empirical class prior log-odds:
$$\Delta b_{\text{Pso} - \text{SD}} = -1.2045 - (-2.9611) = +1.7566 \implies e^{1.7566} \approx \mathbf{5.79}$$
$$\Delta b_{\text{Pso} - \text{PR}} = -1.2045 - (-2.4044) = +1.1999 \implies e^{1.1999} \approx \mathbf{3.32}$$
$$\Delta b_{\text{Pso} - \text{LP}} = -1.2045 - (-1.9706) = +0.7661 \implies e^{0.7661} \approx \mathbf{2.15}$$

Even if the Random Forest outputs identical probabilities across all classes, Platt scaling multiplies the odds of Psoriasis by **$5.79\times$** relative to Seborrheic Dermatitis, **$3.32\times$** relative to Pityriasis Rosea, and **$2.15\times$** relative to Lichen Planus!

---

## 4. Mathematical Verification on Specific Cases

### Test Case A: Perfectly Uniform Raw Probability ($p_{\text{raw}} = [0.25, 0.25, 0.25, 0.25]$)
When the underlying model has complete uncertainty:

1. **Unnormalized Sigmoid Outputs**:
   - Psoriasis: $\sigma(3.5610 \times 0.25 - 1.2045) = \sigma(-0.31425) = \mathbf{0.42211}$
   - Lichen Planus: $\sigma(2.4140 \times 0.25 - 1.9706) = \sigma(-1.36710) = \mathbf{0.20309}$
   - Pityriasis Rosea: $\sigma(2.9705 \times 0.25 - 2.4044) = \sigma(-1.66178) = \mathbf{0.15951}$
   - Seborrheic Dermatitis: $\sigma(3.1697 \times 0.25 - 2.9611) = \sigma(-2.16868) = \mathbf{0.10263}$
2. **Sum of Sigmoids**:
   $$S = 0.42211 + 0.20309 + 0.15951 + 0.10263 = \mathbf{0.88734}$$
3. **Normalized Calibrated Vector**:
   - **Psoriasis**: $0.42211 / 0.88734 = \mathbf{47.57\%}$
   - **Lichen Planus**: $0.20309 / 0.88734 = \mathbf{22.89\%}$
   - **Pityriasis Rosea**: $0.15951 / 0.88734 = \mathbf{17.98\%}$
   - **Seborrheic Dermatitis**: $0.10263 / 0.88734 = \mathbf{11.56\%}$

**Verdict**: A uniform raw prediction is transformed into a **$47.57\%$ Psoriasis prediction**, establishing Psoriasis as the argmax with more than double the probability of the runner-up.

---

### Test Case B: Audit Example (Raw LP = 35%, Raw Psoriasis = 20%)
Consider an input where the Random Forest clearly favors **Lichen Planus**:
$$p_{\text{raw}} = [p_{\text{Pso}} = 0.20, p_{\text{LP}} = 0.35, p_{\text{PR}} = 0.25, p_{\text{SD}} = 0.20]$$
- **Raw Top-1 Decision**: **Lichen Planus (35%)**

1. **Unnormalized Sigmoid Outputs**:
   - Psoriasis: $\sigma(3.5610 \times 0.20 - 1.2045) = \sigma(-0.49230) = \mathbf{0.37936}$
   - Lichen Planus: $\sigma(2.4140 \times 0.35 - 1.9706) = \sigma(-1.12570) = \mathbf{0.24496}$
   - Pityriasis Rosea: $\sigma(2.9705 \times 0.25 - 2.4044) = \sigma(-1.66178) = \mathbf{0.15951}$
   - Seborrheic Dermatitis: $\sigma(3.1697 \times 0.20 - 2.9611) = \sigma(-2.32716) = \mathbf{0.08888}$
2. **Sum of Sigmoids**:
   $$S = 0.37936 + 0.24496 + 0.15951 + 0.08888 = \mathbf{0.87271}$$
3. **Normalized Calibrated Vector**:
   - **Psoriasis**: $0.37936 / 0.87271 = \mathbf{43.47\%}$
   - **Lichen Planus**: $0.24496 / 0.87271 = \mathbf{28.07\%}$
   - **Pityriasis Rosea**: $0.15951 / 0.87271 = \mathbf{18.28\%}$
   - **Seborrheic Dermatitis**: $0.08888 / 0.87271 = \mathbf{10.19\%}$

**Verdict**: Despite the base classifier giving **Lichen Planus a 15-point lead (35% vs 20%)**, Platt scaling **inverts the ranking**, outputting **Psoriasis at 43.47%** and demoting Lichen Planus to 28.07%!

---

## 5. Reproduction of the 246-Sample Calibration Split Analysis

We evaluated the exact 246 samples from `artifacts/phase7/calibration_handoff.joblib` (`h7.calibration_raw_probabilities` and `h7.calibration_true_labels`) using the frozen Platt models.

### Confusion Matrices

#### Raw Random Forest Argmax Confusion Matrix
```
True \ Pred             Psoriasis  Lichen_Planus  Pityriasis_Rosea  Seborrheic_Dermatitis    Support
Psoriasis                     113             12                 6                      6        137
Lichen_Planus                  17             32                 5                      1         55
Pityriasis_Rosea               10              1                24                      0         35
Seborrheic_Dermatitis           1              0                 1                     17         19
Total Predicted               141             45                36                     24        246
```

#### Platt Calibrated Argmax Confusion Matrix
```
True \ Pred             Psoriasis  Lichen_Planus  Pityriasis_Rosea  Seborrheic_Dermatitis    Support
Psoriasis                     137              0                 0                      0        137
Lichen_Planus                  53              1                 1                      0         55
Pityriasis_Rosea               31              0                 4                      0         35
Seborrheic_Dermatitis          17              0                 0                      2         19
Total Predicted               238              1                 5                      2        246
```

---

### Class Prediction Distribution Comparison

| Class | Ground Truth Count | Raw RF Predictions | Calibrated Argmax Predictions | Net Change ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Psoriasis** | 137 (55.69%) | 141 (57.32%) | **238 (96.75%)** | **+97 (+68.8%)** |
| **Lichen Planus** | 55 (22.36%) | 45 (18.29%) | **1 (0.41%)** | **-44 (-97.8%)** |
| **Pityriasis Rosea** | 35 (14.23%) | 36 (14.63%) | **5 (2.03%)** | **-31 (-86.1%)** |
| **Seborrheic Dermatitis** | 19 (7.72%) | 24 (9.76%) | **2 (0.81%)** | **-22 (-91.7%)** |

---

### Comprehensive Performance Metrics Comparison

| Metric | Raw RF Argmax | Calibrated Argmax | Absolute Impact | Diagnostic Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Macro-F1** | **0.7299** | **0.2892** | **-0.4407** | **Catastrophic Collapse** |
| **Balanced Accuracy** | **0.7468** | **0.3094** | **-0.4373** | **Catastrophic Collapse** |
| **Overall Accuracy** | **0.7561** | **0.5854** | **-0.1707** | Severe Drop |
| **MCC** | **0.6023** | **0.2135** | **-0.3888** | Severe Drop |

---

### Detailed Flips Breakdown
- **Total Flips**: **97 / 246 (39.43%)**
- **Non-Psoriasis $\to$ Psoriasis**: **97**
  - Raw Lichen Planus $\to$ Platt Psoriasis: **44**
  - Raw Pityriasis Rosea $\to$ Platt Psoriasis: **31**
  - Raw Seborrheic Dermatitis $\to$ Platt Psoriasis: **22**
- **Psoriasis $\to$ Non-Psoriasis**: **0**
- **Other $\to$ Other**: **0**

### Split Partition Breakdown
- On `val_calib` (123 samples used to fit Platt):
  - Flips to Psoriasis: **53 / 123 (43.09%)**
  - Calibrated Psoriasis predictions: **119 / 123 (96.75%)**
  - Macro-F1: $0.7197 \to 0.2701$ (matches `reports/phase8/calibration_comparison.csv` row 3 exactly)
- On `val_conf` (123 holdout samples handed to conformal prediction):
  - Flips to Psoriasis: **44 / 123 (35.77%)**
  - Calibrated Psoriasis predictions: **119 / 123 (96.75%)**
  - Macro-F1: $0.7387 \to 0.3085$

---

## 6. Statistical Consequence of Independent Sigmoid Renormalization

### Do the Sigmoids Sum to 1?
**NO.** Because the four Platt models are fit as independent One-vs-Rest binary regressions, their outputs do not constitute a multinomial distribution.
On the 246 calibration samples, the unnormalized sum of sigmoids ($\sum_{k=1}^4 s_k$) has:
- **Minimum**: **0.8459**
- **Maximum**: **1.1405**
- **Mean**: **1.0039**
- **Standard Deviation**: **0.0692**

### Statistical Mechanism of the Artifact
When independent binary sigmoids are fitted on imbalanced data:
1. The intercept $b_k$ approximates $\text{logit}(\pi_k) = \log\frac{\pi_k}{1 - \pi_k}$, where $\pi_k$ is the class prevalence.
2. In multiclass settings, normalizing independent sigmoids via $p_k = s_k / \sum s_j$ assumes that the relative odds remain calibrated.
3. However, when the underlying classifier is uncertain, the slopes $w_k \cdot p_k$ are small, and the output is almost entirely governed by $b_k$.
4. Dividing by $\sum s_j$ does not cancel out the disparity between $b_{\text{Pso}} = -1.20$ and $b_{\text{SD}} = -2.96$. Instead, it locks the output near the prior distribution:
   $$\text{Prior-Dominated Output: } [47.6\%, 22.9\%, 18.0\%, 11.6\%]$$
5. Taking the argmax of this normalized vector guarantees that **any ambiguous image will be classified as Psoriasis**.

---

## 7. Phase 8 Model Selection Review

We inspected `reports/phase8/calibration_comparison.csv`:
```csv
method,role,nll,brier,ece,macro_f1_diagnostic
uncalibrated,baseline,0.8065,0.4492,0.2488,0.7197
platt,PRIMARY,0.8833,0.4902,0.1479,0.2701
isotonic,SECONDARY,0.4657,0.2908,0.0533,0.7876
```

### Why Platt Was Chosen
1. **Preregistration**: The research protocol explicitly designated Platt scaling as the **PRIMARY** method. Line 14 of `run_aef_crc_phase8.py` states:
   > *"No adaptive selection, ECE threshold (e.g. >= 0.01), or data-dependent cascade is permitted to replace Platt as the primary calibration method."*
2. **Metric Focus**: Calibration quality was judged strictly by **ECE** (Expected Calibration Error) and Brier score. Platt reduced ECE from $0.2488$ to $0.1479$.
3. **The Blind Spot**: Phase 8 treated `macro_f1_diagnostic` as a supporting observation, assuming that downstream inference would use the Random Forest's class decisions. It was never intended that `argmax(calibrated_probs)` would override the classifier's decisions in production!

---

## 8. Separation of Concepts: Calibration vs. Decision Rule vs. Conformal Prediction

To resolve this issue cleanly, three distinct components must be separated:

1. **Point-Class Decision Rule (Classification)**:
   - Purpose: Determine which class label $\hat{y} \in \mathcal{Y}$ best matches the image.
   - Mechanism: Should reflect the trained classifier's discriminant function ($\operatorname{argmax}_{k} P_{\text{raw}}(Y=k \mid X)$) under equal loss.
2. **Probability Calibration (Confidence Assessment)**:
   - Purpose: Map raw ensemble vote scores to true empirical frequencies ($P(\text{correct} \mid \text{conf} = c) \approx c$).
   - Flaw in current code: Because Platt scaling incorporates uncoupled base-rate priors, it distorts relative rankings under high entropy.
3. **Conformal Prediction (Uncertainty & Safety Gating)**:
   - Purpose: Guarantee $(1 - \alpha)$ coverage by outputting a set of plausible classes.
   - At threshold $1 - \hat{q} = 0.2178$, conformal prediction correctly flags diffuse cases as multi-class (`SPECIALIST_REVIEW_REQUIRED`). It is functioning as intended.

---

## 9. External Image Amplification

On external images (e.g. from Google or clinical mobile photos), domain shift naturally causes the Random Forest to output diffuse probabilities, such as:
$$p_{\text{raw}} = [\text{Pso: } 0.20, \text{LP: } 0.25, \text{PR: } 0.30, \text{SD: } 0.25]$$
Under Platt transformation:
- Psoriasis logit: $3.561 \times 0.20 - 1.205 = -0.4923 \implies s_{\text{Pso}} = 0.3794$
- PR logit: $2.971 \times 0.30 - 2.404 = -1.5131 \implies s_{\text{PR}} = 0.1805$
- LP logit: $2.414 \times 0.25 - 1.971 = -1.3675 \implies s_{\text{LP}} = 0.2030$
- SD logit: $3.170 \times 0.25 - 2.961 = -2.1686 \implies s_{\text{SD}} = 0.1026$
- Normalized:
  $$\text{Psoriasis: } \mathbf{43.83\%} \quad\mid\quad \text{LP: } 23.45\% \quad\mid\quad \text{PR: } 20.85\% \quad\mid\quad \text{SD: } 11.87\%$$

Even though the Random Forest believed **Pityriasis Rosea was the most likely class (30%)** and Psoriasis was the least likely (20%), Platt calibration forces **Psoriasis into the argmax with 43.8%**!
This conclusively explains why users observe calibrated probabilities in the **40%–57% range** collapsing to Psoriasis on non-Psoriasis images.

---

## 10. Status of the Proposed Raw-Argmax Rule

The proposed change:
$$\text{pred\_class} = \text{classes}[\operatorname{argmax}(\text{raw\_probs}[0])]$$
is **NOT merely a speculative hypothesis**.

It is **directly supported by the empirical evidence**:
1. On the exact 246 calibration samples:
   - Raw Argmax Macro-F1: **0.7299** (vs. **0.2892** Calibrated Argmax)
   - Raw Argmax Balanced Accuracy: **0.7468** (vs. **0.3094** Calibrated Argmax)
   - Raw Argmax Psoriasis Predictions: **141 / 246 (57.32%)** (vs. **238 / 246 (96.75%)** Calibrated Argmax)
2. In 5-fold cross-validation (`run_final_cv` in Phase 7):
   - The reported Macro-F1 of **0.6906** was computed using **raw RF argmax** (`clf.predict()`).
   - Using calibrated argmax in production was a silent behavioral divergence that introduced the 0.2892 collapse into deployment.

---

## 11. Verification Deliverables Created on Disk

1. Comprehensive Report: [`reports/final_bias_audit/platt_argmax_verification.md`](file:///c:/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP/reports/final_bias_audit/platt_argmax_verification.md)
2. Sample-by-Sample Comparison CSV: [`reports/final_bias_audit/platt_argmax_comparison.csv`](file:///c:/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP/reports/final_bias_audit/platt_argmax_comparison.csv)
3. Full Metric JSON: [`reports/final_bias_audit/platt_verification_metrics.json`](file:///c:/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP/reports/final_bias_audit/platt_verification_metrics.json)

---

## 12. Absolute Stop Condition

All verifications have been conducted strictly through read-only inspection of frozen artifacts.
- No code was edited.
- No models were retrained.
- No V1 artifacts were modified.
- The locked 243-image test partition was not accessed.
- Execution has stopped cleanly for your review.
