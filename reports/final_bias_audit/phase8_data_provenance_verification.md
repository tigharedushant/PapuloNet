# Phase 8 Data Provenance & Calibration Verification Report
**AEF-CRC / PapuloNet Dermatology Diagnosis Pipeline**

- **Project Root**: `/mnt/c/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP`
- **Audit Date**: 2026-09-19
- **Status**: **VERIFIED (READ-ONLY INSPECTION)**
- **Locked Test Isolation**: **100% UNTOUCHED (0 / 243 Overlap)**
- **Code & Model State**: **ZERO CODE MODIFICATIONS / ZERO RETRAINING**

---

## 1. Executive Summary of Provenance Findings

An exhaustive forensic inspection of the Phase 8 calibration artifacts (`artifacts/phase7/calibration_handoff.joblib`, `artifacts/phase8/conformal_handoff.joblib`, `artifacts/phase9/final_pipeline_handoff.joblib`), reports (`reports/phase8/calibration_comparison.csv`, `validation_partition_summary.csv`), and source code (`modules/calibration.py`, `run_aef_crc_phase8.py`) was conducted to trace the exact data provenance of the calibration metrics.

### Key Confirmations:
1. **The Outer Validation Set ($N=246$)** originates from `output/06_final_split/val/` and is strictly disjoint from all cross-validation training folds (Union of folds = 1,146 development images).
2. **Stratified 50/50 Partition ($123 + 123$)**:
   - **Fit Set (`val_calib` / $D_{\text{prob}}$, $N=123$)**: Used exclusively to fit the 4 independent binary Platt sigmoids and secondary Isotonic calibrators.
   - **Held-Out Evaluation Set (`val_conf` / $D_{\text{conf}}$, $N=123$)**: Held out during calibrator fitting; transformed via fitted Platt models and handed to Phase 9 for conformal quantile estimation.
3. **Metric Reconciliation**:
   - The reported **Macro-F1 = 0.7299**, **Platt Macro-F1 = 0.2892**, **97 / 246 flips**, and **238 / 246 Psoriasis predictions** were calculated on **C) ALL 246 COMBINED OUTER VALIDATION SAMPLES**.
   - The Phase 8 report `reports/phase8/calibration_comparison.csv` logged **uncalibrated Macro-F1 = 0.7197** and **Platt Macro-F1 = 0.2701**, which were evaluated strictly on the **Fit Set (`val_calib`, $N=123$)**.
   - On the **Held-Out Evaluation Set (`val_conf`, $N=123$)**, the Raw RF Macro-F1 is **0.7387** and Platt Macro-F1 is **0.3085**.
4. **Consistency of Collapse Across Both Partitions**:
   The collapse to Psoriasis is identical on both subsets:
   - On `val_calib` (Fit Set, $N=123$): **119 / 123 (96.75%)** predicted Psoriasis; 53 non-Pso $\to$ Pso flips, 0 reverse flips.
   - On `val_conf` (Held-Out, $N=123$): **119 / 123 (96.75%)** predicted Psoriasis; 44 non-Pso $\to$ Pso flips, 0 reverse flips.
   - Combined ($N=246$): **238 / 246 (96.75%)** predicted Psoriasis; 97 non-Pso $\to$ Pso flips, 0 reverse flips.
   - This proves the failure is **not an overfit artifact**; it generalizes identically to completely unseen held-out validation data.
5. **Locked Test Protection**:
   The 243-image locked test set (`output/06_final_split/test/`) has **exactly 0 overlap** with `val_calib`, `val_conf`, or the combined calibration cohort.

---

## 2. Cohort Accounting & Provenance Breakdown

### A. The 123 Samples Used to FIT the Platt Calibrators (`val_calib` / $D_{\text{prob}}$)
- **Selection Mechanism**: Derived via `modules/calibration.py:partition_outer_validation()` with fixed parameters `random_seed=42` and `calib_fraction=0.5`.
- **Sample Count**: Exactly **123 samples**.
- **Class Breakdown**:
  - Psoriasis: 68
  - Lichen Planus: 27
  - Pityriasis Rosea: 18
  - Seborrheic Dermatitis: 10
- **Sample IDs**:
  - First 5: `PSD_00000958`, `PSD_00001567`, `PSD_00001693`, `PSD_00001537`, `PSD_00001702`
  - Last 5: `PSD_00001468`, `PSD_00000838`, `PSD_00003273`, `PSD_00000837`, `PSD_00001338`
- **Usage**:
  - In `run_aef_crc_phase8.py:146-150`, these 123 samples were used to fit the per-class `LogisticRegression` models stored in `h8.platt_models`.

### B. The 123 Samples Held Out for Evaluation / Conformal (`val_conf` / $D_{\text{conf}}$)
- **Selection Mechanism**: Complementary stratified partition generated simultaneously by `partition_outer_validation()`.
- **Sample Count**: Exactly **123 samples**.
- **Class Breakdown**:
  - Psoriasis: 69
  - Lichen Planus: 28
  - Pityriasis Rosea: 17
  - Seborrheic Dermatitis: 9
- **Sample IDs**:
  - First 5: `PSD_00001445`, `PSD_00001523`, `PSD_00000982`, `PSD_00003188`, `PSD_00000844`
  - Last 5: `PSD_00001767`, `PSD_00000824`, `PSD_00003147`, `PSD_00003229`, `PSD_00001649`
- **Verification**:
  - `set(val_calib_ids).isdisjoint(set(val_conf_ids))` is **True** ($0$ overlap).
  - Matches `h8.val_conf_psd_ids` stored in `artifacts/phase8/conformal_handoff.joblib` 100% bit-for-bit.
- **Usage**:
  - Transformed by the fitted Platt models to generate `val_conf_calibrated_probabilities` (stored in `h8` and `h9`), which Phase 9 consumed to calculate conformal quantiles ($\hat{q} = 0.7822$).

---

## 3. Metric Reconciliation Across Partitions

The metrics were computed across three evaluation scopes to reconcile all reported audit figures:

| Evaluation Scope | Subset Description | N | Raw RF Macro-F1 | Platt Macro-F1 | Flips (Non-Pso $\to$ Pso) | Reverse Flips | Total Pso Preds | Pso Pred % |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Fit Set** | `val_calib` ($D_{\text{prob}}$) | 123 | **0.7197** | **0.2701** | 53 | 0 | 119 / 123 | **96.75%** |
| **2. Held-Out Eval Set** | `val_conf` ($D_{\text{conf}}$) | 123 | **0.7387** | **0.3085** | 44 | 0 | 119 / 123 | **96.75%** |
| **3. Combined Outer Val** | Full Calibration Handoff | 246 | **0.7299** | **0.2892** | **97** | **0** | **238 / 246** | **96.75%** |

### Origin of Specific Reported Numbers:
- **`0.7197` (Raw) & `0.2701` (Platt)**: These are the exact values recorded in `reports/phase8/calibration_comparison.csv` under `macro_f1_diagnostic`. They were calculated on **Scope 1: Fit Set (`val_calib`, $N=123$)**.
- **`0.7299` (Raw) & `0.2892` (Platt)**: These were calculated during the forensic bias audit on **Scope 3: Combined Outer Validation ($N=246$)** using `artifacts/phase7/calibration_handoff.joblib` and `artifacts/phase9/final_pipeline_handoff.joblib`.
- **`97 / 246` Flips**: Exactly $53$ flips from `val_calib` plus $44$ flips from `val_conf` = **97 flips across all 246 outer validation samples**.
- **`238 / 246` Psoriasis Predictions**: Exactly $119$ in `val_calib` plus $119$ in `val_conf` = **238 Psoriasis predictions out of 246 (96.75%)**.

---

## 4. Locked Test Isolation Audit

To guarantee absolute scientific integrity, the outer test set was cross-referenced against all calibration partitions:

| Partition | Total Samples | Overlap with Locked Test Set (N=243) | Isolation Status |
|:---|:---:|:---:|:---:|
| **Development Cohort (Folds 0–4)** | 1,146 | **0** | **100% ISOLATED** |
| **Fit Set (`val_calib`)** | 123 | **0** | **100% ISOLATED** |
| **Held-Out Eval Set (`val_conf`)** | 123 | **0** | **100% ISOLATED** |
| **Combined Outer Val Set** | 246 | **0** | **100% ISOLATED** |
| **Quarantined Augmented Set** | 264 | **0** | **100% ISOLATED** |
| **Locked Outer Test Set** | 243 | 243 (self) | **PRISTINE & UNTOUCHED** |

---

## 5. Exact Data Provenance Reference Table

| Metric / Finding | Samples Used | N | Fit or Held-Out? | Source Artifact / Path |
|:---|:---|:---:|:---:|:---|
| **Calibrator Fitting ($w_k, b_k$)** | `val_calib` stratified partition | 123 | **Fit set** | `artifacts/phase8/conformal_handoff.joblib` (`platt_models`) |
| **Conformal Quantile Fitting ($\hat{q}=0.7822$)** | `val_conf` stratified partition | 123 | **Held-out relative to calibrator fit** | `artifacts/phase9/final_pipeline_handoff.joblib` (`marginal_q_hat`) |
| **Raw RF Macro-F1 = 0.7197** | `val_calib` ($D_{\text{prob}}$) | 123 | **Fit set** | `reports/phase8/calibration_comparison.csv` (line 1: `uncalibrated`) |
| **Platt Macro-F1 = 0.2701** | `val_calib` ($D_{\text{prob}}$) | 123 | **Fit set** | `reports/phase8/calibration_comparison.csv` (line 2: `platt`) |
| **Isotonic Macro-F1 = 0.7876** | `val_calib` ($D_{\text{prob}}$) | 123 | **Fit set** | `reports/phase8/calibration_comparison.csv` (line 3: `isotonic`) |
| **Held-Out Raw RF Macro-F1 = 0.7387** | `val_conf` ($D_{\text{conf}}$) | 123 | **Held-out evaluation set** | Evaluated on `val_conf_psd_ids` from `h8` / `h7` |
| **Held-Out Platt Macro-F1 = 0.3085** | `val_conf` ($D_{\text{conf}}$) | 123 | **Held-out evaluation set** | Evaluated on `val_conf_calibrated_probabilities` from `h8` |
| **Audit Raw RF Macro-F1 = 0.7299** | Combined outer validation | 246 | **Combined (123 fit + 123 held-out)** | `artifacts/phase7/calibration_handoff.joblib` (`calibration_raw_probabilities`) |
| **Audit Platt Macro-F1 = 0.2892** | Combined outer validation | 246 | **Combined (123 fit + 123 held-out)** | `reports/final_bias_audit/platt_argmax_comparison.csv` |
| **Total Flips = 97 / 246 (39.43%)** | Combined outer validation | 246 | **Combined (53 fit + 44 held-out)** | `reports/final_bias_audit/platt_argmax_comparison.csv` (`flipped_to_psoriasis == True`) |
| **Reverse Flips = 0 / 246 (0.00%)** | Combined outer validation | 246 | **Combined (0 fit + 0 held-out)** | `reports/final_bias_audit/platt_argmax_comparison.csv` |
| **Psoriasis Preds = 238 / 246 (96.75%)** | Combined outer validation | 246 | **Combined (119 fit + 119 held-out)** | `reports/final_bias_audit/platt_argmax_comparison.csv` (`calibrated_argmax == 'Psoriasis'`) |
| **Locked Test Protection (0 Overlap)** | Locked test cohort | 243 | **Strictly locked final test set** | `output/06_final_split/test/` & `reports/aef_crc/fold_plan.csv` |

---
*Phase 8 Data Provenance Verification Complete. Zero code or artifacts modified. Zero models retrained.*
