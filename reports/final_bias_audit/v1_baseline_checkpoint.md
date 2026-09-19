# V1 BASELINE CHECKPOINT & PROVENANCE INVENTORY
**PapuloNet / AEF-CRC Dermatology Diagnosis Pipeline**

- **Project Root**: `/mnt/c/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP`
- **Checkpoint Date**: 2026-09-19
- **Status**: **PERMANENT READ-ONLY HISTORICAL BASELINE INVENTORY**
- **Working Tree Cleanliness**: **UNCOMMITTED CHANGES PRESENT (TAGGING POSTPONED PER RULE 8)**

---

> [!IMPORTANT]
> **GOVERNANCE MANDATES**:
> 1. **V1 IS THE HISTORICAL BASELINE.**
> 2. **V1 MUST NOT BE OVERWRITTEN BY V2.**
> 3. **V1 RESULTS MUST NOT BE MIXED WITH V2 RESULTS.**
> 4. **ALL V2 ARTIFACTS AND REPORTS WILL LIVE EXCLUSIVELY IN `_v2` DIRECTORIES.**

---

## 1. Git Repository State

- **Current Branch**: `main`
- **Current HEAD Commit SHA**: `32c801a788c5e72c7099060364c8220b12da7220`
- **Commit Message**: `"Complete AEF-CRC phases 5 through 7"`
- **Git Tag Status**: **NOT TAGGED** (Working tree is not clean; tagging withheld per Rule 8).

### Working Tree Status Audit:
The working tree contains uncommitted changes and untracked files resulting from the execution and verification of Phases 8, 9, 10, the application adapter, and the bias audit:

#### A. Modified Tracked Files (8 files):
1. `modules/calibration_handoff.py`: Added handoff fields for Phase 8/9.
2. `modules/inference.py`: Implemented inference orchestration and Platt argmax.
3. `modules/xai_gradcam.py`: Fixed half-precision type matching for RTX 2050 GPU.
4. `modules/xai_shap.py`: Fixed `export_csvs` return signature and feature mapping.
5. `run_aef_crc_phase8.py`: Phase 8 runner with Platt scaling and Isotonic comparator.
6. `run_aef_crc_phase9.py`: Phase 9 runner with Marginal & Mondrian conformal sets.
7. `tests/test_phase9_conformal.py`: Contract tests for conformal calibration.
8. `tests/test_xai_contract.py`: Contract tests for Grad-CAM and TreeSHAP.

#### B. Untracked Files & Directories:
1. `app/` (`app/streamlit_app.py` UI implementation).
2. `modules/app_adapter.py` (Tested inference adapter for Streamlit).
3. `reports/final_bias_audit/` (All bias audit and red-team documents).
4. `reports/phase8/` (`calibration_comparison.csv`, `validation_partition_summary.csv`, etc.).
5. `reports/phase9/` (`conformal_summary.csv`, `mondrian_class_quantiles.csv`).
6. `reports/phase10/` (`phase10_report.md`, `phase10_manifest.json`, `selected_samples.csv`, Grad-CAM & SHAP plots).
7. `run_aef_crc_phase10.py` (Phase 10 XAI execution runner).
8. `tests/test_app_adapter.py`, `tests/test_phase10_runner.py`, `tests/test_streamlit_app.py`.

---

## 2. Frozen Dataset & Partition Provenance (Phase 2)

| Asset | Path | SHA-256 Digest | Status |
|:---|:---|:---|:---:|
| **Dataset Freeze Manifest** | `reports/aef_crc/dataset_freeze.json` | `964d91f8178d5bb7123a7cc02388cd57912bba1e836b2a3ab70b9fa98441919f` | **FROZEN** |
| **Stratified Fold Plan** | `reports/aef_crc/fold_plan.csv` | `3c54f09f834632f8f771b46d5a98fb773aa6ee78878b00a0c58806c575b8f7b9` | **FROZEN** |
| **Dataset Metadata** | `reports/metadata.csv` | `b443fa6c1f8ac3d02347de05a0e2a1274995ac61db5146bd8521923facca81e5` | **FROZEN** |
| **Locked Outer Test Set** | `output/06_final_split/test/` ($N=243$) | *Quarantined directory* | **UNTOUCHED** |
| **Quarantined Augmented** | `output/` ($N=264$, `fold_id: -1`) | *Quarantined records* | **EXCLUDED** |

---

## 3. V1 Phase-by-Phase Provenance & Checkpoint Inventory

### Phase 3: Deep Feature Baseline
- **Winner / Architecture**: `P3-BASE` (EfficientNet-B0 backbone, ImageNet pretraining, two-stage transfer learning).
- **Representation ID**: `efficientnet_b0_6760c4f151acc2d2`
- **Output Dimension**: 1280-D (Global Average Pooling)
- **Winning Report**: `reports/phase3/winner.json`
- **Feature Cache**: `artifacts/phase3/deep_features/P3-BASE/`

### Phase 4: Handcrafted Feature Extraction
- **Extracted Dimensions**: GLCM: 12-D, LBP: 18-D, HOG-PCA: 32-D, Color LAB: 6-D (Combined Active: 68-D).
- **PCA Discipline**: Fold-local PCA fitted on training partitions only.
- **Reports**: `reports/phase4/fold_summary.csv`, `reports/phase4/handcrafted_only_results.csv`.

### Phase 5: Multimodal Fusion
- **Selected Arm**: **`A7`** (`EfficientNet + GLCM + LBP + LAB`)
- **Total Fusion Dimension**: **1316-D**
- **Performance**: Mean 5-Fold Macro-F1 = `0.7270 ± 0.0190`, Balanced Accuracy = `0.7264`.
- **Reports**: `reports/phase5/fusion_results.csv`, `reports/phase5/phase5_manifest.json`.

### Phase 6: Binary Dragonfly Feature Selection (BDA)
- **Selected Feature Count**: **194 / 1316 features**
- **Branch Breakdown**: Deep: 190, GLCM: 1, LBP: 3, Color LAB: 0.
- **Production Mask Artifact**: `artifacts/phase6/production_bda_mask.joblib`
  - SHA-256: `aa8ce69fa535bcba702bae311dec0c7f1af238e5e92f4dcf47cc8aceb038b94f`
- **Metadata**: `reports/phase6/production_bda_mask_metadata.json`
- **V1 Known Defect**: Unweighted proxy Logistic Regression (`class_weight=None`), pairwise Jaccard stability = `0.0761`.

### Phase 7: Final Retrained Random Forest & Handoff
- **Training Population**: Reunified 1,146 development images (Folds 0–4).
- **Evaluation Population**: 246 outer validation images (Holdout calibration set).
- **Primary Artifact**: `artifacts/phase7/calibration_handoff.joblib`
  - SHA-256: `c0048de44b1528c7abd46bdaa37f6eb5d5ec068a38bb0a18cf1d35c90a8a0e95`
- **Metrics on Outer Validation (Raw RF)**: Macro-F1 = `0.7299`, Balanced Accuracy = `0.7468`, Accuracy = `0.7561`, MCC = `0.6023`.
- **Reports**: `reports/phase7/final_cv_summary.csv`, `reports/phase7/per_class_metrics.csv`, `reports/phase7/calibration_set_metrics.csv`.

### Phase 8: Probability Calibration
- **Calibration Split**: 123 fit (`val_calib`) / 123 eval (`val_conf`).
- **Primary Method**: Platt Scaling (4 independent binary sigmoids).
- **Primary Artifact**: `artifacts/phase8/conformal_handoff.joblib`
  - SHA-256: `412d4a0d01e340f1e8b1ec8ba38030997ca388a7c86541a924bb173d4999d664`
- **V1 Known Defect**: In `modules/inference.py`, argmax over Platt probabilities causes 97 flips to Psoriasis and 96.75% Psoriasis collapse (Macro-F1 drops to `0.2892`).
- **Reports**: `reports/phase8/calibration_comparison.csv`, `reports/phase8/validation_partition_summary.csv`.

### Phase 9: Conformal Prediction
- **Handoff Artifact**: `artifacts/phase9/final_pipeline_handoff.joblib`
  - SHA-256: `ee1097456f963afd009e552dfaa0cd4f031fd7f8b602777e1bbc5ffa55cacc91`
- **Marginal Threshold**: $\hat{q} = 0.7822$ at $\alpha = 0.10$ (Empirical coverage = `91.1%`, Mean set size = `1.55`).
- **Reports**: `reports/phase9/conformal_summary.csv`, `reports/phase9/mondrian_class_quantiles.csv`.

### Phase 10: Explainable AI (XAI)
- **Report**: `reports/phase10/phase10_report.md`
- **Manifest**: `reports/phase10/phase10_manifest.json`
  - SHA-256: `4eec87e9d1c52ad607a427fbcfd91d72c79d8c407f550a7c6b2ea537f4a55718`
- **Sample Visualizations**: `reports/phase10/gradcam/` and `reports/phase10/shap/` for 8 validation samples.

---

## 4. V1 Final Metric Summary (Historical Baseline)

| Pipeline Component | Scope | Macro-F1 | Balanced Acc | Accuracy | MCC | Psoriasis Predicted % |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Phase 3: P3-BASE** | 5-Fold CV ($N=1146$) | 0.7099 | 0.7637 | 0.7330 | 0.5987 | 46.3% |
| **Phase 4: Combined** | 5-Fold CV ($N=1146$) | 0.5275 | 0.6014 | 0.5986 | 0.3652 | 53.8% |
| **Phase 5: A7 Fusion** | 5-Fold CV ($N=1146$) | 0.7270 | 0.7264 | 0.7513 | 0.6179 | 58.2% |
| **Phase 6: BDA Mask** | 5-Fold CV ($N=1146$) | 0.6994 | 0.6980 | 0.7350 | 0.5720 | 59.1% |
| **Phase 7: Final RF** | 5-Fold CV ($N=1146$) | 0.6906 | 0.6923 | 0.7303 | 0.5547 | 59.5% |
| **Phase 7: Raw RF** | Outer Val ($N=246$) | **0.7299** | **0.7468** | **0.7561** | **0.6023** | **57.3%** |
| **Phase 8: Platt Argmax** | Outer Val ($N=246$) | **0.2892** | **0.3094** | **0.5854** | **0.2135** | **96.75%** |

---

## 5. Certification of Checkpoint Completion

We formally certify:
- The complete V1 baseline state is inventoried and preserved.
- No V1 code or artifact has been modified or overwritten.
- The locked 243-image test partition has not been opened.
- No model has been retrained.
- Tagging is held until the user explicitly directs how to handle the uncommitted Phase 8–10 files.

*V1 Baseline Inventory Complete.*
