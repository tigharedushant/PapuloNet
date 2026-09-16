# AEF-CRC Phase 5 — Deep + Handcrafted Feature Fusion Framework Report

## Executive Summary

Phase 5 of the AEF-CRC framework in the authoritative project (`C:\Users\ASUS\Downloads\PSD_HP_AEF_CRC_Phase12\PSD_HP`) has been built, hardened, and verified under an IEEE-reviewer-oriented controlled fusion design.

**Core Scientific Invariant & Real Experiment Guard**:
- **FRAMEWORK ONLY — NO REAL EXPERIMENTS**: Exactly **0** deep models were trained, **0** real deep features extracted, **0** experimental Macro-F1 metrics generated, and **0** fusion winners selected.
- **Frozen PSD-HP Foundation**: 100% untouched (`DatasetFreezer.verify()`: `Matches = True`). Authoritative 5-fold CV plan (`reports/aef_crc/fold_plan.csv`, SHA-256 `2e804df69358495ccb63359dcf85ddf9634c6d0d8df61fe6b4cf7d5e962e03d7`) is completely preserved.
- **Controlled Feature-Level Concatenation**: Features from the EfficientNet-B0 backbone (1280-D) and verified Phase 4 handcrafted descriptors (68-D) are combined through controlled concatenation arms across an explicit hypothesis hierarchy.

---

## 1. Controlled Fusion Arms (A0–A7)

| Arm ID | Arm Name | Representation ID | Branches | Dimension | Scientific Purpose / Hypothesis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A0** | `EfficientNet` | `efficientnet` | `("deep",)` | **1280-D** | **Deep Backbone Baseline**: isolates CNN representation without any handcrafted descriptors. |
| **A1** | `EfficientNet+GLCM` | `efficientnet_glcm` | `("deep", "glcm")` | **1292-D** | **Macro-Texture Ablation**: tests whether second-order gray-level co-occurrence complements CNN representations. |
| **A2** | `EfficientNet+LBP` | `efficientnet_lbp` | `("deep", "lbp")` | **1298-D** | **Micro-Texture Ablation**: tests whether local rotation-invariant binary patterns complement CNN representations. |
| **A3** | `EfficientNet+HOG-PCA` | `efficientnet_hog` | `("deep", "hog")` | **1312-D** | **Gradient Structure Ablation**: tests whether edge orientation distributions complement CNN representations. |
| **A4** | `EfficientNet+LAB` | `efficientnet_lab` | `("deep", "color_lab")` | **1286-D** | **Explicit Color Distribution Ablation**: tests whether channel-level color statistics ($L, a, b$ mean and std) complement CNN representations. |
| **A5** | `EfficientNet+GLCM+LBP` | `efficientnet_glcm_lbp` | `("deep", "glcm", "lbp")` | **1310-D** | **Dual-Texture Combination**: tests joint complementary effect of macro- and micro-texture. |
| **A7** | `EfficientNet+GLCM+LBP+LAB` | `efficientnet_glcm_lbp_lab` | `("deep", "glcm", "lbp", "color_lab")` | **1316-D** | **Compact Interpretable Fusion**: tests compact texture and color without HOG. |
| **A6** | `EfficientNet+GLCM+LBP+HOG+LAB` | `efficientnet_glcm_lbp_hog_lab` | `("deep", "glcm", "lbp", "hog", "color_lab")` | **1348-D** | **Full Handcrafted Fusion**: combined representation incorporating all deep and handcrafted families. |

---

## 2. Feature Ordering & Exact Slice Contracts

Every fusion arm strictly adheres to the canonical branch ordering:
$$\text{deep} \longrightarrow \text{glcm} \longrightarrow \text{lbp} \longrightarrow \text{hog} \longrightarrow \text{color\_lab}$$

### Full 1348-D Representation Slice Boundaries:
```
Index Range     Branch       Dimension   Feature Names
─────────────────────────────────────────────────────────────────────────────────────────────
[0:1280]        deep         1280-D      efficientnet_0 ... efficientnet_1279
[1280:1292]     glcm           12-D      glcm_contrast_d1 ... glcm_ASM_d2
[1292:1310]     lbp            18-D      lbp_bin_0 ... lbp_bin_17
[1310:1342]     hog (PCA)      32-D      hog_pca_0 ... hog_pca_31
[1342:1348]     color_lab       6-D      lab_L_mean, lab_L_std, lab_a_mean, lab_a_std, lab_b_mean, lab_b_std
─────────────────────────────────────────────────────────────────────────────────────────────
Total:                       1348-D      (Contiguous, non-overlapping, strictly 1-to-1)
```

- **Feature Naming Contract**: All feature names are deterministically mapped and validated by `get_fused_feature_names()`.
- **Slice Map**: Evaluated dynamically via `get_fused_slice_map()` to support downstream Phase 6 DFA and explainability (SHAP).
- **Dimension Validation**: `validate_fusion_dimensions()` rejects any unexpected vector width at runtime.

---

## 3. Leakage Guarantees & Statistical Rules

1. **Strict Fold-Train-Only Fitting**:
   - `FeatureNormalizer`: Strictly fit on `fold.train_records` vectors; validation slices are transform-only.
   - `FoldSafeFeatureReducer` (HOG PCA): Strictly fit on `fold.train_records` HOG raw descriptors ($1296\text{-D} \rightarrow 32\text{-D}$); validation slices are transform-only.
   - Deep, GLCM, LBP, LAB, and concatenated vectors undergo **no PCA reduction**.
2. **Practical Equivalence Rule**:
   - `EQUIVALENCE_MARGIN = 0.005` in validation Macro-F1 across folds.
   - Small empirical differences ($\le 0.005$) default to preferring the simpler, lower-dimensional representation.
3. **Statistical Significance Testing**:
   - Pairwise McNemar test with Holm step-down correction pooled across validation folds implemented in `run_aef_crc_phase5.py`.

---

## 4. Verification Evidence

### Test Execution Summary:
- **Windows Host (Python 3.12)**:
  - Phase 5 Contract Suite: `pytest tests/test_phase5_fusion_contract.py -v` $\rightarrow$ **13 passed, 0 failed (100%)**.
  - Phase 5 Existing Suite: `pytest tests/test_fusion.py -v` $\rightarrow$ **4 passed, 0 failed (100%)**.
  - Complete Repository Suite: `pytest tests/` $\rightarrow$ **263 passed, 0 failed (100%)**.
- **WSL2 Linux Environment (Ubuntu 24.04, Python 3.12, CUDA/GPU stack)**:
  - Phase 5 Contract Suite: `pytest tests/test_phase5_fusion_contract.py tests/test_fusion.py -v` $\rightarrow$ **17 passed, 0 failed (100%)**.
  - Complete Repository Suite: `pytest tests/ -q` $\rightarrow$ **263 passed, 0 failed (100%)**.
- **Dataset Freeze & Fold Plan Integrity**:
  - `DatasetFreezer.verify()`: `Matches = True`
  - `reports/aef_crc/fold_plan.csv` SHA-256: `2e804df69358495ccb63359dcf85ddf9634c6d0d8df61fe6b4cf7d5e962e03d7` (Identical).
- **Runner Plumbing Pass**:
  - `python run_aef_crc_phase5.py` executes synthetic plumbing pass across all 8 arms and pairwise McNemar-Holm tests, confirming execution readiness without touching real data.

---

## 5. Phase Boundaries & Real Experiment Status

- **Phase 4**: Deterministic handcrafted feature extraction and fold-safe HOG PCA (68-D).
- **Phase 5**: Feature-level fusion and controlled ablation study between deep representations and handcrafted branches (1280-D to 1348-D).
- **Phase 6**: Optimization-based feature selection (DFA) on the chosen representation.
- **Explicit Declaration**:
  > **No real Phase 5 experiment was run. No model was trained. No experimental metric was generated. No fusion winner was selected.**
