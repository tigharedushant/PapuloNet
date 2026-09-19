# AEF-CRC / PAPULONET — FINAL BRUTAL FORENSIC AUDIT REPORT
**Authoritative Diagnostic & Root Cause Investigation**  
**Date**: September 19, 2026  
**Target Repository**: `C:\Users\ASUS\Downloads\PSD_HP_AEF_CRC_PopuloNet\PSD_HP`  
**Status**: COMPLETE FORENSIC AUDIT (CODE & ARTIFACTS FROZEN; NO RETRAINING PERFORMED)

---

## 1. Executive Finding

The observed phenomenon — where external and internal dermatological images from non-Psoriasis classes repeatedly collapse to a **Psoriasis** prediction with calibrated probabilities in the **40%–57%** range — is **not** a mystery, nor is it mere "overfitting."

It is the direct, compounded result of **two catastrophic systemic mechanisms operating in series**, preceded by training imbalance and feature-selection damage:

1. **The Immediate Mechanical "Smoking Gun" — Platt Scaling Argmax Inversion (`modules/inference.py`, Line 445)**:
   - In `modules/inference.py`, the predicted class is computed as:
     $$\text{pred\_idx} = \operatorname{argmax}(\text{calib\_probs}[0])$$
   - Phase 8 fitted four **independent 1D Logistic Regression models** on the 246-image holdout calibration split, where Psoriasis has a **55.7% base rate** while Seborrheic Dermatitis has a **7.7% base rate**.
   - The learned logistic intercepts are severely skewed by class prevalence:
     $$\text{Psoriasis: } -1.2045 \quad\text{vs}\quad \text{Lichen Planus: } -1.9706 \quad\text{vs}\quad \text{Pityriasis Rosea: } -2.4044 \quad\text{vs}\quad \text{Seborrheic Dermatitis: } -2.9611$$
   - When an external, ambiguous, or out-of-distribution image arrives, the underlying Random Forest produces diffuse, uncertain raw probabilities (e.g., $[0.25, 0.25, 0.25, 0.25]$ or $[0.20, 0.35, 0.25, 0.20]$ where Lichen Planus is top-1).
   - Platt calibration multiplies these raw probabilities by the uncoupled sigmoid transforms and normalizes by their sum, automatically mapping diffuse outputs into **$41\% - 48\%$ for Psoriasis**, pushing Psoriasis to the top of the vector.
   - **Empirical Proof on the Calibration Set**: When evaluated on the 246 holdout calibration images, Platt scaling **flips 97 out of 246 predictions (39.43% of the entire dataset) from non-Psoriasis to Psoriasis**, and flips **0** in the opposite direction. Consequently, **238 out of 246 calibration samples (96.75%) collapse to a Psoriasis point prediction** under calibrated argmax!

2. **The Secondary Amplifier — Feature Selection (BDA) Damage (`modules/feature_selection.py`, Lines 340–345)**:
   - The Binary Dragonfly Algorithm (BDA) objective optimized a proxy `LogisticRegression(max_iter=300)` that was **completely unweighted** (`class_weight=None`, no sample weights), running on an imbalanced dataset (55.7% Psoriasis).
   - To maximize linear separation of the majority class, BDA selected **190 deep features, 1 GLCM feature, 3 LBP features, and ZERO LAB color features**.
   - Pruning out 1,090 deep features and 100% of color features destroyed the subtle color/texture discriminators separating Lichen Planus (violaceous hue) and Pityriasis Rosea (collarette scale) from Psoriasis (silvery scale on erythematous plaque).
   - Across the 5 development folds, BDA **degraded Macro-F1 by $-0.0527$** ($0.7264 \rightarrow 0.6737$) and **increased Psoriasis false positives by $+26.0\%$ (from 127 to 160)**.

3. **The Upstream Origin — Severe Training Class Imbalance (7.25:1) in Phase 3**:
   - In the authoritative 1,146-image development cohort, Psoriasis comprises **638 images (55.67%)**, Lichen Planus **258 (22.51%)**, Pityriasis Rosea **162 (14.14%)**, and Seborrheic Dermatitis only **88 (7.68%)**.
   - Standard categorical cross-entropy fine-tuning caused EfficientNet-B0 to allocate the vast majority of its feature variance to Psoriasis manifold geometry.
   - Handcrafted features contribute $<4.3\%$ of downstream RF split importance, leaving the system almost entirely dependent on the skewed deep backbone.

---

## 2. Actual V1 Executable Pipeline Architecture

The executable pipeline was traced from user upload in `app/streamlit_app.py` through `modules/app_adapter.py`, `modules/inference.py`, and `artifacts/phase9/final_pipeline_handoff.joblib`.

### End-to-End Pipeline Execution Table

| Stage | Implementation File & Method | Input Dimension | Output Dimension | Preprocessing / Normalization | Parameters & Artifact Provenance | Training / Inference Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Input Validation** | `modules/inference.py`<br>`AEFCRCInferenceEngine.validate_input_image` | Image input (Path, PIL, bytes) | PIL Image (RGB) | Dimension check ($\ge 32\times 32$), format validation, RGB conversion | Strict validation; rejects corrupt/empty inputs | **MATCH** |
| **2. Image Preprocessing** | `modules/inference.py`<br>`AEFCRCInferenceEngine.preprocess_image` | PIL Image (RGB) | 1. Tensor: `(1, 224, 224, 3)` float32<br>2. RGB: `(224, 224, 3)` uint8 | `ConditionalPreprocessor` in `standard` mode (pass-through) + Keras `preprocess_input` (zero-centered $[-1, 1]$) | `image_size=224`, `preprocessing_mode="standard"` | **MATCH** (P3-BASE winner used standard mode) |
| **3a. Deep Feature Extraction** | `modules/inference.py`<br>`AEFCRCInferenceEngine.extract_multimodal_features` | Tensor `(1, 224, 224, 3)` | 1280-D vector (float32) | Global Average Pooling output from penultimate layer of frozen backbone | `artifacts/phase3/P3-BASE/fold_00/best_model.keras`<br>(`representation_id: efficientnet_b0_6760c4f151acc2d2`) | **MATCH** |
| **3b. Handcrafted Features** | `modules/inference.py`<br>`extract_multimodal_features` using `HandcraftedFeatureExtractor` | RGB `(224, 224, 3)` uint8 | - GLCM: 12-D<br>- LBP: 18-D<br>- LAB: 6-D | GLCM (distances 1,2, angles 0,45,90,135); LBP (uniform $P=8, R=1$); LAB (mean/std per channel) | Phase 4 Handcrafted Extractor (`config.handcrafted`) | **MATCH** |
| **3c. Feature Normalization & Fusion** | `modules/inference.py`<br>`extract_multimodal_features` | Raw 1316-D vector | Normalized 1316-D fused vector | Per-branch `FeatureNormalizer` fitted on training cohort (StandardScaler) | Canonical order: `deep` (1280), `glcm` (12), `lbp` (18), `color_lab` (6) | **MATCH** |
| **3d. Feature Selection Masking** | `modules/inference.py`<br>`AEFCRCInferenceEngine.apply_feature_mask` | 1316-D vector | **194-D vector** | Boolean indexing via production mask | `artifacts/phase6/production_bda_mask.joblib`<br>(Active: deep=190, glcm=1, lbp=3, lab=0) | **MATCH** |
| **4. Base Classifier** | `modules/inference.py`<br>`AEFCRCInferenceEngine.predict_raw_probabilities` | 194-D vector | 4-D raw probabilities (float64) | Random Forest ensemble voting (`predict_proba`) | `RandomForestClassifier(n_estimators=300, criterion='gini', max_features='sqrt', bootstrap=True, class_weight=None, random_state=42)` | **MATCH** |
| **5. Probability Calibration** | `modules/inference.py`<br>`AEFCRCInferenceEngine.apply_calibration` | 4-D raw probabilities | 4-D calibrated probabilities | One-vs-Rest Platt scaling: $\sigma(w_c p_c + b_c)$, normalized by sum | Four 1D `LogisticRegression` models fitted on 246 calibration samples (`artifacts/phase9/final_pipeline_handoff.joblib`) | **MATCH** |
| **6. Point Prediction** | `modules/inference.py`<br>`predict` Line 445 | 4-D calibrated probabilities | `pred_class` (str), `calib_conf` (float) | **ARGMAX OF CALIBRATED PROBABILITIES**:<br>`pred_idx = np.argmax(calib_probs[0])` | Class order: `['Psoriasis', 'Lichen_Planus', 'Pityriasis_Rosea', 'Seborrheic_Dermatitis']` | **CRITICAL ANOMALY: Argmax inverted from raw RF** |
| **7. Conformal Prediction** | `modules/inference.py`<br>`AEFCRCInferenceEngine.predict_conformal_sets` | 4-D calibrated probabilities | Marginal prediction set (List[str]), Mondrian set (List[str]) | Nonconformity score $s_i = 1 - \hat{P}(y_i \mid x_i)$. Inclusion threshold: $p_c \ge 1 - \hat{q}$ | Marginal $\hat{q} = 0.782166$ (Threshold = 0.217834); Nominal coverage = 90% | **MATCH** |
| **8. Clinical Review Status** | `modules/output_schema.py`<br>`determine_conformal_review_status` | Marginal set size | Status enum & explanation string | Singleton $\rightarrow$ `STANDARD_OUTPUT`; Multi-class $\rightarrow$ `SPECIALIST_REVIEW_REQUIRED`; Empty $\rightarrow$ `SPECIALIST_REVIEW_REQUIRED` | Clinical safety gating contract | **MATCH** |
| **9. Explainable AI (XAI)** | `modules/app_adapter.py`<br>`run_app_inference` | Original image + 194-D masked features | Grad-CAM overlay PIL, TreeSHAP DataFrame | Grad-CAM on EfficientNet `top_conv` pre-softmax logit; TreeSHAP on RF 194-D features | ProductionTreeSHAPExplainer + GradCAMExplainer | **MATCH** |

---

## 3. Class Distribution — Verified From Authoritative Files

Class counts and split roles were computed directly from `reports/aef_crc/fold_plan.csv` and `reports/metadata.csv` (total records in fold plan: 6,483; unique images: 1,909 non-augmented + 264 excluded augmented).

### Authoritative Dataset Partitioning Matrix

| Split Role | Psoriasis | Lichen Planus | Pityriasis Rosea | Seborrheic Dermatitis | Total Images | % Psoriasis | Class Ratio (Pso : SebDerm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development Cohort (Outer Train, 5 Folds)** | **638** | **258** | **162** | **88** | **1,146** | **55.67%** | **7.25 : 1** |
| - Fold 0 (Train / Val) | 510 / 128 | 206 / 52 | 130 / 32 | 70 / 18 | 916 / 230 | 55.68% / 55.65% | 7.29 : 1 |
| - Fold 1 (Train / Val) | 510 / 128 | 207 / 51 | 129 / 33 | 71 / 17 | 917 / 229 | 55.62% / 55.90% | 7.18 : 1 |
| - Fold 2 (Train / Val) | 510 / 128 | 207 / 51 | 129 / 33 | 71 / 17 | 917 / 229 | 55.62% / 55.90% | 7.18 : 1 |
| - Fold 3 (Train / Val) | 511 / 127 | 206 / 52 | 130 / 32 | 70 / 18 | 917 / 229 | 55.73% / 55.46% | 7.30 : 1 |
| - Fold 4 (Train / Val) | 511 / 127 | 206 / 52 | 130 / 32 | 70 / 18 | 917 / 229 | 55.73% / 55.46% | 7.30 : 1 |
| **Held-out Val (Phase 8 Calibration Split)** | **137** | **55** | **35** | **19** | **246** | **55.69%** | **7.21 : 1** |
| **Locked Outer Test Partition (UNTOUCHED)** | **136** | **55** | **34** | **18** | **243** | **55.97%** | **7.56 : 1** |
| **Excluded Augmented Records (Quarantined)** | 0 | 0 | 0 | 264 | 264 | 0.00% | 0 : 264 |
| **Total Permitted Non-Augmented Images** | **911** | **368** | **231** | **125** | **1,635** | **55.72%** | **7.29 : 1** |

### Imbalance Evaluation
- **Severity**: Extreme. Psoriasis constitutes **$55.7\%$ of every single split**, while Seborrheic Dermatitis constitutes **only $7.4\% - 7.7\%$**.
- **Imbalance Ratio**: Psoriasis outnumbers Seborrheic Dermatitis by **$7.25 : 1$**, Pityriasis Rosea by **$3.94 : 1$**, and Lichen Planus by **$2.47 : 1$**.
- **Plausibility as a Bias Driver**: In any machine learning pipeline without explicit class-prior adjustment, an unconstrained model will optimize global accuracy by shifting its decision boundaries toward the 55.7% majority class.

---

## 4. Critical Investigation: Did Phase 3 Actually Use Class Weighting?

### Findings from Executable Code & Manifests

| Question | Forensic Verdict | Exact Evidence in Code / Artifacts |
| :--- | :--- | :--- |
| **A. Was categorical cross-entropy used?** | **YES** | `modules/efficientnet_model.py`, Line 63 & Line 93: `loss="categorical_crossentropy"`. |
| **B. Was focal loss used?** | **NEVER IMPLEMENTED** | Grep across entire codebase finds zero occurrences of focal loss implementation. It was mentioned only as a future comment in `README.md` (Line 223). |
| **C. Were class weights calculated?** | **YES** | `modules/aef_input_validator.py`, Line 356: `compute_train_fold_class_weights` calculated $w_c = N / (C \cdot N_c)$.<br>In `fold_00/manifest.json`: Pso: 0.4490, LP: 1.1117, PR: 1.7615, SD: 3.2714. |
| **D. Were weights passed to `model.fit()`?** | **INDIRECTLY VIA SAMPLE_WEIGHT** | `modules/training.py`, Lines 16–21 & 327–331: `class_weight` parameter was removed from `model.fit()` because Keras 3 does not support `class_weight` when `x` is a `tf.data.Dataset`. Weights were delivered via `sample_weight` inside `train_ds`. |
| **E. Did `build_dataset()` compute sample weights?** | **YES** | `modules/image_loader.py`, Line 143: `sample_weights = [float(class_weights[r.mapped_class]) for r in records]`; Line 194: yields `(image, one_hot_label, sample_weight)`. |
| **F. Were weights active in Stage 1 and Stage 2?** | **YES** | `train_ds` was passed to `model.fit()` in both Stage 1 (Line 406) and Stage 2 (Line 414). |
| **G. Did P3-BASE, P3-PRE, P3-AUG use same strategy?** | **YES** | All three Phase 3 experiments shared `modules/training.py:run_experiment()` and identical dataset construction. |

### Technical Distinction: Sample-Weighted Cross-Entropy vs. Focal Loss
While sample weighting was mathematically applied to the loss batches, **sample-weighted cross-entropy is fundamentally distinct from focal loss**:
- Sample weighting scales the gradient magnitude uniformly across all examples of a class by $w_c$.
- It does **not** down-weight well-classified, easy examples ($p_t \to 1$).
- Because Psoriasis images are numerous (510 per training fold) and share visually salient erythematous plaque textures, the cumulative gradient from easy Psoriasis examples still completely overwhelms the minority gradients, causing the convolutional kernels to specialize on Psoriasis patterns.

---

## 5. Phase 3 Confusion Analysis: Standalone EfficientNet Backbone

We inspected the actual Phase 3 confusion matrices across all 5 development folds (`artifacts/phase3/P3-BASE/fold_0{0..4}/confusion_matrix.json`).

### Pooled 5-Fold Development Confusion Matrix (P3-BASE Standalone)

```
True \ Pred             Psoriasis  Lichen_Planus  Pityriasis_Rosea  Seborrheic_Dermatitis    Support
Psoriasis                     457             83                58                     40        638
Lichen_Planus                  49            176                25                      8        258
Pityriasis_Rosea               18              8               134                      2        162
Seborrheic_Dermatitis           7              6                 2                     73         88
Total Predicted               531            273               219                    123       1146
```

### Standalone Phase 3 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support | Predicted Count | Pred / Support Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Psoriasis** | **0.8606** | **0.7163** | **0.7819** | 638 | 531 | **0.832** |
| **Lichen Planus** | 0.6447 | 0.6822 | 0.6629 | 258 | 273 | 1.058 |
| **Pityriasis Rosea** | 0.6119 | 0.8272 | 0.7034 | 162 | 219 | 1.352 |
| **Seborrheic Dermatitis** | 0.5935 | 0.8295 | 0.6919 | 88 | 123 | 1.398 |

### Critical Finding: Phase 3 Did NOT Have a Psoriasis Argmax Bias
- In Phase 3 standalone, **Psoriasis was predicted 531 times (46.34%)**, which is **LESS than its true prevalence of 55.67%**.
- Total Psoriasis False Positives in Phase 3 was only **74**:
  - True Lichen Planus $\rightarrow$ Psoriasis: 49 / 258 (18.99%)
  - True Pityriasis Rosea $\rightarrow$ Psoriasis: 18 / 162 (11.11%)
  - True Seborrheic Dermatitis $\rightarrow$ Psoriasis: 7 / 88 (7.95%)
- In fact, the sample weights in Phase 3 caused the standalone head to **under-predict Psoriasis** (recall: 71.63%) and over-predict minority classes (Pityriasis Rosea recall: 82.72%; Seborrheic Dermatitis recall: 82.95%).
- **Conclusion**: The Psoriasis prediction collapse did **not** originate in the Phase 3 classification head decisions. It entered downstream during multimodal fusion, feature selection, and calibration!

---

## 6. EfficientNet Representation & Phase 10 XAI Inspection

While Phase 3's classification head was constrained by sample weights, what was happening inside the **1280-D penultimate representation vector**?

### Dense Head Parameters (`fold_00/best_model.keras`)
- **Kernel Shape**: $(1280, 4)$ | **Bias Shape**: $(4,)$
- **Learned Biases**:
  $$\text{Psoriasis: } +0.0171 \quad\mid\quad \text{Lichen Planus: } -0.0244 \quad\mid\quad \text{Pityriasis Rosea: } +0.0106 \quad\mid\quad \text{Seborrheic Dermatitis: } -0.0039$$
- **L2 Norm of Incoming Weights**:
  $$\text{Psoriasis: } 2.0547 \quad\mid\quad \text{Lichen Planus: } 1.9884 \quad\mid\quad \text{Pityriasis Rosea: } 2.0651 \quad\mid\quad \text{Seborrheic Dermatitis: } 2.0612$$
- The weights and biases of the Dense head are balanced. However, when individual images pass through the backbone, the **deep feature activations** themselves are skewed.

### Phase 10 Representative Samples Table (`reports/phase10/phase10_manifest.json`)

| Sample ID | True Class | App Predicted Class | Raw RF Conf | Calib Conf | Pre-Softmax Pso Logit | Grad-CAM Target | Top SHAP Branch | Deep Branch % | Marginal Prediction Set |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `PSD_00000917` | Psoriasis | Psoriasis | 32.7% | 48.5% | +18.96 | `top_conv` | `deep` | 95.92% | `{Psoriasis, Pityriasis_Rosea}` |
| `PSD_00001647` | Psoriasis | Psoriasis | 29.0% | 47.4% | +6.93 | `top_conv` | `deep` | 97.60% | `{Psoriasis, Pityriasis_Rosea}` |
| `PSD_00000639` | Lichen Planus | **Psoriasis** | 47.0% | 57.5% | **+20.25** | `top_conv` | `deep` | 97.71% | `{Psoriasis, Pityriasis_Rosea}` |
| `PSD_00003153` | Lichen Planus | **Psoriasis** | **21.3%** | **43.2%** | -1.00 | `top_conv` | `deep` | 98.72% | `{Psoriasis, Lichen_Planus}` |
| `PSD_00000735` | Pityriasis Rosea | **Psoriasis** | 31.0% | 49.5% | **+11.45** | `top_conv` | `deep` | 98.69% | `{Psoriasis, Pityriasis_Rosea}` |
| `PSD_00003209` | Pityriasis Rosea | Lichen Planus | 63.7% | 43.7% | -2.75 | `top_conv` | `deep` | 96.63% | `{Psoriasis, Lichen_Planus}` |
| `PSD_00001693` | SebDerm | **Psoriasis** | **17.0%** | **41.6%** | +2.07 | `top_conv` | `deep` | 95.70% | `{Psoriasis, Lichen_Planus}` |
| `PSD_00001760` | SebDerm | **Psoriasis** | **24.7%** | **47.2%** | +3.73 | `top_conv` | `deep` | 96.47% | `{Psoriasis}` |

### Key Forensic Insights from Phase 10
1. **Backbone Misdirection**: For `PSD_00000639` (Lichen Planus), EfficientNet's pre-softmax logit for Psoriasis was **$+20.25$**, and for `PSD_00000735` (Pityriasis Rosea), it was **$+11.45$**. The convolutional features extracted by the backbone fire vigorously on erythematous lesions regardless of the true pathology.
2. **Total Deep Dominance in TreeSHAP**: Across all 8 samples, **$95.7\% - 98.7\%$ of Random Forest split attribution belongs to the `deep` branch**. Handcrafted GLCM, LBP, and LAB features contribute only $1.3\% - 4.3\%$. The downstream model is functionally a linear projection of the EfficientNet embedding.
3. **Low Raw Confidences Inflated by Calibration**: For `PSD_00001693` (SebDerm), raw RF confidence for Psoriasis was **only 17.0%**, but Platt calibration boosted it to **41.6%**, turning it into the argmax! For `PSD_00003153`, raw RF was **21.3%**, boosted to **43.2%**!

---

## 7. BDA Feature Selection Analysis: Discarding Minority Information

In Phase 6, Binary Dragonfly Algorithm (BDA) was selected as the primary feature selection method over GA and RFE (`reports/phase6/feature_refinement_summary.csv`).

### The BDA Formulation (`modules/feature_selection.py`, Lines 280–350)
- **Fitness Function**:
  $$\text{Fitness}(\mathbf{m}) = \text{Macro-F1}_{\text{nested\_val}}(\mathbf{m}) - 0.0005 \cdot \sum_{i=1}^{D} m_i$$
- **The Proxy Classifier**:
  ```python
  clf = LogisticRegression(max_iter=300, random_state=random_seed)
  clf.fit(X_bda_train[:, mask], y_bda_train)
  y_pred = clf.predict(X_bda_val[:, mask])
  ```
  **CRITICAL DEFECT**: The proxy classifier used to evaluate every dragonfly feature subset was `LogisticRegression` with **`class_weight=None` and NO sample weights**.
  In `y_bda_train`, Psoriasis represents $55.7\%$ of the data. An unweighted Logistic Regression on an imbalanced dataset naturally favors features that cleanly separate the majority class.

### Features Selected in Production BDA Mask (`reports/phase6/production_bda_mask_metadata.json`)
- **Total Selected**: 194 / 1316 features
- **Deep Features**: **190 / 1280** (97.9% of selected features)
- **GLCM Texture**: **1 / 12**
- **LBP Texture**: **3 / 18**
- **LAB Color**: **0 / 6 (100% ELIMINATED)**

### Empirical Comparison: Full A7 (1316-D) vs. BDA (194-D) Across 5 Folds

| Metric | Full A7 (1316-D) $\rightarrow$ RF | BDA (194-D) $\rightarrow$ RF | Absolute Change | Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Macro-F1 (Mean $\pm$ Std)** | **$0.7264 \pm 0.0179$** | **$0.6737 \pm 0.0373$** | **$-0.0527$** | **Severe Degradation** |
| **Balanced Accuracy** | **$0.7266 \pm 0.0176$** | **$0.6655 \pm 0.0347$** | **$-0.0611$** | **Severe Degradation** |
| **Overall Accuracy** | $0.7661 \pm 0.0160$ | $0.7260 \pm 0.0274$ | $-0.0401$ | Degradation |
| **MCC** | $0.6170 \pm 0.0283$ | $0.5443 \pm 0.0485$ | $-0.0727$ | Degradation |
| **Total Psoriasis False Positives** | **127 / 508 (25.0%)** | **160 / 508 (31.5%)** | **+33 (+26.0% relative)** | **Worsened Bias** |
| - True Lichen Planus $\rightarrow$ Pso | 74 / 258 (28.7%) | 88 / 258 (34.1%) | +14 (+18.9% relative) | Worsened |
| - True Pityriasis Rosea $\rightarrow$ Pso | 37 / 162 (22.8%) | 52 / 162 (32.1%) | +15 (+40.5% relative) | Worsened |
| - True SebDerm $\rightarrow$ Pso | 16 / 88 (18.2%) | 20 / 88 (22.7%) | +4 (+25.0% relative) | Worsened |
| **Total Psoriasis Predictions** | 666 / 1146 (58.1%) | 691 / 1146 (60.3%) | +25 | Shifted further into majority |

### BDA Instability
- **Pairwise Jaccard Stability**: $\mathbf{0.0761}$ (`reports/phase6/feature_refinement_summary.csv`).
- A Jaccard index of 0.0761 means that the feature masks selected across different folds shared **less than 8% of features in common**. BDA was chasing stochastic noise on the nested validation folds rather than identifying robust, invariant disease signatures.

---

## 8. Random Forest Classifier Imbalance Handling

We inspected the production classifier in `artifacts/phase9/final_pipeline_handoff.joblib`:
```python
RandomForestClassifier(
    n_estimators=300,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)
```

### How Imbalance Was Handled During Fit
In `run_aef_crc_phase7.py`, Line 196:
```python
sw_train = compute_sample_weights(data.y_train, class_weights)
clf.fit(data.X_train[:, mask], data.y_train, sample_weight=sw_train)
```
- Fold-local sample weights **were** computed and passed to `clf.fit()`.
- However, with `bootstrap=True`, scikit-learn's `RandomForestClassifier` samples observations with replacement according to `sample_weight`.
- Because the trees are grown without depth constraints (`max_depth=None`, `min_samples_leaf=1`) on a feature space containing 190 deep features, individual decision paths easily split on majority-class idiosyncrasies.
- When an external or unfamiliar image is evaluated, tree traversals that fail to match specific minority-class thresholds drop into background leaves where the base rate is dominated by Psoriasis.
- This is why moving from the Phase 3 Dense head to Random Forest on full A7 increased Psoriasis false positives from **74 to 127**.

---

## 9. Calibration Analysis: The Mathematical Smoking Gun

Phase 8 evaluated probability calibration on the 246-image holdout validation split (`reports/phase8/`).
- Primary method: **Platt scaling** (One-vs-Rest 1D Logistic Regression per class).
- Secondary method: Isotonic regression.
- Selected and frozen into production: **Platt scaling**.

### The Platt Scaling Model Parameters (`artifacts/phase9/final_pipeline_handoff.joblib`)
Each class has an independent 1D model: $P(y = c \mid p_c) = \sigma(w_c \cdot p_c + b_c)$:

| Class | Slope ($w_c$) | Intercept ($b_c$) | Base Rate in Calibration Set |
| :--- | :---: | :---: | :---: |
| **Psoriasis** | **+3.5610** | **-1.2045** | **55.69% (137 / 246)** |
| **Lichen Planus** | +2.4140 | -1.9706 | 22.36% (55 / 246) |
| **Pityriasis Rosea** | +2.9705 | -2.4044 | 14.23% (35 / 246) |
| **Seborrheic Dermatitis** | +3.1697 | **-2.9611** | **7.72% (19 / 246)** |

### Mathematical Mechanism of Prior Injection
Notice the huge disparity between the Psoriasis intercept ($-1.2045$) and the Seborrheic Dermatitis intercept ($-2.9611$).
The difference in logits is $\Delta b = -1.2045 - (-2.9611) = +1.7566$.
In terms of odds ratios:
$$e^{1.7566} \approx \mathbf{5.79}$$
Even if the Random Forest outputs equal raw probabilities for Psoriasis and Seborrheic Dermatitis, Platt calibration multiplies the odds of Psoriasis by **nearly $6\times$ relative to Seborrheic Dermatitis**!

### Behavior on Uniform Raw Probability ($p = [0.25, 0.25, 0.25, 0.25]$)
When the model has complete uncertainty:
- $\sigma(3.5610 \cdot 0.25 - 1.2045) = \sigma(-0.3142) = 0.4221$ (Psoriasis)
- $\sigma(2.4140 \cdot 0.25 - 1.9706) = \sigma(-1.3671) = 0.2031$ (Lichen Planus)
- $\sigma(2.9705 \cdot 0.25 - 2.4044) = \sigma(-1.6618) = 0.1595$ (Pityriasis Rosea)
- $\sigma(3.1697 \cdot 0.25 - 2.9611) = \sigma(-2.1687) = 0.1026$ (Seborrheic Dermatitis)

Normalizing by sum ($0.4221 + 0.2031 + 0.1595 + 0.1026 = 0.8873$):
- **Psoriasis**: **47.57%**
- **Lichen Planus**: **22.89%**
- **Pityriasis Rosea**: **17.98%**
- **Seborrheic Dermatitis**: **11.56%**

### Catastrophic Prediction Inversion (Rank Flips)
We experimentally proved that Platt calibration actively inverts raw predictions:
- **Case 1**: Raw RF predicts **Lichen Planus top-1** ($p_{\text{LP}} = 0.35, p_{\text{Pso}} = 0.20, p_{\text{PR}} = 0.25, p_{\text{SD}} = 0.20$):
  - Platt Calibrated Output: **Psoriasis = 43.47%**, Lichen Planus = 28.07%, PR = 18.28%, SD = 10.19%.
  - **Argmax Flips to Psoriasis!**
- **Case 2**: Raw RF predicts **Pityriasis Rosea top-1** ($p_{\text{PR}} = 0.35, p_{\text{Pso}} = 0.22, p_{\text{LP}} = 0.23, p_{\text{SD}} = 0.20$):
  - Platt Calibrated Output: **Psoriasis = 44.82%**, PR = 23.02%, LP = 22.10%, SD = 10.06%.
  - **Argmax Flips to Psoriasis!**

### Empirical Verification on the Entire 246-Image Calibration Split
Running `inspect_calib_flips.py` on the real Phase 7 calibration handoff artifact yielded:
```
Total calibration samples: 246
Total prediction flips between Raw RF and Platt Calibrated: 97 / 246 (39.43%)
Predictions flipped FROM non-Psoriasis TO Psoriasis: 97
  Raw Lichen Planus          -> Platt Psoriasis: 44
  Raw Pityriasis Rosea       -> Platt Psoriasis: 31
  Raw Seborrheic Dermatitis  -> Platt Psoriasis: 22
Predictions flipped FROM Psoriasis TO non-Psoriasis: 0

Raw RF Psoriasis Predictions:        141 / 246 (57.32%)
Platt Calibrated Psoriasis Predictions: 238 / 246 (96.75%)
```
**Out of 246 validation images, 238 (96.75%) are predicted as Psoriasis under calibrated argmax.**
This explains why external images from other disease classes repeatedly show **40%–57% Psoriasis predictions** in the Streamlit application: whenever an image has any ambiguity, Platt scaling forcibly flips the top class to Psoriasis!

---

## 10. Conformal Prediction: Why It Is Not at Fault

In `modules/inference.py`, Split-Conformal prediction runs after calibration:
- Nonconformity score: $s_i = 1 - \hat{P}(y_i \mid x_i)$.
- Marginal threshold: $1 - \hat{q} = 1 - 0.782166 = \mathbf{0.217834}$ (21.78%).
- Any class with calibrated probability $p_c \ge 0.2178$ is included in the conformal set.

### Interaction with the Psoriasis Bias
1. When an ambiguous or external image arrives, Platt calibration outputs e.g. $\text{Pso} \approx 45\%, \text{PR} \approx 23\%, \text{LP} \approx 22\%, \text{SD} \approx 10\%$.
2. Both Psoriasis and PR (or LP) exceed $21.78\%$, so the conformal prediction set is:
   $$\mathcal{C}(x) = \{\text{Psoriasis}, \text{Pityriasis\_Rosea}\}$$
3. Because $|\mathcal{C}(x)| = 2 > 1$, conformal prediction correctly flags:
   `review_status = SPECIALIST_REVIEW_REQUIRED`.
4. **Conclusion**: Conformal prediction is functioning exactly as designed. It is successfully flagging that the prediction is ambiguous and multi-class. The fault lies entirely in `modules/inference.py` Line 445 selecting `pred_class = classes[argmax(calib_probs)]`, which displays "Psoriasis (45.2%)" as the headline point prediction.

---

## 11. Domain Shift, Preprocessing, and External Images

### Preprocessing Consistency
- Inference preprocessing in `modules/inference.py` was compared against Phase 3 training preprocessing in `modules/image_loader.py`.
- **Verdict: PERFECT MATCH**. P3-BASE used `preprocessing_mode="standard"` (a pass-through no-op) followed by OpenCV resize to $224\times 224$ and Keras EfficientNet `preprocess_input`. Inference executes the identical sequence. Preprocessing discrepancies are **excluded** as a cause.

### External Domain Shift Characteristics
Why do external images fail worse than internal validation images?
1. **Source Dataset Confounding**:
   - In our training data, **85.5% of Psoriasis images come from DermNet** (`reports/source_contribution_report.csv`).
   - DermNet images have characteristic JPEG compression artifacts, clinical lighting, framing (often tightly cropped on plaques), and specific skin-tone distributions.
2. **Out-of-Distribution Feature Drift**:
   - When an external image (e.g. from Google or another hospital) is fed into EfficientNet, its embedding does not land inside the tight minority clusters (Lichen Planus or Pityriasis Rosea).
   - Instead, it falls into the broad, diffuse Psoriasis representation manifold (which spans 638 training images).
3. **The Compounding Failure Chain**:
   $$\text{External Image} \xrightarrow{\text{OOD Shift}} \text{Ambiguous Deep Embedding} \xrightarrow{\text{RF Tree Traversal}} \text{Diffuse Raw Probs } [0.20-0.30] \xrightarrow{\text{Platt Intercept Shift}} \text{Psoriasis Argmax } (41\%-56\%)$$

---

## 12. Data Leakage, Source Bias, and Quality Safeguards

- **Locked Test Set Partition**: Verified completely untouched. `output/06_final_split/test/` contains exactly 243 images. No test images were read, indexed, or evaluated.
- **Quarantined Augmentations**: All 264 unverified augmentations of Seborrheic Dermatitis were successfully quarantined in Phase 2 and excluded from CV and final retrain.
- **Deduplication**: 632 exact/near duplicates were pruned during dataset harmonization (`reports/duplicates_report.csv`).
- **Source Confounding Limitation**: `reports/source_contribution_report.csv` shows that class and dataset source are confounded:
  - Psoriasis is 85.5% DermNet.
  - Seborrheic Dermatitis is 78.7% SkinDisNet.
  - While exact near-duplicates were eliminated, cross-source domain shift makes minority classes vulnerable when evaluated against external distributions.

---

## 13. Why Internal Cross-Validation Looked Acceptable

A critical paradox in the user's observation is:
> *"How can internal CV report Macro-F1 $\approx 0.69 - 0.73$ while the deployed app repeatedly fails on external images?"*

There are **three precise mathematical reasons**:

1. **CV Evaluated Raw RF Predictions, NOT Platt Calibrated Predictions**:
   - In `run_aef_crc_phase7.py`, Line 85:
     ```python
     y_pred = clf.predict(data.X_val[:, mask])  # RAW RF PREDICTIONS
     metrics = compute_fold_metrics(data.y_val, list(y_pred), classes, fold.fold_index)
     ```
   - In cross-validation, the model was scored on `clf.predict()` (Raw Random Forest argmax).
   - In Raw Random Forest, the 5-fold CV Psoriasis prediction rate was **58.1%** (A7) and **60.3%** (BDA).
   - **Platt calibration was NEVER evaluated in 5-fold CV**. It was fitted once in Phase 8 and plugged into inference, where it inverted 39.4% of validation predictions into Psoriasis ($96.75\%$ Psoriasis rate).
2. **Within-Domain Test Slices**:
   - The 5 development folds were carved from the same multi-source pool. Images from DermNet Psoriasis appeared in both train and val folds, allowing the backbone to leverage acquisition shortcuts.
3. **Macro-F1 Masks Majority Infiltration**:
   - In Phase 7 CV, Lichen Planus recall was only **51.1%** (48.9% error rate), and Pityriasis Rosea precision was only **65.5%**. The bias was already present internally; it was simply tolerated by the aggregate metric.

---

## 14. Authoritative Root-Cause Tree

Ranked strictly by empirical evidence and confidence:

```
Psoriasis Top-1 Prediction Collapse
 ├── 1. [CONFIDENCE: HIGH] Platt Calibration Argmax Inversion (modules/inference.py L445)
 │    ├── Evidence: Uncoupled 1D logistic regressions with unconstrained intercepts (-1.20 vs -2.96)
 │    ├── Proof: Inverts 97/246 (39.4%) of calibration predictions into Psoriasis (96.75% Psoriasis total)
 │    ├── Effect: Maps uniform/diffuse raw probabilities [0.25, 0.25, 0.25, 0.25] -> 47.6% Psoriasis
 │    └── Trigger: pred_idx = np.argmax(calib_probs[0]) in inference.py
 │
 ├── 2. [CONFIDENCE: HIGH] BDA Feature Selection Damage (modules/feature_selection.py L340)
 │    ├── Evidence: Proxy classifier was unweighted LogisticRegression without sample weights
 │    ├── Proof: Discarded 1,090 deep features and 100% of LAB color features (190 deep, 0 LAB)
 │    ├── Effect: Degraded 5-fold Macro-F1 by -0.0527; surged Psoriasis FPs by +26.0% (127 -> 160)
 │    └── Instability: Pairwise Jaccard stability across folds was only 0.0761 (<8% overlap)
 │
 ├── 3. [CONFIDENCE: HIGH] Severe Dataset Class Imbalance (7.25:1) (reports/aef_crc/fold_plan.csv)
 │    ├── Evidence: 638 Psoriasis (55.67%) vs 88 SebDerm (7.68%) in development cohort
 │    ├── Proof: Class counts verified from fold_plan.csv and metadata.csv
 │    └── Effect: Massive prior dominance across all learning stages (backbone, RF, calibration)
 │
 ├── 4. [CONFIDENCE: HIGH] EfficientNet Backbone Representation Skew (modules/efficientnet_model.py)
 │    ├── Evidence: Pre-softmax logit for Psoriasis was +20.25 on true Lichen Planus (Phase 10 XAI)
 │    ├── Proof: TreeSHAP confirms 95.7% - 98.7% of decision weight comes from deep branch
 │    └── Effect: Downstream models inherit a feature space where Psoriasis occupies the vast majority of volume
 │
 ├── 5. [CONFIDENCE: MEDIUM] Random Forest Majority Bias (modules/fusion.py)
 │    ├── Evidence: Unconstrained trees (max_depth=None) with Gini impurity splitting
 │    └── Proof: Psoriasis FPs jumped from 74 in Phase 3 Dense head to 127 in Full A7 Random Forest
 │
 ├── 6. [CONFIDENCE: MEDIUM] External Image Domain Shift
 │    ├── Evidence: External images from clinical cameras/web differ from DermNet training distribution
 │    └── Effect: OOD features induce high entropy in RF, triggering the Platt intercept prior collapse
 │
 ├── 7. [CONFIDENCE: LOW] Preprocessing Mismatch (DISPROVEN)
 │    └── Proof: P3-BASE training preprocessing and inference preprocessing are identical (standard mode)
 │
 └── 8. [CONFIDENCE: LOW] Conformal Prediction Flaw (DISPROVEN)
      └── Proof: Conformal thresholding correctly outputs {Pso, PR} and triggers SPECIALIST_REVIEW_REQUIRED
```

---

## 15. Root Cause Summary Table

| Rank | Cause ID | Description | Affected Stage | Confidence | Empirical Proof |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | `CALIB_ARGMAX_FLIP` | Platt Calibration Intercept Prior Inversion | Inference (`modules/inference.py:445`) | **HIGH** | Flips 97/246 (39.4%) calibration samples to Psoriasis; collapses 96.8% of calibration set to Psoriasis. |
| **2** | `BDA_FEATURE_DAMAGE` | BDA Discards Minority Texture & Color Features | Feature Selection (`modules/feature_selection.py`) | **HIGH** | Prunes all LAB features (LAB=0); drops Macro-F1 by 0.0527; increases Psoriasis FPs by +26% (127 $\to$ 160). |
| **3** | `CLASS_IMBALANCE_PRIOR` | Severe 7.25:1 Training Class Imbalance | Data & Cohort (`fold_plan.csv`) | **HIGH** | Psoriasis represents 55.7% of all splits (638 Pso vs 88 SebDerm). |
| **4** | `P3_BACKBONE_BIAS` | Deep Representation Skewed Toward Psoriasis | Backbone (`modules/efficientnet_model.py`) | **HIGH** | Pre-softmax Pso logit is +20.25 on LP; SHAP is 95.7%–98.7% deep branch. |
| **5** | `RF_MAJORITY_BIAS` | Random Forest Gini Impurity Majority Preference | Classifier (`modules/fusion.py`) | **MEDIUM** | Increases Psoriasis FPs from 74 (Phase 3) to 127 (Phase 5). |
| **6** | `DOMAIN_SHIFT_OOD` | Acquisition & Demographics Shift on External Images | Inference Input | **MEDIUM** | OOD inputs induce high entropy, triggering prior collapse. |
| **7** | `PREPROC_MISMATCH` | Preprocessing Code Discrepancy | Preprocessing | **LOW (Disproven)** | Exact match between training and inference code paths. |
| **8** | `CONFORMAL_FAULT` | Conformal Prediction Threshold Flaw | Conformal Prediction | **LOW (Disproven)** | Faithfully outputs multi-class sets and triggers specialist review. |

---

## 16. Proposed V2 Solution & Controlled Experimentation Roadmap

### Principle: Do NOT Change Multiple Components Simultaneously
To maintain causal interpretability and adhere to strict scientific rigor, changes must be tested in a **staged, controlled hierarchy**.

### The Two-Stage V2 Resolution

#### Stage 1 (Immediate, Zero-Retraining Inference Correction): Uncouple Argmax from Calibration
- **The Finding**: The Random Forest classifier was trained to output class probabilities. Platt calibration was trained to estimate posterior probabilities. Using `argmax(calib_probs)` introduces massive prior bias.
- **Hypothesis to Test in V2 Inference**:
  - Determine `predicted_class` via **Raw Random Forest Argmax**:
    $$\text{pred\_idx} = \operatorname{argmax}(\text{raw\_probs}[0])$$
  - Present `calibrated_confidence` and `calibrated_probabilities` strictly as **posterior certainty estimates**, NOT as the class decision rule.
  - Or apply **Prior-Adjusted Calibration** (Temperature Scaling or Isotonic with uniform class priors) so that calibration cannot alter the Bayes decision rule.

#### Stage 2 (Controlled Retraining Experiment: Phase 3 V2): Alpha-Balanced Focal Loss
- If retraining is pursued, it must occur in an isolated namespace: `artifacts/phase3_v2/` and `reports/phase3_v2/`.
- **The Intervention**: Replace ordinary categorical cross-entropy with **$\alpha$-Balanced Focal Loss**:
  $$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  with $\gamma = 2.0$ and $\alpha_t = w_c / \sum w_c$ (normalized class weights).
- **Controlled Invariants**:
  - Exact same 5 frozen folds from `fold_plan.csv`.
  - Exact same 1,146 outer training images.
  - Exact same EfficientNet-B0 architecture and checkpoint selection rule.
  - Exact same epochs (15 Stage 1, 10 Stage 2) and random seeds.
- **Evaluation Criteria**:
  - Pooled Macro-F1 across 5 folds.
  - Per-class recall and precision (specifically Lichen Planus and Pityriasis Rosea).
  - Number of Psoriasis false positives on minority validation samples.

#### Stage 3 (Feature Selection V2): Balanced BDA or Full A7 Retention
- Given that full A7 (1316-D) achieved **0.7264 Macro-F1** and BDA degraded it to **0.6737**, feature selection should either:
  1. Be omitted entirely (retaining Full A7, which preserves all 6 LAB color features and texture descriptors), OR
  2. Re-fit BDA with a **Balanced-F1 / Minimum-Recall objective** and enforce a **Handcrafted Feature Quota** (requiring at least 15% of selected features to come from GLCM, LBP, and LAB).

---

## 17. Strict Protocol Constraints: What Must NOT Be Changed

To preserve scientific provenance and integrity:
1. **DO NOT OVERWRITE V1 ARTIFACTS**:
   - `artifacts/phase3/`
   - `artifacts/phase6/`
   - `artifacts/phase7/`
   - `artifacts/phase8/`
   - `artifacts/phase9/`
   - `reports/phase10/`
2. **DO NOT ACCESS OR EVALUATE THE LOCKED TEST SET**:
   - `output/06_final_split/test/` (243 images) remains completely untouched until final clinical validation.
3. **DO NOT TUNE PARAMETERS AGAINST EXTERNAL GOOGLE IMAGES**:
   - External images are qualitative stress tests only, never validation sets for hyperparameter selection.

---

## 18. Exact Next Action

1. **Acknowledge and review this forensic report** (`reports/final_bias_audit/final_bias_audit.md`).
2. **Decide whether to execute Stage 1 (Inference Adapter prior correction)** or **Stage 2 (Controlled Phase 3 V2 Focal Loss retraining)**.
3. **Maintain frozen status on all V1 assets**.

---

## 19. Final Question Answered Directly

> **"Why does the current PAPULONET V1 pipeline repeatedly predict Psoriasis for external images from other disease classes, and what is the minimum scientifically defensible change we should test first?"**

### Direct Answer

1. **Why it happens**:
   The collapse to Psoriasis is driven by **Platt scaling intercept bias operating at the point of prediction**:
   - In `modules/inference.py`, line 445 computes `pred_class = argmax(calib_probs)`.
   - Platt scaling was fit on an imbalanced calibration set (55.7% Psoriasis vs 7.7% SebDerm), learning intercepts that give Psoriasis a **$5.8\times$ odds advantage** over minority classes.
   - When an external image is processed, the domain shift creates uncertainty in the Random Forest, yielding diffuse probabilities ($20\% - 35\%$ per class).
   - Platt calibration automatically scales diffuse inputs into **$41\% - 56\%$ for Psoriasis**.
   - Because inference selects the argmax of calibrated probabilities, **Platt scaling inverts the prediction into Psoriasis** (empirically proven to flip 39.4% of validation images and collapse 96.75% of the calibration set to Psoriasis).
   - This was compounded upstream by BDA feature selection, which used an unweighted logistic proxy and pruned away all 6 LAB color features and 1,090 deep features, increasing Psoriasis false positives by +26%.

2. **The Minimum Scientifically Defensible Change to Test First**:
   **Test uncoupling the class argmax from Platt calibration at inference time (Zero-Retrain Test)**:
   In `modules/inference.py`, set:
   $$\text{pred\_class} = \text{classes}[\operatorname{argmax}(\text{raw\_probs}[0])]$$
   while retaining `calibrated_probabilities` and conformal sets as confidence/ambiguity indicators.
   - This immediately prevents Platt scaling from inverting minority-class predictions into Psoriasis on uncertain inputs.
   - It requires **zero retraining, zero modification of frozen model weights, and zero changes to the feature pipeline**.
   - If retraining is subsequently desired, the minimum causal change is a **controlled Phase 3 V2 experiment comparing $\alpha$-Balanced Focal Loss against Categorical Cross-Entropy** on the identical 5 development folds.
