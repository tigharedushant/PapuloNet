# V2 METHODOLOGY FREEZE & EXPERIMENTAL PROTOCOL
**AEF-CRC / PapuloNet Dermatology Diagnosis Pipeline (V2 Rebuild)**

- **Project Root**: `/mnt/c/Users/ASUS/Downloads/PSD_HP_AEF_CRC_PopuloNet/PSD_HP`
- **Date of Freeze**: 2026-09-19
- **Status**: **FROZEN METHODOLOGY PROTOCOL (NO EXECUTION / NO TRAINING YET)**
- **Locked Test Isolation**: **STRICTLY LOCKED & UNTOUCHED** (`output/06_final_split/test/`)
- **V1 Protection**: **FROZEN & PRESERVED** (Zero V1 files overwritten)

---

## 1. V1 Lessons Incorporated into V2

The forensic audit of the V1 pipeline identified three major failure modes that led to the severe production **Psoriasis top-1 collapse**:

1. **Inference Decision Decoupling Failure**:
   In V1 (`modules/inference.py`), the classification prediction was computed as $\text{pred\_idx} = \operatorname{argmax}(\text{calibrated\_probs})$ instead of the classifier's native decision rule $\operatorname{argmax}(\text{raw\_probs})$. Because calibration was conducted via 4 independent binary sigmoids with class-prevalence-dependent intercepts ($b_{\text{Pso}} = -1.2045$ vs. $b_{\text{SD}} = -2.9611$, representing a $+1.7566$ log-odds advantage for Psoriasis), any uncertain, diffuse probability vector collapsed to Psoriasis after normalization.
   - **V2 Mandate**: The point-classification decision rule $\hat{Y} \in \{1, \dots, C\}$ must be defined independently of calibration (e.g., raw classifier argmax or cost-sensitive Bayes decision rule). Calibration will only be used for posterior probability estimates and conformal prediction.

2. **Multiclass Calibration Distortion**:
   V1 relied on one-vs-rest Platt scaling (4 uncoupled 1D logistic regressions). Sigmoids fitted independently on imbalanced data fail to preserve ranking and severely distort multi-class posterior distributions when renormalized.
   - **V2 Mandate**: One-vs-rest independent sigmoid calibration is strictly prohibited. V2 must evaluate multiclass calibration schemes (Multiclass Temperature Scaling, Vector Scaling, Dirichlet Calibration, or Multiclass Isotonic Regression) that model joint class interactions.

3. **Compromised Feature Selection (Phase 6 BDA)**:
   In V1, the Binary Dragonfly Algorithm used an unweighted proxy Logistic Regression (`class_weight=None`) on an imbalanced training set (55.7% Psoriasis). This led to extreme feature instability (pairwise Jaccard index of $0.0761$), stripped out almost all handcrafted features (retaining 190 deep, 1 GLCM, 3 LBP, 0 LAB), and degraded cross-validation Macro-F1 from $0.7270$ (A7 Full) down to $0.6906$.
   - **V2 Mandate**: Full multimodal fusion (A7 1316-D) is the primary production path. BDA is demoted to a secondary controlled comparator, evaluated only with a balanced proxy and strict stability checks.

4. **Missing Focal Loss Evaluation**:
   Phase 3 in V1 documented focal loss but only executed standard categorical cross-entropy with inverse class sample weights.
   - **V2 Mandate**: Phase 3 V2 will execute a rigorous, controlled, 3-arm head-to-head loss comparison: Class-Weighted Cross-Entropy vs. Focal Loss vs. Class-Weighted Focal Loss.

---

## 2. Phase 2: Frozen Foundation (Reused 100% Unchanged)

Phase 2 was forensically verified as completely leak-free, correctly stratified, and mathematically sound. It remains frozen and authoritative for V2.

### Authoritative Files & Verified Hashes
- **Dataset Freeze Manifest**: `reports/aef_crc/dataset_freeze.json`
  - SHA-256: `964d91f8178d5bb7123a7cc02388cd57912bba1e836b2a3ab70b9fa98441919f`
- **Fold Plan**: `reports/aef_crc/fold_plan.csv`
  - SHA-256: `3c54f09f834632f8f771b46d5a98fb773aa6ee78878b00a0c58806c575b8f7b9`
- **Metadata**: `reports/metadata.csv`
  - SHA-256: `b443fa6c1f8ac3d02347de05a0e2a1274995ac61db5146bd8521923facca81e5`

### Cohort Accounting
- **Total Images**: **1,899 images**
- **Permitted Development Cohort (5-Fold CV)**: **1,146 original images**
  - Psoriasis: 638 (55.67%)
  - Lichen Planus: 258 (22.51%)
  - Pityriasis Rosea: 162 (14.14%)
  - Seborrheic Dermatitis: 88 (7.68%)
- **Outer Validation Partition (`val/`)**: **246 original images**
  - Reserved strictly for Phase 8 calibration and Phase 9 conformal prediction.
- **Outer Test Partition (`test/` - LOCKED)**: **243 original images**
  - Strictly quarantined and untouchable throughout V2 development.
- **Quarantined SkinDisNet Augmented Images**: **264 images** (`fold_id: -1`)
  - Excluded from all CV training, evaluation, and final retrain.

### Fold Distribution Invariance
Folds 0–4 partition the 1,146 development images into disjoint validation sets:
- **Fold 0**: 916 train / 230 val
- **Fold 1**: 917 train / 229 val
- **Fold 2**: 917 train / 229 val
- **Fold 3**: 917 train / 229 val
- **Fold 4**: 917 train / 229 val

---

## 3. Phase 3 V2: Controlled Loss & Imbalance Experiment

### Objective
Determine whether Focal Loss or Class-Weighted Focal Loss outperforms the V1 baseline (Class-Weighted Categorical Cross-Entropy) in mitigating majority-class bias and enhancing minority-class discrimination without degrading overall accuracy.

### Experimental Arms (3 Controlled Treatments)

| Arm ID | Loss Function | Loss Formula | Class Weight / $\alpha$ Handling | Purpose |
|:---:|:---|:---|:---|:---|
| **V2-A** | Class-Weighted Categorical Cross-Entropy | $\mathcal{L} = -\sum_{c=1}^C w_c y_c \log(\hat{p}_c)$ | $w_c = \frac{N_{\text{train}}}{C \cdot N_c}$ passed via `sample_weight` | V1 Baseline Replication |
| **V2-B** | Multi-Class Focal Loss (Unweighted) | $\mathcal{L} = -\sum_{c=1}^C (1 - \hat{p}_c)^\gamma y_c \log(\hat{p}_c)$ | $\alpha_c = 1.0$ (uniform), $\gamma = 2.0$ | Pure Hard-Example Mining |
| **V2-C** | Class-Weighted Multi-Class Focal Loss | $\mathcal{L} = -\sum_{c=1}^C \alpha_c (1 - \hat{p}_c)^\gamma y_c \log(\hat{p}_c)$ | $\alpha_c = w_c = \frac{N_{\text{train}}}{C \cdot N_c}$, $\gamma = 2.0$ | Joint Hard-Example + Imbalance Compensation |

### Exact Mathematical Specification & Numerical Stability
For a ground-truth one-hot vector $\mathbf{y} \in \{0, 1\}^C$ and predicted softmax probabilities $\hat{\mathbf{p}} \in (0, 1)^C$:

1. **Probability Clipping for Numerical Stability**:
   $$\hat{p}_c^{\text{clip}} = \operatorname{clip}(\hat{p}_c, \epsilon, 1.0 - \epsilon), \quad \epsilon = 10^{-7}$$
2. **Focal Modulating Factor**:
   $$(1 - \hat{p}_c^{\text{clip}})^\gamma, \quad \text{preregistered } \gamma = 2.0$$
3. **Weight Application Discipline**:
   - In **V2-A**, sample weights $w_i = w_{c(i)}$ are passed natively into `model.fit(train_ds)`. Loss is standard `categorical_crossentropy`.
   - In **V2-B**, custom `CategoricalFocalLoss(gamma=2.0, alpha=None)` is compiled into the model. No `sample_weight` is passed.
   - In **V2-C**, class weights $\alpha_c = w_c$ are embedded directly into the loss function $\sum_c \alpha_c (1 - \hat{p}_c)^\gamma y_c \log(\hat{p}_c)$, **OR** `CategoricalFocalLoss(gamma=2.0)` is trained with `sample_weight=sw_train`. Under no circumstances will weights be applied both in the loss definition and via `sample_weight` (zero double-weighting guarantee).

### Invariant Hyperparameters (Strictly Held Constant Across V2-A, V2-B, V2-C)
- **Backbone**: EfficientNet-B0 (`imagenet` weights, `pooling='avg'`, 1280-D).
- **Input Dimensions**: $224 \times 224 \times 3$, bilinear resize, $[0, 1]$ standard normalization.
- **Classification Head**: GlobalAveragePooling (1280-D) $\to$ Dropout ($p=0.20$) $\to$ Dense (4 classes, `softmax`, `dtype='float32'`).
- **Two-Stage Schedule**:
  - Stage 1: Backbone frozen, Adam ($\text{lr} = 10^{-4}$), 15 epochs.
  - Stage 2: Top 20 layers unfrozen (`unfrozen_layers=20`), BatchNorm frozen, Adam ($\text{lr} = 10^{-5}$), 10 epochs.
- **Augmentation**: None during baseline feature extraction (Standard preprocessing `preprocessing_mode='standard'`).
- **Checkpoint Selection**: Minimum validation loss (`val_loss`) per fold.
- **Random Seed**: Synchronized `seed = 42`.

---

## 4. Phase 3 V2: Evaluation Metrics & Model Selection Rule

### Mandatory Evaluation Metrics (Per Fold and 5-Fold Aggregate)
1. **Aggregate**: Macro-F1, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Accuracy, Weighted F1.
2. **Per-Class Breakdown**: Precision, Recall, F1-Score for Psoriasis, Lichen Planus, Pityriasis Rosea, Seborrheic Dermatitis.
3. **Imbalance Diagnostics**:
   - Psoriasis False Positive Rate ($\text{FPR}_{\text{Pso}} = \frac{\text{FP}_{\text{Pso}}}{\text{FP}_{\text{Pso}} + \text{TN}_{\text{Pso}}}$).
   - Minority Class Recall Floor: $\min(\text{Recall}_{\text{LP}}, \text{Recall}_{\text{PR}}, \text{Recall}_{\text{SD}})$.
   - Prediction Class Distribution vs. True Support.

### Preregistered Phase 3 Selection Rule
The winner of Phase 3 V2 will be selected using a multi-criteria clinical objective, **NOT Macro-F1 alone**:

$$\text{Score} = \text{Macro-F1} + \text{Balanced Accuracy} + \text{MCC} - 0.5 \times \text{FPR}_{\text{Pso}}$$

**Gating Criteria**:
1. **Minority Recall Constraint**: The selected arm must achieve $\text{Recall} \ge 65.0\%$ on all three minority classes (LP, PR, SD).
2. **Practical Equivalence**: If the top candidate exceeds V2-A by $\le 0.005$ in Macro-F1 and Balanced Accuracy, V2-A is retained for parsimony.
3. **Tie-Breaker**: Highest Seborrheic Dermatitis Recall (the most underrepresented class, $N=88$).

---

## 5. Phase 4 V2: Handcrafted Feature Extraction Protocol

### Verified Feature Definitions (68-D Total Active Handcrafted)
1. **GLCM (12-D)**: Gray-Level Co-occurrence Matrix texture features (contrast, dissimilarity, homogeneity, energy, correlation, ASM at 4 directions $[0, \pi/4, \pi/2, 3\pi/4]$ with distance $d=1$).
2. **LBP (18-D)**: Local Binary Pattern histograms using rotation-invariant uniform patterns:
   - $(P=8, R=1.0) \implies 10$ bins.
   - $(P=16, R=2.0) \implies 18$ bins $\to$ concatenated and normalized.
3. **HOG-PCA (32-D)**: Histograms of Oriented Gradients ($9$ orientations, $8 \times 8$ pixels per cell, $2 \times 2$ cells per block = 1,296 raw features) $\to$ reduced to **32 components** via fold-local PCA.
4. **Color LAB (6-D)**: Mean and standard deviation of L, a, b color channels in CIELAB space.

### Strict Fold-Local PCA Discipline
- PCA is fit strictly on $\mathbf{X}_{\text{train}}^{\text{HOG}}$ for each fold.
- $\mathbf{X}_{\text{val}}^{\text{HOG}}$ is transformed using the fold-specific PCA model.
- Handcrafted features are deterministic: caches from V1 will be verified against V2 inputs; if identical, they are linked or copied to `artifacts/phase4_v2/` without modification.

---

## 6. Phase 5 V2: Multimodal Fusion Protocol

### Evaluated Fusion Layouts
All 8 fusion arms from V1 remain candidates:
- **A0**: Deep Backbone Only (1280-D)
- **A1**: Deep + GLCM (1292-D)
- **A2**: Deep + LBP (1298-D)
- **A3**: Deep + HOG-PCA (1312-D)
- **A4**: Deep + Color LAB (1286-D)
- **A5**: Deep + GLCM + LBP (1310-D)
- **A6**: Full Handcrafted (Deep + GLCM + LBP + HOG-PCA + LAB, 1348-D)
- **A7 (Primary Candidate)**: Selective Fusion (Deep + GLCM + LBP + LAB, **1316-D**)

### Classifier Configuration
- **Primary Model**: `RandomForestClassifier` ($n_{\text{estimators}}=300$, `criterion='gini'`, `max_features='sqrt'`, `random_state=42`).
- **Weighting**: `class_weight=None` in constructor; fold-local `sample_weight=sw_train` passed to `clf.fit()` to prevent double-weighting.
- **Secondary Comparator**: Multinomial `LogisticRegression` with balanced sample weights.

---

## 7. Phase 6 V2: Feature Selection Protocol (BDA Decision)

### Policy Decision: A7 Full is the Primary Production Path
Because Phase 6 BDA in V1 failed severely (Jaccard = $0.0761$, eliminated handcrafted features, degraded Macro-F1), **the full A7 1316-D representation is designated as the V2 Primary Production Architecture.**

### BDA V2 as a Secondary Controlled Comparator Only
If BDA is evaluated in V2, it must strictly comply with the following protocol:
1. **Balanced Proxy Classifier**:
   ```python
   # V2 BDA Requirement:
   clf = LogisticRegression(max_iter=300, class_weight='balanced', random_state=seed)
   ```
   Alternatively, compute fold-local sample weights and pass `sample_weight=sw_train` to `clf.fit()`.
2. **Stability Precondition**:
   BDA will be accepted into production **ONLY IF**:
   - Mean pairwise Jaccard stability across the 5 folds exceeds **$0.40$**.
   - Cross-validation Macro-F1 exceeds full A7 by at least **$+0.005$**.
   - Minority class recalls are not degraded compared to full A7.
3. **Automatic Fallback**:
   If these conditions are not met, BDA is rejected, and full A7 is deployed for Phase 7.

---

## 8. Phase 7 V2: Final Classifier Retraining Protocol

### Retraining Architecture
- **Input Representation**: Selected Phase 5/6 Representation (Full A7 1316-D by default).
- **Training Population**: Reunified development cohort = **1,146 images** (Folds 0–4).
- **Excluded Cohorts**:
  - Quarantined augmented images (264) remain strictly excluded.
  - Outer validation holdout (246) remains untouched (reserved for Phase 8/9).
  - Outer test partition (243) remains strictly locked.
- **Sample Weighting**: Computed on all 1,146 development images:
  $$w_c = \frac{1146}{4 \cdot N_c}$$
- **Output Artifact**: `artifacts/phase7_v2/calibration_handoff.joblib`.

---

## 9. Phase 8 V2: Calibration Redesign

### Prohibition of V1 Platt Method
- **FORBIDDEN**: Independent binary sigmoid scaling evaluated via $\operatorname{argmax}(\text{calibrated\_probs})$.

### Decoupling Architecture
In V2, the pipeline explicitly separates:
1. **Categorical Diagnosis Decision ($\hat{Y}$)**:
   $$\hat{Y} = \operatorname{argmax}_{c} \hat{p}_{\text{raw}, c}$$
   (or a Cost-Sensitive Bayes Risk Decision Rule: $\hat{Y} = \operatorname{argmin}_i \sum_j C_{ij} \hat{p}_j$).
2. **Confidence Metric**:
   Properly calibrated posterior probabilities $\mathbf{P}_{\text{cal}}(Y \mid X)$.
3. **Conformal Uncertainty Set**:
   $C(X) \subseteq \{1, \dots, C\}$ constructed at $1 - \alpha$ guarantee.

### Multiclass Calibration Candidates

| Candidate Method | Mathematical Formulation | Parameters | Pros / Cons |
|:---|:---|:---:|:---|
| **Multiclass Temperature Scaling (Primary)** | $\mathbf{P}_{\text{cal}} = \operatorname{softmax}(\mathbf{z} / T)$ where $\mathbf{z} = \ln(\hat{\mathbf{p}}_{\text{raw}} + \epsilon)$ | 1 scalar ($T > 0$) | **Preserves argmax exactly**; monotonic; prevents rank inversion; optimizes NLL. |
| **Vector / Matrix Scaling** | $\mathbf{P}_{\text{cal}} = \operatorname{softmax}(\mathbf{W}\mathbf{z} + \mathbf{b})$ | $C$ or $C \times C$ | More expressive; risk of overfitting on small calibration splits ($N=123$). |
| **Multiclass Isotonic Regression** | Non-parametric monotonic binning per class with softmax normalization | Non-parametric | Achieved ECE = 0.0533 in V1 audit; must be tested with raw argmax decoupling. |
| **Dirichlet Calibration** | Dirichlet distribution multinomial model | $C \times C$ | Sound Bayesian formulation; requires regularization. |

### Protocol for Calibration Selection
- **Fitting Set**: $D_{\text{prob}}$ ($N = 123$ images, stratified outer validation slice).
- **Evaluation Set**: $D_{\text{conf}}$ ($N = 123$ images, held-out outer validation slice).
- **Selection Criteria**:
  1. Rank-order preservation (zero label flips between raw RF and calibrated output).
  2. Lowest Expected Calibration Error (ECE, 10 bins).
  3. Lowest Multiclass Brier Score and Negative Log-Likelihood (NLL).

---

## 10. Phase 9 V2: Conformal Prediction Protocol

### Formal Guarantees
Using the calibrated probabilities on $D_{\text{conf}}$ ($N = 123$), construct conformal prediction sets at nominal coverage $1 - \alpha = 0.90$ ($\alpha = 0.10$):
1. **Marginal Conformal Prediction**:
   - Non-conformity score: $s_i = 1 - \hat{P}_{\text{cal}}(y_i \mid x_i)$.
   - Quantile threshold: $\hat{q} = \text{Quantile}\left(\frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$.
   - Prediction set: $C(x) = \{c : \hat{P}_{\text{cal}}(c \mid x) \ge 1 - \hat{q}\}$.
2. **Class-Conditional Mondrian Conformal Prediction**:
   - Independent quantiles $\hat{q}_c$ calculated per class $c \in \{1, 2, 3, 4\}$.
   - Explicit sparse-class flag if $N_c < 15$ (specifically for Seborrheic Dermatitis, $N \approx 9$).
3. **Reporting Metrics**: Empirical coverage, mean prediction set size, singleton rate, empty set rate.
4. **Boundary**: Conformal sets will NOT be represented as out-of-distribution (OOD) detectors.

---

## 11. Phase 10 V2: Explainable AI (XAI) Protocol

### Dual-Tier Interpretability Engine
1. **Spatial Visual Attention (Grad-CAM)**:
   - Target: Pre-softmax logit layer of the fine-tuned V2 EfficientNet-B0 backbone.
   - Evaluated on verified validation samples across all 4 disease classes.
   - Output: High-resolution heatmap overlays ($224 \times 224$) highlighting localized lesion features.
2. **Feature & Branch Attribution (TreeSHAP)**:
   - Target: Production Random Forest model over the final selected feature set (1316-D).
   - Parameters: `feature_perturbation='tree_path_dependent'`, `model_output='raw'`.
   - Output: Summary plots and per-sample branch importance (Deep vs. GLCM vs. LBP vs. LAB).
   - Evaluation: Verify that handcrafted features receive non-trivial attribution and deep features do not completely saturate decisions.

---

## 12. Complete V2 Experimental Rebuild vs. V1 Comparison

| Phase | V1 Problem | V2 Methodology Change | Scientific Rationale | Retraining Required? |
|:---|:---|:---|:---|:---:|
| **Phase 2** | None (100% sound) | **Reuse 100% unchanged** | Verified leak-free; preserves identical splits and locked test. | **NO** |
| **Phase 3** | Focal Loss was missing in code; only weighted cross-entropy executed | **Controlled 3-arm experiment**: V2-A (Weighted CE), V2-B (Focal $\gamma=2$), V2-C (Weighted Focal) | Explicitly evaluate hard-example mining vs. frequency weighting. | **YES** |
| **Phase 4** | None (feature definitions were sound) | **Re-extract / link deterministic 68-D handcrafted features** | Maintain fold-local PCA and strict normalization discipline. | **NO** (reuse deterministic logic) |
| **Phase 5** | None (A7 selected legitimately) | **Re-evaluate fusion arms A0–A7 on new V2 Phase 3 backbone** | Ensure handcrafted features complement the updated deep representation. | **YES** |
| **Phase 6** | Unweighted proxy LR; Jaccard = 0.0761; degraded performance | **Full A7 1316-D is Primary Path**; BDA demoted to secondary comparator with balanced proxy | Prevent catastrophic feature loss and unstable selection. | **YES** |
| **Phase 7** | Final RF was fitted on defective 194-D BDA mask | **Fit final RF on verified representation (A7 1316-D)** with fold-local sample weights | Deploy optimal feature representation to final classifier. | **YES** |
| **Phase 8** | Platt sigmoids caused 97 flips to Psoriasis and 96.75% collapse | **Multiclass Temperature Scaling / Isotonic Regression**; strictly decouple raw argmax decision | Eliminate intercept bias; guarantee rank preservation. | **YES** |
| **Phase 9** | Conformal sets inherited distorted Platt probabilities | **Re-fit conformal quantiles on verified V2 calibrated probabilities** | Ensure valid marginal and Mondrian coverage without majority-set inflation. | **YES** (recompute) |
| **Phase 10** | 7/8 samples collapsed to Psoriasis; deep branch dominated 97% | **Regenerate Grad-CAM and TreeSHAP on final V2 model** | Provide genuine multi-modal interpretability across all classes. | **YES** (recompute) |
| **Inference** | `pred_idx = argmax(calib_probs)` caused production collapse | **`pred_idx = argmax(raw_probs)`**; calibrated probs reported as confidence only | Restore true classifier decision boundary. | **NO** (code design rule) |

---

## 13. Artifact & Report Isolation Strategy

All V2 artifacts and reports will be written to dedicated `_v2` directories to ensure that V1 history is completely preserved:

```
PSD_HP/
├── artifacts/
│   ├── phase3/          <-- V1 (PRESERVED)
│   ├── phase3_v2/       <-- V2 (NEW)
│   ├── phase4_v2/       <-- V2 (NEW)
│   ├── phase5_v2/       <-- V2 (NEW)
│   ├── phase6_v2/       <-- V2 (NEW)
│   ├── phase7_v2/       <-- V2 (NEW)
│   ├── phase8_v2/       <-- V2 (NEW)
│   ├── phase9_v2/       <-- V2 (NEW)
│   └── phase10_v2/      <-- V2 (NEW)
└── reports/
    ├── phase3/          <-- V1 (PRESERVED)
    ├── phase3_v2/       <-- V2 (NEW)
    ├── ...
    └── phase10_v2/      <-- V2 (NEW)
```

---

## 14. Conditions Under Which V2 is Considered Successful

The V2 pipeline will be certified as **successful and production-ready** if and only if all of the following conditions are met on cross-validation and outer validation:

1. **Macro-F1 & Balanced Accuracy**: 5-fold CV Macro-F1 $\ge 0.72$ and Balanced Accuracy $\ge 0.74$.
2. **Minority Class Performance**:
   - Lichen Planus Recall $\ge 65.0\%$
   - Pityriasis Rosea Recall $\ge 70.0\%$
   - Seborrheic Dermatitis Recall $\ge 70.0\%$
3. **Psoriasis False Positive Control**:
   - Psoriasis False Positive Rate $\le 15.0\%$ (no majority-class flooding).
   - Zero systematic argmax inversion flips from non-Psoriasis to Psoriasis.
4. **Calibration Quality**:
   - Multiclass ECE $\le 0.10$ on the outer validation set.
   - Raw decision ranking preserved ($T > 0$ temperature scaling or rank-preserving mapping).
5. **Conformal Validity**:
   - Marginal coverage $\ge 89.0\%$ at $\alpha = 0.10$.
   - Mean set size $\le 1.80$.
6. **Interpretability Balance**:
   - TreeSHAP demonstrates statistically significant contributions ($> 5\%$) from handcrafted texture/color features on appropriate classes.

---

## 15. Final Safety & Protection Statement

We certify:
- **Zero training has been executed.**
- **Zero production code has been modified.**
- **The locked 243-image test partition remains completely pristine and untouched.**
- **The V1 pipeline and audit artifacts remain 100% intact.**

*V2 Methodology Protocol Frozen. Awaiting explicit user approval before any code implementation or training execution.*
