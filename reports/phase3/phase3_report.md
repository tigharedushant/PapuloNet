# AEF-CRC Phase 3: IEEE-Standard Controlled Evaluation & Single-Winner Certification Report

**Date/Time (UTC)**: 2026-09-17T14:29:37.106662+00:00
**Pipeline Version**: AEF-CRC Hardened Phase 3
**Verification Status**: CERTIFIED / PASSED

---

## 1. Executive Summary & Selection Decision

- **Authoritative Winning Arm**: `P3-BASE`
- **Primary Selection Metric**: Mean 5-Fold Validation Macro-F1 = **0.7099 ± 0.0186**
- **Mean Validation Accuracy**: **0.7330 ± 0.0214**
- **Mean Balanced Accuracy**: **0.7637 ± 0.0178**
- **Mean Matthews Correlation (MCC)**: **0.5987 ± 0.0386**
- **Selection Decision Rationale**: P3-BASE retained for parsimony. Top candidate P3-AUG Macro-F1 is 0.705752 vs P3-BASE 0.709905 (delta -0.004153 <= 0.005; practically equivalent or inferior).
- **Preprocessing Empirical Verdict**: REMOVE conditional preprocessing (Macro-F1 delta -0.0754 <= 0.005 margin; prefer simpler standard config)
- **Augmentation Empirical Verdict**: REMOVE training augmentation (Macro-F1 delta -0.0042 <= 0.005 margin; prefer simpler unaugmented config)
- **Machine-Readable Contract**: [`winner.json`](winner.json)
- **Fold-by-Fold Metrics Table**: [`fold_metrics.csv`](fold_metrics.csv)
- **Cross-Validation Summary Table**: [`fold_summary.csv`](fold_summary.csv)

---

## 2. Dataset Freeze & Integrity Verification

Phase 3 strictly operates upon the read-only, certified dataset foundation established in Phase 1:
- **Dataset Directory**: `datasets/` (verified 1,899 primary JPEG clinical dermatology images)
- **Metadata Source**: `reports/metadata.csv` (SHA-256: `606bdb9d5c8f622474ff29e6beb0dfa85d775d31d6dcd8261d6fbc0727d64e16`)
- **Dataset Freeze State**: `reports/aef_crc/dataset_freeze.json` (SHA-256: `5a2471fb972f4824fc5eba19b4ac8ed9cf12bdd8e44cf0654104bf0c837b491d`)
- **Data Integrity Invariant**: All images are read directly as inputs without modifying the underlying raw or harmonized files.

---

## 3. 5-Fold Cross-Validation Architecture & Partition Integrity

Cross-validation strictly executes the authoritative Phase 2 partition plan (`reports/aef_crc/fold_plan.csv`):
- **CV Population ($N_{CV}$)**: Exactly 1,146 original training images.
- **Quarantine Enforcement**: Exactly 264 pre-augmented SkinDisNet disk images strictly quarantined and excluded from all CV folds.
- **Holdout Partitions**: Exactly 246 outer validation and 243 outer test images kept completely untouched and unreferenced during Phase 3 model selection.
- **Disjointness Invariant**: Every one of the 1,146 CV samples is evaluated in validation exactly once; fold validation slices are strictly pairwise-disjoint.

---

## 4. Hardware, Runtime & Reproducibility Environment

- **Device Type**: `GPU`
- **Compute Hardware**: `NVIDIA GeForce RTX 2050`
- **Device Count**: `1`
- **Execution Strategy**: `OneDeviceStrategy`
- **Fallback Triggered**: `False` (No silent fallback; explicitly audited)
- **TensorFlow Version**: `2.18.0`
- **Python Version**: `3.12.3`
- **CUDA / cuDNN Support**: `Enabled`
- **Mixed Precision Policy**: `mixed_float16`
- **Global Random Seed**: `42` (strictly synchronized across Python, NumPy, TensorFlow, and cuDNN deterministic ops)

---

## 5. Controlled Experimental Arms Protocol

Three tightly controlled arms were executed across all 5 authoritative folds ($3 \times 5 = 15$ total training runs):

| Arm ID | Preprocessing Mode | Training Augmentation | Epochs (Stage 1 / 2) | Batch Size | Learning Rates | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P3-BASE** | Standard (resize + norm) | OFF | 15 / 10 | 16 | 1e-4 / 1e-5 | Authoritative Baseline |
| **P3-PRE** | Conditional (dull razor + CLAHE) | OFF | 15 / 10 | 16 | 1e-4 / 1e-5 | Evaluate Preprocessing Delta |
| **P3-AUG** | Standard (resize + norm) | Fold-Safe On-the-Fly | 15 / 10 | 16 | 1e-4 / 1e-5 | Evaluate Augmentation Delta |

All other training conditions (EfficientNet-B0 backbone, ImageNet pretraining, optimizer, sample weighting formula, loss function, checkpointing criteria) were held strictly invariant.

---

## 6. Aggregate Cross-Validation Performance Comparison

Performance summary aggregated across the 5 authoritative cross-validation folds ($mean \pm std$):

| Experiment Arm | Macro-F1 (Primary) | Accuracy | Balanced Accuracy | Weighted F1 | Matthews Corr (MCC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **P3-BASE** | **0.7099 ± 0.0186** | 0.7330 ± 0.0214 | 0.7637 ± 0.0178 | 0.7369 ± 0.0237 | 0.5987 ± 0.0386 |
| **P3-PRE** | **0.6345 ± 0.0282** | 0.6597 ± 0.0248 | 0.6910 ± 0.0295 | 0.6661 ± 0.0259 | 0.4946 ± 0.0451 |
| **P3-AUG** | **0.7058 ± 0.0139** | 0.7295 ± 0.0176 | 0.7598 ± 0.0159 | 0.7337 ± 0.0198 | 0.5934 ± 0.0321 |

---

## 7. Fold-by-Fold Performance Breakdown (All 15 Authoritative Runs)

| Experiment | Fold | Macro-F1 | Accuracy | Balanced Acc | Weighted F1 | MCC | Sel. Stage | Sel. Epoch | Best Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| P3-BASE | Fold 0 | 0.7397 | 0.7609 | 0.7840 | 0.7684 | 0.6514 | Stage 2 | 8 | 0.6229 |
| P3-BASE | Fold 1 | 0.7187 | 0.7511 | 0.7644 | 0.7564 | 0.6282 | Stage 2 | 8 | 0.6036 |
| P3-BASE | Fold 2 | 0.7064 | 0.7336 | 0.7810 | 0.7380 | 0.6030 | Stage 2 | 9 | 0.6582 |
| P3-BASE | Fold 3 | 0.6838 | 0.7162 | 0.7359 | 0.7173 | 0.5608 | Stage 2 | 9 | 0.6874 |
| P3-BASE | Fold 4 | 0.7009 | 0.7031 | 0.7533 | 0.7045 | 0.5501 | Stage 2 | 9 | 0.6383 |
| P3-PRE | Fold 0 | 0.6511 | 0.6652 | 0.6827 | 0.6727 | 0.4999 | Stage 2 | 6 | 0.8017 |
| P3-PRE | Fold 1 | 0.6554 | 0.6900 | 0.7071 | 0.6987 | 0.5521 | Stage 2 | 7 | 0.7543 |
| P3-PRE | Fold 2 | 0.6049 | 0.6332 | 0.6775 | 0.6387 | 0.4540 | Stage 2 | 9 | 0.7964 |
| P3-PRE | Fold 3 | 0.5962 | 0.6288 | 0.6500 | 0.6335 | 0.4336 | Stage 2 | 5 | 0.8519 |
| P3-PRE | Fold 4 | 0.6648 | 0.6812 | 0.7376 | 0.6868 | 0.5332 | Stage 2 | 9 | 0.7812 |
| P3-AUG | Fold 0 | 0.7311 | 0.7565 | 0.7762 | 0.7639 | 0.6430 | Stage 2 | 9 | 0.6261 |
| P3-AUG | Fold 1 | 0.7013 | 0.7336 | 0.7480 | 0.7398 | 0.6030 | Stage 2 | 9 | 0.5981 |
| P3-AUG | Fold 2 | 0.7064 | 0.7336 | 0.7810 | 0.7380 | 0.6030 | Stage 2 | 9 | 0.6548 |
| P3-AUG | Fold 3 | 0.6890 | 0.7205 | 0.7407 | 0.7222 | 0.5681 | Stage 2 | 9 | 0.6851 |
| P3-AUG | Fold 4 | 0.7009 | 0.7031 | 0.7533 | 0.7045 | 0.5501 | Stage 2 | 9 | 0.6358 |

---

## 8. Comprehensive Per-Class Performance Breakdown

Per-class evaluation metrics aggregated across folds ($mean \pm std$):

### Per-Class Metrics: `P3-BASE`

| Disease Class | Precision | Recall | F1-Score | Total Val Support |
| :--- | :---: | :---: | :---: | :---: |
| **Psoriasis** | 0.8631 ± 0.0508 | 0.7163 ± 0.0211 | 0.7822 ± 0.0269 | 638 |
| **Lichen_Planus** | 0.6471 ± 0.0368 | 0.6823 ± 0.1046 | 0.6594 ± 0.0505 | 258 |
| **Pityriasis_Rosea** | 0.6148 ± 0.0515 | 0.8269 ± 0.0176 | 0.7044 ± 0.0387 | 162 |
| **Seborrheic_Dermatitis** | 0.6036 ± 0.0840 | 0.8294 ± 0.0824 | 0.6936 ± 0.0597 | 88 |

### Per-Class Metrics: `P3-PRE`

| Disease Class | Precision | Recall | F1-Score | Total Val Support |
| :--- | :---: | :---: | :---: | :---: |
| **Psoriasis** | 0.8156 ± 0.0468 | 0.6473 ± 0.0042 | 0.7212 ± 0.0179 | 638 |
| **Lichen_Planus** | 0.5771 ± 0.0585 | 0.6048 ± 0.0916 | 0.5857 ± 0.0517 | 258 |
| **Pityriasis_Rosea** | 0.4955 ± 0.0453 | 0.7282 ± 0.0752 | 0.5893 ± 0.0547 | 162 |
| **Seborrheic_Dermatitis** | 0.5563 ± 0.0772 | 0.7837 ± 0.1055 | 0.6418 ± 0.0392 | 88 |

### Per-Class Metrics: `P3-AUG`

| Disease Class | Precision | Recall | F1-Score | Total Val Support |
| :--- | :---: | :---: | :---: | :---: |
| **Psoriasis** | 0.8604 ± 0.0456 | 0.7132 ± 0.0216 | 0.7793 ± 0.0250 | 638 |
| **Lichen_Planus** | 0.6423 ± 0.0399 | 0.6822 ± 0.0966 | 0.6577 ± 0.0493 | 258 |
| **Pityriasis_Rosea** | 0.6143 ± 0.0527 | 0.8146 ± 0.0214 | 0.6995 ± 0.0393 | 162 |
| **Seborrheic_Dermatitis** | 0.5915 ± 0.0767 | 0.8294 ± 0.0824 | 0.6865 ± 0.0602 | 88 |

---

## 9. Out-of-Fold Aggregated Confusion Matrices (4x4)

Element-wise sum of confusion matrices across the 5 non-overlapping validation partitions:

### Confusion Matrix: `P3-BASE` (Total Out-of-Fold Samples = 1,146)

| True \ Pred | Psoriasis | Lichen_Planus | Pityriasis_Rosea | Seborrheic_Dermatitis | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Psoriasis** | 457 | 83 | 58 | 40 | **638** |
| **Lichen_Planus** | 49 | 176 | 25 | 8 | **258** |
| **Pityriasis_Rosea** | 18 | 8 | 134 | 2 | **162** |
| **Seborrheic_Dermatitis** | 7 | 6 | 2 | 73 | **88** |

### Confusion Matrix: `P3-PRE` (Total Out-of-Fold Samples = 1,146)

| True \ Pred | Psoriasis | Lichen_Planus | Pityriasis_Rosea | Seborrheic_Dermatitis | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Psoriasis** | 413 | 100 | 84 | 41 | **638** |
| **Lichen_Planus** | 58 | 156 | 33 | 11 | **258** |
| **Pityriasis_Rosea** | 26 | 12 | 118 | 6 | **162** |
| **Seborrheic_Dermatitis** | 11 | 5 | 3 | 69 | **88** |

### Confusion Matrix: `P3-AUG` (Total Out-of-Fold Samples = 1,146)

| True \ Pred | Psoriasis | Lichen_Planus | Pityriasis_Rosea | Seborrheic_Dermatitis | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Psoriasis** | 455 | 84 | 57 | 42 | **638** |
| **Lichen_Planus** | 49 | 176 | 25 | 8 | **258** |
| **Pityriasis_Rosea** | 19 | 9 | 132 | 2 | **162** |
| **Seborrheic_Dermatitis** | 7 | 6 | 2 | 73 | **88** |

---

## 10. Practical Equivalence Margin Analysis & Parsimony Rule

Under the predefined IEEE-grade experimental protocol:
- **Practical Equivalence Threshold ($\Delta_{{equiv}}$)**: $0.005$ in validation Macro-F1.
- **Decision Logic**: If a more complex arm (conditional preprocessing or on-the-fly augmentation) outperforms the baseline by $\le 0.005$, the simpler baseline (`P3-BASE`) is preferred to maximize scientific parsimony and clinical interpretability.
- **Baseline (`P3-BASE`) Macro-F1**: `0.7099`
- **Conditional Preprocessing (`P3-PRE`) Macro-F1**: `0.6345` (Delta vs BASE: `-0.0754`)
- **Augmentation (`P3-AUG`) Macro-F1**: `0.7058` (Delta vs BASE: `-0.0042`)
- **Formal Decision Rationale**: P3-BASE retained for parsimony. Top candidate P3-AUG Macro-F1 is 0.705752 vs P3-BASE 0.709905 (delta -0.004153 <= 0.005; practically equivalent or inferior).

---

## 11. Deep Feature Representation & Downstream Phase 4 Handoff Contract

- **Certified Architecture**: `P3-BASE` (Backbone: EfficientNet-B0)
- **Extracted Layer**: Global average pooling representation (`1280-D`)
- **Cache Location**: `artifacts/phase3/deep_features/`
- **Feature Vectors Stored**:
  - `train_features_fold_XX.npy` & `train_labels_fold_XX.npy` (exactly 916 or 917 training samples per fold)
  - `val_features_fold_XX.npy` & `val_labels_fold_XX.npy` (exactly 230 or 229 validation samples per fold)
- **Integrity Contract**: Downstream Phase 4 (Handcrafted Feature Extraction) and Phase 5/6 (Fusion and Ensemble Classification) consume these exact frozen feature representations without requiring retraining of the deep CNN backbone.

---

## 12. Verification & Audit Compliance Checklist

- [x] **Zero Data Leakage**: Quarantined augmented images excluded; holdout val/test partitions untouched.
- [x] **Fixed Epoch Schedule**: Exactly 15 epochs in Stage 1 and 10 epochs in Stage 2 executed without early stopping bias.
- [x] **Sample Weighting Correctness**: Class weights computed strictly from each fold's training slice via $w_c = N_{{train}} / (C \cdot N_c)$.
- [x] **Checkpoint Integrity**: Model reload with `compile=False` performed before fold evaluation.
- [x] **Disjoint Validation**: All 1,146 CV images evaluated exactly once across the 5 disjoint validation splits.
- [x] **Device Provenance**: Hardware execution context tracked and audited in every fold manifest and summary table.

---
*Report compiled automatically by AEF-CRC Phase 3 Certification Engine.*