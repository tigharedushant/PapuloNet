# AEF-CRC Phase 7: Final Model + Evaluation Pipeline Report

## 1. Executive Summary
Phase 7 finalized and evaluated the complete AEF-CRC pipeline:
- **Representation**: A7 (EfficientNet+GLCM+LBP+LAB, 1316-D).
- **Deep Features**: P3-BASE backbone checkpoint (`artifacts/phase3/P3-BASE/fold_00/best_model.keras`).
- **Feature Selection**: Production Binary Dragonfly Algorithm (BDA) mask (`artifacts/phase6/production_bda_mask.joblib`, 194 features).
- **Classifier**: Primary Random Forest (300 estimators, Gini criterion, sqrt max features, fold-local sample weights).

## 2. 5-Fold Stratified Cross-Validation (Task 1)

| Fold | Selected Features | Macro-F1 |
|---|---|---|
| Fold 0 | 238 | 0.7193 |
| Fold 1 | 198 | 0.7339 |
| Fold 2 | 199 | 0.6292 |
| Fold 3 | 197 | 0.6649 |
| Fold 4 | 197 | 0.7058 |
| **Mean ± Std** | **205.8 ± 18.0** | **0.6906 ± 0.0384** |

### Aggregate Cross-Validation Metrics
- **Macro-F1**: `0.6906 ± 0.0384`
- **Balanced Accuracy**: `0.6923 ± 0.0290`
- **MCC**: `0.5547 ± 0.0525`
- **Accuracy**: `0.7303 ± 0.0294`
- **Weighted F1**: `0.7239 ± 0.0357`

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Psoriasis** | 0.7762 ± 0.0393 | 0.8276 ± 0.0134 | 0.8004 ± 0.0200 | 638 |
| **Lichen Planus** | 0.6568 ± 0.0616 | 0.5115 ± 0.1358 | 0.5676 ± 0.1000 | 258 |
| **Pityriasis Rosea** | 0.6546 ± 0.0771 | 0.6917 ± 0.0681 | 0.6712 ± 0.0666 | 162 |
| **Seborrheic Dermatitis** | 0.7207 ± 0.0943 | 0.7386 ± 0.0690 | 0.7234 ± 0.0504 | 88 |

### Summed Confusion Matrix
```
Rows: True class, Columns: Predicted class
Class Order: ['Psoriasis', 'Lichen_Planus', 'Pityriasis_Rosea', 'Seborrheic_Dermatitis']

                      Psoriasis  Lichen_Planus  Pityriasis_Rosea  Seborrheic_Dermatitis
Psoriasis                   528             54                34                     22
Lichen_Planus                98            132                23                      5
Pityriasis_Rosea             39             11               112                      0
Seborrheic_Dermatitis        17              3                 3                     65
```

## 3. Final Production Model Retrain (Task 2)
- **Training Cohort**: 1146 non-augmented outer-train images (reunified across all 5 folds).
- **Quarantine Enforcement**: All 264 unverified augmentations strictly excluded.
- **Feature Selection**: Frozen production BDA mask (`artifacts/phase6/production_bda_mask.joblib`).
  - Total features selected: **194** (190 deep, 1 glcm, 3 lbp, 0 color_lab).
- **Classifier Fit**: Fitted once on unified 1146-image cohort with class weights applied.

## 4. Calibration-Set Evaluation & Handoff (Task 3)
- **Calibration Split**: 246 images from `splitter.py` held-out validation set.
- **Performance on Calibration Set**:
  - Macro-F1: **0.7354**
  - Balanced Accuracy: **0.7540**
  - MCC: **0.6110**
  - Accuracy: **0.7602**
  - Weighted F1: **0.7585**
- **Handoff Artifact**: `artifacts/phase7/calibration_handoff.joblib` written and verified on disk.

## 5. Locked Test Set Isolation
- **Status**: **STRICTLY UNTOUCHED**
- **Number of Images**: 243 images in `final_split/test/`
- **Audit**: The locked test set was neither read, indexed, nor evaluated at any point during Phase 6 or Phase 7.
