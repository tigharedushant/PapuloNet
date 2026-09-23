# AEF-CRC Phase 10: Explainable AI (XAI) & Interpretability Report

## Executive Summary
- **Timestamp**: 2026-09-22T08:53:39.079663+00:00
- **Representation ID**: `efficientnet_b0_43d581b96f8ec368`
- **Fusion Layout**: `A7` (Full dimension: 1316-D)
- **Production Feature Selection**: BDA active features = 642 / 1316
- **Production Classifier**: `random_forest` (n_features_in = 642)
- **Probability Calibration**: `Temperature_scaling` scaling
- **Conformal Coverage (Nominal)**: 90.0% (Marginal quantile $\hat{q} = 0.7382$)
- **Evaluated Cohort**: 8 representative validation samples (Holdout val cohort; locked test set strictly isolated).
- **Validation Subset Accuracy**: 3/8 (37.5%)

---

## Scientific Rigor & Methodological Boundaries

> [!IMPORTANT]
> **Two-Tier Interpretability Division of Responsibility**:
> 1. **Grad-CAM (Spatial CNN Representation)**: Operates exclusively on the TensorFlow/Keras EfficientNet-B0 backbone, targeting the **pre-softmax logit** of the Phase 3 Dense-4 classification head. It identifies spatial lesion regions driving visual feature extraction.
> 2. **TreeSHAP (Tabular Multimodal Decision)**: Operates on the production Random Forest classifier across the **194 BDA-selected features** using `model_output='raw'` and `feature_perturbation='tree_path_dependent'`. It quantifies tabular feature contributions and branch importance.
> 3. **Non-Explanation Boundary**: Neither Grad-CAM nor TreeSHAP explains the Platt probability calibration mapping or conformal prediction set boundaries.
> 4. **Locked Outer Test Partition**: The 243-image locked test set was completely untouched and unaccessed.

---

## Sample Evaluation & Explanation Summary

| Sample ID | True Class | Predicted Class | Calib Conf | Conformal Set | Review Status | Top SHAP Branch | Overlay |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PSD_00000917` | Psoriasis | Pityriasis_Rosea | 71.7% | `Psoriasis;Pityriasis_Rosea` | `SPECIALIST_REVIEW_REQUIRED` | deep (96.7%) | [View](gradcam/PSD_00000917/gradcam_overlay.png) |
| `PSD_00001647` | Psoriasis | Pityriasis_Rosea | 91.8% | `Pityriasis_Rosea` | `STANDARD_OUTPUT` | deep (94.0%) | [View](gradcam/PSD_00001647/gradcam_overlay.png) |
| `PSD_00000639` | Lichen_Planus | Pityriasis_Rosea | 61.2% | `Psoriasis;Pityriasis_Rosea` | `SPECIALIST_REVIEW_REQUIRED` | deep (94.0%) | [View](gradcam/PSD_00000639/gradcam_overlay.png) |
| `PSD_00003153` | Lichen_Planus | Lichen_Planus | 75.3% | `Lichen_Planus` | `STANDARD_OUTPUT` | deep (94.7%) | [View](gradcam/PSD_00003153/gradcam_overlay.png) |
| `PSD_00000735` | Pityriasis_Rosea | Pityriasis_Rosea | 46.3% | `Psoriasis;Pityriasis_Rosea` | `SPECIALIST_REVIEW_REQUIRED` | deep (95.9%) | [View](gradcam/PSD_00000735/gradcam_overlay.png) |
| `PSD_00003209` | Pityriasis_Rosea | Lichen_Planus | 99.4% | `Lichen_Planus` | `STANDARD_OUTPUT` | deep (92.9%) | [View](gradcam/PSD_00003209/gradcam_overlay.png) |
| `PSD_00001693` | Seborrheic_Dermatitis | Lichen_Planus | 49.7% | `Lichen_Planus` | `STANDARD_OUTPUT` | deep (94.9%) | [View](gradcam/PSD_00001693/gradcam_overlay.png) |
| `PSD_00001760` | Seborrheic_Dermatitis | Seborrheic_Dermatitis | 83.9% | `Seborrheic_Dermatitis` | `STANDARD_OUTPUT` | deep (92.6%) | [View](gradcam/PSD_00001760/gradcam_overlay.png) |

---

## Artifact Inventory
- **Selected Samples Table**: `selected_samples.csv`
- **Phase 10 Manifest**: `phase10_manifest.json`
- **Grad-CAM Visualizations**:
  - `PSD_00000917`: `gradcam/PSD_00000917/original.png`, `gradcam/PSD_00000917/gradcam_heatmap.png`, `gradcam/PSD_00000917/gradcam_overlay.png`
  - `PSD_00001647`: `gradcam/PSD_00001647/original.png`, `gradcam/PSD_00001647/gradcam_heatmap.png`, `gradcam/PSD_00001647/gradcam_overlay.png`
  - `PSD_00000639`: `gradcam/PSD_00000639/original.png`, `gradcam/PSD_00000639/gradcam_heatmap.png`, `gradcam/PSD_00000639/gradcam_overlay.png`
  - `PSD_00003153`: `gradcam/PSD_00003153/original.png`, `gradcam/PSD_00003153/gradcam_heatmap.png`, `gradcam/PSD_00003153/gradcam_overlay.png`
  - `PSD_00000735`: `gradcam/PSD_00000735/original.png`, `gradcam/PSD_00000735/gradcam_heatmap.png`, `gradcam/PSD_00000735/gradcam_overlay.png`
  - `PSD_00003209`: `gradcam/PSD_00003209/original.png`, `gradcam/PSD_00003209/gradcam_heatmap.png`, `gradcam/PSD_00003209/gradcam_overlay.png`
  - `PSD_00001693`: `gradcam/PSD_00001693/original.png`, `gradcam/PSD_00001693/gradcam_heatmap.png`, `gradcam/PSD_00001693/gradcam_overlay.png`
  - `PSD_00001760`: `gradcam/PSD_00001760/original.png`, `gradcam/PSD_00001760/gradcam_heatmap.png`, `gradcam/PSD_00001760/gradcam_overlay.png`
- **TreeSHAP Feature Importances**:
  - `PSD_00000917`: `shap/PSD_00000917/shap_feature_importance.csv`, `shap/PSD_00000917/shap_block_importance.csv`
  - `PSD_00001647`: `shap/PSD_00001647/shap_feature_importance.csv`, `shap/PSD_00001647/shap_block_importance.csv`
  - `PSD_00000639`: `shap/PSD_00000639/shap_feature_importance.csv`, `shap/PSD_00000639/shap_block_importance.csv`
  - `PSD_00003153`: `shap/PSD_00003153/shap_feature_importance.csv`, `shap/PSD_00003153/shap_block_importance.csv`
  - `PSD_00000735`: `shap/PSD_00000735/shap_feature_importance.csv`, `shap/PSD_00000735/shap_block_importance.csv`
  - `PSD_00003209`: `shap/PSD_00003209/shap_feature_importance.csv`, `shap/PSD_00003209/shap_block_importance.csv`
  - `PSD_00001693`: `shap/PSD_00001693/shap_feature_importance.csv`, `shap/PSD_00001693/shap_block_importance.csv`
  - `PSD_00001760`: `shap/PSD_00001760/shap_feature_importance.csv`, `shap/PSD_00001760/shap_block_importance.csv`

---

*AEF-CRC Phase 10 Interpretability Pipeline Complete.*