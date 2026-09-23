# AEF-CRC / PapuloNet Phase 3 V2: Controlled Loss Arm Evaluation & Selection Report

**Date/Time (UTC)**: 2026-09-20T03:22:50.670627+00:00
**Pipeline Version**: PapuloNet V2 Phase 3
**Verification Status**: CERTIFIED / PASSED

---

## 1. Executive Summary & Selection Decision

- **Authoritative Winning Arm**: `P3-V2-Focal`
- **Primary Selection Metric**: Mean 5-Fold Validation Macro-F1 = **0.7545 ± 0.0242**
- **Mean Validation Accuracy**: **0.7914 ± 0.0224**
- **Mean Balanced Accuracy**: **0.7397 ± 0.0244**
- **Mean Matthews Correlation (MCC)**: **0.6545 ± 0.0391**
- **Selection Decision Rationale**: P3-V2-Focal won with Macro-F1 0.754528, exceeding P3-V2-CE (0.709864) by +0.044665 (threshold > 0.005).
- **Machine-Readable Contract**: [`winner.json`](winner.json)
- **Fold-by-Fold Metrics Table**: [`fold_metrics.csv`](fold_metrics.csv)
- **Cross-Validation Summary Table**: [`fold_summary.csv`](fold_summary.csv)

---

## 2. Dataset Freeze & Partition Integrity

- **Authoritative CV Population**: Exactly 1,146 original non-augmented training images.
- **Quarantine Enforcement**: Exactly 264 pre-augmented SkinDisNet disk images strictly quarantined from CV.
- **Outer Holdout Validation**: Exactly 246 images held out strictly for calibration.
- **Outer Holdout Test (LOCKED)**: Exactly 243 images locked and never accessed.
- **Partition Disjointness**: Each CV sample is validated exactly once across the 5 folds.

---

## 3. Controlled Experimental Arms Protocol

Three loss arms executed under strictly invariant architecture, schedule, and optimizer:

| Arm ID | Loss Formulation | Gamma | Class Weighting Policy | Optimizer | Epochs (S1 / S2) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P3-V2-CE** | Categorical Cross-Entropy | N/A | Fold-local sample_weights | Adam (clipnorm=1.0) | 15 / 10 |
| **P3-V2-Focal** | Categorical Focal Loss | 2.0 | None (unweighted) | Adam (clipnorm=1.0) | 15 / 10 |
| **P3-V2-WFocal** | Categorical Focal Loss | 2.0 | Fold-local sample_weights | Adam (clipnorm=1.0) | 15 / 10 |

---

## 4. Multi-Metric Results Comparison

| Arm ID | Macro-F1 (Mean ± SD) | Balanced Acc | Accuracy | MCC | Psoriasis Recall | Lichen Planus Recall | Pityriasis Rosea Recall | Seborrheic Derm Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P3-V2-CE** | 0.7099 ± 0.0183 | 0.7591 | 0.7347 | 0.5988 | 0.7226 | 0.6939 | 0.8023 | 0.8176 |
| **P3-V2-Focal** | 0.7545 ± 0.0242 | 0.7397 | 0.7914 | 0.6545 | 0.8761 | 0.6514 | 0.7282 | 0.7033 |
| **P3-V2-WFocal** | 0.7312 ± 0.0166 | 0.7659 | 0.7591 | 0.6249 | 0.7711 | 0.6900 | 0.7960 | 0.8065 |

---

## 5. Selection Decision & Parsimony Rule

- **Primary Criterion**: Mean 5-fold CV Macro-F1 with 0.005 practical-equivalence margin over baseline `P3-V2-CE`.
- **Decision**: P3-V2-Focal won with Macro-F1 0.754528, exceeding P3-V2-CE (0.709864) by +0.044665 (threshold > 0.005).

---

*(Report generated automatically by `run_aef_crc_phase3_v2.py`)*