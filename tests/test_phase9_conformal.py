"""
tests/test_phase9_conformal.py

Targeted tests for Phase 9 Conformal Prediction:
  - Exact split-conformal quantile arithmetic: k = ceil((n + 1) * (1 - alpha))
    Verifies n=10, alpha=0.10 -> k=10.
    Verifies n=123, alpha=0.10 -> k=112.
  - Prediction set construction for pooled marginal and class-conditional Mondrian.
  - Set evaluation metrics (empirical coverage, mean set size, size distribution).
  - BDA mask compatibility validation (1348-D contract, bool dtype, feature count).
  - ConformalArtifact serialization round-trip through joblib.
  - Leakage guards: validation partition disjointness and outer test exclusion.

NOTE: Per Strict Execution Guard, this test file is created for static integrity
and future CI/CD; it MUST NOT be executed during the current turn.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from modules.conformal import (
    compute_nonconformity_scores,
    compute_conformal_quantile,
    predict_conformal_sets,
    predict_conformal_sets_mondrian,
    fit_marginal_conformal,
    fit_class_conditional_mondrian,
    evaluate_conformal_sets,
)
from modules.calibration import partition_outer_validation
from modules.calibration_handoff import (
    validate_bda_mask_compatibility,
    validate_conformal_handoff_provenance,
    ConformalHandoff,
    ConformalArtifact,
    save_conformal_artifact,
    load_conformal_artifact,
    load_conformal_handoff,
    get_active_run_id,
)
from config.config import get_config
from run_aef_crc_phase6 import align_phase3_winner_config
from modules.experiment_config import representation_id

CLASSES = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]


# ============================================================
# 1. Exact Conformal Quantile Arithmetic
# ============================================================

def test_conformal_quantile_formula_arithmetic():
    """
    Verifies the pre-registered quantile formula:
      k = ceil((n + 1) * (1 - alpha))
    """
    # Case A: n = 10, alpha = 0.10 -> (11) * 0.90 = 9.9 -> ceil = 10 (not 11)
    k_10 = int(math.ceil((10 + 1) * (1.0 - 0.10)))
    assert k_10 == 10, f"Expected k=10 for n=10, alpha=0.10, got {k_10}"

    # Case B: n = 123, alpha = 0.10 -> (124) * 0.90 = 111.6 -> ceil = 112
    k_123 = int(math.ceil((123 + 1) * (1.0 - 0.10)))
    assert k_123 == 112, f"Expected k=112 for n=123, alpha=0.10, got {k_123}"

    # Verify via compute_conformal_quantile on uniform scores
    scores_10 = np.linspace(0.1, 1.0, 10)
    q_10 = compute_conformal_quantile(scores_10, alpha=0.10)
    assert q_10 == scores_10[9]  # 10th element (0-indexed 9)

    scores_123 = np.linspace(0.01, 1.0, 123)
    q_123 = compute_conformal_quantile(scores_123, alpha=0.10)
    assert q_123 == scores_123[111]  # 112th element (0-indexed 111)


def test_conformal_quantile_edge_cases():
    scores = np.array([0.2, 0.5, 0.8])
    with pytest.raises(ValueError):
        compute_conformal_quantile(np.array([]), alpha=0.10)
    with pytest.raises(ValueError):
        compute_conformal_quantile(scores, alpha=0.0)
    with pytest.raises(ValueError):
        compute_conformal_quantile(scores, alpha=1.0)


# ============================================================
# 2. Nonconformity Scores & Prediction Sets
# ============================================================

def test_nonconformity_score_computation():
    probs = np.array([
        [0.70, 0.10, 0.10, 0.10],
        [0.05, 0.80, 0.10, 0.05],
        [0.20, 0.30, 0.40, 0.10],
    ])
    true_indices = np.array([0, 1, 2])
    scores = compute_nonconformity_scores(probs, true_indices)

    assert np.allclose(scores, [0.30, 0.20, 0.60])


def test_predict_conformal_sets_marginal():
    probs = np.array([
        [0.60, 0.25, 0.10, 0.05],
        [0.35, 0.35, 0.20, 0.10],
        [0.05, 0.05, 0.80, 0.10],
    ])
    # Suppose q_hat = 0.70 -> threshold = 1.0 - 0.70 = 0.30
    q_hat = 0.70
    sets = predict_conformal_sets(probs, q_hat, CLASSES)

    # Sample 0: probs >= 0.30 are [0.60] -> ["Psoriasis"]
    assert sets[0] == ["Psoriasis"]
    # Sample 1: probs >= 0.30 are [0.35, 0.35] -> ["Psoriasis", "Lichen_Planus"]
    assert sets[1] == ["Psoriasis", "Lichen_Planus"]
    # Sample 2: probs >= 0.30 are [0.80] -> ["Pityriasis_Rosea"]
    assert sets[2] == ["Pityriasis_Rosea"]


def test_predict_conformal_sets_mondrian():
    probs = np.array([
        [0.60, 0.30, 0.05, 0.05],
    ])
    # Class-specific thresholds:
    # Psoriasis q=0.50 (thresh 0.50), Lichen_Planus q=0.80 (thresh 0.20)
    q_hat_by_class = {
        "Psoriasis": 0.50,
        "Lichen_Planus": 0.80,
        "Pityriasis_Rosea": 0.90,
        "Seborrheic_Dermatitis": 0.90,
    }
    sets = predict_conformal_sets_mondrian(probs, q_hat_by_class, CLASSES)
    # Psoriasis (0.60 >= 0.50) -> True; Lichen_Planus (0.30 >= 0.20) -> True
    assert sets[0] == ["Psoriasis", "Lichen_Planus"]


# ============================================================
# 3. Conformal Evaluation Metrics
# ============================================================

def test_evaluate_conformal_sets_metrics():
    pred_sets = [
        ["Psoriasis"],
        ["Lichen_Planus", "Psoriasis"],
        ["Pityriasis_Rosea"],
        [],  # empty set
    ]
    true_labels = [
        "Psoriasis",
        "Lichen_Planus",
        "Seborrheic_Dermatitis",  # missed
        "Seborrheic_Dermatitis",  # missed
    ]
    metrics = evaluate_conformal_sets(pred_sets, true_labels, CLASSES, nominal_coverage=0.90)

    # 2 out of 4 correct -> marginal coverage = 0.50
    assert metrics.marginal_coverage == 0.50
    assert metrics.mean_set_size == 1.0  # (1 + 2 + 1 + 0) / 4 = 1.0
    assert metrics.median_set_size == 1.0
    assert metrics.singleton_fraction == 0.50  # 2 of 4
    assert metrics.empty_set_fraction == 0.25  # 1 of 4
    assert metrics.set_size_distribution[0] == 1
    assert metrics.set_size_distribution[1] == 2
    assert metrics.set_size_distribution[2] == 1


# ============================================================
# 4. BDA Mask Compatibility Validation
# ============================================================

def test_validate_bda_mask_compatibility():
    class DummyClassifier:
        n_features_in_ = 200

    # Valid mask
    valid_mask = np.zeros(1348, dtype=bool)
    valid_mask[:200] = True
    clf = DummyClassifier()
    validate_bda_mask_compatibility(valid_mask, clf, expected_dim=1348)

    # Wrong dimension
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_bda_mask_compatibility(np.zeros(1280, dtype=bool), clf, expected_dim=1348)

    # Wrong dtype
    with pytest.raises(TypeError, match="boolean dtype"):
        validate_bda_mask_compatibility(np.zeros(1348, dtype=float), clf, expected_dim=1348)

    # Incompatible with classifier feature count
    bad_count_mask = np.zeros(1348, dtype=bool)
    bad_count_mask[:150] = True
    with pytest.raises(ValueError, match="incompatibility"):
        validate_bda_mask_compatibility(bad_count_mask, clf, expected_dim=1348)


# ============================================================
# 5. Validation Partition Disjointness (Leakage Guard)
# ============================================================

def test_validation_partition_exactness_and_disjointness():
    # Synthetic 246-sample validation set
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    labels = ["Psoriasis"] * 80 + ["Lichen_Planus"] * 80 + ["Pityriasis_Rosea"] * 50 + ["Seborrheic_Dermatitis"] * 36
    assert len(labels) == 246
    psd_ids = [f"IMG_{i:04d}" for i in range(246)]

    res = partition_outer_validation(psd_ids, labels, classes, random_seed=42, calib_fraction=0.5)

    assert len(res.val_calib_psd_ids) == 123
    assert len(res.val_conf_psd_ids) == 123
    assert set(res.val_calib_psd_ids).isdisjoint(set(res.val_conf_psd_ids))

    # All 4 classes represented in both splits
    for cls in classes:
        assert res.per_class_calib_counts[cls] > 0
        assert res.per_class_conf_counts[cls] > 0


# ============================================================
# 6. ConformalArtifact Round-Trip
# ============================================================

def test_conformal_artifact_serialization_round_trip(tmp_path):
    artifact = ConformalArtifact(
        representation_id="repr_test",
        experiment_id="exp_test",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=42,
        class_order=CLASSES,
        final_classifier=object(),
        selected_feature_mask=np.ones(1348, dtype=bool),
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "lab": 6},
        calibration_method="platt",
        marginal_q_hat=0.72,
        mondrian_q_hat={"Psoriasis": 0.70, "Lichen_Planus": 0.75, "Pityriasis_Rosea": 0.68, "Seborrheic_Dermatitis": 0.85},
        alpha=0.10,
        n_conf_samples=123,
        per_class_conf_counts={"Psoriasis": 40, "Lichen_Planus": 40, "Pityriasis_Rosea": 25, "Seborrheic_Dermatitis": 18},
    )

    path = tmp_path / "conformal_artifact.joblib"
    save_conformal_artifact(artifact, path)
    reloaded = load_conformal_artifact(path)

    assert reloaded.marginal_q_hat == 0.72
    assert reloaded.mondrian_q_hat["Seborrheic_Dermatitis"] == 0.85
    assert reloaded.alpha == 0.10
    assert reloaded.n_conf_samples == 123
    assert reloaded.calibration_method == "platt"


# ============================================================
# 7. Phase 9 Provenance & P3-BASE Alignment Validation
# ============================================================

def test_phase9_config_alignment_with_p3_base_winner():
    """
    Verifies that aligning config with the authoritative Phase 3 winner (P3-BASE)
    yields the exact expected representation_id: efficientnet_b0_6760c4f151acc2d2,
    whereas raw unaligned config produces the stale default b43260d357d5e51f.
    """
    raw_config = get_config()
    raw_repr = representation_id(raw_config)
    assert raw_repr == "efficientnet_b0_b43260d357d5e51f"

    aligned_config, winner_name = align_phase3_winner_config(get_config())
    aligned_repr = representation_id(aligned_config)
    assert aligned_repr == "efficientnet_b0_6760c4f151acc2d2"
    assert winner_name == "P3-BASE"


def test_phase9_validate_conformal_handoff_provenance_strictness():
    """
    Verifies that validate_conformal_handoff_provenance:
    - Accepts valid handoffs with the authoritative P3-BASE representation_id
    - Strictly rejects mismatched representation_id (including the stale b43260d357d5e51f)
    - Strictly rejects mismatched random_seed
    """
    aligned_config, _ = align_phase3_winner_config(get_config())
    active_run_id = get_active_run_id(aligned_config) or ""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]

    valid_handoff = ConformalHandoff(
        run_id=active_run_id,
        representation_id="efficientnet_b0_6760c4f151acc2d2",
        experiment_id="P3-BASE",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=42,
        class_order=classes,
        final_classifier=object(),
        selected_feature_mask=np.ones(194, dtype=bool),
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
        calibration_method="platt",
        calibration_method_reason="Frozen primary method: Platt scaling",
        val_conf_psd_ids=[f"CONF_{i:04d}" for i in range(123)],
        val_conf_true_labels=["Psoriasis"] * 123,
        val_conf_calibrated_probabilities=np.ones((123, 4)) / 4.0,
        alpha=0.10,
    )

    # Valid handoff must pass
    validate_conformal_handoff_provenance(valid_handoff, aligned_config)

    # Stale / mismatched representation_id must be strictly rejected
    stale_handoff = ConformalHandoff(
        run_id=active_run_id,
        representation_id="efficientnet_b0_b43260d357d5e51f",
        experiment_id="P3-BASE",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=42,
        class_order=classes,
        final_classifier=object(),
        selected_feature_mask=np.ones(194, dtype=bool),
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
        calibration_method="platt",
        calibration_method_reason="Frozen primary method: Platt scaling",
        val_conf_psd_ids=[f"CONF_{i:04d}" for i in range(123)],
        val_conf_true_labels=["Psoriasis"] * 123,
        val_conf_calibrated_probabilities=np.ones((123, 4)) / 4.0,
        alpha=0.10,
    )
    with pytest.raises(RuntimeError, match="representation_id"):
        validate_conformal_handoff_provenance(stale_handoff, aligned_config)

    # Mismatched random seed must be strictly rejected
    seed_mismatch_handoff = ConformalHandoff(
        run_id=active_run_id,
        representation_id="efficientnet_b0_6760c4f151acc2d2",
        experiment_id="P3-BASE",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=999,
        class_order=classes,
        final_classifier=object(),
        selected_feature_mask=np.ones(194, dtype=bool),
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
        calibration_method="platt",
        calibration_method_reason="Frozen primary method: Platt scaling",
        val_conf_psd_ids=[f"CONF_{i:04d}" for i in range(123)],
        val_conf_true_labels=["Psoriasis"] * 123,
        val_conf_calibrated_probabilities=np.ones((123, 4)) / 4.0,
        alpha=0.10,
    )
    with pytest.raises(RuntimeError, match="random_seed"):
        validate_conformal_handoff_provenance(seed_mismatch_handoff, aligned_config)

    # Mismatched run_id must be strictly rejected
    run_id_mismatch_handoff = ConformalHandoff(
        run_id="DIFFERENT_RUN_ID",
        representation_id="efficientnet_b0_6760c4f151acc2d2",
        experiment_id="P3-BASE",
        classifier_name="random_forest",
        feature_selection_method="bda",
        random_seed=42,
        class_order=classes,
        final_classifier=object(),
        selected_feature_mask=np.ones(194, dtype=bool),
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
        calibration_method="platt",
        calibration_method_reason="Frozen primary method: Platt scaling",
        val_conf_psd_ids=[f"CONF_{i:04d}" for i in range(123)],
        val_conf_true_labels=["Psoriasis"] * 123,
        val_conf_calibrated_probabilities=np.ones((123, 4)) / 4.0,
        alpha=0.10,
    )
    with pytest.raises(RuntimeError, match="run_id"):
        validate_conformal_handoff_provenance(run_id_mismatch_handoff, aligned_config)


def test_phase9_existing_phase8_artifact_passes_provenance():
    """
    Verifies that the actual Phase 8 artifact on disk passes provenance validation
    under the aligned Phase 3 winner configuration, and is rejected under unaligned config.
    """
    aligned_config, _ = align_phase3_winner_config(get_config())
    handoff_path = aligned_config.aef_crc_phase8_artifacts_dir / "conformal_handoff.joblib"
    if not handoff_path.exists():
        pytest.skip("Phase 8 artifact conformal_handoff.joblib not present on disk.")

    real_handoff = load_conformal_handoff(handoff_path)
    assert real_handoff.representation_id == "efficientnet_b0_6760c4f151acc2d2"
    assert real_handoff.calibration_method == "platt"
    assert len(real_handoff.val_conf_true_labels) == 123

    # Aligned config accepts the artifact
    validate_conformal_handoff_provenance(real_handoff, aligned_config)

    # Raw unaligned config rejects the artifact with RuntimeError
    raw_config = get_config()
    with pytest.raises(RuntimeError, match="Cross-run artifact provenance mismatch"):
        validate_conformal_handoff_provenance(real_handoff, raw_config)

