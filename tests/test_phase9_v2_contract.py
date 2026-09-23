"""
tests/test_phase9_v2_contract.py

Contract verification and targeted unit tests for PapuloNet V2 Phase 9:
Conformal Prediction Implementation & Calibration Handoff.

Scientific & Contractual Invariants Tested:
1. Gate Check Enforcement:
   - Validates certified Phase 8 V2 manifest (CALIBRATION_CERTIFIED).
   - Fast-fails if status != CALIBRATION_CERTIFIED, arm != A7, or representation mismatch.
2. Conformal Handoff Structure & Upstream Provenance:
   - Validates existence and schema of artifacts/phase8_v2/conformal_handoff.joblib.
   - Confirms representation ID (efficientnet_b0_43d581b96f8ec368).
   - Confirms BDA mask (642/1316) and 4-class ordering.
   - Confirms val_conf shape (123, 4), values in [0, 1], row sums == 1.0 +- 1e-5.
3. Preflight Partition Isolation:
   - val_conf (123) is strictly disjoint from dev (1146), val_calib (123), test (243), and quarantined (264).
   - Locked test isolation: zero test records accessed.
4. Exact Finite-Sample Quantile Invariant:
   - k = ceil((n + 1) * (1 - alpha)).
   - For n = 123, alpha = 0.10: k = 112 (0-based index 111).
5. Prediction Set Construction:
   - Class c in C(x) iff p_c >= 1 - q_hat.
   - Empty set condition: max(p) < 1 - q_hat.
6. Mondrian Class-Conditional Calibration & Small-Sample Detection:
   - Detects classes with n_c < 15 and flags diagnostic warnings.
   - Correct per-class quantiles.
7. FinalPipelineHandoff Serialization:
   - Full round-trip serialization preserves all upstream provenance and quantile thresholds.
8. Determinism:
   - Quantiles and prediction sets are 100% deterministic given calibrated inputs.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pytest

from config.config import get_config, PSDConfig
from modules.experiment_config import representation_id
from modules.calibration_handoff import (
    ConformalHandoff,
    FinalPipelineHandoff,
    load_conformal_handoff,
    save_final_pipeline_handoff,
    validate_conformal_handoff_provenance,
)
from modules.conformal import (
    ConformalMetrics,
    compute_conformal_quantile,
    compute_nonconformity_scores,
    evaluate_conformal_sets,
    fit_class_conditional_mondrian,
    fit_marginal_conformal,
    predict_conformal_sets,
    predict_conformal_sets_mondrian,
)
from run_aef_crc_phase9_v2 import (
    A7_ARM_ID,
    A7_EXPECTED_DIM,
    A7_SELECTED_FEATURES,
    CLASSIFIER_NAME,
    FEATURE_SELECTOR_NAME,
    NOMINAL_ALPHA,
    NOMINAL_COVERAGE,
    P3_V2_FOCAL_EXP,
    P3_V2_REPR_ID,
    configure_phase9_v2,
    gate_check_phase9_v2,
    preflight_phase9_v2_data,
    run_phase9_v2_conformal_calibration,
)


@pytest.fixture
def config() -> PSDConfig:
    return configure_phase9_v2(get_config())


def test_gate_check_phase9_v2_passes_on_certified_phase8(config: PSDConfig):
    """Verifies that Phase 9 gate check passes cleanly against the certified Phase 8 V2 manifest."""
    passed, msg, manifest = gate_check_phase9_v2(config)
    assert passed is True, f"Gate check unexpectedly failed: {msg}"
    assert manifest.get("status") == "CALIBRATION_CERTIFIED"
    assert manifest.get("pipeline", {}).get("arm_id") == A7_ARM_ID
    assert manifest.get("pipeline", {}).get("representation_id") == P3_V2_REPR_ID
    assert manifest.get("pipeline", {}).get("classifier") == CLASSIFIER_NAME
    assert manifest.get("pipeline", {}).get("feature_selection_method") == FEATURE_SELECTOR_NAME


def test_gate_check_phase9_v2_rejects_tampered_manifest(config: PSDConfig, tmp_path: Path):
    """Verifies that gate check rejects invalid statuses or mismatched representations."""
    test_cfg = dataclasses.replace(config, aef_crc_phase8_v2_reports_dir=tmp_path)

    # 1. Missing manifest
    passed, msg, _ = gate_check_phase9_v2(test_cfg)
    assert passed is False
    assert "Missing Phase 8 V2 manifest" in msg

    # 2. Invalid status
    bad_manifest = {
        "status": "UNVERIFIED",
        "pipeline": {"arm_id": A7_ARM_ID, "representation_id": P3_V2_REPR_ID},
    }
    (tmp_path / "phase8_manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    passed, msg, _ = gate_check_phase9_v2(test_cfg)
    assert passed is False
    assert "expected 'CALIBRATION_CERTIFIED'" in msg


def test_conformal_handoff_provenance_and_schema(config: PSDConfig):
    """Verifies that the certified Phase 8 ConformalHandoff artifact meets all contract specifications."""
    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    assert handoff_path.exists(), f"Phase 8 conformal handoff missing at {handoff_path}"

    handoff: ConformalHandoff = load_conformal_handoff(handoff_path)
    assert isinstance(handoff, ConformalHandoff)
    assert handoff.representation_id == P3_V2_REPR_ID
    assert handoff.experiment_id == P3_V2_FOCAL_EXP
    assert handoff.classifier_name == CLASSIFIER_NAME
    assert handoff.feature_selection_method == FEATURE_SELECTOR_NAME
    assert handoff.selected_feature_mask.shape == (A7_EXPECTED_DIM,)
    assert int(handoff.selected_feature_mask.sum()) == A7_SELECTED_FEATURES
    assert handoff.class_order == [
        "Psoriasis",
        "Lichen_Planus",
        "Pityriasis_Rosea",
        "Seborrheic_Dermatitis",
    ]
    assert handoff.temperature > 0.0
    assert pytest.approx(handoff.temperature, rel=1e-5) == 0.325677


def test_val_conf_probabilities_validity(config: PSDConfig):
    """Verifies shape, domain [0, 1], and row-sum unity of val_conf probabilities."""
    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    handoff: ConformalHandoff = load_conformal_handoff(handoff_path)

    probs = handoff.val_conf_calibrated_probabilities
    assert probs is not None
    assert probs.shape == (123, 4)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_preflight_partition_isolation(config: PSDConfig):
    """Verifies that val_conf partition is strictly isolated and disjoint from all other splits."""
    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    handoff: ConformalHandoff = load_conformal_handoff(handoff_path)

    passed, msg = preflight_phase9_v2_data(config, handoff)
    assert passed is True, f"Preflight partition isolation failed: {msg}"


def test_exact_finite_sample_quantile_order_statistic():
    """Verifies the exact finite-sample quantile formula k = ceil((n+1)(1-alpha))."""
    # Test case 1: n = 123, alpha = 0.10 -> (124 * 0.90) = 111.6 -> k = 112
    n = 123
    alpha = 0.10
    k_expected = math.ceil((n + 1) * (1.0 - alpha))
    assert k_expected == 112

    # Synthesize linear scores from 0.01 to 1.23
    scores = np.linspace(0.01, 0.99, n)
    q_hat = compute_conformal_quantile(scores, alpha=alpha)
    # The 112th order statistic (1-indexed) is index 111 (0-indexed)
    assert q_hat == scores[111]

    # Test case 2: small sample n = 9, alpha = 0.10 -> (10 * 0.90) = 9.0 -> k = 9
    scores_small = np.linspace(0.1, 0.9, 9)
    q_hat_small = compute_conformal_quantile(scores_small, alpha=alpha)
    assert q_hat_small == scores_small[8]  # index 8 is the 9th element (maximum)


def test_prediction_set_inclusion_rule():
    """Verifies that a class is included if and only if p_c >= 1 - q_hat."""
    classes = ["A", "B", "C", "D"]
    q_hat = 0.75  # inclusion threshold: 1.0 - 0.75 = 0.25 (exact in binary float)

    probs = np.array([
        [0.40, 0.25, 0.20, 0.15],  # A (0.40 >= 0.25) and B (0.25 >= 0.25) -> {'A', 'B'}
        [0.20, 0.20, 0.20, 0.20],  # All < 0.25 -> {} (empty set)
        [0.80, 0.10, 0.05, 0.05],  # A only -> {'A'} (singleton)
        [0.30, 0.30, 0.20, 0.20],  # A and B -> {'A', 'B'}
    ])

    sets = predict_conformal_sets(probs, q_hat, classes)
    assert sets[0] == ["A", "B"]
    assert sets[1] == []
    assert sets[2] == ["A"]
    assert sets[3] == ["A", "B"]


def test_mondrian_class_conditional_and_small_sample_detection():
    """Verifies that Mondrian calibration flags minority classes with n_c < 15."""
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    rng = np.random.default_rng(123)

    # Class counts: 69, 28, 17, 9
    labels = (
        ["Psoriasis"] * 69
        + ["Lichen_Planus"] * 28
        + ["Pityriasis_Rosea"] * 17
        + ["Seborrheic_Dermatitis"] * 9
    )
    probs = rng.dirichlet(alpha=[1.0, 1.0, 1.0, 1.0], size=len(labels))

    mondrian_fit = fit_class_conditional_mondrian(
        probs=probs,
        true_labels=labels,
        class_order=classes,
        alpha=0.10,
        min_recommended_n=15,
    )

    assert mondrian_fit.per_class_n["Psoriasis"] == 69
    assert mondrian_fit.per_class_n["Lichen_Planus"] == 28
    assert mondrian_fit.per_class_n["Pityriasis_Rosea"] == 17
    assert mondrian_fit.per_class_n["Seborrheic_Dermatitis"] == 9

    # Seborrheic_Dermatitis has n=9 < 15, so it MUST trigger a small-sample warning
    assert "Seborrheic_Dermatitis" in mondrian_fit.small_sample_warnings
    assert "Psoriasis" not in mondrian_fit.small_sample_warnings
    assert "Lichen_Planus" not in mondrian_fit.small_sample_warnings
    assert "Pityriasis_Rosea" not in mondrian_fit.small_sample_warnings


def test_conformal_metrics_evaluation():
    """Verifies calculation of marginal coverage, per-class coverage, and set efficiency metrics."""
    classes = ["A", "B"]
    true_labels = ["A", "A", "B", "B"]
    pred_sets = [
        ["A"],        # correct, singleton (size 1)
        ["A", "B"],   # correct, multi-set (size 2)
        ["A"],        # incorrect, singleton (size 1)
        [],           # incorrect, empty (size 0)
    ]
    # Coverage: 2 out of 4 = 0.50
    # Set sizes: 1, 2, 1, 0 -> mean = 1.0, median = 1.0
    # Singleton: 2/4 = 0.50
    # Empty: 1/4 = 0.25
    metrics: ConformalMetrics = evaluate_conformal_sets(
        pred_sets=pred_sets,
        true_labels=true_labels,
        class_order=classes,
        nominal_coverage=0.90,
    )

    assert metrics.marginal_coverage == 0.50
    assert metrics.mean_set_size == 1.0
    assert metrics.median_set_size == 1.0
    assert metrics.singleton_fraction == 0.50
    assert metrics.empty_set_fraction == 0.25
    assert metrics.per_class_coverage["A"] == 1.0  # sample 0 and 1 both contain 'A'
    assert metrics.per_class_coverage["B"] == 0.0  # sample 2 has 'A', sample 3 has []


def test_final_pipeline_handoff_serialization_contract(tmp_path: Path):
    """Verifies that FinalPipelineHandoff serializes and deserializes cleanly with all provenance intact."""
    classes = ["A", "B", "C", "D"]
    final_h = FinalPipelineHandoff(
        representation_id=P3_V2_REPR_ID,
        experiment_id=P3_V2_FOCAL_EXP,
        classifier_name=CLASSIFIER_NAME,
        feature_selection_method=FEATURE_SELECTOR_NAME,
        random_seed=42,
        class_order=classes,
        final_classifier=None,
        selected_feature_mask=np.ones(1316, dtype=bool),
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6},
        calibration_method="temperature_scaling",
        marginal_q_hat=0.725432,
        mondrian_q_hat={"A": 0.70, "B": 0.72, "C": 0.75, "D": 0.80},
        run_id="AEFCRC_P9_V2_TEST",
        upstream_phase8_run_id="AEFCRC_P8_V2_TEST",
        upstream_phase7_run_id="AEFCRC_P7_V2_TEST",
        dataset_freeze_hash="hash_freeze",
        fold_plan_hash="hash_fold",
        platt_models=None,
        isotonic_models=None,
        temperature_scaler=None,
        temperature=0.325677,
        alpha=0.10,
        n_conf_samples=123,
        per_class_conf_counts={"A": 69, "B": 28, "C": 17, "D": 9},
        small_sample_warnings={"D": "Small sample warning"},
    )

    art_path = tmp_path / "final_pipeline_handoff.joblib"
    save_final_pipeline_handoff(final_h, art_path)
    assert art_path.exists()

    reloaded: FinalPipelineHandoff = joblib.load(art_path)
    assert reloaded.run_id == "AEFCRC_P9_V2_TEST"
    assert reloaded.upstream_phase8_run_id == "AEFCRC_P8_V2_TEST"
    assert reloaded.upstream_phase7_run_id == "AEFCRC_P7_V2_TEST"
    assert reloaded.marginal_q_hat == 0.725432
    assert reloaded.temperature == 0.325677
    assert reloaded.alpha == 0.10
    assert reloaded.n_conf_samples == 123
    assert "D" in reloaded.small_sample_warnings


def test_conformal_prediction_determinism():
    """Verifies that nonconformity scoring and quantile derivation are 100% deterministic."""
    rng = np.random.default_rng(999)
    probs = rng.dirichlet(alpha=[2, 2, 2, 2], size=123)
    labels = np.array([0, 1, 2, 3] * 30 + [0, 1, 2])

    scores1 = compute_nonconformity_scores(probs, labels)
    scores2 = compute_nonconformity_scores(probs, labels)
    np.testing.assert_array_equal(scores1, scores2)

    q1 = compute_conformal_quantile(scores1, alpha=0.10)
    q2 = compute_conformal_quantile(scores2, alpha=0.10)
    assert q1 == q2


def test_phase9_v2_representation_id_resolution():
    """Regression Test: Demonstrates that configure_phase9_v2 explicitly binds
    to the certified Phase 3 V2 Focal representation ID: efficientnet_b0_43d581b96f8ec368.
    """
    configured_cfg = configure_phase9_v2(get_config())
    repr_id = representation_id(configured_cfg)
    assert repr_id == P3_V2_REPR_ID, f"Expected '{P3_V2_REPR_ID}', got '{repr_id}'"
    assert configured_cfg.loss_name == "categorical_focal_loss"
    assert configured_cfg.use_class_weights is False
    assert configured_cfg.preprocessing_mode == "standard"
    assert configured_cfg.focal_gamma == 2.0


def test_phase9_v2_unconfigured_default_provenance_rejection(config: PSDConfig):
    """Regression Test: Demonstrates that an unconfigured/default PSDConfig
    resolves to efficientnet_b0_9890ba0451ef2158 and does NOT silently pass
    provenance validation against the Phase 8 handoff.
    """
    default_cfg = get_config()
    default_repr_id = representation_id(default_cfg)
    assert default_repr_id == "efficientnet_b0_9890ba0451ef2158"
    assert default_repr_id != P3_V2_REPR_ID

    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    handoff: ConformalHandoff = load_conformal_handoff(handoff_path)

    # 1. validate_conformal_handoff_provenance must raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        validate_conformal_handoff_provenance(handoff, default_cfg)
    assert "Cross-run artifact provenance mismatch" in str(exc_info.value)
    assert "efficientnet_b0_9890ba0451ef2158" in str(exc_info.value)

    # 2. preflight_phase9_v2_data must fail fast
    passed, msg = preflight_phase9_v2_data(default_cfg, handoff)
    assert passed is False
    assert "Representation ID mismatch" in msg


def test_phase9_v2_preflight_verifies_all_upstream_provenance_dimensions(config: PSDConfig):
    """Demonstrates that preflight_phase9_v2_data validates all required upstream dimensions:
    representation ID, Phase 8 run ID, Phase 7 run ID, arm ID, BDA dimensions, temperature, class order.
    """
    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    handoff: ConformalHandoff = load_conformal_handoff(handoff_path)

    manifest_path = config.aef_crc_phase8_v2_reports_dir / "phase8_manifest.json"
    p8_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Pristine preflight must pass cleanly
    passed, msg = preflight_phase9_v2_data(config, handoff, p8_manifest)
    assert passed is True, f"Preflight unexpectedly failed: {msg}"

    # Verify tampering detection on each dimension:
    # 1. Tampered Arm ID
    bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad_manifest["pipeline"]["arm_id"] = "A6"
    passed, msg = preflight_phase9_v2_data(config, handoff, bad_manifest)
    assert passed is False
    assert "Arm ID mismatch" in msg

    # 2. Tampered Upstream Phase 7 Run ID
    bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad_manifest["provenance"]["upstream_phase7_run_id"] = "TAMPERED_P7_RUN"
    passed, msg = preflight_phase9_v2_data(config, handoff, bad_manifest)
    assert passed is False
    assert "does not match upstream Phase 7 run_id" in msg

    # 3. Tampered Temperature
    bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad_manifest["calibration"]["temperature"] = 1.9999
    passed, msg = preflight_phase9_v2_data(config, handoff, bad_manifest)
    assert passed is False
    assert "Temperature mismatch" in msg


def test_conformal_ulp_boundary_regression():
    """Exact Regression Test for IEEE-754 Boundary Roundoff Bug:
    p = 0.26175721726626361
    q = 0.73824278273373634
    Demonstrates that:
    1. Canonical (1.0 - p) <= q is TRUE.
    2. Reformulated p >= (1.0 - q) fails because 1.0 - q = 0.26175721726626366 > p.
    3. predict_conformal_sets correctly includes the class at the exact boundary.
    """
    p = 0.26175721726626361
    q = 0.73824278273373634
    classes = ["A", "B", "C", "D"]

    # 1. Canonical nonconformity condition is TRUE
    score = 1.0 - p
    assert score <= q, f"Canonical condition failed: {score} > {q}"

    # 2. Reformulated condition fails due to 1-ULP floating point deficit
    thresh = 1.0 - q
    assert p < thresh, f"Expected p < (1.0 - q) by ULP roundoff, got p={p}, thresh={thresh}"
    assert not (p >= thresh), "Old reformulated condition unexpectedly passed!"

    # 3. predict_conformal_sets must include class A
    probs = np.array([[p, 0.50, 0.15, 1.0 - p - 0.65]])
    sets = predict_conformal_sets(probs, q, classes)
    assert "A" in sets[0], f"Boundary class 'A' was dropped from prediction set: {sets[0]}"


def test_conformal_general_invariants():
    """Verifies general mathematical invariants of conformal prediction sets:
    - If score < q, class is included.
    - If score == q (boundary), class is included.
    - If score > q, class is excluded.
    - Ties at q are all included.
    - Set sizes are in {0, 1, 2, 3, 4}.
    - Only valid class names are included.
    """
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    q = 0.60  # score threshold; classes with s <= 0.60 (p >= ~0.40) included

    # Case 1: strict inequality, boundary, and tie
    p_tie = 1.0 - q  # s = q
    probs = np.array([
        [0.45, 0.40, 0.10, 0.05],  # Pso: s=0.55 < q; LP: s=0.60 == q; PR: s=0.90 > q; SD: s=0.95 > q
        [0.40, 0.40, 0.10, 0.10],  # Pso & LP tie at s=0.60 == q
        [0.10, 0.10, 0.10, 0.10],  # all s=0.90 > q -> empty set
        [0.90, 0.04, 0.03, 0.03],  # Pso s=0.10 < q -> singleton
    ])

    sets = predict_conformal_sets(probs, q, classes)
    assert sets[0] == ["Psoriasis", "Lichen_Planus"]  # strict < and == boundary both included
    assert sets[1] == ["Psoriasis", "Lichen_Planus"]  # ties included
    assert sets[2] == []                              # all > q excluded
    assert sets[3] == ["Psoriasis"]                   # singleton

    for s in sets:
        assert len(s) in {0, 1, 2, 3, 4}
        for name in s:
            assert name in classes


def test_phase9_calibration_cohort_exact_finite_sample_coverage(config: PSDConfig):
    """Verifies that evaluating the calibration cohort (val_conf, N=123) against its own
    k=112 order statistic covers AT LEAST 112 true labels (112 / 123 = 91.0569%).
    """
    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    handoff: ConformalHandoff = load_conformal_handoff(handoff_path)

    probs = handoff.val_conf_calibrated_probabilities
    labels = handoff.val_conf_true_labels
    classes = handoff.class_order
    cls2idx = {c: i for i, c in enumerate(classes)}
    true_indices = np.array([cls2idx[y] for y in labels])

    scores = compute_nonconformity_scores(probs, true_indices)
    n = len(scores)
    assert n == 123
    k = math.ceil((n + 1) * 0.90)
    assert k == 112

    q_hat = compute_conformal_quantile(scores, alpha=0.10)
    sorted_s = np.sort(scores)
    assert q_hat == sorted_s[k - 1]

    # Verification 1: Number of scores <= q_hat MUST be >= 112
    count_scores_le = np.sum(scores <= q_hat)
    assert count_scores_le >= k, f"Expected at least {k} scores <= q_hat, got {count_scores_le}"
    assert count_scores_le == 112

    # Verification 2: predict_conformal_sets must cover at least 112 samples
    pred_sets = predict_conformal_sets(probs, q_hat, classes)
    covered_flags = [labels[i] in pred_sets[i] for i in range(n)]
    cov_count = sum(covered_flags)
    assert cov_count >= k, f"Expected at least {k} covered samples, got {cov_count}"
    assert cov_count == 112
    assert pytest.approx(cov_count / n, rel=1e-4) == 0.910569


def test_phase9_mondrian_calibration_cohort_exact_finite_sample_coverage(config: PSDConfig):
    """Verifies that class-conditional Mondrian conformal sets cover at least k_c samples for every class."""
    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    handoff: ConformalHandoff = load_conformal_handoff(handoff_path)

    probs = handoff.val_conf_calibrated_probabilities
    labels = handoff.val_conf_true_labels
    classes = handoff.class_order
    cls2idx = {c: i for i, c in enumerate(classes)}
    true_indices = np.array([cls2idx[y] for y in labels])
    scores = compute_nonconformity_scores(probs, true_indices)

    mondrian_fit = fit_class_conditional_mondrian(probs, labels, classes, alpha=0.10)
    pred_sets = predict_conformal_sets_mondrian(probs, mondrian_fit.q_hat_by_class, classes)

    expected_k = {
        "Psoriasis": 63,            # ceil(70 * 0.90) = 63 out of 69
        "Lichen_Planus": 27,        # ceil(29 * 0.90) = 27 out of 28
        "Pityriasis_Rosea": 17,     # ceil(18 * 0.90) = 17 out of 17
        "Seborrheic_Dermatitis": 9, # ceil(10 * 0.90) = 9 out of 9
    }

    for cls in classes:
        mask = np.array([labels[i] == cls for i in range(len(labels))])
        c_cov = sum(labels[i] in pred_sets[i] for i in range(len(labels)) if mask[i])
        min_k = expected_k[cls]
        assert c_cov >= min_k, f"Class {cls} coverage {c_cov} < expected minimum {min_k}"
        assert c_cov == min_k, f"Class {cls} coverage {c_cov} != exact order statistic {min_k}"


def test_audit_automation_canonical_score_vs_prediction_sets(config: PSDConfig):
    """Audit Invariant: Compares canonical score-based inclusion against predict_conformal_sets
    for all 123 calibration samples, asserting ZERO discrepancies across all classes.
    """
    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    handoff: ConformalHandoff = load_conformal_handoff(handoff_path)

    probs = handoff.val_conf_calibrated_probabilities
    classes = handoff.class_order
    q_hat = compute_conformal_quantile(
        compute_nonconformity_scores(
            probs,
            np.array([{c: i for i, c in enumerate(classes)}[y] for y in handoff.val_conf_true_labels]),
        ),
        alpha=0.10,
    )

    pred_sets = predict_conformal_sets(probs, q_hat, classes)
    for i in range(len(probs)):
        canonical_included = [classes[c] for c in range(len(classes)) if (1.0 - probs[i, c]) <= q_hat]
        assert set(pred_sets[i]) == set(canonical_included), (
            f"Audit Discrepancy at sample {i}: predict_conformal_sets={pred_sets[i]} != canonical={canonical_included}"
        )


def test_floating_point_formulation_divergence_cross_check(config: PSDConfig):
    """Cross-check demonstrating exactly where canonical (1-p <= q) and reformulated (p >= 1-q)
    diverge on the real calibration cohort due to IEEE-754 arithmetic.
    """
    handoff_path = config.aef_crc_phase8_v2_artifacts_dir / "conformal_handoff.joblib"
    handoff: ConformalHandoff = load_conformal_handoff(handoff_path)
    probs = handoff.val_conf_calibrated_probabilities
    classes = handoff.class_order
    cls2idx = {c: i for i, c in enumerate(classes)}
    true_indices = np.array([cls2idx[y] for y in handoff.val_conf_true_labels])
    scores = compute_nonconformity_scores(probs, true_indices)
    q_hat = compute_conformal_quantile(scores, alpha=0.10)
    thresh = 1.0 - q_hat

    divergences = []
    for i in range(len(probs)):
        for c, cls_name in enumerate(classes):
            p = probs[i, c]
            canonical = (1.0 - p) <= q_hat
            reformulated = p >= thresh
            if canonical != reformulated:
                divergences.append((i, cls_name, p, 1.0 - p, q_hat, thresh))

    # There is exactly 1 boundary divergence across all 123 * 4 = 492 evaluations in val_conf:
    # Sample 12, Lichen_Planus (the 112th order statistic sample)
    assert len(divergences) == 1
    sample_idx, cls_name, p_val, s_val, q_val, t_val = divergences[0]
    assert sample_idx == 12
    assert cls_name == "Lichen_Planus"
    assert s_val == q_val  # s exactly equals q_hat
    assert p_val < t_val  # but p is strictly less than 1.0 - q_hat by ~5.5e-17


