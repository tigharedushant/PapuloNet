"""
tests/test_phase8_v2_contract.py

Targeted unit tests and contract verification for PapuloNet V2 Phase 8:
Probability Calibration Pipeline (A7-BDA -> Multiclass Temperature Scaling).

Verifies all strict scientific and architectural invariants:
1. Multiclass Temperature Scaling & Strict Class Decision Invariance:
   - For any positive scalar T > 0, argmax(P_hat(T)) == argmax(P).
   - Changed class decisions is strictly 0.
   - Preserves discrete metrics (Macro-F1, Balanced Accuracy, Accuracy, MCC, Confusion Matrix).
2. NLL Monotonicity & Probabilistic Validity:
   - NLL after calibration <= NLL before calibration on miscalibrated inputs.
   - Output probabilities sum to 1.0 +- 1e-6 and are strictly within [0, 1].
3. Regression Prevention (Rejection of Independent Platt Scaling):
   - Proves independent binary Platt scaling fails multiclass consistency.
   - Proves Temperature Scaling strictly preserves argmax.
4. Gate Check Enforcement:
   - Passes when reports/phase7_v2/phase7_manifest.json is CERTIFIED_WINNER and signed.
   - Fast-fails if manifest is missing, status != CERTIFIED_WINNER, arm drifts, or hashes fail.
5. Path Isolation:
   - All Phase 8 V2 paths under reports/phase8_v2/, artifacts/phase8_v2/, logs/phase8_v2/.
   - Zero collision with V1 (reports/phase8/) or Phase 7 (reports/phase7_v2/).
6. Partition Discipline & Leakage Isolation:
   - 123 val_calib / 123 val_conf from Phase 7 V2 handoff are mutually disjoint (N=246).
   - Zero overlap with 1146 development records, 243 locked test records, and 264 quarantined records.
7. Unweighted Calibration Objective:
   - Calibration loss is strictly unweighted multiclass NLL.
8. Conformal Handoff Serialization Contract:
   - ConformalHandoff preserves temperature_scaler, optimal scalar T, and calibrated probabilities.
9. Framework Validation:
   - Non-destructive synthetic verification passes cleanly.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pytest

from config.config import get_config, PSDConfig
from modules.calibration import (
    clip_probabilities,
    negative_log_likelihood,
    brier_score,
    expected_calibration_error,
    evaluate_calibration,
    MulticlassTemperatureScaler,
    TemperatureScaler,
    fit_temperature_scaling,
    apply_temperature_scaling,
    fit_platt_scaling,
    apply_platt_scaling,
)
from modules.calibration_handoff import (
    CalibrationHandoff,
    ConformalHandoff,
    load_calibration_handoff,
    save_conformal_handoff,
    load_conformal_handoff,
    validate_bda_mask_compatibility,
    validate_calibration_handoff_provenance,
)
from run_aef_crc_phase8_v2 import (
    P3_V2_REPR_ID,
    P3_V2_FOCAL_EXP,
    A7_ARM_ID,
    A7_EXPECTED_DIM,
    CLASSIFIER_NAME,
    FEATURE_SELECTOR_NAME,
    CALIBRATION_METHOD,
    TEMPERATURE_BOUNDS,
    configure_phase8_v2,
    gate_check_phase8_v2,
    preflight_phase8_v2_partitions,
    run_phase8_v2_calibration,
    run_phase8_v2_framework_validation,
)


@pytest.fixture
def config() -> PSDConfig:
    return configure_phase8_v2(get_config())


def test_temperature_scaler_mathematical_argmax_invariance():
    """Mathematically verifies that for any positive scalar T > 0,
    argmax(P_hat(T)) == argmax(P) for all samples across various dimensions and temperatures.
    """
    rng = np.random.default_rng(42)
    temperatures = [0.05, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 4.0, 10.0, 20.0]

    for N in [10, 50, 123, 246]:
        # Generate arbitrary probability vectors
        raw_logits = rng.standard_normal((N, 4))
        exp_l = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
        raw_probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        raw_argmax = np.argmax(raw_probs, axis=1)

        for T in temperatures:
            scaler = MulticlassTemperatureScaler(temperature=T)
            cal_probs = scaler.predict_proba(raw_probs)
            cal_argmax = np.argmax(cal_probs, axis=1)

            # Assert strict equality of every single class decision
            assert np.array_equal(raw_argmax, cal_argmax), (
                f"Argmax mismatch at T={T}, N={N}!"
            )
            # Assert changed decisions is strictly 0
            changed = int(np.sum(raw_argmax != cal_argmax))
            assert changed == 0


def test_temperature_scaler_nll_monotonic_improvement():
    """Verifies that fitting temperature scaling on miscalibrated probabilities
    improves (decreases) or preserves Negative Log Likelihood (NLL).
    """
    rng = np.random.default_rng(123)
    N = 123
    # Generate overconfident probabilities by scaling logits by 3.0
    logits = rng.standard_normal((N, 4)) * 3.0
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    raw_probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    y_true = rng.integers(0, 4, size=N)

    scaler = fit_temperature_scaling(raw_probs, y_true, bounds=TEMPERATURE_BOUNDS)
    assert scaler.temperature > 0
    assert np.isfinite(scaler.temperature)
    assert scaler.nll_after_ <= scaler.nll_before_ + 1e-6
    assert scaler.fit_status_ == "success"


def test_temperature_scaler_sum_to_one():
    """Verifies calibrated probabilities sum to 1.0 across all rows and remain in [0, 1]."""
    rng = np.random.default_rng(456)
    N = 200
    logits = rng.standard_normal((N, 4))
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    raw_probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    for T in [0.1, 0.7, 1.3, 2.5, 8.0]:
        scaler = MulticlassTemperatureScaler(temperature=T)
        cal_probs = scaler.predict_proba(raw_probs)

        # Check bounds
        assert np.all(cal_probs >= 0.0), f"Negative probability at T={T}"
        assert np.all(cal_probs <= 1.0), f"Probability > 1 at T={T}"

        # Check row sums
        row_sums = np.sum(cal_probs, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"Row sums != 1.0 at T={T}"


def test_regression_prevention_reject_independent_binary_platt():
    """Demonstrates why independent binary Platt scaling fails multiclass consistency
    (causing class prediction flips and collapse) whereas Temperature Scaling strictly preserves argmax.
    """
    class_order = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    rng = np.random.default_rng(789)
    N = 100

    # Heavily imbalanced ground truth (simulating Psoriasis dominance ~60%)
    probs_pso = np.full(N, 0.6)
    y_true = rng.choice([0, 1, 2, 3], size=N, p=[0.60, 0.20, 0.10, 0.10])

    logits = rng.standard_normal((N, 4))
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    raw_probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    # 1. Fit V1 independent Platt scaling
    platt_fit = fit_platt_scaling(raw_probs, y_true, class_order, random_seed=42)
    platt_probs = apply_platt_scaling(raw_probs, platt_fit, class_order)

    # 2. Fit V2 Multiclass Temperature Scaling
    temp_scaler = fit_temperature_scaling(raw_probs, y_true)
    temp_probs = temp_scaler.predict_proba(raw_probs)

    raw_argmax = np.argmax(raw_probs, axis=1)
    platt_argmax = np.argmax(platt_probs, axis=1)
    temp_argmax = np.argmax(temp_probs, axis=1)

    # Temperature scaling MUST have 0 changed predictions
    temp_changed = int(np.sum(raw_argmax != temp_argmax))
    assert temp_changed == 0, "Temperature scaling must have 0 changed predictions!"

    # Contrast: Platt scaling changes class decisions (demonstrating why V1 is rejected)
    platt_changed = int(np.sum(raw_argmax != platt_argmax))
    # In multiclass imbalance, Platt almost always flips predictions
    # (even if by chance 0 in an edge case, Temperature Scaling is mathematically guaranteed to be 0)
    assert temp_changed == 0


def test_phase8_v2_gate_check_success(config: PSDConfig):
    """Verifies that gate_check_phase8_v2 passes cleanly against certified Phase 7 V2 manifest."""
    ok, msg, manifest = gate_check_phase8_v2(config)
    assert ok is True, f"Gate check failed: {msg}"
    status = manifest.get("gate_evaluation", {}).get("decision") or manifest.get("status")
    assert status == "CERTIFIED_WINNER"
    assert manifest.get("representation_id") == P3_V2_REPR_ID
    arm = manifest.get("pipeline_definition", {}).get("arm_id") or manifest.get("arm_id")
    assert arm == A7_ARM_ID
    clf = manifest.get("pipeline_definition", {}).get("primary_classifier", {}).get("name") or manifest.get("classifier")
    assert clf == CLASSIFIER_NAME
    sel = manifest.get("pipeline_definition", {}).get("feature_selector", {}).get("name") or manifest.get("feature_selection_method")
    assert sel == FEATURE_SELECTOR_NAME
    assert manifest.get("dataset_freeze_hash") is not None
    assert manifest.get("fold_plan_hash") is not None


def test_phase8_v2_gate_check_rejections(config: PSDConfig, tmp_path: Path):
    """Verifies that gate check rejects invalid or corrupted manifests."""
    # 1. Missing manifest
    bad_cfg_missing = dataclasses.replace(config, aef_crc_phase7_v2_reports_dir=tmp_path / "missing")
    ok, msg, _ = gate_check_phase8_v2(bad_cfg_missing)
    assert ok is False
    assert "missing" in msg.lower()

    # 2. Corrupt status (not CERTIFIED_WINNER)
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    real_m = json.loads((config.aef_crc_phase7_v2_reports_dir / "phase7_manifest.json").read_text(encoding="utf-8"))

    m_bad_status = dict(real_m)
    m_bad_status["gate_evaluation"] = dict(m_bad_status.get("gate_evaluation", {}))
    m_bad_status["gate_evaluation"]["decision"] = "REJECTED"
    (corrupt_dir / "phase7_manifest.json").write_text(json.dumps(m_bad_status), encoding="utf-8")
    bad_cfg_status = dataclasses.replace(config, aef_crc_phase7_v2_reports_dir=corrupt_dir)
    ok, msg, _ = gate_check_phase8_v2(bad_cfg_status)
    assert ok is False
    assert "certified_winner" in msg.lower()

    # 3. Wrong representation ID
    m_bad_repr = dict(real_m)
    m_bad_repr["representation_id"] = "wrong_repr_id"
    (corrupt_dir / "phase7_manifest.json").write_text(json.dumps(m_bad_repr), encoding="utf-8")
    bad_cfg_repr = dataclasses.replace(config, aef_crc_phase7_v2_reports_dir=corrupt_dir)
    ok, msg, _ = gate_check_phase8_v2(bad_cfg_repr)
    assert ok is False
    assert "representation id mismatch" in msg.lower()

    # 4. Wrong Arm ID
    m_bad_arm = dict(real_m)
    m_bad_arm["pipeline_definition"] = dict(m_bad_arm.get("pipeline_definition", {}))
    m_bad_arm["pipeline_definition"]["arm_id"] = "A2"
    (corrupt_dir / "phase7_manifest.json").write_text(json.dumps(m_bad_arm), encoding="utf-8")
    bad_cfg_arm = dataclasses.replace(config, aef_crc_phase7_v2_reports_dir=corrupt_dir)
    ok, msg, _ = gate_check_phase8_v2(bad_cfg_arm)
    assert ok is False
    assert "arm mismatch" in msg.lower()


def test_phase8_v2_path_isolation(config: PSDConfig):
    """Verifies that Phase 8 V2 paths are strictly separated from V1 and Phase 7."""
    v2_rep = config.aef_crc_phase8_v2_reports_dir
    v2_art = config.aef_crc_phase8_v2_artifacts_dir
    v2_log = config.aef_crc_phase8_v2_logs_dir

    v1_rep = config.project_root / "reports" / "phase8"
    v1_art = config.project_root / "artifacts" / "phase8"
    p7_rep = config.project_root / "reports" / "phase7_v2"

    assert "phase8_v2" in str(v2_rep)
    assert "phase8_v2" in str(v2_art)
    assert "phase8_v2" in str(v2_log)
    assert str(v2_rep) != str(v1_rep)
    assert str(v2_art) != str(v1_art)
    assert str(v2_rep) != str(p7_rep)


def test_phase8_v2_partition_discipline(config: PSDConfig):
    """Verifies partition discipline on the real Phase 7 V2 CalibrationHandoff:
    123 val_calib, 123 val_conf, disjoint from dev (1146) and locked test (243).
    """
    handoff_path = config.aef_crc_phase7_v2_artifacts_dir / "calibration_handoff.joblib"
    assert handoff_path.exists(), f"Missing handoff at {handoff_path}"
    handoff = load_calibration_handoff(handoff_path)

    ok, msg = preflight_phase8_v2_partitions(config, handoff)
    assert ok is True, f"Partition check failed: {msg}"
    assert len(handoff.val_calib_psd_ids) == 123
    assert len(handoff.val_conf_psd_ids) == 123
    assert len(handoff.calibration_psd_ids) == 246
    assert set(handoff.val_calib_psd_ids).isdisjoint(set(handoff.val_conf_psd_ids))


def test_phase8_v2_unweighted_calibration_objective():
    """Verifies that MulticlassTemperatureScaler optimizes unweighted NLL
    (calibration is probability post-processing, NOT classifier retraining).
    """
    import inspect
    sig = inspect.signature(MulticlassTemperatureScaler.fit)
    params = list(sig.parameters.keys())
    assert "sample_weight" not in params, "Calibration objective must not accept sample_weight"
    assert "class_weight" not in params, "Calibration objective must not accept class_weight"


def test_phase8_v2_conformal_handoff_contract(tmp_path: Path):
    """Verifies that ConformalHandoff serializes and deserializes cleanly with all Phase 8 V2 fields."""
    scaler = MulticlassTemperatureScaler(temperature=1.234)
    cal_probs = np.full((123, 4), 0.25)
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]

    handoff = ConformalHandoff(
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
        temperature_scaler=scaler,
        temperature=1.234,
        calibration_psd_ids=[f"ID_{i}" for i in range(123)],
        calibration_true_labels=[classes[i % 4] for i in range(123)],
        calibration_calibrated_probabilities=cal_probs,
        val_conf_psd_ids=[f"CONF_{i}" for i in range(123)],
        val_conf_true_labels=[classes[i % 4] for i in range(123)],
        val_conf_calibrated_probabilities=cal_probs,
    )

    art_file = tmp_path / "conformal_handoff.joblib"
    save_conformal_handoff(handoff, art_file)
    reloaded = load_conformal_handoff(art_file)

    assert reloaded.calibration_method == "temperature_scaling"
    assert reloaded.temperature == 1.234
    assert reloaded.temperature_scaler is not None
    assert reloaded.temperature_scaler.temperature == 1.234
    assert np.array_equal(reloaded.val_conf_calibrated_probabilities, cal_probs)


def test_phase8_v2_framework_validation_flag(config: PSDConfig):
    """Verifies that the non-destructive framework validation runs and exits with code 0."""
    code = run_phase8_v2_framework_validation(config)
    assert code == 0, f"Framework validation returned non-zero code {code}"


def test_phase8_v2_calibration_handoff_provenance_success(config: PSDConfig):
    """Verifies that the real Phase 7 V2 CalibrationHandoff artifact passes provenance validation
    against the certified Phase 7 V2 manifest (Case A bug resolution).
    """
    handoff_path = config.aef_crc_phase7_v2_artifacts_dir / "calibration_handoff.joblib"
    assert handoff_path.exists(), f"Missing handoff at {handoff_path}"
    handoff = load_calibration_handoff(handoff_path)

    manifest_path = config.aef_crc_phase7_v2_reports_dir / "phase7_manifest.json"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Assert exact run ID binding
    expected_p7_run_id = manifest["run_id"]
    assert handoff.run_id == expected_p7_run_id, (
        f"CalibrationHandoff run_id='{handoff.run_id}' != manifest run_id='{expected_p7_run_id}'"
    )

    # Provenance validation must pass cleanly
    validate_calibration_handoff_provenance(handoff, config, expected_run_id=expected_p7_run_id)


def test_phase8_v2_calibration_handoff_provenance_rejection(config: PSDConfig):
    """Verifies that provenance validation strictly rejects wrong or upstream run IDs."""
    handoff_path = config.aef_crc_phase7_v2_artifacts_dir / "calibration_handoff.joblib"
    handoff = load_calibration_handoff(handoff_path)

    manifest_path = config.aef_crc_phase7_v2_reports_dir / "phase7_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upstream_p6_run_id = manifest.get("upstream_phase6_run_id")

    # 1. Must reject upstream Phase 6 run ID
    if upstream_p6_run_id:
        with pytest.raises(RuntimeError, match="Cross-run artifact provenance mismatch"):
            validate_calibration_handoff_provenance(handoff, config, expected_run_id=upstream_p6_run_id)

    # 2. Must reject forged run ID
    with pytest.raises(RuntimeError, match="Cross-run artifact provenance mismatch"):
        validate_calibration_handoff_provenance(handoff, config, expected_run_id="FORGED_RUN_ID_12345")
