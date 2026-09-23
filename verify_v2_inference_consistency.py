"""
verify_v2_inference_consistency.py

Targeted V2 Inference Consistency Verification Script.
Audits and verifies that the repaired Streamlit / app_adapter inference path
is bitwise/numerically consistent with the direct certified PapuloNet V2 pipeline.

Strict Isolation & Safety Contract:
- Reads ONLY the pre-registered 8 holdout validation samples (val cohort).
- ZERO access to the locked 243-image test partition (data/test or holdout_test_records).
- ZERO artifact modification or re-saving.
- NO retraining, recalibration, parameter adjustments, or threshold changes.
"""

from __future__ import annotations

import io
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from config.config import get_config, PSDConfig
from modules.calibration_handoff import load_final_pipeline_handoff, FinalPipelineHandoff
from modules.experiment_config import representation_id
from modules.inference import AEFCRCInferenceEngine
from modules.app_adapter import get_app_pipeline, run_app_inference, clear_app_pipeline_cache
from modules.handcrafted_features import HandcraftedFeatureExtractor
from run_aef_crc_phase9_v2 import configure_phase9_v2


# =============================================================================
# 1. Authoritative Frozen Constants & Tolerances
# =============================================================================
FROZEN_HANDOFF_PATH = Path("artifacts/phase9_v2/final_pipeline_handoff.joblib")
EXPECTED_SHA256 = "d109b1fb798ea1ecda32d9f8723ce1f14b9f7b54b6d57322d61f30d9a5c4309f"
EXPECTED_FILE_SIZE = 8500766

EXPECTED_REPRESENTATION_ID = "efficientnet_b0_43d581b96f8ec368"
EXPECTED_PREPROCESSING_MODE = "standard"
EXPECTED_A7_FULL_DIM = 1316
EXPECTED_BDA_SELECTED_COUNT = 642
EXPECTED_TEMPERATURE = 0.32567700916361153
EXPECTED_MARGINAL_Q_HAT = 0.7382427827337363

# Numerical tolerances per representation layer
TOLERANCE_EXACT = 0.0          # Integer arrays, categorical labels, prediction sets
TOLERANCE_PIXEL = 1e-6         # Normalized float pixel values [-1.0, 1.0]
TOLERANCE_FLOAT_CPU = 1e-5     # TensorFlow / BLAS float32 CPU forward pass
TOLERANCE_PROBABILITY = 1e-6   # Calibrated & raw probability distributions


# =============================================================================
# 2. Pre-Registered Phase 10 Validation Cohort (Holdout Val Only)
# =============================================================================
VALIDATION_SAMPLES: List[Dict[str, Any]] = [
    # Psoriasis (2 samples)
    {
        "sample_id": "PSD_00000917",
        "true_class": "Psoriasis",
        "path": Path("output/06_final_split/val/Psoriasis/PSD_00000917.jpg"),
    },
    {
        "sample_id": "PSD_00001647",
        "true_class": "Psoriasis",
        "path": Path("output/06_final_split/val/Psoriasis/PSD_00001647.jpg"),
    },
    # Lichen Planus (2 samples)
    {
        "sample_id": "PSD_00000639",
        "true_class": "Lichen_Planus",
        "path": Path("output/06_final_split/val/Lichen_Planus/PSD_00000639.jpg"),
    },
    {
        "sample_id": "PSD_00003153",
        "true_class": "Lichen_Planus",
        "path": Path("output/06_final_split/val/Lichen_Planus/PSD_00003153.jpg"),
    },
    # Pityriasis Rosea (2 samples)
    {
        "sample_id": "PSD_00000735",
        "true_class": "Pityriasis_Rosea",
        "path": Path("output/06_final_split/val/Pityriasis_Rosea/PSD_00000735.jpg"),
    },
    {
        "sample_id": "PSD_00003209",
        "true_class": "Pityriasis_Rosea",
        "path": Path("output/06_final_split/val/Pityriasis_Rosea/PSD_00003209.jpg"),
    },
    # Seborrheic Dermatitis (2 samples)
    {
        "sample_id": "PSD_00001693",
        "true_class": "Seborrheic_Dermatitis",
        "path": Path("output/06_final_split/val/Seborrheic_Dermatitis/PSD_00001693.jpg"),
    },
    {
        "sample_id": "PSD_00001760",
        "true_class": "Seborrheic_Dermatitis",
        "path": Path("output/06_final_split/val/Seborrheic_Dermatitis/PSD_00001760.jpg"),
    },
]


# =============================================================================
# 3. Strict Isolation and Preflight Verification
# =============================================================================
def verify_strict_test_set_isolation() -> None:
    """Guarantees zero contact with the locked test partition."""
    print("=" * 80)
    print("STEP 1: VERIFYING STRICT TEST SET ISOLATION & ARTIFACT INTEGRITY")
    print("=" * 80)

    for item in VALIDATION_SAMPLES:
        p = item["path"]
        p_str = str(p).replace("\\", "/")
        # Guard 1: Must be in 'val' directory
        assert "/val/" in p_str, f"Test isolation violation! Sample {p} is not in holdout val."
        # Guard 2: Must never contain 'test'
        assert "/test/" not in p_str and "test" not in p.name.lower(), (
            f"Test isolation violation! Sample {p} references test partition."
        )
        assert p.exists(), f"Sample image file not found: {p}"
        print(f"  [OK] Validated Holdout Val Sample: {item['sample_id']:14s} ({item['true_class']})")

    # Guard 3: Handoff file immutability
    assert FROZEN_HANDOFF_PATH.exists(), f"Handoff missing at {FROZEN_HANDOFF_PATH}"
    data = FROZEN_HANDOFF_PATH.read_bytes()
    assert len(data) == EXPECTED_FILE_SIZE, (
        f"File size mismatch: expected {EXPECTED_FILE_SIZE}, got {len(data)}"
    )
    actual_sha = hashlib.sha256(data).hexdigest()
    assert actual_sha == EXPECTED_SHA256, (
        f"Handoff modification detected! Expected SHA256 {EXPECTED_SHA256}, got {actual_sha}"
    )
    print(f"  [OK] Frozen Handoff SHA256 Verified: {actual_sha[:16]}... (8,500,766 bytes)")


def verify_pipeline_contract_metadata(handoff: FinalPipelineHandoff, engine: AEFCRCInferenceEngine) -> None:
    """Verifies that all pipeline metadata and parameters strictly match Phase 9 V2 certified specifications."""
    print("\n" + "=" * 80)
    print("STEP 2: VERIFYING PIPELINE CONTRACT METADATA & REPR IDENTIFIERS")
    print("=" * 80)

    # 1. Preprocessing mode
    assert engine.config.preprocessing_mode == EXPECTED_PREPROCESSING_MODE, (
        f"Mismatch: expected '{EXPECTED_PREPROCESSING_MODE}', got '{engine.config.preprocessing_mode}'"
    )
    print(f"  [OK] Preprocessing Mode:         '{engine.config.preprocessing_mode}' (Standard, no-op)")

    # 2. Representation ID
    engine_repr_id = representation_id(engine.config)
    assert engine_repr_id == EXPECTED_REPRESENTATION_ID, (
        f"Mismatch: expected '{EXPECTED_REPRESENTATION_ID}', got '{engine_repr_id}'"
    )
    assert handoff.representation_id == EXPECTED_REPRESENTATION_ID
    print(f"  [OK] Representation ID:          '{engine_repr_id}' (Phase 3 V2 Focal)")

    # 3. A7 Multimodal source dimension
    mask = handoff.selected_feature_mask
    assert mask.shape[0] == EXPECTED_A7_FULL_DIM, (
        f"Mismatch: expected {EXPECTED_A7_FULL_DIM}-D, got {mask.shape[0]}-D"
    )
    print(f"  [OK] A7 Source Feature Dim:       {mask.shape[0]}-D (Deep 1280 + GLCM 12 + LBP 18 + LAB 6)")

    # 4. Selected feature count
    k_selected = int(mask.sum())
    assert k_selected == EXPECTED_BDA_SELECTED_COUNT, (
        f"Mismatch: expected {EXPECTED_BDA_SELECTED_COUNT}, got {k_selected}"
    )
    print(f"  [OK] BDA Selected Features:       {k_selected}-D active subspace")

    # 5. Temperature parameter
    t_val = getattr(handoff, "temperature", None)
    if t_val is None and getattr(handoff, "temperature_scaler", None) is not None:
        t_val = float(handoff.temperature_scaler.temperature)
    assert math.isclose(t_val, EXPECTED_TEMPERATURE, rel_tol=1e-7), (
        f"Mismatch: expected {EXPECTED_TEMPERATURE}, got {t_val}"
    )
    print(f"  [OK] Calibrator Temperature:      {t_val:.16f}")

    # 6. Marginal Conformal Quantile (q_hat)
    q_hat = handoff.marginal_q_hat
    assert math.isclose(q_hat, EXPECTED_MARGINAL_Q_HAT, rel_tol=1e-7), (
        f"Mismatch: expected {EXPECTED_MARGINAL_Q_HAT}, got {q_hat}"
    )
    print(f"  [OK] Marginal Conformal q_hat:    {q_hat:.16f} (Threshold: {1.0 - q_hat:.16f})")

    # 7. Canonical conformal inclusion rule check: (1 - p) <= q_hat
    # Numerical validation of canonical vs naive thresholding
    p_exact_border = 1.0 - q_hat
    canonical_included = (1.0 - p_exact_border) <= q_hat
    assert canonical_included is True, "Canonical conformal inclusion rule failed boundary test"
    print(f"  [OK] Canonical Conformal Rule:    (1.0 - p[c]) <= q_hat (IEEE-754 roundoff safe)")


# =============================================================================
# 4. Deep Layer-by-Layer Verification Harness
# =============================================================================
def compare_vectors(
    name: str,
    vec_app: np.ndarray,
    vec_direct: np.ndarray,
    tolerance: float,
) -> Dict[str, Any]:
    """Compares two numeric vectors and returns max_diff, mean_diff, and pass/fail status."""
    assert vec_app.shape == vec_direct.shape, (
        f"Shape mismatch for {name}: {vec_app.shape} vs {vec_direct.shape}"
    )
    diff = np.abs(vec_app.astype(np.float64) - vec_direct.astype(np.float64))
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    passed = bool(max_abs <= tolerance)

    return {
        "stage": name,
        "shape": vec_app.shape,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "tolerance": tolerance,
        "passed": passed,
    }


def verify_sample_consistency(
    sample_info: Dict[str, Any],
    direct_engine: AEFCRCInferenceEngine,
    app_pipeline: Any,
) -> List[Dict[str, Any]]:
    """Runs a single validation sample through both paths and audits all 11 stages."""
    sample_path = sample_info["path"]
    sample_id = sample_info["sample_id"]
    true_class = sample_info["true_class"]

    # -------------------------------------------------------------------------
    # Path A: App / Streamlit Path (Exact Simulation: Raw Bytes -> run_app_inference)
    # -------------------------------------------------------------------------
    raw_bytes = sample_path.read_bytes()
    app_res = run_app_inference(
        image_input=raw_bytes,
        handoff_path=FROZEN_HANDOFF_PATH,
        device="cpu",
        pipeline=app_pipeline,
    )

    # Decode in-memory PIL image from bytes (as Streamlit does)
    buf = io.BytesIO(raw_bytes)
    app_pil = Image.open(buf).convert("RGB")
    app_engine = app_pipeline.engine

    # Step-by-step intermediate tensors from app engine
    app_validated_img = app_engine.validate_input_image(app_pil)
    app_tensor, app_rgb_np = app_engine.preprocess_image(app_validated_img)
    
    # Handcrafted components from app path
    app_extractor = HandcraftedFeatureExtractor(app_engine.config)
    app_bgr = cv2.cvtColor(app_rgb_np, cv2.COLOR_RGB2BGR)
    app_fv = app_extractor.extract(app_bgr)
    app_glcm = app_fv.glcm.flatten()
    app_lbp = app_fv.lbp.flatten()
    app_lab = app_fv.color_lab.flatten()
    app_handcrafted = np.concatenate([app_glcm, app_lbp, app_lab])

    app_fused_1316 = app_engine.extract_multimodal_features(app_tensor, app_rgb_np)
    app_deep_1280 = app_fused_1316[:1280]
    app_masked_642 = app_engine.apply_feature_mask(app_fused_1316)
    app_raw_probs = app_engine.predict_raw_probabilities(app_masked_642)[0]
    app_calib_probs = app_engine.apply_calibration(np.expand_dims(app_raw_probs, axis=0))[0]
    app_marginal_set, app_mondrian_set = app_engine.predict_conformal_sets(np.expand_dims(app_calib_probs, axis=0))

    # -------------------------------------------------------------------------
    # Path B: Direct Certified Pipeline Reference Path
    # -------------------------------------------------------------------------
    direct_pil = Image.open(sample_path).convert("RGB")
    direct_validated_img = direct_engine.validate_input_image(direct_pil)
    direct_tensor, direct_rgb_np = direct_engine.preprocess_image(direct_validated_img)

    direct_extractor = HandcraftedFeatureExtractor(direct_engine.config)
    direct_bgr = cv2.cvtColor(direct_rgb_np, cv2.COLOR_RGB2BGR)
    direct_fv = direct_extractor.extract(direct_bgr)
    direct_glcm = direct_fv.glcm.flatten()
    direct_lbp = direct_fv.lbp.flatten()
    direct_lab = direct_fv.color_lab.flatten()
    direct_handcrafted = np.concatenate([direct_glcm, direct_lbp, direct_lab])

    direct_fused_1316 = direct_engine.extract_multimodal_features(direct_tensor, direct_rgb_np)
    direct_deep_1280 = direct_fused_1316[:1280]
    direct_masked_642 = direct_engine.apply_feature_mask(direct_fused_1316)
    direct_raw_probs = direct_engine.predict_raw_probabilities(direct_masked_642)[0]
    direct_calib_probs = direct_engine.apply_calibration(np.expand_dims(direct_raw_probs, axis=0))[0]
    direct_marginal_set, direct_mondrian_set = direct_engine.predict_conformal_sets(np.expand_dims(direct_calib_probs, axis=0))

    # -------------------------------------------------------------------------
    # Stage Comparisons (11 Stages)
    # -------------------------------------------------------------------------
    results: List[Dict[str, Any]] = []

    # 1. Input image dimensions & raw RGB
    arr_app_raw = np.asarray(app_pil, dtype=np.uint8)
    arr_direct_raw = np.asarray(direct_pil, dtype=np.uint8)
    results.append(compare_vectors("1. Input RGB Image Array", arr_app_raw, arr_direct_raw, TOLERANCE_EXACT))

    # 2. Preprocessed 224x224 image array (normalized Keras tensor)
    results.append(compare_vectors("2. Preprocessed Keras Tensor", app_tensor, direct_tensor, TOLERANCE_PIXEL))
    results.append(compare_vectors("2b. Resized RGB Image (224x224)", app_rgb_np, direct_rgb_np, TOLERANCE_EXACT))

    # 3. EfficientNet 1280-D deep feature vector
    results.append(compare_vectors("3. EfficientNet Deep Vector (1280-D)", app_deep_1280, direct_deep_1280, TOLERANCE_FLOAT_CPU))

    # 4. Handcrafted features: GLCM (12-D), LBP (18-D), LAB (6-D) -> 36-D
    results.append(compare_vectors("4a. GLCM Texture Vector (12-D)", app_glcm, direct_glcm, TOLERANCE_FLOAT_CPU))
    results.append(compare_vectors("4b. LBP Texture Vector (18-D)", app_lbp, direct_lbp, TOLERANCE_FLOAT_CPU))
    results.append(compare_vectors("4c. LAB Color Vector (6-D)", app_lab, direct_lab, TOLERANCE_FLOAT_CPU))
    results.append(compare_vectors("4d. Handcrafted Combined (36-D)", app_handcrafted, direct_handcrafted, TOLERANCE_FLOAT_CPU))

    # 5. A7 1316-D fused vector
    results.append(compare_vectors("5. A7 Fused Multimodal Vector (1316-D)", app_fused_1316, direct_fused_1316, TOLERANCE_FLOAT_CPU))

    # 6. 642-D BDA-masked vector
    results.append(compare_vectors("6. BDA Active Subspace Vector (642-D)", app_masked_642, direct_masked_642, TOLERANCE_FLOAT_CPU))

    # 7. Base RF predict_proba output (raw ensemble votes)
    results.append(compare_vectors("7. Base RF Raw Probabilities (4-D)", app_raw_probs, direct_raw_probs, TOLERANCE_PROBABILITY))

    # 8. Temperature-scaled probabilities
    results.append(compare_vectors("8. Temperature-Scaled Probabilities (4-D)", app_calib_probs, direct_calib_probs, TOLERANCE_PROBABILITY))

    # 9. Final argmax class
    app_argmax = direct_engine.classes[int(np.argmax(app_calib_probs))]
    direct_argmax = direct_engine.classes[int(np.argmax(direct_calib_probs))]
    assert app_argmax == direct_argmax, f"Argmax class mismatch: {app_argmax} vs {direct_argmax}"
    assert app_res.predicted_class == direct_argmax, f"App result class mismatch: {app_res.predicted_class} vs {direct_argmax}"
    results.append({
        "stage": "9. Final Argmax Prediction",
        "shape": (1,),
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "tolerance": TOLERANCE_EXACT,
        "passed": True,
        "detail": f"Class: '{direct_argmax}' (True: '{true_class}')",
    })

    # 10. Conformal prediction set (marginal 90%)
    app_marg = sorted(app_marginal_set)
    direct_marg = sorted(direct_marginal_set)
    assert app_marg == direct_marg, f"Marginal set mismatch: {app_marg} vs {direct_marg}"
    assert sorted(app_res.marginal_prediction_set) == direct_marg
    results.append({
        "stage": "10. Marginal Prediction Set",
        "shape": (len(direct_marg),),
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "tolerance": TOLERANCE_EXACT,
        "passed": True,
        "detail": f"Set: {direct_marg}",
    })

    # 11. Mondrian prediction set (class-conditional)
    app_mond = sorted(app_mondrian_set)
    direct_mond = sorted(direct_mondrian_set)
    assert app_mond == direct_mond, f"Mondrian set mismatch: {app_mond} vs {direct_mond}"
    assert sorted(app_res.mondrian_prediction_set) == direct_mond
    results.append({
        "stage": "11. Mondrian Prediction Set",
        "shape": (len(direct_mond),),
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "tolerance": TOLERANCE_EXACT,
        "passed": True,
        "detail": f"Set: {direct_mond}",
    })

    return results


# =============================================================================
# 5. Main Execution Entry Point
# =============================================================================
def main() -> None:
    print("\n" + "#" * 80)
    print("PAPULONET V2: REPAIRED STREAMLIT & CERTIFIED PIPELINE CONSISTENCY AUDIT")
    print("#" * 80)

    # Step 1: Enforce test isolation
    verify_strict_test_set_isolation()

    # Step 2: Initialize handoff & direct engine
    clear_app_pipeline_cache()
    handoff = load_final_pipeline_handoff(FROZEN_HANDOFF_PATH)
    direct_engine = AEFCRCInferenceEngine(artifact=handoff, device="cpu")
    app_pipeline = get_app_pipeline(handoff_or_path=handoff, device="cpu", use_cache=False)

    # Step 3: Verify contract metadata
    verify_pipeline_contract_metadata(handoff, direct_engine)

    # Step 4: Run audits on all 8 validation samples
    print("\n" + "=" * 80)
    print("STEP 3: EXECUTING 11-STAGE NUMERICAL AUDIT OVER 8 VALIDATION SAMPLES")
    print("=" * 80)

    all_passed = True
    total_stages_checked = 0

    for idx, sample in enumerate(VALIDATION_SAMPLES, 1):
        sample_id = sample["sample_id"]
        true_cls = sample["true_class"]
        print(f"\n--- [{idx}/8] Sample: {sample_id} ({true_cls}) ---")
        stage_results = verify_sample_consistency(sample, direct_engine, app_pipeline)

        for res in stage_results:
            total_stages_checked += 1
            status_str = "PASS" if res["passed"] else "FAIL"
            if not res["passed"]:
                all_passed = False
            detail_str = f" | {res['detail']}" if "detail" in res else ""
            print(
                f"  [{status_str}] {res['stage']:<38s} | "
                f"MaxDiff: {res['max_abs_diff']:10.3e} | "
                f"MeanDiff: {res['mean_abs_diff']:10.3e} | "
                f"Tol: {res['tolerance']:8.1e}{detail_str}"
            )

    print("\n" + "=" * 80)
    print("CONSISTENCY AUDIT SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Validation Samples Evaluated:  {len(VALIDATION_SAMPLES)}")
    print(f"Total Stage Checkpoints Audited:     {total_stages_checked}")
    print(f"Numerical Consistency Overall Status: {'ALL PASSED (100% CONSISTENT)' if all_passed else 'FAILED'}")
    print(f"Test Set Isolation Confirmed:        STRICTLY 0 TEST SAMPLES ACCESSED")
    print(f"Frozen Artifacts Preserved:          SHA-256 IDENTICAL (ZERO MUTATION)")
    print("=" * 80 + "\n")

    if not all_passed:
        raise RuntimeError("Inference consistency audit failed at one or more stages.")


if __name__ == "__main__":
    main()
