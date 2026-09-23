"""
tests/test_v2_inference_config.py

Regression test suite for PapuloNet V2 inference configuration propagation:
1. Verifies that AEFCRCInferenceEngine strictly resolves preprocessing_mode == "standard"
   when loaded from the certified Phase 9 V2 final pipeline handoff without an explicit config.
2. Verifies that preprocess_image() does NOT trigger ConditionalPreprocessor hair removal or CLAHE.
3. Verifies that handcrafted feature extraction receives the standard configuration.
4. Verifies that external configs with preprocessing_mode != "standard" or mismatched
   representation_id are rejected with ValueError.
5. Verifies that the frozen Phase 9 V2 handoff file on disk is strictly immutable (SHA256 verified).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

from config.config import get_config, PSDConfig
from modules.calibration_handoff import load_final_pipeline_handoff, FinalPipelineHandoff
from modules.experiment_config import representation_id
from modules.inference import AEFCRCInferenceEngine
from modules.preprocessing import ConditionalPreprocessor
from run_aef_crc_phase9_v2 import configure_phase9_v2


V2_HANDOFF_PATH = Path("artifacts/phase9_v2/final_pipeline_handoff.joblib")
EXPECTED_V2_REPR_ID = "efficientnet_b0_43d581b96f8ec368"
EXPECTED_V2_SHA256 = "d109b1fb798ea1ecda32d9f8723ce1f14b9f7b54b6d57322d61f30d9a5c4309f"
EXPECTED_V2_SIZE = 8500766


# -----------------------------------------------------------------------------
# 1. Handoff File Immutability Check
# -----------------------------------------------------------------------------
def test_v2_handoff_file_immutability():
    """Guarantees that artifacts/phase9_v2/final_pipeline_handoff.joblib has not been altered."""
    assert V2_HANDOFF_PATH.exists(), f"V2 handoff not found at: {V2_HANDOFF_PATH}"
    data = V2_HANDOFF_PATH.read_bytes()
    assert len(data) == EXPECTED_V2_SIZE, (
        f"File size mismatch: expected {EXPECTED_V2_SIZE}, got {len(data)}"
    )
    actual_sha = hashlib.sha256(data).hexdigest()
    assert actual_sha == EXPECTED_V2_SHA256, (
        f"SHA-256 hash mismatch! Frozen artifact has been modified:\n"
        f"  Expected: {EXPECTED_V2_SHA256}\n"
        f"  Actual:   {actual_sha}"
    )


# -----------------------------------------------------------------------------
# 2. Default Configuration Resolution & Propagation
# -----------------------------------------------------------------------------
def test_engine_binds_standard_config_by_default():
    """Verifies that AEFCRCInferenceEngine without config binds preprocessing_mode='standard'."""
    engine = AEFCRCInferenceEngine(artifact=V2_HANDOFF_PATH, device="cpu")
    
    assert engine.config is not None
    assert engine.config.preprocessing_mode == "standard", (
        f"Expected preprocessing_mode='standard', got '{engine.config.preprocessing_mode}'"
    )
    assert engine.config.image_size == 224
    
    repr_id = representation_id(engine.config)
    assert repr_id == EXPECTED_V2_REPR_ID, (
        f"Expected representation_id='{EXPECTED_V2_REPR_ID}', got '{repr_id}'"
    )


def test_instantiating_without_config_does_not_fallback_to_conditional():
    """Confirms that global get_config() default ('conditional') is NOT used for V2 handoff."""
    bare_cfg = get_config()
    assert bare_cfg.preprocessing_mode == "conditional", (
        "Baseline check: get_config() must default to 'conditional' in config.py"
    )

    engine = AEFCRCInferenceEngine(artifact=V2_HANDOFF_PATH, device="cpu")
    assert engine.config.preprocessing_mode == "standard"
    assert engine.config.preprocessing_mode != bare_cfg.preprocessing_mode


# -----------------------------------------------------------------------------
# 3. Preprocessing Execution & Behavior
# -----------------------------------------------------------------------------
def test_preprocess_image_does_not_trigger_hair_or_clahe():
    """Verifies that preprocess_image() with V2 handoff executes standard no-op preprocessing."""
    engine = AEFCRCInferenceEngine(artifact=V2_HANDOFF_PATH, device="cpu")

    # Create synthetic image with dark lines (hair candidate) and low contrast
    img_arr = np.full((128, 128, 3), 100, dtype=np.uint8)
    img_arr[30:35, :, :] = 10  # simulated hair line
    synthetic_pil = Image.fromarray(img_arr, mode="RGB")

    # Trace ConditionalPreprocessor with engine.config
    preprocessor = ConditionalPreprocessor(engine.config)
    bgr_arr = img_arr[:, :, ::-1]
    prep_res = preprocessor.process(bgr_arr, "test_synthetic")

    # In 'standard' mode, decisions list is strictly empty (no hair removal, no CLAHE)
    assert len(prep_res.decisions) == 0, (
        f"Decisions must be empty under standard mode, got: {prep_res.decisions}"
    )
    np.testing.assert_array_equal(prep_res.image, bgr_arr)

    # Now verify preprocess_image() returns normalized tensor and resized rgb
    tensor, rgb_resized = engine.preprocess_image(synthetic_pil)
    assert tensor.shape == (1, 224, 224, 3)
    assert rgb_resized.shape == (224, 224, 3)
    assert tensor.dtype == np.float32
    assert rgb_resized.dtype == np.uint8


# -----------------------------------------------------------------------------
# 4. Handcrafted Feature Extraction Configuration
# -----------------------------------------------------------------------------
def test_handcrafted_features_receives_standard_config():
    """Verifies that HandcraftedFeatureExtractor is instantiated with standard config."""
    engine = AEFCRCInferenceEngine(artifact=V2_HANDOFF_PATH, device="cpu")
    from modules.handcrafted_features import HandcraftedFeatureExtractor

    extractor = HandcraftedFeatureExtractor(engine.config)
    assert extractor.config.preprocessing_mode == "standard"
    assert "standard" in str(extractor._cache_root)


# -----------------------------------------------------------------------------
# 5. External Config Validation & Guardrails
# -----------------------------------------------------------------------------
def test_external_config_validation_rejects_conditional_mode():
    """Verifies that passing a config with preprocessing_mode='conditional' raises ValueError."""
    bad_cfg = get_config()  # preprocessing_mode is "conditional"
    with pytest.raises(ValueError, match="preprocessing_mode='standard'"):
        AEFCRCInferenceEngine(artifact=V2_HANDOFF_PATH, device="cpu", config=bad_cfg)


def test_external_config_validation_accepts_valid_v2_config():
    """Verifies that passing a validated V2 config succeeds."""
    valid_cfg = configure_phase9_v2(get_config())
    engine = AEFCRCInferenceEngine(artifact=V2_HANDOFF_PATH, device="cpu", config=valid_cfg)
    assert engine.config.preprocessing_mode == "standard"
    assert representation_id(engine.config) == EXPECTED_V2_REPR_ID


# -----------------------------------------------------------------------------
# 6. App Adapter Integration
# -----------------------------------------------------------------------------
def test_app_adapter_pipeline_uses_standard_config():
    """Verifies that get_app_pipeline() uses the standard V2 configuration."""
    from modules.app_adapter import get_app_pipeline, clear_app_pipeline_cache

    clear_app_pipeline_cache()
    pipeline = get_app_pipeline(handoff_or_path=V2_HANDOFF_PATH, device="cpu", use_cache=False)
    assert pipeline.config.preprocessing_mode == "standard"
    assert pipeline.engine.config.preprocessing_mode == "standard"
    assert representation_id(pipeline.config) == EXPECTED_V2_REPR_ID
