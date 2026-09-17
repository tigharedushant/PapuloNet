"""
tests/test_phase3_cache_validation.py

Tests for Phase 3 fold cache validation in modules/training.py:
1. matching current provenance -> cached fold may be reused;
2. mismatched dataset freeze hash -> cached fold rejected/retrained;
3. mismatched fold-plan hash -> cached fold rejected/retrained;
4. missing provenance field -> cached fold rejected/retrained;
5. wrong experiment/representation -> cached fold rejected/retrained;
6. missing required artifact files -> cached fold rejected/retrained.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from config.config import get_config
from modules.experiment_config import representation_id
from modules.training import validate_cached_fold


@pytest.fixture
def mock_fold_dir(tmp_path: Path):
    """Creates a mock fold directory with valid artifacts matching current repo hashes."""
    config = get_config()
    
    # Read authoritative hashes
    freeze_p = config.aef_crc_reports_dir / "dataset_freeze.json"
    plan_p = config.aef_crc_reports_dir / "fold_plan.csv"
    freeze_hash = hashlib.sha256(freeze_p.read_bytes()).hexdigest()
    plan_hash = hashlib.sha256(plan_p.read_bytes()).hexdigest()
    repr_id = representation_id(config)
    
    fold_dir = tmp_path / "fold_00"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    (fold_dir / "best_model.keras").write_bytes(b"PK\x03\x04mock_keras_weights")
    
    manifest = {
        "fold_index": 0,
        "random_seed": config.random_seed,
        "preprocessing_mode": config.preprocessing_mode,
        "training_time_augmentation": False,
        "representation_id": repr_id,
        "dataset_freeze_hash": freeze_hash,
        "fold_plan_hash": plan_hash,
        "macro_f1": 0.725,
        "accuracy": 0.750,
        "device_type": "GPU",
    }
    (fold_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    
    cm_data = {
        "class_order": list(config.target_classes),
        "matrix": [[10, 1, 0, 0], [1, 15, 2, 0], [0, 1, 12, 1], [0, 0, 1, 8]],
        "per_class": {
            cls: {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 10}
            for cls in config.target_classes
        },
    }
    (fold_dir / "confusion_matrix.json").write_text(json.dumps(cm_data), encoding="utf-8")
    
    return fold_dir, config, freeze_hash, plan_hash, repr_id


def test_cache_validation_matching_provenance(mock_fold_dir):
    """1. Matching current provenance -> cached fold may be reused."""
    fold_dir, config, _, _, _ = mock_fold_dir
    is_valid, reason = validate_cached_fold(
        fold_dir=fold_dir,
        config=config,
        fold_index=0,
        experiment_name="P3-BASE",
        apply_training_time_augmentation=False,
    )
    assert is_valid is True
    assert "Verified matching" in reason


def test_cache_validation_mismatched_freeze_hash(mock_fold_dir):
    """2. Mismatched dataset freeze hash -> cached fold rejected/retrained."""
    fold_dir, config, _, _, _ = mock_fold_dir
    manifest_p = fold_dir / "manifest.json"
    m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    m_data["dataset_freeze_hash"] = "stale_hash_deadbeef12345678"
    manifest_p.write_text(json.dumps(m_data), encoding="utf-8")
    
    is_valid, reason = validate_cached_fold(
        fold_dir=fold_dir,
        config=config,
        fold_index=0,
        experiment_name="P3-BASE",
        apply_training_time_augmentation=False,
    )
    assert is_valid is False
    assert "dataset_freeze_hash mismatch" in reason


def test_cache_validation_mismatched_plan_hash(mock_fold_dir):
    """3. Mismatched fold-plan hash -> cached fold rejected/retrained."""
    fold_dir, config, _, _, _ = mock_fold_dir
    manifest_p = fold_dir / "manifest.json"
    m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    m_data["fold_plan_hash"] = "stale_plan_hash_abcdef987654"
    manifest_p.write_text(json.dumps(m_data), encoding="utf-8")
    
    is_valid, reason = validate_cached_fold(
        fold_dir=fold_dir,
        config=config,
        fold_index=0,
        experiment_name="P3-BASE",
        apply_training_time_augmentation=False,
    )
    assert is_valid is False
    assert "fold_plan_hash mismatch" in reason


def test_cache_validation_missing_provenance_field(mock_fold_dir):
    """4. Missing provenance field -> cached fold rejected/retrained."""
    fold_dir, config, _, _, _ = mock_fold_dir
    manifest_p = fold_dir / "manifest.json"
    original_m_data = json.loads(manifest_p.read_text(encoding="utf-8"))

    for field in ["dataset_freeze_hash", "fold_plan_hash", "representation_id", "random_seed"]:
        m_data = dict(original_m_data)
        del m_data[field]
        manifest_p.write_text(json.dumps(m_data), encoding="utf-8")
        
        is_valid, reason = validate_cached_fold(
            fold_dir=fold_dir,
            config=config,
            fold_index=0,
            experiment_name="P3-BASE",
            apply_training_time_augmentation=False,
        )
        assert is_valid is False
        assert f"Missing {field}" in reason or "mismatch" in reason


def test_cache_validation_wrong_experiment_or_representation(mock_fold_dir):
    """5. Wrong experiment/representation -> cached fold rejected/retrained."""
    fold_dir, config, _, _, _ = mock_fold_dir
    manifest_p = fold_dir / "manifest.json"
    m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    m_data["representation_id"] = "efficientnet_b0_wrong_repr_id_123"
    manifest_p.write_text(json.dumps(m_data), encoding="utf-8")
    
    is_valid, reason = validate_cached_fold(
        fold_dir=fold_dir,
        config=config,
        fold_index=0,
        experiment_name="P3-BASE",
        apply_training_time_augmentation=False,
    )
    assert is_valid is False
    assert "representation_id mismatch" in reason


def test_cache_validation_missing_artifact_files(mock_fold_dir):
    """6. Missing required artifact files -> cached fold rejected/retrained."""
    fold_dir, config, _, _, _ = mock_fold_dir
    (fold_dir / "best_model.keras").unlink()
    
    is_valid, reason = validate_cached_fold(
        fold_dir=fold_dir,
        config=config,
        fold_index=0,
        experiment_name="P3-BASE",
        apply_training_time_augmentation=False,
    )
    assert is_valid is False
    assert "Missing best_model.keras" in reason
