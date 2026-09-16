"""
tests/test_phase3_synthetic_framework.py

Comprehensive synthetic test suite for Phase 3 reusable framework.
Tests all 10 framework components using synthetic imagery, synthetic labels,
and toy arrays without requiring the real clinical dataset.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from config.config import get_config, PSDConfig
from modules.framework_validation import (
    validate_configuration,
    validate_fold_definitions,
    validate_preprocessing,
    validate_augmentation,
    validate_model_construction,
    validate_device_detection,
    validate_training_configuration,
    validate_checkpoint_selection,
    validate_evaluation_logic,
    validate_winner_selection,
    validate_phase3_framework,
)
from modules.evaluation import (
    AggregatedMetrics,
    compute_fold_metrics,
    select_phase3_winner,
)


@pytest.fixture
def base_config() -> PSDConfig:
    return get_config()


def test_master_framework_validation_all_passed(base_config):
    """Verifies that all 10 framework validation checks pass cleanly on baseline configuration."""
    ok, errors = validate_phase3_framework(base_config, verbose=False)
    assert ok is True, f"Framework validation failed with errors: {errors}"
    assert len(errors) == 0


def test_validate_configuration_detects_deviations(base_config):
    """Verifies that validate_configuration flags unauthorized hyperparameter drift."""
    # 1. Image size deviation
    bad_cfg1 = dataclasses.replace(base_config, image_size=256)
    ok1, errs1 = validate_configuration(bad_cfg1)
    assert ok1 is False
    assert any("image_size" in e for e in errs1)

    # 2. Stage 1 epochs deviation
    bad_cfg2 = dataclasses.replace(base_config, stage1_epochs=20)
    ok2, errs2 = validate_configuration(bad_cfg2)
    assert ok2 is False
    assert any("stage1_epochs" in e for e in errs2)

    # 3. Backbone deviation
    bad_cfg3 = dataclasses.replace(base_config, backbone="resnet50")
    ok3, errs3 = validate_configuration(bad_cfg3)
    assert ok3 is False
    assert any("backbone" in e for e in errs3)

    # 4. Dropout deviation
    bad_cfg4 = dataclasses.replace(base_config, dropout_rate=0.5)
    ok4, errs4 = validate_configuration(bad_cfg4)
    assert ok4 is False
    assert any("dropout_rate" in e for e in errs4)


def test_validate_fold_definitions_invariants(base_config):
    """Verifies that fold plan passes K=5, disjointness, and quarantine invariants."""
    ok, errors = validate_fold_definitions(base_config)
    assert ok is True, f"Fold invariants failed: {errors}"


def test_validate_preprocessing_safeguards(base_config):
    """Verifies standard vs conditional preprocessing behavior and density safeguards."""
    ok, errors = validate_preprocessing(base_config)
    assert ok is True, f"Preprocessing validation failed: {errors}"


def test_validate_augmentation_refusal_logic(base_config):
    """Verifies that augmentation and sample weights are strictly refused on validation splits."""
    ok, errors = validate_augmentation(base_config)
    assert ok is True, f"Augmentation validation failed: {errors}"


def test_validate_model_construction_architecture(base_config):
    """Verifies 1280-D global pooling, 2-stage unfreezing schedule, and probability summation."""
    ok, errors = validate_model_construction(base_config)
    assert ok is True, f"Model construction validation failed: {errors}"


def test_validate_device_detection_telemetry(base_config):
    """Verifies device detection structure and complete dictionary serialization."""
    ok, errors = validate_device_detection(base_config)
    assert ok is True, f"Device detection validation failed: {errors}"


def test_validate_training_configuration_class_weights(base_config):
    """Verifies balanced class weights calculation w_c = N_train / (C * N_c) and callbacks."""
    ok, errors = validate_training_configuration(base_config)
    assert ok is True, f"Training configuration validation failed: {errors}"


def test_validate_checkpoint_selection_val_loss(base_config):
    """Verifies that checkpoint selection strictly follows minimal validation loss."""
    ok, errors = validate_checkpoint_selection(base_config)
    assert ok is True, f"Checkpoint selection validation failed: {errors}"


def test_validate_evaluation_metrics_and_confusion(base_config):
    """Verifies multi-metric computation and exact C x C (4 x 4) confusion matrix geometry."""
    ok, errors = validate_evaluation_logic(base_config)
    assert ok is True, f"Evaluation logic validation failed: {errors}"


def test_validate_winner_selection_parsimony_threshold(base_config):
    """Verifies 0.005 equivalence threshold and parsimony preference for P3-BASE."""
    ok, errors = validate_winner_selection(base_config)
    assert ok is True, f"Winner selection validation failed: {errors}"
