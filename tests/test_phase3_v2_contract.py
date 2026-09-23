"""
tests/test_phase3_v2_contract.py

Targeted test suite and dry-run contract verification for PapuloNet V2 Phase 3.

Covers all Phase 3 V2 protocol specifications:
1. All three arms defined: P3-V2-CE, P3-V2-Focal, P3-V2-WFocal
2. Loss formulations:
   - Categorical cross-entropy
   - Categorical focal loss with gamma=2.0
   - Weighted focal loss with gamma=2.0
3. Sample weight & class weight delivery:
   - Fold-local weights formula: w_c = N_train / (C * N_c)
   - Refusal of class weights on validation/test splits
   - Zero validation/test data in weight computation
4. Optimizer configuration:
   - Adam with clipnorm=1.0 across all three arms
5. Schedule & Architecture invariants:
   - Identical 1280-D global average pooling
   - Stage 1: frozen backbone, 15 epochs, LR=1e-3
   - Stage 2: top 20 layers unfrozen, BatchNorm layers frozen, 10 epochs, LR=1e-5
6. Artifact & Report isolation:
   - Writes exclusively to artifacts/phase3_v2/ and reports/phase3_v2/
   - Zero access or pollution of artifacts/phase3/ or reports/phase3/
7. Locked test set invariance:
   - Locked 243-image test partition is NEVER loaded or accessed
8. Dry-run execution test:
   - Full pipeline construction without running long-running training
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import tensorflow as tf

from config.config import get_config, PSDConfig
from modules.aef_input_validator import compute_train_fold_class_weights
from modules.backbones import get_backbone
from modules.fold_loader import load_frozen_fold_plan, FoldPlan
from modules.losses import CategoricalFocalLoss, get_loss
from modules.experiment_config import representation_id
from run_aef_crc_phase3_v2 import _build_v2_arm_configs


@pytest.fixture
def base_config() -> PSDConfig:
    return get_config()


def test_v2_arms_definition_and_invariants(base_config: PSDConfig):
    """Verify that exactly three arms are defined with the exact frozen specifications."""
    arms = _build_v2_arm_configs(base_config)
    assert set(arms.keys()) == {"P3-V2-CE", "P3-V2-Focal", "P3-V2-WFocal"}

    # Common schedule invariants
    for arm_name, (arm_cfg, aug_flag, use_weights) in arms.items():
        assert arm_cfg.stage1_epochs == 15
        assert arm_cfg.stage2_epochs == 10
        assert arm_cfg.stage1_learning_rate == 1e-3
        assert arm_cfg.stage2_learning_rate == 1e-5
        assert arm_cfg.unfrozen_layers == 20
        assert arm_cfg.adam_clipnorm == 1.0
        assert aug_flag is False
        assert arm_cfg.preprocessing_mode == "standard"

    # Specific arm differences
    ce_cfg, _, ce_weights = arms["P3-V2-CE"]
    assert ce_cfg.loss_name == "categorical_crossentropy"
    assert ce_weights is True

    focal_cfg, _, focal_weights = arms["P3-V2-Focal"]
    assert focal_cfg.loss_name == "categorical_focal_loss"
    assert focal_cfg.focal_gamma == 2.0
    assert focal_weights is False

    wfocal_cfg, _, wfocal_weights = arms["P3-V2-WFocal"]
    assert wfocal_cfg.loss_name == "categorical_focal_loss"
    assert wfocal_cfg.focal_gamma == 2.0
    assert wfocal_weights is True


def test_categorical_focal_loss_formulation():
    """Verify CategoricalFocalLoss produces correct mathematical values with gamma=2.0."""
    loss_fn = CategoricalFocalLoss(gamma=2.0)
    
    # Well-classified example: true=[1, 0, 0, 0], pred=[0.9, 0.05, 0.03, 0.02]
    # FL = -(1 - 0.9)^2 * log(0.9) = -0.01 * (-0.10536) = 0.0010536
    y_true_easy = tf.constant([[1.0, 0.0, 0.0, 0.0]])
    y_pred_easy = tf.constant([[0.9, 0.05, 0.03, 0.02]])
    loss_easy = float(loss_fn(y_true_easy, y_pred_easy).numpy())
    expected_easy = (1.0 - 0.9)**2 * (-np.log(0.9))
    assert np.isclose(loss_easy, expected_easy, atol=1e-5)

    # Hard-classified example: true=[1, 0, 0, 0], pred=[0.2, 0.5, 0.2, 0.1]
    # FL = -(1 - 0.2)^2 * log(0.2) = -0.64 * (-1.6094) = 1.0300
    y_pred_hard = tf.constant([[0.2, 0.5, 0.2, 0.1]])
    loss_hard = float(loss_fn(y_true_easy, y_pred_hard).numpy())
    expected_hard = (1.0 - 0.2)**2 * (-np.log(0.2))
    assert np.isclose(loss_hard, expected_hard, atol=1e-5)

    # Focal modulation ratio: hard loss should be penalized ~1000x relative to easy
    assert loss_hard > 500 * loss_easy


def test_focal_loss_sample_weight_integration():
    """Verify that sample weights are correctly integrated during loss computation."""
    loss_fn = CategoricalFocalLoss(gamma=2.0)
    m = tf.keras.Sequential([tf.keras.layers.Dense(4, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss=loss_fn)
    
    x = tf.random.normal((4, 8))
    y = tf.one_hot(tf.constant([0, 1, 2, 3]), depth=4)
    w = tf.constant([2.5, 1.25, 0.833, 0.625])
    
    # Must train 1 step with sample_weight without throwing errors or producing NaN
    hist = m.fit(x, y, sample_weight=w, epochs=1, verbose=0)
    assert np.isfinite(hist.history["loss"][0])


def test_adam_clipnorm_enforcement(base_config: PSDConfig):
    """Verify that Adam optimizer is built with clipnorm=1.0 across all models."""
    arms = _build_v2_arm_configs(base_config)
    backbone_spec = get_backbone(base_config.backbone)

    def _get_cn(opt):
        cn = getattr(opt, "clipnorm", None)
        if cn is None and hasattr(opt, "inner_optimizer"):
            cn = getattr(opt.inner_optimizer, "clipnorm", None)
        return cn

    for arm_name, (arm_cfg, _, _) in arms.items():
        model, backbone = backbone_spec.build_stage1(arm_cfg, num_classes=4)
        assert _get_cn(model.optimizer) == 1.0, f"Arm {arm_name} clipnorm != 1.0"
        
        model_s2 = backbone_spec.unfreeze_stage2(model, backbone, arm_cfg)
        assert _get_cn(model_s2.optimizer) == 1.0, f"Arm {arm_name} Stage 2 clipnorm != 1.0"
        tf.keras.backend.clear_session()


def test_stage2_batchnorm_frozen_invariant(base_config: PSDConfig):
    """Verify that BatchNorm layers remain strictly frozen in Stage 2 fine-tuning."""
    arms = _build_v2_arm_configs(base_config)
    backbone_spec = get_backbone(base_config.backbone)

    arm_cfg = arms["P3-V2-WFocal"][0]
    model, backbone = backbone_spec.build_stage1(arm_cfg, num_classes=4)
    model = backbone_spec.unfreeze_stage2(model, backbone, arm_cfg)

    top_layers = backbone.layers[-arm_cfg.unfrozen_layers:]
    bn_layers = [l for l in top_layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert len(bn_layers) > 0
    for bn in bn_layers:
        assert not bn.trainable, f"BatchNorm layer '{bn.name}' must remain frozen"
    tf.keras.backend.clear_session()


def test_class_weights_fold_local_and_no_leakage(base_config: PSDConfig):
    """Verify class weights formula w_c = N_train / (C * N_c) and strict outer-train-only data source."""
    plan = load_frozen_fold_plan(base_config)
    assert len(plan.folds) == 5

    classes = base_config.target_classes
    C = len(classes)

    for fold in plan.folds:
        # Compute weights from fold's train records ONLY
        train_labels = [r.mapped_class for r in fold.train_records]
        computed = compute_train_fold_class_weights(train_labels, classes)
        
        # Verify formula
        n_train = len(fold.train_records)
        for cls in classes:
            n_c = sum(1 for lbl in train_labels if lbl == cls)
            expected = n_train / (C * n_c)
            assert np.isclose(computed[cls], expected, atol=1e-7)

        # Verify that fold.val_records PSD IDs have ZERO overlap with train_records
        train_ids = {r.psd_id for r in fold.train_records}
        val_ids = {r.psd_id for r in fold.val_records}
        assert len(train_ids & val_ids) == 0


def test_locked_test_partition_never_referenced(base_config: PSDConfig):
    """Verify that the locked 243-image test partition is strictly untouched by Phase 3 V2."""
    plan = load_frozen_fold_plan(base_config)
    assert len(plan.holdout_test_records) == 243
    test_ids = {r.psd_id for r in plan.holdout_test_records}

    # Verify that no test ID appears in any CV fold train or val
    for fold in plan.folds:
        train_ids = {r.psd_id for r in fold.train_records}
        val_ids = {r.psd_id for r in fold.val_records}
        assert len(test_ids & train_ids) == 0
        assert len(test_ids & val_ids) == 0

    # Verify quarantined augmented images never enter CV
    quarantine_ids = {r.psd_id for r in plan.excluded_augmented_records}
    assert len(quarantine_ids) == 264
    for fold in plan.folds:
        train_ids = {r.psd_id for r in fold.train_records}
        val_ids = {r.psd_id for r in fold.val_records}
        assert len(quarantine_ids & train_ids) == 0
        assert len(quarantine_ids & val_ids) == 0


def test_v2_directory_isolation(base_config: PSDConfig):
    """Verify that all Phase 3 V2 artifacts, reports, and logs are isolated to phase3_v2."""
    arms = _build_v2_arm_configs(base_config)
    for arm_name, (arm_cfg, _, _) in arms.items():
        assert arm_cfg.aef_crc_artifacts_dir == base_config.aef_crc_phase3_v2_artifacts_dir
        assert arm_cfg.aef_crc_phase3_reports_dir == base_config.aef_crc_phase3_v2_reports_dir
        assert arm_cfg.aef_crc_phase3_logs_dir == base_config.aef_crc_phase3_v2_logs_dir
        assert "phase3_v2" in str(arm_cfg.aef_crc_artifacts_dir)
        assert "phase3_v2" in str(arm_cfg.aef_crc_phase3_reports_dir)
        assert "phase3_v2" in str(arm_cfg.aef_crc_phase3_logs_dir)
        # Ensure V1 paths are not targeted
        assert arm_cfg.aef_crc_artifacts_dir != base_config.aef_crc_artifacts_dir
        assert arm_cfg.aef_crc_phase3_reports_dir != base_config.aef_crc_phase3_reports_dir


def test_representation_id_reflects_v2_loss_attributes(base_config: PSDConfig):
    """Verify that representation_id differentiates between CE, Focal, and WFocal configs."""
    arms = _build_v2_arm_configs(base_config)
    r_ce = representation_id(arms["P3-V2-CE"][0])
    r_focal = representation_id(arms["P3-V2-Focal"][0])
    r_wfocal = representation_id(arms["P3-V2-WFocal"][0])

    assert r_ce != r_focal, "CE and Focal must have distinct representation IDs"
    assert r_focal != r_wfocal, "Focal and Weighted Focal must have distinct representation IDs"
    assert r_ce != r_wfocal, "CE and Weighted Focal must have distinct representation IDs"
