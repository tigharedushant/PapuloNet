"""
tests/test_phase3_framework_independent.py

Regression tests for the two Phase 3 modules that do NOT require a
deep learning framework: modules/preprocessing.py and
modules/evaluation.py. These were verified ad-hoc during Phase 3
delivery; this file makes that verification permanent and re-runnable,
rather than a one-off script whose result is only in a chat transcript.

modules/image_loader.py, efficientnet_model.py, and training.py are
NOT tested here -- they require TensorFlow, which is unavailable in
this sandbox. See run_aef_crc_phase3.py's own honest failure message
for that boundary; it is not duplicated with fake tests here.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from config.config import get_config
from modules.preprocessing import ConditionalPreprocessor
from modules.evaluation import compute_fold_metrics, aggregate_fold_metrics, mcnemar_test


# ============================================================
# Synthetic image builders (smooth structure, not pixel noise --
# noise breaks both the blackhat hair-detector and the contrast
# metric in unrealistic ways; see Phase 3 delivery notes)
# ============================================================

def _clean_image():
    img = np.full((224, 224, 3), 200, dtype=np.uint8)
    cv2.circle(img, (112, 112), 60, (90, 70, 60), -1)
    return cv2.GaussianBlur(img, (9, 9), 0)


def _low_contrast_image():
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    cv2.circle(img, (112, 112), 60, (135, 133, 130), -1)
    return cv2.GaussianBlur(img, (9, 9), 0)


def _hair_image():
    img = _clean_image()
    rng = np.random.default_rng(1)
    for _ in range(20):
        x1, y1 = rng.integers(0, 224, 2)
        x2, y2 = x1 + rng.integers(-120, 120), y1 + rng.integers(-120, 120)
        cv2.line(img, (x1, y1), (x2, y2), (15, 15, 15), 2)
    return img


# ============================================================
# preprocessing.py
# ============================================================

def test_clean_image_triggers_neither_operation():
    cfg = get_config()
    pre = ConditionalPreprocessor(cfg)
    result = pre.process(_clean_image(), "TEST_clean")
    assert all(not d.applied for d in result.decisions)


def test_low_contrast_image_triggers_only_clahe():
    cfg = get_config()
    pre = ConditionalPreprocessor(cfg)
    result = pre.process(_low_contrast_image(), "TEST_low_contrast")
    applied = {d.operation: d.applied for d in result.decisions}
    assert applied["clahe"] is True
    assert applied["hair_removal"] is False


def test_hair_image_triggers_only_hair_removal():
    cfg = get_config()
    pre = ConditionalPreprocessor(cfg)
    result = pre.process(_hair_image(), "TEST_hair")
    applied = {d.operation: d.applied for d in result.decisions}
    assert applied["hair_removal"] is True
    assert applied["clahe"] is False


def test_untriggered_image_is_pixel_identical():
    cfg = get_config()
    pre = ConditionalPreprocessor(cfg)
    clean = _clean_image()
    result = pre.process(clean, "TEST_untouched")
    assert np.array_equal(result.image, clean)


def test_triggered_hair_removal_actually_modifies_pixels():
    cfg = get_config()
    pre = ConditionalPreprocessor(cfg)
    hair = _hair_image()
    result = pre.process(hair, "TEST_modified")
    assert not np.array_equal(result.image, hair)


def test_standard_mode_is_always_a_noop():
    cfg = dataclasses.replace(get_config(), preprocessing_mode="standard")
    pre = ConditionalPreprocessor(cfg)
    hair = _hair_image()  # would trigger hair removal in conditional mode
    result = pre.process(hair, "TEST_p0")
    assert np.array_equal(result.image, hair)
    assert result.decisions == []


# ============================================================
# image_loader.py -- argument validation only (runs before the
# TensorFlow import inside build_dataset(), so testable without TF)
# ============================================================

def test_augmentation_on_validation_data_is_refused():
    from modules.image_loader import build_dataset
    cfg = get_config()
    try:
        build_dataset(cfg, [], [], training=False, apply_training_time_augmentation=True)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_class_weights_on_validation_data_is_refused():
    from modules.image_loader import build_dataset
    cfg = get_config()
    try:
        build_dataset(cfg, [], [], training=False, apply_training_time_augmentation=False, class_weights={"A": 1.0})
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_nan_class_weight_is_refused():
    from modules.image_loader import build_dataset
    cfg = get_config()
    try:
        build_dataset(cfg, [], ["A"], training=True, apply_training_time_augmentation=False, class_weights={"A": float("nan")})
        assert False, "should have raised ValueError"
    except ValueError:
        pass

_CLASSES = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]


def test_perfect_predictions_give_macro_f1_of_one():
    y_true = ["Psoriasis"] * 10 + ["Lichen_Planus"] * 8 + ["Pityriasis_Rosea"] * 5 + ["Seborrheic_Dermatitis"] * 3
    m = compute_fold_metrics(y_true, list(y_true), _CLASSES, fold_index=0)
    assert abs(m.macro_f1 - 1.0) < 1e-9
    assert m.confusion.sum() - m.confusion.diagonal().sum() == 0


def test_never_predicted_class_gets_zero_recall_not_a_crash():
    y_true = ["Psoriasis"] * 10 + ["Seborrheic_Dermatitis"] * 3
    y_pred = ["Psoriasis"] * 13
    m = compute_fold_metrics(y_true, y_pred, _CLASSES, fold_index=0)
    assert m.per_class["Seborrheic_Dermatitis"].recall == 0.0
    assert m.per_class["Psoriasis"].recall == 1.0


def test_aggregate_across_identical_folds_has_zero_std():
    y_true = ["Psoriasis"] * 10 + ["Lichen_Planus"] * 8 + ["Pityriasis_Rosea"] * 4 + ["Seborrheic_Dermatitis"] * 3
    fold_metrics = [compute_fold_metrics(y_true, list(y_true), _CLASSES, i) for i in range(3)]
    agg = aggregate_fold_metrics(fold_metrics)
    assert abs(agg.macro_f1_mean - 1.0) < 1e-9
    assert agg.macro_f1_std < 1e-9
    assert agg.n_folds == 3


def test_mcnemar_below_threshold_returns_none_not_a_fabricated_pvalue():
    y_true = ["X"] * 10
    y_pred_a = ["Y"] * 5 + ["X"] * 5
    y_pred_b = ["X"] * 5 + ["Y"] * 5
    assert mcnemar_test(y_true, y_pred_a, y_pred_b) is None


def test_mcnemar_above_threshold_returns_a_real_result():
    y_true = ["X"] * 30
    y_pred_a = ["Y"] * 25 + ["X"] * 5
    y_pred_b = ["X"] * 25 + ["Y"] * 5
    result = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert result is not None
    assert result["n01"] == 25 and result["n10"] == 5
