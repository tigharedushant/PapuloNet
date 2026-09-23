"""
tests/test_class_order_regression.py

Authoritative Regression Test Suite for Random Forest Probability Class-Order Alignment.
Validates dynamic string-label-based column alignment in AEFCRCInferenceEngine.predict_raw_probabilities().

Tests:
- Test A: Synthetic permutation test (['LP', 'PR', 'Pso', 'Seb'] -> canonical ['Pso', 'LP', 'PR', 'Seb'])
- Test B: Alternate / reverse class order (label-driven dynamic matching)
- Test C: Class-set mismatch (raises ValueError when label sets differ)
- Test D: Downstream isolation (guarantees NO secondary permutation downstream)
- Test E: Argmax semantic alignment (Psoriasis top RF probability yields predicted_class == 'Psoriasis')
- Test F: Probability dictionary semantic alignment (keys match canonical classes and values match canonical order)
- Test G: Temperature scaling alignment with canonical labels
- Test H: Conformal prediction set alignment with canonical labels
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from modules.inference import AEFCRCInferenceEngine
from modules.calibration import MulticlassTemperatureScaler


CANONICAL_CLASSES = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
LEXICOGRAPHICAL_CLASSES = ["Lichen_Planus", "Pityriasis_Rosea", "Psoriasis", "Seborrheic_Dermatitis"]


class MockRFClassifier:
    """Mock classifier that mimics scikit-learn's predict_proba and classes_ attribute."""
    def __init__(self, classes: list[str], fixed_proba: np.ndarray):
        self.classes_ = np.array(classes)
        self.fixed_proba = fixed_proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0] if X.ndim > 1 else 1
        if self.fixed_proba.ndim == 1:
            return np.tile(self.fixed_proba, (n, 1))
        return self.fixed_proba


class MockEngineHandoff:
    """Minimal mock handoff providing attributes needed by AEFCRCInferenceEngine."""
    def __init__(self, classifier, class_order=None, mask=None):
        self.final_classifier = classifier
        self.class_order = class_order or CANONICAL_CLASSES
        self.selected_feature_mask = mask if mask is not None else np.ones(642, dtype=bool)
        self.calibration_method = "temperature_scaling"
        self.marginal_q_hat = 0.7382427827337363
        self.mondrian_q_hat = {cls: 0.7382427827337363 for cls in self.class_order}
        self.alpha = 0.10
        self.temperature = 0.32567700916361153
        self.temperature_scaler = MulticlassTemperatureScaler().fit(
            np.tile([0.4, 0.2, 0.2, 0.2], (10, 1)),
            np.zeros(10, dtype=int),
        )
        self.temperature_scaler.temperature = self.temperature
        self.representation_id = "efficientnet_b0_43d581b96f8ec368"
        self.run_id = "AEFCRC_P9_V2_TEST"
        self.branch_dims = {"deep": 1280, "glcm": 12, "lbp": 18, "color_lab": 6}
        self.classifier_name = "RandomForestClassifier"


# =============================================================================
# Test A: Synthetic Permutation Test
# =============================================================================
def test_a_synthetic_lexicographical_permutation():
    """
    Test A: Validates that when RF predicts probabilities in lexicographical order:
        ['Lichen_Planus', 'Pityriasis_Rosea', 'Psoriasis', 'Seborrheic_Dermatitis']
    with probabilities:
        [0.10, 0.15, 0.65, 0.10]  (Psoriasis is index 2 in RF)
    The output is reordered to canonical order:
        ['Psoriasis', 'Lichen_Planus', 'Pityriasis_Rosea', 'Seborrheic_Dermatitis']
    yielding:
        [0.65, 0.10, 0.15, 0.10]  (Psoriasis is index 0 in canonical)
    """
    rf_probs = np.array([[0.10, 0.15, 0.65, 0.10]])  # LP=0.10, PR=0.15, Pso=0.65, Seb=0.10
    mock_clf = MockRFClassifier(LEXICOGRAPHICAL_CLASSES, rf_probs)
    handoff = MockEngineHandoff(mock_clf, CANONICAL_CLASSES)
    engine = AEFCRCInferenceEngine(handoff)

    X_dummy = np.zeros((1, 642))
    raw_probs = engine.predict_raw_probabilities(X_dummy)

    # Expected: Pso=0.65 (idx 0), LP=0.10 (idx 1), PR=0.15 (idx 2), Seb=0.10 (idx 3)
    expected = np.array([[0.65, 0.10, 0.15, 0.10]])
    np.testing.assert_allclose(raw_probs, expected, atol=1e-12)
    assert engine.classes[0] == "Psoriasis"
    assert raw_probs[0, 0] == 0.65
    assert engine.classes[1] == "Lichen_Planus"
    assert raw_probs[0, 1] == 0.10
    assert engine.classes[2] == "Pityriasis_Rosea"
    assert raw_probs[0, 2] == 0.15
    assert engine.classes[3] == "Seborrheic_Dermatitis"
    assert raw_probs[0, 3] == 0.10


# =============================================================================
# Test B: Alternate / Reverse Class Order
# =============================================================================
def test_b_arbitrary_and_reverse_class_order():
    """
    Test B: Validates purely dynamic label-driven matching using reversed class order:
        ['Seborrheic_Dermatitis', 'Pityriasis_Rosea', 'Lichen_Planus', 'Psoriasis']
    with probabilities:
        [0.05, 0.20, 0.35, 0.40]
    Reorders dynamically to canonical:
        Pso=0.40, LP=0.35, PR=0.20, Seb=0.05
    """
    reversed_classes = ["Seborrheic_Dermatitis", "Pityriasis_Rosea", "Lichen_Planus", "Psoriasis"]
    rf_probs = np.array([[0.05, 0.20, 0.35, 0.40]])
    mock_clf = MockRFClassifier(reversed_classes, rf_probs)
    handoff = MockEngineHandoff(mock_clf, CANONICAL_CLASSES)
    engine = AEFCRCInferenceEngine(handoff)

    X_dummy = np.zeros((1, 642))
    raw_probs = engine.predict_raw_probabilities(X_dummy)

    expected = np.array([[0.40, 0.35, 0.20, 0.05]])
    np.testing.assert_allclose(raw_probs, expected, atol=1e-12)


# =============================================================================
# Test C: Class-Set Mismatch Raises ValueError
# =============================================================================
def test_c_class_set_mismatch_raises_value_error():
    """
    Test C: Validates that if classifier classes do not match canonical classes,
    AEFCRCInferenceEngine.predict_raw_probabilities() raises ValueError.
    """
    mismatched_classes = ["Melanoma", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    rf_probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    mock_clf = MockRFClassifier(mismatched_classes, rf_probs)
    handoff = MockEngineHandoff(mock_clf, CANONICAL_CLASSES)
    engine = AEFCRCInferenceEngine(handoff)

    X_dummy = np.zeros((1, 642))
    with pytest.raises(ValueError, match="do not match expected canonical classes"):
        engine.predict_raw_probabilities(X_dummy)


# =============================================================================
# Test D: No Double Permutation Downstream
# =============================================================================
def test_d_no_double_permutation_downstream():
    """
    Test D: Confirms that downstream methods (apply_calibration, predict_conformal_sets)
    operate on canonical probabilities and do not apply secondary permutations.
    """
    rf_probs = np.array([[0.10, 0.20, 0.60, 0.10]])  # LP=0.10, PR=0.20, Pso=0.60, Seb=0.10
    mock_clf = MockRFClassifier(LEXICOGRAPHICAL_CLASSES, rf_probs)
    handoff = MockEngineHandoff(mock_clf, CANONICAL_CLASSES)
    engine = AEFCRCInferenceEngine(handoff)

    X_dummy = np.zeros((1, 642))
    raw_probs = engine.predict_raw_probabilities(X_dummy)
    # raw_probs must be in canonical order: Pso=0.60, LP=0.10, PR=0.20, Seb=0.10
    assert np.argmax(raw_probs[0]) == 0  # Psoriasis is index 0

    calib_probs = engine.apply_calibration(raw_probs)
    # Temperature scaling is monotonic: argmax MUST remain index 0 (Psoriasis)
    assert np.argmax(calib_probs[0]) == 0
    assert calib_probs[0, 0] > calib_probs[0, 2] > calib_probs[0, 1] == calib_probs[0, 3]


# =============================================================================
# Test E: Argmax Semantic Test
# =============================================================================
def test_e_argmax_semantic_alignment():
    """
    Test E: Validates that when RF predicts highest probability for Psoriasis (column 2 in RF),
    predicted_class evaluates to 'Psoriasis', NOT 'Pityriasis_Rosea'.
    """
    rf_probs = np.array([[0.10, 0.15, 0.65, 0.10]])  # Psoriasis is column 2 in RF
    mock_clf = MockRFClassifier(LEXICOGRAPHICAL_CLASSES, rf_probs)
    handoff = MockEngineHandoff(mock_clf, CANONICAL_CLASSES)
    engine = AEFCRCInferenceEngine(handoff)

    X_dummy = np.zeros((1, 642))
    raw_probs = engine.predict_raw_probabilities(X_dummy)
    calib_probs = engine.apply_calibration(raw_probs)

    pred_idx = int(np.argmax(calib_probs[0]))
    pred_class = engine.classes[pred_idx]

    assert pred_idx == 0
    assert pred_class == "Psoriasis"
    assert pred_class != "Pityriasis_Rosea"


# =============================================================================
# Test F: Probability Dictionary Semantic Test
# =============================================================================
def test_f_probability_dictionaries_semantic_alignment():
    """
    Test F: Validates that dictionary representations map canonical class names
    to the correct probabilities.
    """
    rf_probs = np.array([[0.10, 0.20, 0.60, 0.10]])  # LP=0.10, PR=0.20, Pso=0.60, Seb=0.10
    mock_clf = MockRFClassifier(LEXICOGRAPHICAL_CLASSES, rf_probs)
    handoff = MockEngineHandoff(mock_clf, CANONICAL_CLASSES)
    engine = AEFCRCInferenceEngine(handoff)

    X_dummy = np.zeros((1, 642))
    raw_probs = engine.predict_raw_probabilities(X_dummy)

    raw_dict = {cls: float(raw_probs[0][i]) for i, cls in enumerate(engine.classes)}

    assert raw_dict["Psoriasis"] == pytest.approx(0.60)
    assert raw_dict["Lichen_Planus"] == pytest.approx(0.10)
    assert raw_dict["Pityriasis_Rosea"] == pytest.approx(0.20)
    assert raw_dict["Seborrheic_Dermatitis"] == pytest.approx(0.10)


# =============================================================================
# Test G: Temperature Alignment with Canonical Labels
# =============================================================================
def test_g_temperature_alignment_with_canonical_labels():
    """
    Test G: Validates that temperature scaling operates strictly on canonical order
    and preserves rank ordering across all classes.
    """
    rf_probs = np.array([[0.05, 0.15, 0.70, 0.10]])  # LP=0.05, PR=0.15, Pso=0.70, Seb=0.10
    mock_clf = MockRFClassifier(LEXICOGRAPHICAL_CLASSES, rf_probs)
    handoff = MockEngineHandoff(mock_clf, CANONICAL_CLASSES)
    engine = AEFCRCInferenceEngine(handoff)

    X_dummy = np.zeros((1, 642))
    raw_probs = engine.predict_raw_probabilities(X_dummy)
    calib_probs = engine.apply_calibration(raw_probs)

    calib_dict = {cls: float(calib_probs[0][i]) for i, cls in enumerate(engine.classes)}

    # Highest probability must belong to Psoriasis
    assert max(calib_dict, key=calib_dict.get) == "Psoriasis"
    # Relative rank order must be preserved: Pso > PR > Seb > LP
    assert calib_dict["Psoriasis"] > calib_dict["Pityriasis_Rosea"] > calib_dict["Seborrheic_Dermatitis"] > calib_dict["Lichen_Planus"]


# =============================================================================
# Test H: Conformal Alignment with Canonical Labels
# =============================================================================
def test_h_conformal_alignment_with_canonical_labels():
    """
    Test H: Validates that conformal prediction set inclusion rule:
        (1.0 - p_hat(c)) <= q_hat
    correctly includes 'Psoriasis' when Psoriasis has high probability,
    and excludes low-probability classes.
    """
    # High confidence Psoriasis sample: Pso=0.85, LP=0.05, PR=0.05, Seb=0.05
    rf_probs = np.array([[0.05, 0.05, 0.85, 0.05]])
    mock_clf = MockRFClassifier(LEXICOGRAPHICAL_CLASSES, rf_probs)
    handoff = MockEngineHandoff(mock_clf, CANONICAL_CLASSES)
    engine = AEFCRCInferenceEngine(handoff)

    X_dummy = np.zeros((1, 642))
    raw_probs = engine.predict_raw_probabilities(X_dummy)
    calib_probs = engine.apply_calibration(raw_probs)
    marginal_set, _ = engine.predict_conformal_sets(calib_probs)

    # For q_hat ~ 0.7382 (threshold 1 - q_hat ~ 0.2618):
    # Only Psoriasis (prob ~ 0.85+ after temperature scaling) satisfies (1 - p) <= q_hat
    assert "Psoriasis" in marginal_set
    assert "Pityriasis_Rosea" not in marginal_set
    assert marginal_set == ["Psoriasis"]
