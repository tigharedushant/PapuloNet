"""
tests/test_representation_hardening.py

Targeted tests for THIS phase's Task 1 (harden representation identity)
and Task 3 (be honest about the PyTorch/backend boundary).

Does not re-test what tests/test_phase_replaceability.py already covers
(representation_id stable across classifier/selector changes; changes
with backbone/preprocessing/seed) -- this file covers the NEW fields
that review added (batch_size, learning rates, epochs, dropout,
image_size, unfrozen_layers, training_time_augmentation) and the
manifest-level collision check (validate_deep_feature_cache now checks
representation_id, tested in test_deep_feature_contract.py) from the
config-identity side.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from config.config import get_config
from modules.experiment_config import representation_id, _REPRESENTATION_FIELDS


# ============================================================
# Task 1: every representation-defining field actually changes the id
# ============================================================

# One deliberately-different value per field -- exercises the exact gap
# this phase closed (previously ALL of these except random_seed were
# silently ignored by representation_id).
_FIELD_OVERRIDES = {
    "image_size": 299,
    "batch_size": 64,
    "stage1_learning_rate": 5e-4,
    "stage2_learning_rate": 5e-6,
    "stage1_epochs": 20,
    "stage2_epochs": 15,
    "unfrozen_layers": 40,
    "dropout_rate": 0.5,
    "training_time_augmentation": "true",
}


@pytest.mark.parametrize("field", sorted(_FIELD_OVERRIDES))
def test_representation_id_changes_when_training_hyperparameter_changes(field):
    base = get_config()
    changed = dataclasses.replace(base, **{field: _FIELD_OVERRIDES[field]})
    assert representation_id(base) != representation_id(changed), (
        f"representation_id did not change when '{field}' changed -- this field is listed in "
        f"_REPRESENTATION_FIELDS but isn't actually affecting the hash."
    )


def test_representation_fields_table_has_no_downstream_only_field():
    """Guards the boundary in the other direction: classifier_name and
    feature_selection_method must NEVER end up in this tuple (that
    would make representation_id needlessly change on a downstream-only
    choice, defeating the whole point of the split)."""
    assert "classifier_name" not in _REPRESENTATION_FIELDS
    assert "feature_selection_method" not in _REPRESENTATION_FIELDS


def test_representation_id_format_is_backbone_prefixed_hash():
    """Not asserting the exact hash algorithm (an implementation detail)
    -- just the externally-relevant shape: starts with the backbone
    name, deterministic, collision-resistant length."""
    config = get_config()
    rid = representation_id(config)
    assert rid.startswith(f"{config.backbone}_")
    suffix = rid[len(config.backbone) + 1:]
    assert re.fullmatch(r"[0-9a-f]{16}", suffix), f"expected a 16-char hex suffix, got {suffix!r}"


def test_representation_id_identical_for_two_separately_constructed_equal_configs():
    """Determinism across separate Python objects, not just repeated
    calls on the same object -- what actually matters when Phase 3 and
    Phase 5 build their configs independently."""
    a = get_config()
    b = dataclasses.replace(get_config())  # a fresh, independently-constructed but equal config
    assert representation_id(a) == representation_id(b)


# ============================================================
# Task 3: honest disclosure of the TF/Keras training-loop boundary
# ============================================================

def test_training_module_discloses_the_keras_specific_boundary():
    """This is a documentation-freshness check, not a behavioral one:
    modules/training.py must keep saying, in its own docstring, that
    the training LOOP (not just model construction) is Keras-specific
    -- so if someone later actually generalizes it without updating the
    docs, this test is the tripwire that says the claim is now stale."""
    import inspect
    import modules.training as training_module
    source = inspect.getsource(training_module)
    assert "NOT abstract the training loop" in source
    assert "TensorFlow/Keras-specific" in source
    assert "PyTorch" in source


def test_backbone_registry_has_exactly_one_entry_still():
    """Task 3: do not claim replaceability that hasn't been built.
    EfficientNet-B0 remains the only registered backbone this phase --
    a second entry appearing here without a second real implementation
    module would be exactly the 'claim ConvNeXt/Swin compatibility
    before it's proven' this task explicitly forbids."""
    from modules.backbones import BACKBONE_REGISTRY
    assert set(BACKBONE_REGISTRY) == {"efficientnet_b0"}
