"""
modules/experiment_config.py

AEF-CRC: experiment identity + validation.

The experiment-defining fields already live directly on PSDConfig
(backbone, preprocessing_mode, classifier_name, feature_selection_method,
random_seed, and Phase 3's training hyperparameters) -- see config.py's
own comment for why a second parallel config class was deliberately
rejected. This module provides three things a plain dataclass field set
doesn't give you for free:

  1. representation_id(config) -- collision-safe identity for what a
     Phase-3 deep-feature cache actually IS. Hardened on this review:
     previously just f"{backbone}__{preprocessing_mode}__seed{seed}",
     which silently ignored every OTHER thing that materially changes a
     trained representation -- stage1/stage2 learning rate, epoch
     counts, image size, dropout rate, unfrozen-layer count, training-
     time augmentation. Two configs differing ONLY in stage2_learning_rate
     would previously have gotten the SAME representation_id despite
     being two genuinely different trained models -- a real cache-
     collision risk, not a hypothetical one. Now a SHA-256 hash (first
     16 hex chars -- collision risk at that length is negligible for
     this project's actual number of experiments, and this is an
     identity check, not a security boundary) over a canonical dict of
     every field in _REPRESENTATION_FIELDS below, prefixed with the
     backbone name for human readability. Adding a newly-discovered
     representation-defining field later is one line in that tuple, not
     a risk of forgetting to update a manually concatenated string.

  2. experiment_id(config) -- representation_id() plus classifier_name
     and feature_selection_method: the FULL downstream identity.
     Changing classifier_name or feature_selection_method changes
     experiment_id but NOT representation_id (verified by test) --
     that is the concrete meaning of "changing downstream classifier (e.g. Random Forest -> Logistic Regression) must not
     force recomputation of the trained deep representation." This
     property already holds structurally today (Phase 3's deep-feature
     cache is keyed by `source_experiment`, a name chosen once when
     training runs, completely independent of config.classifier_name/
     feature_selection_method -- those are only ever read downstream,
     in Phase 5/6). representation_id/experiment_id do not change that
     mechanism; they make the property explicit and testable rather
     than an accident of which functions happen to read which config
     fields. Neither id is currently used to NAME an actual cache or
     report directory -- source_experiment remains a free string chosen
     by whoever runs Phase 3 (modules.training.run_experiment's
     existing, tested parameter), unchanged this phase.

  3. validate_experiment_config(config) -- fails fast, at experiment
     setup, if config.backbone/classifier_name/feature_selection_method
     names a registry key that doesn't exist, rather than failing deep
     inside training.py after real compute has already been spent.
"""

from __future__ import annotations

import hashlib
import json

from config.config import PSDConfig
from modules.backbones import BACKBONE_REGISTRY
from modules.fusion import CLASSIFIER_REGISTRY
from modules.feature_selection import SELECTOR_REGISTRY


# Every PSDConfig field that materially defines what
# artifacts/phase3/deep_features/<experiment>/ actually contains, i.e.
# changing it produces (or legitimately could produce) a different
# trained representation even with backbone/dataset held fixed. NOT
# included: classifier_name, feature_selection_method (downstream-only,
# see module docstring), early_stopping_patience (affects HOW LONG
# training may run, not what it converges toward, given the same
# data/seed/LR schedule -- borderline, deliberately left out to avoid
# invalidating a cache over a knob that doesn't change the model this
# project has actually been observed to produce; revisit if that
# assumption turns out wrong). config.image_size is declared in TWO
# places in config.py (Phase 1's generic image handling AND Phase 3's
# EfficientNet input size) -- both are literally the same dataclass
# field (Python keeps only the later declaration), so there is only one
# value to read here, not a real ambiguity.
_REPRESENTATION_FIELDS = (
    "backbone", "preprocessing_mode", "random_seed",
    "image_size", "batch_size",
    "stage1_learning_rate", "stage2_learning_rate",
    "stage1_epochs", "stage2_epochs",
    "unfrozen_layers", "dropout_rate", "training_time_augmentation",
)

# Pretrained-weight source is a fixed constant today, not a config
# field (see modules.efficientnet_model.build_stage1_model's
# weights="imagenet") -- included as a literal in the hash so the day
# it DOES become configurable, changing it changes representation_id
# too, not just the day it happens to matter.
_PRETRAINED_SOURCE = "imagenet"


def representation_id(config: PSDConfig) -> str:
    """Two configs with the same representation_id could validly read
    the SAME cached deep-feature vectors; two with a different value
    for ANY field in _REPRESENTATION_FIELDS could not, and now don't
    collide -- see module docstring for the specific gap this closed."""
    canonical = {name: getattr(config, name) for name in _REPRESENTATION_FIELDS}
    canonical["pretrained_source"] = _PRETRAINED_SOURCE
    encoded = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{config.backbone}_{digest}"


def experiment_id(config: PSDConfig) -> str:
    """Full downstream identity: representation_id() plus the two
    choices that only affect Phase 5/6, never the cached representation
    itself."""
    return f"{representation_id(config)}__{config.classifier_name}__{config.feature_selection_method}"


def validate_experiment_config(config: PSDConfig) -> None:
    errors = []
    if config.backbone not in BACKBONE_REGISTRY:
        errors.append(f"backbone='{config.backbone}' not in {sorted(BACKBONE_REGISTRY.keys())}")
    if config.classifier_name not in CLASSIFIER_REGISTRY:
        errors.append(f"classifier_name='{config.classifier_name}' not in {sorted(CLASSIFIER_REGISTRY.keys())}")
    if config.feature_selection_method not in SELECTOR_REGISTRY:
        errors.append(f"feature_selection_method='{config.feature_selection_method}' not in {sorted(SELECTOR_REGISTRY.keys())}")
    if config.preprocessing_mode not in ("standard", "conditional"):
        errors.append(f"preprocessing_mode='{config.preprocessing_mode}' not in ('standard', 'conditional')")
    if errors:
        raise ValueError("Invalid experiment configuration:\n  " + "\n  ".join(errors))
