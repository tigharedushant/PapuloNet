"""
modules/backbones.py

AEF-CRC: Backbone registry.

Design decision (brutal review, as requested): this is a plain
dict-keyed registry of small dataclasses wrapping existing functions --
NOT a class hierarchy / ABC / plugin-decorator system. This project
already has exactly this pattern, proven, in modules/adapters.py's
_ADAPTER_REGISTRY (dataset adapters keyed by string, resolved at
config-read time). Reusing that established convention here is
deliberate: consistent with the rest of the codebase, trivially
testable (a dict lookup + a call), and there is nothing about swapping
CNN backbones that benefits from more machinery than that. An ABC
hierarchy would add indirection without adding any capability this
dict doesn't already have.

BackboneSpec bundles the four functions modules/training.py actually
calls -- no more, no less; this is the real, minimal contract already
implicit in how training.py used efficientnet_model.py directly:

  build_stage1(config, num_classes) -> (model, backbone_object)
  unfreeze_stage2(model, backbone_object, config) -> model
  describe(model) -> ModelInfo
  extract_features(config, backbone_object, records, experiment_name,
                    fold_index, split_name, checkpoint_path) -> out_dir

modules/efficientnet_model.py is UNCHANGED by this file -- its four
functions are only referenced, not modified, preserving all of Phase
3's tested behavior exactly.

Adding a future backbone (ConvNeXt, Swin-V2) means: write one new
module implementing these same four functions (own file, own framework
choice if ever needed -- Part "framework decision" above), then add one
entry to BACKBONE_REGISTRY. Nothing in modules/training.py,
modules/fusion.py, or modules/feature_selection.py needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from modules.efficientnet_model import (
    build_stage1_model, unfreeze_for_stage2, describe_model, extract_deep_features,
)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    build_stage1: Callable
    unfreeze_stage2: Callable
    describe: Callable
    extract_features: Callable
    output_dim: int  # documented contract, checked by extract_deep_features's own assertion today


BACKBONE_REGISTRY = {
    "efficientnet_b0": BackboneSpec(
        name="efficientnet_b0",
        build_stage1=build_stage1_model,
        unfreeze_stage2=unfreeze_for_stage2,
        describe=describe_model,
        extract_features=extract_deep_features,
        output_dim=1280,
    ),
}


def get_backbone(name: str) -> BackboneSpec:
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Registered: {sorted(BACKBONE_REGISTRY.keys())}. "
            f"Fails fast here rather than at a deep call site inside training.py."
        )
    return BACKBONE_REGISTRY[name]


class DeepFeatureCacheError(ValueError):
    """Raised when a deep-feature cache's manifest doesn't match what the
    caller expects to load: wrong experiment/fold/split, wrong backbone,
    wrong dimension, or a PSD-ID mismatch (duplicate or missing). This is
    the concrete gap this check closes -- previously nothing verified a
    cache's identity before modules.efficientnet_model.load_deep_feature
    read individual .npy files one at a time; a mismatch would only
    surface later as a confusing shape error inside a classifier's
    .fit(), or not at all if dimensions happened to coincide."""


def validate_deep_feature_cache(
    config, experiment_name: str, fold_index: int, split_name: str,
    expected_psd_ids, backbone_name: str,
) -> None:
    """Checks the cache's manifest.json BEFORE any individual vector is
    read (modules.fusion.build_fusion_fold calls this once per fold/split
    for the 'deep' branch, not once per PSD ID). Fails loudly -- raises,
    never warns-and-continues -- on any of: missing manifest, wrong
    experiment_name/fold_index/split/backbone_name, wrong feature_dim
    (checked against get_backbone(backbone_name).output_dim, so this
    works for any future backbone without a hardcoded number here),
    wrong representation_id (Task 1, this review: source_experiment is a
    free string chosen by whoever runs Phase 3 -- this catches two
    genuinely different trained representations, e.g. differing only in
    stage2_learning_rate, that happened to reuse the same
    source_experiment name), duplicate PSD IDs in the manifest, or an
    expected PSD ID absent from the manifest's recorded set."""
    from modules.efficientnet_model import load_deep_feature_manifest
    from modules.experiment_config import representation_id

    manifest = load_deep_feature_manifest(config, experiment_name, fold_index, split_name)
    expected_dim = get_backbone(backbone_name).output_dim
    expected_ids = list(expected_psd_ids)
    expected_repr_id = representation_id(config)

    errors = []
    if manifest.get("experiment_name") != experiment_name:
        errors.append(f"manifest experiment_name={manifest.get('experiment_name')!r} != expected {experiment_name!r}")
    if manifest.get("fold_index") != fold_index:
        errors.append(f"manifest fold_index={manifest.get('fold_index')!r} != expected {fold_index!r}")
    if manifest.get("split") != split_name:
        errors.append(f"manifest split={manifest.get('split')!r} != expected {split_name!r}")
    if manifest.get("backbone_name") != backbone_name:
        errors.append(f"manifest backbone_name={manifest.get('backbone_name')!r} != expected {backbone_name!r}")
    if manifest.get("feature_dim") != expected_dim:
        errors.append(
            f"manifest feature_dim={manifest.get('feature_dim')!r} != backbone {backbone_name!r}'s "
            f"declared output_dim {expected_dim}"
        )
    if manifest.get("representation_id") != expected_repr_id:
        errors.append(
            f"manifest representation_id={manifest.get('representation_id')!r} != current config's "
            f"{expected_repr_id!r} -- this cache was produced by a DIFFERENT training configuration "
            f"(learning rate, epochs, image size, ...) even though backbone/dimension match."
        )

    manifest_ids = manifest.get("psd_ids")
    if manifest_ids is None:
        errors.append("manifest missing 'psd_ids' list")
    else:
        if len(set(manifest_ids)) != len(manifest_ids):
            errors.append("manifest 'psd_ids' contains duplicate PSD IDs")
        missing = sorted(set(expected_ids) - set(manifest_ids))
        if missing:
            errors.append(f"{len(missing)} expected PSD ID(s) not present in cache, e.g. {missing[:5]}")

    if errors:
        raise DeepFeatureCacheError(
            f"Deep-feature cache for experiment={experiment_name!r} fold={fold_index} "
            f"split={split_name!r} does not match what was requested:\n  " + "\n  ".join(errors)
        )
