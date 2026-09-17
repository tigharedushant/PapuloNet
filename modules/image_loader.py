"""
modules/image_loader.py

AEF-CRC Phase 3: Image Loading (tf.data).

*** NOT EXECUTED IN THIS SANDBOX ***
TensorFlow is not installed here and there is no network access to
install it (verified: `pip install tensorflow` fails with no matching
distribution). This module is written to the same standard as every
executed module in this project, but it has NOT been run, and per
this project's explicit rule against fabricating results, that
distinction is not being blurred. Run it in an environment with
TensorFlow (Colab, or any machine with network access) and treat its
first real run there as the actual verification.

Bridges modules/fold_loader.py's Fold objects (file paths + labels)
and modules/preprocessing.py's ConditionalPreprocessor into batched,
prefetched tf.data.Dataset objects ready for model.fit().

Design constraints from the Phase 3 brief, followed here:
  - Part 21: batch loading, prefetch, no full-dataset RAM load.
  - Part 13: training-time augmentation applied ONLY to the training
    split, generated dynamically (never written to disk as new files,
    which would be indistinguishable from PSD-HP's own source-derived
    augmented images -- a confusion this project has already spent two
    review rounds eliminating at the data layer; this loader does not
    reintroduce it at the modeling layer).
  - Part 15: this loader outputs raw model logits/probabilities. It
    never labels anything "calibrated confidence" -- that term is
    reserved for Phase 8.

--- class_weight + tf.data fix (found on Phase 3 review, before any
    real training run -- this is a documented API contract issue,
    not something that needed a live run to discover) ---
The original version of this module produced (image, one_hot_label)
batches, and modules/training.py passed the resulting tf.data.Dataset
to model.fit(..., class_weight=class_weight_dict). Per Keras's own
documented contract for Model.fit(): "class_weight ... is not
supported when x is a dataset, generator, or keras.utils.Sequence
instance, instead provide the sample_weights directly." Passing both
a Dataset and class_weight together is exactly the combination Keras's
own docs say is unsupported -- this would either be silently ignored
or raise an error depending on version, meaning fold-local class
weighting (Part 12, a stated PRIMARY imbalance strategy) could have
silently not been applied at all.

Fix: build_dataset() now computes a per-example sample_weight directly
in the pipeline (from the same fold.class_weights dict, looked up by
each record's known class label at dataset-construction time) and
yields (image, one_hot_label, sample_weight) for training data. This
is the path Keras's own docs recommend, and it is version-independent
-- it does not rely on Keras's internal (and, for Dataset inputs,
unsupported) class_weight-to-sample_weight conversion at all.
Validation/test data yields plain (image, one_hot_label) with no
weighting, matching Part 12's intent that class weighting is a
TRAINING-loss adjustment, not something that should distort how
validation metrics are read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config.config import PSDConfig
from modules.fold_loader import Fold
from modules.aef_input_validator import ImageRecord
from modules.preprocessing import ConditionalPreprocessor


def _class_to_index(class_order: List[str]) -> dict:
    return {cls: i for i, cls in enumerate(class_order)}


def build_dataset(
    config: PSDConfig,
    records: List[ImageRecord],
    class_order: List[str],
    training: bool,
    apply_training_time_augmentation: bool,
    class_weights: Optional[Dict[str, float]] = None,
    batch_size: Optional[int] = None,
):
    """Returns a tf.data.Dataset of (image, one_hot_label) batches for
    val/test, or (image, one_hot_label, sample_weight) batches for
    training when class_weights is provided.

    training: if True, shuffles every epoch. If False (val/test), no
    shuffling -- order is preserved so predictions can be matched back
    to specific records for confusion matrices and McNemar's test.

    apply_training_time_augmentation: Part 13/14's B0 vs B1 switch.
    Must be False whenever training=False -- enforced below, not left
    to the caller to remember (Part 13: "never applied to
    validation/test").

    class_weights: fold.class_weights from fold_loader.py, i.e. ALREADY
    computed from this fold's training portion only. Passing this for
    training=False data is refused below -- weighting validation/test
    was never the intent (Part 12) and doing so would make validation
    metrics harder to interpret against the unweighted test-time metric.
    NaN weights (a class with zero samples in this fold -- see
    compute_train_fold_class_weights) are refused explicitly rather
    than silently propagated as NaN sample weights, which would corrupt
    the loss for every example of that class without an obvious error.

    Validation of the arguments happens BEFORE the TensorFlow import
    below, deliberately -- so these checks (and their tests) don't
    require TensorFlow to be installed at all.
    """
    if apply_training_time_augmentation and not training:
        raise ValueError(
            "apply_training_time_augmentation=True with training=False -- "
            "Part 13 explicitly forbids training-time augmentation on "
            "validation/test data. This is refused, not silently corrected."
        )
    if class_weights is not None and not training:
        raise ValueError(
            "class_weights provided with training=False -- class weighting is a "
            "training-loss adjustment (Part 12), not intended for validation/test. Refused."
        )
    if class_weights is not None:
        for cls, w in class_weights.items():
            if w != w:  # NaN
                raise ValueError(
                    f"class_weights['{cls}'] is NaN (zero samples of that class in this fold's "
                    f"training portion). Refusing to build a training dataset with an undefined "
                    f"sample weight rather than silently corrupting the loss for that class."
                )

    import tensorflow as tf  # deferred import -- see module docstring

    label_to_index = _class_to_index(class_order)
    preprocessor = ConditionalPreprocessor(config)

    paths = [str(r.file_path) for r in records]
    labels = [label_to_index[r.mapped_class] for r in records]
    psd_ids = [r.psd_id for r in records]

    if class_weights is not None:
        sample_weights = [float(class_weights[r.mapped_class]) for r in records]
        path_ds = tf.data.Dataset.from_tensor_slices((paths, labels, psd_ids, sample_weights))
    else:
        path_ds = tf.data.Dataset.from_tensor_slices((paths, labels, psd_ids))

    def _load_and_preprocess(*args):
        if class_weights is not None:
            path, label, psd_id, sample_weight = args
        else:
            path, label, psd_id = args
            sample_weight = None

        def _py_load(path_bytes, psd_id_bytes):
            import cv2
            cv2.setNumThreads(1)
            path_str = path_bytes.numpy().decode("utf-8")
            psd_id_str = psd_id_bytes.numpy().decode("utf-8")
            p = Path(path_str)
            if not p.exists():
                import re
                m = re.match(r"^([a-zA-Z]):[/\\](.*)", path_str)
                if m:
                    wsl_p = Path(f"/mnt/{m.group(1).lower()}/{m.group(2).replace('\\', '/')}")
                    if wsl_p.exists():
                        p = wsl_p
                else:
                    m = re.match(r"^/mnt/([a-zA-Z])/(.*)", path_str)
                    if m:
                        win_p = Path(f"{m.group(1).upper()}:/{m.group(2)}")
                        if win_p.exists():
                            p = win_p
            path_str = str(p)
            img_bgr = cv2.imread(path_str)
            if img_bgr is None:
                raise IOError(f"Failed to read image at {path_str} (psd_id={psd_id_str})")
            result = preprocessor.process(img_bgr, psd_id_str)
            img_rgb = result.image[:, :, ::-1]  # BGR (OpenCV) -> RGB (Keras/EfficientNet convention)
            img_resized = cv2.resize(img_rgb, (config.image_size, config.image_size))
            return img_resized.astype(np.float32)

        image = tf.py_function(func=_py_load, inp=[path, psd_id], Tout=tf.float32)
        image.set_shape([config.image_size, config.image_size, 3])
        # EfficientNet's own preprocess_input handles ImageNet-pretrained
        # normalization -- deliberately not hand-rolled here, to avoid
        # ever silently mismatching what the pretrained weights expect.
        from tensorflow.keras.applications.efficientnet import preprocess_input
        image = preprocess_input(image)

        one_hot_label = tf.one_hot(label, depth=len(class_order))

        if sample_weight is not None:
            return image, one_hot_label, sample_weight
        return image, one_hot_label
    ds = path_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=len(records), seed=config.random_seed, reshuffle_each_iteration=True)
        if apply_training_time_augmentation:
            ds = ds.map(_conservative_augment, num_parallel_calls=tf.data.AUTOTUNE)

    effective_batch_size = batch_size if batch_size is not None else config.batch_size
    ds = ds.batch(effective_batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _conservative_augment(*args):
    """Part 13: 'conservative medically appropriate transformations,'
    explicitly NOT 'aggressive transformations that could alter disease
    morphology.' Horizontal flip and small rotation/brightness only --
    no large rotations, no shearing, no color-channel swaps (papulosquamous
    lesion color IS diagnostically relevant; hue jitter is deliberately
    excluded, unlike a generic image-classification augmentation recipe).

    Accepts *args (2- or 3-tuple) so this works identically whether or
    not sample_weight is present in the pipeline -- augmentation must
    never touch the weight, only the image."""
    import tensorflow as tf

    image = args[0]
    rest = args[1:]
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.92, upper=1.08)
    return (image,) + rest
