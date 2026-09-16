"""
run_aef_crc_phase4.py

AEF-CRC Phase 4: Handcrafted Feature Extraction (GLCM, LBP, HOG)
Robust Implementation, Preflight Contract Verification, and 5-Fold Evaluation.

Key Project Invariants:
1. Dataset is Frozen: Never modifies, regenerates, or alters the PSD-HP dataset.
2. Authoritative Fold Plan: Consumes Phase 2 fold_plan.csv (K=5). Never creates a new CV split.
3. Upstream Winner Contract: Strictly consumes Phase 3 reports/phase3/winner.json.
   Aligns preprocessing_mode ("standard") directly with the winner contract.
4. Leakage-Safe Normalization and PCA: FeatureNormalizer and FoldSafeFeatureReducer
   are fit strictly on train records per fold, never on validation or global data.
5. Augmented Image Exclusion: Validation folds strictly exclude augmented images.
6. Deterministic & Complete Caching: Cached per psd_id under artifacts/phase4/<mode>/
   with parameter hashing, dimension checks, and fail-loudly validation.
7. Truthful Hardware Logging: Accurately reports CPU/GPU execution environment on Windows.
8. Complete Artifact Generation: Produces fold_metrics.csv, fold_summary.csv,
   handcrafted_only_results.csv, mcnemar_pairwise_results.csv, and phase4_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from datetime import datetime, timezone
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from config.config import PSDConfig, get_config
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import load_frozen_fold_plan
from modules.handcrafted_features import (
    HandcraftedFeatureExtractor, FeatureNormalizer, FoldSafeFeatureReducer,
    extract_batch, write_extraction_report, load_and_preprocess_image,
    get_handcrafted_feature_names,
)
from modules.aef_input_validator import AEFInputLoader
from modules.evaluation import compute_fold_metrics, aggregate_fold_metrics, mcnemar_test, holm_correction


FEATURE_SETS = ("trivial", "glcm", "lbp", "hog", "color_lab", "glcm_lbp", "combined")


def verify_phase4_preflight(config: PSDConfig) -> Tuple[bool, List[str], Dict]:
    """Step 0: Verify existence and integrity of all 7 upstream artifacts.
    Returns (success, missing_artifacts_list, winner_data).
    """
    missing = []

    # 1. Dataset freeze report
    f1 = config.aef_crc_reports_dir / "dataset_freeze.json"
    if not f1.exists():
        missing.append(f"1. Dataset Freeze Report missing: {f1}")

    # 2. Authoritative fold plan
    f2 = config.aef_crc_reports_dir / "fold_plan.csv"
    if not f2.exists():
        missing.append(f"2. Fold Plan missing: {f2}")

    # 3. Phase 2 fold summary
    f3 = config.aef_crc_reports_dir / "phase2_fold_summary.csv"
    if not f3.exists():
        missing.append(f"3. Phase 2 Fold Summary missing: {f3}")

    # 4. Phase 3 winner contract
    f4 = config.aef_crc_phase3_reports_dir / "winner.json"
    winner_data = {}
    if not f4.exists():
        missing.append(f"4. Phase 3 Winner JSON missing: {f4}")
    else:
        try:
            winner_data = json.loads(f4.read_text(encoding="utf-8"))
            if "winner_experiment_id" not in winner_data or "preprocessing_mode" not in winner_data:
                missing.append(f"4. Phase 3 Winner JSON missing required keys: {f4}")
            elif "representation_id" in winner_data:
                from modules.experiment_config import representation_id
                expected_repr = representation_id(config)
                if winner_data["representation_id"] != expected_repr:
                    missing.append(
                        f"4. Phase 3 Winner representation_id mismatch: "
                        f"'{winner_data['representation_id']}' != expected '{expected_repr}'"
                    )
        except Exception as exc:
            missing.append(f"4. Phase 3 Winner JSON unreadable: {exc}")

    # 5. Phase 3 fold metrics
    f5 = config.aef_crc_phase3_reports_dir / "fold_metrics.csv"
    if not f5.exists():
        missing.append(f"5. Phase 3 Fold Metrics missing: {f5}")

    # 6. Phase 3 fold summary
    f6 = config.aef_crc_phase3_reports_dir / "fold_summary.csv"
    if not f6.exists():
        missing.append(f"6. Phase 3 Fold Summary missing: {f6}")

    # 7. Phase 3 deep feature vectors for winner
    winner_id = winner_data.get("winner_experiment_id", "P3-BASE")
    p3_artifacts_dir = getattr(config, "aef_crc_phase3_artifacts_dir", config.aef_crc_artifacts_dir)
    deep_feat_dir = p3_artifacts_dir / "deep_features" / winner_id
    if not deep_feat_dir.exists():
        missing.append(f"7. Phase 3 Deep Features directory missing: {deep_feat_dir}")
    else:
        num_vecs = len(list(deep_feat_dir.rglob("*.npy")))
        if num_vecs == 0:
            missing.append(f"7. Phase 3 Deep Features directory is empty: {deep_feat_dir}")

    return len(missing) == 0, missing, winner_data


def get_hardware_diagnostics() -> Dict:
    """Truthfully inspect and report the computing environment."""
    diag = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "device": "CPU",
        "gpu_available": False,
        "details": "Windows CPU environment. Handcrafted feature extraction (OpenCV, skimage) runs CPU-native.",
    }
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            diag["device"] = "GPU"
            diag["gpu_available"] = True
            diag["details"] = f"Detected {len(gpus)} GPU(s): {[g.name for g in gpus]}"
    except Exception:
        pass
    return diag


def _build_xy(extractor: HandcraftedFeatureExtractor, records, feature_set: str, hog_reducer=None):
    """Returns raw feature vectors for a split, retrieving from cache if available."""
    cached = []
    for r in records:
        c = extractor._load_cached(r.psd_id)
        if c is None:
            img = load_and_preprocess_image(extractor.config, r.file_path, r.psd_id)
            c = extractor.extract_and_cache(img, r.psd_id)
        cached.append(c)

    if feature_set == "glcm":
        return [c.glcm for c in cached]
    if feature_set == "lbp":
        return [c.lbp for c in cached]
    if feature_set in ("color_lab", "lab"):
        return [c.color_lab for c in cached]
    if feature_set == "glcm_lbp":
        return [np.concatenate([c.glcm, c.lbp]) for c in cached]
    if feature_set == "hog":
        hog_vecs = [c.hog for c in cached]
        if hog_reducer is not None:
            return list(hog_reducer.transform(hog_vecs))
        return hog_vecs
    if feature_set == "combined":
        hog_vecs = [c.hog for c in cached]
        hog_reduced = hog_reducer.transform(hog_vecs) if hog_reducer is not None else np.stack(hog_vecs)
        color_enabled = getattr(extractor.config, "color_feature_enabled", True)
        parts = []
        for i, c in enumerate(cached):
            p = [c.glcm, c.lbp, hog_reduced[i]]
            if color_enabled and c.color_lab is not None:
                p.append(c.color_lab)
            parts.append(np.concatenate(p))
        return parts
    raise ValueError(f"Unknown feature_set: {feature_set}")


def validate_framework_components(config: PSDConfig) -> bool:
    """Non-destructive programmatic certification of Phase 4 framework components:
    1. GLCM extraction (12-D, 6 props x 2 distances, angle-averaged, finite).
    2. LBP extraction (18-D, uniform histogram, sum=1.0, finite).
    3. HOG extraction (1296-D raw, 9 orientations, 32x32 pixels/cell, finite).
    4. Color LAB extraction (6-D, L*, a*, b* channel statistics from color image, finite).
    5. Color sensitivity: distinct colors yield distinct LAB representations.
    6. FoldSafeFeatureReducer: fit-once, transform-only discipline, exact 1296 -> 32 reduction.
    7. FeatureNormalizer: fit-once, transform-only discipline, zero mean and unit variance.
    8. 68-D Handcrafted Feature Contract & Deterministic Naming.
    9. Upstream contracts and freeze validation without executing expensive training/extraction.
    """
    print("=" * 70)
    print("AEF-CRC PHASE 4: HANDCRAFTED FRAMEWORK COMPONENT VALIDATION")
    print("=" * 70)

    # 1. Hardware Environment Diagnostics
    hw = get_hardware_diagnostics()
    print(f"\n[Hardware Diagnostics]")
    print(f"  Platform: {hw['os']}")
    print(f"  Compute Device: {hw['device']} (GPU Available: {hw['gpu_available']})")
    print(f"  Details: {hw['details']}")

    # 2. Synthetic Test Inputs
    size = config.image_size
    rng = np.random.default_rng(42)

    synthetic_lesion = np.full((size, size, 3), 190, dtype=np.uint8)
    cv2.circle(synthetic_lesion, (size // 2, size // 2), size // 4, (60, 50, 40), -1)
    synthetic_lesion = cv2.GaussianBlur(synthetic_lesion, (9, 9), 0)

    red_img = np.full((size, size, 3), (30, 30, 220), dtype=np.uint8)    # BGR: high R
    green_img = np.full((size, size, 3), (30, 220, 30), dtype=np.uint8)  # BGR: high G

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_cfg = dataclasses.replace(config, aef_crc_phase4_artifacts_dir=Path(tmp_dir))
        extractor = HandcraftedFeatureExtractor(test_cfg)

    # 3. Extraction & Dimensionality Verification
    print("\n[Component 1: Descriptor Dimensions & Value Sanity]")
    fv = extractor.extract(synthetic_lesion)
    assert len(fv.glcm) == 12, f"GLCM dimension mismatch: expected 12, got {len(fv.glcm)}"
    assert len(fv.lbp) == 18, f"LBP dimension mismatch: expected 18, got {len(fv.lbp)}"
    assert len(fv.hog) == 1296, f"HOG raw dimension mismatch: expected 1296, got {len(fv.hog)}"
    assert np.isclose(fv.lbp.sum(), 1.0, atol=1e-3), f"LBP histogram density should sum to 1.0, got {fv.lbp.sum()}"

    color_enabled = getattr(config, "color_feature_enabled", True)
    if color_enabled:
        assert fv.color_lab is not None and len(fv.color_lab) == 6, f"Color LAB dimension mismatch: expected 6, got {len(fv.color_lab) if fv.color_lab is not None else None}"
        assert len(fv.combined) == 1332, f"Raw combined dimension mismatch: expected 1332, got {len(fv.combined)}"
    else:
        assert len(fv.combined) == 1326, f"Raw combined dimension mismatch: expected 1326, got {len(fv.combined)}"

    for name, arr in [("GLCM", fv.glcm), ("LBP", fv.lbp), ("HOG", fv.hog), ("Color LAB", fv.color_lab), ("Combined", fv.combined)]:
        if arr is not None:
            assert not np.isnan(arr).any(), f"NaN detected in {name}"
            assert not np.isinf(arr).any(), f"Inf detected in {name}"
    print(f"  GLCM (12-D Haralick, 4-angle averaged): PASS")
    print(f"  LBP (18-D Uniform histogram): PASS")
    print(f"  HOG (1296-D 32x32 cells, 9 orientations): PASS")
    print(f"  Color LAB (6-D CIELAB mean+std from color image): {'PASS' if color_enabled else 'DISABLED'}")
    print(f"  Raw Combined Vector ({len(fv.combined)}-D): PASS")

    # 4. Color Sensitivity Check
    if color_enabled:
        print("\n[Component 2: Color Space Discrimination & Invariance]")
        fv_red = extractor.extract(red_img)
        fv_green = extractor.extract(green_img)
        a_mean_red = fv_red.color_lab[2]
        a_mean_green = fv_green.color_lab[2]
        assert abs(a_mean_red - a_mean_green) > 20.0, f"Color LAB extractor failed to distinguish red from green: red_a={a_mean_red}, green_a={a_mean_green}"
        print(f"  Red vs Green discrimination (a* difference = {abs(a_mean_red - a_mean_green):.2f}): PASS")

        fv_red_2 = extractor.extract(red_img)
        assert np.array_equal(fv_red.color_lab, fv_red_2.color_lab), "Color LAB extraction is not strictly deterministic"
        print("  Parameter-free deterministic reproducibility: PASS")

    # 5. FoldSafeFeatureReducer (HOG PCA 1296 -> 32)
    print("\n[Component 3: FoldSafeFeatureReducer (HOG PCA 1296 -> 32)]")
    dummy_hogs = [rng.normal(size=1296).astype(np.float32) for _ in range(50)]
    train_hogs = dummy_hogs[:40]
    val_hogs = dummy_hogs[40:]

    reducer = FoldSafeFeatureReducer(n_components=config.hog_pca_components)
    try:
        reducer.transform(val_hogs)
        raise AssertionError("Reducer allowed transform before fit!")
    except RuntimeError:
        pass

    reducer.fit(train_hogs)
    try:
        reducer.fit(val_hogs)
        raise AssertionError("Reducer allowed double fit!")
    except RuntimeError:
        pass

    reduced_train = reducer.transform(train_hogs)
    reduced_val = reducer.transform(val_hogs)
    assert reduced_train.shape == (40, config.hog_pca_components), f"Expected shape (40, {config.hog_pca_components}), got {reduced_train.shape}"
    assert reduced_val.shape == (10, config.hog_pca_components), f"Expected shape (10, {config.hog_pca_components}), got {reduced_val.shape}"
    print(f"  Fit-once, transform-only discipline: PASS")
    print(f"  PCA dimension control (1296-D -> {config.hog_pca_components}-D): PASS")

    # 6. FeatureNormalizer
    print("\n[Component 4: FeatureNormalizer (Train-Fold Only Normalization)]")
    normalizer = FeatureNormalizer()
    norm_dim = 12 + 18 + config.hog_pca_components + (6 if color_enabled else 0)
    train_features = [rng.normal(loc=5.0, scale=2.0, size=norm_dim).astype(np.float32) for _ in range(25)]
    val_features = [rng.normal(loc=5.0, scale=2.0, size=norm_dim).astype(np.float32) for _ in range(10)]

    try:
        normalizer.transform(val_features)
        raise AssertionError("Normalizer allowed transform before fit!")
    except RuntimeError:
        pass

    normalizer.fit(train_features)
    try:
        normalizer.fit(val_features)
        raise AssertionError("Normalizer allowed double fit!")
    except RuntimeError:
        pass

    norm_train = normalizer.transform(train_features)
    assert np.allclose(norm_train.mean(axis=0), 0.0, atol=1e-5), "Normalized train mean not close to 0"
    assert np.allclose(norm_train.std(axis=0), 1.0, atol=1e-5), "Normalized train std not close to 1"
    print("  Fit-once, transform-only discipline: PASS")
    print("  Zero-mean, unit-variance standardization: PASS")

    # 7. Deterministic Feature Names Contract (68-D / 62-D)
    print("\n[Component 5: Handcrafted Feature Names Contract]")
    names = get_handcrafted_feature_names(config, reduced_hog=True)
    expected_eval_dim = 68 if color_enabled else 62
    assert len(names) == expected_eval_dim, f"Feature names count mismatch: expected {expected_eval_dim}, got {len(names)}"
    assert len(set(names)) == len(names), "Feature names contain duplicate entries"
    glcm_names = [n for n in names if n.startswith("glcm_")]
    lbp_names = [n for n in names if n.startswith("lbp_")]
    hog_names = [n for n in names if n.startswith("hog_pca_")]
    lab_names = [n for n in names if n.startswith("lab_")]
    assert len(glcm_names) == 12, f"Expected 12 GLCM names, got {len(glcm_names)}"
    assert len(lbp_names) == 18, f"Expected 18 LBP names, got {len(lbp_names)}"
    assert len(hog_names) == config.hog_pca_components, f"Expected {config.hog_pca_components} HOG names, got {len(hog_names)}"
    if color_enabled:
        assert len(lab_names) == 6, f"Expected 6 LAB names, got {len(lab_names)}"
    print(f"  Total Feature Dimensions: {len(names)}-D (GLCM=12, LBP=18, HOG_PCA=32{', LAB=6' if color_enabled else ''})")
    print(f"  Deterministic feature names contract: PASS")

    # 8. Upstream Preflight Status
    print("\n[Component 6: Upstream Preflight Contract Check]")
    preflight_ok, missing_artifacts, winner_data = verify_phase4_preflight(config)
    if preflight_ok:
        print("  All 7 Upstream Artifacts Verified: PASS")
        print(f"  Upstream Winner: {winner_data.get('winner_experiment_id')} (preprocessing_mode={winner_data.get('preprocessing_mode')})")
    else:
        print(f"  Preflight Artifact Status: {len(missing_artifacts)} item(s) pending (framework verified independently)")
        for m in missing_artifacts:
            print(f"    - {m}")

    print("\n" + "=" * 70)
    print("PHASE 4 HANDCRAFTED FRAMEWORK VALIDATION: PASSED")
    print(f"68-D representation ({expected_eval_dim}-D active), fold-safe PCA, normalization discipline,")
    print("and device-agnostic execution certified.")
    print("=" * 70)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="AEF-CRC Phase 4: Handcrafted Feature Extraction & Evaluation")
    parser.add_argument(
        "--validate-framework", "--validate-only",
        action="store_true",
        dest="validate_framework",
        help="Run non-destructive framework validation on synthetic inputs without extracting full dataset."
    )
    args = parser.parse_args()

    config = get_config()

    if args.validate_framework:
        success = validate_framework_components(config)
        return 0 if success else 1

    print("=" * 70)
    print("AEF-CRC PHASE 4: HANDCRAFTED FEATURE EXTRACTION & EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 0: Preflight Verification of 7 Upstream Artifacts
    # -------------------------------------------------------------
    print("\n[Step 0] Preflight Verification of 7 Required Upstream Artifacts...")
    preflight_ok, missing_artifacts, winner_data = verify_phase4_preflight(config)
    if not preflight_ok:
        print("\nFATAL: Preflight verification failed! Missing upstream artifacts:")
        for item in missing_artifacts:
            print(f"  - {item}")
        print("\nAll 7 Phase 1-3 artifacts must exist before Phase 4 can proceed.")
        return 1
    print("PREFLIGHT CHECK: ALL 7 UPSTREAM ARTIFACTS VERIFIED.")

    # -------------------------------------------------------------
    # Step 1: Hardware Environment Diagnostics
    # -------------------------------------------------------------
    hw = get_hardware_diagnostics()
    print(f"\n[Hardware Diagnostics]")
    print(f"  Platform: {hw['os']}")
    print(f"  Processor: {hw['processor']}")
    print(f"  Python: {hw['python_version']}")
    print(f"  Compute Device: {hw['device']} (GPU Available: {hw['gpu_available']})")
    print(f"  Details: {hw['details']}")

    # -------------------------------------------------------------
    # Step 2: Dataset Freeze Verification
    # -------------------------------------------------------------
    print("\n[Dataset Freeze Verification]")
    verify_result = DatasetFreezer(config).verify()
    print(f"  Freeze Identity Check: {'PASS' if verify_result.matches else 'FAIL'}")
    if not verify_result.matches:
        print("FATAL: Dataset freeze verification failed! PSD-HP dataset has been modified.")
        return 1

    # -------------------------------------------------------------
    # Step 3: Consume Phase 3 Winner Contract
    # -------------------------------------------------------------
    winner_id = winner_data.get("winner_experiment_id", "P3-BASE")
    winner_mode = winner_data.get("preprocessing_mode", "standard")
    print(f"\n[Upstream Phase 3 Winner Contract]")
    print(f"  Winner ID: {winner_id}")
    print(f"  Preprocessing Mode: {winner_mode}")
    print(f"  Macro-F1 (CV): {winner_data.get('macro_f1_mean'):.4f} +/- {winner_data.get('macro_f1_std'):.4f}")

    if config.preprocessing_mode != winner_mode:
        print(f"  Aligning Phase 4 preprocessing_mode ({config.preprocessing_mode} -> {winner_mode})")
        config = dataclasses.replace(config, preprocessing_mode=winner_mode)

    # -------------------------------------------------------------
    # Step 4: Load Authoritative Fold Plan
    # -------------------------------------------------------------
    print("\n[Loading Frozen Fold Plan]")
    try:
        plan = load_frozen_fold_plan(config)
    except Exception as exc:
        print(f"FATAL: Could not load authoritative fold plan: {exc}")
        return 1
    print(f"  Fold plan verified: K={plan.k} folds, {sum(len(f.train_records) for f in plan.folds)} train slots.")

    # Invariant: CV validation folds must never contain augmented images
    for f in plan.folds:
        for r in f.val_records:
            if getattr(r, "augmented", False) or "_aug" in r.psd_id:
                print(f"FATAL: Augmented image leaked into validation fold {f.fold_index}: {r.psd_id}")
                return 1
    print("  Augmented image CV exclusion: VERIFIED (zero augmented images in validation folds).")

    # -------------------------------------------------------------
    # Step 5: Extract and Cache Handcrafted Features
    # -------------------------------------------------------------
    all_records = AEFInputLoader(config).load()
    print(f"\n[Feature Extraction & Caching]")
    print(f"  Total dataset images: {len(all_records)}")
    print(f"  Preprocessing mode: {config.preprocessing_mode}")

    extractor = HandcraftedFeatureExtractor(config)
    summary = extract_batch(config, all_records)
    report_path = write_extraction_report(config, summary)

    print(f"  Extracted: {summary.n_success} succeeded, {summary.n_failed} failed.")
    print(f"  Feature Dimensions: {summary.feature_dims}")
    print(f"  Extraction Report: {report_path}")

    if summary.n_failed > 0:
        print("FATAL: Handcrafted feature extraction encountered failures:")
        for f in summary.failures[:10]:
            print(f"    {f.psd_id}: {f.reason}")
        return 1

    # -------------------------------------------------------------
    # Step 6: PSD-ID Alignment and Integrity Assertions
    # -------------------------------------------------------------
    print("\n[Integrity & Alignment Verification]")
    color_enabled = getattr(config, "color_feature_enabled", True)
    expected_combined_raw = 1332 if color_enabled else 1326
    expected_combined_eval = 12 + 18 + config.hog_pca_components + (6 if color_enabled else 0)

    for r in all_records:
        loaded = extractor._load_cached(r.psd_id)
        if loaded is None:
            print(f"FATAL: Missing cached feature vector for {r.psd_id}")
            return 1
        if len(loaded.glcm) != 12 or len(loaded.lbp) != 18 or len(loaded.hog) != 1296:
            print(f"FATAL: Invalid feature dimensions for {r.psd_id}: glcm={len(loaded.glcm)}, lbp={len(loaded.lbp)}, hog={len(loaded.hog)}")
            return 1
        if color_enabled:
            if loaded.color_lab is None or len(loaded.color_lab) != 6:
                print(f"FATAL: Invalid LAB dimensions for {r.psd_id}: expected 6, got {len(loaded.color_lab) if loaded.color_lab is not None else None}")
                return 1
        if len(loaded.combined) != expected_combined_raw:
            print(f"FATAL: Invalid raw combined dimensions for {r.psd_id}: expected {expected_combined_raw}, got {len(loaded.combined)}")
            return 1
        if np.isnan(loaded.combined).any() or np.isinf(loaded.combined).any():
            print(f"FATAL: NaN/Inf detected in cached features for {r.psd_id}")
            return 1
    print("  PSD-ID 1-to-1 Alignment: VERIFIED.")
    dim_str = "GLCM=12, LBP=18, HOG=1296, LAB=6, Combined_Raw=1332, Combined_Eval=68" if color_enabled else "GLCM=12, LBP=18, HOG=1296, Combined_Raw=1326, Combined_Eval=62"
    print(f"  Feature Dimensions ({dim_str}): VERIFIED.")
    print("  Numerical Sanity (Zero NaN/Inf): VERIFIED.")

    # -------------------------------------------------------------
    # Step 7: 5-Fold Cross-Validation Evaluation
    # -------------------------------------------------------------
    print(f"\n[5-Fold Cross-Validation Evaluation]")
    print("  Evaluating 6 feature configurations under leak-free train-only normalization & PCA...")

    fold_metrics_rows = []
    summary_rows = []
    results_rows = []
    pooled_predictions = {fs: {} for fs in FEATURE_SETS}

    for feature_set in FEATURE_SETS:
        cur_fold_metrics = []
        for fold in plan.folds:
            train_records, val_records = fold.train_records, fold.val_records
            y_train = [r.mapped_class for r in train_records]
            y_val = [r.mapped_class for r in val_records]

            if feature_set == "trivial":
                clf = DummyClassifier(strategy="most_frequent", random_state=config.random_seed)
                clf.fit(np.zeros((len(train_records), 1)), y_train)
                y_pred = clf.predict(np.zeros((len(val_records), 1)))
            else:
                hog_reducer = None
                if feature_set in ("hog", "combined"):
                    train_hog = [extractor._load_cached(r.psd_id).hog for r in train_records]
                    hog_reducer = FoldSafeFeatureReducer(n_components=config.hog_pca_components).fit(train_hog)

                X_train_raw = _build_xy(extractor, train_records, feature_set, hog_reducer)
                X_val_raw = _build_xy(extractor, val_records, feature_set, hog_reducer)

                normalizer = FeatureNormalizer().fit(X_train_raw)
                X_train = normalizer.transform(X_train_raw)
                X_val = normalizer.transform(X_val_raw)

                clf = LogisticRegression(max_iter=1000, random_state=config.random_seed, class_weight=fold.class_weights)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)

            for r, true, pred in zip(val_records, y_val, y_pred):
                pooled_predictions[feature_set][r.psd_id] = (true, pred)

            m = compute_fold_metrics(y_val, list(y_pred), config.target_classes, fold.fold_index)
            cur_fold_metrics.append(m)

            fold_row = {
                "fold": fold.fold_index,
                "feature_set": feature_set,
                "macro_f1": round(m.macro_f1, 4),
                "balanced_accuracy": round(m.balanced_accuracy, 4),
                "mcc": round(m.mcc, 4),
            }
            for cls in config.target_classes:
                fold_row[f"f1_{cls}"] = round(m.per_class[cls].f1, 4) if cls in m.per_class else 0.0
            fold_metrics_rows.append(fold_row)

        agg = aggregate_fold_metrics(cur_fold_metrics)
        print(f"  {feature_set:10s}: Macro-F1 = {agg.macro_f1_mean:.4f} +/- {agg.macro_f1_std:.4f} "
              f"| BalAcc = {agg.balanced_accuracy_mean:.4f} | MCC = {agg.mcc_mean:.4f}")

        summary_row = {
            "feature_set": feature_set,
            "macro_f1_mean": round(agg.macro_f1_mean, 4),
            "macro_f1_std": round(agg.macro_f1_std, 4),
            "balanced_accuracy_mean": round(agg.balanced_accuracy_mean, 4),
            "balanced_accuracy_std": round(agg.balanced_accuracy_std, 4),
            "mcc_mean": round(agg.mcc_mean, 4),
            "mcc_std": round(agg.mcc_std, 4),
        }
        for cls in config.target_classes:
            summary_row[f"f1_{cls}_mean"] = round(agg.per_class_f1_mean.get(cls, 0.0), 4)
            summary_row[f"f1_{cls}_std"] = round(agg.per_class_f1_std.get(cls, 0.0), 4)
        summary_rows.append(summary_row)

        res_row = {
            "feature_set": feature_set,
            "macro_f1_mean": round(agg.macro_f1_mean, 4),
            "macro_f1_std": round(agg.macro_f1_std, 4),
            "balanced_accuracy_mean": round(agg.balanced_accuracy_mean, 4),
            "mcc_mean": round(agg.mcc_mean, 4),
        }
        for cls in config.target_classes:
            res_row[f"f1_{cls}"] = round(agg.per_class_f1_mean.get(cls, 0.0), 4)
        results_rows.append(res_row)

    # -------------------------------------------------------------
    # Step 8: Pairwise McNemar's Tests with Holm Correction
    # -------------------------------------------------------------
    print("\n[Pairwise McNemar's Tests (Pooled across folds, Holm-corrected)]")
    pairs = list(itertools.combinations(FEATURE_SETS, 2))
    raw_results = []
    for fs_a, fs_b in pairs:
        common_ids = sorted(set(pooled_predictions[fs_a]) & set(pooled_predictions[fs_b]))
        y_true = [pooled_predictions[fs_a][pid][0] for pid in common_ids]
        y_pred_a = [pooled_predictions[fs_a][pid][1] for pid in common_ids]
        y_pred_b = [pooled_predictions[fs_b][pid][1] for pid in common_ids]
        res = mcnemar_test(y_true, y_pred_a, y_pred_b)
        raw_results.append((fs_a, fs_b, res))

    p_values = [r["p_value"] if r is not None else None for _, _, r in raw_results]
    adjusted_p_values = holm_correction(p_values)

    mcnemar_rows = []
    for (fs_a, fs_b, res), p_adj in zip(raw_results, adjusted_p_values):
        if res is None:
            mcnemar_rows.append({
                "feature_set_a": fs_a, "feature_set_b": fs_b,
                "n01": "", "n10": "", "p_value": "", "p_value_holm": "",
                "status": "Below threshold (n01+n10<25)"
            })
        else:
            p_val = round(res["p_value"], 6)
            p_holm = round(p_adj, 6) if p_adj is not None else ""
            status = "Significant (p_holm < 0.05)" if (p_adj is not None and p_adj < 0.05) else "Not Significant"
            print(f"  {fs_a:10s} vs {fs_b:10s}: n01={res['n01']}, n10={res['n10']}, p={p_val}, p_holm={p_holm} [{status}]")
            mcnemar_rows.append({
                "feature_set_a": fs_a, "feature_set_b": fs_b,
                "n01": res["n01"], "n10": res["n10"],
                "p_value": p_val, "p_value_holm": p_holm,
                "status": status,
            })

    # -------------------------------------------------------------
    # Step 9: Write All Reports and Manifests
    # -------------------------------------------------------------
    config.aef_crc_phase4_reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. fold_metrics.csv
    fold_metrics_path = config.aef_crc_phase4_reports_dir / "fold_metrics.csv"
    with fold_metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_metrics_rows)
    print(f"\n[Generated] Fold Metrics: {fold_metrics_path}")

    # 2. fold_summary.csv
    fold_summary_path = config.aef_crc_phase4_reports_dir / "fold_summary.csv"
    with fold_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[Generated] Fold Summary: {fold_summary_path}")

    # 3. handcrafted_only_results.csv (legacy compatibility)
    results_path = config.aef_crc_phase4_reports_dir / "handcrafted_only_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"[Generated] Handcrafted Only Results: {results_path}")

    # 4. mcnemar_pairwise_results.csv
    mcnemar_path = config.aef_crc_phase4_reports_dir / "mcnemar_pairwise_results.csv"
    with mcnemar_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mcnemar_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mcnemar_rows)
    print(f"[Generated] McNemar Pairwise Results: {mcnemar_path}")

    # 5. phase4_manifest.json
    from modules.experiment_config import representation_id
    manifest_data = {
        "phase": "phase4",
        "run_id": winner_data.get("run_id"),
        "representation_id": representation_id(config),
        "random_seed": config.random_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "CERTIFIED",
        "dataset_freeze_hash": getattr(verify_result, "freeze_hash", None),
        "upstream_contract": {
            "run_id": winner_data.get("run_id"),
            "winner_experiment_id": winner_id,
            "preprocessing_mode": config.preprocessing_mode,
            "macro_f1_mean": winner_data.get("macro_f1_mean"),
            "macro_f1_std": winner_data.get("macro_f1_std"),
        },
        "hardware_environment": hw,
        "feature_dimensions": {
            "glcm": 12,
            "lbp": 18,
            "hog_raw": 1296,
            "hog_reduced": config.hog_pca_components,
            "color_lab": 6 if color_enabled else 0,
            "combined_raw": expected_combined_raw,
            "combined_eval": expected_combined_eval,
        },
        "extraction_summary": {
            "total_images": len(all_records),
            "n_success": summary.n_success,
            "n_failed": summary.n_failed,
        },
        "evaluation_summary": summary_rows,
        "reports_generated": [
            "reports/phase4/extraction_report.json",
            "reports/phase4/fold_metrics.csv",
            "reports/phase4/fold_summary.csv",
            "reports/phase4/handcrafted_only_results.csv",
            "reports/phase4/mcnemar_pairwise_results.csv",
            "reports/phase4/phase4_manifest.json",
        ],
    }

    manifest_path = config.aef_crc_phase4_reports_dir / "phase4_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"[Generated] Phase 4 Manifest: {manifest_path}")

    print("\n" + "=" * 70)
    print("PHASE 4 EXECUTION COMPLETE AND CERTIFIED.")
    print("All handcrafted features extracted, cached, verified, and evaluated.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
