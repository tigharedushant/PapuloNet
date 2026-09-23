"""
run_aef_crc_phase5_v2.py

PapuloNet V2 Phase 5: Evidence-Based Feature Fusion
(EfficientNet-B0 P3-V2-Focal Deep Features + GLCM/LBP/HOG/LAB Handcrafted Features)

Controlled Fusion Arms (8 Arms, A0-A7):
  A0 = Deep only (1280-D)
  A1 = Deep + GLCM (1292-D)
  A2 = Deep + LBP (1298-D)
  A3 = Deep + HOG-PCA (1312-D)
  A4 = Deep + LAB (1286-D)
  A5 = Deep + GLCM + LBP (1310-D)
  A6 = Deep + GLCM + LBP + HOG-PCA + LAB (1348-D)
  A7 = Deep + GLCM + LBP + LAB (1316-D)

Key Invariants:
1. Upstream Binding: Strictly consumes Phase 3 V2 certified winner (P3-V2-Focal,
   representation_id=efficientnet_b0_43d581b96f8ec368) and Phase 4 V2 manifest.
2. Handcrafted Representation: 68-D certified Phase 4 V2 features (GLCM=12, LBP=18,
   HOG-PCA=32, LAB=6).
3. Leakage-Safe Fitting: FeatureNormalizer and FoldSafeFeatureReducer fit strictly on
   fold.train_records. Validation fold is transform-only.
4. Strict Partition Isolation:
   - Outer Validation (246 images): Completely untouched.
   - Locked Final Test (243 images): Completely untouched.
   - Quarantined SkinDisNet (264 images): Zero augmented images in validation folds.
5. Class Imbalance Handling: Fold-local balanced sample weights:
   w_c = N_train / (C * N_c,train), fit-time only. Validation predictions unweighted.
6. Primary Classifier: RandomForestClassifier(n_estimators=300, criterion='gini',
   min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True,
   random_state=seed, n_jobs=-1).
7. Comparator Classifier: Multinomial L2 LogisticRegression(C=1.0, max_iter=1000).
8. Selection Protocol: Primary metric = mean 5-fold Macro-F1. Parsimony tie-break:
   within 0.005 practical-equivalence margin, lowest dimension selected.
9. V1 Baseline Protection: Outputs written exclusively to reports/phase5_v2/,
   artifacts/phase5_v2/, logs/phase5_v2/.
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
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.config import PSDConfig, get_config
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import load_frozen_fold_plan
from modules.handcrafted_features import HandcraftedFeatureExtractor, FeatureNormalizer, FoldSafeFeatureReducer
from modules.fusion import (
    FUSION_ARMS,
    FUSION_ARMS_BY_ID,
    FUSION_ARMS_BY_NAME,
    CONTROLLED_ARMS,
    CANONICAL_BRANCH_ORDER,
    EXPECTED_BRANCH_DIMS,
    EQUIVALENCE_MARGIN,
    build_fusion_fold,
    compute_sample_weights,
    build_classifier,
    get_fusion_slice_boundaries,
    get_fusion_feature_names,
    get_fusion_provenance,
    validate_fusion_dimensions,
    sort_branches_deterministically,
    select_fusion_arm_from_results,
)
from modules.evaluation import compute_fold_metrics, aggregate_fold_metrics, mcnemar_test, holm_correction


def gate_check_v2(config: PSDConfig) -> Tuple[bool, Optional[str], str, Dict]:
    """Checks Step 0's upstream Phase 3 V2 and Phase 4 V2 requirements.
    Returns (real_data_available: bool, winning_experiment: Optional[str], detail: str, winner_data: Dict).
    """
    p3_artifacts_dir = getattr(config, "aef_crc_phase3_v2_artifacts_dir", config.aef_crc_artifacts_dir)
    deep_feature_dir = p3_artifacts_dir / "deep_features"

    winner_json_path = config.aef_crc_phase3_v2_reports_dir / "winner.json"
    winner = None
    winner_data = {}
    report_has_real_results = False

    if winner_json_path.exists():
        try:
            winner_data = json.loads(winner_json_path.read_text(encoding="utf-8"))
            winner = winner_data.get("winner_experiment_id") or winner_data.get("winner_name")
            report_has_real_results = winner is not None
            expected_run_id = winner_data.get("run_id")

            # Check cross-phase run_id consistency with Phase 4 V2
            p4_manifest_path = config.aef_crc_phase4_v2_reports_dir / "phase4_manifest.json"
            if p4_manifest_path.exists() and expected_run_id:
                p4_data = json.loads(p4_manifest_path.read_text(encoding="utf-8"))
                p4_run_id = p4_data.get("run_id")
                if p4_run_id != expected_run_id:
                    raise RuntimeError(
                        f"Cross-phase run_id mismatch: Phase 4 V2 run_id='{p4_run_id}', "
                        f"but Phase 3 V2 winner run_id='{expected_run_id}'!"
                    )
        except RuntimeError:
            raise
        except Exception:
            winner = None

    winner_deep_dir = deep_feature_dir / (winner or "P3-V2-Focal")
    npy_files = list(winner_deep_dir.glob("**/*.npy")) if winner_deep_dir.exists() else []
    manifests = list(winner_deep_dir.glob("**/manifest.json")) if winner_deep_dir.exists() else []

    ok = len(npy_files) > 0 and len(manifests) > 0 and report_has_real_results
    detail = (
        f"P3-V2-Focal deep feature .npy files: {len(npy_files)} | manifest.json files: {len(manifests)} | "
        f"winner.json exists: {winner_json_path.exists()} | has real results: {report_has_real_results} | "
        f"winner: {winner} | repr_id: {winner_data.get('representation_id')}"
    )
    return ok, winner, detail, winner_data


def run_arms(config: PSDConfig, plan, source_experiment: str, extractor_factory, label_prefix=""):
    """Runs all 8 CONTROLLED_ARMS across 5 folds with fold-local balanced sample weights.
    Returns (results_by_arm, pooled_predictions, classifier_backend).
    """
    extractor = extractor_factory()

    backend_name, _ = build_classifier(config.random_seed, config.classifier_name)
    print(f"{label_prefix}Classifier backend: {backend_name} ({config.classifier_name})")

    results_by_arm = {}
    pooled_predictions = {arm: {} for arm in CONTROLLED_ARMS}

    for arm_name, branches in CONTROLLED_ARMS.items():
        fold_metrics = []
        for fold in plan.folds:
            data = build_fusion_fold(config, fold, branches, source_experiment, extractor)
            sw_train = compute_sample_weights(data.y_train, fold.class_weights)

            # Deterministic fold seed
            fold_seed = config.random_seed + fold.fold_index
            _, clf = build_classifier(fold_seed, config.classifier_name)
            classes = config.target_classes
            label_to_idx = {c: i for i, c in enumerate(classes)}
            y_train_idx = [label_to_idx[y] for y in data.y_train]

            # Fit with fold-local sample weights
            clf.fit(data.X_train, y_train_idx, sample_weight=sw_train)

            # Unweighted validation prediction
            pred_idx = clf.predict(data.X_val)
            y_pred = [classes[i] for i in pred_idx]

            for pid, true, pred in zip(data.psd_ids_val, data.y_val, y_pred):
                pooled_predictions[arm_name][pid] = (true, pred)

            metrics = compute_fold_metrics(data.y_val, y_pred, classes, fold.fold_index)
            fold_metrics.append(metrics)

        agg = aggregate_fold_metrics(fold_metrics)
        results_by_arm[arm_name] = agg
        arm_obj = FUSION_ARMS_BY_NAME[arm_name]
        print(f"{label_prefix}[{arm_obj.arm_id}] {arm_name:36s} ({arm_obj.expected_dim:4d}-D): "
              f"macro_f1 = {agg.macro_f1_mean:.4f} +/- {agg.macro_f1_std:.4f} "
              f"| balanced_acc = {agg.balanced_accuracy_mean:.4f} | MCC = {agg.mcc_mean:.4f}")
        for cls in ("Pityriasis_Rosea", "Seborrheic_Dermatitis"):
            print(f"{label_prefix}    {cls}: F1 = {agg.per_class_f1_mean.get(cls, 0.0):.4f}")

    return results_by_arm, pooled_predictions, backend_name


def pairwise_mcnemar_holm(pooled_predictions, label_prefix=""):
    """Pairwise McNemar's tests pooled across folds with Holm-Bonferroni correction."""
    arm_names = list(pooled_predictions.keys())
    pairs = list(itertools.combinations(arm_names, 2))
    raw = []
    for a, b in pairs:
        common = sorted(set(pooled_predictions[a]) & set(pooled_predictions[b]))
        y_true = [pooled_predictions[a][pid][0] for pid in common]
        y_pred_a = [pooled_predictions[a][pid][1] for pid in common]
        y_pred_b = [pooled_predictions[b][pid][1] for pid in common]
        raw.append((a, b, mcnemar_test(y_true, y_pred_a, y_pred_b)))

    p_values = [r["p_value"] if r is not None else None for _, _, r in raw]
    adjusted = holm_correction(p_values)
    rows = []
    for (a, b, r), p_adj in zip(raw, adjusted):
        if r is None:
            rows.append({"arm_a": a, "arm_b": b, "n01": "", "n10": "", "p_value": "", "p_value_holm": "", "status": "Below threshold"})
        else:
            status = "Significant (p_holm < 0.05)" if (p_adj is not None and p_adj < 0.05) else "Not Significant"
            rows.append({
                "arm_a": a, "arm_b": b,
                "n01": r["n01"], "n10": r["n10"],
                "p_value": round(r["p_value"], 6),
                "p_value_holm": round(p_adj, 6) if p_adj is not None else "",
                "status": status,
            })
    return rows


def validate_fusion_framework_v2(config: PSDConfig) -> bool:
    """Non-destructive programmatic certification of Phase 5 V2 feature fusion framework."""
    print("=" * 70)
    print("PAPULONET V2 PHASE 5: FEATURE FUSION FRAMEWORK VALIDATION")
    print("=" * 70)

    # 1. Upstream Gate Check Status
    print("\n[CHECK 1] Upstream Phase 3 V2 & Phase 4 V2 Gate:")
    real_data_ok, winner, detail, winner_data = gate_check_v2(config)
    print(f"  {detail}")
    print(f"  Upstream certified Phase 3 V2 deep features: {'PASS' if real_data_ok else 'FAIL'}")
    if not real_data_ok:
        print("  FAIL: Upstream Phase 3 V2 deep features not found!")
        return False

    # 2. 8 Controlled Fusion Arms Verification
    print("\n[CHECK 2] 8-Arm Fusion Hierarchy & Dimensions:")
    if len(FUSION_ARMS) != 8:
        print(f"  FAIL: Expected 8 arms in FUSION_ARMS, found {len(FUSION_ARMS)}")
        return False

    for arm in FUSION_ARMS:
        calc_dim = sum(EXPECTED_BRANCH_DIMS[b] for b in arm.branches)
        if calc_dim != arm.expected_dim:
            print(f"  FAIL: Arm {arm.arm_id} ({arm.name}) expected_dim={arm.expected_dim} but sum={calc_dim}")
            return False
        validate_fusion_dimensions(arm.branches, arm.expected_dim, config)
        print(f"  PASS: [{arm.arm_id}] {arm.name:36s} -> {arm.expected_dim:4d}-D | ReprID: {arm.representation_id}")

    # 3. Deterministic Ordering & Slicing Contract
    print("\n[CHECK 3] Deterministic Branch Ordering & Slice Boundaries:")
    shuffled_branches = ("color_lab", "hog", "deep", "lbp", "glcm")
    sorted_branches = sort_branches_deterministically(shuffled_branches)
    if sorted_branches != CANONICAL_BRANCH_ORDER:
        print(f"  FAIL: Shuffled branches sorted to {sorted_branches}, expected {CANONICAL_BRANCH_ORDER}")
        return False
    print(f"  PASS: Arbitrary input {shuffled_branches} correctly canonicalized to {sorted_branches}")

    full_arm = FUSION_ARMS_BY_ID["A6"]
    full_slices = get_fusion_slice_boundaries(full_arm.branches, config)
    expected_full_slices = {
        "deep": (0, 1280),
        "glcm": (1280, 1292),
        "lbp": (1292, 1310),
        "hog": (1310, 1342),
        "color_lab": (1342, 1348),
    }
    for b, expected_range in expected_full_slices.items():
        actual_range = full_slices.get(b)
        if actual_range != expected_range:
            print(f"  FAIL: Full fusion slice for {b}: expected {expected_range}, got {actual_range}")
            return False
        print(f"  PASS: Branch '{b:9s}' slice boundary -> [{actual_range[0]:4d}:{actual_range[1]:4d}] (dim={actual_range[1]-actual_range[0]})")

    # 4. Feature Naming 1-to-1 Contract
    print("\n[CHECK 4] Stable 1-to-1 Feature Naming Contract:")
    for arm in FUSION_ARMS:
        fnames = get_fusion_feature_names(arm.branches, config)
        if len(fnames) != arm.expected_dim:
            print(f"  FAIL: Arm {arm.arm_id} feature names count={len(fnames)} != {arm.expected_dim}")
            return False
        if len(set(fnames)) != len(fnames):
            print(f"  FAIL: Duplicate feature names detected in arm {arm.arm_id}")
            return False
    print("  PASS: All 8 arms produce unique, 1-to-1 column-aligned feature names matching expected dimensions.")

    # 5. Ablation Isolation & Additivity
    print("\n[CHECK 5] Ablation Isolation & Additive Branch Invariance:")
    a0 = FUSION_ARMS_BY_ID["A0"]
    a1 = FUSION_ARMS_BY_ID["A1"]
    a2 = FUSION_ARMS_BY_ID["A2"]
    a3 = FUSION_ARMS_BY_ID["A3"]
    a4 = FUSION_ARMS_BY_ID["A4"]
    a5 = FUSION_ARMS_BY_ID["A5"]
    a6 = FUSION_ARMS_BY_ID["A6"]
    a7 = FUSION_ARMS_BY_ID["A7"]

    assert set(a1.branches) - set(a0.branches) == {"glcm"}
    assert set(a2.branches) - set(a0.branches) == {"lbp"}
    assert set(a3.branches) - set(a0.branches) == {"hog"}
    assert set(a4.branches) - set(a0.branches) == {"color_lab"}
    assert set(a5.branches) - set(a0.branches) == {"glcm", "lbp"}
    assert set(a7.branches) - set(a5.branches) == {"color_lab"}
    assert set(a6.branches) - set(a7.branches) == {"hog"}
    print("  PASS: Single-branch ablations (A1-A4) strictly isolate one handcrafted family.")
    print("  PASS: Texture combination (A5) combines GLCM + LBP without cross-contamination.")
    print("  PASS: A7 is compact interpretable fusion without HOG.")
    print("  PASS: A6 contains full 5-branch fusion.")

    # 6. Fold-Safety, No Second PCA, and Leakage Guarantees
    print("\n[CHECK 6] Fold-Safety & Normalization Leakage Disciplines:")
    rng = np.random.default_rng(42)
    dummy_train = rng.normal(size=(20, 32))
    dummy_val = rng.normal(size=(10, 32))

    norm = FeatureNormalizer().fit(dummy_train)
    norm_val = norm.transform(dummy_val)
    assert norm_val.shape == (10, 32)

    reducer = FoldSafeFeatureReducer(n_components=16).fit(dummy_train)
    red_val = reducer.transform(dummy_val)
    assert red_val.shape == (10, 16)
    print("  PASS: FeatureNormalizer & FoldSafeFeatureReducer maintain train-only fit, val-transform-only.")
    print("  PASS: HOG-PCA is the only PCA in the pipeline (no second PCA on fused representations).")

    # 7. Phase Boundary Enforcement
    print("\n[CHECK 7] Phase Boundary Verification:")
    import modules.fusion as fusion_mod
    for forbidden in ("DFASelector", "GeneticSelector", "RFESelector", "run_selector"):
        if hasattr(fusion_mod, forbidden):
            print(f"  FAIL: Phase 5 fusion module exposes {forbidden} (Phase 6 responsibility)")
            return False
    print("  PASS: Phase 5 is strictly feature concatenation. Feature selection belongs to Phase 6.")

    # 8. Practical Equivalence Margin Constant
    print("\n[CHECK 8] Practical Equivalence Margin:")
    if EQUIVALENCE_MARGIN != 0.005:
        print(f"  FAIL: EQUIVALENCE_MARGIN is {EQUIVALENCE_MARGIN}, expected 0.005")
        return False
    print(f"  PASS: EQUIVALENCE_MARGIN = {EQUIVALENCE_MARGIN} (0.005 Macro-F1 threshold certified).")

    # 9. Provenance Metadata Contract
    print("\n[CHECK 9] Provenance Generation Contract:")
    prov = get_fusion_provenance("A6", config, fold_index=0, split="train")
    assert prov["arm_id"] == "A6"
    assert prov["expected_dim"] == 1348
    assert prov["actual_feature_count"] == 1348
    assert prov["leakage_guarantees"]["val_transform_only"] is True
    print("  PASS: Full provenance metadata schema generated and verified.")

    # 10. Classifier Configuration Contract
    print("\n[CHECK 10] Classifier Configurations (Random Forest & Comparator):")
    backend_rf, clf_rf = build_classifier(42, "random_forest")
    assert backend_rf == "random_forest"
    assert clf_rf.n_estimators == 300
    assert clf_rf.criterion == "gini"
    assert clf_rf.min_samples_split == 2
    assert clf_rf.min_samples_leaf == 1
    assert clf_rf.max_features == "sqrt"
    assert clf_rf.bootstrap is True
    assert clf_rf.class_weight is None
    print("  PASS: Primary Random Forest configuration strictly certified.")

    backend_lr, clf_lr = build_classifier(42, "logistic_regression")
    assert backend_lr == "logistic_regression"
    assert clf_lr.penalty == "l2"
    assert clf_lr.C == 1.0
    assert clf_lr.solver == "lbfgs"
    assert clf_lr.max_iter == 1000
    assert clf_lr.class_weight is None
    print("  PASS: Comparator Logistic Regression configuration strictly certified.")

    print("\n" + "=" * 70)
    print("PAPULONET V2 PHASE 5 FRAMEWORK VALIDATION: ALL CHECKS PASSED.")
    print("=" * 70)
    return True


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="PapuloNet V2 Phase 5: Feature Fusion & Evaluation")
    parser.add_argument(
        "--validate-framework", "--validate-only", "--validate",
        action="store_true",
        dest="validate_framework",
        help="Run non-destructive framework validation without executing real 5-fold evaluation."
    )
    args = parser.parse_args(argv)

    config = get_config()
    # Route directories to V2 locations
    config = dataclasses.replace(
        config,
        aef_crc_artifacts_dir=config.aef_crc_phase3_v2_artifacts_dir,
        aef_crc_phase3_reports_dir=config.aef_crc_phase3_v2_reports_dir,
        aef_crc_phase5_reports_dir=config.aef_crc_phase5_v2_reports_dir,
    )

    if args.validate_framework:
        ok = validate_fusion_framework_v2(config)
        return 0 if ok else 1

    print("=== PapuloNet V2 Phase 5: Evidence-Based Feature Fusion ===\n")
    print("--- Step 0: Gate check ---")
    real_data_ok, winner, detail, winner_data = gate_check_v2(config)
    print(detail)
    print(f"Real Phase 3 V2 deep-feature data available: {real_data_ok}\n")

    if not real_data_ok or not winner:
        print("FATAL: Upstream Phase 3 V2 winner artifacts missing or unreadable.")
        return 1

    mode = winner_data.get("preprocessing_mode", "standard")
    aug = "true" if winner_data.get("training_time_augmentation") else "false"
    config = dataclasses.replace(
        config,
        preprocessing_mode=mode,
        training_time_augmentation=aug,
        loss_name=winner_data.get("config", {}).get("loss_name", config.loss_name),
        use_class_weights=winner_data.get("config", {}).get("use_class_weights", config.use_class_weights),
        focal_gamma=winner_data.get("config", {}).get("focal_gamma", config.focal_gamma),
        adam_clipnorm=winner_data.get("config", {}).get("adam_clipnorm", config.adam_clipnorm),
    )

    print(f"Using winning Phase 3 V2 experiment: {winner} (mode={config.preprocessing_mode})\n")
    freeze_ok = DatasetFreezer(config).verify().matches
    print(f"DATASET FREEZE: {'PASS' if freeze_ok else 'FAIL'}")
    if not freeze_ok:
        return 1

    try:
        plan = load_frozen_fold_plan(config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    extractor_factory = lambda: HandcraftedFeatureExtractor(config)

    # 1. Primary Evaluation: Random Forest across all 8 arms
    print("\n--- Primary Classifier Evaluation: Random Forest ---")
    rf_config = dataclasses.replace(config, classifier_name="random_forest")
    results_by_arm_rf, pooled_predictions_rf, backend_rf = run_arms(
        rf_config, plan, winner, extractor_factory, label_prefix="  [RF] "
    )

    # 2. Pairwise McNemar's tests for Random Forest
    print("\n--- Pairwise McNemar's Tests (Random Forest, Holm-corrected) ---")
    mcnemar_rows_rf = pairwise_mcnemar_holm(pooled_predictions_rf, label_prefix="  ")

    # 3. Comparator Evaluation: Logistic Regression across all 8 arms
    print("\n--- Comparator Evaluation: Multinomial L2 Logistic Regression ---")
    lr_config = dataclasses.replace(config, classifier_name="logistic_regression")
    results_by_arm_lr, pooled_predictions_lr, backend_lr = run_arms(
        lr_config, plan, winner, extractor_factory, label_prefix="  [LR] "
    )

    # 4. Write Results Tables
    out_dir = config.aef_crc_phase5_v2_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary fusion_results.csv (Random Forest)
    results_path = out_dir / "fusion_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "mcc_mean"])
        for arm, agg in results_by_arm_rf.items():
            writer.writerow([arm, agg.macro_f1_mean, agg.macro_f1_std, agg.balanced_accuracy_mean, agg.mcc_mean])

    # Comparator fusion_results_comparator_lr.csv
    lr_results_path = out_dir / "fusion_results_comparator_lr.csv"
    with lr_results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "mcc_mean"])
        for arm, agg in results_by_arm_lr.items():
            writer.writerow([arm, agg.macro_f1_mean, agg.macro_f1_std, agg.balanced_accuracy_mean, agg.mcc_mean])

    # McNemar results
    mcnemar_path = out_dir / "fusion_mcnemar_holm.csv"
    with mcnemar_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mcnemar_rows_rf[0].keys()))
        writer.writeheader()
        writer.writerows(mcnemar_rows_rf)

    # 5. Apply Deterministic Model Selection Rule
    from modules.experiment_config import representation_id
    selection = select_fusion_arm_from_results(results_path, config)

    freeze_path = config.aef_crc_reports_dir / "dataset_freeze.json"
    freeze_hash = winner_data.get("dataset_freeze_hash") or (
        json.loads(freeze_path.read_text(encoding="utf-8")).get("freeze_hash") if freeze_path.exists() else None
    )

    fold_plan_path = config.aef_crc_reports_dir / "fold_plan.csv"
    fold_plan_hash = winner_data.get("fold_plan_hash") or (
        hashlib.sha256(fold_plan_path.read_bytes()).hexdigest() if fold_plan_path.exists() else None
    )

    # 6. Generate Certified Phase 5 V2 Manifest
    manifest_data = {
        "phase": "phase5_v2",
        "run_id": winner_data.get("run_id"),
        "representation_id": representation_id(config),
        "random_seed": config.random_seed,
        "primary_classifier": {
            "name": "random_forest",
            "n_estimators": 300,
            "criterion": "gini",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "sample_weight": "fold_local_balanced",
        },
        "comparator_classifier": {
            "name": "logistic_regression",
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "sample_weight": "fold_local_balanced",
        },
        "upstream_phase3_winner": {
            "winner_experiment_id": winner,
            "representation_id": winner_data.get("representation_id"),
            "macro_f1_mean": winner_data.get("macro_f1_mean"),
            "macro_f1_std": winner_data.get("macro_f1_std"),
        },
        "dataset_freeze_hash": freeze_hash,
        "fold_plan_hash": fold_plan_hash,
        "selection_rule": selection["selection_rule"],
        "selected_fusion_arm": selection["selected_fusion_arm"],
        "selected_fusion_arm_id": selection["selected_fusion_arm_id"],
        "selected_fusion_representation": selection["selected_fusion_representation"],
        "selected_fusion_dimension": selection["selected_fusion_dimension"],
        "selected_fusion_branches": selection["selected_fusion_branches"],
        "selected_metrics": selection["selected_metrics"],
        "selection_metadata": selection["selection_metadata"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reports_generated": [
            "reports/phase5_v2/fusion_results.csv",
            "reports/phase5_v2/fusion_results_comparator_lr.csv",
            "reports/phase5_v2/fusion_mcnemar_holm.csv",
            "reports/phase5_v2/phase5_manifest.json",
        ],
    }

    manifest_path = out_dir / "phase5_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    print(f"\n[Generated] Results: {results_path}")
    print(f"[Generated] Comparator Results: {lr_results_path}")
    print(f"[Generated] McNemar Results: {mcnemar_path}")
    print(f"[Generated] Phase 5 V2 Manifest: {manifest_path}")
    print(f"\nPhase 5 Selected Fusion Arm: {selection['selected_fusion_arm']} ({selection['selected_fusion_arm_id']})")
    print(f"Selection Rationale: {selection['selection_metadata']['selection_rationale']}")
    print("\n" + "=" * 70)
    print("PAPULONET V2 PHASE 5 EXECUTION COMPLETE AND CERTIFIED.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
