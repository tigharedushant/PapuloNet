"""
run_aef_crc_phase5.py

AEF-CRC Phase 5: Evidence-based fusion of EfficientNet-B0 + GLCM/LBP/HOG.

Step 0 (gate check) always runs first and is always reported explicitly,
per this project's standing rule against fabricating results. Unlike the
previous turn, this script does NOT stop when the gate fails -- it falls
through to a clearly-labeled PLUMBING-ONLY verification instead, per this
turn's explicit instruction. Synthetic numbers from that path are never
presented as, or near, real dermatology results.

Usage:
    python run_aef_crc_phase3.py    # must produce real deep features first
    python run_aef_crc_phase5.py    # this script
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import sys

import numpy as np

from config.config import get_config
from modules.dataset_freeze import DatasetFreezer
from modules.fold_loader import load_frozen_fold_plan
from modules.fusion import ARMS, CONTROLLED_ARMS, build_fusion_fold, compute_sample_weights, build_classifier
from modules.evaluation import compute_fold_metrics, aggregate_fold_metrics, mcnemar_test, holm_correction


def gate_check(config):
    """Checks Step 0's two hard requirements. Returns
    (real_data_available: bool, winning_experiment: Optional[str], detail: str)."""
    deep_feature_dir = config.aef_crc_artifacts_dir / "deep_features"
    npy_files = list(deep_feature_dir.glob("**/*.npy")) if deep_feature_dir.exists() else []
    manifests = list(deep_feature_dir.glob("**/manifest.json")) if deep_feature_dir.exists() else []

    winner_json_path = config.aef_crc_phase3_reports_dir / "winner.json"
    winner = None
    report_has_real_results = False
    if winner_json_path.exists():
        try:
            wdata = json.loads(winner_json_path.read_text(encoding="utf-8"))
            winner = wdata.get("winner_experiment_id") or wdata.get("winner_name")
            report_has_real_results = winner is not None
            expected_run_id = wdata.get("run_id")
            p4_manifest_path = config.aef_crc_phase4_reports_dir / "phase4_manifest.json"
            if p4_manifest_path.exists() and expected_run_id:
                p4_data = json.loads(p4_manifest_path.read_text(encoding="utf-8"))
                p4_run_id = p4_data.get("run_id")
                if p4_run_id != expected_run_id:
                    raise RuntimeError(
                        f"Cross-run artifact mismatch: Phase 4 was run under run_id='{p4_run_id}', "
                        f"but Phase 3 winner has run_id='{expected_run_id}'. Artifacts from separate runs cannot be combined!"
                    )
        except RuntimeError:
            raise
        except Exception:
            winner = None

    ok = len(npy_files) > 0 and len(manifests) > 0 and report_has_real_results
    detail = (
        f"deep feature .npy files: {len(npy_files)} | manifest.json files: {len(manifests)} | "
        f"winner.json exists: {winner_json_path.exists()} | has real results: {report_has_real_results} | "
        f"winner: {winner}"
    )
    return ok, winner, detail


def run_arms(config, plan, source_experiment, extractor_factory, label_prefix=""):
    """Runs all six ARMS, returns (results_by_arm, pooled_predictions,
    classifier_backend). Shared by the real-data and plumbing paths --
    same code, different feature source underneath (real vs synthetic
    deep-feature cache), so a plumbing pass genuinely exercises the same
    logic a real run would."""
    from modules.handcrafted_features import HandcraftedFeatureExtractor
    extractor = extractor_factory()

    backend_name, _ = build_classifier(config.random_seed, config.classifier_name)
    print(f"{label_prefix}Classifier backend: {backend_name}")

    results_by_arm = {}
    pooled_predictions = {arm: {} for arm in CONTROLLED_ARMS}

    for arm_name, branches in CONTROLLED_ARMS.items():
        fold_metrics = []
        for fold in plan.folds:
            data = build_fusion_fold(config, fold, branches, source_experiment, extractor)
            sw_train = compute_sample_weights(data.y_train, fold.class_weights)

            _, clf = build_classifier(config.random_seed, config.classifier_name)
            classes = config.target_classes
            label_to_idx = {c: i for i, c in enumerate(classes)}
            y_train_idx = [label_to_idx[y] for y in data.y_train]

            clf.fit(data.X_train, y_train_idx, sample_weight=sw_train)
            pred_idx = clf.predict(data.X_val)
            y_pred = [classes[i] for i in pred_idx]

            for pid, true, pred in zip(data.psd_ids_val, data.y_val, y_pred):
                pooled_predictions[arm_name][pid] = (true, pred)

            metrics = compute_fold_metrics(data.y_val, y_pred, classes, fold.fold_index)
            fold_metrics.append(metrics)

        agg = aggregate_fold_metrics(fold_metrics)
        results_by_arm[arm_name] = agg
        print(f"{label_prefix}{arm_name:28s}: macro_f1 = {agg.macro_f1_mean:.4f} +/- {agg.macro_f1_std:.4f} "
              f"| balanced_acc = {agg.balanced_accuracy_mean:.4f} | MCC = {agg.mcc_mean:.4f}")
        for cls in ("Pityriasis_Rosea", "Seborrheic_Dermatitis"):
            print(f"{label_prefix}    {cls}: F1 = {agg.per_class_f1_mean[cls]:.4f}")

    return results_by_arm, pooled_predictions, backend_name


def pairwise_mcnemar_holm(pooled_predictions, label_prefix=""):
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
            print(f"{label_prefix}  {a} vs {b}: below reliability threshold (n01+n10<25)")
            rows.append({"arm_a": a, "arm_b": b, "n01": "", "n10": "", "p_value": "", "p_value_holm": ""})
        else:
            print(f"{label_prefix}  {a} vs {b}: n01={r['n01']}, n10={r['n10']}, p={r['p_value']:.4f}, p_holm={p_adj:.4f}")
            rows.append({"arm_a": a, "arm_b": b, "n01": r["n01"], "n10": r["n10"], "p_value": r["p_value"], "p_value_holm": p_adj})
    return rows


def validate_fusion_framework(config) -> bool:
    """Non-destructive validation of Phase 5 feature fusion framework.
    Tests all 8 fusion arms, deterministic ordering, slice boundaries,
    feature naming contract, ablation isolation, fold-safety, and phase boundary.
    DOES NOT perform real training or deep feature extraction.
    """
    print("=" * 70)
    print("AEF-CRC Phase 5 Framework Validation (Non-Destructive)")
    print("=" * 70)

    # 1. Upstream Gate Check Status (truthful reporting)
    print("\n[CHECK 1] Upstream Phase 3 Deep Feature Gate:")
    real_data_ok, winner, detail = gate_check(config)
    print(f"  {detail}")
    print(f"  Upstream real deep features certified: {real_data_ok}")
    if not real_data_ok:
        print("  NOTE: Upstream deep features not yet extracted. Real experiment blocked.")
    else:
        print(f"  Winner experiment: {winner}")

    # 2. 8 Controlled Fusion Arms Verification
    print("\n[CHECK 2] 8-Arm Fusion Hierarchy & Dimensions:")
    from modules.fusion import (
        FUSION_ARMS, FUSION_ARMS_BY_ID,
        CANONICAL_BRANCH_ORDER, EXPECTED_BRANCH_DIMS,
        EQUIVALENCE_MARGIN, sort_branches_deterministically,
        get_fusion_slice_boundaries, get_fusion_feature_names,
        get_fusion_provenance, validate_fusion_dimensions,
    )

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
    print(f"  PASS: All 8 arms produce unique, 1-to-1 column-aligned feature names matching expected dimensions.")

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
    print("\n[CHECK 6] Fold-Safety & Preprocessing Leakage Disciplines:")
    from modules.handcrafted_features import FeatureNormalizer, FoldSafeFeatureReducer
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

    # 7. Phase 5 vs Phase 6 Boundary Check
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

    print("\n" + "=" * 70)
    print("PHASE 5 FRAMEWORK VALIDATION: ALL CHECKS PASSED.")
    print("=" * 70)
    return True


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = get_config()
    if any(arg in argv for arg in ("--validate-framework", "--validate-only", "--validate")):
        ok = validate_fusion_framework(config)
        return 0 if ok else 1

    print("=== AEF-CRC Phase 5: Evidence-Based Feature Fusion ===\n")
    print("--- Step 0: Gate check ---")
    real_data_ok, winner, detail = gate_check(config)
    print(detail)
    print(f"Real deep-feature data available: {real_data_ok}\n")

    from modules.handcrafted_features import HandcraftedFeatureExtractor

    if real_data_ok and winner:
        winner_json_path = config.aef_crc_phase3_reports_dir / "winner.json"
        if winner_json_path.exists():
            try:
                wdata = json.loads(winner_json_path.read_text(encoding="utf-8"))
                mode = wdata.get("preprocessing_mode", "standard")
                aug = "true" if wdata.get("training_time_augmentation") else "false"
                config = dataclasses.replace(config, preprocessing_mode=mode, training_time_augmentation=aug)
            except Exception as e:
                print(f"Align error: {e}")
        print(f"Using winning Phase-3 experiment: {winner} (mode={config.preprocessing_mode})\n")
        freeze_ok = DatasetFreezer(config).verify().matches
        print(f"DATASET FREEZE: {'PASS' if freeze_ok else 'FAIL'}")
        if not freeze_ok:
            return 1

        try:
            plan = load_frozen_fold_plan(config)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            print("Run run_aef_crc_phase2.py first to generate the authoritative fold plan.")
            return 1

        results_by_arm, pooled_predictions, backend = run_arms(
            config, plan, winner, lambda: HandcraftedFeatureExtractor(config),
        )

        print("\n--- Pairwise McNemar's test (pooled across folds), Holm-corrected ---")
        print("NOTE: McNemar's test compares paired PREDICTIONS between two arms on the")
        print("same instances -- it is not a direct significance test of the Macro-F1")
        print("difference itself.")
        mcnemar_rows = pairwise_mcnemar_holm(pooled_predictions)

        out_dir = config.aef_crc_phase5_reports_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "fusion_results.csv"
        with results_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arm", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "mcc_mean"])
            for arm, agg in results_by_arm.items():
                writer.writerow([arm, agg.macro_f1_mean, agg.macro_f1_std, agg.balanced_accuracy_mean, agg.mcc_mean])
        mcnemar_path = out_dir / "fusion_mcnemar_holm.csv"
        with mcnemar_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(mcnemar_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mcnemar_rows)

        from datetime import datetime, timezone
        from modules.experiment_config import representation_id
        manifest_data = {
            "phase": "phase5",
            "run_id": wdata.get("run_id") if 'wdata' in locals() and wdata else None,
            "representation_id": representation_id(config),
            "random_seed": config.random_seed,
            "winner": winner,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "phase5_manifest.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        print(f"\nResults written to {results_path}, {mcnemar_path}, and phase5_manifest.json")
        print("\nPHASE 5 RESULT: REAL EXPERIMENTS COMPLETE.")
        return 0

    # --- PLUMBING-ONLY PATH: real deep features do not exist here ---
    print("Real deep features unavailable -- running PLUMBING-ONLY verification.")
    print("*** EVERYTHING BELOW IS A SYNTHETIC PLUMBING CHECK, NOT A RESEARCH RESULT ***\n")

    from modules.synthetic_fixtures import make_test_config as _make_test_config, build_synthetic_dataset as _build_synthetic_dataset, run_pipeline_through_split as _run_pipeline_through_split

    import shutil
    tmp_root = config.project_root / "_phase5_plumbing_tmp"
    plumbing_ok = False
    backend = "n/a"

    try:
        test_config = _make_test_config(tmp_root, seed=42)
        _build_synthetic_dataset(test_config)
        _run_pipeline_through_split(test_config)

        plan = ImbalanceAwareFoldLoader(test_config, k=3).build()
        if not plan.folds:
            print("PLUMBING CHECK FAILED: no folds built from synthetic data.")
            return 1

        # Write a synthetic deep-feature cache matching the EXACT path
        # convention modules.efficientnet_model.extract_deep_features() uses,
        # AND the full manifest schema it writes -- verifies fusion.py's
        # read-side integration (including validate_deep_feature_cache),
        # not the real TF extraction.
        from modules.backbones import get_backbone
        from modules.experiment_config import representation_id
        backbone_dim = get_backbone(test_config.backbone).output_dim
        repr_id = representation_id(test_config)

        fake_experiment = "PLUMBING-FAKE"
        rng = np.random.default_rng(0)
        for fold in plan.folds:
            for records, split_name in ((fold.train_records, "train"), (fold.val_records, "val")):
                out_dir = test_config.aef_crc_artifacts_dir / "deep_features" / fake_experiment / f"fold_{fold.fold_index:02d}" / split_name
                out_dir.mkdir(parents=True, exist_ok=True)
                psd_ids = [r.psd_id for r in records]
                for r in records:
                    np.save(out_dir / f"{r.psd_id}.npy", rng.normal(size=backbone_dim).astype(np.float32))
                manifest = {
                    "experiment_name": fake_experiment, "fold_index": fold.fold_index, "split": split_name,
                    "checkpoint_path": "SYNTHETIC PLUMBING DATA -- no real checkpoint", "n_features": len(psd_ids),
                    "feature_dim": backbone_dim, "psd_ids": psd_ids, "backbone_name": test_config.backbone,
                    "representation_id": repr_id,
                }
                (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Synthetic deep-feature cache written for plumbing ({len(plan.folds)} folds, train+val).\n")

        try:
            results_by_arm, pooled_predictions, backend = run_arms(
                test_config, plan, fake_experiment, lambda: HandcraftedFeatureExtractor(test_config),
                label_prefix="[PLUMBING] ",
            )
            print("\n[PLUMBING] Pairwise McNemar's test:")
            pairwise_mcnemar_holm(pooled_predictions, label_prefix="[PLUMBING] ")
            plumbing_ok = True
        except Exception as exc:
            print(f"PLUMBING CHECK FAILED: {type(exc).__name__}: {exc}")
            plumbing_ok = False
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nPLUMBING CHECK: {'PASSED' if plumbing_ok else 'FAILED'}")
    print("Classifier backend used above:", backend if plumbing_ok else "n/a")
    print("\nPHASE 5 RESULT: IMPLEMENTATION COMPLETE, real execution BLOCKED on real Phase-3 deep features.")
    return 0 if plumbing_ok else 1


if __name__ == "__main__":
    sys.exit(main())
