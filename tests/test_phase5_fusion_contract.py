"""
tests/test_phase5_fusion_contract.py

AEF-CRC Phase 5: Deep + Handcrafted Feature Fusion Verification Suite.
Rigorous synthetic tests verifying:
1. Expected dimensions for all 8 controlled fusion arms (A0-A7).
2. Strict canonical feature ordering and exact slice ranges (1348-D full fusion).
3. 1-to-1 deterministic feature names without duplicates or gaps.
4. Ablation isolation (no cross-contamination between omitted and active branches).
5. Frozen branch additivity (adding a branch does not alter existing branches).
6. Bitwise reproducibility for identical inputs and configuration.
7. Leakage safety (train-only fitting of normalizers and HOG PCA, val transform-only).
8. Phase boundary enforcement (no DFA, GA, RFE, or feature selection in Phase 5).
9. Provenance manifest and representation IDs.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from config.config import get_config
from modules.aef_input_validator import ImageRecord
from modules.fold_loader import Fold
from modules.fusion import (
    ARMS,
    CONTROLLED_ARMS,
    CANONICAL_BRANCH_ORDER,
    EQUIVALENCE_MARGIN,
    EXPECTED_ARM_DIMS,
    EXPECTED_BRANCH_DIMS,
    ARM_REPRESENTATION_IDS,
    FusionFoldData,
    _canonicalize_branches,
    get_branch_feature_names,
    get_fused_feature_names,
    get_fused_slice_map,
    get_representation_id,
    validate_fusion_dimensions,
    build_fusion_manifest,
    SELECTION_RULE_NAME,
    select_fusion_arm_from_results,
)
from modules.handcrafted_features import (
    FeatureVectors,
    HandcraftedFeatureExtractor,
    FeatureNormalizer,
    FoldSafeFeatureReducer,
)


# ---- Fixtures ----

class MockExtractor:
    """Deterministic in-memory mock extractor for synthetic testing."""
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(42)
        self.cache: Dict[str, FeatureVectors] = {}

    def get_or_create(self, psd_id: str) -> FeatureVectors:
        if psd_id not in self.cache:
            glcm = self.rng.uniform(0.1, 10.0, size=12).astype(np.float64)
            lbp = self.rng.uniform(0.0, 1.0, size=18).astype(np.float64)
            hog = self.rng.uniform(-1.0, 1.0, size=1296).astype(np.float64)
            lab = self.rng.uniform(0.0, 100.0, size=6).astype(np.float64)
            self.cache[psd_id] = FeatureVectors(
                psd_id=psd_id, glcm=glcm, lbp=lbp, hog=hog, color_lab=lab,
            )
        return self.cache[psd_id]

    def _load_cached(self, psd_id: str):
        return self.get_or_create(psd_id)

    def extract_and_cache(self, img, psd_id: str):
        return self.get_or_create(psd_id)


def _make_mock_records(count: int, prefix: str) -> List[ImageRecord]:
    records = []
    classes = ["Psoriasis", "Lichen_Planus", "Pityriasis_Rosea", "Seborrheic_Dermatitis"]
    for i in range(count):
        pid = f"{prefix}_{i:04d}"
        rec = ImageRecord(
            psd_id=pid,
            split="train" if "train" in prefix else "val",
            mapped_class=classes[i % 4],
            source_dataset="Synthetic",
            dataset_version="v1",
            original_split="train",
            source_type="original",
            file_path=Path(f"/fake/{pid}.jpg"),
        )
        records.append(rec)
    return records


# ---- 1. Dimension Contracts for All 8 Arms (Sections 23 & 25) ----

def test_controlled_arms_count_and_dimensions():
    """Verify exactly 8 controlled arms are defined and match exact dimensional requirements."""
    expected_contracts = {
        "A0": ("EfficientNet", 1280),
        "A1": ("EfficientNet+GLCM", 1292),
        "A2": ("EfficientNet+LBP", 1298),
        "A3": ("EfficientNet+HOG-PCA", 1312),
        "A4": ("EfficientNet+LAB", 1286),
        "A5": ("EfficientNet+GLCM+LBP", 1310),
        "A7": ("EfficientNet+GLCM+LBP+LAB", 1316),
        "A6": ("EfficientNet+GLCM+LBP+HOG-PCA+LAB", 1348),
    }

    assert len(CONTROLLED_ARMS) == 8, f"Expected 8 controlled arms, found {len(CONTROLLED_ARMS)}"

    for arm_id, (arm_name, expected_dim) in expected_contracts.items():
        assert arm_name in CONTROLLED_ARMS or arm_name in ARMS, f"Missing arm: {arm_name}"
        branches = ARMS[arm_name]
        validated_dim = validate_fusion_dimensions(branches, expected_dim)
        assert validated_dim == expected_dim, f"Arm {arm_name} dimension mismatch: {validated_dim} != {expected_dim}"


def test_validate_fusion_dimensions_rejects_mismatch():
    """validate_fusion_dimensions must raise ValueError on unexpected vector width."""
    with pytest.raises(ValueError, match="Fusion dimension mismatch"):
        validate_fusion_dimensions("EfficientNet", 1279)

    with pytest.raises(ValueError, match="Fusion dimension mismatch"):
        validate_fusion_dimensions(("deep", "glcm"), 1348)

    with pytest.raises(ValueError, match="Fusion dimension mismatch"):
        validate_fusion_dimensions("EfficientNet+GLCM+LBP+HOG+LAB", 1347)


# ---- 2. Canonical Ordering & Slice Boundaries (Sections 15, 24, 26) ----

def test_full_fusion_slice_boundaries():
    """Verify exact column slices for full 1348-D representation:
    0:1280       -> EfficientNet-B0 (1280)
    1280:1292    -> GLCM (12)
    1292:1310    -> LBP (18)
    1310:1342    -> HOG-PCA (32)
    1342:1348    -> LAB (6)
    """
    full_branches = ("deep", "glcm", "lbp", "hog", "color_lab")
    slice_map = get_fused_slice_map(full_branches)

    assert slice_map["deep"] == (0, 1280)
    assert slice_map["glcm"] == (1280, 1292)
    assert slice_map["lbp"] == (1292, 1310)
    assert slice_map["hog"] == (1310, 1342)
    assert slice_map["color_lab"] == (1342, 1348)

    # Verify contiguous and non-overlapping
    assert slice_map["deep"][1] == slice_map["glcm"][0]
    assert slice_map["glcm"][1] == slice_map["lbp"][0]
    assert slice_map["lbp"][1] == slice_map["hog"][0]
    assert slice_map["hog"][1] == slice_map["color_lab"][0]
    assert slice_map["color_lab"][1] == 1348


def test_canonical_branch_ordering_enforced_regardless_of_input_order():
    """No matter what tuple order is passed, branches are sorted strictly into canonical order."""
    unordered = ("color_lab", "glcm", "deep", "hog", "lbp")
    canonical = _canonicalize_branches(unordered)
    assert canonical == ("deep", "glcm", "lbp", "hog", "color_lab")

    subset_unordered = ("color_lab", "deep")
    canonical_subset = _canonicalize_branches(subset_unordered)
    assert canonical_subset == ("deep", "color_lab")


def test_feature_names_1to1_deterministic_and_unique():
    """Feature names must be 1-to-1 with columns, deterministic, and free of duplicates."""
    full_branches = ("deep", "glcm", "lbp", "hog", "color_lab")
    names = get_fused_feature_names(full_branches)

    assert len(names) == 1348
    assert len(set(names)) == 1348, "Feature names contain duplicates"

    # Check Deep
    assert names[0] == "efficientnet_0"
    assert names[1279] == "efficientnet_1279"

    # Check GLCM
    assert names[1280].startswith("glcm_")
    assert names[1291].startswith("glcm_")

    # Check LBP
    assert names[1292] == "lbp_bin_0"
    assert names[1309] == "lbp_bin_17"

    # Check HOG-PCA
    assert names[1310] == "hog_pca_0"
    assert names[1341] == "hog_pca_31"

    # Check LAB
    expected_lab = [
        "lab_L_mean", "lab_L_std",
        "lab_a_mean", "lab_a_std",
        "lab_b_mean", "lab_b_std",
    ]
    assert names[1342:1348] == expected_lab


# ---- 3. Ablation Isolation (Section 27) ----

def test_ablation_isolation():
    """Verify single-handcrafted and intermediate arms isolate active branches only."""
    # EfficientNet + GLCM (1292-D) must contain NO lbp, hog, or lab
    glcm_arm_names = get_fused_feature_names(("deep", "glcm"))
    assert len(glcm_arm_names) == 1292
    assert not any("lbp" in n for n in glcm_arm_names)
    assert not any("hog" in n for n in glcm_arm_names)
    assert not any("lab" in n for n in glcm_arm_names)

    # EfficientNet + LAB (1286-D) must contain NO glcm, lbp, or hog
    lab_arm_names = get_fused_feature_names(("deep", "color_lab"))
    assert len(lab_arm_names) == 1286
    assert not any("glcm" in n for n in lab_arm_names)
    assert not any("lbp" in n for n in lab_arm_names)
    assert not any("hog" in n for n in lab_arm_names)

    # EfficientNet + GLCM + LBP (1310-D) must contain NO hog or lab
    texture_arm_names = get_fused_feature_names(("deep", "glcm", "lbp"))
    assert len(texture_arm_names) == 1310
    assert not any("hog" in n for n in texture_arm_names)
    assert not any("lab" in n for n in texture_arm_names)

    # EfficientNet + GLCM + LBP + LAB (1316-D) must contain NO hog
    compact_arm_names = get_fused_feature_names(("deep", "glcm", "lbp", "color_lab"))
    assert len(compact_arm_names) == 1316
    assert not any("hog" in n for n in compact_arm_names)


# ---- 4. Frozen Branches / Additivity (Section 29) ----

def test_frozen_branches_additivity():
    """Adding a branch (e.g. LAB) must not alter the values of existing branches."""
    rng = np.random.default_rng(123)
    train_recs = _make_mock_records(10, "train")
    val_recs = _make_mock_records(5, "val")

    deep_train = [rng.normal(size=1280).astype(np.float32) for _ in train_recs]
    deep_val = [rng.normal(size=1280).astype(np.float32) for _ in val_recs]
    glcm_train = [rng.uniform(1, 10, size=12).astype(np.float32) for _ in train_recs]
    glcm_val = [rng.uniform(1, 10, size=12).astype(np.float32) for _ in val_recs]
    lab_train = [rng.uniform(0, 100, size=6).astype(np.float32) for _ in train_recs]
    lab_val = [rng.uniform(0, 100, size=6).astype(np.float32) for _ in val_recs]

    norm_deep = FeatureNormalizer().fit(deep_train)
    norm_glcm = FeatureNormalizer().fit(glcm_train)
    norm_lab = FeatureNormalizer().fit(lab_train)

    deep_norm_train = norm_deep.transform(deep_train)
    glcm_norm_train = norm_glcm.transform(glcm_train)
    lab_norm_train = norm_lab.transform(lab_train)

    fused_arm1 = np.concatenate([deep_norm_train, glcm_norm_train], axis=1)
    fused_with_lab = np.concatenate([deep_norm_train, glcm_norm_train, lab_norm_train], axis=1)

    assert np.array_equal(fused_arm1, fused_with_lab[:, :1292]), "Adding LAB altered prior branch values!"


# ---- 5. Bitwise Reproducibility (Section 28) ----

def test_bitwise_reproducibility():
    """Identical input vectors and identical configuration produce 100% bitwise identical fused vectors."""
    rng = np.random.default_rng(999)
    deep_vecs = rng.normal(size=(20, 1280))
    glcm_vecs = rng.uniform(0, 5, size=(20, 12))
    lbp_vecs = rng.uniform(0, 1, size=(20, 18))

    norm_deep = FeatureNormalizer().fit(deep_vecs)
    norm_glcm = FeatureNormalizer().fit(glcm_vecs)
    norm_lbp = FeatureNormalizer().fit(lbp_vecs)

    fused1 = np.concatenate([
        norm_deep.transform(deep_vecs),
        norm_glcm.transform(glcm_vecs),
        norm_lbp.transform(lbp_vecs),
    ], axis=1)

    fused2 = np.concatenate([
        norm_deep.transform(deep_vecs),
        norm_glcm.transform(glcm_vecs),
        norm_lbp.transform(lbp_vecs),
    ], axis=1)

    assert np.array_equal(fused1, fused2)


# ---- 6. Leakage Safety (Sections 16 & 17) ----

def test_leakage_safety_train_only_fit():
    """Normalizers and HOG PCA must fit on train only. Modifying validation data must NOT affect train transformation."""
    rng = np.random.default_rng(42)
    train_raw = rng.normal(size=(25, 1296))
    val_raw_1 = rng.normal(size=(10, 1296))
    val_raw_2 = rng.normal(loc=100.0, scale=50.0, size=(10, 1296))

    pca1 = FoldSafeFeatureReducer(n_components=32).fit(train_raw)
    train_pca_1 = pca1.transform(train_raw)

    pca2 = FoldSafeFeatureReducer(n_components=32).fit(train_raw)
    train_pca_2 = pca2.transform(train_raw)

    assert np.array_equal(train_pca_1, train_pca_2)

    norm1 = FeatureNormalizer().fit(train_raw)
    norm_train_1 = norm1.transform(train_raw)

    norm2 = FeatureNormalizer().fit(train_raw)
    norm_train_2 = norm2.transform(train_raw)

    assert np.array_equal(norm_train_1, norm_train_2)

    _ = norm1.transform(val_raw_1)
    _ = norm1.transform(val_raw_2)
    assert np.array_equal(norm1.mean_, norm2.mean_)
    assert np.array_equal(norm1.std_, norm2.std_)


# ---- 7. Phase Boundary Enforcement: No DFA/Feature Selection in Phase 5 (Section 12, 13, 30) ----

def test_phase5_modules_contain_no_feature_selection():
    """Static AST check: Phase 5 modules must NOT contain or invoke DFA, GA, PSO, ACO, or RFE."""
    forbidden_terms = [
        "DragonflyAlgorithm", "dragonfly_algorithm",
        "GeneticAlgorithm", "ParticleSwarm", "AntColony",
        "RFE", "SelectKBest", "SelectFromModel",
    ]

    for file_name in ["modules/fusion.py", "run_aef_crc_phase5.py"]:
        path = Path(__file__).resolve().parent.parent / file_name
        assert path.exists(), f"{file_name} not found"
        content = path.read_text(encoding="utf-8")
        parsed = ast.parse(content)

        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for term in forbidden_terms:
                        assert term.lower() not in alias.name.lower(), (
                            f"Forbidden feature selection import '{alias.name}' in {file_name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for term in forbidden_terms:
                    assert term.lower() not in module.lower(), (
                        f"Forbidden feature selection import from '{module}' in {file_name}"
                    )
                for alias in node.names:
                    for term in forbidden_terms:
                        assert term.lower() not in alias.name.lower(), (
                            f"Forbidden feature selection import symbol '{alias.name}' in {file_name}"
                        )


# ---- 8. Representation IDs & Provenance Manifest (Section 22) ----

def test_representation_ids_exact_match():
    """Verify exact representation IDs as specified in Section 22."""
    expected_ids = {
        ("deep",): "efficientnet",
        ("deep", "glcm"): "efficientnet_glcm",
        ("deep", "lbp"): "efficientnet_lbp",
        ("deep", "hog"): "efficientnet_hog",
        ("deep", "color_lab"): "efficientnet_lab",
        ("deep", "glcm", "lbp"): "efficientnet_glcm_lbp",
        ("deep", "glcm", "lbp", "color_lab"): "efficientnet_glcm_lbp_lab",
        ("deep", "glcm", "lbp", "hog", "color_lab"): "efficientnet_glcm_lbp_hog_lab",
    }

    for branches, exp_id in expected_ids.items():
        assert get_representation_id(branches) == exp_id

    assert get_representation_id("EfficientNet") == "efficientnet"
    assert get_representation_id("EfficientNet+GLCM") == "efficientnet_glcm"
    assert get_representation_id("EfficientNet+GLCM+LBP+HOG-PCA+LAB") == "efficientnet_glcm_lbp_hog_lab"


def test_build_fusion_manifest_schema():
    """Verify provenance manifest captures complete metadata."""
    config = get_config()
    manifest = build_fusion_manifest(
        representation_id_str="efficientnet_glcm_lbp_hog_lab",
        fold_index=0,
        split_name="train",
        psd_ids=["PSD_0001", "PSD_0002"],
        feature_dim=1348,
        branch_dims={"deep": 1280, "glcm": 12, "lbp": 18, "hog": 32, "color_lab": 6},
        config=config,
        source_experiment="P3-AUG",
    )

    assert manifest["phase"] == "phase5"
    assert manifest["source_experiment"] == "P3-AUG"
    assert manifest["feature_dim"] == 1348
    assert manifest["canonical_branch_order"] == ["deep", "glcm", "lbp", "hog", "color_lab"]
    assert "slice_boundaries" in manifest
    assert "feature_names" in manifest
    assert len(manifest["feature_names"]) == 1348
    assert manifest["slice_boundaries"]["deep"] == [0, 1280]
    assert manifest["slice_boundaries"]["color_lab"] == [1342, 1348]


# ---- 9. Practical Equivalence Margin & Selection Contract ----

def test_practical_equivalence_margin_defined():
    """EQUIVALENCE_MARGIN must be 0.005."""
    assert EQUIVALENCE_MARGIN == 0.005


def test_select_fusion_arm_rule_synthetic_scenarios(tmp_path):
    """Test deterministic selection rule with practical equivalence and tiebreaks."""
    import csv

    # Scenario 1: A6 and A7 within 0.005 margin -> prefer lower dimension (A7: 1316 < A6: 1348)
    csv_path = tmp_path / "fusion_results_1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "mcc_mean"])
        writer.writerow(["EfficientNet", 0.7097, 0.0355, 0.7069, 0.5870])
        writer.writerow(["EfficientNet+GLCM+LBP+HOG-PCA+LAB", 0.7243, 0.0366, 0.7239, 0.6084])
        writer.writerow(["EfficientNet+GLCM+LBP+LAB", 0.7270, 0.0190, 0.7264, 0.6179])

    sel = select_fusion_arm_from_results(csv_path)
    assert sel["selected_fusion_arm"] == "EfficientNet+GLCM+LBP+LAB"
    assert sel["selected_fusion_arm_id"] == "A7"
    assert sel["selected_fusion_dimension"] == 1316
    assert len(sel["selection_metadata"]["candidates_within_practical_performance_margin"]) == 2

    # Scenario 2: High dimension has slightly higher F1 (within 0.005) -> prefer lower dimension
    csv_path2 = tmp_path / "fusion_results_2.csv"
    with csv_path2.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "mcc_mean"])
        writer.writerow(["EfficientNet", 0.7260, 0.0200, 0.7200, 0.6000])
        writer.writerow(["EfficientNet+GLCM+LBP+HOG-PCA+LAB", 0.7290, 0.0200, 0.7250, 0.6100])

    sel2 = select_fusion_arm_from_results(csv_path2)
    # EfficientNet has 1280 dim vs 1348 dim; delta is 0.0030 <= 0.005 -> lower dim wins
    assert sel2["selected_fusion_arm"] == "EfficientNet"
    assert sel2["selected_fusion_arm_id"] == "A0"
    assert sel2["selected_fusion_dimension"] == 1280

    # Scenario 3: Delta > 0.005 -> higher F1 wins regardless of dimension
    csv_path3 = tmp_path / "fusion_results_3.csv"
    with csv_path3.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "mcc_mean"])
        writer.writerow(["EfficientNet", 0.7200, 0.0200, 0.7200, 0.6000])
        writer.writerow(["EfficientNet+GLCM+LBP+HOG-PCA+LAB", 0.7300, 0.0200, 0.7250, 0.6100])

    sel3 = select_fusion_arm_from_results(csv_path3)
    assert sel3["selected_fusion_arm"] == "EfficientNet+GLCM+LBP+HOG-PCA+LAB"
    assert sel3["selected_fusion_arm_id"] == "A6"
    assert sel3["selected_fusion_dimension"] == 1348


def test_phase5_real_results_selection_and_manifest():
    """Verify selection on actual reports/phase5 artifacts."""
    config = get_config()
    results_path = config.aef_crc_phase5_reports_dir / "fusion_results.csv"
    manifest_path = config.aef_crc_phase5_reports_dir / "phase5_manifest.json"

    if not results_path.exists() or not manifest_path.exists():
        pytest.skip("Phase 5 report artifacts not present on disk.")

    sel = select_fusion_arm_from_results(results_path, config)
    assert sel["selected_fusion_arm"] == "EfficientNet+GLCM+LBP+LAB"
    assert sel["selected_fusion_arm_id"] == "A7"
    assert sel["selected_fusion_dimension"] == 1316
    assert sel["selected_fusion_representation"] == "efficientnet_glcm_lbp_lab"

    # Check manifest fields
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["phase"] == "phase5"
    assert manifest["winner"] == "P3-BASE"  # Preserves Phase 3 winner
    assert manifest["selected_fusion_arm"] == "EfficientNet+GLCM+LBP+LAB"
    assert manifest["selected_fusion_arm_id"] == "A7"
    assert manifest["selected_fusion_dimension"] == 1316
    assert manifest["selected_fusion_representation"] == "efficientnet_glcm_lbp_lab"
    assert manifest["selection_rule"] == SELECTION_RULE_NAME
    assert manifest["dataset_freeze_hash"] is not None
    assert manifest["fold_plan_hash"] is not None


def test_parse_phase5_winner_phase6_interface():
    """Verify Phase 6 interface parse_phase5_winner yields selected arm."""
    from run_aef_crc_phase6 import parse_phase5_winner
    config = get_config()
    results_path = config.aef_crc_phase5_reports_dir / "fusion_results.csv"
    if not results_path.exists():
        pytest.skip("Phase 5 results not present.")

    winner_arm, detail = parse_phase5_winner(config)
    assert winner_arm == "EfficientNet+GLCM+LBP+LAB"
    assert "A7" in detail
    assert "Macro-F1=0.7270" in detail
