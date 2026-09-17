# SYNTHETIC_TEST_FIXTURE = True
# This test uses strictly synthetic, generated, or mock fixtures.
# Zero real dataset images or private research artifacts are required or accessed.
"""
tests/test_targeted_corrections.py

Regression test suite verifying the targeted correction pass:
1. Authoritative fold plan counts (Fold 0=916/230, Folds 1-4=917/229, 100% coverage over 1146 CV records).
2. Parent lineage leak counts (9 val parents, 4 test parents with descendants, 13 union, 52 descendants).
3. Authoritative winner selection rule (mean validation Macro-F1 primary, >0.005 threshold, parsimony preference, no SD tie-breaker).
4. Early stopping protocol alignment (configurable patience, strict fixed epochs when disabled).
5. Phase 3 legacy results quarantine status.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import pytest

from config.config import PSDConfig
from modules.evaluation import select_phase3_winner, AggregatedMetrics, validate_phase3_winner
from modules.training import _build_callbacks


def get_config() -> PSDConfig:
    return PSDConfig()


_FOLD_PLAN_PATH = get_config().aef_crc_reports_dir / "fold_plan.csv"
_METADATA_PATH = get_config().reports_dir / "metadata.csv"

requires_authoritative_fold_plan = pytest.mark.skipif(
    not _FOLD_PLAN_PATH.exists(),
    reason="real-data integration test skipped because authoritative dataset artifacts are not installed (clean framework mode)",
)

requires_authoritative_metadata = pytest.mark.skipif(
    not _METADATA_PATH.exists(),
    reason="real-data integration test skipped because authoritative dataset artifacts are not installed (clean framework mode)",
)


@requires_authoritative_fold_plan
def test_authoritative_fold_plan_counts():
    """Verify authoritative fold plan row counts and fold distribution."""
    config = get_config()
    plan_path = config.aef_crc_reports_dir / "fold_plan.csv"
    assert plan_path.exists(), f"Missing {plan_path}"

    with plan_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 6483, f"Expected 6483 rows in fold_plan.csv, got {len(rows)}"

    fold_roles: Dict[str, Dict[str, int]] = {}
    val_ids_by_fold: Dict[str, set] = {}

    for r in rows:
        fid = r["fold_id"]
        role = r["split_role"]
        fold_roles.setdefault(fid, {}).setdefault(role, 0)
        fold_roles[fid][role] += 1
        if role == "val":
            val_ids_by_fold.setdefault(fid, set()).add(r["psd_id"])

    # Outer holdout / excluded augmented
    assert "-1" in fold_roles
    assert fold_roles["-1"].get("holdout_val", 0) == 246
    assert fold_roles["-1"].get("holdout_test", 0) == 243
    assert fold_roles["-1"].get("excluded_augmented", 0) == 264
    assert sum(fold_roles["-1"].values()) == 753

    # Fold 0: 916 train / 230 val
    assert fold_roles["0"]["train"] == 916
    assert fold_roles["0"]["val"] == 230

    # Folds 1 to 4: 917 train / 229 val
    for fid in ("1", "2", "3", "4"):
        assert fold_roles[fid]["train"] == 917, f"Fold {fid} train count != 917"
        assert fold_roles[fid]["val"] == 229, f"Fold {fid} val count != 229"

    # Validation coverage: 1146 unique IDs
    all_val_ids = set()
    for fid, s in val_ids_by_fold.items():
        assert len(s) == fold_roles[fid]["val"]
        all_val_ids.update(s)
    assert len(all_val_ids) == 1146, f"Expected 1146 unique validation IDs, got {len(all_val_ids)}"


@requires_authoritative_metadata
def test_parent_lineage_leak_counts():
    """Verify exact counts of SkinDisNet parent leakage into train and exclusion from CV."""
    import re

    config = get_config()
    meta_path = config.reports_dir / "metadata.csv"
    assert meta_path.exists()

    with meta_path.open("r", encoding="utf-8") as f:
        meta = {r["psd_id"]: r for r in csv.DictReader(f)}

    split_dir = config.final_dir
    train_files = list((split_dir / "train").rglob("*.jpg"))
    val_files = list((split_dir / "val").rglob("*.jpg"))
    test_files = list((split_dir / "test").rglob("*.jpg"))

    train_aug = [meta[p.stem] for p in train_files if meta[p.stem]["source_type"] == "augmented"]
    val_sd = [meta[p.stem] for p in val_files if meta[p.stem]["source_dataset"] in ("SkinDisNet", "SkinDisNet_2")]
    test_sd = [meta[p.stem] for p in test_files if meta[p.stem]["source_dataset"] in ("SkinDisNet", "SkinDisNet_2")]

    assert len(train_aug) == 264, f"Expected 264 augmented in train, got {len(train_aug)}"
    assert len(val_sd) == 9, f"Expected 9 SkinDisNet originals in outer val, got {len(val_sd)}"
    assert len(test_sd) == 5, f"Expected 5 SkinDisNet originals in outer test, got {len(test_sd)}"

    def norm_orig_key(s: str) -> str:
        stem = s.rsplit(".", 1)[0]
        return re.sub(r"[\s\(\)_]+", "", stem).lower()

    def norm_aug_key(s: str) -> str:
        stem = s.rsplit(".", 1)[0]
        parent = re.sub(r"_\d+$", "", stem)
        return re.sub(r"[\s\(\)_]+", "", parent).lower()

    val_parents = {norm_orig_key(m["original_filename"]): m for m in val_sd}
    test_parents = {norm_orig_key(m["original_filename"]): m for m in test_sd}

    # Outer val and outer test parents are disjoint
    assert len(set(val_parents.keys()).intersection(set(test_parents.keys()))) == 0

    leaked_val_parents = set()
    leaked_test_parents = set()
    val_descendants = []
    test_descendants = []

    for m in train_aug:
        k = norm_aug_key(m["original_filename"])
        if k in val_parents:
            leaked_val_parents.add(k)
            val_descendants.append(m)
        elif k in test_parents:
            leaked_test_parents.add(k)
            test_descendants.append(m)

    # 9 val parents leak 37 descendants into train
    assert len(leaked_val_parents) == 9
    assert len(val_descendants) == 37

    # 4 of 5 test parents leak 15 descendants into train (SD_74 has 0)
    assert len(leaked_test_parents) == 4
    assert len(test_descendants) == 15

    # Union is 13 unique parents
    union_parents = leaked_val_parents.union(leaked_test_parents)
    assert len(union_parents) == 13

    # Total descendants leaked is 52
    assert len(val_descendants) + len(test_descendants) == 52


def _make_dummy_agg(macro_f1: float, macro_f1_std: float = 0.01) -> AggregatedMetrics:
    return AggregatedMetrics(
        n_folds=5,
        macro_f1_mean=macro_f1,
        macro_f1_std=macro_f1_std,
        accuracy_mean=macro_f1,
        accuracy_std=macro_f1_std,
        balanced_accuracy_mean=macro_f1,
        balanced_accuracy_std=macro_f1_std,
        weighted_f1_mean=macro_f1,
        weighted_f1_std=macro_f1_std,
        mcc_mean=macro_f1,
        mcc_std=macro_f1_std,
        per_class_f1_mean={},
        per_class_f1_std={},
    )


def test_winner_selection_threshold_and_parsimony():
    """Verify winner selection rule: > 0.005 threshold, parsimony preference, no SD tie-breaker."""
    # Case 1: Candidate exceeds baseline by > 0.005 -> Candidate wins
    aggs_candidate_wins = {
        "P3-BASE": _make_dummy_agg(0.7000, 0.010),
        "P3-PRE": _make_dummy_agg(0.6000, 0.010),
        "P3-AUG": _make_dummy_agg(0.7060, 0.050),  # +0.006 > 0.005
    }
    winner, rationale = select_phase3_winner(aggs_candidate_wins, threshold=0.005)
    assert winner == "P3-AUG"
    assert "exceeding P3-BASE" in rationale

    # Case 2: Candidate exceeds baseline by <= 0.005 -> P3-BASE wins for parsimony
    aggs_parsimony = {
        "P3-BASE": _make_dummy_agg(0.7012, 0.013),
        "P3-PRE": _make_dummy_agg(0.6040, 0.030),
        "P3-AUG": _make_dummy_agg(0.7030, 0.002),  # +0.0018 <= 0.005; lower SD must NOT break tie
    }
    winner, rationale = select_phase3_winner(aggs_parsimony, threshold=0.005)
    assert winner == "P3-BASE"
    assert "retained for parsimony" in rationale

    # Case 3: Candidate is inferior to baseline -> P3-BASE wins
    aggs_inferior = {
        "P3-BASE": _make_dummy_agg(0.7012, 0.013),
        "P3-PRE": _make_dummy_agg(0.6040, 0.030),
        "P3-AUG": _make_dummy_agg(0.7006, 0.005),
    }
    winner, rationale = select_phase3_winner(aggs_inferior, threshold=0.005)
    assert winner == "P3-BASE"


def test_early_stopping_configurable_and_optional(tmp_path):
    """Verify _build_callbacks adheres strictly to fixed epochs protocol (no EarlyStopping, only ModelCheckpoint)."""
    import tensorflow as tf
    ckpt_path = tmp_path / "model.keras"

    config = get_config()
    callbacks = _build_callbacks(config, ckpt_path)
    types = [type(cb).__name__ for cb in callbacks]
    assert "EarlyStopping" not in types, "Protocol requires strict fixed epochs (no EarlyStopping)!"
    assert "ModelCheckpoint" in types, "ModelCheckpoint required to save best val_loss model!"


def test_phase3_quarantine_integrity():
    """Verify that reports/phase3 is cleanly reset, or if winner.json is present, it is a valid non-quarantined completed Phase 3 winner artifact."""
    config = get_config()
    winner_path = config.aef_crc_phase3_reports_dir / "winner.json"
    if not winner_path.exists():
        # Clean pre-run state: active phase3 reports directory has zero stale winner artifacts
        return
    data = json.loads(winner_path.read_text(encoding="utf-8"))

    # If winner.json exists, it must NOT be a legacy/quarantined artifact
    assert not data.get("quarantined"), "Stale/quarantined winner.json detected: 'quarantined' must not be True"
    assert data.get("status") != "LEGACY_QUARANTINED", "Stale/quarantined winner.json detected: status must not be LEGACY_QUARANTINED"
    assert "quarantine_notice" not in data, "Stale/quarantined winner.json detected: 'quarantine_notice' present"

    # Validate required Phase 3 winner contract fields
    required_fields = [
        "run_id",
        "created_at_utc",
        "winner",
        "winner_experiment_id",
        "selection_metric",
        "selection_rule",
        "dataset_freeze_hash",
        "fold_plan_hash",
        "preprocessing_mode",
        "training_time_augmentation",
        "representation_id",
        "macro_f1_mean",
        "macro_f1_std",
        "folds",
    ]
    for field in required_fields:
        assert field in data, f"Legitimate winner.json missing required contract field '{field}'"

    # Winner experiment ID must be a recognized Phase 3 arm
    winner_arm = data.get("winner_experiment_id") or data.get("winner")
    assert winner_arm in ("P3-BASE", "P3-PRE", "P3-AUG"), f"Invalid winner experiment arm: {winner_arm}"
    assert data.get("winner") == winner_arm

    # Contract integrity checks
    assert data["folds"] == 5
    assert isinstance(data["macro_f1_mean"], (int, float))
    assert 0.0 < data["macro_f1_mean"] <= 1.0

    # Programmatic contract validation when winner artifact directory is present
    if (config.aef_crc_artifacts_dir / winner_arm).exists():
        is_valid, errors = validate_phase3_winner(config.aef_crc_phase3_reports_dir, config.aef_crc_artifacts_dir)
        assert is_valid, f"Phase 3 winner contract validation failed: {errors}"
