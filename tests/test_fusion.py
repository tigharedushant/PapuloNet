"""
tests/test_fusion.py

Targeted tests for genuinely new Phase-5 risk surface: PSD-ID alignment
enforcement in build_fusion_fold, branch-wise dimension tracking, and
the test-isolation bug this phase's plumbing run surfaced (Phase 3/4/5
artifact dirs previously defaulted to the REAL project paths inside
_make_test_config, silently contaminating them).

Does not re-test what tests/test_leakage.py already covers (outer
split, CV fold isolation, class weights) -- those are unchanged by
Phase 5 and remain that file's responsibility.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from config.config import get_config
from modules.synthetic_fixtures import make_test_config as _make_test_config, build_synthetic_dataset as _build_synthetic_dataset, run_pipeline_through_split as _run_pipeline_through_split
from modules.fold_loader import ImbalanceAwareFoldLoader
from modules.fusion import ARMS, build_fusion_fold, compute_sample_weights


def test_make_test_config_isolates_phase345_dirs(tmp_path):
    """Regression test for the contamination bug found this phase:
    aef_crc_artifacts_dir / aef_crc_phase3_reports_dir / etc. must all
    resolve under tmp_path, never under the real project root."""
    real_config = get_config()
    test_config = _make_test_config(tmp_path, seed=42)

    for field in (
        "aef_crc_artifacts_dir", "aef_crc_phase3_reports_dir", "aef_crc_phase3_logs_dir",
        "aef_crc_phase4_artifacts_dir", "aef_crc_phase4_reports_dir", "aef_crc_phase5_reports_dir",
        "aef_crc_phase7_reports_dir", "aef_crc_phase7_artifacts_dir",
        "aef_crc_phase8_reports_dir", "aef_crc_phase8_artifacts_dir",
    ):
        test_path = getattr(test_config, field)
        real_path = getattr(real_config, field)
        assert str(test_path).startswith(str(tmp_path)), f"{field} not isolated: {test_path}"
        assert test_path != real_path, f"{field} matches the real project path -- would contaminate it"


def test_build_fusion_fold_refuses_psd_id_collision_between_train_and_val(tmp_path):
    """If a PSD ID somehow appeared in both fold.train_records and
    fold.val_records (should be structurally impossible upstream, but
    this is the same 'verify, don't assume' discipline as the rest of
    the project), build_fusion_fold must refuse rather than silently
    build a fusion vector against an ambiguous fold."""
    import dataclasses as dc
    from modules.aef_input_validator import ImageRecord
    from pathlib import Path

    config = _make_test_config(tmp_path, seed=42)
    rec_train = ImageRecord(
        psd_id="PSD_DUPLICATE", split="train", mapped_class="Psoriasis",
        source_dataset="DermNet", dataset_version="DermNet", original_split="train",
        source_type="original", file_path=Path("/fake/a.jpg"),
    )
    rec_val = dc.replace(rec_train, split="val")  # SAME psd_id, wrongly also in val

    from modules.fold_loader import Fold
    bad_fold = Fold(fold_index=0, train_records=[rec_train], val_records=[rec_val], class_weights={"Psoriasis": 1.0})

    try:
        build_fusion_fold(config, bad_fold, ("glcm",), "FAKE-EXP")
        assert False, "should have raised AssertionError for train/val PSD ID collision"
    except AssertionError:
        pass


def test_compute_sample_weights_matches_fold_class_weights():
    weights = {"Psoriasis": 0.5, "Lichen_Planus": 2.0}
    labels = ["Psoriasis", "Psoriasis", "Lichen_Planus"]
    result = compute_sample_weights(labels, weights)
    assert np.allclose(result, [0.5, 0.5, 2.0])


def test_all_arms_defined_with_deep_as_common_branch():
    from modules.fusion import FUSION_ARMS
    assert len(FUSION_ARMS) == 8
    assert len(ARMS) == 8
    for arm_name, branches in ARMS.items():
        assert "deep" in branches, f"{arm_name} does not include the deep branch"
    assert ARMS["EfficientNet"] == ("deep",)
    assert set(ARMS["EfficientNet+GLCM+LBP+HOG"]) == {"deep", "glcm", "lbp", "hog"}
    assert set(ARMS["EfficientNet+GLCM+LBP+HOG-PCA+LAB"]) == {"deep", "glcm", "lbp", "hog", "color_lab"}
