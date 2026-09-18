"""
tests/test_phase6_phase7_reports.py

Targeted tests verifying the integrity, completeness, and provenance of the
Phase 6 and Phase 7 report artifacts and their consistency with persisted artifacts.
"""

import csv
import json
from pathlib import Path
import joblib
import numpy as np
import pytest

from config.config import get_config


def test_phase6_report_artifacts_exist_and_consistent():
    config = get_config()
    p6_dir = config.project_root / "reports" / "phase6"
    
    # 1. feature_refinement_results.csv
    res_path = p6_dir / "feature_refinement_results.csv"
    assert res_path.exists(), "feature_refinement_results.csv missing"
    with open(res_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 10, f"Expected 10 fold results (5 BDA, 5 GA), got {len(reader)}"
    bda_f1 = [float(r["macro_f1"]) for r in reader if r["method"] == "bda"]
    ga_f1 = [float(r["macro_f1"]) for r in reader if r["method"] == "ga"]
    assert len(bda_f1) == 5 and len(ga_f1) == 5
    assert np.isclose(np.mean(bda_f1), 0.6994, atol=1e-3)
    assert np.isclose(np.mean(ga_f1), 0.6947, atol=1e-3)

    # 2. feature_refinement_summary.csv
    sum_path = p6_dir / "feature_refinement_summary.csv"
    assert sum_path.exists()
    with open(sum_path, "r", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert len(summary_rows) == 2
    assert summary_rows[0]["method"] == "bda" and summary_rows[0]["role"] == "Primary"
    assert summary_rows[1]["method"] == "ga" and summary_rows[1]["role"] == "Comparator"

    # 3. pairwise_mcnemar.csv
    mcn_path = p6_dir / "pairwise_mcnemar.csv"
    assert mcn_path.exists()
    with open(mcn_path, "r", encoding="utf-8") as f:
        mcn_rows = list(csv.DictReader(f))
    assert len(mcn_rows) == 1
    assert float(mcn_rows[0]["p_value"]) == 0.9247

    # 4. phase6_manifest.json
    man_path = p6_dir / "phase6_manifest.json"
    assert man_path.exists()
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    assert manifest["phase"] == "phase6"
    assert manifest["status"] == "CERTIFIED"
    assert manifest["upstream_phase5_winner"] == "A7"
    assert manifest["upstream_phase3_winner"] == "P3-BASE"
    assert manifest["input_dimension"] == 1316
    assert manifest["production_bda_mask"]["selected_features"] == 194

    # 5. Consistency with production_bda_mask.joblib
    mask_artifact = config.aef_crc_phase6_artifacts_dir / "production_bda_mask.joblib"
    assert mask_artifact.exists()
    payload = joblib.load(mask_artifact)
    assert payload["selected_count"] == manifest["production_bda_mask"]["selected_features"]
    assert payload["total_dim"] == manifest["input_dimension"]


def test_phase7_report_artifacts_exist_and_consistent():
    config = get_config()
    p7_dir = config.project_root / "reports" / "phase7"

    # 1. final_cv_fold_metrics.csv
    cv_path = p7_dir / "final_cv_fold_metrics.csv"
    assert cv_path.exists()
    with open(cv_path, "r", encoding="utf-8") as f:
        cv_rows = list(csv.DictReader(f))
    assert len(cv_rows) == 5
    f1s = [float(r["macro_f1"]) for r in cv_rows]
    assert np.isclose(np.mean(f1s), 0.6906, atol=1e-3)

    # 2. final_cv_summary.csv
    sum_path = p7_dir / "final_cv_summary.csv"
    assert sum_path.exists()
    with open(sum_path, "r", encoding="utf-8") as f:
        metrics = {r["metric"]: float(r["mean"]) for r in csv.DictReader(f)}
    assert metrics["Macro-F1"] == 0.6906
    assert metrics["Balanced Accuracy"] == 0.6923
    assert metrics["MCC"] == 0.5547

    # 3. per_class_metrics.csv
    per_class_path = p7_dir / "per_class_metrics.csv"
    assert per_class_path.exists()
    with open(per_class_path, "r", encoding="utf-8") as f:
        classes = [r["class"] for r in csv.DictReader(f)]
    assert set(classes) == set(config.target_classes)

    # 4. confusion_matrix.csv
    cm_path = p7_dir / "confusion_matrix.csv"
    assert cm_path.exists()
    with open(cm_path, "r", encoding="utf-8") as f:
        cm_rows = list(csv.DictReader(f))
    assert len(cm_rows) == 4
    total_samples = sum(int(r[col]) for r in cm_rows for col in config.target_classes)
    assert total_samples == 1146, f"Sum of confusion matrix must equal 1146, got {total_samples}"

    # 5. calibration_set_metrics.csv
    calib_path = p7_dir / "calibration_set_metrics.csv"
    assert calib_path.exists()
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}
    assert calib["Macro-F1"] == 0.7354
    assert calib["n_calibration_samples"] == 246

    # 6. phase7_manifest.json
    man_path = p7_dir / "phase7_manifest.json"
    assert man_path.exists()
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    assert manifest["phase"] == "phase7"
    assert manifest["status"] == "CERTIFIED"
    assert manifest["selected_fusion_arm"] == "EfficientNet+GLCM+LBP+LAB"
    assert manifest["selected_fusion_dimension"] == 1316
    assert manifest["source_experiment"] == "P3-BASE"
    assert manifest["classifier_name"] == "random_forest"
    assert manifest["feature_selection_method"] == "bda"

    # 7. Consistency with calibration_handoff.joblib
    handoff_path = config.aef_crc_phase7_artifacts_dir / "calibration_handoff.joblib"
    assert handoff_path.exists()
    handoff = joblib.load(handoff_path)
    assert handoff.source_experiment == "P3-BASE"
    assert handoff.feature_arm == "EfficientNet+GLCM+LBP+LAB"
    assert handoff.classifier_name == "random_forest"
    assert handoff.feature_selection_method == "bda"
    assert len(handoff.selected_feature_mask) == 1316
    assert int(handoff.selected_feature_mask.sum()) == 194
    assert len(handoff.calibration_true_labels) == 246


def test_locked_test_set_isolation():
    config = get_config()
    test_dir = config.final_dir / "test"
    test_files = list(test_dir.rglob("*.png")) + list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.jpeg"))
    assert len(test_files) == 243, f"Expected exactly 243 locked test images, found {len(test_files)}"
