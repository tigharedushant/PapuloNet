"""
tests/test_cross_class_duplicates.py

Regression test for item 3 (Phase-4 review): a global cross-class
duplicate-detection pass, additive to the existing within-class
behavior, report-only (never moves/deletes a file).

Reuses tests/test_leakage.py's config/image helpers rather than
duplicating them.
"""

from __future__ import annotations

import csv
import dataclasses

import numpy as np
from PIL import Image

from tests.test_leakage import _make_test_config
from modules.provenance import ProvenanceTracker, ProvenanceRecord
from modules.duplicate_detector import DuplicateDetector


def _register(tracker: ProvenanceTracker, psd_id: str, mapped_class: str, source: str = "DermNet") -> None:
    tracker.register(ProvenanceRecord(
        psd_id=psd_id, source_dataset=source, original_class=mapped_class,
        original_filename=f"{psd_id}.jpg", original_path=f"/fake/{psd_id}.jpg", mapped_class=mapped_class,
    ))


def test_identical_image_in_two_classes_is_flagged_not_removed(tmp_path):
    config = _make_test_config(tmp_path, seed=42)

    # Build extracted_dir directly (bypassing the full adapter/mapper
    # pipeline -- this test targets duplicate_detector.py specifically,
    # not the whole pipeline test_leakage.py already covers).
    identical_arr = np.random.default_rng(5).integers(0, 255, (64, 64, 3), dtype=np.uint8)

    class_a_dir = config.extracted_dir / "Psoriasis"
    class_b_dir = config.extracted_dir / "Lichen_Planus"
    class_a_dir.mkdir(parents=True, exist_ok=True)
    class_b_dir.mkdir(parents=True, exist_ok=True)

    file_a = class_a_dir / "PSD_00000001__test.jpg"
    file_b = class_b_dir / "PSD_00000002__test.jpg"
    Image.fromarray(identical_arr).save(file_a)
    Image.fromarray(identical_arr).save(file_b)  # byte-identical content, different class

    # A genuinely different image in class A -- must NOT be flagged.
    different_arr = np.random.default_rng(999).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    file_c = class_a_dir / "PSD_00000003__test.jpg"
    Image.fromarray(different_arr).save(file_c)

    tracker = ProvenanceTracker()
    _register(tracker, "PSD_00000001", "Psoriasis")
    _register(tracker, "PSD_00000002", "Lichen_Planus")
    _register(tracker, "PSD_00000003", "Psoriasis")
    result = DuplicateDetector(config, tracker).detect_all()

    # Flagged, not removed: both files must still physically exist in
    # extracted_dir -- this pass never moves/deletes anything.
    assert file_a.exists() and file_b.exists() and file_c.exists()

    cross_pairs = {(d.psd_id_a, d.psd_id_b) for d in result.cross_class_duplicates}
    cross_pairs |= {(b, a) for a, b in cross_pairs}  # order-independent membership check
    assert ("PSD_00000001", "PSD_00000002") in cross_pairs or ("PSD_00000002", "PSD_00000001") in cross_pairs

    # The genuinely different image must not appear in any cross-class pair.
    involved_ids = {d.psd_id_a for d in result.cross_class_duplicates} | {d.psd_id_b for d in result.cross_class_duplicates}
    assert "PSD_00000003" not in involved_ids

    # Report file written and matches in-memory result.
    report_path = config.reports_dir / "cross_class_duplicates_report.csv"
    assert report_path.exists()
    with report_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result.cross_class_duplicates)


def test_within_class_behavior_unchanged_by_cross_class_addition(tmp_path):
    """Regression guard: adding the cross-class pass must not change
    within-class duplicate removal at all -- same trigger, same
    kept/removed logic, verified directly."""
    config = _make_test_config(tmp_path, seed=42)
    class_dir = config.extracted_dir / "Psoriasis"
    class_dir.mkdir(parents=True, exist_ok=True)

    identical_arr = np.random.default_rng(7).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    file_a = class_dir / "PSD_00000001__test.jpg"
    file_b = class_dir / "PSD_00000002__test.jpg"
    Image.fromarray(identical_arr).save(file_a)
    Image.fromarray(identical_arr).save(file_b)

    tracker = ProvenanceTracker()
    _register(tracker, "PSD_00000001", "Psoriasis")
    _register(tracker, "PSD_00000002", "Psoriasis")
    result = DuplicateDetector(config, tracker).detect_all()

    assert result.total_removed == 1  # one of the two identical images moved out
    assert len(result.groups) == 1
    assert result.groups[0].target_class == "Psoriasis"
