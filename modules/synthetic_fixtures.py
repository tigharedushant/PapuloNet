"""
modules/synthetic_fixtures.py

Shared synthetic-data fixtures for plumbing checks and test isolation.

Why this module exists (Phase 6, Step -1): run_aef_crc_phase5.py's
plumbing-only fallback path used to import _make_test_config /
_build_synthetic_dataset / _run_pipeline_through_split directly from
tests/test_leakage.py, which has `import pytest` at module scope. In
any environment without pytest installed, importing test_leakage.py
for its fixtures crashed the plumbing fallback -- confirmed by actually
running it, not inferred. A production entry point must never depend
on a test file's import graph.

tests/test_leakage.py and tests/test_fusion.py now import these same
functions from here instead of defining/duplicating them -- single
source of truth, test files keep their own `import pytest` for their
own @pytest.fixture/pytest.raises usage, which is correct and expected.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image

from config.config import PSDConfig, DatasetConfig, get_config
from modules.scanner import DatasetScanner
from modules.mapper import ClassMapper
from modules.extractor import TargetClassExtractor
from modules.duplicate_detector import DuplicateDetector
from modules.quality_assessor import QualityAssessor
from modules.standardizer import ImageStandardizer
from modules.metadata import MetadataGenerator
from modules.splitter import DatasetSplitter
from modules.provenance import ProvenanceTracker


def make_test_config(tmp_path: Path, seed: int = 42) -> PSDConfig:
    base = get_config()
    datasets_dir = tmp_path / "datasets"
    return dataclasses.replace(
        base,
        project_root=tmp_path,
        datasets_dir=datasets_dir,
        output_dir=tmp_path / "output",
        reports_dir=tmp_path / "reports",
        logs_dir=tmp_path / "logs",
        extracted_dir=tmp_path / "output" / "01_extracted",
        ambiguous_review_dir=tmp_path / "output" / "02_ambiguous_review",
        duplicates_removed_dir=tmp_path / "output" / "03_duplicates_removed",
        low_quality_dir=tmp_path / "output" / "04_low_quality_excluded",
        harmonized_dir=tmp_path / "output" / "05_harmonized",
        final_dir=tmp_path / "output" / "06_final_split",
        aef_crc_output_dir=tmp_path / "output" / "aef_crc",
        aef_crc_reports_dir=tmp_path / "reports" / "aef_crc",
        aef_crc_artifacts_dir=tmp_path / "artifacts" / "phase3",
        aef_crc_phase3_reports_dir=tmp_path / "reports" / "phase3",
        aef_crc_phase3_logs_dir=tmp_path / "logs" / "phase3",
        aef_crc_phase4_artifacts_dir=tmp_path / "artifacts" / "phase4",
        aef_crc_phase4_reports_dir=tmp_path / "reports" / "phase4",
        aef_crc_phase5_reports_dir=tmp_path / "reports" / "phase5",
        aef_crc_phase6_reports_dir=tmp_path / "reports" / "phase6",
        aef_crc_phase6_artifacts_dir=tmp_path / "artifacts" / "phase6",
        aef_crc_phase7_reports_dir=tmp_path / "reports" / "phase7",
        aef_crc_phase7_artifacts_dir=tmp_path / "artifacts" / "phase7",
        aef_crc_phase8_reports_dir=tmp_path / "reports" / "phase8",
        aef_crc_phase8_artifacts_dir=tmp_path / "artifacts" / "phase8",
        random_seed=seed,
        datasets=[
            DatasetConfig(
                name="DermNet", path=datasets_dir / "dermnet",
                description="test", adapter_kind="dermnet",
            ),
            DatasetConfig(
                name="SkinDisNet", path=datasets_dir / "skindisnet",
                description="test", adapter_kind="skindisnet",
            ),
        ],
    )


def make_img(path: Path, seed_val: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed_val)
    arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def build_synthetic_dataset(config: PSDConfig, seed_offset: int = 0) -> None:
    """DermNet: no augmentation, 2 splits, per-image classifiable filenames.
    SkinDisNet: Preprocessed (original) + Augmented, deliberately more
    augmented images than original ones (mirrors the real Seborrheic
    Dermatitis situation this whole fix exists for)."""
    counter = [seed_offset]

    def next_seed():
        counter[0] += 1
        return counter[0]

    dermnet_root = config.datasets[0].path
    for split in ("train", "test"):
        for cls, n in (("Psoriasis", 30), ("Lichen Planus", 20)):
            for i in range(n):
                make_img(
                    dermnet_root / split / cls / f"{cls.replace(' ', '_').lower()}_{split}_{i}.jpg",
                    next_seed(),
                )

    skindisnet_root = config.datasets[1].path
    for i in range(6):
        make_img(skindisnet_root / "SkinDisNet" / "Preprocessed" / "SD" / f"sd_orig_{i}.jpg", next_seed())
    for i in range(20):
        make_img(skindisnet_root / "SkinDisNet" / "Augmented" / "SD" / f"sd_aug_{i}.jpg", next_seed())
    for i in range(15):
        make_img(skindisnet_root / "SkinDisNet" / "Preprocessed" / "AD" / f"ad_orig_{i}.jpg", next_seed())


def run_pipeline_through_split(config: PSDConfig) -> None:
    """Runs the exact real stages needed to reach output/06_final_split/,
    same modules main.py calls, same order."""
    for d in config.required_dirs():
        d.mkdir(parents=True, exist_ok=True)

    tracker = ProvenanceTracker()
    scan_results = DatasetScanner(config).scan_all()
    decisions = ClassMapper(config).map_all(scan_results)
    TargetClassExtractor(config, tracker).extract_all(decisions)
    DuplicateDetector(config, tracker).detect_all()
    QualityAssessor(config, tracker).assess_all()
    ImageStandardizer(config, tracker).standardize_all()
    MetadataGenerator(config).generate(tracker)
    DatasetSplitter(config).split_all()
