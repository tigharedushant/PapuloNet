"""
pipeline/harmonization_pipeline.py

The PSD-HP orchestrator.

Defines the ordered sequence of harmonization stages and runs every
one of them, threading state from one stage to the next. As of this
revision, that shared state includes a single ProvenanceTracker
instance, created once here and passed by reference into every stage
that needs to read or write per-image lineage (extractor,
duplicate_detector, quality_assessor, standardizer, metadata). This
is what lets those stages update one persistent record instead of
each maintaining its own partial view of "what happened to this
image."

Stage sequence:

    1.  Dataset Validation         (validator.py)
    2.  Dataset Scanning           (scanner.py)
    3.  Class Mapping              (mapper.py)
    4.  Coverage Verification      (coverage.py)
    5.  Target Class Extraction    (extractor.py)          -- provenance begins
    6.  Duplicate Detection        (duplicate_detector.py) -- updates provenance
    7.  Quality Assessment         (quality_assessor.py)   -- updates provenance
    8.  Image Standardization      (standardizer.py)       -- updates provenance
    9.  Metadata Generation        (metadata.py)           -- serializes provenance
    10. Dataset Statistics         (statistics.py)
    11. Train/Val/Test Split       (splitter.py)
    12. Integrity Verification     (integrity.py)          -- NEW: conservation check
    13. Reproducibility Manifest   (manifest.py)            -- NEW: config snapshot + manifest
    14. Report Generation          (report_generator.py)
"""

from __future__ import annotations

from typing import List, Optional

from config.config import PSDConfig
from utils.logger import get_module_logger

from modules.validator import DatasetValidator, DatasetValidationResult
from modules.scanner import DatasetScanner, DatasetScanResult
from modules.mapper import ClassMapper, MappingDecision
from modules.coverage import CoverageVerifier, CoverageSummary
from modules.provenance import ProvenanceTracker, ProvenanceRecord
from modules.extractor import TargetClassExtractor, ExtractionResult
from modules.duplicate_detector import DuplicateDetector, DuplicateDetectionResult
from modules.quality_assessor import QualityAssessor, QualityAssessmentResult
from modules.standardizer import ImageStandardizer, StandardizationResult
from modules.metadata import MetadataGenerator
from modules.statistics import DatasetStatistics, DatasetStatisticsSummary
from modules.splitter import DatasetSplitter, SplitResult
from modules.integrity import IntegrityVerifier, IntegrityResult
from modules.manifest import ManifestGenerator
from modules.report_generator import ReportGenerator


class HarmonizationPipeline:
    """Runs the full PSD-HP stage sequence, in order, threading a shared
    ProvenanceTracker through every stage that touches per-image lineage."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("pipeline", config.logs_dir, config.log_level)
        self.tracker = ProvenanceTracker()

        self.validation_results: List[DatasetValidationResult] = []
        self.scan_results: List[DatasetScanResult] = []
        self.mapping_decisions: List[MappingDecision] = []
        self.coverage_summary: Optional[CoverageSummary] = None
        self.extraction_result: Optional[ExtractionResult] = None
        self.duplicate_result: Optional[DuplicateDetectionResult] = None
        self.quality_result: Optional[QualityAssessmentResult] = None
        self.standardization_result: Optional[StandardizationResult] = None
        self.metadata_records: List[ProvenanceRecord] = []
        self.statistics_summary: Optional[DatasetStatisticsSummary] = None
        self.split_result: Optional[SplitResult] = None
        self.integrity_result: Optional[IntegrityResult] = None

    def run(self) -> None:
        self.logger.info("########## PSD-HP HARMONIZATION PIPELINE START ##########")

        ManifestGenerator(self.config).write_config_snapshot()

        self._run_stage_1_validation()
        if self.validation_results and all(r.status == "FAIL" for r in self.validation_results):
            self.logger.error(
                "Every configured dataset FAILED validation — stopping here. "
                "Fix reports/dataset_validation_report.txt before re-running."
            )
            self.logger.info("########## PSD-HP HARMONIZATION PIPELINE END (early stop) ##########")
            return

        self._run_stage_2_scanning()
        self._run_stage_3_mapping()
        self._run_stage_4_coverage()
        self._run_stage_5_extraction()
        self._run_stage_6_duplicate_detection()
        self._run_stage_7_quality_assessment()
        self._run_stage_8_standardization()
        self._run_stage_9_metadata_generation()
        self._run_stage_10_statistics()
        self._run_stage_11_split()
        self._run_stage_12_integrity_verification()
        self._run_stage_13_manifest()
        self._run_stage_14_report_generation()

        self.logger.info("PSD-HP: all stages completed. See reports/harmonization_report.pdf for the full summary.")
        self.logger.info("########## PSD-HP HARMONIZATION PIPELINE END ##########")

    # ---- Stage implementations ----

    def _run_stage_1_validation(self) -> None:
        self.logger.info("--- Stage 1: Dataset Validation ---")
        self.validation_results = DatasetValidator(self.config).validate_all()

    def _run_stage_2_scanning(self) -> None:
        self.logger.info("--- Stage 2: Dataset Scanning ---")
        self.scan_results = DatasetScanner(self.config).scan_all()

    def _run_stage_3_mapping(self) -> None:
        self.logger.info("--- Stage 3: Class Mapping ---")
        self.mapping_decisions = ClassMapper(self.config).map_all(self.scan_results)

    def _run_stage_4_coverage(self) -> None:
        self.logger.info("--- Stage 4: Coverage Verification ---")
        self.coverage_summary = CoverageVerifier(self.config).verify(self.mapping_decisions)

    def _run_stage_5_extraction(self) -> None:
        self.logger.info("--- Stage 5: Target Class Extraction ---")
        self.extraction_result = TargetClassExtractor(self.config, self.tracker).extract_all(self.mapping_decisions)

    def _run_stage_6_duplicate_detection(self) -> None:
        self.logger.info("--- Stage 6: Duplicate Detection ---")
        self.duplicate_result = DuplicateDetector(self.config, self.tracker).detect_all()

    def _run_stage_7_quality_assessment(self) -> None:
        self.logger.info("--- Stage 7: Quality Assessment ---")
        self.quality_result = QualityAssessor(self.config, self.tracker).assess_all()

    def _run_stage_8_standardization(self) -> None:
        self.logger.info("--- Stage 8: Image Standardization ---")
        self.standardization_result = ImageStandardizer(self.config, self.tracker).standardize_all()

    def _run_stage_9_metadata_generation(self) -> None:
        self.logger.info("--- Stage 9: Metadata Generation ---")
        self.metadata_records = MetadataGenerator(self.config).generate(self.tracker)

    def _run_stage_10_statistics(self) -> None:
        self.logger.info("--- Stage 10: Dataset Statistics ---")
        self.statistics_summary = DatasetStatistics(self.config).compute(self.metadata_records)

    def _run_stage_11_split(self) -> None:
        self.logger.info("--- Stage 11: Train/Val/Test Split ---")
        self.split_result = DatasetSplitter(self.config).split_all()

    def _run_stage_12_integrity_verification(self) -> None:
        self.logger.info("--- Stage 12: Integrity Verification ---")
        self.integrity_result = IntegrityVerifier(self.config).verify(self.metadata_records)

    def _run_stage_13_manifest(self) -> None:
        self.logger.info("--- Stage 13: Reproducibility Manifest ---")
        per_class_final_counts = self.statistics_summary.per_class_counts if self.statistics_summary else {}
        ManifestGenerator(self.config).write_dataset_manifest(per_class_final_counts)

    def _run_stage_14_report_generation(self) -> None:
        self.logger.info("--- Stage 14: Report Generation ---")
        ReportGenerator(self.config).generate(
            validation_results=self.validation_results,
            scan_results=self.scan_results,
            mapping_decisions=self.mapping_decisions,
            coverage_summary=self.coverage_summary,
            extraction_result=self.extraction_result,
            duplicate_result=self.duplicate_result,
            quality_result=self.quality_result,
            standardization_result=self.standardization_result,
            statistics_summary=self.statistics_summary,
            split_result=self.split_result,
        )
