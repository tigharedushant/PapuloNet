"""
run_aef_crc_phase1.py

AEF-CRC Phase 1 entry point: dataset validation + input loader +
imbalance analysis, run against PSD-HP's actual final_split/ output.

Deliberately a separate script from main.py -- main.py's job is
"run the PSD-HP harmonization pipeline" and nothing else. AEF-CRC is
a new, independent modeling layer that CONSUMES PSD-HP's output; it
does not belong inside PSD-HP's own entry point.

Usage:
    python main.py                  # PSD-HP harmonization (unchanged)
    python run_aef_crc_phase1.py    # AEF-CRC Phase 1, after main.py has been run at least once
"""

from __future__ import annotations

import sys

from config.config import get_config
from modules.dataset_freeze import DatasetFreezer
from modules.aef_input_validator import AEFInputLoader, AEFDatasetValidator, Phase1ReportWriter


def main() -> int:
    config = get_config()

    print("=== AEF-CRC Phase 1: Dataset Validation + Imbalance Analysis ===")
    print(f"Reading PSD-HP final split from: {config.final_dir}")
    print(f"Reading PSD-HP metadata from:    {config.reports_dir / 'metadata.csv'}")
    print()

    # Step 0: Dataset freeze verification (if freeze exists)
    freezer = DatasetFreezer(config)
    freeze_file = config.aef_crc_reports_dir / "dataset_freeze.json"
    if freeze_file.exists():
        print("--- Step 0: Verifying against existing dataset freeze ---")
        verify_result = freezer.verify()
        print(f"Dataset Freeze Verification: {'PASSED' if verify_result.matches else 'FAILED'}")
        if not verify_result.matches:
            for m in verify_result.mismatches[:10]:
                print(f"  [{m.kind}] {m.detail}")
            print("\nDataset has changed from freeze identity! Aborting Phase 1.")
            return 1
        print()

    records = AEFInputLoader(config).load()
    report = AEFDatasetValidator(config).run(records)
    Phase1ReportWriter(config).write(report)

    print(f"Total images: {report.total_images}")
    print(f"Split counts: {report.split_counts}")
    print(f"Class counts: {report.class_counts}")
    print(f"Source counts: {report.source_counts}")
    if report.imbalance_ratio is not None:
        print(f"Imbalance ratio: {report.imbalance_ratio:.2f}")
    print(f"Image issues: {len(report.image_issues)}")
    print(f"Orphan disk files: {len(report.orphan_files)}")
    print(f"Missing disk files: {len(report.missing_disk_files)}")
    print(f"Leakage findings: {len(report.leakage_findings)}")
    for finding in report.leakage_findings:
        print(f"  [{finding.severity}] [{finding.kind}] {finding.detail}")
    print()
    print(f"PHASE 1 RESULT: {'PASSED' if report.passed else 'FAILED'}")
    if not report.passed:
        for reason in report.fail_reasons:
            print(f"  - {reason}")
    else:
        # Step 3: Write authoritative dataset_freeze.json snapshot
        freeze_path = freezer.freeze(overwrite=True)
        print(f"Authoritative dataset freeze snapshot created: {freeze_path.name}")

    print()
    print(f"Full reports written to: {config.aef_crc_reports_dir}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
