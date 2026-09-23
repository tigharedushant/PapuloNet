"""
run_aef_crc_test.py

Dedicated entry point for evaluating the held-out, locked AEF-CRC outer test set (N=243).

STRICT SCIENTIFIC GUARD:
The 243 outer-test images represent the final benchmark for this study.
To prevent data snooping, hypothesis contamination, or premature leakage:
- This script requires explicit authorization via `--confirm-locked-test-evaluation`.
- It will refuse execution if any required production artifact is absent.
- It executes exactly once on the frozen pipeline.
"""

import argparse
import sys
from pathlib import Path

from config.config import get_config
from run_aef_crc_phase9_v2 import configure_phase9_v2
from modules.calibration_handoff import load_final_pipeline_handoff
from modules.test_evaluation import evaluate_locked_outer_test, LockedOuterTestGuardError
from utils.logger import get_module_logger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AEF-CRC Locked Outer Test Evaluation (N=243)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--confirm-locked-test-evaluation",
        "--run-final-test",
        dest="confirm_locked_test_evaluation",
        action="store_true",
        default=False,
        help="Explicit security confirmation required to execute outer test evaluation.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=None,
        help="Path to the frozen final pipeline handoff artifact (joblib).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save test evaluation reports and calibration plots.",
    )

    args = parser.parse_args()
    config = configure_phase9_v2(get_config())
    logger = get_module_logger("test_evaluation", config.logs_dir, config.log_level)

    # 1. Check security guard flag
    if not args.confirm_locked_test_evaluation:
        print("\n" + "=" * 80)
        print("  [AEF-CRC EXECUTION GUARD ACTIVE] -- OUTER TEST EVALUATION LOCKED")
        print("=" * 80)
        print("The 243 outer-test images are strictly held out to preserve research validity.")
        print("Test evaluation is only permitted after all development phases (Phases 2-9)")
        print("are completely executed and frozen.")
        print("\nTo authorize final test evaluation on frozen artifacts, execute:")
        print("  python run_aef_crc_test.py --confirm-locked-test-evaluation\n")
        print("=" * 80)
        return 0

    # 2. Locate production pipeline artifact
    artifact_path = args.artifact_path or (
        config.project_root / "artifacts" / "phase9_v2" / "final_pipeline_handoff.joblib"
    )

    if not artifact_path.exists():
        logger.error(
            f"Production pipeline artifact not found at {artifact_path}. "
            "Outer test evaluation cannot proceed without the frozen Phase 9 handoff artifact."
        )
        print(f"\n[ERROR] Missing required frozen artifact: {artifact_path}")
        print("Please complete Phases 3-9 production pipeline execution before evaluating the test set.")
        return 1

    try:
        logger.info(f"Loading frozen pipeline artifact: {artifact_path}")
        pipeline_artifact = load_final_pipeline_handoff(artifact_path)
    except Exception as exc:
        logger.error(f"Failed to load pipeline artifact: {exc}")
        print(f"\n[ERROR] Corrupt or incompatible artifact: {exc}")
        return 1

    # 3. Execute locked outer test evaluation
    try:
        logger.info("Starting authoritative outer test benchmark evaluation (N=243)...")
        report = evaluate_locked_outer_test(
            config=config,
            pipeline_artifact=pipeline_artifact,
            confirm_execution=True,
            output_dir=args.output_dir,
        )

        print("\n" + "=" * 80)
        print("  AEF-CRC LOCKED OUTER TEST BENCHMARK RESULTS (N=243)")
        print("=" * 80)
        cm = report.classification_metrics
        cal = report.calibration_metrics
        conf = report.conformal_metrics

        print(f"\n[CLASSIFICATION PERFORMANCE]")
        print(f"  Macro-F1 (Primary)   : {cm['macro_f1']:.4f}")
        print(f"  Overall Accuracy     : {cm['accuracy']:.4f}")
        print(f"  Balanced Accuracy    : {cm['balanced_accuracy']:.4f}")

        print(f"\n[PROBABILITY CALIBRATION]")
        print(f"  Raw ECE              : {cal['raw_ece']:.4f}")
        print(f"  Platt-Calibrated ECE : {cal['platt_calibrated_ece']:.4f} (Improvement: {cal['ece_improvement']:+.4f})")
        print(f"  Raw Brier Score      : {cal['raw_multiclass_brier']:.4f}")
        print(f"  Platt Brier Score    : {cal['platt_calibrated_brier']:.4f} (Improvement: {cal['brier_improvement']:+.4f})")

        print(f"\n[SPLIT-CONFORMAL PREDICTION (Target = {conf['nominal_target_coverage']*100:.1f}%)]")
        print(f"  Empirical Coverage   : {conf['marginal_empirical_coverage']*100:.2f}% (Delta: {conf['coverage_delta']*100:+.2f}%)")
        print(f"  Mean Set Size        : {conf['mean_set_size']:.2f}")
        print(f"  Median Set Size      : {conf['median_set_size']:.1f}")
        print(f"  Singleton Fraction   : {conf['singleton_fraction']*100:.1f}%")
        print(f"  Ambiguous Fraction   : {conf['ambiguous_fraction']*100:.1f}%")
        print(f"  Empty Set Fraction   : {conf['empty_set_fraction']*100:.1f}%")
        print("=" * 80 + "\n")

        return 0

    except LockedOuterTestGuardError as exc:
        logger.error(f"Outer test execution rejected: {exc}")
        print(f"\n[GUARD REJECTION] {exc}")
        return 1
    except Exception as exc:
        logger.exception(f"Unexpected error during test evaluation: {exc}")
        print(f"\n[UNEXPECTED ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
