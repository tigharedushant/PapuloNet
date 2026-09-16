"""
run_aef_crc_infer.py

Dedicated entry point for single-image inference and Explainable AI (XAI).

SCIENTIFIC CONTRACT:
- Operates on arbitrary NEW, unseen dermatological images.
- Completely separate from the 243-image locked outer test partition.
- Executes the full 9-stage inference pipeline using frozen production artifacts.
- Supports optional Grad-CAM (spatial CNN representation) and TreeSHAP (tabular RF attributions).
"""

import argparse
import sys
from pathlib import Path

from config.config import get_config
from modules.calibration_handoff import load_final_pipeline_handoff, load_conformal_artifact
from modules.inference import AEFCRCInferenceEngine
from utils.logger import get_module_logger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AEF-CRC Single-Image Inference & XAI Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input dermoscopy/clinical image file.",
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
        help="Directory to save explanation JSON and XAI visualizations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Device to use for EfficientNet inference and Grad-CAM backpropagation.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="Whether to compute and export Grad-CAM and TreeSHAP explanation bundles.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Sample identifier for reporting. Defaults to image filename stem.",
    )

    args = parser.parse_args()
    config = get_config()
    logger = get_module_logger("inference_cli", config.logs_dir, config.log_level)

    # 1. Validate image path
    image_path = args.image
    if not image_path.exists() or not image_path.is_file():
        print(f"\n[ERROR] Input image file not found: {image_path}")
        return 1

    sample_id = args.sample_id or image_path.stem

    # 2. Locate production pipeline artifact
    artifact_path = args.artifact_path or (
        config.project_root / "artifacts" / "phase9" / "final_pipeline_handoff.joblib"
    )

    if not artifact_path.exists():
        # Fallback to conformal artifact if pipeline handoff not yet compiled
        fallback_path = config.project_root / "artifacts" / "phase9" / "conformal_artifact.joblib"
        if fallback_path.exists():
            artifact_path = fallback_path
        else:
            print(f"\n[ERROR] Production artifact not found at {artifact_path}")
            print("Please ensure the AEF-CRC pipeline has completed Phase 9 to freeze model artifacts.")
            return 1

    try:
        logger.info(f"Loading inference pipeline artifact: {artifact_path}")
        try:
            artifact = load_final_pipeline_handoff(artifact_path)
        except Exception:
            artifact = load_conformal_artifact(artifact_path)
    except Exception as exc:
        logger.error(f"Failed to load artifact: {exc}")
        print(f"\n[ERROR] Corrupt or incompatible artifact: {exc}")
        return 1

    # 3. Initialize inference engine
    engine = AEFCRCInferenceEngine(artifact, device=args.device)

    # 4. Resolve output directory
    out_dir = args.output_dir or (config.project_root / "output" / "xai" / sample_id)

    # 5. Execute inference
    print("\n" + "=" * 80)
    print(f"  AEF-CRC SINGLE-IMAGE INFERENCE: {image_path.name}")
    print("=" * 80)

    try:
        result = engine.predict(
            image_input=image_path,
            sample_id=sample_id,
            explain=args.explain,
            xai_output_dir=out_dir if args.explain else None,
        )

        resp = result.response_bundle

        if not resp.is_valid_input:
            print(f"\n[STATUS: INPUT_QUALITY_FAILURE]")
            print(f"  {resp.review_status_explanation}")
            print("  No disease prediction was fabricated for this degenerate input.\n")
            return 1

        print(f"\n[PRIMARY PREDICTION]")
        print(f"  Predicted Disease Class   : {resp.predicted_class}")
        print(f"  Calibrated Confidence     : {resp.calibrated_confidence*100:.2f}% (Platt scaling)")
        print(f"  Raw RF Margin Confidence  : {resp.raw_confidence*100:.2f}%")

        print(f"\n[CLASS PROBABILITY DISTRIBUTION]")
        for c_name, p_calib in resp.calibrated_probabilities.items():
            p_raw = resp.raw_probabilities.get(c_name, 0.0)
            print(f"  - {c_name:<24}: Calibrated = {p_calib*100:6.2f}% | Raw RF = {p_raw*100:6.2f}%")

        print(f"\n[SPLIT-CONFORMAL PREDICTION SET (Confidence = {resp.nominal_coverage*100:.0f}%)]")
        print(f"  Marginal Prediction Set   : {resp.conformal_prediction_set}")
        print(f"  Prediction Set Size       : {resp.prediction_set_size}")
        print(f"  Conformal Cutoff (1 - q)  : {resp.marginal_threshold:.4f}")

        print(f"\n[CLINICAL TRIAGE STATUS]")
        print(f"  Review Status             : {resp.review_status}")
        print(f"  Triage Rationale          : {resp.review_status_explanation}")

        if args.explain and resp.xai_explanation is not None:
            xai = resp.xai_explanation
            print(f"\n[EXPLAINABLE AI (XAI) GENERATED]")
            print(f"  Grad-CAM Layer Target     : {xai.gradcam_target_layer}")
            print(f"  Grad-CAM Target Class     : {xai.gradcam_target_class}")
            print(f"  Grad-CAM Overlay Saved    : {xai.gradcam_overlay_path}")
            print(f"  SHAP Explainer Mode       : tree_path_dependent")
            print(f"  SHAP Feature Rankings     : {out_dir / 'shap_feature_importance.csv'}")
            print(f"  SHAP Block Aggregations   : {out_dir / 'shap_block_importance.csv'}")
            print(f"  Explanation Metadata      : {out_dir / 'explanation.json'}")

        print("\n" + "=" * 80 + "\n")
        return 0

    except Exception as exc:
        logger.exception(f"Inference execution failed: {exc}")
        print(f"\n[EXECUTION ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
