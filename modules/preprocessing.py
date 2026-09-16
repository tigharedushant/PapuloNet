"""
modules/preprocessing.py

AEF-CRC Phase 3: Conditional Preprocessing.

Two operations, both trigger-based, both OFF by default per image
unless the image's own measured signal crosses a configured threshold:

  1. Hair-artifact removal (DullRazor-style: morphological blackhat +
     inpainting), triggered only when a meaningful fraction of pixels
     look hair-like.
  2. CLAHE local contrast enhancement, triggered only when the image's
     overall contrast is measurably low.

--- Brutal rule (Part 7 of the Phase 3 brief), taken literally ---
"Do not assume that every dark line is hair." "If hair detection
confidence is low: DO NOTHING." "False-positive preprocessing can be
worse than leaving the image untouched."

This module is built around that asymmetry deliberately. The hair
coverage threshold (config.hair_coverage_threshold) exists precisely
to avoid triggering on ordinary dark lesion structure, ink markings,
or scanner artifacts that happen to be linear and dark but are NOT
hair -- the blackhat transform alone cannot tell those apart with
certainty, so this module only acts when the AREA covered by
hair-like structure is large enough that leaving it untouched is
more likely to hurt than a conservative inpaint is to help. When in
doubt, this module does nothing to the image, by design.

Two modes, matching Part 9's P0/P1 ablation exactly:
  - "standard" (P0): no-op. Returns the image unchanged. Exists so
    P0 and P1 can be run through the identical code path and compared
    fairly, rather than P0 meaning "skip this module entirely" (which
    would make the two experiments harder to compare apples-to-apples).
  - "conditional" (P1): trigger-based, as described above.

Every decision is logged with enough detail to audit later --
image/PSD ID, the measured trigger value, the threshold it was
compared against, and whether the operation was applied. Never just
"applied=true/false" with no reasoning attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config.config import PSDConfig
from utils.logger import get_module_logger


@dataclass
class PreprocessingDecision:
    psd_id: str
    operation: str        # "hair_removal" | "clahe"
    trigger_value: float
    threshold: float
    applied: bool
    runtime_ms: float = 0.0


@dataclass
class PreprocessingResult:
    image: np.ndarray                 # BGR, uint8, same shape as input
    decisions: List[PreprocessingDecision]


class ConditionalPreprocessor:
    def __init__(self, config: PSDConfig) -> None:
        self.config = config
        self.logger = get_module_logger("preprocessing", config.aef_crc_phase3_logs_dir, config.log_level)

    # ---- Public API ----

    def process(self, image_bgr: np.ndarray, psd_id: str) -> PreprocessingResult:
        if self.config.preprocessing_mode == "standard":
            return PreprocessingResult(image=image_bgr.copy(), decisions=[])

        if self.config.preprocessing_mode != "conditional":
            raise ValueError(
                f"Unknown preprocessing_mode '{self.config.preprocessing_mode}' -- "
                f"expected 'standard' or 'conditional'."
            )

        import time
        decisions: List[PreprocessingDecision] = []
        result = image_bgr.copy()

        t0 = time.perf_counter()
        result, hair_decision = self._maybe_remove_hair(result, psd_id)
        hair_decision.runtime_ms = (time.perf_counter() - t0) * 1000
        decisions.append(hair_decision)

        t0 = time.perf_counter()
        result, contrast_decision = self._maybe_apply_clahe(result, psd_id)
        contrast_decision.runtime_ms = (time.perf_counter() - t0) * 1000
        decisions.append(contrast_decision)

        return PreprocessingResult(image=result, decisions=decisions)

    # ---- Hair-artifact removal ----

    def _hair_coverage_fraction(self, image_bgr: np.ndarray) -> float:
        """Fraction of pixels classified as hair-like via a morphological
        blackhat transform (finds thin dark structures against a lighter
        background -- the standard DullRazor-family signal for hair)."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        # Otsu threshold on the blackhat response isolates only the
        # strongest, most hair-consistent responses -- deliberately not
        # a fixed low threshold, which would over-trigger on any dark
        # texture (lesion borders, ink) rather than specifically
        # thin/linear structures the blackhat transform is designed for.
        _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(np.count_nonzero(mask)) / mask.size

    def _maybe_remove_hair(self, image_bgr: np.ndarray, psd_id: str):
        coverage = self._hair_coverage_fraction(image_bgr)
        threshold = self.config.hair_coverage_threshold
        applied = coverage >= threshold

        if applied:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Density safeguard (B21): if mask covers > 30% of pixels, inpainting is unsafe
            if coverage > 0.30:
                self.logger.warning(
                    f"{psd_id}: hair mask covers {coverage:.2%} of image (>30% safeguard limit) -- "
                    f"skipping inpainting to prevent severe lesion destruction."
                )
                applied = False
            else:
                image_bgr = cv2.inpaint(image_bgr, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
                self.logger.info(f"{psd_id}: hair removal APPLIED (coverage={coverage:.4f} >= threshold={threshold:.4f})")
        else:
            self.logger.debug(f"{psd_id}: hair removal skipped (coverage={coverage:.4f} < threshold={threshold:.4f})")

        return image_bgr, PreprocessingDecision(psd_id, "hair_removal", coverage, threshold, applied)

    # ---- CLAHE contrast enhancement ----

    def _contrast_std(self, image_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def _maybe_apply_clahe(self, image_bgr: np.ndarray, psd_id: str):
        std_before = self._contrast_std(image_bgr)
        threshold = self.config.contrast_std_threshold
        applied = std_before < threshold

        if applied:
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
            image_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            std_after = self._contrast_std(image_bgr)
            self.logger.info(
                f"{psd_id}: CLAHE APPLIED (contrast std {std_before:.2f} < threshold {threshold:.2f}, "
                f"now {std_after:.2f})"
            )
        else:
            self.logger.debug(f"{psd_id}: CLAHE skipped (contrast std {std_before:.2f} >= threshold {threshold:.2f})")

        return image_bgr, PreprocessingDecision(psd_id, "clahe", std_before, threshold, applied)


class PreprocessingLogWriter:
    """Persists every PreprocessingDecision to a CSV -- the audit trail
    Part 6 requires ("Every preprocessing decision should be logged")."""

    def __init__(self, config: PSDConfig) -> None:
        self.config = config

    def write(self, all_decisions: List[PreprocessingDecision], filename: str = "preprocessing_log.csv") -> Path:
        import csv
        self.config.aef_crc_phase3_reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.aef_crc_phase3_reports_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["psd_id", "operation", "trigger_value", "threshold", "applied", "runtime_ms"])
            for d in all_decisions:
                writer.writerow([d.psd_id, d.operation, f"{d.trigger_value:.6f}", f"{d.threshold:.6f}", d.applied, f"{d.runtime_ms:.4f}"])
        return path
