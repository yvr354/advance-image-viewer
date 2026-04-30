"""
Image quality metrics engine.
All metrics are objective and numerical — no human-eye judgment.
"""

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class QualityResult:
    overall_score: float        # 0–100 composite
    verdict: str                # PASS / FAIL

    # Exposure
    mean_brightness: float
    overexposed_pct: float      # % pixels above threshold
    underexposed_pct: float     # % pixels below threshold
    exposure_ok: bool

    # Contrast
    rms_contrast: float         # Root-mean-square contrast
    michelson_contrast: float   # (max-min)/(max+min)
    dynamic_range_stops: float

    # Noise
    noise_level: float          # Estimated noise std dev in flat regions
    snr_db: float               # Signal-to-noise ratio in dB

    # Histogram stats
    hist_mean: float
    hist_std: float
    hist_min: int
    hist_max: int
    hist_median: float


class QualityEngine:
    def __init__(
        self,
        overexpose_threshold: int = 250,
        underexpose_threshold: int = 5,
        pass_threshold: float = 60.0,
    ):
        self.overexpose_threshold = overexpose_threshold
        self.underexpose_threshold = underexpose_threshold
        self.pass_threshold = pass_threshold

    def analyze(self, image: np.ndarray) -> QualityResult:
        gray = self._to_gray_8bit(image)
        flat = gray.flatten().astype(np.float64)

        mean_b = float(np.mean(flat))
        std_b  = float(np.std(flat))
        min_b  = int(np.min(flat))
        max_b  = int(np.max(flat))
        median_b = float(np.median(flat))

        over_pct  = float(np.sum(gray > self.overexpose_threshold) / gray.size * 100)
        under_pct = float(np.sum(gray < self.underexpose_threshold) / gray.size * 100)
        exposure_ok = (over_pct < 1.0) and (under_pct < 1.0)

        rms_contrast = float(std_b / 255.0 * 100)
        denom = float(max_b + min_b)
        michelson = float((max_b - min_b) / denom) if denom > 0 else 0.0
        dr_stops = float(np.log2(max_b / max(min_b, 1))) if max_b > 0 else 0.0

        noise_level = self._estimate_noise(gray)
        signal = max(mean_b, 1.0)
        snr_db = float(20 * np.log10(signal / max(noise_level, 0.001)))

        overall = self._composite_score(
            exposure_ok, over_pct, under_pct, rms_contrast, noise_level, snr_db
        )
        verdict = "PASS" if overall >= self.pass_threshold else "FAIL"

        return QualityResult(
            overall_score=overall,
            verdict=verdict,
            mean_brightness=mean_b,
            overexposed_pct=over_pct,
            underexposed_pct=under_pct,
            exposure_ok=exposure_ok,
            rms_contrast=rms_contrast,
            michelson_contrast=michelson,
            dynamic_range_stops=dr_stops,
            noise_level=noise_level,
            snr_db=snr_db,
            hist_mean=mean_b,
            hist_std=std_b,
            hist_min=min_b,
            hist_max=max_b,
            hist_median=median_b,
        )

    def compute_histogram(self, image: np.ndarray, bins: int = 256) -> dict:
        """Return histogram data for all channels."""
        result = {}
        if image.ndim == 2:
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
            result["gray"] = hist.flatten()
        else:
            for i, ch in enumerate(["red", "green", "blue"]):
                hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
                result[ch] = hist.flatten()
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
            result["luma"] = hist.flatten()
        return result

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        """Estimate noise using Laplacian method (fast, no flat-region search needed)."""
        h, w = gray.shape
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
        filtered = cv2.filter2D(gray.astype(np.float64), -1, kernel)
        sigma = np.sqrt(np.abs(np.mean(filtered ** 2)))
        # Normalize to 0–100 scale
        return float(min(sigma / 2.0, 100.0))

    def _composite_score(
        self,
        exposure_ok: bool,
        over_pct: float,
        under_pct: float,
        rms_contrast: float,
        noise: float,
        snr_db: float,
    ) -> float:
        score = 100.0
        if not exposure_ok:
            score -= min((over_pct + under_pct) * 2, 40)
        score -= min(noise * 0.5, 20)
        if snr_db < 20:
            score -= (20 - snr_db) * 1.5
        if rms_contrast < 5:
            score -= (5 - rms_contrast) * 2
        return float(max(score, 0.0))

    @staticmethod
    def _to_gray_8bit(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
