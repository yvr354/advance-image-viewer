"""
Focus / sharpness scoring engine.
Three metrics — all objective, no human-eye judgment.

  Laplacian variance  — fast, reliable, best for general use
  Tenengrad (Sobel)   — robust on textured surfaces
  Brenner function    — simple, good for smooth surfaces
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FocusResult:
    score: float            # 0 – 1000
    verdict: str            # PERFECT / GOOD / SOFT / BLURRY
    heatmap: np.ndarray     # Float32 NxN, each cell = score
    metric: str             # which metric was used


class FocusEngine:
    PERFECT_THRESHOLD = 700
    GOOD_THRESHOLD    = 400
    SOFT_THRESHOLD    = 200

    def __init__(self, metric: str = "laplacian", grid: int = 8):
        self.metric = metric      # laplacian | tenengrad | brenner
        self.grid = grid          # NxN grid for heatmap

    def set_thresholds(self, perfect: int, good: int, soft: int):
        self.PERFECT_THRESHOLD = perfect
        self.GOOD_THRESHOLD = good
        self.SOFT_THRESHOLD = soft

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def analyze(self, image: np.ndarray) -> FocusResult:
        gray = self._to_gray_8bit(image)
        score = self._score(gray)
        heatmap = self._build_heatmap(gray)
        verdict = self._verdict(score)
        return FocusResult(score=score, verdict=verdict, heatmap=heatmap, metric=self.metric)

    def score_only(self, image: np.ndarray) -> float:
        """Fast path — score only, no heatmap (for live view)."""
        gray = self._to_gray_8bit(image)
        return self._score(gray)

    def best_frame(self, frames: list[np.ndarray]) -> Tuple[int, float]:
        """Return (index, score) of the sharpest frame in a list."""
        scores = [self.score_only(f) for f in frames]
        idx = int(np.argmax(scores))
        return idx, scores[idx]

    # ------------------------------------------------------------------ #
    #  Metric implementations
    # ------------------------------------------------------------------ #

    def _score(self, gray: np.ndarray) -> float:
        if self.metric == "tenengrad":
            return self._tenengrad(gray)
        elif self.metric == "brenner":
            return self._brenner(gray)
        else:
            return self._laplacian(gray)

    @staticmethod
    def _laplacian(gray: np.ndarray) -> float:
        """Variance of Laplacian — higher = sharper."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        score = lap.var()
        # Normalize to 0–1000 scale (empirically calibrated)
        return float(min(score / 5.0, 1000.0))

    @staticmethod
    def _tenengrad(gray: np.ndarray) -> float:
        """Sobel gradient energy — robust on textured surfaces."""
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        score = np.mean(sx ** 2 + sy ** 2)
        return float(min(score / 50.0, 1000.0))

    @staticmethod
    def _brenner(gray: np.ndarray) -> float:
        """Brenner gradient — fast, good for smooth/flat surfaces."""
        diff = gray.astype(np.float64)
        score = np.sum((diff[:, 2:] - diff[:, :-2]) ** 2)
        score /= gray.size
        return float(min(score / 2.0, 1000.0))

    # ------------------------------------------------------------------ #
    #  Heatmap
    # ------------------------------------------------------------------ #

    def _build_heatmap(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        gh = max(1, min(int(self.grid), h))
        gw = max(1, min(int(self.grid), w))
        heatmap = np.zeros((gh, gw), dtype=np.float32)

        for row in range(gh):
            for col in range(gw):
                y0 = int(row * h / gh)
                y1 = int((row + 1) * h / gh)
                x0 = int(col * w / gw)
                x1 = int((col + 1) * w / gw)
                cell = gray[y0:y1, x0:x1]
                heatmap[row, col] = self._score(cell)

        return heatmap

    def heatmap_to_rgb(self, heatmap: np.ndarray, max_score: float = 1000.0) -> np.ndarray:
        """Convert NxN score grid to RGB image for overlay."""
        norm = np.clip(heatmap / max_score, 0, 1).astype(np.float32)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_RdYlGn)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _verdict(self, score: float) -> str:
        if score >= self.PERFECT_THRESHOLD:
            return "PERFECT"
        elif score >= self.GOOD_THRESHOLD:
            return "GOOD"
        elif score >= self.SOFT_THRESHOLD:
            return "SOFT"
        else:
            return "BLURRY"

    @staticmethod
    def _to_gray_8bit(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
