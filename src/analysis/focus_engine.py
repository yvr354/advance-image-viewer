"""
Focus / sharpness engine — industrial grade.

Techniques used (all peer-reviewed, used in Cognex / Keyence / Halcon):
  Laplacian Variance  — Var(∇²f)          high-freq energy via 2nd derivative
  Tenengrad           — mean(Gx²+Gy²)     gradient magnitude via Sobel
  Brenner Gradient    — Σ(f(x+2)-f(x))²   2-pixel diff, fast, good for smooth surfaces

Per-cell scoring:
  Each metric is computed per grid cell, then BOTH metrics are normalized
  0–100 relative to the best cell in THIS image (not an absolute threshold).
  Fusion: score = 0.55*lap + 0.45*ten   (Laplacian slightly favored — less noise-sensitive)

Verdicts are relative (% of the sharpest cell):
  SHARP   ≥ 72 %   — good for inspection
  SOFT    ≥ 38 %   — marginal — re-check lens
  BLURRY  <  38 %  — reject / refocus

Tilt detection:
  Linear regression over row-means and col-means.
  If slope is significant (>5 pts/side), part or lens may be tilted.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Tuple


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class FocusGridData:
    """Rich per-cell grid produced by FocusEngine.analyze()."""
    scores:    np.ndarray    # (rows, cols) float32, 0–100 relative
    raw_lap:   np.ndarray    # (rows, cols) raw laplacian per cell
    raw_ten:   np.ndarray    # (rows, cols) raw tenengrad per cell
    rows:      int
    cols:      int
    img_h:     int
    img_w:     int
    best_cell:  Tuple[int, int]   # (row, col) of highest score
    worst_cell: Tuple[int, int]   # (row, col) of lowest score
    pct_sharp:  float
    pct_soft:   float
    pct_blurry: float
    tilt_h:     float   # horizontal tilt slope (pts per 100px right)
    tilt_v:     float   # vertical tilt slope (pts per 100px down)
    tilt_warn:  str     # human-readable tilt description, "" if none


@dataclass
class FocusResult:
    score:    float          # 0–1000 (absolute whole-image score)
    verdict:  str            # PERFECT / GOOD / SOFT / BLURRY
    heatmap:  np.ndarray     # (rows, cols) float32 raw scores (legacy compat)
    metric:   str
    grid:     FocusGridData  # full rich grid for UI rendering


# ── Engine ─────────────────────────────────────────────────────────────────

class FocusEngine:
    PERFECT_THRESHOLD = 700
    GOOD_THRESHOLD    = 400
    SOFT_THRESHOLD    = 200

    SHARP_PCT  = 72.0   # relative % threshold
    SOFT_PCT   = 38.0
    TILT_SLOPE = 5.0    # pts/side to flag tilt

    def __init__(self, metric: str = "laplacian", grid: int = 8):
        self.metric = metric
        self.grid   = max(4, grid)   # min 4×4

    def set_thresholds(self, perfect: int, good: int, soft: int):
        self.PERFECT_THRESHOLD = perfect
        self.GOOD_THRESHOLD    = good
        self.SOFT_THRESHOLD    = soft

    # ── Public API ──────────────────────────────────────────────────────────

    def analyze(self, image: np.ndarray) -> FocusResult:
        gray        = self._to_gray_8bit(image)
        whole_score = self._laplacian(gray)
        grid_data   = self._build_grid(gray)
        verdict     = self._verdict(whole_score)
        return FocusResult(
            score   = whole_score,
            verdict = verdict,
            heatmap = grid_data.raw_lap,
            metric  = "laplacian+tenengrad fusion",
            grid    = grid_data,
        )

    def score_only(self, image: np.ndarray) -> float:
        gray = self._to_gray_8bit(image)
        return self._laplacian(gray)

    def best_frame(self, frames) -> Tuple[int, float]:
        scores = [self.score_only(f) for f in frames]
        idx = int(np.argmax(scores))
        return idx, scores[idx]

    # ── Whole-image metrics ─────────────────────────────────────────────────

    @staticmethod
    def _laplacian(gray: np.ndarray) -> float:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(min(lap.var() / 5.0, 1000.0))

    @staticmethod
    def _tenengrad(gray: np.ndarray) -> float:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(min(np.mean(sx**2 + sy**2) / 50.0, 1000.0))

    @staticmethod
    def _brenner(gray: np.ndarray) -> float:
        d = gray.astype(np.float64)
        return float(min(np.sum((d[:, 2:] - d[:, :-2])**2) / gray.size / 2.0, 1000.0))

    # ── Grid analysis ───────────────────────────────────────────────────────

    def _build_grid(self, gray: np.ndarray) -> FocusGridData:
        h, w = gray.shape
        R = min(self.grid, h)
        C = min(self.grid, w)

        lap = np.zeros((R, C), dtype=np.float32)
        ten = np.zeros((R, C), dtype=np.float32)

        # Compute both metrics per cell
        for r in range(R):
            for c in range(C):
                y0 = int(r * h / R);  y1 = int((r+1) * h / R)
                x0 = int(c * w / C);  x1 = int((c+1) * w / C)
                cell = gray[y0:y1, x0:x1]
                if cell.size < 16:
                    continue
                lap[r, c] = self._laplacian(cell)
                ten[r, c] = self._tenengrad(cell)

        # Normalize each metric 0-1 relative to its own max in this image
        lap_max = lap.max() if lap.max() > 0 else 1.0
        ten_max = ten.max() if ten.max() > 0 else 1.0
        lap_n = lap / lap_max
        ten_n = ten / ten_max

        # Fuse: weighted combination → 0–100 relative score
        fused = (0.55 * lap_n + 0.45 * ten_n) * 100.0

        # Best / worst cells
        best_flat  = int(np.argmax(fused))
        worst_flat = int(np.argmin(fused))
        best_cell  = (best_flat  // C, best_flat  % C)
        worst_cell = (worst_flat // C, worst_flat % C)

        # Per-cell verdict counts
        sharp_mask  = fused >= self.SHARP_PCT
        soft_mask   = (fused >= self.SOFT_PCT) & ~sharp_mask
        blurry_mask = fused < self.SOFT_PCT
        total = R * C
        pct_sharp  = 100.0 * sharp_mask.sum()  / total
        pct_soft   = 100.0 * soft_mask.sum()   / total
        pct_blurry = 100.0 * blurry_mask.sum() / total

        # Tilt detection via linear regression on row/col means
        row_means = fused.mean(axis=1)   # shape (R,)
        col_means = fused.mean(axis=0)   # shape (C,)
        tilt_v = self._tilt_slope(row_means)
        tilt_h = self._tilt_slope(col_means)
        tilt_warn = self._tilt_description(tilt_h, tilt_v)

        return FocusGridData(
            scores    = fused.astype(np.float32),
            raw_lap   = lap,
            raw_ten   = ten,
            rows      = R,
            cols      = C,
            img_h     = h,
            img_w     = w,
            best_cell  = best_cell,
            worst_cell = worst_cell,
            pct_sharp  = pct_sharp,
            pct_soft   = pct_soft,
            pct_blurry = pct_blurry,
            tilt_h     = tilt_h,
            tilt_v     = tilt_v,
            tilt_warn  = tilt_warn,
        )

    @staticmethod
    def _tilt_slope(means: np.ndarray) -> float:
        """Slope of linear fit in pts per full span (pts/side)."""
        n = len(means)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        # polyfit degree 1
        coeffs = np.polyfit(x, means, 1)
        return float(coeffs[0] * (n - 1))   # total drop/rise left→right

    def _tilt_description(self, h_slope: float, v_slope: float) -> str:
        parts = []
        th = self.TILT_SLOPE
        if abs(h_slope) > th:
            if h_slope > 0:
                parts.append("sharper on RIGHT — part/lens tilted left→right")
            else:
                parts.append("sharper on LEFT — part/lens tilted right→left")
        if abs(v_slope) > th:
            if v_slope > 0:
                parts.append("sharper at BOTTOM — part/lens tilted top→bottom")
            else:
                parts.append("sharper at TOP — part/lens tilted bottom→top")
        return "  |  ".join(parts)

    # ── Legacy heatmap RGB (kept for backward compat) ──────────────────────

    def heatmap_to_rgb(self, heatmap: np.ndarray, max_score: float = None) -> np.ndarray:
        # Normalize relative to THIS image's best cell — not a fixed scale.
        # A blurry image will still show variation (best region = green, worst = red).
        peak = float(heatmap.max()) if heatmap.max() > 0 else 1.0
        norm = np.clip(heatmap / peak, 0, 1).astype(np.float32)
        # Green → yellow → red  (high score = green = sharp)
        r = np.clip(2.0 * (1.0 - norm), 0, 1)
        g = np.clip(2.0 * norm,         0, 1)
        b = np.zeros_like(norm)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        return rgb

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _verdict(self, score: float) -> str:
        if score >= self.PERFECT_THRESHOLD: return "PERFECT"
        if score >= self.GOOD_THRESHOLD:    return "GOOD"
        if score >= self.SOFT_THRESHOLD:    return "SOFT"
        return "BLURRY"

    @staticmethod
    def _to_gray_8bit(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image.astype(np.uint8)
