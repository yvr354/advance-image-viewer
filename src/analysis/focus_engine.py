"""
Focus / sharpness engine — industrial grade.

Metrics (peer-reviewed — Pertuz et al. 2013, Pattern Recognition 46(5)):
  Laplacian Variance  — Var(∇²I)          best at low noise; most used in industry
  Tenengrad           — Σ(Gx²+Gy²)        best noise robustness — Pertuz #1 ranking
  Brenner Gradient    — Σ(I[x+2]-I[x])²   fastest; good validation metric on clean cameras

Scoring modes:
  RELATIVE    — cell score = cell_raw / image_best_cell_raw × 100
                Useful for seeing spatial variation. Meaningless for absolute judgment.
                A completely blurry image can score 100% in its best cell. DO NOT trust
                this for pass/fail decisions.

  AUTO_REF    — cell score = cell_raw / session_best_cell_raw × 100
                Automatically uses the sharpest image seen this session as reference.
                More honest. Still session-dependent.

  LOCKED_REF  — cell score = cell_raw / locked_reference_cell_raw × 100
                Uses a deliberately captured reference image as baseline.
                100% = as sharp as your calibrated reference. This is production-grade.

Confidence:
  HIGH   — LOCKED_REF mode AND Laplacian verdict agrees with Tenengrad verdict
  MEDIUM — AUTO_REF mode, OR metrics are one category apart
  LOW    — RELATIVE mode only (no reference)

Verdict thresholds (apply to % of reference OR absolute score in RELATIVE mode):
  SHARP   ≥ 72%  — acceptable for defect inspection
  SOFT    ≥ 38%  — marginal — review lens/distance/vibration
  BLURRY  < 38%  — reject — do not use for inspection

Tilt detection:
  Linear regression over row-means and col-means (np.polyfit degree 1).
  Slope > TILT_SLOPE pts/span → field curvature or part/camera tilt.
"""

import json
import datetime
import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from src.analysis.mask_engine import MaskData


# ── Reference image ────────────────────────────────────────────────────────

@dataclass
class FocusReference:
    """
    Per-cell raw scores captured from a known-good (sharp) image.
    Used as the denominator for reference-based scoring.
    """
    raw_lap:   np.ndarray   # (rows, cols) raw Laplacian variance per cell
    raw_ten:   np.ndarray   # (rows, cols) raw Tenengrad energy per cell
    whole_lap: float        # whole-image Laplacian variance (unscaled)
    whole_ten: float        # whole-image Tenengrad energy (unscaled)
    rows:      int
    cols:      int
    source:    str = ""     # filename of the source image
    locked_at: str = ""     # ISO timestamp when locked
    mode:      str = "auto" # "auto" | "locked"

    def save(self, path: str):
        data = {
            "raw_lap":   self.raw_lap.tolist(),
            "raw_ten":   self.raw_ten.tolist(),
            "whole_lap": self.whole_lap,
            "whole_ten": self.whole_ten,
            "rows":      self.rows,
            "cols":      self.cols,
            "source":    self.source,
            "locked_at": self.locked_at,
            "mode":      self.mode,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FocusReference":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            raw_lap   = np.array(d["raw_lap"],  dtype=np.float32),
            raw_ten   = np.array(d["raw_ten"],  dtype=np.float32),
            whole_lap = float(d["whole_lap"]),
            whole_ten = float(d["whole_ten"]),
            rows      = int(d["rows"]),
            cols      = int(d["cols"]),
            source    = d.get("source",    ""),
            locked_at = d.get("locked_at", ""),
            mode      = d.get("mode",      "locked"),
        )


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class FocusGridData:
    """Rich per-cell grid produced by FocusEngine.analyze()."""
    scores:    np.ndarray          # (rows, cols) float32, 0–100
    raw_lap:   np.ndarray          # (rows, cols) raw Laplacian variance per cell
    raw_ten:   np.ndarray          # (rows, cols) raw Tenengrad per cell
    rows:      int
    cols:      int
    img_h:     int
    img_w:     int
    best_cell:  Tuple[int, int]
    worst_cell: Tuple[int, int]
    pct_sharp:  float
    pct_soft:   float
    pct_blurry: float
    tilt_h:     float
    tilt_v:     float
    tilt_warn:  str
    # Scoring context
    scoring_mode: str = "RELATIVE"   # RELATIVE | AUTO_REF | LOCKED_REF


@dataclass
class FocusResult:
    score:    float          # 0–1000 whole-image Laplacian score (absolute)
    verdict:  str            # PERFECT / GOOD / SOFT / BLURRY
    heatmap:  np.ndarray     # (rows, cols) raw_lap for legacy heatmap rendering
    metric:   str
    grid:     FocusGridData
    # ── Honest reporting fields ───────────────────────────────────────────
    raw_lap:      float = 0.0       # unscaled Laplacian variance (whole image)
    raw_ten:      float = 0.0       # unscaled Tenengrad energy (whole image)
    raw_brenner:  float = 0.0       # unscaled Brenner gradient (whole image)
    confidence:   str   = "LOW"     # HIGH | MEDIUM | LOW
    scoring_mode: str   = "RELATIVE"
    ref_pct:      float = 0.0       # % of reference whole-image score (0 = no ref)
    ref_source:   str   = ""        # filename of the reference image


# ── Engine ─────────────────────────────────────────────────────────────────

class FocusEngine:

    # Whole-image absolute verdict thresholds (Laplacian variance / 5)
    PERFECT_THRESHOLD = 700
    GOOD_THRESHOLD    = 400
    SOFT_THRESHOLD    = 200

    # Per-cell % thresholds (apply to both RELATIVE and REF modes)
    SHARP_PCT  = 72.0
    SOFT_PCT   = 38.0
    TILT_SLOPE = 5.0

    def __init__(self, metric: str = "laplacian", grid: int = 8):
        self.metric = metric
        self.grid   = max(4, grid)

    def set_thresholds(self, perfect: int, good: int, soft: int):
        self.PERFECT_THRESHOLD = perfect
        self.GOOD_THRESHOLD    = good
        self.SOFT_THRESHOLD    = soft

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(self, image: np.ndarray,
                reference: Optional[FocusReference] = None,
                mask=None) -> FocusResult:
        """
        Analyze focus. If reference is provided, scores are % of reference.
        If mask (MaskData) is provided, metrics are computed only inside the mask.
        Without reference: RELATIVE mode — honest but limited.
        """
        gray      = self._to_gray_8bit(image)
        mask_arr  = mask.to_array(gray.shape[:2]) if mask is not None else None

        # Whole-image raw metrics (unscaled — used for honest reporting)
        raw_lap     = self._raw_laplacian(gray,  mask_arr)
        raw_ten     = self._raw_tenengrad(gray,  mask_arr)
        raw_brenner = self._raw_brenner(gray,    mask_arr)

        # Whole-image score (Laplacian scaled to 0–1000 for verdict)
        whole_score = float(min(raw_lap / 5.0, 1000.0))
        verdict     = self._verdict(whole_score)

        # Build reference-aware, mask-aware grid
        grid_data = self._build_grid(gray, reference, mask_arr)

        # Confidence: how much to trust this verdict
        scoring_mode = grid_data.scoring_mode
        confidence   = self._compute_confidence(raw_lap, raw_ten, scoring_mode)

        # Reference percentage (whole-image)
        ref_pct    = 0.0
        ref_source = ""
        if reference is not None and reference.whole_lap > 0:
            ref_pct    = min(raw_lap / reference.whole_lap * 100.0, 100.0)
            ref_source = reference.source

        return FocusResult(
            score        = whole_score,
            verdict      = verdict,
            heatmap      = grid_data.raw_lap,
            metric       = "Laplacian+Tenengrad fusion (Pertuz 2013)",
            grid         = grid_data,
            raw_lap      = raw_lap,
            raw_ten      = raw_ten,
            raw_brenner  = raw_brenner,
            confidence   = confidence,
            scoring_mode = scoring_mode,
            ref_pct      = ref_pct,
            ref_source   = ref_source,
        )

    def make_reference(self, image: np.ndarray,
                       source: str = "", mode: str = "locked") -> FocusReference:
        """Build a FocusReference from a known-good image."""
        gray = self._to_gray_8bit(image)
        R = min(self.grid, gray.shape[0])
        C = min(self.grid, gray.shape[1])
        h, w = gray.shape

        lap = np.zeros((R, C), dtype=np.float32)
        ten = np.zeros((R, C), dtype=np.float32)
        for r in range(R):
            for c in range(C):
                y0, y1 = int(r*h/R), int((r+1)*h/R)
                x0, x1 = int(c*w/C), int((c+1)*w/C)
                cell = gray[y0:y1, x0:x1]
                if cell.size < 16:
                    continue
                lap[r, c] = self._raw_laplacian(cell)
                ten[r, c] = self._raw_tenengrad(cell)

        return FocusReference(
            raw_lap   = lap,
            raw_ten   = ten,
            whole_lap = self._raw_laplacian(gray),
            whole_ten = self._raw_tenengrad(gray),
            rows      = R,
            cols      = C,
            source    = source,
            locked_at = datetime.datetime.now().isoformat(),
            mode      = mode,
        )

    def score_only(self, image: np.ndarray) -> float:
        gray = self._to_gray_8bit(image)
        return float(min(self._raw_laplacian(gray) / 5.0, 1000.0))

    def best_frame(self, frames) -> Tuple[int, float]:
        scores = [self.score_only(f) for f in frames]
        idx = int(np.argmax(scores))
        return idx, scores[idx]

    # ── Whole-image raw metrics (unscaled) ────────────────────────────────

    @staticmethod
    def _raw_laplacian(gray: np.ndarray,
                       mask: np.ndarray = None) -> float:
        """Variance of Laplacian — Pertuz 2013 recommended for low-noise cameras."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        if mask is not None:
            vals = lap[mask.astype(bool)]
            return float(np.var(vals)) if vals.size >= 16 else 0.0
        return float(lap.var())

    @staticmethod
    def _raw_tenengrad(gray: np.ndarray,
                       mask: np.ndarray = None) -> float:
        """Sum of squared Sobel gradients — Pertuz 2013 top overall ranking."""
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        g  = sx**2 + sy**2
        if mask is not None:
            vals = g[mask.astype(bool)]
            return float(np.mean(vals)) if vals.size >= 16 else 0.0
        return float(np.mean(g))

    @staticmethod
    def _raw_brenner(gray: np.ndarray,
                     mask: np.ndarray = None) -> float:
        """Brenner gradient — fastest; good validation on high-SNR cameras."""
        d = gray.astype(np.float64)
        b = (d[:, 2:] - d[:, :-2]) ** 2
        if mask is not None:
            # Only include positions where BOTH columns are valid
            m = mask[:, 2:].astype(bool) & mask[:, :-2].astype(bool)
            vals = b[m]
            return float(np.mean(vals)) if vals.size >= 16 else 0.0
        return float(np.mean(b))

    # Legacy scaled versions (keep backward compat with score_only / batch)
    @staticmethod
    def _laplacian(gray: np.ndarray) -> float:
        return float(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 5.0, 1000.0))

    @staticmethod
    def _tenengrad(gray: np.ndarray) -> float:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(min(np.mean(sx**2 + sy**2) / 50.0, 1000.0))

    # ── Grid analysis ──────────────────────────────────────────────────────

    def _build_grid(self, gray: np.ndarray,
                    reference: Optional[FocusReference] = None,
                    mask: np.ndarray = None) -> FocusGridData:
        h, w = gray.shape
        R = min(self.grid, h)
        C = min(self.grid, w)

        lap     = np.zeros((R, C), dtype=np.float32)
        ten     = np.zeros((R, C), dtype=np.float32)
        invalid = np.zeros((R, C), dtype=bool)   # cells with <15% valid pixels

        for r in range(R):
            for c in range(C):
                y0, y1 = int(r*h/R), int((r+1)*h/R)
                x0, x1 = int(c*w/C), int((c+1)*w/C)
                cell = gray[y0:y1, x0:x1]
                if cell.size < 16:
                    invalid[r, c] = True
                    continue

                cell_mask = mask[y0:y1, x0:x1] if mask is not None else None

                # Skip cell if less than 15% of its pixels are valid
                if cell_mask is not None and cell_mask.mean() < 0.15:
                    invalid[r, c] = True
                    continue

                lap[r, c] = self._raw_laplacian(cell, cell_mask)
                ten[r, c] = self._raw_tenengrad(cell, cell_mask)

        # ── Score computation ────────────────────────────────────────────
        ref_ok = (reference is not None and
                  reference.rows == R and reference.cols == C)

        if ref_ok:
            # Reference-based: how sharp is each cell vs the reference cell?
            # Clamp at 100% — can't score above reference
            EPS = 0.01   # prevent div-by-zero on smooth reference cells
            lap_ratio = np.minimum(lap / np.maximum(reference.raw_lap, EPS), 1.0)
            ten_ratio = np.minimum(ten / np.maximum(reference.raw_ten, EPS), 1.0)
            fused = (0.55 * lap_ratio + 0.45 * ten_ratio) * 100.0
            scoring_mode = "LOCKED_REF" if reference.mode == "locked" else "AUTO_REF"
        else:
            # Relative: normalize to THIS image's own best cell
            lap_max = lap.max() if lap.max() > 0 else 1.0
            ten_max = ten.max() if ten.max() > 0 else 1.0
            fused = (0.55 * (lap / lap_max) + 0.45 * (ten / ten_max)) * 100.0
            scoring_mode = "RELATIVE"

        fused = fused.astype(np.float32)
        # Mark invalid cells with -1 (excluded from best/worst/percentages)
        fused[invalid] = -1.0

        valid_mask = ~invalid
        if valid_mask.any():
            valid_scores = fused[valid_mask]
            best_flat_v  = int(np.argmax(valid_scores))
            worst_flat_v = int(np.argmin(valid_scores))
            valid_idx    = np.argwhere(valid_mask)
            best_cell    = tuple(valid_idx[best_flat_v])
            worst_cell   = tuple(valid_idx[worst_flat_v])
        else:
            best_cell  = (0, 0)
            worst_cell = (0, 0)

        valid_cells = valid_mask.sum()
        if valid_cells > 0:
            sharp_mask  = valid_mask & (fused >= self.SHARP_PCT)
            soft_mask   = valid_mask & (fused >= self.SOFT_PCT) & (fused < self.SHARP_PCT)
            blurry_mask = valid_mask & (fused >= 0) & (fused < self.SOFT_PCT)
            pct_sharp  = 100.0 * sharp_mask.sum()  / valid_cells
            pct_soft   = 100.0 * soft_mask.sum()   / valid_cells
            pct_blurry = 100.0 * blurry_mask.sum() / valid_cells
        else:
            pct_sharp = pct_soft = pct_blurry = 0.0

        row_means = fused.mean(axis=1)
        col_means = fused.mean(axis=0)
        tilt_v    = self._tilt_slope(row_means)
        tilt_h    = self._tilt_slope(col_means)
        tilt_warn = self._tilt_description(tilt_h, tilt_v)

        return FocusGridData(
            scores       = fused,
            raw_lap      = lap,
            raw_ten      = ten,
            rows         = R,
            cols         = C,
            img_h        = h,
            img_w        = w,
            best_cell    = best_cell,
            worst_cell   = worst_cell,
            pct_sharp    = pct_sharp,
            pct_soft     = pct_soft,
            pct_blurry   = pct_blurry,
            tilt_h       = tilt_h,
            tilt_v       = tilt_v,
            tilt_warn    = tilt_warn,
            scoring_mode = scoring_mode,
        )

    # ── Confidence ─────────────────────────────────────────────────────────

    def _compute_confidence(self, raw_lap: float, raw_ten: float,
                             scoring_mode: str) -> str:
        """
        Confidence reflects both the scoring mode and metric agreement.
        Laplacian and Tenengrad use different sensitivity profiles.
        If they disagree on the verdict category → lower confidence.
        """
        # Scale both metrics to comparable 0–1000 range
        lap_score = min(raw_lap / 5.0,  1000.0)
        ten_score = min(raw_ten / 50.0, 1000.0)   # empirical scaling

        lap_verdict = self._verdict(lap_score)
        ten_verdict = self._verdict(ten_score)

        _RANK = {"BLURRY": 0, "SOFT": 1, "GOOD": 2, "PERFECT": 3}
        gap = abs(_RANK[lap_verdict] - _RANK[ten_verdict])
        metrics_agree = (gap == 0)
        metrics_close  = (gap <= 1)

        if scoring_mode == "LOCKED_REF":
            if metrics_agree: return "HIGH"
            if metrics_close: return "MEDIUM"
            return "LOW"
        elif scoring_mode == "AUTO_REF":
            if metrics_agree: return "MEDIUM"
            return "LOW"
        else:  # RELATIVE
            return "LOW"

    # ── Tilt ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tilt_slope(means: np.ndarray) -> float:
        n = len(means)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, means, 1)
        return float(coeffs[0] * (n - 1))

    def _tilt_description(self, h_slope: float, v_slope: float) -> str:
        parts = []
        th = self.TILT_SLOPE
        if abs(h_slope) > th:
            parts.append("sharper on RIGHT" if h_slope > 0
                         else "sharper on LEFT")
        if abs(v_slope) > th:
            parts.append("sharper at BOTTOM" if v_slope > 0
                         else "sharper at TOP")
        return "  |  ".join(parts)

    # ── Heatmap RGB ────────────────────────────────────────────────────────

    def heatmap_to_rgb(self, heatmap: np.ndarray,
                       max_score: float = None) -> np.ndarray:
        peak = float(heatmap.max()) if heatmap.max() > 0 else 1.0
        norm = np.clip(heatmap / peak, 0, 1).astype(np.float32)
        r = np.clip(2.0 * (1.0 - norm), 0, 1)
        g = np.clip(2.0 * norm,         0, 1)
        b = np.zeros_like(norm)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    # ── Helpers ────────────────────────────────────────────────────────────

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
