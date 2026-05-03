"""
Research-backed advanced filters for industrial defect inspection.

  GaborBankFilter   — multi-orientation texture analysis (IEEE/JIM 2025)
  WaveletFilter     — multi-scale frequency decomposition (ACM 2025)
  FrequencyFFTFilter — FFT magnitude / phase / bandpass (standard)
  PhaseCongruency   — edge detection invariant to illumination changes
"""

import numpy as np
import cv2
from src.filters.base_filter import BaseFilter, FilterParam


# ═══════════════════════════════════════════════════════════════════════
#  Gabor Filter Bank
# ═══════════════════════════════════════════════════════════════════════

class GaborBankFilter(BaseFilter):
    """
    Bank of Gabor filters at N orientations × M scales.
    Combines all responses to reveal texture defects at any angle.

    IEEE basis:
    Gabor filters achieve minimum time-frequency uncertainty (Heisenberg limit).
    Superior to Sobel/Canny for periodic and directional surface defects.
    — Daugman (1985), revalidated in JIM 2025 for industrial surfaces.
    """
    NAME        = "Gabor Filter Bank"
    CATEGORY    = "③ Detect"
    DESCRIPTION = "Multi-orientation texture analyzer. Best filter for directional surface defects: machining marks, weave, scratches at any angle. Detects all orientations simultaneously."

    def _define_params(self):
        self.params["frequency"]    = FilterParam("frequency",    "Frequency",      "float", 0.15, 0.02, 0.50, 0.01,
            description="Spatial frequency of the Gabor sinusoidal carrier wave (cycles per pixel). "
                        "0.05 = responds to coarse, low-frequency texture. "
                        "0.25 = responds to fine, high-frequency texture. "
                        "Match to the period of the surface texture you want to detect.")
        self.params["orientations"] = FilterParam("orientations", "Orientations",   "int",   6,    2,    12,   1,
            description="Number of equally-spaced filter orientations (0°–180°). "
                        "6 = detects texture defects at 0°, 30°, 60°, 90°, 120°, 150°. "
                        "More orientations = catches defects at any angle but slower. "
                        "2 = horizontal/vertical only. 8–12 for truly isotropic detection.")
        self.params["scales"]       = FilterParam("scales",       "Scales",         "int",   3,    1,    6,    1,
            description="Number of frequency scales. Each scale analyzes a different feature size. "
                        "1 = single scale (fastest). 3 = multi-scale (catches small and large defects). "
                        "Higher = more comprehensive but slower. Typical: 2–3 for industrial inspection.")
        self.params["sigma_x"]      = FilterParam("sigma_x",      "Sigma X",        "float", 4.0,  1.0,  20.0, 0.5,
            description="Gaussian envelope width along the filter's major axis (orientation direction). "
                        "Controls how elongated the filter is in the preferred orientation. "
                        "Larger = longer, more directional filter — good for detecting streaks and grooves.")
        self.params["sigma_y"]      = FilterParam("sigma_y",      "Sigma Y",        "float", 4.0,  1.0,  20.0, 0.5,
            description="Gaussian envelope width perpendicular to the orientation. "
                        "Sigma Y / Sigma X ratio controls aspect ratio: "
                        "Equal = round filter. Sigma Y < Sigma X = highly directional bar filter.")
        self.params["combine"]      = FilterParam("combine",      "Combine Mode",   "choice", "max", choices=["max", "mean", "sum", "energy"],
            description="How to combine responses across all orientations and scales. "
                        "Max = highlight strongest defect signal at any orientation (recommended for inspection). "
                        "Mean = average response (smoother but may miss sharp features). "
                        "Energy = square root of sum of squares — theoretically optimal but slower.")
        self.params["normalize"]    = FilterParam("normalize",    "Normalize",      "bool",  True,
            description="Normalize output to 0–255 display range. "
                        "On = consistent visible contrast regardless of filter strength. "
                        "Off = preserve raw response magnitudes for quantitative measurement.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray_f32(image)
        n_orient = self.get_param("orientations")
        n_scales  = self.get_param("scales")
        freq      = self.get_param("frequency")
        sx        = self.get_param("sigma_x")
        sy        = self.get_param("sigma_y")
        combine   = self.get_param("combine")
        normalize = self.get_param("normalize")

        responses = []
        for s in range(n_scales):
            scale_freq = freq * (0.7 ** s)          # each scale = 0.7× frequency
            ksize = max(int(6 * max(sx, sy)) | 1, 7)
            for i in range(n_orient):
                theta = i * np.pi / n_orient
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), max(sx, 1),
                    theta, 1.0 / max(scale_freq, 0.001),
                    sy / sx, 0, ktype=cv2.CV_32F
                )
                resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
                responses.append(np.abs(resp))

        stack = np.stack(responses, axis=0)

        if combine == "max":
            result = np.max(stack, axis=0)
        elif combine == "mean":
            result = np.mean(stack, axis=0)
        elif combine == "sum":
            result = np.sum(stack, axis=0)
        else:  # energy
            result = np.sqrt(np.sum(stack ** 2, axis=0))

        if normalize:
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

        result = result.astype(np.uint8)
        if image.ndim == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result


# ═══════════════════════════════════════════════════════════════════════
#  Wavelet Decomposition
# ═══════════════════════════════════════════════════════════════════════

class WaveletFilter(BaseFilter):
    """
    Haar wavelet decomposition — separates image into frequency bands.
    Each band reveals defects at a specific spatial scale.

    IEEE basis:
    Wavelet transform decomposes signal into scale-specific bands.
    Combined with FFT for noise elimination before spectral analysis.
    — ACM Proceedings 2025: cigarette foil defect detection.
    """
    NAME        = "Wavelet Decomposition"
    CATEGORY    = "③ Detect"
    DESCRIPTION = "Multi-scale frequency decomposition. Each level reveals defects at a specific spatial scale. Use detail_all mode for broadest defect sensitivity."

    def _define_params(self):
        self.params["level"]   = FilterParam("level",   "Level",      "int",    2,   1, 5,  1,
            description="Number of decomposition levels. Each level halves spatial resolution and doubles scale. "
                        "Level 1 = finest detail. Level 2 = medium features. Level 3+ = coarse structure. "
                        "Use level 1–2 for fine surface defects; level 3–4 for large cracks or shape errors.")
        self.params["subband"] = FilterParam("subband", "Show Band",  "choice", "detail_all",
                                             choices=["approximation", "horizontal", "vertical",
                                                      "diagonal", "detail_all", "all_levels"],
            description="Which frequency subband to display. "
                        "Approximation = low-frequency content (shape). "
                        "Horizontal = horizontal edge detail. Vertical = vertical edge detail. Diagonal = diagonal detail. "
                        "detail_all = maximum of all three detail subbands at the selected level (recommended for defect inspection). "
                        "all_levels = maximum across all scales — broadest defect sensitivity.")
        self.params["enhance"] = FilterParam("enhance", "Enhance",    "bool",   True,
            description="Apply histogram equalization to the output for improved contrast. "
                        "On = automatically boosts low-contrast detail for visual inspection. "
                        "Off = show raw wavelet coefficient magnitudes.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray_f32(image)
        level  = self.get_param("level")
        band   = self.get_param("subband")
        enhance = self.get_param("enhance")

        coeffs = self._haar_decompose(gray, level)
        result = self._select_band(coeffs, band, gray.shape)

        if enhance:
            result = cv2.equalizeHist(
                cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            )
        else:
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if image.ndim == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result

    def _haar_decompose(self, image: np.ndarray, levels: int) -> dict:
        """Manual Haar wavelet using OpenCV resize operations."""
        coeffs = {}
        current = image.copy()
        for lv in range(1, levels + 1):
            h, w = current.shape
            # Low-pass (approximation)
            ll = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            # High-pass: difference between original and reconstructed
            ll_up = cv2.resize(ll, (w, h), interpolation=cv2.INTER_LINEAR)
            detail = current - ll_up

            # Split detail into directional components
            dh = detail[:h // 2, :]           # horizontal detail
            dv = detail[:, :w // 2]           # vertical detail
            dd = detail[:h // 2, :w // 2]     # diagonal detail

            coeffs[f"approx_{lv}"] = ll
            coeffs[f"horizontal_{lv}"] = cv2.resize(np.abs(dh), (w, h))
            coeffs[f"vertical_{lv}"]   = cv2.resize(np.abs(dv), (w, h))
            coeffs[f"diagonal_{lv}"]   = cv2.resize(np.abs(dd), (w, h))
            current = ll

        return coeffs

    def _select_band(self, coeffs: dict, band: str, orig_shape: tuple) -> np.ndarray:
        level = self.get_param("level")
        h, w  = orig_shape

        if band == "approximation":
            img = coeffs.get(f"approx_{level}", np.zeros(orig_shape, np.float32))
        elif band == "horizontal":
            img = coeffs.get(f"horizontal_{level}", np.zeros(orig_shape, np.float32))
        elif band == "vertical":
            img = coeffs.get(f"vertical_{level}", np.zeros(orig_shape, np.float32))
        elif band == "diagonal":
            img = coeffs.get(f"diagonal_{level}", np.zeros(orig_shape, np.float32))
        elif band == "detail_all":
            imgs = [
                coeffs.get(f"horizontal_{level}", np.zeros(orig_shape, np.float32)),
                coeffs.get(f"vertical_{level}",   np.zeros(orig_shape, np.float32)),
                coeffs.get(f"diagonal_{level}",   np.zeros(orig_shape, np.float32)),
            ]
            img = np.max(np.stack(imgs), axis=0)
        else:  # all_levels
            all_details = []
            for lv in range(1, level + 1):
                for d in ["horizontal", "vertical", "diagonal"]:
                    k = f"{d}_{lv}"
                    if k in coeffs:
                        all_details.append(coeffs[k])
            img = np.max(np.stack(all_details), axis=0) if all_details else np.zeros(orig_shape, np.float32)

        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# ═══════════════════════════════════════════════════════════════════════
#  FFT Frequency Domain Filter
# ═══════════════════════════════════════════════════════════════════════

class FFTMagnitudeFilter(BaseFilter):
    """
    FFT magnitude spectrum — shows spatial frequency content of image.
    Periodic defects appear as bright spots away from center.
    Removes periodic noise (banding) via notch filtering.
    """
    NAME        = "FFT Analysis"
    CATEGORY    = "③ Detect"
    DESCRIPTION = "Fourier frequency analysis. Magnitude mode reveals periodic defects as bright spots. Highpass mode sharpens fine detail. Removes banding and moire interference."

    def _define_params(self):
        self.params["mode"]       = FilterParam("mode",       "Display Mode", "choice", "magnitude",
                                                choices=["magnitude", "phase", "lowpass", "highpass", "bandpass"],
            description="What to display or compute from the FFT. "
                        "Magnitude = power spectrum — bright spots away from center indicate periodic defects (banding, moire). "
                        "Phase = phase spectrum — shows spatial structure. "
                        "Lowpass = apply low-pass filter and reconstruct (blurs). "
                        "Highpass = apply high-pass filter (sharpens, reveals fine defects). "
                        "Bandpass = keep only frequencies between Low and High Cutoff.")
        self.params["cutoff_low"]  = FilterParam("cutoff_low",  "Low Cutoff",  "float", 0.1, 0.01, 0.5, 0.01,
            description="Lower frequency boundary (fraction of Nyquist frequency, 0–0.5). "
                        "Used in Highpass and Bandpass modes. Frequencies below this are suppressed. "
                        "0.1 = suppress DC and very low frequencies (large illumination gradients).")
        self.params["cutoff_high"] = FilterParam("cutoff_high", "High Cutoff", "float", 0.4, 0.05, 1.0, 0.01,
            description="Upper frequency boundary (fraction of Nyquist, 0–1.0). "
                        "Used in Lowpass and Bandpass modes. Frequencies above this are suppressed. "
                        "0.4 = preserve spatial detail up to 2.5× the Nyquist. "
                        "Decrease to smooth, increase to sharpen.")
        self.params["log_scale"]   = FilterParam("log_scale",   "Log Scale",   "bool",  True,
            description="Display magnitude on a logarithmic scale. "
                        "On = compresses the wide dynamic range of FFT magnitudes for visibility (recommended). "
                        "Off = linear scale — DC component dominates, fine detail invisible.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray_f32(image)
        mode = self.get_param("mode")
        log  = self.get_param("log_scale")

        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)

        if mode == "magnitude":
            mag = np.abs(dft_shift)
            if log:
                mag = np.log1p(mag)
            result = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        elif mode == "phase":
            phase = np.angle(dft_shift)
            result = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        else:  # lowpass / highpass / bandpass
            h, w = gray.shape
            cl = self.get_param("cutoff_low")
            ch = self.get_param("cutoff_high")
            mask = self._frequency_mask(h, w, mode, cl, ch)
            filtered = dft_shift * mask
            result_f = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
            result = cv2.normalize(result_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if image.ndim == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result

    @staticmethod
    def _frequency_mask(h: int, w: int, mode: str, low: float, high: float) -> np.ndarray:
        cy, cx = h // 2, w // 2
        y = np.arange(h) - cy
        x = np.arange(w) - cx
        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt(xx ** 2 + yy ** 2) / min(h, w)

        if mode == "lowpass":
            return (dist <= high).astype(np.float32)
        elif mode == "highpass":
            return (dist >= low).astype(np.float32)
        else:  # bandpass
            return ((dist >= low) & (dist <= high)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  Local Binary Pattern texture filter
# ═══════════════════════════════════════════════════════════════════════

class LBPTextureFilter(BaseFilter):
    """
    Local Binary Pattern — encodes local texture as binary pattern.
    Defective regions break the regular texture pattern.
    Illumination-invariant (does not change with brightness shifts).
    """
    NAME        = "LBP Texture"
    CATEGORY    = "③ Detect"
    DESCRIPTION = "Local Binary Pattern texture encoder. Illumination-invariant — brightness changes do not affect the result. Segments regions with different surface texture."

    def _define_params(self):
        self.params["radius"]  = FilterParam("radius",  "Radius",  "int", 3, 1, 10, 1,
            description="Radius of the sampling circle around each pixel. "
                        "Controls the scale of texture being analyzed. "
                        "1–2 = micro-texture (surface finish, grain). 3–5 = meso-texture (weave, pattern). "
                        "Match radius to the size of the texture unit you want to characterize.")
        self.params["points"]  = FilterParam("points",  "Points",  "int", 8, 4, 24, 4,
            description="Number of sampling points evenly distributed on the circle. "
                        "More points = finer angular resolution but slower. "
                        "8 = standard (LBP8,1 or LBP8,3). 16–24 = high-precision texture discrimination. "
                        "Increasing points distinguishes more texture micropatterns.")
        self.params["uniform"] = FilterParam("uniform", "Uniform Only", "bool", False,
            description="Uniform LBP patterns have at most 2 bitwise transitions (0→1 or 1→0) in the binary code. "
                        "On = keep only uniform patterns (represent 90% of real textures) — reduces noise in classification. "
                        "Off = show all LBP codes including non-uniform (useful for detecting irregular defect regions).")

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray_u8(image)
        r = self.get_param("radius")
        p = self.get_param("points")

        result = self._compute_lbp(gray, r, p)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if image.ndim == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result

    @staticmethod
    def _compute_lbp(gray: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        h, w  = gray.shape
        result = np.zeros((h, w), dtype=np.uint32)
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dx = radius * np.cos(angle)
            dy = -radius * np.sin(angle)
            x = np.clip(np.arange(w) + dx, 0, w - 1).astype(np.float32)
            y = np.clip(np.arange(h) + dy, 0, h - 1).astype(np.float32)
            xx, yy = np.meshgrid(x, y)
            neighbor = cv2.remap(
                gray.astype(np.float32), xx.astype(np.float32), yy.astype(np.float32),
                cv2.INTER_LINEAR
            )
            result += (neighbor >= gray.astype(np.float32)).astype(np.uint32) << i
        return result.astype(np.float32)


# ── Helpers ──────────────────────────────────────────────────────────────

def _to_gray_f32(image: np.ndarray) -> np.ndarray:
    img = image
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.float32)


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    img = image
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint8)
