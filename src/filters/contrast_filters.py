"""Contrast, tone, and brightness filters."""

import numpy as np
import cv2
from src.filters.base_filter import BaseFilter, FilterParam


class BrightnessContrastFilter(BaseFilter):
    NAME = "Brightness / Contrast"
    CATEGORY = "Contrast & Tone"

    def _define_params(self):
        self.params["brightness"] = FilterParam("brightness", "Brightness", "int", 0, -255, 255, 1,
            description="Shift all pixel values up (positive) or down (negative). "
                        "+50 makes the image brighter; -50 makes it darker. "
                        "Range: -255 to +255.")
        self.params["contrast"] = FilterParam("contrast", "Contrast", "int", 0, -255, 255, 1,
            description="Expand (positive) or compress (negative) the tonal range. "
                        "Higher contrast makes darks darker and brights brighter. "
                        "Range: -255 to +255.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        b = self.get_param("brightness")
        c = self.get_param("contrast")
        img = image.astype(np.float32)
        img = img * (1 + c / 255.0) + b
        return np.clip(img, 0, 255).astype(np.uint8)


class GammaFilter(BaseFilter):
    NAME = "Gamma Correction"
    CATEGORY = "Contrast & Tone"

    def _define_params(self):
        self.params["gamma"] = FilterParam("gamma", "Gamma", "float", 1.0, 0.1, 5.0, 0.05,
            description="Non-linear brightness adjustment. "
                        "1.0 = no change. "
                        "< 1.0 brightens shadows (useful for underexposed images). "
                        "> 1.0 darkens midtones (useful for overexposed images). "
                        "Typical range: 0.5–2.0.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        g = self.get_param("gamma")
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image if image.dtype == np.uint8 else (image >> 8).astype(np.uint8), lut)


class CLAHEFilter(BaseFilter):
    NAME = "CLAHE"
    CATEGORY = "Contrast & Tone"

    def _define_params(self):
        self.params["clip_limit"] = FilterParam("clip_limit", "Clip Limit", "float", 2.0, 0.5, 10.0, 0.5,
            description="Maximum contrast amplification per tile. "
                        "Higher = more aggressive enhancement but may introduce noise. "
                        "Lower = gentler, more natural result. "
                        "Typical: 2.0–4.0. Used in medical and industrial inspection.")
        self.params["tile_size"] = FilterParam("tile_size", "Tile Size", "int", 8, 4, 32, 4,
            description="Size of local region for contrast computation (pixels). "
                        "Smaller tiles = more local adaptation, finds fine low-contrast detail. "
                        "Larger tiles = more global, smoother result. "
                        "Typical: 8×8. Use 4 for very fine defects.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        clip = self.get_param("clip_limit")
        tile = self.get_param("tile_size")
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        img8 = (image >> 8).astype(np.uint8) if image.dtype == np.uint16 else image.astype(np.uint8)
        if img8.ndim == 3:
            lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return clahe.apply(img8)


class HistogramEqualizationFilter(BaseFilter):
    NAME = "Histogram Equalization"
    CATEGORY = "Contrast & Tone"

    def _define_params(self):
        pass

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = (image >> 8).astype(np.uint8) if image.dtype == np.uint16 else image.astype(np.uint8)
        if img8.ndim == 3:
            yuv = cv2.cvtColor(img8, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        return cv2.equalizeHist(img8)


class LevelsFilter(BaseFilter):
    NAME = "Levels"
    CATEGORY = "Contrast & Tone"

    def _define_params(self):
        self.params["in_black"] = FilterParam("in_black", "Input Black", "int", 0, 0, 254, 1,
            description="Set the darkest input value to map to pure black (0). "
                        "Pixels at or below this value become black. "
                        "Increase to clip dark areas and boost contrast in bright regions.")
        self.params["in_white"] = FilterParam("in_white", "Input White", "int", 255, 1, 255, 1,
            description="Set the brightest input value to map to pure white (255). "
                        "Pixels at or above this value become white. "
                        "Decrease to clip bright areas and reveal detail in dark regions.")
        self.params["gamma"] = FilterParam("gamma", "Midtones", "float", 1.0, 0.1, 5.0, 0.05,
            description="Adjust the midpoint of the tonal range without clipping. "
                        "< 1.0 brightens midtones. > 1.0 darkens midtones.")
        self.params["out_black"] = FilterParam("out_black", "Output Black", "int", 0, 0, 254, 1,
            description="Minimum output brightness. Raise to prevent pure black output "
                        "(useful for display calibration or preventing crushed shadows).")
        self.params["out_white"] = FilterParam("out_white", "Output White", "int", 255, 1, 255, 1,
            description="Maximum output brightness. Lower to prevent pure white output "
                        "(useful for display calibration or preventing blown highlights).")

    def apply(self, image: np.ndarray) -> np.ndarray:
        ib = self.get_param("in_black")
        iw = self.get_param("in_white")
        g  = self.get_param("gamma")
        ob = self.get_param("out_black")
        ow = self.get_param("out_white")
        img = image.astype(np.float32)
        img = np.clip((img - ib) / max(iw - ib, 1), 0, 1)
        img = np.power(img, 1.0 / g)
        img = img * (ow - ob) + ob
        return np.clip(img, 0, 255).astype(np.uint8)


class NormalizeFilter(BaseFilter):
    NAME = "Normalize"
    CATEGORY = "Contrast & Tone"

    def _define_params(self):
        pass

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx == mn:
            return image
        return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


class InvertFilter(BaseFilter):
    NAME = "Invert"
    CATEGORY = "Contrast & Tone"

    def _define_params(self):
        pass

    def apply(self, image: np.ndarray) -> np.ndarray:
        max_val = 65535 if image.dtype == np.uint16 else 255
        return (max_val - image.astype(np.int32)).clip(0, max_val).astype(image.dtype)
