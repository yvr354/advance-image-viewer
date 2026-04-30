"""Contrast, tone, and brightness filters."""

import numpy as np
import cv2
from src.filters.base_filter import BaseFilter, FilterParam


class BrightnessContrastFilter(BaseFilter):
    NAME = "Brightness / Contrast"
    CATEGORY = "Contrast & Tone"

    def _define_params(self):
        self.params["brightness"] = FilterParam("brightness", "Brightness", "int", 0, -255, 255, 1)
        self.params["contrast"]   = FilterParam("contrast",   "Contrast",   "int", 0, -255, 255, 1)

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
        self.params["gamma"] = FilterParam("gamma", "Gamma", "float", 1.0, 0.1, 5.0, 0.05)

    def apply(self, image: np.ndarray) -> np.ndarray:
        g = self.get_param("gamma")
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image if image.dtype == np.uint8 else (image >> 8).astype(np.uint8), lut)


class CLAHEFilter(BaseFilter):
    NAME = "CLAHE"
    CATEGORY = "Contrast & Tone"

    def _define_params(self):
        self.params["clip_limit"] = FilterParam("clip_limit", "Clip Limit", "float", 2.0, 0.5, 10.0, 0.5)
        self.params["tile_size"]  = FilterParam("tile_size",  "Tile Size",  "int",   8,   4,   32,   4)

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
        self.params["in_black"]  = FilterParam("in_black",  "Input Black",  "int",   0,   0, 254, 1)
        self.params["in_white"]  = FilterParam("in_white",  "Input White",  "int", 255,   1, 255, 1)
        self.params["gamma"]     = FilterParam("gamma",     "Midtones",  "float", 1.0, 0.1, 5.0, 0.05)
        self.params["out_black"] = FilterParam("out_black", "Output Black", "int",   0,   0, 254, 1)
        self.params["out_white"] = FilterParam("out_white", "Output White", "int", 255,   1, 255, 1)

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
