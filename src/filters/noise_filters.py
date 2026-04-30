"""Smoothing, noise reduction, and morphological filters."""

import numpy as np
import cv2
from src.filters.base_filter import BaseFilter, FilterParam


class GaussianBlurFilter(BaseFilter):
    NAME = "Gaussian Blur"
    CATEGORY = "Smoothing & Noise"

    def _define_params(self):
        self.params["sigma"] = FilterParam("sigma", "Sigma", "float", 1.0, 0.5, 20.0, 0.5)

    def apply(self, image: np.ndarray) -> np.ndarray:
        s = self.get_param("sigma")
        k = int(s * 6) + 1
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(_to_8bit(image), (k, k), s)


class MedianFilter(BaseFilter):
    NAME = "Median Filter"
    CATEGORY = "Smoothing & Noise"

    def _define_params(self):
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "choice", 3, choices=[3, 5, 7, 9, 11, 15, 21])

    def apply(self, image: np.ndarray) -> np.ndarray:
        k = self.get_param("ksize")
        return cv2.medianBlur(_to_8bit(image), k)


class BilateralFilter(BaseFilter):
    NAME = "Bilateral Filter"
    CATEGORY = "Smoothing & Noise"

    def _define_params(self):
        self.params["diameter"]    = FilterParam("diameter",    "Diameter",     "int",   9,   3, 25, 2)
        self.params["sigma_color"] = FilterParam("sigma_color", "Sigma Color",  "float", 75, 10, 200, 5)
        self.params["sigma_space"] = FilterParam("sigma_space", "Sigma Space",  "float", 75, 10, 200, 5)

    def apply(self, image: np.ndarray) -> np.ndarray:
        d  = self.get_param("diameter")
        sc = self.get_param("sigma_color")
        ss = self.get_param("sigma_space")
        return cv2.bilateralFilter(_to_8bit(image), d, sc, ss)


class NLMeansFilter(BaseFilter):
    NAME = "NL-Means Denoise"
    CATEGORY = "Smoothing & Noise"

    def _define_params(self):
        self.params["h"]          = FilterParam("h",          "H Strength",      "float", 10.0, 1.0, 50.0, 1.0)
        self.params["template_w"] = FilterParam("template_w", "Template Window", "int",   7,    3,   21,   2)
        self.params["search_w"]   = FilterParam("search_w",   "Search Window",   "int",   21,   7,   63,   2)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_8bit(image)
        h  = self.get_param("h")
        tw = self.get_param("template_w")
        sw = self.get_param("search_w")
        if img8.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img8, None, h, h, tw, sw)
        return cv2.fastNlMeansDenoising(img8, None, h, tw, sw)


class TopHatFilter(BaseFilter):
    NAME = "Top-Hat"
    CATEGORY = "Morphological"

    def _define_params(self):
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 15, 3, 51, 2)
        self.params["shape"] = FilterParam("shape", "Shape", "choice", "ellipse", choices=["rect", "ellipse", "cross"])

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_gray_8bit(image)
        k = self.get_param("ksize")
        sh = self.get_param("shape")
        shape_map = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}
        kernel = cv2.getStructuringElement(shape_map[sh], (k, k))
        result = cv2.morphologyEx(img8, cv2.MORPH_TOPHAT, kernel)
        if image.ndim == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result


class BlackHatFilter(BaseFilter):
    NAME = "Black-Hat"
    CATEGORY = "Morphological"

    def _define_params(self):
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 15, 3, 51, 2)
        self.params["shape"] = FilterParam("shape", "Shape", "choice", "ellipse", choices=["rect", "ellipse", "cross"])

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_gray_8bit(image)
        k = self.get_param("ksize")
        sh = self.get_param("shape")
        shape_map = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}
        kernel = cv2.getStructuringElement(shape_map[sh], (k, k))
        result = cv2.morphologyEx(img8, cv2.MORPH_BLACKHAT, kernel)
        if image.ndim == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result


class MorphOpenFilter(BaseFilter):
    NAME = "Morphological Open"
    CATEGORY = "Morphological"

    def _define_params(self):
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 5, 3, 31, 2)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_8bit(image)
        k = self.get_param("ksize")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(img8, cv2.MORPH_OPEN, kernel)


class MorphCloseFilter(BaseFilter):
    NAME = "Morphological Close"
    CATEGORY = "Morphological"

    def _define_params(self):
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 5, 3, 31, 2)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_8bit(image)
        k = self.get_param("ksize")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(img8, cv2.MORPH_CLOSE, kernel)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _to_8bit(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16:
        return (image >> 8).astype(np.uint8)
    return image.astype(np.uint8)


def _to_gray_8bit(image: np.ndarray) -> np.ndarray:
    img = _to_8bit(image)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img
