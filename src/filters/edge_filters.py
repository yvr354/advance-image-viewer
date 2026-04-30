"""Edge detection and sharpening filters."""

import numpy as np
import cv2
from src.filters.base_filter import BaseFilter, FilterParam


class UnsharpMaskFilter(BaseFilter):
    NAME = "Unsharp Mask"
    CATEGORY = "Sharpening & Edge"

    def _define_params(self):
        self.params["radius"]    = FilterParam("radius",    "Radius",    "float", 1.0, 0.5, 10.0, 0.5)
        self.params["strength"]  = FilterParam("strength",  "Strength",  "float", 1.5, 0.0,  5.0, 0.1)
        self.params["threshold"] = FilterParam("threshold", "Threshold", "int",   0,   0,    50,   1)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_8bit(image)
        r = self.get_param("radius")
        s = self.get_param("strength")
        t = self.get_param("threshold")
        ksize = int(r * 2) * 2 + 1
        blurred = cv2.GaussianBlur(img8, (ksize, ksize), r)
        diff = img8.astype(np.int16) - blurred.astype(np.int16)
        mask = np.abs(diff) >= t
        result = img8.astype(np.float32) + s * diff * mask
        return np.clip(result, 0, 255).astype(np.uint8)


class CannyEdgeFilter(BaseFilter):
    NAME = "Canny Edge"
    CATEGORY = "Sharpening & Edge"

    def _define_params(self):
        self.params["threshold1"] = FilterParam("threshold1", "Low Threshold",  "int", 50,  0, 500, 5)
        self.params["threshold2"] = FilterParam("threshold2", "High Threshold", "int", 150, 0, 500, 5)
        self.params["aperture"]   = FilterParam("aperture",   "Aperture Size",  "choice", 3, choices=[3, 5, 7])

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray_8bit(image)
        t1 = self.get_param("threshold1")
        t2 = self.get_param("threshold2")
        ap = self.get_param("aperture")
        edges = cv2.Canny(gray, t1, t2, apertureSize=ap)
        if image.ndim == 3:
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges


class SobelEdgeFilter(BaseFilter):
    NAME = "Sobel Edge"
    CATEGORY = "Sharpening & Edge"

    def _define_params(self):
        self.params["direction"] = FilterParam("direction", "Direction", "choice", "combined", choices=["x", "y", "combined"])
        self.params["ksize"]     = FilterParam("ksize",     "Kernel",    "choice", 3,           choices=[3, 5, 7])

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray_8bit(image)
        k = self.get_param("ksize")
        d = self.get_param("direction")
        if d == "x":
            result = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        elif d == "y":
            result = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        else:
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
            result = np.hypot(sx, sy)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if image.ndim == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result


class LaplacianSharpenFilter(BaseFilter):
    NAME = "Laplacian Sharpen"
    CATEGORY = "Sharpening & Edge"

    def _define_params(self):
        self.params["strength"] = FilterParam("strength", "Strength", "float", 1.0, 0.1, 5.0, 0.1)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_8bit(image)
        s = self.get_param("strength")
        gray = _to_gray_8bit(image)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_norm = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img8.ndim == 3:
            lap_rgb = cv2.cvtColor(lap_norm, cv2.COLOR_GRAY2RGB)
            result = img8.astype(np.float32) + s * lap_rgb.astype(np.float32)
        else:
            result = img8.astype(np.float32) + s * lap_norm.astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8)


class DoGFilter(BaseFilter):
    NAME = "Difference of Gaussians"
    CATEGORY = "Sharpening & Edge"

    def _define_params(self):
        self.params["sigma1"] = FilterParam("sigma1", "Sigma 1 (fine)",   "float", 1.0, 0.5, 10.0, 0.5)
        self.params["sigma2"] = FilterParam("sigma2", "Sigma 2 (coarse)", "float", 2.0, 1.0, 20.0, 0.5)

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray_8bit(image).astype(np.float32)
        s1 = self.get_param("sigma1")
        s2 = self.get_param("sigma2")
        k1 = _ksize(s1)
        k2 = _ksize(s2)
        g1 = cv2.GaussianBlur(gray, (k1, k1), s1)
        g2 = cv2.GaussianBlur(gray, (k2, k2), s2)
        dog = g1 - g2
        result = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if image.ndim == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result


class MorphGradientFilter(BaseFilter):
    NAME = "Morphological Gradient"
    CATEGORY = "Sharpening & Edge"

    def _define_params(self):
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 3, 3, 21, 2)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_gray_8bit(image)
        k = self.get_param("ksize")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        result = cv2.morphologyEx(img8, cv2.MORPH_GRADIENT, kernel)
        if image.ndim == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result


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


def _ksize(sigma: float) -> int:
    k = int(sigma * 6) + 1
    return k if k % 2 == 1 else k + 1
