"""Edge detection and sharpening filters."""

import numpy as np
import cv2
from src.filters.base_filter import BaseFilter, FilterParam


class UnsharpMaskFilter(BaseFilter):
    NAME = "Unsharp Mask"
    CATEGORY = "② Enhance"
    DESCRIPTION = "Industry-standard sharpening. Adds a scaled high-pass layer to the image. Makes edge features crisper without changing overall brightness."

    def _define_params(self):
        self.params["radius"]    = FilterParam("radius",    "Radius",    "float", 1.0, 0.5, 10.0, 0.5,
            description="Blur radius used to create the unsharp mask. "
                        "Larger radius sharpens broader, lower-frequency edges. "
                        "Smaller radius sharpens fine detail and micro-texture. "
                        "Typical: 1.0–2.0. For scratches/pits: 0.5–1.0.")
        self.params["strength"]  = FilterParam("strength",  "Strength",  "float", 1.5, 0.0,  5.0, 0.1,
            description="How much sharpening is applied. "
                        "1.0 = subtle. 2.0 = strong. > 3.0 = aggressive, may create halos. "
                        "Typical: 1.5 for defect inspection. Reduce if noise is amplified.")
        self.params["threshold"] = FilterParam("threshold", "Threshold", "int",   0,   0,    50,   1,
            description="Minimum edge contrast required before sharpening is applied. "
                        "0 = sharpen everything including noise. "
                        "5–10 = skip smooth areas and noise, sharpen only real edges. "
                        "Use > 0 on noisy images to avoid amplifying grain.")

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
    CATEGORY = "③ Detect"
    DESCRIPTION = "Optimal edge detector (Canny 1986). Finds thin, precise, connected edges using gradient magnitude + hysteresis. Best for scratch boundaries and part outlines."

    def _define_params(self):
        self.params["threshold1"] = FilterParam("threshold1", "Low Threshold",  "int", 50,  0, 500, 5,
            description="Weak edge threshold. Pixels with gradient below this are discarded as background. "
                        "Pixels between Low and High are kept only if connected to a strong edge. "
                        "Lower = more edges detected (more noise). Typical: 30–80.")
        self.params["threshold2"] = FilterParam("threshold2", "High Threshold", "int", 150, 0, 500, 5,
            description="Strong edge threshold. Pixels above this are always kept as edges. "
                        "Ratio of High:Low should be 2:1 or 3:1 for best results. "
                        "Higher = fewer, cleaner edges. Typical: 100–200.")
        self.params["aperture"]   = FilterParam("aperture",   "Aperture Size",  "choice", 3, choices=[3, 5, 7],
            description="Sobel kernel size used for gradient computation. "
                        "3 = fast, good for fine detail. 5 = smoother, less noise-sensitive. "
                        "7 = largest, best for broad edges. Default 3 works for most inspection tasks.")

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
    CATEGORY = "③ Detect"
    DESCRIPTION = "Directional gradient edge map. Faster than Canny, good for coarse edge extraction or direction-specific defect analysis (X only, Y only, or combined)."

    def _define_params(self):
        self.params["direction"] = FilterParam("direction", "Direction", "choice", "combined", choices=["x", "y", "combined"],
            description="Which edge direction to detect. "
                        "X = vertical edges (left-right transitions). "
                        "Y = horizontal edges (top-bottom transitions). "
                        "Combined = magnitude of both X and Y — finds edges in all directions. "
                        "Use Combined for general inspection; X or Y to isolate cracks in one direction.")
        self.params["ksize"]     = FilterParam("ksize",     "Kernel",    "choice", 3,           choices=[3, 5, 7],
            description="Sobel kernel size. Larger kernels smooth out noise before computing gradient. "
                        "3 = sharp, sensitive to fine edges. 5 = balanced. 7 = broad edges only. "
                        "Use 3 for precision inspection; 5 or 7 for noisy images.")

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
    CATEGORY = "② Enhance"
    DESCRIPTION = "Second-derivative sharpening. More aggressive than Unsharp Mask. Excellent for revealing micro-cracks, fine pits, and surface discontinuities."

    def _define_params(self):
        self.params["strength"] = FilterParam("strength", "Strength", "float", 1.0, 0.1, 5.0, 0.1,
            description="Amount of Laplacian edge enhancement added to the image. "
                        "1.0 = moderate sharpening. 2.0+ = strong. "
                        "Unlike Unsharp Mask, this uses second-order derivatives — more sensitive to sharp discontinuities. "
                        "Excellent for revealing micro-cracks and pit edges. May amplify noise on rough surfaces.")

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
    CATEGORY = "③ Detect"
    DESCRIPTION = "Band-pass spatial filter — isolates features at one specific size scale. Targets defects of a chosen size while suppressing everything else."

    def _define_params(self):
        self.params["sigma1"] = FilterParam("sigma1", "Sigma 1 (fine)",   "float", 1.0, 0.5, 10.0, 0.5,
            description="Fine-scale Gaussian blur sigma. Controls the scale of the narrow filter. "
                        "Smaller = captures very fine detail / high-frequency edges. "
                        "Must be smaller than Sigma 2. Typical: 1.0.")
        self.params["sigma2"] = FilterParam("sigma2", "Sigma 2 (coarse)", "float", 2.0, 1.0, 20.0, 0.5,
            description="Coarse-scale Gaussian blur sigma. Controls the scale of the wide filter. "
                        "The DoG result = fine blur minus coarse blur — isolates a frequency band. "
                        "Ratio of Sigma2:Sigma1 ≈ 1.6 mimics the human visual system (Marr-Hildreth). "
                        "Increase gap between sigma1 and sigma2 to detect broader features.")

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
    CATEGORY = "③ Detect"
    DESCRIPTION = "Boundary extractor (dilation minus erosion). Produces thick connected edge outlines. Use when edges must form closed regions for area measurement."

    def _define_params(self):
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 3, 3, 21, 2,
            description="Structuring element size for morphological gradient (dilation minus erosion). "
                        "Result highlights the boundary / edge of every object at the specified scale. "
                        "3 = thin, sharp boundaries. 9+ = thick outlines. "
                        "Particularly useful for detecting raised or recessed features on machined surfaces.")

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
