"""Smoothing, noise reduction, and morphological filters."""

import numpy as np
import cv2
from src.filters.base_filter import BaseFilter, FilterParam


class GaussianBlurFilter(BaseFilter):
    NAME = "Gaussian Blur"
    CATEGORY = "Smoothing & Noise"

    def _define_params(self):
        self.params["sigma"] = FilterParam("sigma", "Sigma", "float", 1.0, 0.5, 20.0, 0.5,
            description="Standard deviation of the Gaussian kernel — controls blur radius. "
                        "1.0 = gentle smoothing of single-pixel noise. "
                        "3.0+ = heavy blur, removes fine texture. "
                        "Use low values (0.5–1.5) before edge detection to suppress camera noise without losing edges.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        s = self.get_param("sigma")
        k = int(s * 6) + 1
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(_to_8bit(image), (k, k), s)


class MedianFilter(BaseFilter):
    NAME = "Median Filter"
    CATEGORY = "Smoothing & Noise"

    def _define_params(self):
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "choice", 3, choices=[3, 5, 7, 9, 11, 15, 21],
            description="Size of the median filter window (pixels). "
                        "Replaces each pixel with the median of its neighbors — preserves edges better than Gaussian. "
                        "3 = removes isolated hot/cold pixels. 5–7 = removes salt-and-pepper noise. "
                        "9+ = heavy smoothing. Best choice for removing impulse noise (stuck pixels, dust).")

    def apply(self, image: np.ndarray) -> np.ndarray:
        k = self.get_param("ksize")
        return cv2.medianBlur(_to_8bit(image), k)


class BilateralFilter(BaseFilter):
    NAME = "Bilateral Filter"
    CATEGORY = "Smoothing & Noise"

    def _define_params(self):
        self.params["diameter"]    = FilterParam("diameter",    "Diameter",     "int",   9,   3, 25, 2,
            description="Pixel neighborhood size. Larger = more smoothing but slower. "
                        "5 = fast, subtle. 9 = balanced (recommended). 15+ = strong but slow. "
                        "Bilateral filter preserves sharp edges while smoothing flat regions — ideal for pre-inspection denoising.")
        self.params["sigma_color"] = FilterParam("sigma_color", "Sigma Color",  "float", 75, 10, 200, 5,
            description="Color space similarity range. "
                        "Pixels within this intensity range of the center pixel are included in the average. "
                        "High (> 100) = aggressive smoothing across larger intensity differences. "
                        "Low (< 30) = only smooth pixels of very similar brightness — preserve more edge contrast.")
        self.params["sigma_space"] = FilterParam("sigma_space", "Sigma Space",  "float", 75, 10, 200, 5,
            description="Spatial distance weighting. Controls how far spatially pixels can be and still influence the result. "
                        "High = larger area of influence (more blur). Low = only nearest neighbors. "
                        "Usually set equal to Sigma Color. Typical: 75.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        d  = self.get_param("diameter")
        sc = self.get_param("sigma_color")
        ss = self.get_param("sigma_space")
        return cv2.bilateralFilter(_to_8bit(image), d, sc, ss)


class NLMeansFilter(BaseFilter):
    NAME = "NL-Means Denoise"
    CATEGORY = "Smoothing & Noise"

    def _define_params(self):
        self.params["h"]          = FilterParam("h",          "H Strength",      "float", 10.0, 1.0, 50.0, 1.0,
            description="Filter strength. Controls how aggressively noise is removed. "
                        "Higher = more noise removed but also more fine detail lost. "
                        "3–5 = preserve texture. 10 = balanced. 20+ = heavy smoothing. "
                        "Best denoising algorithm for inspection — non-local averaging preserves structure that bilateral misses.")
        self.params["template_w"] = FilterParam("template_w", "Template Window", "int",   7,    3,   21,   2,
            description="Size of the patch used to compare similarity between regions. "
                        "7×7 is standard. Larger = more context for comparison but slower. "
                        "Increase if the surface has large repeating texture patterns.")
        self.params["search_w"]   = FilterParam("search_w",   "Search Window",   "int",   21,   7,   63,   2,
            description="How far (in pixels) to search for similar patches. "
                        "21 = search 21×21 area around each pixel. Larger = better quality but significantly slower. "
                        "For high-resolution industrial cameras, 35–63 gives better results.")

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
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 15, 3, 51, 2,
            description="Structuring element size. "
                        "Top-Hat = original minus morphological opening. "
                        "Reveals bright features smaller than the kernel — highlights bright spots, protrusions, scratches on dark background. "
                        "Set larger than the defect size for best isolation. Typical: 15–25 for pits/protrusions.")
        self.params["shape"] = FilterParam("shape", "Shape", "choice", "ellipse", choices=["rect", "ellipse", "cross"],
            description="Shape of the structuring element. "
                        "Ellipse = most natural, preferred for circular or arbitrary features. "
                        "Rect = sensitive to axis-aligned features. Cross = only affects horizontal/vertical structures.")

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
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 15, 3, 51, 2,
            description="Structuring element size. "
                        "Black-Hat = morphological closing minus original. "
                        "Reveals dark features smaller than the kernel — highlights voids, pits, scratches on bright background. "
                        "Complement of Top-Hat: use Black-Hat when defects are darker than the surface.")
        self.params["shape"] = FilterParam("shape", "Shape", "choice", "ellipse", choices=["rect", "ellipse", "cross"],
            description="Shape of the structuring element. "
                        "Ellipse = most natural, preferred for circular or arbitrary features. "
                        "Rect = sensitive to axis-aligned features. Cross = only affects horizontal/vertical structures.")

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
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 5, 3, 31, 2,
            description="Structuring element size (erosion then dilation). "
                        "Removes bright noise spots smaller than kernel. Separates touching objects. "
                        "Used in defect detection pipeline to eliminate isolated false-positive pixels before blob counting. "
                        "Set to the minimum defect size to preserve real defects while removing noise.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_8bit(image)
        k = self.get_param("ksize")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(img8, cv2.MORPH_OPEN, kernel)


class MorphCloseFilter(BaseFilter):
    NAME = "Morphological Close"
    CATEGORY = "Morphological"

    def _define_params(self):
        self.params["ksize"] = FilterParam("ksize", "Kernel Size", "int", 5, 3, 31, 2,
            description="Structuring element size (dilation then erosion). "
                        "Fills small dark holes inside bright objects. Joins nearby blobs. "
                        "Used after thresholding to merge fragmented defect blobs into one solid region. "
                        "A scratch with small gaps will be joined into a single blob for correct area measurement.")

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
