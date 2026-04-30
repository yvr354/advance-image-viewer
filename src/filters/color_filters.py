"""False color, LUT, and channel filters."""

import numpy as np
import cv2
from src.filters.base_filter import BaseFilter, FilterParam

LUT_MAP = {
    "JET":     cv2.COLORMAP_JET,
    "HOT":     cv2.COLORMAP_HOT,
    "COOL":    cv2.COLORMAP_COOL,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA":  cv2.COLORMAP_PLASMA,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "BONE":    cv2.COLORMAP_BONE,
    "OCEAN":   cv2.COLORMAP_OCEAN,
    "PINK":    cv2.COLORMAP_PINK,
}


class FalseColorFilter(BaseFilter):
    NAME = "False Color (LUT)"
    CATEGORY = "False Color & LUT"

    def _define_params(self):
        self.params["lut"]       = FilterParam("lut",       "Color Map", "choice", "JET", choices=list(LUT_MAP.keys()))
        self.params["range_min"] = FilterParam("range_min", "Input Min", "int", 0,   0, 254, 1)
        self.params["range_max"] = FilterParam("range_max", "Input Max", "int", 255, 1, 255, 1)

    def apply(self, image: np.ndarray) -> np.ndarray:
        lut_name = self.get_param("lut")
        rmin = self.get_param("range_min")
        rmax = self.get_param("range_max")
        gray = _to_gray_8bit(image)
        gray = np.clip((gray.astype(np.float32) - rmin) / max(rmax - rmin, 1) * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray, LUT_MAP.get(lut_name, cv2.COLORMAP_JET))
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


class ClippingHighlightFilter(BaseFilter):
    NAME = "Clipping Highlight"
    CATEGORY = "False Color & LUT"

    def _define_params(self):
        self.params["over_threshold"]  = FilterParam("over_threshold",  "Overexpose Threshold",  "int", 250, 200, 255, 1)
        self.params["under_threshold"] = FilterParam("under_threshold", "Underexpose Threshold", "int",   5,   0,  50, 1)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_8bit(image)
        if img8.ndim == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        result = img8.copy()
        ot = self.get_param("over_threshold")
        ut = self.get_param("under_threshold")
        gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
        result[gray >= ot] = [255, 0, 0]    # Red = overexposed
        result[gray <= ut] = [0, 0, 255]    # Blue = underexposed
        return result


class ChannelSplitFilter(BaseFilter):
    NAME = "Channel Split"
    CATEGORY = "False Color & LUT"

    def _define_params(self):
        self.params["channel"] = FilterParam("channel", "Channel", "choice", "gray", choices=["red", "green", "blue", "gray"])

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_8bit(image)
        ch = self.get_param("channel")
        if img8.ndim == 2:
            return img8
        if ch == "red":
            result = img8[:, :, 0]
        elif ch == "green":
            result = img8[:, :, 1]
        elif ch == "blue":
            result = img8[:, :, 2]
        else:
            result = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


class ChannelMixerFilter(BaseFilter):
    NAME = "Channel Mixer"
    CATEGORY = "False Color & LUT"

    def _define_params(self):
        self.params["rr"] = FilterParam("rr", "R→R", "float", 1.0, 0.0, 2.0, 0.05)
        self.params["rg"] = FilterParam("rg", "G→R", "float", 0.0, 0.0, 2.0, 0.05)
        self.params["rb"] = FilterParam("rb", "B→R", "float", 0.0, 0.0, 2.0, 0.05)
        self.params["gr"] = FilterParam("gr", "R→G", "float", 0.0, 0.0, 2.0, 0.05)
        self.params["gg"] = FilterParam("gg", "G→G", "float", 1.0, 0.0, 2.0, 0.05)
        self.params["gb"] = FilterParam("gb", "B→G", "float", 0.0, 0.0, 2.0, 0.05)
        self.params["br"] = FilterParam("br", "R→B", "float", 0.0, 0.0, 2.0, 0.05)
        self.params["bg"] = FilterParam("bg", "G→B", "float", 0.0, 0.0, 2.0, 0.05)
        self.params["bb"] = FilterParam("bb", "B→B", "float", 1.0, 0.0, 2.0, 0.05)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img8 = _to_8bit(image)
        if img8.ndim == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        f = img8.astype(np.float32)
        r = f[:, :, 0] * self.get_param("rr") + f[:, :, 1] * self.get_param("rg") + f[:, :, 2] * self.get_param("rb")
        g = f[:, :, 0] * self.get_param("gr") + f[:, :, 1] * self.get_param("gg") + f[:, :, 2] * self.get_param("gb")
        b = f[:, :, 0] * self.get_param("br") + f[:, :, 1] * self.get_param("bg") + f[:, :, 2] * self.get_param("bb")
        result = np.stack([r, g, b], axis=2)
        return np.clip(result, 0, 255).astype(np.uint8)


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
