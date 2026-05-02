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
        self.params["lut"]       = FilterParam("lut",       "Color Map", "choice", "JET", choices=list(LUT_MAP.keys()),
            description="Scientific color map applied to grayscale intensity. "
                        "JET = blue→green→red (widest contrast, common in engineering). "
                        "HOT = black→red→orange→yellow→white (thermal imaging style). "
                        "VIRIDIS/INFERNO/PLASMA = perceptually uniform (best for quantitative measurement — equal steps in brightness = equal steps in value). "
                        "BONE/OCEAN/PINK = low-contrast, useful for subtle texture. "
                        "Use VIRIDIS or INFERNO for reports and publications.")
        self.params["range_min"] = FilterParam("range_min", "Input Min", "int", 0,   0, 254, 1,
            description="Minimum input intensity mapped to the start of the color map. "
                        "Pixels at or below this value get the first color (e.g. blue in JET). "
                        "Increase to clip dark pixels and stretch color range over the region of interest.")
        self.params["range_max"] = FilterParam("range_max", "Input Max", "int", 255, 1, 255, 1,
            description="Maximum input intensity mapped to the end of the color map. "
                        "Pixels at or above this value get the last color (e.g. red in JET). "
                        "Decrease to clip bright pixels — useful to focus color contrast on mid-tones.")

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
        self.params["over_threshold"]  = FilterParam("over_threshold",  "Overexpose Threshold",  "int", 250, 200, 255, 1,
            description="Pixels at or above this intensity are highlighted RED (overexposed/saturated). "
                        "250 = flag near-saturated pixels. 255 = only flag clipped (pure white) pixels. "
                        "Overexposed regions lose all detail — defects in these areas cannot be detected. "
                        "Lower this threshold if you need to identify specular reflections or blown highlights.")
        self.params["under_threshold"] = FilterParam("under_threshold", "Underexpose Threshold", "int",   5,   0,  50, 1,
            description="Pixels at or below this intensity are highlighted BLUE (underexposed/crushed). "
                        "5 = flag near-black pixels. 0 = only flag pure black (clipped shadows). "
                        "Underexposed regions lose shadow detail — cracks in dark areas may be invisible. "
                        "Increase threshold if deep shadows are hiding defects.")

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
        self.params["channel"] = FilterParam("channel", "Channel", "choice", "gray", choices=["red", "green", "blue", "gray"],
            description="Which color channel to isolate and display. "
                        "Gray = standard luminance (RGB weighted average). "
                        "Red = shows R channel only — useful for red-absorbing coatings, blood stains. "
                        "Green = most sensitive for human vision, maximum SNR for most cameras. "
                        "Blue = shows B channel — useful for UV-fluorescent defects, blue-absorbing coatings. "
                        "For Bayer-pattern industrial cameras, green channel has 2× more pixels than R or B.")

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
        self.params["rr"] = FilterParam("rr", "R→R", "float", 1.0, 0.0, 2.0, 0.05,
            description="Amount of Red channel contributing to output Red. 1.0 = normal. 0 = no red in output red.")
        self.params["rg"] = FilterParam("rg", "G→R", "float", 0.0, 0.0, 2.0, 0.05,
            description="Amount of Green channel contributing to output Red. "
                        "Increase to add green-channel content into the red channel output.")
        self.params["rb"] = FilterParam("rb", "B→R", "float", 0.0, 0.0, 2.0, 0.05,
            description="Amount of Blue channel contributing to output Red. "
                        "Increase to blend blue into red output.")
        self.params["gr"] = FilterParam("gr", "R→G", "float", 0.0, 0.0, 2.0, 0.05,
            description="Amount of Red channel contributing to output Green.")
        self.params["gg"] = FilterParam("gg", "G→G", "float", 1.0, 0.0, 2.0, 0.05,
            description="Amount of Green channel contributing to output Green. 1.0 = normal.")
        self.params["gb"] = FilterParam("gb", "B→G", "float", 0.0, 0.0, 2.0, 0.05,
            description="Amount of Blue channel contributing to output Green.")
        self.params["br"] = FilterParam("br", "R→B", "float", 0.0, 0.0, 2.0, 0.05,
            description="Amount of Red channel contributing to output Blue.")
        self.params["bg"] = FilterParam("bg", "G→B", "float", 0.0, 0.0, 2.0, 0.05,
            description="Amount of Green channel contributing to output Blue.")
        self.params["bb"] = FilterParam("bb", "B→B", "float", 1.0, 0.0, 2.0, 0.05,
            description="Amount of Blue channel contributing to output Blue. 1.0 = normal. "
                        "Channel Mixer lets you create custom band combinations — e.g. emphasize a specific spectral signature. "
                        "Use case: increase R→G to make red defects more visible in the green channel.")

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
