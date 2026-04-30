"""
Multi-illumination image fusion engine.
Assigns multiple grayscale images (different lighting) to RGB channels.
Defects invisible in any single image become color-visible in composite.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FusionChannel:
    image: Optional[np.ndarray] = None
    path: str = ""
    weight: float = 1.0
    assign_r: bool = False
    assign_g: bool = False
    assign_b: bool = False


class IlluminationFusion:
    MAX_INPUTS = 8

    def __init__(self):
        self.inputs: List[FusionChannel] = []

    def add_image(self, image: np.ndarray, path: str = "") -> int:
        if len(self.inputs) >= self.MAX_INPUTS:
            raise ValueError("Maximum 8 input images")
        ch = FusionChannel(image=self._to_gray(image), path=path)
        self.inputs.append(ch)
        return len(self.inputs) - 1

    def remove_image(self, index: int):
        if 0 <= index < len(self.inputs):
            self.inputs.pop(index)

    def clear(self):
        self.inputs.clear()

    def set_assignment(self, index: int, r: bool, g: bool, b: bool, weight: float = 1.0):
        if 0 <= index < len(self.inputs):
            self.inputs[index].assign_r = r
            self.inputs[index].assign_g = g
            self.inputs[index].assign_b = b
            self.inputs[index].weight = weight

    def compose(self) -> Optional[np.ndarray]:
        """Build RGB composite from channel assignments."""
        if not self.inputs:
            return None

        # Find reference shape
        h, w = self.inputs[0].image.shape[:2]

        r_acc = np.zeros((h, w), dtype=np.float64)
        g_acc = np.zeros((h, w), dtype=np.float64)
        b_acc = np.zeros((h, w), dtype=np.float64)
        r_cnt = g_cnt = b_cnt = 0

        for ch in self.inputs:
            if ch.image is None:
                continue
            img = self._resize_to(ch.image, h, w).astype(np.float64) * ch.weight
            if ch.assign_r:
                r_acc += img
                r_cnt += 1
            if ch.assign_g:
                g_acc += img
                g_cnt += 1
            if ch.assign_b:
                b_acc += img
                b_cnt += 1

        if r_cnt:
            r_acc /= r_cnt
        if g_cnt:
            g_acc /= g_cnt
        if b_cnt:
            b_acc /= b_cnt

        composite = np.stack([r_acc, g_acc, b_acc], axis=2)
        composite = cv2.normalize(composite, None, 0, 255, cv2.NORM_MINMAX)
        return composite.astype(np.uint8)

    def difference(self, idx_a: int, idx_b: int) -> Optional[np.ndarray]:
        """Absolute difference between two input images."""
        if idx_a >= len(self.inputs) or idx_b >= len(self.inputs):
            return None
        a = self.inputs[idx_a].image.astype(np.float32)
        b = self.inputs[idx_b].image.astype(np.float32)
        diff = np.abs(a - b)
        return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def average_fusion(self) -> Optional[np.ndarray]:
        """Weighted average of all inputs."""
        if not self.inputs:
            return None
        h, w = self.inputs[0].image.shape[:2]
        acc = np.zeros((h, w), dtype=np.float64)
        total_weight = 0.0
        for ch in self.inputs:
            if ch.image is None:
                continue
            acc += self._resize_to(ch.image, h, w).astype(np.float64) * ch.weight
            total_weight += ch.weight
        if total_weight > 0:
            acc /= total_weight
        return cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def max_fusion(self) -> Optional[np.ndarray]:
        """Pixel-wise maximum across all inputs."""
        if not self.inputs:
            return None
        h, w = self.inputs[0].image.shape[:2]
        result = np.zeros((h, w), dtype=np.float64)
        for ch in self.inputs:
            if ch.image is None:
                continue
            img = self._resize_to(ch.image, h, w).astype(np.float64)
            result = np.maximum(result, img)
        return result.astype(np.uint8)

    def min_fusion(self) -> Optional[np.ndarray]:
        """Pixel-wise minimum across all inputs."""
        if not self.inputs:
            return None
        h, w = self.inputs[0].image.shape[:2]
        result = np.ones((h, w), dtype=np.float64) * 255
        for ch in self.inputs:
            if ch.image is None:
                continue
            img = self._resize_to(ch.image, h, w).astype(np.float64)
            result = np.minimum(result, img)
        return result.astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image.astype(np.uint8)

    @staticmethod
    def _resize_to(image: np.ndarray, h: int, w: int) -> np.ndarray:
        if image.shape[:2] == (h, w):
            return image
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
