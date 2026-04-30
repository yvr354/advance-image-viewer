"""Base class for all processing filters."""

from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class FilterParam:
    name: str
    label: str
    type: str           # float | int | bool | choice
    value: Any
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    choices: list = field(default_factory=list)


class BaseFilter(ABC):
    NAME = "Base Filter"
    CATEGORY = "General"

    def __init__(self):
        self.enabled = True
        self.opacity = 1.0          # 0.0 – 1.0 blend with previous layer
        self.blend_mode = "normal"  # normal | multiply | screen | overlay | difference
        self.params: Dict[str, FilterParam] = {}
        self._define_params()

    @abstractmethod
    def _define_params(self):
        """Subclasses define their FilterParam entries here."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply filter. Input and output are uint8 or uint16 numpy arrays."""

    def get_param(self, name: str) -> Any:
        return self.params[name].value

    def set_param(self, name: str, value: Any):
        if name in self.params:
            self.params[name].value = value

    def to_dict(self) -> dict:
        return {
            "filter": self.__class__.__name__,
            "enabled": self.enabled,
            "opacity": self.opacity,
            "blend_mode": self.blend_mode,
            "params": {k: v.value for k, v in self.params.items()},
        }

    def from_dict(self, data: dict):
        self.enabled = data.get("enabled", True)
        self.opacity = data.get("opacity", 1.0)
        self.blend_mode = data.get("blend_mode", "normal")
        for k, v in data.get("params", {}).items():
            self.set_param(k, v)

    def _blend(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Blend processed result with original using opacity and blend_mode."""
        if self.opacity >= 1.0 and self.blend_mode == "normal":
            return processed

        orig = original.astype(np.float32) / 255.0
        proc = processed.astype(np.float32) / 255.0

        if self.blend_mode == "multiply":
            blended = orig * proc
        elif self.blend_mode == "screen":
            blended = 1 - (1 - orig) * (1 - proc)
        elif self.blend_mode == "overlay":
            mask = orig < 0.5
            blended = np.where(mask, 2 * orig * proc, 1 - 2 * (1 - orig) * (1 - proc))
        elif self.blend_mode == "difference":
            blended = np.abs(proc - orig)
        else:
            blended = proc

        result = orig * (1 - self.opacity) + blended * self.opacity
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
