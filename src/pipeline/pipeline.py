"""
Processing pipeline — ordered stack of filters applied to an image.
Non-destructive: original image is never modified.
"""

import json
import numpy as np
from typing import List, Optional
from src.filters.base_filter import BaseFilter
from src.pipeline.filter_registry import FILTER_REGISTRY


class Pipeline:
    def __init__(self):
        self.layers: List[BaseFilter] = []

    def add(self, filter_instance: BaseFilter, index: Optional[int] = None):
        if index is None:
            self.layers.append(filter_instance)
        else:
            self.layers.insert(index, filter_instance)

    def remove(self, index: int):
        if 0 <= index < len(self.layers):
            self.layers.pop(index)

    def move_up(self, index: int):
        if index > 0:
            self.layers[index - 1], self.layers[index] = self.layers[index], self.layers[index - 1]

    def move_down(self, index: int):
        if index < len(self.layers) - 1:
            self.layers[index + 1], self.layers[index] = self.layers[index], self.layers[index + 1]

    def clear(self):
        self.layers.clear()

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply all enabled layers in order."""
        result = image.copy()
        for layer in self.layers:
            if not layer.enabled:
                continue
            try:
                processed = layer.apply(result)
                if layer.opacity < 1.0 or layer.blend_mode != "normal":
                    processed = layer._blend(result, processed)
                result = processed
            except Exception:
                pass  # Skip broken layer, continue pipeline
        return result

    def __len__(self):
        return len(self.layers)

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        data = [layer.to_dict() for layer in self.layers]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.layers.clear()
        for item in data:
            cls_name = item.get("filter")
            cls = FILTER_REGISTRY.get(cls_name)
            if cls:
                instance = cls()
                instance.from_dict(item)
                self.layers.append(instance)

    def to_list(self) -> list:
        return [layer.to_dict() for layer in self.layers]
