import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImageData:
    """Central data container for one loaded image."""
    path: str = ""
    raw: Optional[np.ndarray] = None        # Original pixels, never modified
    display: Optional[np.ndarray] = None    # After pipeline processing
    bit_depth: int = 8
    channels: int = 1
    width: int = 0
    height: int = 0
    filename: str = ""
    metadata: dict = field(default_factory=dict)

    # Quality metrics (populated by analysis engine)
    focus_score: float = -1.0
    focus_verdict: str = ""          # PERFECT / GOOD / SOFT / BLURRY
    focus_map: Optional[np.ndarray] = None   # NxN float array
    exposure_ok: bool = True
    overexposed_pct: float = 0.0
    underexposed_pct: float = 0.0
    noise_level: float = 0.0
    contrast_score: float = 0.0
    quality_score: float = -1.0      # 0–100 composite
    quality_verdict: str = ""        # PASS / FAIL

    def is_loaded(self) -> bool:
        return self.raw is not None

    def is_grayscale(self) -> bool:
        return self.channels == 1

    def is_16bit(self) -> bool:
        return self.bit_depth == 16

    def shape_str(self) -> str:
        if self.raw is None:
            return ""
        return f"{self.width}×{self.height}  {self.bit_depth}-bit  {'Gray' if self.is_grayscale() else 'RGB'}"
