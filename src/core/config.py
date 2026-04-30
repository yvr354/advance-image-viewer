import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class FocusThresholds:
    perfect: int = 700
    good: int = 400
    soft: int = 200


@dataclass
class Config:
    # Paths
    last_folder: str = ""
    recent_files: list = field(default_factory=list)

    # Viewer
    default_zoom: float = 1.0
    show_grid: bool = False
    show_minimap: bool = True
    show_crosshair: bool = True
    interpolation_mode: str = "nearest"  # nearest = pixel-perfect

    # Focus
    focus_metric: str = "laplacian"      # laplacian | tenengrad | brenner
    focus_thresholds: FocusThresholds = field(default_factory=FocusThresholds)
    focus_grid_size: int = 8             # NxN grid for heatmap
    show_focus_heatmap: bool = False

    # Quality
    overexpose_threshold: int = 250
    underexpose_threshold: int = 5

    # UI layout
    panel_inspector_visible: bool = True
    panel_pipeline_visible: bool = True
    panel_browser_visible: bool = True
    filmstrip_visible: bool = True
    filmstrip_size: int = 80

    # Pipeline
    last_pipeline_preset: str = ""

    # Camera
    camera_serial: str = ""
    camera_exposure_us: float = 10000.0
    camera_gain_db: float = 0.0

    # Window geometry
    window_width: int = 1600
    window_height: int = 900
    window_maximized: bool = False

    _config_path: str = field(default="", init=False, repr=False, compare=False)

    def load(self, path: str = ""):
        if not path:
            path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
        self._config_path = os.path.abspath(path)
        if not os.path.exists(self._config_path):
            return
        try:
            with open(self._config_path, "r") as f:
                data = json.load(f)
            for key, value in data.items():
                if key == "focus_thresholds" and isinstance(value, dict):
                    self.focus_thresholds = FocusThresholds(**value)
                elif hasattr(self, key):
                    setattr(self, key, value)
        except Exception:
            pass

    def save(self):
        if not self._config_path:
            self._config_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
            )
        data = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        data["focus_thresholds"] = asdict(self.focus_thresholds)
        # Keep recent files to last 50
        data["recent_files"] = self.recent_files[-50:]
        with open(self._config_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_recent(self, path: str):
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:50]
