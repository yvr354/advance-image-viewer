"""
3D surface visualization from a 2D image.
Uses image intensity as height (Z) — no special hardware needed.
Renders as interactive 3D mesh using PyQtGraph GLViewWidget.

Why this is powerful for defect inspection:
  A 0.5% brightness difference invisible in 2D becomes a visible
  bump or valley when rendered as a 3D surface.
  The engineer can rotate, tilt, and zoom the surface to inspect
  any region from any angle — impossible with a flat 2D view.
"""

import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QComboBox, QCheckBox, QGroupBox,
)
from PyQt6.QtCore import Qt
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from src.ui.tooltips import TIP


class Surface3DPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._surface: gl.GLSurfacePlotItem | None = None
        self._build()

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Controls bar
        ctrl = QHBoxLayout()

        self._downsample = QSlider(Qt.Orientation.Horizontal)
        self._downsample.setRange(1, 16)
        self._downsample.setValue(4)
        self._downsample.setFixedWidth(80)
        self._downsample.setToolTip(TIP["3d_downsample"])
        self._downsample.valueChanged.connect(self._update_surface)

        self._z_scale = QSlider(Qt.Orientation.Horizontal)
        self._z_scale.setRange(1, 200)
        self._z_scale.setValue(30)
        self._z_scale.setFixedWidth(100)
        self._z_scale.setToolTip(TIP["3d_z_scale"])
        self._z_scale.valueChanged.connect(self._update_surface)

        self._colormap = QComboBox()
        self._colormap.addItems(["thermal", "viridis", "plasma", "grays", "cyclic"])
        self._colormap.setToolTip(TIP["3d_colormap"])
        self._colormap.currentTextChanged.connect(self._update_surface)

        self._smooth_cb = QCheckBox("Smooth")
        self._smooth_cb.setChecked(True)
        self._smooth_cb.setToolTip(TIP["3d_smooth"])
        self._smooth_cb.stateChanged.connect(self._update_surface)

        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.clicked.connect(self._reset_view)

        ctrl.addWidget(QLabel("Resolution:"))
        ctrl.addWidget(self._downsample)
        ctrl.addWidget(QLabel("  Z Scale:"))
        ctrl.addWidget(self._z_scale)
        ctrl.addWidget(QLabel("  Color:"))
        ctrl.addWidget(self._colormap)
        ctrl.addWidget(self._smooth_cb)
        ctrl.addStretch()
        ctrl.addWidget(self._reset_btn)
        layout.addLayout(ctrl)

        # 3D viewport
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("#0A0A0F")
        self._view.setCameraPosition(distance=200, elevation=30, azimuth=45)
        layout.addWidget(self._view)

        # Grid
        grid = gl.GLGridItem()
        grid.setColor((40, 40, 60, 80))
        self._view.addItem(grid)

        # Info label
        self._info = QLabel("Load an image — surface will render here.  Drag to rotate.  Scroll to zoom.")
        self._info.setStyleSheet("color: #44445A; font-size: 10px;")
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._info)

    # ── Public API ────────────────────────────────────────────────────────

    def set_image(self, image: np.ndarray):
        self._image = image
        self._update_surface()

    def clear(self):
        if self._surface:
            self._view.removeItem(self._surface)
            self._surface = None

    # ── Surface computation ───────────────────────────────────────────────

    def _update_surface(self):
        if self._image is None:
            return

        ds    = self._downsample.value()
        zs    = self._z_scale.value() / 10.0
        cmap  = self._colormap.currentText()
        smooth = self._smooth_cb.isChecked()

        gray = self._to_gray(self._image)

        # Downsample for performance — full 10K mesh would be 100M vertices
        h, w = gray.shape
        gray_ds = cv2.resize(gray, (w // ds, h // ds), interpolation=cv2.INTER_AREA)

        if smooth:
            gray_ds = cv2.GaussianBlur(gray_ds, (3, 3), 0.8)

        # Height map: pixel intensity → Z coordinate
        z = gray_ds.astype(np.float32) / 255.0 * zs * 50

        # Build color map
        colors = self._make_colors(gray_ds, cmap)

        # Remove old surface
        if self._surface:
            self._view.removeItem(self._surface)

        self._surface = gl.GLSurfacePlotItem(
            z=z,
            colors=colors,
            shader="shaded",
            smooth=smooth,
        )
        self._surface.translate(
            -z.shape[1] / 2, -z.shape[0] / 2, 0
        )
        self._view.addItem(self._surface)

        pts = gray_ds.shape[0] * gray_ds.shape[1]
        self._info.setText(
            f"Surface: {gray_ds.shape[1]}×{gray_ds.shape[0]} pts  "
            f"({pts:,} vertices)  |  Z scale: {zs:.1f}×  |  "
            f"Original: {w}×{h}  |  Downsample: {ds}×  —  "
            f"Drag=Rotate  Scroll=Zoom  Middle=Pan"
        )

    def _make_colors(self, gray: np.ndarray, cmap_name: str) -> np.ndarray:
        """Map grayscale intensity to RGBA colors."""
        norm = gray.astype(np.float32) / 255.0

        if cmap_name == "thermal":
            r = np.clip(norm * 2,         0, 1)
            g = np.clip(norm * 2 - 0.5,   0, 1)
            b = np.clip(1 - norm * 2,     0, 1)
        elif cmap_name == "viridis":
            # Simplified viridis approximation
            r = np.clip(0.267 + norm * 0.733, 0, 1)
            g = np.clip(0.004 + norm * 0.871, 0, 1)
            b = np.clip(0.329 + norm * 0.121, 0, 1)
        elif cmap_name == "plasma":
            r = np.clip(0.05 + norm * 0.95, 0, 1)
            g = np.clip(0.03 + norm * 0.45 * np.sin(norm * np.pi), 0, 1)
            b = np.clip(0.53 - norm * 0.50, 0, 1)
        elif cmap_name == "grays":
            r = g = b = norm
        else:  # cyclic
            r = (np.sin(norm * 2 * np.pi) + 1) / 2
            g = (np.sin(norm * 2 * np.pi + 2.094) + 1) / 2
            b = (np.sin(norm * 2 * np.pi + 4.189) + 1) / 2

        alpha = np.ones_like(norm)
        colors = np.stack([r, g, b, alpha], axis=2)
        return colors.astype(np.float32)

    def _reset_view(self):
        self._view.setCameraPosition(distance=200, elevation=30, azimuth=45)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        img = image
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)
