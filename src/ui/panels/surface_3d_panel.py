"""
3D surface visualization from a 2D image.
Uses image intensity as height (Z) — no special hardware needed.

Performance: computation runs in a background QThread so the UI
never freezes. Slider changes are debounced 400ms to avoid
rebuilding the mesh on every tick.
"""

import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QComboBox, QCheckBox, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl

from src.ui.tooltips import TIP


# ── Background worker ──────────────────────────────────────────────────────

class SurfaceWorker(QThread):
    """Computes mesh data off the UI thread."""
    result_ready = pyqtSignal(object, object, object)  # z, colors, info_str

    def __init__(self, image, ds, zs, cmap, smooth):
        super().__init__()
        self._image  = image
        self._ds     = ds
        self._zs     = zs
        self._cmap   = cmap
        self._smooth = smooth

    def run(self):
        try:
            gray = self._to_gray(self._image)
            h, w = gray.shape
            new_w = max(4, w // self._ds)
            new_h = max(4, h // self._ds)
            gray_ds = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if self._smooth:
                gray_ds = cv2.GaussianBlur(gray_ds, (3, 3), 0.8)

            z      = gray_ds.astype(np.float32) / 255.0 * self._zs * 50
            colors = self._make_colors(gray_ds, self._cmap)
            pts    = new_w * new_h
            info   = (
                f"Surface: {new_w}×{new_h}  ({pts:,} vertices)  |  "
                f"Z scale: {self._zs:.1f}×  |  Original: {w}×{h}  |  "
                f"Downsample: {self._ds}×  —  Drag=Rotate  Scroll=Zoom"
            )
            self.result_ready.emit(z, colors, info)
        except Exception as e:
            self.result_ready.emit(None, None, f"Error: {e}")

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        img = image
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)

    @staticmethod
    def _make_colors(gray: np.ndarray, cmap_name: str) -> np.ndarray:
        norm = gray.astype(np.float32) / 255.0
        if cmap_name == "thermal":
            r = np.clip(norm * 2,           0, 1)
            g = np.clip(norm * 2 - 0.5,     0, 1)
            b = np.clip(1 - norm * 2,       0, 1)
        elif cmap_name == "viridis":
            r = np.clip(0.267 + norm * 0.733, 0, 1)
            g = np.clip(0.004 + norm * 0.871, 0, 1)
            b = np.clip(0.329 + norm * 0.121, 0, 1)
        elif cmap_name == "plasma":
            r = np.clip(0.05 + norm * 0.95,                        0, 1)
            g = np.clip(0.03 + norm * 0.45 * np.sin(norm * np.pi), 0, 1)
            b = np.clip(0.53 - norm * 0.50,                        0, 1)
        elif cmap_name == "grays":
            r = g = b = norm
        else:  # cyclic
            r = (np.sin(norm * 2 * np.pi)           + 1) / 2
            g = (np.sin(norm * 2 * np.pi + 2.094)   + 1) / 2
            b = (np.sin(norm * 2 * np.pi + 4.189)   + 1) / 2
        alpha  = np.ones_like(norm)
        return np.stack([r, g, b, alpha], axis=2).astype(np.float32)


# ── Panel ──────────────────────────────────────────────────────────────────

class Surface3DPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image:   np.ndarray | None = None
        self._surface: gl.GLSurfacePlotItem | None = None
        self._worker:  SurfaceWorker | None = None

        # Debounce: rebuild 400ms after the last slider/combo change
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(400)
        self._debounce.timeout.connect(self._start_worker)

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
        self._downsample.setValue(8)          # default 8× — fast first render
        self._downsample.setFixedWidth(80)
        self._downsample.setToolTip(TIP["3d_downsample"])
        self._downsample.valueChanged.connect(self._schedule_update)

        self._z_scale = QSlider(Qt.Orientation.Horizontal)
        self._z_scale.setRange(1, 200)
        self._z_scale.setValue(30)
        self._z_scale.setFixedWidth(100)
        self._z_scale.setToolTip(TIP["3d_z_scale"])
        self._z_scale.valueChanged.connect(self._schedule_update)

        self._colormap = QComboBox()
        self._colormap.addItems(["thermal", "viridis", "plasma", "grays", "cyclic"])
        self._colormap.setToolTip(TIP["3d_colormap"])
        self._colormap.currentTextChanged.connect(self._schedule_update)

        self._smooth_cb = QCheckBox("Smooth")
        self._smooth_cb.setChecked(True)
        self._smooth_cb.setToolTip(TIP["3d_smooth"])
        self._smooth_cb.stateChanged.connect(self._schedule_update)

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

        work = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(work, stretch=1)

        # 3D viewport
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("#0A0A0F")
        self._view.setCameraPosition(distance=145, elevation=24, azimuth=42)
        work.addWidget(self._view)

        grid = gl.GLGridItem()
        grid.setColor((40, 40, 60, 80))
        self._view.addItem(grid)

        ref_panel = QWidget()
        ref_layout = QVBoxLayout(ref_panel)
        ref_layout.setContentsMargins(8, 0, 0, 0)
        ref_layout.setSpacing(6)
        self._ref_title = QLabel("Original Image")
        self._ref_title.setStyleSheet("color: #8888AA; font-size: 10px; font-weight: 700;")
        self._ref_image = QLabel("Load an image")
        self._ref_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ref_image.setMinimumWidth(280)
        self._ref_image.setStyleSheet("background: #0A0A0F; border: 1px solid #252535;")
        self._ref_meta = QLabel("")
        self._ref_meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ref_meta.setStyleSheet("color: #44445A; font-size: 10px;")
        ref_layout.addWidget(self._ref_title)
        ref_layout.addWidget(self._ref_image, stretch=1)
        ref_layout.addWidget(self._ref_meta)
        work.addWidget(ref_panel)
        work.setSizes([1400, 360])

        self._info = QLabel("Load an image — surface renders here.  Drag=Rotate  Scroll=Zoom")
        self._info.setStyleSheet("color: #44445A; font-size: 10px;")
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._info)

    # ── Public API ────────────────────────────────────────────────────────

    def set_image(self, image: np.ndarray):
        self._image = image
        self._update_reference()
        self._schedule_update()

    def clear(self):
        if self._surface:
            self._view.removeItem(self._surface)
            self._surface = None

    # ── Update logic ─────────────────────────────────────────────────────

    def _schedule_update(self):
        """Restart the debounce timer — fires _start_worker 400ms after last change."""
        self._debounce.start()

    def _start_worker(self):
        if self._image is None:
            return

        # Cancel previous worker if still running
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()

        self._info.setText("Computing surface mesh…")

        self._worker = SurfaceWorker(
            self._image,
            self._downsample.value(),
            self._z_scale.value() / 10.0,
            self._colormap.currentText(),
            self._smooth_cb.isChecked(),
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.start()

    def _on_result(self, z, colors, info: str):
        """Called from main thread when worker is done — safe to update GL."""
        if z is None:
            self._info.setText(info)
            return

        if self._surface:
            self._view.removeItem(self._surface)

        self._surface = gl.GLSurfacePlotItem(
            z=z,
            colors=colors,
            shader="shaded",
            smooth=self._smooth_cb.isChecked(),
        )
        self._surface.translate(-z.shape[1] / 2, -z.shape[0] / 2, 0)
        self._view.addItem(self._surface)
        self._info.setText(info)

    def _reset_view(self):
        self._view.setCameraPosition(distance=145, elevation=24, azimuth=42)

    def _update_reference(self):
        if self._image is None:
            return
        img8 = self._to_preview_8bit(self._image)
        if img8.ndim == 2:
            h, w = img8.shape
            qimg = QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w = img8.shape[:2]
            qimg = QImage(img8.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        self._ref_image.setPixmap(
            pixmap.scaled(
                self._ref_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        dtype = "16-bit" if self._image.dtype == np.uint16 else "8-bit"
        channels = "Gray" if self._image.ndim == 2 else f"{self._image.shape[2]} ch"
        self._ref_meta.setText(f"{w} x {h}   {dtype}   {channels}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_reference()

    @staticmethod
    def _to_preview_8bit(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        return np.ascontiguousarray(image)
