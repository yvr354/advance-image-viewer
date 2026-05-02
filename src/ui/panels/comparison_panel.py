"""
Professional A/B Image Comparison Panel.

Layout:
  ┌─ Header bar ──────────────────────────────────────────────────────────┐
  │  [A: filename  W×H]  [B: filename  W×H]  [⇄ Swap]  Amp──●──  Mode▼  │
  ├─ Metrics bar ─────────────────────────────────────────────────────────┤
  │  SSIM: 0.947  PSNR: 38.2 dB  Diff: 1.8%  Max Δ: 43  ✓ PASS          │
  ├─────────────────────┬─────────────────────┬───────────────────────────┤
  │                     │                     │                           │
  │   REFERENCE  (A)    │   TEST  (B)         │   DIFF MAP                │
  │                     │                     │   red = changed           │
  ├─────────────────────┴─────────────────────┴───────────────────────────┤
  │  Pixel Loupe:  [A patch]  ΔR/G/B  [B patch]    X: 1024  Y: 768       │
  └───────────────────────────────────────────────────────────────────────┘

Usage:
  • Click panel A or B header to activate it (bright blue border).
  • Then click any image in the File Browser — it loads into the active panel.
  • Or drag an image file directly onto A or B.
  • Zoom/pan any panel — all three stay in sync.
  • Hover anywhere — pixel loupe shows both A and B at that position.
"""

import os
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QSlider, QComboBox,
    QFileDialog, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QDragEnterEvent, QDropEvent

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

from src.core.image_loader import load_image
from src.ui.panels.gl_viewer import GLImageViewer
from src.ui.theme import ACCENT, TEXT_PRIMARY, TEXT_SECONDARY, BG_RAISED, BG_CONTROL


# ── Background diff / metrics worker ──────────────────────────────────────

class DiffWorker(QThread):
    """Computes diff image + metrics off the main thread."""
    done = pyqtSignal(np.ndarray, dict, bool)   # diff_rgb, metrics, auto_normalized

    def __init__(self, img_a: np.ndarray, img_b: np.ndarray, amp: float):
        super().__init__()
        self._a   = img_a
        self._b   = img_b
        self._amp = amp

    def run(self):
        try:
            a, b = self._normalise(self._a), self._normalise(self._b)

            # Resize B to A's size if different
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

            ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
            bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY) if b.ndim == 3 else b

            # Raw absolute difference
            diff_raw = cv2.absdiff(ag.astype(np.int16), bg.astype(np.int16))
            diff_raw = diff_raw.astype(np.float32)

            # Display: amplify then auto-normalize so even tiny diffs are visible.
            # If max after amplification < 32, stretch to full range automatically.
            amplified = diff_raw * self._amp
            amp_max   = float(np.max(amplified))
            if amp_max < 32 and amp_max > 0:
                # Auto-stretch: tiny differences (e.g. JPEG artifacts of 3-4 px)
                # become fully visible instead of invisible near-black
                amplified = amplified * (255.0 / amp_max)
            amp_clipped = np.clip(amplified, 0, 255).astype(np.uint8)
            diff_color  = cv2.applyColorMap(amp_clipped, cv2.COLORMAP_HOT)
            diff_rgb    = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

            # Metrics
            n        = ag.size
            nonzero  = float(np.count_nonzero(diff_raw > 5)) / n * 100
            max_dev  = int(np.max(diff_raw))
            mean_dev = float(np.mean(diff_raw))

            metrics = {
                "pixel_diff": nonzero,
                "max_dev":    max_dev,
                "mean_dev":   mean_dev,
                "ssim":       None,
                "psnr":       None,
                "verdict":    "PASS" if nonzero < 2.0 else "FAIL",
            }

            if _SKIMAGE:
                try:
                    metrics["ssim"] = float(structural_similarity(ag, bg, data_range=255))
                    metrics["psnr"] = float(peak_signal_noise_ratio(ag, bg, data_range=255))
                    metrics["verdict"] = "PASS" if metrics["ssim"] >= 0.92 else "FAIL"
                except Exception:
                    pass

            self.done.emit(diff_rgb, metrics, amp_max < 32 and amp_max > 0)
        except Exception:
            pass

    @staticmethod
    def _normalise(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint16:
            return (img >> 8).astype(np.uint8)
        if img.dtype != np.uint8:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img


# ── Slot panel (A or B) ───────────────────────────────────────────────────

class SlotPanel(QWidget):
    """One viewer slot — header bar + GLImageViewer. Emits activated on click."""

    activated = pyqtSignal(str)   # 'A' or 'B'
    load_requested = pyqtSignal(str, str)   # slot, path (from drag-drop)

    _ACTIVE_HDR   = "background:#0E2438; color:#00AAFF; padding:4px 8px; font-weight:700; font-size:10px;"
    _INACTIVE_HDR = "background:#0A1520; color:#445566; padding:4px 8px; font-size:10px;"
    _ACTIVE_BORDER   = "QWidget#SlotPanel { border: 2px solid #00AAFF; }"
    _INACTIVE_BORDER = "QWidget#SlotPanel { border: 2px solid #1A2535; }"

    def __init__(self, slot: str, parent=None):
        super().__init__(parent)
        self.slot    = slot      # 'A' or 'B'
        self._active = False
        self._path   = ""
        self._w = self._h = 0
        self.setObjectName("SlotPanel")
        self.setAcceptDrops(True)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Header click area
        self._header = QLabel(self._default_text())
        self._header.setFixedHeight(26)
        self._header.setStyleSheet(self._INACTIVE_HDR)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.mousePressEvent = lambda _: self.activated.emit(self.slot)
        lay.addWidget(self._header)

        # GL viewer
        self.viewer = GLImageViewer()
        self.viewer.setAcceptDrops(False)   # panel handles drops
        orig = self.viewer.mousePressEvent
        def _fwd(ev, _o=orig):
            self.activated.emit(self.slot)
            _o(ev)
        self.viewer.mousePressEvent = _fwd
        lay.addWidget(self.viewer, stretch=1)

        self._set_active(False)

    def _default_text(self):
        label = "REFERENCE" if self.slot == "A" else "TEST"
        return f"  {self.slot} — {label}   ·   click to activate, then pick image in browser"

    def set_active(self, active: bool):
        self._active = active
        self._set_active(active)

    def _set_active(self, active: bool):
        self._header.setStyleSheet(self._ACTIVE_HDR if active else self._INACTIVE_HDR)
        self.setStyleSheet(self._ACTIVE_BORDER if active else self._INACTIVE_BORDER)

    def set_image(self, path: str, img: np.ndarray):
        self._path = path
        self._h, self._w = img.shape[:2]
        self.viewer.set_image(img)
        self._refresh_header()

    def _refresh_header(self):
        name = os.path.basename(self._path)
        if len(name) > 28:
            name = name[:14] + "…" + name[-12:]
        label = "REFERENCE" if self.slot == "A" else "TEST"
        self._header.setText(
            f"  {self.slot} — {label}   ·   {name}   {self._w}×{self._h} px"
        )

    def update_zoom(self, zoom: float):
        if not self._path:
            return
        name = os.path.basename(self._path)
        if len(name) > 28:
            name = name[:14] + "…" + name[-12:]
        label = "REFERENCE" if self.slot == "A" else "TEST"
        pct   = f"{zoom*100:.0f}%"
        self._header.setText(
            f"  {self.slot} — {label}   ·   {name}   {self._w}×{self._h} px   {pct}"
        )

    # Drag-drop
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.activated.emit(self.slot)
                self.load_requested.emit(self.slot, path)
                break


# ── Diff panel ────────────────────────────────────────────────────────────

class DiffPanel(QWidget):
    """Third panel — shows computed diff map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._amp = 4
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._header = QLabel("  DIFF MAP   ·   load both images to compute")
        self._header.setFixedHeight(26)
        self._header.setStyleSheet(
            "background:#101A10; color:#336633; padding:4px 8px; font-size:10px;"
        )
        lay.addWidget(self._header)

        self.viewer = GLImageViewer()
        lay.addWidget(self.viewer, stretch=1)

    def set_diff(self, diff_rgb: np.ndarray, amp: int, auto_normalized: bool = False):
        self._amp = amp
        self.viewer.set_image(diff_rgb)
        note = "  ⚡ auto-normalized (differences are very small)" if auto_normalized else f"  amplify ×{amp}"
        self._header.setText(f"  DIFF MAP   ·   hot = changed  · {note}")
        self._header.setStyleSheet(
            "background:#101A10; color:#44AA44; padding:4px 8px; "
            "font-size:10px; font-weight:700;"
        )

    def clear(self):
        self._header.setText("  DIFF MAP   ·   load both images to compute")
        self._header.setStyleSheet(
            "background:#101A10; color:#336633; padding:4px 8px; font-size:10px;"
        )


# ── Pixel Loupe ───────────────────────────────────────────────────────────

class PixelLoupe(QWidget):
    """Shows A and B pixel patches at hover position side by side."""

    PATCH_PX = 16   # pixels extracted from image
    DISP_PX  = 96   # display size of loupe square

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_a: np.ndarray | None = None
        self._img_b: np.ndarray | None = None
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(16)

        def _col(title):
            col = QVBoxLayout()
            col.setSpacing(2)
            ttl = QLabel(title)
            ttl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:9px;")
            ttl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img = QLabel()
            img.setFixedSize(self.DISP_PX, self.DISP_PX)
            img.setStyleSheet("background:#08080F; border:1px solid #1A2535;")
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("R:—  G:—  B:—")
            val.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:9px; font-weight:600;")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(ttl)
            col.addWidget(img)
            col.addWidget(val)
            return col, img, val

        col_a, self._img_lbl_a, self._val_a = _col("A — pixel")
        col_b, self._img_lbl_b, self._val_b = _col("B — pixel")

        # Delta column
        delta_col = QVBoxLayout()
        delta_col.setSpacing(4)
        delta_ttl = QLabel("Δ (A−B)")
        delta_ttl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:9px;")
        delta_ttl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._delta_lbl = QLabel("—")
        self._delta_lbl.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:11px; font-weight:700;"
        )
        self._delta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._delta_lbl.setWordWrap(True)
        self._delta_lbl.setFixedWidth(80)
        delta_col.addWidget(delta_ttl)
        delta_col.addStretch()
        delta_col.addWidget(self._delta_lbl)
        delta_col.addStretch()

        self._coord_lbl = QLabel("X: —   Y: —")
        self._coord_lbl.setStyleSheet(f"color:{ACCENT}; font-size:10px; font-weight:600;")
        self._coord_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addLayout(col_a)
        lay.addLayout(delta_col)
        lay.addLayout(col_b)
        lay.addStretch()
        lay.addWidget(self._coord_lbl, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.setFixedHeight(self.DISP_PX + 50)
        self.setStyleSheet(f"background:{BG_RAISED}; border-top:1px solid #1A2535;")

    def update(self, ix: int, iy: int, img_a, img_b):
        self._coord_lbl.setText(f"X: {ix}   Y: {iy}")
        px_a = self._render_patch(img_a, ix, iy, self._img_lbl_a, self._val_a)
        px_b = self._render_patch(img_b, ix, iy, self._img_lbl_b, self._val_b)
        if px_a is not None and px_b is not None:
            dr = int(px_a[0]) - int(px_b[0])
            dg = int(px_a[1]) - int(px_b[1])
            db = int(px_a[2]) - int(px_b[2])
            self._delta_lbl.setText(f"R:{dr:+d}\nG:{dg:+d}\nB:{db:+d}")
        else:
            self._delta_lbl.setText("—")

    def _render_patch(self, img, cx, cy, lbl, val_lbl):
        if img is None:
            lbl.clear()
            val_lbl.setText("R:—  G:—  B:—")
            return None
        img8 = self._to_8bit(img)
        if img8.ndim == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        h, w = img8.shape[:2]
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return None

        # Extract patch
        half = self.PATCH_PX // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, x1 + self.PATCH_PX), min(h, y1 + self.PATCH_PX)
        patch = img8[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        # Scale up (nearest-neighbour — keep pixels sharp)
        scale = self.DISP_PX // self.PATCH_PX
        big   = cv2.resize(patch, (patch.shape[1]*scale, patch.shape[0]*scale),
                           interpolation=cv2.INTER_NEAREST)
        qimg  = QImage(big.data, big.shape[1], big.shape[0],
                       big.shape[1]*3, QImage.Format.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg.copy()))

        # Pixel value at cursor
        px = img8[cy, cx]
        if px.ndim == 0 or len(px) < 3:
            val_lbl.setText(f"L:{int(px)}")
            return (int(px),) * 3
        val_lbl.setText(f"R:{px[0]}  G:{px[1]}  B:{px[2]}")
        return (int(px[0]), int(px[1]), int(px[2]))

    @staticmethod
    def _to_8bit(img):
        if img.dtype == np.uint16:
            return (img >> 8).astype(np.uint8)
        if img.dtype != np.uint8:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img


# ── Main comparison panel ─────────────────────────────────────────────────

class ComparisonPanel(QWidget):
    """
    Professional A/B/Diff comparison panel.
    External API:
      load_path(path)  — loads into active slot (called from main window / browser)
      active_slot      — 'A' or 'B'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_a:  np.ndarray | None = None
        self._img_b:  np.ndarray | None = None
        self._active  = "A"
        self._diff_worker: DiffWorker | None = None
        self._flicker_timer = QTimer()
        self._flicker_timer.timeout.connect(self._flicker_tick)
        self._flicker_state = False
        self._build()

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def active_slot(self) -> str:
        return self._active

    def load_path(self, path: str):
        """Load image file into the currently active slot."""
        try:
            data = load_image(path)
            img  = data.raw
        except Exception as e:
            return
        if self._active == "A":
            self._img_a = img
            self._slot_a.set_image(path, img)
        else:
            self._img_b = img
            self._slot_b.set_image(path, img)
        self._trigger_diff()

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())
        root.addWidget(self._build_metrics_bar())
        root.addWidget(self._build_viewers(), stretch=1)
        root.addWidget(self._build_loupe())

        self._activate("A")

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(38)
        bar.setStyleSheet(
            f"background:#080E18; border-bottom:1px solid #1A2535;"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)

        _BTN = (
            "QPushButton { background:#0A1828; color:#5588AA; border:1px solid #1A2A3A; "
            "              padding:3px 10px; font-size:10px; border-radius:3px; }"
            "QPushButton:hover { background:#112233; color:#88CCEE; }"
        )

        # Load A
        self._load_a_btn = QPushButton("📂  Load Reference (A)")
        self._load_a_btn.setStyleSheet(_BTN)
        self._load_a_btn.setToolTip("Load reference / known-good image into slot A")
        self._load_a_btn.clicked.connect(lambda: self._browse_and_load("A"))
        lay.addWidget(self._load_a_btn)

        # Load B
        self._load_b_btn = QPushButton("📂  Load Test (B)")
        self._load_b_btn.setStyleSheet(_BTN)
        self._load_b_btn.setToolTip("Load test / inspection image into slot B")
        self._load_b_btn.clicked.connect(lambda: self._browse_and_load("B"))
        lay.addWidget(self._load_b_btn)

        lay.addSpacing(12)

        # Swap
        swap_btn = QPushButton("⇄  Swap A↔B")
        swap_btn.setStyleSheet(_BTN)
        swap_btn.setToolTip("Swap reference and test images")
        swap_btn.clicked.connect(self._swap)
        lay.addWidget(swap_btn)

        lay.addSpacing(12)

        # Diff mode
        mode_lbl = QLabel("Diff:")
        mode_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:10px;")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Diff Map", "Blend 50/50", "Flicker"])
        self._mode_combo.setToolTip(
            "Diff Map   — hot colormap, red = changed pixels\n"
            "Blend      — 50% A + 50% B overlay\n"
            "Flicker    — alternates A and B every 600ms"
        )
        self._mode_combo.setFixedWidth(110)
        self._mode_combo.setStyleSheet(
            "QComboBox { background:#0A1828; color:#88AACC; border:1px solid #1A2A3A; "
            "            padding:2px 6px; font-size:10px; border-radius:3px; }"
            "QComboBox::drop-down { border:none; }"
        )
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        lay.addWidget(mode_lbl)
        lay.addWidget(self._mode_combo)

        lay.addSpacing(12)

        # Amplify slider
        amp_lbl = QLabel("Amplify:")
        amp_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:10px;")
        self._amp_slider = QSlider(Qt.Orientation.Horizontal)
        self._amp_slider.setRange(1, 20)
        self._amp_slider.setValue(4)
        self._amp_slider.setFixedWidth(80)
        self._amp_slider.setToolTip(
            "Boost subtle pixel differences so they become visible.\n"
            "1× = raw diff  ·  10× = 10x brighter  ·  20× = maximum"
        )
        self._amp_val_lbl = QLabel("×4")
        self._amp_val_lbl.setStyleSheet(f"color:{ACCENT}; font-size:10px; font-weight:700;")
        self._amp_val_lbl.setFixedWidth(24)
        self._amp_slider.valueChanged.connect(self._on_amp_changed)
        lay.addWidget(amp_lbl)
        lay.addWidget(self._amp_slider)
        lay.addWidget(self._amp_val_lbl)

        lay.addStretch()

        # Active slot indicator
        self._active_lbl = QLabel("Active: A")
        self._active_lbl.setStyleSheet(
            f"color:{ACCENT}; font-size:10px; font-weight:700;"
        )
        lay.addWidget(self._active_lbl)

        return bar

    def _build_metrics_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet(
            "background:#060C12; border-bottom:1px solid #1A2535;"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(0)

        self._metrics_lbl = QLabel(
            "Load both images to see comparison metrics"
        )
        self._metrics_lbl.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:10px;"
        )
        lay.addWidget(self._metrics_lbl)
        lay.addStretch()
        return bar

    def _build_viewers(self) -> QWidget:
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)

        self._slot_a = SlotPanel("A")
        self._slot_b = SlotPanel("B")
        self._diff_panel = DiffPanel()

        self._splitter.addWidget(self._slot_a)
        self._splitter.addWidget(self._slot_b)
        self._splitter.addWidget(self._diff_panel)
        self._splitter.setSizes([1, 1, 1])

        # Activate on click
        self._slot_a.activated.connect(self._activate)
        self._slot_b.activated.connect(self._activate)

        # Drag-drop load
        self._slot_a.load_requested.connect(lambda _s, p: self._load_into("A", p))
        self._slot_b.load_requested.connect(lambda _s, p: self._load_into("B", p))

        # Zoom display in headers
        self._slot_a.viewer.zoom_changed.connect(self._slot_a.update_zoom)
        self._slot_b.viewer.zoom_changed.connect(self._slot_b.update_zoom)

        # Linked zoom/pan across all three
        self._slot_a.viewer.view_state_changed.connect(self._sync_from_a)
        self._slot_b.viewer.view_state_changed.connect(self._sync_from_b)

        # Pixel loupe hover
        self._slot_a.viewer.pixel_hovered.connect(self._on_hover)
        self._slot_b.viewer.pixel_hovered.connect(self._on_hover)

        return self._splitter

    def _build_loupe(self) -> QWidget:
        self._loupe = PixelLoupe()
        return self._loupe

    # ── Activation ───────────────────────────────────────────────────────

    def _activate(self, slot: str):
        self._active = slot
        self._slot_a.set_active(slot == "A")
        self._slot_b.set_active(slot == "B")
        self._active_lbl.setText(f"Active: {slot}")

    # ── Load ─────────────────────────────────────────────────────────────

    def _browse_and_load(self, slot: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load image into slot {slot}",
            "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"
        )
        if path:
            self._activate(slot)
            self._load_into(slot, path)

    def _load_into(self, slot: str, path: str):
        try:
            data = load_image(path)
            img  = data.raw
        except Exception:
            return
        if slot == "A":
            self._img_a = img
            self._slot_a.set_image(path, img)
        else:
            self._img_b = img
            self._slot_b.set_image(path, img)
        self._trigger_diff()

    # ── Diff computation ──────────────────────────────────────────────────

    def _trigger_diff(self):
        if self._img_a is None or self._img_b is None:
            return
        mode = self._mode_combo.currentText()
        amp  = self._amp_slider.value()

        if self._diff_worker and self._diff_worker.isRunning():
            self._diff_worker.quit()

        if mode == "Blend 50/50":
            self._show_blend()
            return

        self._diff_worker = DiffWorker(self._img_a, self._img_b, amp)
        self._diff_worker.done.connect(self._on_diff_done)
        self._diff_worker.start()

    def _on_diff_done(self, diff_rgb: np.ndarray, metrics: dict, auto_normalized: bool):
        self._diff_panel.set_diff(diff_rgb, self._amp_slider.value(), auto_normalized)
        self._show_metrics(metrics)

    def _show_blend(self):
        if self._img_a is None or self._img_b is None:
            return
        a = self._to_8bit(self._img_a)
        b = self._to_8bit(self._img_b)
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        if b.ndim == 2:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        blend = cv2.addWeighted(a, 0.5, b, 0.5, 0)
        self._diff_panel.viewer.set_image(blend)
        self._diff_panel._header.setText("  BLEND   ·   50% A + 50% B overlay")

    def _show_metrics(self, m: dict):
        parts = []
        if m["ssim"] is not None:
            parts.append(f"SSIM: {m['ssim']:.3f}")
        if m["psnr"] is not None:
            psnr_str = f"{m['psnr']:.1f} dB" if m["psnr"] < 100 else "∞"
            parts.append(f"PSNR: {psnr_str}")
        parts.append(f"Diff pixels: {m['pixel_diff']:.1f}%")
        parts.append(f"Max Δ: {m['max_dev']}")
        parts.append(f"Mean Δ: {m['mean_dev']:.1f}")

        verdict  = m["verdict"]
        v_color  = "#44FF88" if verdict == "PASS" else "#FF4444"
        v_symbol = "✓" if verdict == "PASS" else "✗"
        metrics_str = "   │   ".join(parts)
        self._metrics_lbl.setText(
            f"{metrics_str}   │   "
            f"<span style='color:{v_color}; font-weight:700;'>{v_symbol} {verdict}</span>"
        )
        self._metrics_lbl.setTextFormat(Qt.TextFormat.RichText)

    # ── Sync zoom/pan ─────────────────────────────────────────────────────

    def _sync_from_a(self, zoom, ox, oy):
        for v in [self._slot_b.viewer, self._diff_panel.viewer]:
            v.blockSignals(True)
            v.set_view_state(zoom, ox, oy)
            v.blockSignals(False)
        self._slot_b.update_zoom(zoom)

    def _sync_from_b(self, zoom, ox, oy):
        for v in [self._slot_a.viewer, self._diff_panel.viewer]:
            v.blockSignals(True)
            v.set_view_state(zoom, ox, oy)
            v.blockSignals(False)
        self._slot_a.update_zoom(zoom)

    # ── Pixel loupe ───────────────────────────────────────────────────────

    def _on_hover(self, ix: int, iy: int, _pixel):
        self._loupe.update(ix, iy, self._img_a, self._img_b)

    # ── Controls ──────────────────────────────────────────────────────────

    def _swap(self):
        self._img_a, self._img_b = self._img_b, self._img_a
        path_a = self._slot_a._path
        path_b = self._slot_b._path
        img_a  = self._img_a
        img_b  = self._img_b
        if img_a is not None:
            self._slot_a.set_image(path_b, img_a)
        if img_b is not None:
            self._slot_b.set_image(path_a, img_b)
        self._trigger_diff()

    def _on_mode_changed(self, mode: str):
        if mode == "Flicker":
            self._flicker_timer.start(600)
        else:
            self._flicker_timer.stop()
            if mode == "Blend 50/50":
                self._show_blend()
            else:
                self._trigger_diff()

    def _flicker_tick(self):
        if self._img_a is None or self._img_b is None:
            return
        img = self._img_a if self._flicker_state else self._img_b
        label = "A — REFERENCE" if self._flicker_state else "B — TEST"
        self._diff_panel.viewer.set_image(img)
        self._diff_panel._header.setText(f"  FLICKER   ·   showing {label}")
        self._flicker_state = not self._flicker_state

    def _on_amp_changed(self, val: int):
        self._amp_val_lbl.setText(f"×{val}")
        mode = self._mode_combo.currentText()
        if mode == "Diff Map":
            self._trigger_diff()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_8bit(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint16:
            return (img >> 8).astype(np.uint8)
        if img.dtype != np.uint8:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img
