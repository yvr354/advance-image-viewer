"""
Image Comparison Panel — Task 4

Load multiple images of the same scene (different formats, exposures,
camera settings) and instantly rank them by objective quality metrics.

Use cases:
  - Compare TIFF vs PNG vs JPEG — see compression quality loss
  - Compare 10 captures of same part — find sharpest
  - Compare exposure settings — find optimal exposure
  - Compare different gain values — find best SNR

Layout:
  Top    — toolbar (load, clear, export best)
  Middle — scrollable card grid, one card per image
           each card: thumbnail + all metrics + rank badge
  Bottom — side-by-side viewer of any two selected images
           with difference overlay option
"""

import os
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QLabel, QGridLayout, QFrame,
    QFileDialog, QComboBox, QCheckBox, QSizePolicy,
    QSplitter, QGroupBox, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor

from src.core.image_loader import load_image, is_supported
from src.analysis.focus_engine import FocusEngine
from src.analysis.quality_engine import QualityEngine
from src.ui.theme import (
    COLOR_PERFECT, COLOR_GOOD, COLOR_WARN, COLOR_FAIL,
    BG_RAISED, BG_CONTROL, ACCENT, TEXT_PRIMARY, TEXT_SECONDARY,
)


# ── Background worker — analyze one image ──────────────────────────────────

class ImageAnalyzeWorker(QThread):
    result_ready = pyqtSignal(str, object, object)   # path, focus_result, quality_result

    def __init__(self, path: str, focus: FocusEngine, quality: QualityEngine):
        super().__init__()
        self._path    = path
        self._focus   = focus
        self._quality = quality

    def run(self):
        try:
            data  = load_image(self._path)
            fr    = self._focus.analyze(data.raw)
            qr    = self._quality.analyze(data.raw)
            self.result_ready.emit(self._path, fr, qr)
        except Exception:
            pass


# ── Single image card ──────────────────────────────────────────────────────

class ImageCard(QFrame):
    selected    = pyqtSignal(object)   # emits self
    compare_req = pyqtSignal(object)   # emits self — add to compare slot

    THUMB_SIZE = 120

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path          = path
        self.focus_result  = None
        self.quality_result= None
        self.rank          = 0
        self._selected     = False

        self.setFixedWidth(160)
        self.setMinimumHeight(280)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QFrame {{
                background: {BG_RAISED};
                border: 1px solid #252535;
                border-radius: 6px;
            }}
            QFrame:hover {{
                border-color: {ACCENT};
            }}
        """)
        self._build()
        self._load_thumb()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Rank badge
        self._rank_label = QLabel("")
        self._rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rank_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self._rank_label.setFixedHeight(18)
        layout.addWidget(self._rank_label)

        # Thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(self.THUMB_SIZE + 8, self.THUMB_SIZE + 8)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet("background: #0A0A0F; border-radius: 3px;")
        layout.addWidget(self._thumb, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Filename
        name = os.path.basename(self.path)
        if len(name) > 18:
            name = name[:8] + "…" + name[-7:]
        self._name_label = QLabel(name)
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 10px;")
        self._name_label.setToolTip(self.path)
        layout.addWidget(self._name_label)

        # Format + size
        ext  = os.path.splitext(self.path)[1].upper().lstrip(".")
        size = os.path.getsize(self.path) / 1024
        self._meta_label = QLabel(f"{ext}  {size:.0f} KB")
        self._meta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._meta_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9px;")
        layout.addWidget(self._meta_label)

        # Metrics area
        self._metrics_widget = QWidget()
        ml = QVBoxLayout(self._metrics_widget)
        ml.setContentsMargins(2, 2, 2, 2)
        ml.setSpacing(2)

        self._focus_row   = self._make_metric_row("Focus",   "—")
        self._quality_row = self._make_metric_row("Quality", "—")
        self._snr_row     = self._make_metric_row("SNR",     "—")
        self._verdict_label = QLabel("Analyzing…")
        self._verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self._verdict_label.setStyleSheet(f"color: {TEXT_SECONDARY};")

        ml.addLayout(self._focus_row[0])
        ml.addLayout(self._quality_row[0])
        ml.addLayout(self._snr_row[0])
        ml.addWidget(self._verdict_label)
        layout.addWidget(self._metrics_widget)

        # Compare button
        self._cmp_btn = QPushButton("＋ Compare")
        self._cmp_btn.setFixedHeight(22)
        self._cmp_btn.setStyleSheet(f"""
            QPushButton {{
                background: {BG_CONTROL}; color: {ACCENT};
                border: 1px solid {ACCENT}44; border-radius: 3px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {ACCENT}22; border-color: {ACCENT};
            }}
        """)
        self._cmp_btn.clicked.connect(lambda: self.compare_req.emit(self))
        layout.addWidget(self._cmp_btn)

        layout.addStretch()

    def _make_metric_row(self, label: str, value: str):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9px;")
        lbl.setFixedWidth(45)
        val = QLabel(value)
        val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 9px; font-weight: 600;")
        val.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(lbl)
        row.addWidget(val)
        return row, val

    def _load_thumb(self):
        try:
            img = cv2.imread(self.path, cv2.IMREAD_REDUCED_COLOR_4)
            if img is None:
                img = cv2.imread(self.path)
            if img is None:
                return
            h, w = img.shape[:2]
            scale = self.THUMB_SIZE / max(h, w)
            img   = cv2.resize(img, (int(w*scale), int(h*scale)))
            rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg  = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                           rgb.shape[1]*3, QImage.Format.Format_RGB888)
            self._thumb.setPixmap(QPixmap.fromImage(qimg.copy()))
        except Exception:
            pass

    def set_results(self, focus_result, quality_result):
        self.focus_result   = focus_result
        self.quality_result = quality_result

        f  = focus_result
        q  = quality_result

        # Color maps
        fc = COLOR_PERFECT if f.score >= 700 else COLOR_GOOD if f.score >= 400 else COLOR_WARN if f.score >= 200 else COLOR_FAIL
        qc = COLOR_PERFECT if q.verdict == "PASS" else COLOR_FAIL

        self._focus_row[1].setText(f"{f.score:.0f}")
        self._focus_row[1].setStyleSheet(f"color: {fc}; font-size: 9px; font-weight: 700;")

        self._quality_row[1].setText(f"{q.overall_score:.0f}/100")
        self._quality_row[1].setStyleSheet(f"color: {qc}; font-size: 9px; font-weight: 700;")

        self._snr_row[1].setText(f"{q.snr_db:.0f} dB")

        verdict = f"{f.verdict} · {q.verdict}"
        self._verdict_label.setText(verdict)
        self._verdict_label.setStyleSheet(f"color: {fc}; font-size: 9px; font-weight: 700;")

    def set_rank(self, rank: int, total: int):
        self.rank = rank
        if rank == 1:
            self._rank_label.setText("🥇  BEST")
            self._rank_label.setStyleSheet(f"color: {COLOR_PERFECT}; font-size: 10px;")
            self.setStyleSheet(self.styleSheet().replace(
                "border: 1px solid #252535", f"border: 2px solid {COLOR_PERFECT}"
            ))
        elif rank == 2:
            self._rank_label.setText(f"#{rank}")
            self._rank_label.setStyleSheet(f"color: {COLOR_GOOD}; font-size: 10px;")
        elif rank == total:
            self._rank_label.setText(f"#{rank}  WORST")
            self._rank_label.setStyleSheet(f"color: {COLOR_FAIL}; font-size: 10px;")
        else:
            self._rank_label.setText(f"#{rank}")
            self._rank_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")

    def mousePressEvent(self, event):
        self.selected.emit(self)


# ── Side-by-side comparison view ──────────────────────────────────────────

class CompareView(QWidget):
    """Shows two images side by side with difference option."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_a: np.ndarray | None = None
        self._image_b: np.ndarray | None = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._mode = QComboBox()
        self._mode.addItems(["Side by Side", "Difference (A−B)", "Overlay Blend", "Flicker"])
        self._mode.currentTextChanged.connect(self._update)

        self._swap_btn = QPushButton("⇄ Swap")
        self._swap_btn.setFixedWidth(70)
        self._swap_btn.clicked.connect(self._swap)

        self._label_a = QLabel("A: —")
        self._label_b = QLabel("B: —")
        for lbl in [self._label_a, self._label_b]:
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")

        toolbar.addWidget(QLabel("Mode:"))
        toolbar.addWidget(self._mode)
        toolbar.addWidget(self._swap_btn)
        toolbar.addStretch()
        toolbar.addWidget(self._label_a)
        toolbar.addWidget(QLabel(" vs "))
        toolbar.addWidget(self._label_b)
        layout.addLayout(toolbar)

        # Display
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._view_a = QLabel()
        self._view_b = QLabel()
        for v in [self._view_a, self._view_b]:
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.setStyleSheet("background: #0A0A0F; border-radius: 4px;")
            v.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            v.setMinimumSize(200, 150)
        self._splitter.addWidget(self._view_a)
        self._splitter.addWidget(self._view_b)
        layout.addWidget(self._splitter)

        # Diff metrics bar
        self._diff_label = QLabel("")
        self._diff_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._diff_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
        layout.addWidget(self._diff_label)

        self.setMinimumHeight(200)

    def set_image_a(self, image: np.ndarray, name: str):
        self._image_a = image
        self._label_a.setText(f"A: {name}")
        self._update()

    def set_image_b(self, image: np.ndarray, name: str):
        self._image_b = image
        self._label_b.setText(f"B: {name}")
        self._update()

    def _swap(self):
        self._image_a, self._image_b = self._image_b, self._image_a
        a_txt = self._label_a.text()
        b_txt = self._label_b.text()
        self._label_a.setText(b_txt.replace("B:", "A:"))
        self._label_b.setText(a_txt.replace("A:", "B:"))
        self._update()

    def _update(self):
        mode = self._mode.currentText()
        if self._image_a is None and self._image_b is None:
            return

        if mode == "Side by Side":
            self._view_a.setVisible(True)
            self._view_b.setVisible(True)
            if self._image_a is not None:
                self._show(self._view_a, self._image_a)
            if self._image_b is not None:
                self._show(self._view_b, self._image_b)
            self._diff_label.setText("")

        elif mode == "Difference (A−B)":
            if self._image_a is not None and self._image_b is not None:
                diff = self._compute_diff(self._image_a, self._image_b)
                self._view_a.setVisible(True)
                self._view_b.setVisible(True)
                self._show(self._view_a, self._image_a)
                self._show(self._view_b, diff)
                mean_diff = float(np.mean(np.abs(
                    self._image_a.astype(np.float32) - self._image_b.astype(np.float32)
                )))
                self._diff_label.setText(
                    f"Mean absolute difference: {mean_diff:.2f} / 255   "
                    f"({mean_diff/255*100:.2f}%)   "
                    f"{'Images are identical' if mean_diff < 0.5 else 'Images differ'}"
                )

        elif mode == "Overlay Blend":
            if self._image_a is not None and self._image_b is not None:
                a8 = self._to_8bit(self._image_a)
                b8 = self._to_8bit(self._image_b)
                if a8.shape != b8.shape:
                    b8 = cv2.resize(b8, (a8.shape[1], a8.shape[0]))
                blended = cv2.addWeighted(a8, 0.5, b8, 0.5, 0)
                self._view_a.setVisible(True)
                self._view_b.setVisible(False)
                self._show(self._view_a, blended)

    def _compute_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a8 = self._to_8bit(a).astype(np.float32)
        b8 = self._to_8bit(b).astype(np.float32)
        if a8.shape != b8.shape:
            b8 = cv2.resize(b8, (a8.shape[1], a8.shape[0]))
        diff = np.abs(a8 - b8)
        # Amplify 8× so small differences are visible
        diff = np.clip(diff * 8, 0, 255).astype(np.uint8)
        if diff.ndim == 2:
            diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        return diff

    def _show(self, label: QLabel, image: np.ndarray):
        img = self._to_8bit(image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img.shape[:2]
        max_w = max(label.width() - 8, 200)
        max_h = max(label.height() - 8, 150)
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        qimg = QImage(img.data, img.shape[1], img.shape[0],
                      img.shape[1]*3, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg.copy()))

    @staticmethod
    def _to_8bit(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            return (image >> 8).astype(np.uint8)
        return image.astype(np.uint8)


# ── Main Comparison Panel ──────────────────────────────────────────────────

class ComparisonPanel(QWidget):
    """
    Full image comparison panel.
    Load any number of images → automatic quality ranking → side-by-side compare.
    """

    open_image_requested = pyqtSignal(str)   # ask main window to open this image

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: list[ImageCard]       = []
        self._workers: list[ImageAnalyzeWorker] = []
        self._compare_slots: list[ImageCard | None] = [None, None]
        self._sort_key = "quality"

        self.focus_engine   = FocusEngine()
        self.quality_engine = QualityEngine()

        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Toolbar ────────────────────────────────────────────────
        toolbar = QHBoxLayout()

        btn_load = QPushButton("+ Load Images")
        btn_load.setProperty("accent", True)
        btn_load.setToolTip(
            "Load multiple images to compare.\n"
            "Can be same scene in different formats (TIFF vs PNG vs JPEG),\n"
            "different exposures, different capture settings.\n"
            "Automatic quality ranking tells you which is best for AI training."
        )
        btn_load.clicked.connect(self._load_images)

        btn_clear = QPushButton("Clear All")
        btn_clear.setProperty("danger", True)
        btn_clear.clicked.connect(self._clear_all)

        btn_best = QPushButton("⬇ Export Best")
        btn_best.setToolTip("Export the highest-ranked image to a chosen folder.")
        btn_best.clicked.connect(self._export_best)

        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Sort: Quality Score", "Sort: Focus Score", "Sort: SNR", "Sort: Filename"])
        self._sort_combo.setToolTip("Re-rank images by selected metric.")
        self._sort_combo.currentTextChanged.connect(self._resort)

        self._count_label = QLabel("No images loaded")
        self._count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")

        toolbar.addWidget(btn_load)
        toolbar.addWidget(btn_clear)
        toolbar.addWidget(btn_best)
        toolbar.addWidget(self._sort_combo)
        toolbar.addStretch()
        toolbar.addWidget(self._count_label)
        layout.addLayout(toolbar)

        # Instructions
        self._hint = QLabel(
            "Load multiple images of the same scene.  "
            "Click ＋ Compare on any two to compare them side by side.  "
            "Best image for AI training is ranked #1."
        )
        self._hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        # ── Splitter: cards on top, compare view on bottom ─────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Card scroll area
        self._card_scroll = QScrollArea()
        self._card_scroll.setWidgetResizable(True)
        self._card_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._cards_widget = QWidget()
        self._cards_layout = QHBoxLayout(self._cards_widget)
        self._cards_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._cards_layout.setSpacing(10)
        self._cards_layout.setContentsMargins(4, 4, 4, 4)
        self._card_scroll.setWidget(self._cards_widget)
        self._card_scroll.setMinimumHeight(320)

        # Compare view
        self._compare_view = CompareView()
        self._compare_view.setMinimumHeight(220)

        splitter.addWidget(self._card_scroll)
        splitter.addWidget(self._compare_view)
        splitter.setSizes([340, 240])
        layout.addWidget(splitter)

    # ── Load & analyze ─────────────────────────────────────────────

    def _load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Images for Comparison", "",
            "All Images (*.tiff *.tif *.png *.bmp *.jpg *.jpeg *.pgm *.ppm *.exr *.hdr);;"
            "All Files (*)"
        )
        for path in paths:
            if not any(c.path == path for c in self._cards):
                self._add_card(path)

    def _add_card(self, path: str):
        card = ImageCard(path)
        card.selected.connect(self._on_card_selected)
        card.compare_req.connect(self._on_compare_req)
        self._cards.append(card)
        self._cards_layout.addWidget(card)
        self._count_label.setText(f"{len(self._cards)} images")

        # Start analysis worker
        worker = ImageAnalyzeWorker(path, self.focus_engine, self.quality_engine)
        worker.result_ready.connect(self._on_analysis_ready)
        self._workers.append(worker)
        worker.start()

    def _on_analysis_ready(self, path: str, focus_result, quality_result):
        for card in self._cards:
            if card.path == path:
                card.set_results(focus_result, quality_result)
                break
        self._rank_cards()

    def _rank_cards(self):
        """Re-rank all cards that have results."""
        ready = [c for c in self._cards if c.quality_result is not None]
        if not ready:
            return

        key = self._sort_combo.currentText()
        if "Focus" in key:
            ready.sort(key=lambda c: c.focus_result.score, reverse=True)
        elif "SNR" in key:
            ready.sort(key=lambda c: c.quality_result.snr_db, reverse=True)
        elif "Filename" in key:
            ready.sort(key=lambda c: os.path.basename(c.path))
        else:  # Quality Score (default)
            ready.sort(key=lambda c: c.quality_result.overall_score, reverse=True)

        for i, card in enumerate(ready):
            card.set_rank(i + 1, len(ready))

    def _resort(self):
        self._rank_cards()

    # ── Selection & comparison ─────────────────────────────────────

    def _on_card_selected(self, card: ImageCard):
        """Single click → open in main viewer."""
        self.open_image_requested.emit(card.path)

    def _on_compare_req(self, card: ImageCard):
        """Click ＋ Compare → fill slot A then B."""
        try:
            data = load_image(card.path)
            name = os.path.basename(card.path)
            if self._compare_slots[0] is None:
                self._compare_slots[0] = card
                self._compare_view.set_image_a(data.raw, name)
            else:
                self._compare_slots[1] = card
                self._compare_view.set_image_b(data.raw, name)
                # Reset for next pair
                self._compare_slots = [None, None]
        except Exception:
            pass

    # ── Export ─────────────────────────────────────────────────────

    def _export_best(self):
        best = next((c for c in self._cards if c.rank == 1), None)
        if not best:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Best Image",
            os.path.basename(best.path),
            "PNG (*.png);;TIFF (*.tiff);;All Files (*)"
        )
        if dest:
            import shutil
            shutil.copy2(best.path, dest)

    def _clear_all(self):
        for worker in self._workers:
            worker.quit()
        self._workers.clear()
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._compare_slots = [None, None]
        self._count_label.setText("No images loaded")

    # ── Add current image from main viewer ─────────────────────────

    def add_image_from_path(self, path: str):
        """Called by main window to add currently open image to comparison."""
        if path and os.path.isfile(path) and not any(c.path == path for c in self._cards):
            self._add_card(path)
