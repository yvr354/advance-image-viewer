"""Right-side inspector panel — focus, quality, histogram, pixel, ROI, profile, annotations, measurement."""

import math
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout,
    QProgressBar, QSizePolicy, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
import pyqtgraph as pg


class MetricRow(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        self._label = QLabel(label)
        self._label.setFixedWidth(90)
        self._value = QLabel("—")
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        layout.addWidget(self._label, 0, 0)
        layout.addWidget(self._value, 0, 1)
        layout.addWidget(self._bar,   1, 0, 1, 2)

    def set_value(self, text: str, pct: float, color: str = "#4CAF50"):
        self._value.setText(text)
        self._bar.setValue(int(pct))
        self._bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")


class InspectorPanel(QScrollArea):

    annotation_remove_requested = pyqtSignal(int)   # annotation index

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(160)
        self.setMaximumWidth(340)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setSpacing(4)
        self.setWidget(container)

        self._build_mask_status()
        self._build_focus_group()
        self._build_quality_group()
        self._build_histogram_group()
        self._build_pixel_group()
        self._build_roi_group()
        self._build_profile_group()
        self._build_annotation_group()
        self._build_measure_group()

        self._layout.addStretch()

    # ── Mask status banner ─────────────────────────────────────────────────

    def _build_mask_status(self):
        self._mask_status = QLabel("No mask — full image analyzed")
        self._mask_status.setWordWrap(True)
        self._mask_status.setStyleSheet(
            "color:#336633; font-size:9px; padding:3px 4px; "
            "background:#0A1408; border:1px solid #1A2A1A;"
        )
        self._layout.addWidget(self._mask_status)

    def update_mask_status(self, mask):
        if mask is None or not mask.polygons:
            self._mask_status.setText("No mask — full image analyzed")
            self._mask_status.setStyleSheet(
                "color:#336633; font-size:9px; padding:3px 4px; "
                "background:#0A1408; border:1px solid #1A2A1A;"
            )
            return
        n    = len(mask.polygons)
        pct  = mask.coverage_pct()
        auto = " (auto-detected)" if mask.auto_detected else ""
        conf = mask.align_confidence
        if conf < 100:
            conf_txt = f"  align confidence: {conf:.0f}%"
            conf_warn = "⚠ " if conf < 50 else ""
        else:
            conf_txt  = ""
            conf_warn = ""

        self._mask_status.setText(
            f"⬡ MASK ACTIVE{auto} — {n} region(s) — {pct:.1f}% of image\n"
            f"   {conf_warn}Metrics computed inside mask only{conf_txt}"
        )
        color = "#FFAA44" if conf < 50 else "#44FF88"
        self._mask_status.setStyleSheet(
            f"color:{color}; font-size:9px; padding:3px 4px; "
            f"background:#0A140A; border:1px solid #1A3A1A; font-weight:600;"
        )

    # ── Focus ──────────────────────────────────────────────────────────────

    def _build_focus_group(self):
        box = QGroupBox("Focus & Sharpness")
        layout = QVBoxLayout(box)
        layout.setSpacing(3)

        # Verdict + confidence on same line
        top_row = QHBoxLayout()
        self._focus_verdict = QLabel("—")
        self._focus_verdict.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self._focus_confidence = QLabel("")
        self._focus_confidence.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self._focus_confidence.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top_row.addWidget(self._focus_verdict)
        top_row.addStretch()
        top_row.addWidget(self._focus_confidence)
        layout.addLayout(top_row)

        self._focus_score_row = MetricRow("Score")
        layout.addWidget(self._focus_score_row)

        # Raw metrics — always shown so expert can verify
        self._focus_raw_lbl = QLabel("Lap var: —  |  Tenengrad: —  |  Brenner: —")
        self._focus_raw_lbl.setStyleSheet("color:#555577; font-size:9px; font-family:Consolas;")
        self._focus_raw_lbl.setWordWrap(True)
        layout.addWidget(self._focus_raw_lbl)

        # Scoring mode + reference info
        self._focus_mode_lbl = QLabel("Mode: RELATIVE (no reference)")
        self._focus_mode_lbl.setStyleSheet("color:#555577; font-size:9px;")
        self._focus_mode_lbl.setWordWrap(True)
        layout.addWidget(self._focus_mode_lbl)

        self._layout.addWidget(box)

    def update_focus(self, result, current_filename: str = ""):
        color_map = {"PERFECT": "#00e676", "GOOD": "#b2ff59",
                     "SOFT": "#ffab40", "BLURRY": "#ff5252"}
        conf_color = {"HIGH": "#00e676", "MEDIUM": "#ffab40", "LOW": "#888899"}

        color = color_map.get(result.verdict, "white")
        self._focus_verdict.setText(result.verdict)
        self._focus_verdict.setStyleSheet(f"color: {color}; font-weight: bold;")

        conf  = getattr(result, "confidence",   "LOW")
        mode  = getattr(result, "scoring_mode", "RELATIVE")
        r_pct = getattr(result, "ref_pct",      0.0)
        r_src = getattr(result, "ref_source",   "")
        r_lap = getattr(result, "raw_lap",      0.0)
        r_ten = getattr(result, "raw_ten",      0.0)
        r_bre = getattr(result, "raw_brenner",  0.0)

        cc = conf_color.get(conf, "#888899")
        self._focus_confidence.setText(f"● {conf}")
        self._focus_confidence.setStyleSheet(f"color:{cc}; font-size:8px; font-weight:bold;")

        pct = min(result.score / 10, 100)
        self._focus_score_row.set_value(f"{result.score:.0f}", pct, color)

        # Raw numbers — expert can always verify
        self._focus_raw_lbl.setText(
            f"Lap: {r_lap:,.0f}  |  Ten: {r_ten:,.0f}  |  Bren: {r_bre:,.1f}"
        )

        # Detect self-comparison: current image IS the reference — must match by filename,
        # NOT by score (same image in different format would also score ~100%)
        is_self_ref = (mode != "RELATIVE" and r_src and current_filename
                       and current_filename == r_src)

        # Mode label — explicit about what scoring was used
        if mode == "RELATIVE":
            self._focus_mode_lbl.setText(
                "⚠ RELATIVE MODE — normalized to own best cell\n"
                "   Cannot confirm absolute sharpness. Use reference."
            )
            self._focus_mode_lbl.setStyleSheet("color:#886633; font-size:9px;")
        elif is_self_ref:
            self._focus_mode_lbl.setText(
                f"⚠ THIS IMAGE IS THE REFERENCE ({r_src})\n"
                "   Comparing to itself — 100% is not a real score.\n"
                "   Trust the absolute Score and Verdict above."
            )
            self._focus_mode_lbl.setStyleSheet("color:#CC8800; font-size:9px;")
        elif mode == "AUTO_REF":
            self._focus_mode_lbl.setText(
                f"AUTO-REF ({r_src})\n"
                f"   This image: {r_pct:.1f}% of session best"
            )
            self._focus_mode_lbl.setStyleSheet("color:#6699AA; font-size:9px;")
        else:  # LOCKED_REF
            self._focus_mode_lbl.setText(
                f"✓ LOCKED REF ({r_src})\n"
                f"   This image: {r_pct:.1f}% of reference"
            )
            self._focus_mode_lbl.setStyleSheet("color:#00AA66; font-size:9px;")

    # ── Quality ────────────────────────────────────────────────────────────

    def _build_quality_group(self):
        box = QGroupBox("Image Quality")
        layout = QVBoxLayout(box)
        layout.setSpacing(2)

        self._quality_verdict = QLabel("—")
        self._quality_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._quality_verdict.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(self._quality_verdict)

        self._quality_score_row = MetricRow("Overall")
        self._exposure_row      = MetricRow("Exposure")
        self._contrast_row      = MetricRow("Contrast")
        self._noise_row         = MetricRow("Noise")
        self._snr_row           = MetricRow("SNR")

        for row in [self._quality_score_row, self._exposure_row,
                    self._contrast_row, self._noise_row, self._snr_row]:
            layout.addWidget(row)

        self._layout.addWidget(box)

    def update_quality(self, result):
        color = "#00e676" if result.verdict == "PASS" else "#ff5252"
        self._quality_verdict.setText(result.verdict)
        self._quality_verdict.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._quality_score_row.set_value(f"{result.overall_score:.0f}/100", result.overall_score, color)

        exp_ok = result.exposure_ok
        exp_color = "#00e676" if exp_ok else "#ff5252"
        exp_text = "OK" if exp_ok else f"Over:{result.overexposed_pct:.1f}% Under:{result.underexposed_pct:.1f}%"
        self._exposure_row.set_value(exp_text, 100 if exp_ok else 30, exp_color)

        self._contrast_row.set_value(f"{result.rms_contrast:.1f}%", min(result.rms_contrast * 2, 100))
        noise_pct = max(0, 100 - result.noise_level)
        noise_color = "#00e676" if result.noise_level < 10 else "#ffab40"
        self._noise_row.set_value(f"{result.noise_level:.1f}", noise_pct, noise_color)
        snr_pct = min(max((result.snr_db - 0) / 60 * 100, 0), 100)
        self._snr_row.set_value(f"{result.snr_db:.1f} dB", snr_pct)

    # ── Histogram ──────────────────────────────────────────────────────────

    def _build_histogram_group(self):
        self._hist_group = QGroupBox("Histogram")
        layout = QVBoxLayout(self._hist_group)

        self._hist_widget = pg.PlotWidget()
        self._hist_widget.setBackground("#1a1a1a")
        self._hist_widget.setFixedHeight(100)
        self._hist_widget.getAxis("left").hide()
        self._hist_widget.getAxis("bottom").setStyle(tickFont=None)
        self._hist_widget.setMouseEnabled(False, False)
        layout.addWidget(self._hist_widget)

        self._layout.addWidget(self._hist_group)
        self._hist_visible = True

    def update_histogram(self, hist_data: dict):
        self._hist_widget.clear()
        x = np.arange(256)
        colors = {"red": (255, 50, 50), "green": (50, 200, 50), "blue": (50, 100, 255),
                  "gray": (180, 180, 180), "luma": (200, 200, 200)}
        for ch, data in hist_data.items():
            color = colors.get(ch, (180, 180, 180))
            pen = pg.mkPen(color=color, width=1)
            self._hist_widget.plot(x, data, pen=pen, fillLevel=0, brush=(*color, 30))

    def toggle_histogram(self):
        self._hist_visible = not self._hist_visible
        self._hist_group.setVisible(self._hist_visible)

    # ── Pixel inspector ────────────────────────────────────────────────────

    def _build_pixel_group(self):
        box = QGroupBox("Pixel Inspector")
        layout = QGridLayout(box)

        self._px_coord  = QLabel("X: — Y: —")
        self._px_r      = QLabel("R: —")
        self._px_g      = QLabel("G: —")
        self._px_b      = QLabel("B: —")
        self._px_gray   = QLabel("Gray: —")
        self._px_swatch = QLabel()
        self._px_swatch.setFixedSize(24, 24)
        self._px_swatch.setStyleSheet("background: #333; border: 1px solid #555;")

        layout.addWidget(self._px_coord,  0, 0, 1, 2)
        layout.addWidget(self._px_swatch, 1, 0)
        layout.addWidget(self._px_r,      1, 1)
        layout.addWidget(self._px_g,      2, 1)
        layout.addWidget(self._px_b,      3, 1)
        layout.addWidget(self._px_gray,   2, 0)

        self._layout.addWidget(box)

    def update_pixel(self, x: int, y: int, pixel):
        self._px_coord.setText(f"X:{x}  Y:{y}")
        if hasattr(pixel, "__len__") and len(pixel) >= 3:
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            self._px_r.setText(f"R: {r}")
            self._px_g.setText(f"G: {g}")
            self._px_b.setText(f"B: {b}")
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            self._px_gray.setText(f"Gray: {gray}")
            sr, sg, sb = self._display_rgb(r, g, b)
            self._px_swatch.setStyleSheet(f"background: rgb({sr},{sg},{sb}); border: 1px solid #555;")
        else:
            v = int(pixel) if not hasattr(pixel, "__len__") else int(pixel[0])
            self._px_gray.setText(f"Val: {v}")
            self._px_r.setText("")
            self._px_g.setText("")
            self._px_b.setText("")
            sv = self._display_channel(v)
            self._px_swatch.setStyleSheet(f"background: rgb({sv},{sv},{sv}); border: 1px solid #555;")

    @staticmethod
    def _display_channel(value: int) -> int:
        if value > 255:
            value = value >> 8
        return max(0, min(int(value), 255))

    @classmethod
    def _display_rgb(cls, r: int, g: int, b: int) -> tuple[int, int, int]:
        return cls._display_channel(r), cls._display_channel(g), cls._display_channel(b)

    # ── ROI Analysis ───────────────────────────────────────────────────────

    def _build_roi_group(self):
        self._roi_group = QGroupBox("ROI Analysis")
        layout = QVBoxLayout(self._roi_group)
        layout.setSpacing(3)

        self._roi_size_lbl = QLabel("Draw a rectangle on the image")
        self._roi_size_lbl.setStyleSheet("color:#555566; font-size:10px; font-style:italic;")
        layout.addWidget(self._roi_size_lbl)

        # Stats table: header + R/G/B/Gray rows
        tbl = QWidget()
        g = QGridLayout(tbl)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(2)
        for col, hdr in enumerate(["", "Mean", "Std", "Min", "Max"]):
            lbl = QLabel(hdr)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#00B4D8; font-size:9px; font-weight:bold;")
            g.addWidget(lbl, 0, col)

        self._roi_cells: dict[str, list[QLabel]] = {}
        for row, (ch, color) in enumerate([("R", "#FF5252"), ("G", "#00E676"),
                                            ("B", "#448AFF"), ("Gray", "#CCCCDD")], 1):
            name_lbl = QLabel(ch)
            name_lbl.setStyleSheet(f"color:{color}; font-size:9px; font-weight:bold;")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            g.addWidget(name_lbl, row, 0)
            cells = []
            for col in range(1, 5):
                lbl = QLabel("—")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet("font-size:9px;")
                g.addWidget(lbl, row, col)
                cells.append(lbl)
            self._roi_cells[ch] = cells

        layout.addWidget(tbl)
        self._roi_group.setVisible(False)
        self._layout.addWidget(self._roi_group)

    def update_roi_stats(self, image: np.ndarray, ix1: int, iy1: int,
                          ix2: int, iy2: int):
        region = image[iy1:iy2, ix1:ix2]
        if region.size == 0:
            return

        w_px = ix2 - ix1;  h_px = iy2 - iy1
        self._roi_size_lbl.setText(f"Region: {w_px}×{h_px} px  ({w_px*h_px:,} px²)")
        self._roi_size_lbl.setStyleSheet("color:#CCCCDD; font-size:10px;")

        def _stats(arr):
            a = arr.astype(np.float32)
            return f"{a.mean():.1f}", f"{a.std():.1f}", f"{a.min():.0f}", f"{a.max():.0f}"

        if region.ndim == 3 and region.shape[2] >= 3:
            for ch, idx in [("R", 0), ("G", 1), ("B", 2)]:
                vals = _stats(region[:, :, idx])
                for lbl, v in zip(self._roi_cells[ch], vals):
                    lbl.setText(v)
            gray = (0.299 * region[:, :, 0] + 0.587 * region[:, :, 1] +
                    0.114 * region[:, :, 2])
            for lbl, v in zip(self._roi_cells["Gray"], _stats(gray)):
                lbl.setText(v)
        else:
            arr = region if region.ndim == 2 else region[:, :, 0]
            vals = _stats(arr)
            for ch in ["R", "G", "B", "Gray"]:
                for lbl, v in zip(self._roi_cells[ch], vals):
                    lbl.setText(v)

        self._roi_group.setVisible(True)

    # ── Line Profile ───────────────────────────────────────────────────────

    def _build_profile_group(self):
        self._profile_group = QGroupBox("Intensity Line Profile")
        layout = QVBoxLayout(self._profile_group)
        layout.setSpacing(3)

        self._profile_length_lbl = QLabel("Draw a line on the image")
        self._profile_length_lbl.setStyleSheet("color:#555566; font-size:10px; font-style:italic;")
        layout.addWidget(self._profile_length_lbl)

        self._profile_plot = pg.PlotWidget()
        self._profile_plot.setBackground("#0D0D1A")
        self._profile_plot.setFixedHeight(110)
        self._profile_plot.getAxis("left").setStyle(tickFont=QFont("Consolas", 7))
        self._profile_plot.getAxis("bottom").setStyle(tickFont=QFont("Consolas", 7))
        self._profile_plot.showGrid(x=True, y=True, alpha=0.15)
        self._profile_plot.setMouseEnabled(False, False)
        self._profile_plot.setLabel("bottom", "px along line", color="#888899", size="8pt")
        self._profile_plot.setLabel("left", "intensity", color="#888899", size="8pt")
        layout.addWidget(self._profile_plot)

        self._profile_stats_lbl = QLabel("")
        self._profile_stats_lbl.setStyleSheet("color:#888899; font-size:9px;")
        layout.addWidget(self._profile_stats_lbl)

        self._profile_group.setVisible(False)
        self._layout.addWidget(self._profile_group)

    def update_line_profile(self, image: np.ndarray, ix1: int, iy1: int,
                             ix2: int, iy2: int):
        length = int(math.sqrt((ix2 - ix1)**2 + (iy2 - iy1)**2))
        n = max(2, min(512, length))
        xs = np.linspace(ix1, ix2, n).astype(int)
        ys = np.linspace(iy1, iy2, n).astype(int)
        H, W = image.shape[:2]
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)

        self._profile_length_lbl.setText(
            f"Length: {length} px  |  ({ix1},{iy1}) → ({ix2},{iy2})")
        self._profile_length_lbl.setStyleSheet("color:#CCCCDD; font-size:10px;")

        self._profile_plot.clear()
        px_axis = np.arange(n)

        if image.ndim == 3 and image.shape[2] >= 3:
            for ch_idx, (color, name) in enumerate(
                    [(( 255,  80,  80), "R"), ((80, 220, 80), "G"), ((80, 120, 255), "B")]):
                vals = image[ys, xs, ch_idx].astype(np.float32)
                pen = pg.mkPen(color=color, width=1.2)
                self._profile_plot.plot(px_axis, vals, pen=pen, name=name)
            gray = (0.299 * image[ys, xs, 0] + 0.587 * image[ys, xs, 1] +
                    0.114 * image[ys, xs, 2])
            stats_ch = gray
        else:
            arr = image[ys, xs] if image.ndim == 2 else image[ys, xs, 0]
            vals = arr.astype(np.float32)
            pen = pg.mkPen(color=(200, 200, 200), width=1.5)
            self._profile_plot.plot(px_axis, vals, pen=pen)
            stats_ch = vals

        mn = stats_ch.min();  mx = stats_ch.max();  avg = stats_ch.mean()
        self._profile_stats_lbl.setText(
            f"Min: {mn:.0f}  Max: {mx:.0f}  Mean: {avg:.1f}  ΔRange: {mx-mn:.0f}")

        self._profile_group.setVisible(True)

    # ── Annotations ────────────────────────────────────────────────────────

    _ANN_COLORS = {
        "Scratch": "#FF1744", "Pit": "#FF6D00", "Contamination": "#FFD600",
        "Burr": "#E040FB", "Crack": "#FF5252", "OK": "#00E676", "Other": "#00B0FF",
    }

    def _build_annotation_group(self):
        self._ann_group = QGroupBox("Annotations")
        layout = QVBoxLayout(self._ann_group)
        layout.setSpacing(4)

        self._ann_hint = QLabel("Click image to place markers")
        self._ann_hint.setStyleSheet("color:#555566; font-size:10px; font-style:italic;")
        layout.addWidget(self._ann_hint)

        self._ann_table = QTableWidget(0, 4)
        self._ann_table.setHorizontalHeaderLabels(["#", "Label", "X", "Y"])
        self._ann_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._ann_table.horizontalHeader().setDefaultSectionSize(44)
        self._ann_table.setColumnWidth(0, 28)
        self._ann_table.verticalHeader().setVisible(False)
        self._ann_table.setFixedHeight(110)
        self._ann_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._ann_table.setStyleSheet(
            "QTableWidget { font-size:9px; background:#0D0D1A; "
            "gridline-color:#1A1A2A; border:1px solid #1A1A2A; }"
            "QHeaderView::section { background:#141428; color:#00B4D8; "
            "border:none; padding:2px; font-size:9px; }"
        )
        layout.addWidget(self._ann_table)

        btn_row = QHBoxLayout()
        self._ann_clear_btn = QPushButton("Clear All")
        self._ann_clear_btn.setFixedHeight(22)
        self._ann_clear_btn.setStyleSheet(
            "QPushButton { background:#1A1A2A; color:#FF5252; font-size:9px; "
            "border:1px solid #2A2A3A; padding:2px 8px; }"
            "QPushButton:hover { background:#2A1A1A; }"
        )
        self._ann_count_lbl = QLabel("0 annotations")
        self._ann_count_lbl.setStyleSheet("color:#555566; font-size:9px;")
        btn_row.addWidget(self._ann_count_lbl)
        btn_row.addStretch()
        btn_row.addWidget(self._ann_clear_btn)
        layout.addLayout(btn_row)

        self._ann_group.setVisible(False)
        self._layout.addWidget(self._ann_group)

    def show_annotation_tools(self, visible: bool):
        pass  # group visibility controlled entirely by refresh_annotations

    def refresh_annotations(self, annotations: list):
        """Re-populate annotation table from viewer's annotation list."""
        self._ann_table.setRowCount(0)
        for i, ann in enumerate(annotations):
            row = self._ann_table.rowCount()
            self._ann_table.insertRow(row)
            label = ann.get("label", "Other")
            color = QColor(self._ANN_COLORS.get(label, "#00B0FF"))

            items = [
                QTableWidgetItem(str(i + 1)),
                QTableWidgetItem(label),
                QTableWidgetItem(str(ann["ix"])),
                QTableWidgetItem(str(ann["iy"])),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 1:
                    item.setForeground(color)
                self._ann_table.setItem(row, col, item)

        n = len(annotations)
        self._ann_count_lbl.setText(f"{n} annotation{'s' if n != 1 else ''}")
        self._ann_hint.setVisible(False)
        self._ann_table.setVisible(n > 0)
        self._ann_group.setVisible(n > 0)

    # ── Measurement ────────────────────────────────────────────────────────

    def _build_measure_group(self):
        self._measure_group = QGroupBox("Measurement")
        layout = QVBoxLayout(self._measure_group)
        layout.setSpacing(4)

        self._meas_hint = QLabel("Drag on the image to measure distance")
        self._meas_hint.setStyleSheet("color:#555566; font-size:10px; font-style:italic;")
        layout.addWidget(self._meas_hint)

        grid = QGridLayout()
        grid.setSpacing(3)

        def _row_lbl(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color:#888899; font-size:10px;")
            return lbl

        def _val_lbl():
            lbl = QLabel("—")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            lbl.setStyleSheet("color:#CCCCDD; font-size:10px; font-weight:bold;")
            return lbl

        self._meas_dist_px  = _val_lbl()
        self._meas_dist_mm  = _val_lbl()
        self._meas_dx       = _val_lbl()
        self._meas_dy       = _val_lbl()
        self._meas_angle    = _val_lbl()

        grid.addWidget(_row_lbl("Distance"),  0, 0)
        grid.addWidget(self._meas_dist_px,    0, 1)
        grid.addWidget(_row_lbl("(mm)"),      1, 0)
        grid.addWidget(self._meas_dist_mm,    1, 1)
        grid.addWidget(_row_lbl("ΔX"),        2, 0)
        grid.addWidget(self._meas_dx,         2, 1)
        grid.addWidget(_row_lbl("ΔY"),        3, 0)
        grid.addWidget(self._meas_dy,         3, 1)
        grid.addWidget(_row_lbl("Angle"),     4, 0)
        grid.addWidget(self._meas_angle,      4, 1)
        layout.addLayout(grid)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#1A1A2A;")
        layout.addWidget(sep)

        self._calib_lbl = QLabel("Scale: not calibrated")
        self._calib_lbl.setStyleSheet("color:#555566; font-size:9px;")
        layout.addWidget(self._calib_lbl)

        self._measure_group.setVisible(False)
        self._layout.addWidget(self._measure_group)

    def update_measurement(self, ix1: int, iy1: int, ix2: int, iy2: int,
                            mm_per_px: float = 0.0):
        dx = ix2 - ix1;  dy = iy2 - iy1
        dist_px = math.sqrt(dx * dx + dy * dy)
        angle   = math.degrees(math.atan2(abs(dy), abs(dx)))

        self._meas_dist_px.setText(f"{dist_px:.2f} px")
        self._meas_dx.setText(f"{abs(dx)} px")
        self._meas_dy.setText(f"{abs(dy)} px")
        self._meas_angle.setText(f"{angle:.2f}°")

        if mm_per_px > 0:
            self._meas_dist_mm.setText(f"{dist_px * mm_per_px:.4f} mm")
            self._calib_lbl.setText(f"Scale: {mm_per_px:.6f} mm/px")
            self._calib_lbl.setStyleSheet("color:#00B4D8; font-size:9px;")
        else:
            self._meas_dist_mm.setText("— (not calibrated)")
            self._calib_lbl.setText("Scale: not calibrated")
            self._calib_lbl.setStyleSheet("color:#555566; font-size:9px;")

        self._meas_hint.setVisible(False)
        self._measure_group.setVisible(True)

    def scroll_to(self, widget: QWidget):
        """Scroll the inspector so that widget is visible."""
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(50, lambda: self.ensureWidgetVisible(widget, 0, 20))

    def set_calibration_label(self, mm_per_px: float):
        if mm_per_px > 0:
            self._calib_lbl.setText(f"Scale: {mm_per_px:.6f} mm/px")
            self._calib_lbl.setStyleSheet("color:#00B4D8; font-size:9px;")
        else:
            self._calib_lbl.setText("Scale: not calibrated")
            self._calib_lbl.setStyleSheet("color:#555566; font-size:9px;")
