"""Right-side inspector panel — focus metrics, quality scores, histogram, pixel values."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QGridLayout,
    QProgressBar, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt
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
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(220)
        self.setMaximumWidth(280)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setSpacing(4)
        self.setWidget(container)

        self._build_focus_group()
        self._build_quality_group()
        self._build_histogram_group()
        self._build_pixel_group()

        self._layout.addStretch()

    # ------------------------------------------------------------------ #
    #  Focus group
    # ------------------------------------------------------------------ #

    def _build_focus_group(self):
        box = QGroupBox("Focus & Sharpness")
        layout = QVBoxLayout(box)
        layout.setSpacing(2)

        self._focus_verdict = QLabel("—")
        self._focus_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._focus_verdict.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(self._focus_verdict)

        self._focus_score_row = MetricRow("Score")
        layout.addWidget(self._focus_score_row)

        self._focus_metric_label = QLabel("Metric: —")
        self._focus_metric_label.setStyleSheet("color: gray; font-size: 9px;")
        layout.addWidget(self._focus_metric_label)

        self._layout.addWidget(box)

    def update_focus(self, result):
        color_map = {"PERFECT": "#00e676", "GOOD": "#b2ff59", "SOFT": "#ffab40", "BLURRY": "#ff5252"}
        color = color_map.get(result.verdict, "white")
        self._focus_verdict.setText(result.verdict)
        self._focus_verdict.setStyleSheet(f"color: {color}; font-weight: bold;")
        pct = min(result.score / 10, 100)
        self._focus_score_row.set_value(f"{result.score:.0f}", pct, color)
        self._focus_metric_label.setText(f"Metric: {result.metric}")

    # ------------------------------------------------------------------ #
    #  Quality group
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Histogram
    # ------------------------------------------------------------------ #

    def _build_histogram_group(self):
        box = QGroupBox("Histogram")
        layout = QVBoxLayout(box)

        self._hist_widget = pg.PlotWidget()
        self._hist_widget.setBackground("#1a1a1a")
        self._hist_widget.setFixedHeight(100)
        self._hist_widget.getAxis("left").hide()
        self._hist_widget.getAxis("bottom").setStyle(tickFont=None)
        self._hist_widget.setMouseEnabled(False, False)
        layout.addWidget(self._hist_widget)

        self._layout.addWidget(box)
        self._hist_visible = True

    def update_histogram(self, hist_data: dict):
        self._hist_widget.clear()
        x = np.arange(256)
        colors = {"red": (255, 50, 50), "green": (50, 200, 50), "blue": (50, 100, 255), "gray": (180, 180, 180), "luma": (200, 200, 200)}
        for ch, data in hist_data.items():
            color = colors.get(ch, (180, 180, 180))
            pen = pg.mkPen(color=color, width=1)
            self._hist_widget.plot(x, data, pen=pen, fillLevel=0, brush=(*color, 30))

    def toggle_histogram(self):
        self._hist_visible = not self._hist_visible

    # ------------------------------------------------------------------ #
    #  Pixel inspector
    # ------------------------------------------------------------------ #

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
            self._px_swatch.setStyleSheet(f"background: rgb({r},{g},{b}); border: 1px solid #555;")
        else:
            v = int(pixel) if not hasattr(pixel, "__len__") else int(pixel[0])
            self._px_gray.setText(f"Val: {v}")
            self._px_r.setText("")
            self._px_g.setText("")
            self._px_b.setText("")
            self._px_swatch.setStyleSheet(f"background: rgb({v},{v},{v}); border: 1px solid #555;")
