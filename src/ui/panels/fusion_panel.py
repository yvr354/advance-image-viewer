"""Multi-illumination fusion panel."""

import os
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QGroupBox, QCheckBox, QFileDialog, QSlider,
    QComboBox, QDoubleSpinBox, QGridLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal

from src.fusion.illumination_fusion import IlluminationFusion
from src.core.image_loader import load_image


class FusionInputRow(QWidget):
    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, index: int, path: str, fusion: IlluminationFusion, parent=None):
        super().__init__(parent)
        self.index = index
        self.fusion = fusion
        self._build(path)

    def _build(self, path: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._label = QLabel(os.path.basename(path))
        self._label.setFixedWidth(120)
        self._label.setToolTip(path)

        self._cb_r = QCheckBox("R")
        self._cb_g = QCheckBox("G")
        self._cb_b = QCheckBox("B")

        self._weight = QDoubleSpinBox()
        self._weight.setRange(0.0, 2.0)
        self._weight.setSingleStep(0.1)
        self._weight.setValue(1.0)
        self._weight.setFixedWidth(55)
        self._weight.setPrefix("w:")

        btn_del = QPushButton("✕")
        btn_del.setFixedSize(20, 20)
        btn_del.setStyleSheet("color: #ff5252;")
        btn_del.clicked.connect(lambda: self.remove_requested.emit(self))

        for w in [self._cb_r, self._cb_g, self._cb_b, self._weight]:
            if hasattr(w, "stateChanged"):
                w.stateChanged.connect(self._on_changed)
            elif hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._on_changed)

        layout.addWidget(self._label)
        layout.addWidget(self._cb_r)
        layout.addWidget(self._cb_g)
        layout.addWidget(self._cb_b)
        layout.addWidget(self._weight)
        layout.addWidget(btn_del)

    def _on_changed(self):
        self.fusion.set_assignment(
            self.index,
            r=self._cb_r.isChecked(),
            g=self._cb_g.isChecked(),
            b=self._cb_b.isChecked(),
            weight=self._weight.value(),
        )
        self.changed.emit()


class FusionPanel(QWidget):
    composite_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.fusion = IlluminationFusion()
        self._rows: list[FusionInputRow] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        btn_add = QPushButton("+ Add Image")
        btn_add.clicked.connect(self._add_images)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear)
        btn_clear.setStyleSheet("color: #ff5252;")

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["RGB Composite", "Average", "Max", "Min"])
        self._mode_combo.currentTextChanged.connect(self._compose)

        btn_compose = QPushButton("Compose →")
        btn_compose.clicked.connect(self._compose)
        btn_compose.setStyleSheet("background: #1565C0; color: white; font-weight: bold;")

        toolbar.addWidget(btn_add)
        toolbar.addWidget(btn_clear)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("Mode:"))
        toolbar.addWidget(self._mode_combo)
        toolbar.addWidget(btn_compose)
        layout.addLayout(toolbar)

        # Header
        header = QHBoxLayout()
        for text, w in [("Image", 120), ("R", 25), ("G", 25), ("B", 25), ("Weight", 55)]:
            lbl = QLabel(text)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet("color: gray; font-size: 9px;")
            header.addWidget(lbl)
        header.addStretch()
        layout.addLayout(header)

        # Rows scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_widget)
        self._scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll_layout.setSpacing(2)
        self._scroll.setWidget(self._scroll_widget)
        self._scroll.setMaximumHeight(200)
        layout.addWidget(self._scroll)

        self._hint = QLabel("Load multiple images of the same scene with different lighting.\nAssign each to R / G / B channels to reveal defects.")
        self._hint.setStyleSheet("color: gray; font-size: 9px;")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Illumination Images", "",
            "Images (*.tiff *.tif *.png *.bmp *.jpg *.pgm)"
        )
        for path in paths:
            self._add_single(path)

    def _add_single(self, path: str):
        try:
            data = load_image(path)
            idx = self.fusion.add_image(data.raw, path)
            row = FusionInputRow(idx, path, self.fusion)
            row.changed.connect(self._compose)
            row.remove_requested.connect(self._remove_row)
            self._rows.append(row)
            self._scroll_layout.addWidget(row)
        except Exception as e:
            pass

    def _remove_row(self, row: FusionInputRow):
        idx = self._rows.index(row)
        self.fusion.remove_image(idx)
        self._rows.pop(idx)
        self._scroll_layout.removeWidget(row)
        row.deleteLater()
        # Re-index remaining rows
        for i, r in enumerate(self._rows):
            r.index = i

    def _clear(self):
        for row in self._rows:
            self._scroll_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self.fusion.clear()

    def _compose(self):
        mode = self._mode_combo.currentText()
        result = None
        if mode == "RGB Composite":
            result = self.fusion.compose()
        elif mode == "Average":
            result = self.fusion.average_fusion()
        elif mode == "Max":
            result = self.fusion.max_fusion()
        elif mode == "Min":
            result = self.fusion.min_fusion()

        if result is not None:
            self.composite_ready.emit(result)
