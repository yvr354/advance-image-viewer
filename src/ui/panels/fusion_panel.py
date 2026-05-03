"""
Illumination Fusion panel.

Modes
─────
  RGB Composite      — assign images to R / G / B channels with weights
  Photometric Stereo — Woodham least-squares normals; output selector
  RTI Relight        — biquadratic PTM fit then interactive relighting
  Max / Min / Avg    — simple pixel-wise statistics
"""

import os
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QFileDialog, QDoubleSpinBox,
    QButtonGroup, QSlider, QStackedWidget, QSizePolicy, QFrame,
    QListWidget, QListWidgetItem, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer

from src.fusion.illumination_fusion import IlluminationFusion
from src.core.image_loader import load_image


# ── Palette ────────────────────────────────────────────────────────────────────

_DIM   = "#556677"
_TEXT  = "#AABBCC"
_CYAN  = "#00B4D8"
_AMBER = "#D4840A"

_BTN_MODE_ON = (
    "QPushButton { background:#003040; color:#00E5FF; border:1px solid #00B4D8;"
    "  border-radius:3px; padding:3px 8px; font-size:10px; font-weight:700; }"
)
_BTN_MODE_OFF = (
    "QPushButton { background:#0A1420; color:#556677; border:1px solid #1A2A3A;"
    "  border-radius:3px; padding:3px 8px; font-size:10px; }"
    "QPushButton:hover { background:#111F2E; color:#AABBCC; border-color:#2A3A4A; }"
)
_BTN_OUT_ON = (
    "QPushButton { background:#0A2800; color:#2ECC71; border:1px solid #2ECC71;"
    "  border-radius:3px; padding:2px 8px; font-size:10px; font-weight:700; }"
)
_BTN_OUT_OFF = (
    "QPushButton { background:#0A1420; color:#445566; border:1px solid #1A2A3A;"
    "  border-radius:3px; padding:2px 8px; font-size:10px; }"
    "QPushButton:hover { background:#111F2E; color:#AABBCC; }"
)
_BTN_COMPOSE = (
    "QPushButton { background:#1565C0; color:#FFFFFF; border:none; border-radius:4px;"
    "  padding:5px 16px; font-size:11px; font-weight:700; }"
    "QPushButton:hover { background:#1976D2; }"
    "QPushButton:disabled { background:#1A2A3A; color:#445566; }"
)
_BTN_FIT = (
    "QPushButton { background:#2A0A3A; color:#CC88FF; border:1px solid #5A3A7A;"
    "  border-radius:4px; padding:4px 12px; font-size:10px; font-weight:700; }"
    "QPushButton:hover { background:#3A1A4A; }"
    "QPushButton:disabled { background:#1A2A3A; color:#445566; }"
)
_SPIN = (
    "QDoubleSpinBox { background:#0E1820; color:#CCDDEE; border:1px solid #2A3A4A;"
    "  border-radius:3px; padding:1px 3px; font-size:10px; }"
)


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet("background:#1A2A3A;")
    return f


# ── Background worker ──────────────────────────────────────────────────────────

class _FusionWorker(QThread):
    done   = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as e:
            self.failed.emit(str(e))


# ── Image row ──────────────────────────────────────────────────────────────────

class FusionRow(QWidget):
    changed          = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, index: int, path: str, fusion: IlluminationFusion, parent=None):
        super().__init__(parent)
        self.index  = index
        self.fusion = fusion
        self._build(path)

    def _build(self, path: str):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(6)

        # Filename — stretch to take available space
        name = QLabel(os.path.basename(path))
        name.setMinimumWidth(80)
        name.setStyleSheet(f"color:{_TEXT}; font-size:11px; font-weight:600;")
        name.setToolTip(path)
        lay.addWidget(name, stretch=1)

        # ── RGB controls ─────────────────────────────────────────────────
        self._rgb_w = QWidget()
        rgb_lay = QHBoxLayout(self._rgb_w)
        rgb_lay.setContentsMargins(0, 0, 0, 0)
        rgb_lay.setSpacing(6)

        from PyQt6.QtWidgets import QCheckBox
        self._cb_r = QCheckBox("R"); self._cb_g = QCheckBox("G"); self._cb_b = QCheckBox("B")
        for cb, col, tip in [
            (self._cb_r, "#FF5555", "Assign this image to the Red channel of the composite."),
            (self._cb_g, "#55EE55", "Assign this image to the Green channel of the composite."),
            (self._cb_b, "#5599FF", "Assign this image to the Blue channel of the composite."),
        ]:
            cb.setStyleSheet(f"color:{col}; font-size:11px; font-weight:700;")
            cb.setToolTip(tip)
            cb.stateChanged.connect(self._on_rgb)
            rgb_lay.addWidget(cb)

        self._w_spin = QDoubleSpinBox()
        self._w_spin.setRange(0.0, 2.0); self._w_spin.setSingleStep(0.1)
        self._w_spin.setValue(1.0); self._w_spin.setFixedWidth(72)
        self._w_spin.setStyleSheet(_SPIN); self._w_spin.setPrefix("w: ")
        self._w_spin.setDecimals(2)
        self._w_spin.setToolTip(
            "Channel weight (0–2).\n"
            "1.0 = normal contribution.\n"
            "Increase to make this light angle dominate the composite."
        )
        self._w_spin.valueChanged.connect(self._on_rgb)
        rgb_lay.addWidget(self._w_spin)
        lay.addWidget(self._rgb_w)

        # ── Angle controls ───────────────────────────────────────────────
        self._ang_w = QWidget()
        ang_lay = QHBoxLayout(self._ang_w)
        ang_lay.setContentsMargins(0, 0, 0, 0)
        ang_lay.setSpacing(4)

        for txt in ["Az:", "El:"]:
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color:{_DIM}; font-size:11px;")
            ang_lay.addWidget(lbl)
            sp  = QDoubleSpinBox()
            sp.setRange(0, 360 if txt == "Az:" else 90)
            sp.setSingleStep(5); sp.setValue(0 if txt == "Az:" else 45)
            sp.setFixedWidth(68); sp.setStyleSheet(_SPIN); sp.setSuffix("°")
            if txt == "Az:":
                sp.setToolTip(
                    "Azimuth — horizontal angle of the light source (0–360°).\n"
                    "0° = right, 90° = top, 180° = left, 270° = bottom.\n"
                    "Distribute lights evenly: 0°, 90°, 180°, 270° for 4-light setups."
                )
            else:
                sp.setToolTip(
                    "Elevation — vertical angle of the light above the surface (0–90°).\n"
                    "0° = grazing (side-lit, reveals texture).\n"
                    "90° = overhead (flat, minimal shadow).\n"
                    "45° is the recommended default for defect inspection."
                )
            sp.valueChanged.connect(self._on_angles)
            ang_lay.addWidget(sp)
            if txt == "Az:": self._az_sp = sp
            else:            self._el_sp = sp

        self._ang_w.setVisible(False)
        lay.addWidget(self._ang_w)

        # Delete
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(22, 22)
        del_btn.setToolTip("Remove this image from the fusion set.")
        del_btn.setStyleSheet(
            "QPushButton{background:#1A0808;color:#AA4444;border:none;border-radius:3px;font-size:10px;}"
            "QPushButton:hover{background:#2A1010;color:#FF6666;}"
        )
        del_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        lay.addWidget(del_btn)

        self.setStyleSheet("background:#111A24; border-radius:4px;")
        self.setFixedHeight(34)

    def set_mode(self, mode: str):
        angle_modes = ("Photometric Stereo", "RTI Relight")
        self._rgb_w.setVisible(mode == "RGB Composite")
        self._ang_w.setVisible(mode in angle_modes)

    def _on_rgb(self):
        self.fusion.set_assignment(
            self.index,
            r=self._cb_r.isChecked(), g=self._cb_g.isChecked(), b=self._cb_b.isChecked(),
            weight=self._w_spin.value(),
        )
        self.changed.emit()

    def _on_angles(self):
        self.fusion.set_angles(self.index, self._az_sp.value(), self._el_sp.value())
        self.changed.emit()


# ── Main fusion panel ──────────────────────────────────────────────────────────

class FusionPanel(QWidget):
    composite_ready = pyqtSignal(np.ndarray)

    _MODES = ["RGB Composite", "Photometric Stereo", "RTI Relight", "Max / Min / Avg"]

    def __init__(self):
        super().__init__()
        self.fusion = IlluminationFusion()
        self._rows:  list[FusionRow]   = []
        self._worker: _FusionWorker | None = None
        self._rti_fitted = False
        self._build()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Header
        hdr = QLabel("  ⚡  Illumination Fusion")
        hdr.setStyleSheet(f"color:{_CYAN}; font-size:10px; font-weight:700; letter-spacing:1px;")
        root.addWidget(hdr)
        root.addWidget(_sep())

        # Mode selector
        root.addWidget(self._make_section_label("MODE"))
        self._mode_btns: dict[str, QPushButton] = {}
        self._mode_group = QButtonGroup(self)
        _MODE_TIPS = {
            "RGB Composite": (
                "Assign each image to a colour channel (R, G, B) with a weight.\n"
                "Defects hidden under one lighting angle become colour-visible in the merged image.\n"
                "Example: front-light → R, left-light → G, right-light → B."
            ),
            "Photometric Stereo": (
                "Woodham (1980) algorithm — industry gold standard used in Halcon and Cognex.\n"
                "Computes a surface normal for every pixel from ≥ 3 images.\n"
                "Outputs: Gradient Magnitude (best defect signal), Normal Map, Albedo, Height Map.\n"
                "Set the azimuth and elevation for each light source in the image list."
            ),
            "RTI Relight": (
                "Reflectance Transformation Imaging — fits a 6-coefficient polynomial per pixel.\n"
                "After fitting, drag the Azimuth and Elevation sliders to relight the surface\n"
                "in real time and find the angle that best reveals a subtle defect.\n"
                "Requires ≥ 6 images. Fit is slow once; relighting is instant."
            ),
            "Max / Min / Avg": (
                "Simple pixel-wise statistics across all loaded images.\n"
                "Max — brightest pixel wins: emphasises bright defects and high-reflectance areas.\n"
                "Min — darkest pixel wins: reveals pits, shadows, and low-reflectance defects.\n"
                "Average — balanced blend: reduces noise, good for general exposure improvement."
            ),
        }
        row1 = QHBoxLayout(); row1.setSpacing(3); row1.setContentsMargins(0,0,0,0)
        row2 = QHBoxLayout(); row2.setSpacing(3); row2.setContentsMargins(0,0,0,0)
        for i, m in enumerate(self._MODES):
            btn = QPushButton(m)
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setToolTip(_MODE_TIPS[m])
            self._mode_group.addButton(btn)
            self._mode_btns[m] = btn
            btn.clicked.connect(lambda _, mode=m: self._on_mode(mode))
            (row1 if i < 2 else row2).addWidget(btn)
        self._mode_btns[self._MODES[0]].setChecked(True)
        self._refresh_mode_styles(self._MODES[0])
        root.addLayout(row1)
        root.addLayout(row2)
        root.addWidget(_sep())

        # ── From current folder ──────────────────────────────────────────
        self._folder_section = QWidget()
        fs_lay = QVBoxLayout(self._folder_section)
        fs_lay.setContentsMargins(0, 0, 0, 0)
        fs_lay.setSpacing(3)

        folder_hdr = QHBoxLayout()
        folder_lbl = self._make_section_label("FROM CURRENT FOLDER")
        folder_lbl.setToolTip(
            "Images already open in the left browser.\n"
            "Double-click any filename to add it to the fusion set."
        )
        self._folder_count = QLabel("")
        self._folder_count.setStyleSheet(f"color:{_DIM}; font-size:9px;")
        folder_hdr.addWidget(folder_lbl)
        folder_hdr.addStretch(1)
        folder_hdr.addWidget(self._folder_count)
        fs_lay.addLayout(folder_hdr)

        self._folder_list = QListWidget()
        self._folder_list.setFixedHeight(100)
        self._folder_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._folder_list.setStyleSheet(
            "QListWidget { background:#0A1218; border:1px solid #1A2A3A; border-radius:3px;"
            f"  color:{_TEXT}; font-size:10px; outline:none; }}"
            "QListWidget::item { padding:2px 6px; }"
            "QListWidget::item:selected { background:#003040; color:#00E5FF; }"
            "QListWidget::item:hover { background:#111F2E; }"
        )
        self._folder_list.setToolTip(
            "Double-click a filename to add it to the fusion set.\n"
            "Select multiple with Ctrl/Shift, then click Add Selected."
        )
        self._folder_list.itemDoubleClicked.connect(self._add_from_folder_item)
        fs_lay.addWidget(self._folder_list)

        add_sel_btn = QPushButton("＋ Add Selected to Fusion")
        add_sel_btn.setFixedHeight(22)
        add_sel_btn.setStyleSheet(
            f"QPushButton{{background:#0A1A10;color:#2ECC71;border:1px solid #2ECC71;"
            f"border-radius:3px;padding:2px 8px;font-size:10px;font-weight:600;}}"
            f"QPushButton:hover{{background:#0A2A18;}}"
        )
        add_sel_btn.setToolTip(
            "Add all selected images from the folder list to the fusion set.\n"
            "Use Ctrl+Click or Shift+Click to select multiple files."
        )
        add_sel_btn.clicked.connect(self._add_selected_from_folder)
        fs_lay.addWidget(add_sel_btn)

        self._folder_section.setVisible(False)   # hidden until a folder is loaded
        root.addWidget(self._folder_section)
        root.addWidget(_sep())

        # Image list
        img_bar = QHBoxLayout()
        img_bar.addWidget(self._make_section_label("FUSION SET"))
        img_bar.addStretch(1)
        for label, slot, style, tip in [
            ("＋ Add",  self._add_images,
             f"QPushButton{{background:#0E2030;color:{_CYAN};border:1px solid {_CYAN};"
             "border-radius:3px;padding:2px 8px;font-size:10px;font-weight:600;}}"
             "QPushButton:hover{background:#0A3040;}",
             "Load one or more images captured of the same scene\nunder different lighting angles."),
            ("Clear", self._clear,
             "QPushButton{background:#200A0A;color:#DD4444;border:1px solid #552222;"
             "border-radius:3px;padding:2px 8px;font-size:10px;}"
             "QPushButton:hover{background:#300A0A;}",
             "Remove all images and reset the fusion."),
        ]:
            b = QPushButton(label); b.setFixedHeight(22); b.setStyleSheet(style)
            b.setToolTip(tip); b.clicked.connect(slot)
            img_bar.addWidget(b)
        root.addLayout(img_bar)

        # Column header
        self._col_hdr = QLabel("Filename                     R    G    B    Weight")
        self._col_hdr.setStyleSheet(f"color:{_DIM}; font-size:9px; padding-left:6px;")
        root.addWidget(self._col_hdr)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self._sw = QWidget(); self._sw.setStyleSheet("background:transparent;")
        self._sl = QVBoxLayout(self._sw)
        self._sl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._sl.setSpacing(3); self._sl.setContentsMargins(0,0,0,0)
        self._scroll.setWidget(self._sw)
        self._scroll.setMinimumHeight(50)
        root.addWidget(self._scroll, stretch=1)

        self._hint = QLabel(
            "Load images captured under different\n"
            "lighting angles to reveal surface defects."
        )
        self._hint.setStyleSheet(f"color:{_DIM}; font-size:9px; padding:4px;")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)
        root.addWidget(_sep())

        # Mode-specific controls
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_rgb_panel())    # 0
        self._stack.addWidget(self._build_ps_panel())     # 1
        self._stack.addWidget(self._build_rti_panel())    # 2
        self._stack.addWidget(self._build_stat_panel())   # 3
        root.addWidget(self._stack)

        root.addWidget(_sep())

        # Compose
        self._compose_btn = QPushButton("▶  Compose")
        self._compose_btn.setStyleSheet(_BTN_COMPOSE)
        self._compose_btn.setToolTip(
            "Run the selected fusion algorithm and display the result in the main viewer.\n"
            "The result can then be processed further with the Filter Pipeline."
        )
        self._compose_btn.clicked.connect(self._compose)
        root.addWidget(self._compose_btn)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{_AMBER}; font-size:9px;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._status)

    # ── Sub-panels ────────────────────────────────────────────────────────────

    def _build_rgb_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        tip = QLabel(
            "Assign each image to R, G, or B.\n"
            "Defects hidden in one light angle become colour-visible in the composite."
        )
        tip.setStyleSheet(f"color:{_DIM}; font-size:9px;"); tip.setWordWrap(True)
        lay.addWidget(tip)
        return w

    def _build_ps_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(5)
        tip = QLabel("Set azimuth + elevation per light. Needs ≥ 3 images.")
        tip.setStyleSheet(f"color:{_DIM}; font-size:9px;"); tip.setWordWrap(True)
        lay.addWidget(tip)

        lay.addWidget(self._make_section_label("OUTPUT"))
        _PS_TIPS = {
            "Gradient Magnitude": (
                "Best defect signal for most inspections.\n"
                "Shows how sharply the surface slope changes at each pixel.\n"
                "Scratches, pits, and edges appear as bright bands.\n"
                "Feed this into the Filter Pipeline → Canny or False Color for further analysis."
            ),
            "Normal Map": (
                "Surface orientation encoded as RGB colour (Halcon standard).\n"
                "Red = X slope, Green = Y slope, Blue = surface facing camera.\n"
                "Useful for visualising surface topology and checking light calibration."
            ),
            "Albedo": (
                "Surface reflectivity with lighting removed.\n"
                "Shows true material colour and texture, independent of light angle.\n"
                "Use to detect stains, discolouration, or coating defects."
            ),
            "Height Map": (
                "3-D depth reconstruction via Frankot-Chellappa integration.\n"
                "Bright = raised, Dark = recessed.\n"
                "Use for dents, bumps, warps, and embossed/engraved features.\n"
                "Accuracy depends on having well-distributed light angles."
            ),
        }
        self._ps_outputs = ["Gradient Magnitude", "Normal Map", "Albedo", "Height Map"]
        self._ps_sel = self._ps_outputs[0]
        self._ps_group = QButtonGroup(self)
        self._ps_btns: dict[str, QPushButton] = {}
        r1 = QHBoxLayout(); r1.setSpacing(3); r1.setContentsMargins(0,0,0,0)
        r2 = QHBoxLayout(); r2.setSpacing(3); r2.setContentsMargins(0,0,0,0)
        for i, lbl in enumerate(self._ps_outputs):
            btn = QPushButton(lbl); btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setToolTip(_PS_TIPS[lbl])
            self._ps_group.addButton(btn); self._ps_btns[lbl] = btn
            btn.clicked.connect(lambda _, l=lbl: self._set_ps_out(l))
            (r1 if i < 2 else r2).addWidget(btn)
        self._ps_btns[self._ps_outputs[0]].setChecked(True)
        self._refresh_ps_styles(self._ps_outputs[0])
        lay.addLayout(r1); lay.addLayout(r2)

        tip2 = QLabel(
            "Gradient Magnitude — best defect signal (scratches, pits)\n"
            "Normal Map — surface orientation as RGB colour\n"
            "Albedo — material reflectivity, no lighting\n"
            "Height Map — reconstructed 3-D depth"
        )
        tip2.setStyleSheet(f"color:{_DIM}; font-size:9px;"); tip2.setWordWrap(True)
        lay.addWidget(tip2)
        return w

    def _build_rti_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(5)
        tip = QLabel("Set azimuth + elevation per image. Needs ≥ 6 images.\nFit once — then drag sliders to relight in real time.")
        tip.setStyleSheet(f"color:{_DIM}; font-size:9px;"); tip.setWordWrap(True)
        lay.addWidget(tip)

        self._rti_fit_btn = QPushButton("⚙  Fit RTI Polynomials")
        self._rti_fit_btn.setStyleSheet(_BTN_FIT)
        self._rti_fit_btn.setToolTip(
            "Fit a 6-coefficient biquadratic polynomial to every pixel across all images.\n"
            "This is a one-time computation — may take 5–30 seconds for large images.\n"
            "After fitting, the Azimuth and Elevation sliders relight the surface instantly.\n"
            "Drag the sliders to find the light angle that best reveals a hidden defect."
        )
        self._rti_fit_btn.clicked.connect(self._fit_rti)
        lay.addWidget(self._rti_fit_btn)

        self._rti_sliders = QWidget(); self._rti_sliders.setVisible(False)
        sl_lay = QVBoxLayout(self._rti_sliders); sl_lay.setContentsMargins(0,0,0,0); sl_lay.setSpacing(4)

        _RTI_TIPS = {
            "_rti_az": (
                "Azimuth — horizontal light direction (0–360°).\n"
                "Drag left/right to rotate the virtual light around the surface.\n"
                "Scratches and grooves are most visible when the light is perpendicular to them."
            ),
            "_rti_el": (
                "Elevation — vertical light angle above the surface (0–90°).\n"
                "Low values (5–20°) = grazing light — reveals fine texture and scratches.\n"
                "High values (60–90°) = overhead light — reduces shadows, shows colour/albedo."
            ),
        }
        for attr, label, mn, mx, default in [
            ("_rti_az", "Azimuth",   0, 360, 0),
            ("_rti_el", "Elevation", 0,  90, 45),
        ]:
            row = QHBoxLayout(); row.setSpacing(4)
            lbl = QLabel(label); lbl.setFixedWidth(58)
            lbl.setStyleSheet(f"color:{_DIM}; font-size:10px;")
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(mn, mx); sld.setValue(default)
            sld.setToolTip(_RTI_TIPS[attr])
            val_lbl = QLabel(f"{default:3d}°")
            val_lbl.setFixedWidth(32)
            val_lbl.setStyleSheet(f"color:{_CYAN}; font-size:10px;")
            sld.valueChanged.connect(lambda v, vl=val_lbl, a=attr: self._rti_slider_changed(v, vl, a))
            setattr(self, attr + "_sld", sld)
            setattr(self, attr + "_val", val_lbl)
            row.addWidget(lbl); row.addWidget(sld, stretch=1); row.addWidget(val_lbl)
            sl_lay.addLayout(row)

        lay.addWidget(self._rti_sliders)

        self._rti_debounce = QTimer(self)
        self._rti_debounce.setSingleShot(True)
        self._rti_debounce.setInterval(80)
        self._rti_debounce.timeout.connect(self._rti_relight_now)
        return w

    def _build_stat_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(5)
        lay.addWidget(self._make_section_label("OPERATION"))
        self._stat_sel = "Max"
        self._stat_group = QButtonGroup(self)
        self._stat_btns: dict[str, QPushButton] = {}
        row = QHBoxLayout(); row.setSpacing(3); row.setContentsMargins(0,0,0,0)
        _STAT_TIPS = {
            "Max":     "Each output pixel = brightest value across all images.\nBest for revealing bright defects, protrusions, and high-reflectance spots.",
            "Min":     "Each output pixel = darkest value across all images.\nBest for revealing pits, voids, and shadows that appear in at least one light angle.",
            "Average": "Each output pixel = mean of all images.\nReduces noise and balances exposure. Good general-purpose starting point.",
        }
        for op in ["Max", "Min", "Average"]:
            btn = QPushButton(op); btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setToolTip(_STAT_TIPS[op])
            self._stat_group.addButton(btn); self._stat_btns[op] = btn
            btn.clicked.connect(lambda _, o=op: self._set_stat(o))
            row.addWidget(btn)
        self._stat_btns["Max"].setChecked(True)
        self._refresh_stat_styles("Max")
        lay.addLayout(row)
        tip = QLabel(
            "Max = brightest pixel (bright defects).\n"
            "Min = darkest pixel (pits, shadows).\n"
            "Average = balanced exposure blend."
        )
        tip.setStyleSheet(f"color:{_DIM}; font-size:9px;"); tip.setWordWrap(True)
        lay.addWidget(tip)
        return w

    # ── Style helpers ─────────────────────────────────────────────────────────

    def _make_section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{_DIM}; font-size:9px; font-weight:700; letter-spacing:1px;")
        return lbl

    def _refresh_mode_styles(self, active: str):
        for m, btn in self._mode_btns.items():
            btn.setStyleSheet(_BTN_MODE_ON if m == active else _BTN_MODE_OFF)

    def _refresh_ps_styles(self, active: str):
        for lbl, btn in self._ps_btns.items():
            btn.setStyleSheet(_BTN_OUT_ON if lbl == active else _BTN_OUT_OFF)

    def _refresh_stat_styles(self, active: str):
        for op, btn in self._stat_btns.items():
            btn.setStyleSheet(_BTN_OUT_ON if op == active else _BTN_OUT_OFF)

    # ── Mode / output switching ───────────────────────────────────────────────

    def _on_mode(self, mode: str):
        self._stack.setCurrentIndex(self._MODES.index(mode))
        self._refresh_mode_styles(mode)
        col = ("Filename        R  G  B  Weight" if mode == "RGB Composite"
               else "Filename        Azimuth    Elevation" if mode in ("Photometric Stereo", "RTI Relight")
               else "Filename")
        self._col_hdr.setText(col)
        for row in self._rows:
            row.set_mode(mode)
        if mode != "RTI Relight":
            self._rti_fitted = False

    def _set_ps_out(self, lbl: str):
        self._ps_sel = lbl; self._refresh_ps_styles(lbl)

    def _set_stat(self, op: str):
        self._stat_sel = op; self._refresh_stat_styles(op)

    # ── Image management ──────────────────────────────────────────────────────

    def _current_mode(self) -> str:
        for m, btn in self._mode_btns.items():
            if btn.isChecked(): return m
        return self._MODES[0]

    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Light Images", "",
            "Images (*.tiff *.tif *.png *.bmp *.jpg *.pgm)"
        )
        mode = self._current_mode()
        for path in paths:
            self._add_single(path, mode)

    def _add_single(self, path: str, mode: str):
        try:
            data = load_image(path)
            idx  = self.fusion.add_image(data.raw, path)
            row  = FusionRow(idx, path, self.fusion)
            row.set_mode(mode)
            row.changed.connect(lambda: setattr(self, "_rti_fitted", False))
            row.remove_requested.connect(self._remove_row)
            self._rows.append(row)
            self._sl.addWidget(row)
            self._hint.setVisible(False)
            self._rti_fitted = False
        except Exception as e:
            self._status.setText(f"Load failed: {e}")

    def _remove_row(self, row: FusionRow):
        idx = self._rows.index(row)
        self.fusion.remove_image(idx)
        self._rows.pop(idx)
        self._sl.removeWidget(row)
        row.deleteLater()
        for i, r in enumerate(self._rows): r.index = i
        self._hint.setVisible(len(self._rows) == 0)
        self._rti_fitted = False

    def _clear(self):
        for row in self._rows:
            self._sl.removeWidget(row); row.deleteLater()
        self._rows.clear(); self.fusion.clear()
        self._hint.setVisible(True); self._rti_fitted = False
        self._status.setText("")

    # ── Compose ───────────────────────────────────────────────────────────────

    def _compose(self):
        if not self.fusion.entries:
            self._status.setText("No images loaded."); return
        if self._worker and self._worker.isRunning(): return

        mode = self._current_mode()
        _ps_key = {
            "Gradient Magnitude": "gradient",
            "Normal Map":         "normal_map",
            "Albedo":             "albedo",
            "Height Map":         "height",
        }
        if   mode == "RGB Composite":     fn = self.fusion.rgb_composite
        elif mode == "Photometric Stereo":
            out = _ps_key.get(self._ps_sel, "gradient")
            fn  = lambda: self.fusion.photometric_stereo(out)
        elif mode == "RTI Relight":
            az = self._rti_az_sld.value(); el = self._rti_el_sld.value()
            fn = lambda: self.fusion.rti_relight(az, el)
        else:
            ops = {"Max": self.fusion.max_fusion, "Min": self.fusion.min_fusion,
                   "Average": self.fusion.average_fusion}
            fn  = ops.get(self._stat_sel, self.fusion.max_fusion)

        self._status.setText("⏳ Processing…")
        self._compose_btn.setEnabled(False)
        self._worker = _FusionWorker(fn)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result):
        self._compose_btn.setEnabled(True)
        if result is None:
            self._status.setText("Not enough images / channels not assigned."); return
        self._status.setText("")
        if isinstance(result, bool):   # rti_fit returns bool
            return
        out = result
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        self.composite_ready.emit(out)

    def _on_failed(self, error: str):
        self._compose_btn.setEnabled(True)
        self._status.setText(f"Error: {error}")

    # ── RTI ───────────────────────────────────────────────────────────────────

    def _fit_rti(self):
        if len(self.fusion.entries) < 6:
            self._status.setText("RTI needs ≥ 6 images."); return
        self._status.setText("⏳ Fitting polynomials…")
        self._rti_fit_btn.setEnabled(False)
        self._worker = _FusionWorker(self.fusion.rti_fit)
        self._worker.done.connect(self._on_rti_fitted)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_rti_fitted(self, ok):
        self._rti_fit_btn.setEnabled(True)
        if ok:
            self._rti_fitted = True
            self._rti_sliders.setVisible(True)
            self._status.setText("Fitted — drag sliders to relight.")
        else:
            self._status.setText("Fit failed — need ≥ 6 images.")

    def _rti_slider_changed(self, value: int, val_lbl: QLabel, attr: str):
        val_lbl.setText(f"{value:3d}°")
        if self._rti_fitted:
            self._rti_debounce.start()

    def _rti_relight_now(self):
        az = self._rti_az_sld.value()
        el = self._rti_el_sld.value()
        result = self.fusion.rti_relight(az, el)
        if result is not None:
            out = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB) if result.ndim == 2 else result
            self.composite_ready.emit(out)

    # ── Folder image picker ───────────────────────────────────────────────────

    def set_folder_images(self, paths: list[str]):
        """Called by main_window whenever the browser folder changes."""
        self._folder_paths: list[str] = list(paths)
        self._folder_list.clear()
        for path in paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._folder_list.addItem(item)
        count = len(paths)
        self._folder_count.setText(f"{count} image{'s' if count != 1 else ''}")
        self._folder_section.setVisible(count > 0)

    def _add_from_folder_item(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        self._add_single(path, self._current_mode())

    def _add_selected_from_folder(self):
        mode = self._current_mode()
        for item in self._folder_list.selectedItems():
            path = item.data(Qt.ItemDataRole.UserRole)
            self._add_single(path, mode)
