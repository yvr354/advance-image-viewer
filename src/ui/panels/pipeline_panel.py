"""Processing pipeline panel — expert filter organization with workflow steps and presets."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLabel,
    QMenu, QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox, QFrame,
    QSizePolicy, QAbstractScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.pipeline.pipeline import Pipeline
from src.pipeline.filter_registry import (
    FILTER_CATEGORIES, WORKFLOW_STEPS, STEP_COLORS, PRESETS,
)
from src.filters.base_filter import BaseFilter, FilterParam


# ── Styles ───────────────────────────────────────────────────────────────────

_STEP_LABEL_STYLE = {
    "① Pre-process": "color:#5BA8D8; font-size:9px; font-weight:700; letter-spacing:1px;",
    "② Enhance":     "color:#D4A84A; font-size:9px; font-weight:700; letter-spacing:1px;",
    "③ Detect":      "color:#4ABD7A; font-size:9px; font-weight:700; letter-spacing:1px;",
    "④ Visualize":   "color:#C07AE0; font-size:9px; font-weight:700; letter-spacing:1px;",
}

_BTN_PRESET = (
    "QPushButton {"
    "  background: #141E2A; color: #8899AA;"
    "  border: 1px solid #1E2E3E; border-radius: 3px;"
    "  padding: 3px 6px; font-size: 10px; text-align: left;"
    "}"
    "QPushButton:hover { background: #1E2E3E; color: #BBCCDD; border-color: #3A5A7A; }"
    "QPushButton:pressed { background: #0A1620; }"
)

_BTN_ADD = (
    "QPushButton {"
    "  background: #0E2030; color: #00B4D8;"
    "  border: 1px solid #00B4D8; border-radius: 3px;"
    "  padding: 4px 12px; font-size: 11px; font-weight: 600;"
    "}"
    "QPushButton:hover { background: #0A3040; }"
)

_BTN_CLEAR = (
    "QPushButton {"
    "  background: #200A0A; color: #DD4444;"
    "  border: 1px solid #552222; border-radius: 3px;"
    "  padding: 4px 10px; font-size: 11px;"
    "}"
    "QPushButton:hover { background: #300A0A; border-color: #884444; }"
)

_SEP_LINE = "background: #1E2E3E;"


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(_SEP_LINE)
    return line


# ── Filter layer card ─────────────────────────────────────────────────────────

class FilterLayerWidget(QWidget):
    changed           = pyqtSignal()
    remove_requested  = pyqtSignal(object)
    move_up_requested = pyqtSignal(object)
    move_down_requested = pyqtSignal(object)

    def __init__(self, filter_instance: BaseFilter, parent=None):
        super().__init__(parent)
        self.filter = filter_instance
        self._build()

    def _build(self):
        step   = self.filter.CATEGORY
        color  = STEP_COLORS.get(step, "#334455")
        desc   = getattr(self.filter, "DESCRIPTION", "")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(3)

        # ── Header ──────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(4)

        # Step badge (tiny colored label — ①②③④)
        badge = QLabel(step[0])    # just the circled number
        badge.setFixedSize(14, 14)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"background:{color}; color:#FFFFFF; border-radius:7px;"
            f"font-size:8px; font-weight:700;"
        )
        badge.setToolTip(step)
        header.addWidget(badge)

        # Filter name checkbox
        self._enable_cb = QCheckBox(self.filter.NAME)
        self._enable_cb.setChecked(self.filter.enabled)
        self._enable_cb.stateChanged.connect(self._on_enable_changed)
        if desc:
            self._enable_cb.setToolTip(desc)
        font = self._enable_cb.font()
        font.setPointSize(9)
        font.setBold(True)
        self._enable_cb.setFont(font)
        header.addWidget(self._enable_cb, stretch=1)

        # Up / Down / Delete buttons
        for symbol, sig in [("▲", self.move_up_requested),
                             ("▼", self.move_down_requested)]:
            b = QPushButton(symbol)
            b.setFixedSize(18, 18)
            b.setStyleSheet(
                "QPushButton { background:#1A2A3A; color:#6688AA; border:none; border-radius:3px; font-size:9px; }"
                "QPushButton:hover { background:#2A3A4A; color:#AACCEE; }"
            )
            b.clicked.connect(lambda _, s=sig: s.emit(self))
            header.addWidget(b)

        btn_del = QPushButton("✕")
        btn_del.setFixedSize(18, 18)
        btn_del.setStyleSheet(
            "QPushButton { background:#2A1A1A; color:#AA4444; border:none; border-radius:3px; font-size:9px; }"
            "QPushButton:hover { background:#3A1A1A; color:#FF6666; }"
        )
        btn_del.setToolTip("Remove this filter")
        btn_del.clicked.connect(lambda: self.remove_requested.emit(self))
        header.addWidget(btn_del)

        layout.addLayout(header)

        # ── Parameters ──────────────────────────────────────────────
        for name, param in self.filter.params.items():
            row = QHBoxLayout()
            row.setSpacing(6)

            lbl = QLabel(param.label)
            lbl.setFixedWidth(96)
            lbl.setStyleSheet("color:#8899AA; font-size:10px;")
            if param.description:
                lbl.setToolTip(param.description)
            row.addWidget(lbl)

            w = self._build_param_widget(name, param)
            row.addWidget(w, stretch=1)
            layout.addLayout(row)

        # Card border — colored left side shows which step this filter belongs to
        self.setStyleSheet(
            f"FilterLayerWidget {{"
            f"  background:#1A2030; border-radius:4px;"
            f"  border-left:3px solid {color};"
            f"}}"
        )

    def _build_param_widget(self, name: str, param: FilterParam) -> QWidget:
        tip = param.description

        _spin_style = (
            "QSpinBox, QDoubleSpinBox, QComboBox {"
            "  background:#0E1820; color:#CCDDEE; border:1px solid #2A3A4A;"
            "  border-radius:3px; padding:1px 4px; font-size:10px;"
            "}"
            "QSpinBox::up-button, QSpinBox::down-button,"
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {"
            "  width:14px;"
            "}"
        )

        if param.type == "bool":
            w = QCheckBox()
            w.setChecked(bool(param.value))
            w.stateChanged.connect(lambda v, n=name: self._set_and_emit(n, bool(v)))
        elif param.type == "choice":
            w = QComboBox()
            w.setStyleSheet(_spin_style)
            for c in param.choices:
                w.addItem(str(c))
            idx = w.findText(str(param.value))
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.currentTextChanged.connect(lambda v, n=name, p=param: self._set_and_emit(
                n, type(p.value)(v) if not isinstance(p.value, str) else v
            ))
        elif param.type == "int":
            w = QSpinBox()
            w.setStyleSheet(_spin_style)
            w.setRange(int(param.min_val or 0), int(param.max_val or 9999))
            w.setSingleStep(int(param.step or 1))
            w.setValue(int(param.value))
            w.valueChanged.connect(lambda v, n=name: self._set_and_emit(n, v))
        else:  # float
            w = QDoubleSpinBox()
            w.setStyleSheet(_spin_style)
            w.setRange(float(param.min_val or 0), float(param.max_val or 9999))
            w.setSingleStep(float(param.step or 0.1))
            w.setDecimals(2)
            w.setValue(float(param.value))
            w.valueChanged.connect(lambda v, n=name: self._set_and_emit(n, v))

        if tip:
            w.setToolTip(tip)
        return w

    def _set_and_emit(self, name: str, value):
        self.filter.set_param(name, value)
        self.changed.emit()

    def _on_enable_changed(self, state):
        self.filter.enabled = bool(state)
        self.changed.emit()


# ── Pipeline panel ────────────────────────────────────────────────────────────

class PipelinePanel(QWidget):
    pipeline_changed = pyqtSignal()

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._layer_widgets: list[FilterLayerWidget] = []
        self._build()

    # ── Construction ─────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        # ── Presets section ──────────────────────────────────────────
        presets_header = QLabel("  ⚡  Expert Presets")
        presets_header.setFixedHeight(22)
        presets_header.setStyleSheet(
            "background:#0A1420; color:#668899; font-size:10px; font-weight:700;"
            "padding:2px 0px; letter-spacing:1px;"
        )
        presets_header.setToolTip(
            "One-click starting points for common inspection tasks.\n"
            "Each preset adds a proven combination of filters in the correct order.\n"
            "Hover each button to see which filters are added and why."
        )
        root.addWidget(presets_header)

        # Two rows of 3 preset buttons
        preset_names = list(PRESETS.keys())
        for row_idx in range(0, len(preset_names), 3):
            row = QHBoxLayout()
            row.setContentsMargins(2, 2, 2, 2)
            row.setSpacing(3)
            for name in preset_names[row_idx:row_idx + 3]:
                info = PRESETS[name]
                btn = QPushButton(f"{info['icon']}  {name}")
                btn.setStyleSheet(_BTN_PRESET)
                btn.setToolTip(info["tip"])
                btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                btn.clicked.connect(lambda _, n=name: self._apply_preset(n))
                row.addWidget(btn)
            root.addLayout(row)

        root.addWidget(_separator())
        root.addSpacing(4)

        # ── Toolbar: Add Filter + Clear All ──────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(2, 0, 2, 0)
        toolbar.setSpacing(6)

        btn_add = QPushButton("＋  Add Filter")
        btn_add.setStyleSheet(_BTN_ADD)
        btn_add.setToolTip(
            "Browse all 28 filters organized by workflow step.\n"
            "① Pre-process → ② Enhance → ③ Detect → ④ Visualize"
        )
        btn_add.clicked.connect(self._show_add_menu)

        btn_clear = QPushButton("Clear All")
        btn_clear.setStyleSheet(_BTN_CLEAR)
        btn_clear.setToolTip("Remove all filters from the pipeline")
        btn_clear.clicked.connect(self._clear_all)

        toolbar.addWidget(btn_add, stretch=2)
        toolbar.addWidget(btn_clear, stretch=1)
        root.addLayout(toolbar)

        root.addSpacing(4)
        root.addWidget(_separator())
        root.addSpacing(4)

        # ── Empty state hint ─────────────────────────────────────────
        self._empty_label = QLabel(
            "No filters active.\n\n"
            "Use a Preset above, or click  ＋ Add Filter.\n\n"
            "Filters apply top → bottom."
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            "color:#334455; font-size:10px; padding:12px 8px;"
        )
        self._empty_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        root.addWidget(self._empty_label)

        # ── Scrollable pipeline list ─────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setVerticalScrollMode(QAbstractScrollArea.ScrollMode.ScrollPerPixel)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._scroll_widget = QWidget()
        self._scroll_widget.setStyleSheet("background: transparent;")
        self._scroll_layout = QVBoxLayout(self._scroll_widget)
        self._scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll_layout.setSpacing(6)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._scroll_widget)
        self._scroll.setVisible(False)
        root.addWidget(self._scroll, stretch=1)

    # ── Add Filter menu ───────────────────────────────────────────────────────

    def _show_add_menu(self):
        _menu_style = (
            "QMenu { background:#0E1820; color:#AABBCC; border:1px solid #2A3A4A; }"
            "QMenu::item { padding:5px 20px 5px 10px; }"
            "QMenu::item:selected { background:#1E3A50; color:#FFFFFF; }"
            "QMenu::item:disabled { color:#445566; }"
            "QMenu::separator { height:1px; background:#1E2E3E; margin:3px 0px; }"
        )
        menu = QMenu(self)
        menu.setStyleSheet(_menu_style)
        menu.setToolTipsVisible(True)

        for step_name, step_desc in WORKFLOW_STEPS:
            filters = FILTER_CATEGORIES.get(step_name, [])
            if not filters:
                continue

            sub = menu.addMenu(step_name)
            sub.setStyleSheet(_menu_style)
            sub.setToolTipsVisible(True)   # ← enables hover tooltips on actions

            # Step description at top of submenu (disabled, informational)
            info_action = sub.addAction(f"  {step_desc}")
            info_action.setEnabled(False)
            sub.addSeparator()

            for cls in filters:
                action = sub.addAction(cls.NAME)
                desc = getattr(cls, "DESCRIPTION", "")
                if desc:
                    action.setToolTip(desc)   # shown on hover (requires setToolTipsVisible)
                action.setStatusTip(desc)
                action.triggered.connect(lambda checked, c=cls: self._add_filter(c()))

        menu.exec(self.mapToGlobal(self.rect().topLeft()))

    # ── Preset application ────────────────────────────────────────────────────

    def _apply_preset(self, preset_name: str):
        info = PRESETS[preset_name]
        for fname in info["filters"]:
            self._quick_add(fname)

    def _quick_add(self, filter_name: str):
        for filters in FILTER_CATEGORIES.values():
            for cls in filters:
                if cls.NAME == filter_name:
                    self._add_filter(cls())
                    return

    # ── Pipeline management ───────────────────────────────────────────────────

    def _add_filter(self, filter_instance: BaseFilter):
        self.pipeline.add(filter_instance)
        self._add_layer_widget(filter_instance)
        self._update_empty_state()
        self.pipeline_changed.emit()

    def _add_layer_widget(self, filter_instance: BaseFilter):
        widget = FilterLayerWidget(filter_instance)
        widget.changed.connect(self.pipeline_changed.emit)
        widget.remove_requested.connect(self._remove_layer)
        widget.move_up_requested.connect(self._move_up)
        widget.move_down_requested.connect(self._move_down)
        self._layer_widgets.append(widget)
        self._scroll_layout.addWidget(widget)

    def _remove_layer(self, widget: FilterLayerWidget):
        idx = self._layer_widgets.index(widget)
        self.pipeline.remove(idx)
        self._layer_widgets.pop(idx)
        self._scroll_layout.removeWidget(widget)
        widget.deleteLater()
        self._update_empty_state()
        self.pipeline_changed.emit()

    def _move_up(self, widget: FilterLayerWidget):
        idx = self._layer_widgets.index(widget)
        if idx > 0:
            self.pipeline.move_up(idx)
            self._layer_widgets[idx - 1], self._layer_widgets[idx] = \
                self._layer_widgets[idx], self._layer_widgets[idx - 1]
            self._reorder_scroll()
            self.pipeline_changed.emit()

    def _move_down(self, widget: FilterLayerWidget):
        idx = self._layer_widgets.index(widget)
        if idx < len(self._layer_widgets) - 1:
            self.pipeline.move_down(idx)
            self._layer_widgets[idx + 1], self._layer_widgets[idx] = \
                self._layer_widgets[idx], self._layer_widgets[idx + 1]
            self._reorder_scroll()
            self.pipeline_changed.emit()

    def _reorder_scroll(self):
        for w in self._layer_widgets:
            self._scroll_layout.removeWidget(w)
        for w in self._layer_widgets:
            self._scroll_layout.addWidget(w)

    def _clear_all(self):
        for w in self._layer_widgets:
            self._scroll_layout.removeWidget(w)
            w.deleteLater()
        self._layer_widgets.clear()
        self.pipeline.clear()
        self._update_empty_state()
        self.pipeline_changed.emit()

    def _update_empty_state(self):
        has = len(self._layer_widgets) > 0
        self._empty_label.setVisible(not has)
        self._scroll.setVisible(has)

    def refresh(self):
        """Rebuild UI from current pipeline state (e.g. after load from file)."""
        for w in self._layer_widgets:
            self._scroll_layout.removeWidget(w)
            w.deleteLater()
        self._layer_widgets.clear()
        for layer in self.pipeline.layers:
            self._add_layer_widget(layer)
        self._update_empty_state()
