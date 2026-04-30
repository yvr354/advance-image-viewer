"""Processing pipeline panel — add, reorder, configure filter layers."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QCheckBox, QLabel, QMenu, QScrollArea,
    QSlider, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from src.pipeline.pipeline import Pipeline
from src.pipeline.filter_registry import FILTER_CATEGORIES
from src.filters.base_filter import BaseFilter, FilterParam


class FilterLayerWidget(QWidget):
    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)
    move_up_requested = pyqtSignal(object)
    move_down_requested = pyqtSignal(object)

    def __init__(self, filter_instance: BaseFilter, parent=None):
        super().__init__(parent)
        self.filter = filter_instance
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Header row
        header = QHBoxLayout()
        self._enable_cb = QCheckBox(self.filter.NAME)
        self._enable_cb.setChecked(self.filter.enabled)
        self._enable_cb.stateChanged.connect(self._on_enable_changed)

        btn_up   = QPushButton("▲")
        btn_down = QPushButton("▼")
        btn_del  = QPushButton("✕")
        for btn in [btn_up, btn_down, btn_del]:
            btn.setFixedSize(20, 20)
        btn_up.clicked.connect(lambda: self.move_up_requested.emit(self))
        btn_down.clicked.connect(lambda: self.move_down_requested.emit(self))
        btn_del.clicked.connect(lambda: self.remove_requested.emit(self))
        btn_del.setStyleSheet("color: #ff5252;")

        header.addWidget(self._enable_cb)
        header.addStretch()
        header.addWidget(btn_up)
        header.addWidget(btn_down)
        header.addWidget(btn_del)
        layout.addLayout(header)

        # Params
        for name, param in self.filter.params.items():
            row = QHBoxLayout()
            label = QLabel(param.label)
            label.setFixedWidth(100)
            row.addWidget(label)
            widget = self._build_param_widget(name, param)
            row.addWidget(widget)
            layout.addLayout(row)

        self.setStyleSheet("background: #2a2a2a; border-radius: 4px;")

    def _build_param_widget(self, name: str, param: FilterParam) -> QWidget:
        if param.type == "bool":
            w = QCheckBox()
            w.setChecked(bool(param.value))
            w.stateChanged.connect(lambda v, n=name: self._set_and_emit(n, bool(v)))
            return w
        elif param.type == "choice":
            w = QComboBox()
            for c in param.choices:
                w.addItem(str(c))
            current = str(param.value)
            idx = w.findText(current)
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.currentTextChanged.connect(lambda v, n=name, p=param: self._set_and_emit(
                n, type(p.value)(v) if not isinstance(p.value, str) else v
            ))
            return w
        elif param.type == "int":
            w = QSpinBox()
            w.setRange(int(param.min_val or 0), int(param.max_val or 9999))
            w.setSingleStep(int(param.step or 1))
            w.setValue(int(param.value))
            w.valueChanged.connect(lambda v, n=name: self._set_and_emit(n, v))
            return w
        else:  # float
            w = QDoubleSpinBox()
            w.setRange(float(param.min_val or 0), float(param.max_val or 9999))
            w.setSingleStep(float(param.step or 0.1))
            w.setDecimals(2)
            w.setValue(float(param.value))
            w.valueChanged.connect(lambda v, n=name: self._set_and_emit(n, v))
            return w

    def _set_and_emit(self, name: str, value):
        self.filter.set_param(name, value)
        self.changed.emit()

    def _on_enable_changed(self, state):
        self.filter.enabled = bool(state)
        self.changed.emit()


class PipelinePanel(QWidget):
    pipeline_changed = pyqtSignal()

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._layer_widgets: list[FilterLayerWidget] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        btn_add = QPushButton("+ Add Filter")
        btn_add.clicked.connect(self._show_add_menu)
        btn_add.setToolTip("Add a processing filter to the pipeline")
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear_all)
        btn_clear.setStyleSheet("color: #ff5252;")
        toolbar.addWidget(btn_add)
        toolbar.addWidget(btn_clear)
        toolbar.addStretch()

        # Quick-add buttons for most-used filters
        for label, filter_name in [
            ("Sharpen", "Unsharp Mask"),
            ("Denoise", "Bilateral Filter"),
            ("Contrast", "CLAHE"),
            ("Brightness", "Brightness / Contrast"),
            ("Edge Detect", "Canny Edge"),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(
                "QPushButton { background:#1a2a3a; color:#00B4D8; "
                "border:1px solid #00B4D8; border-radius:3px; padding:2px 6px; font-size:10px;}"
                "QPushButton:hover { background:#003344; }"
            )
            btn.setToolTip(f"Quick-add {label} filter")
            btn.clicked.connect(lambda _, n=filter_name: self._quick_add(n))
            toolbar.addWidget(btn)

        layout.addLayout(toolbar)

        # Empty-state hint shown when no filters are active
        self._empty_label = QLabel(
            "No filters active.\n"
            "Click  + Add Filter  or use the quick buttons above to start processing."
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: #444455; font-size: 11px; padding: 20px;"
        )
        layout.addWidget(self._empty_label)

        # Scrollable layer list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_widget)
        self._scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll_layout.setSpacing(4)
        self._scroll.setWidget(self._scroll_widget)
        self._scroll.setVisible(False)   # hidden until first filter added
        layout.addWidget(self._scroll)

    def _update_empty_state(self):
        has_filters = len(self._layer_widgets) > 0
        self._empty_label.setVisible(not has_filters)
        self._scroll.setVisible(has_filters)

    def _quick_add(self, filter_name: str):
        """Add a filter by name — used by quick-add toolbar buttons."""
        from src.pipeline.filter_registry import FILTER_CATEGORIES
        for filters in FILTER_CATEGORIES.values():
            for cls in filters:
                if cls.NAME == filter_name:
                    self._add_filter(cls())
                    return

    def _show_add_menu(self):
        menu = QMenu(self)
        for category, filters in FILTER_CATEGORIES.items():
            sub = menu.addMenu(category)
            for cls in filters:
                action = sub.addAction(cls.NAME)
                action.triggered.connect(lambda checked, c=cls: self._add_filter(c()))
        menu.exec(self.mapToGlobal(self.rect().topLeft()))

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

    def refresh(self):
        """Rebuild UI from current pipeline state (after load)."""
        for w in self._layer_widgets:
            self._scroll_layout.removeWidget(w)
            w.deleteLater()
        self._layer_widgets.clear()
        for layer in self.pipeline.layers:
            self._add_layer_widget(layer)
        self._update_empty_state()
