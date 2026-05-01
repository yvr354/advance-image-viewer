"""
MultiViewer — split-screen container for 1, 2, or 4 viewers.

Fixed 2×2 splitter structure is built ONCE and never changed.
Switching layouts only shows/hides cells — no reparenting, no GL context reset.

Layouts:
  1  — single (cell 0 only)
  2H — two side-by-side  (cells 0 | 1)
  2V — two stacked       (cells 0 / 2)
  4  — 2×2 grid          (all cells)

Usage:
  • Click a panel to make it active (blue border).
  • Then click an image in the File Browser to load it there.
  • Or drag-and-drop an image file onto any panel.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLabel,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.ui.panels.gl_viewer import GLImageViewer


def _zoom_ratio(zoom: float) -> str:
    """Return a human-readable ratio string: 1:1, 2:1, 1:2, etc."""
    if abs(zoom - 1.0) < 0.04:
        return "1:1"
    if zoom > 1.0:
        n = round(zoom * 4) / 4        # round to nearest 0.25
        return f"{n:.2g}:1"
    else:
        n = round((1.0 / zoom) * 4) / 4
        return f"1:{n:.2g}"


# ── Single viewer cell ─────────────────────────────────────────────────────

class ViewerCell(QWidget):
    """Header badge + GLImageViewer. Emits activated(idx) on any click."""

    activated = pyqtSignal(int)

    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self.idx     = idx
        self._active = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────
        self._header = QLabel(f"  Viewer {idx + 1}  —  click here, then pick image in browser")
        self._header.setFixedHeight(22)
        self._header.setFont(QFont("Segoe UI", 8))
        self._set_header_style(False)
        lay.addWidget(self._header)

        # ── GL viewer ─────────────────────────────────────────────
        self.viewer = GLImageViewer()
        self.viewer.setAcceptDrops(True)
        lay.addWidget(self.viewer, stretch=1)

        # Forward any mouse press to activate this cell
        orig = self.viewer.mousePressEvent
        def _fwd(ev, _orig=orig):
            self.activated.emit(self.idx)
            _orig(ev)
        self.viewer.mousePressEvent = _fwd

        self._update_border(False)

    # ── Public API ────────────────────────────────────────────────

    def set_active(self, active: bool):
        self._active = active
        self._set_header_style(active)
        self._update_border(active)

    def set_header_text(self, filename: str, badge: str = "", zoom: float = 0.0):
        name = (filename[:24] + "…") if len(filename) > 26 else filename
        text = f"  [{self.idx+1}]  {name}"
        if zoom > 0:
            pct   = f"{zoom*100:.0f}%"
            ratio = _zoom_ratio(zoom)
            text += f"   {pct}  ({ratio})"
        if badge:
            text += f"   {badge}"
        self._header.setText(text)
        self._last_filename = filename
        self._last_badge    = badge

    def update_zoom(self, zoom: float):
        filename = getattr(self, "_last_filename", f"Viewer {self.idx+1}")
        badge    = getattr(self, "_last_badge",    "")
        self.set_header_text(filename, badge, zoom)

    # ── Internal ─────────────────────────────────────────────────

    def _set_header_style(self, active: bool):
        if active:
            self._header.setStyleSheet(
                "background:#0E2438; color:#00AAFF; "
                "padding-left:6px; font-weight:700;"
            )
        else:
            self._header.setStyleSheet(
                "background:#0C1520; color:#445566; padding-left:6px;"
            )

    def _update_border(self, active: bool):
        color = "#00AAFF" if active else "#1A2535"
        self.setStyleSheet(f"QWidget#ViewerCell {{ border: 2px solid {color}; }}")
        self.setObjectName("ViewerCell")


# ── Multi-viewer container ─────────────────────────────────────────────────

class MultiViewer(QWidget):
    """
    Always-alive 2×2 grid. Layout changes are show/hide only — no reparenting.
    """

    active_changed = pyqtSignal(int)   # emitted when active cell changes

    LAYOUTS = ("1", "2H", "2V", "4")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_idx   = 0
        self._layout_key   = "1"

        # ── Build fixed 2×2 splitter tree (built once, never rebuilt) ──
        self._cells: list[ViewerCell] = [ViewerCell(i) for i in range(4)]
        for cell in self._cells:
            cell.activated.connect(self._activate)

        self._h_top    = QSplitter(Qt.Orientation.Horizontal)
        self._h_top.addWidget(self._cells[0])
        self._h_top.addWidget(self._cells[1])
        self._h_top.setChildrenCollapsible(False)

        self._h_bot    = QSplitter(Qt.Orientation.Horizontal)
        self._h_bot.addWidget(self._cells[2])
        self._h_bot.addWidget(self._cells[3])
        self._h_bot.setChildrenCollapsible(False)

        self._v_main   = QSplitter(Qt.Orientation.Vertical)
        self._v_main.addWidget(self._h_top)
        self._v_main.addWidget(self._h_bot)
        self._v_main.setChildrenCollapsible(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._v_main)

        # Apply initial 1-view layout
        self.set_layout("1")

    # ── Public API ────────────────────────────────────────────────

    @property
    def active_viewer(self) -> GLImageViewer:
        return self._cells[self._active_idx].viewer

    @property
    def active_idx(self) -> int:
        return self._active_idx

    def viewer_at(self, idx: int) -> GLImageViewer:
        return self._cells[idx].viewer

    def n_visible(self) -> int:
        return {"1": 1, "2H": 2, "2V": 2, "4": 4}[self._layout_key]

    def set_layout(self, key: str):
        if key not in self.LAYOUTS:
            return
        self._layout_key = key

        show1 = key in ("2H", "4")       # cell 1 visible?
        show2 = key in ("2V", "4")       # cells 2,3 visible?
        show3 = key == "4"               # cell 3 visible?

        self._cells[1].setVisible(show1)
        self._h_bot.setVisible(show2)
        self._cells[3].setVisible(show3)

        # Equal split sizes
        W = self.width()  or 1200
        H = self.height() or 800

        if show1:
            self._h_top.setSizes([W // 2, W // 2])
        else:
            self._h_top.setSizes([W, 0])

        if show2:
            self._v_main.setSizes([H // 2, H // 2])
            self._h_bot.setSizes(
                [W // 2, W // 2] if show3 else [W, 0]
            )
        else:
            self._v_main.setSizes([H, 0])

        # Clamp active to visible range
        n = self.n_visible()
        if self._active_idx >= n:
            self._activate(0)
        else:
            self._activate(self._active_idx)

    def set_header(self, idx: int, filename: str, badge: str = ""):
        if 0 <= idx < len(self._cells):
            cell = self._cells[idx]
            zoom = getattr(cell, "_last_zoom", 0.0)
            cell.set_header_text(filename, badge, zoom)

    def update_cell_zoom(self, idx: int, zoom: float):
        if 0 <= idx < len(self._cells):
            self._cells[idx]._last_zoom = zoom
            self._cells[idx].update_zoom(zoom)

    # ── Internal ─────────────────────────────────────────────────

    def _activate(self, idx: int):
        n = self.n_visible()
        idx = max(0, min(idx, n - 1))
        for i, cell in enumerate(self._cells):
            cell.set_active(i == idx)
        self._active_idx = idx
        self.active_changed.emit(idx)
