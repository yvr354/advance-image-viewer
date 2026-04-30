"""
VyuhaAI Image Viewer — Main Window
Orchestrates all panels. Every signal connected. Nothing isolated.

Signal flow:
  open_image → GLImageViewer + InspectorPanel + Surface3DPanel + BrowserPanel
  pipeline_changed → re-process → GLImageViewer + Surface3DPanel
  analysis_done → InspectorPanel + GLImageViewer heatmap + status bar
  pixel_hovered → InspectorPanel pixel display
  zoom_changed → status bar zoom label
  fusion composite_ready → GLImageViewer
"""

import os
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QStatusBar, QMenu,
    QFileDialog, QMessageBox, QLabel, QWidget,
    QHBoxLayout, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QKeySequence, QAction, QIcon

from src.core.config import Config
from src.core.image_data import ImageData
from src.core.image_loader import load_image, list_images_in_folder
from src.analysis.focus_engine import FocusEngine
from src.analysis.quality_engine import QualityEngine
from src.pipeline.pipeline import Pipeline

from src.ui.panels.gl_viewer        import GLImageViewer
from src.ui.panels.inspector_panel  import InspectorPanel
from src.ui.panels.pipeline_panel   import PipelinePanel
from src.ui.panels.browser_panel    import BrowserPanel
from src.ui.panels.fusion_panel     import FusionPanel
from src.ui.panels.surface_3d_panel import Surface3DPanel
from src.ui.panels.comparison_panel import ComparisonPanel
from src.ui.theme import VERDICT_COLOR, COLOR_PERFECT, COLOR_WARN, COLOR_FAIL


# ── Background analysis worker ─────────────────────────────────────────────

class AnalysisWorker(QThread):
    """Runs focus + quality analysis off the UI thread — never blocks display."""
    finished = pyqtSignal(object, object)   # FocusResult, QualityResult

    def __init__(self, image: np.ndarray, focus: FocusEngine, quality: QualityEngine):
        super().__init__()
        self._image   = image
        self._focus   = focus
        self._quality = quality

    def run(self):
        focus_result   = self._focus.analyze(self._image)
        quality_result = self._quality.analyze(self._image)
        self.finished.emit(focus_result, quality_result)


# ── Main Window ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self, config: Config):
        super().__init__()
        self.config  = config
        self.current_image: ImageData = ImageData()
        self.folder_images: list[str] = []
        self.folder_index:  int       = -1
        self._analysis_worker: AnalysisWorker | None = None

        self.focus_engine   = FocusEngine(
            metric=config.focus_metric,
            grid=config.focus_grid_size,
        )
        self.focus_engine.set_thresholds(
            config.focus_thresholds.perfect,
            config.focus_thresholds.good,
            config.focus_thresholds.soft,
        )
        self.quality_engine = QualityEngine(
            overexpose_threshold=config.overexpose_threshold,
            underexpose_threshold=config.underexpose_threshold,
        )
        self.pipeline = Pipeline()

        self._build_ui()
        self._connect_signals()
        self._build_menu()
        self._restore_geometry()

    # ═══════════════════════════════════════════════════════════════
    #  UI Construction
    # ═══════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.setWindowTitle("VyuhaAI Image Viewer")

        # ── Central widget: OpenGL viewer ──────────────────────────
        self.viewer = GLImageViewer()
        self.viewer.setAcceptDrops(True)
        self.setCentralWidget(self.viewer)

        # ── Inspector dock (right) ─────────────────────────────────
        self.inspector = InspectorPanel()
        self._dock_inspector = self._make_dock(
            "Inspector", self.inspector,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )

        # ── Pipeline dock (bottom) ─────────────────────────────────
        self.pipeline_panel = PipelinePanel(self.pipeline)
        self._dock_pipeline = self._make_dock(
            "⚙  Processing Pipeline", self.pipeline_panel,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self._dock_pipeline.setMinimumHeight(200)

        # ── Browser dock (left) ────────────────────────────────────
        self.browser = BrowserPanel()
        self._dock_browser = self._make_dock(
            "File Browser", self.browser,
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )

        # ── 3D Surface dock (right, tabbed with inspector) ─────────
        self.surface_3d = Surface3DPanel()
        self._dock_3d = self._make_dock(
            "◈  3D Surface View", self.surface_3d,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.tabifyDockWidget(self._dock_inspector, self._dock_3d)
        self._dock_inspector.raise_()   # Inspector on top by default

        # ── Fusion dock (bottom, tabbed with pipeline) ────────────────
        self.fusion_panel = FusionPanel()
        self._dock_fusion = self._make_dock(
            "⊕  Illumination Fusion", self.fusion_panel,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.tabifyDockWidget(self._dock_pipeline, self._dock_fusion)

        # ── Comparison dock (bottom, tabbed with pipeline) ─────────
        self.comparison_panel = ComparisonPanel()
        self._dock_compare = self._make_dock(
            "⊞  Image Comparison", self.comparison_panel,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.tabifyDockWidget(self._dock_pipeline, self._dock_compare)
        self._dock_pipeline.raise_()

        # All bottom docks: enforce minimum height so they can never collapse
        for dock in [self._dock_pipeline, self._dock_fusion, self._dock_compare]:
            dock.setMinimumHeight(200)

        # ── Status bar ─────────────────────────────────────────────
        self._build_status_bar()

        # Store all docks for menu toggle
        self._docks = {
            "Inspector":          self._dock_inspector,
            "3D Surface View":    self._dock_3d,
            "Processing Pipeline":self._dock_pipeline,
            "Illumination Fusion":self._dock_fusion,
            "Image Comparison":   self._dock_compare,
            "File Browser":       self._dock_browser,
        }

    def _make_dock(self, title: str, widget: QWidget,
                   area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea   |
            Qt.DockWidgetArea.RightDockWidgetArea  |
            Qt.DockWidgetArea.BottomDockWidgetArea |
            Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.addDockWidget(area, dock)
        return dock

    def _build_status_bar(self):
        sb = QStatusBar()
        sb.setSizeGripEnabled(True)

        # Left: filename + metrics
        self._status_main = QLabel("VyuhaAI Image Viewer  —  Open an image  (Ctrl+O)")
        self._status_main.setStyleSheet("color: #8888AA; padding: 0 8px;")

        # Right: zoom level
        self._status_zoom = QLabel("Zoom: —")
        self._status_zoom.setStyleSheet("color: #00B4D8; padding: 0 8px; font-weight: 600;")
        self._status_zoom.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Right: focus verdict badge
        self._status_verdict = QLabel("")
        self._status_verdict.setStyleSheet("padding: 0 10px; font-weight: 700;")

        sb.addWidget(self._status_main, stretch=1)
        sb.addPermanentWidget(self._status_verdict)
        sb.addPermanentWidget(self._status_zoom)
        self.setStatusBar(sb)

    # ═══════════════════════════════════════════════════════════════
    #  Signal Wiring — everything connected to everything
    # ═══════════════════════════════════════════════════════════════

    def _connect_signals(self):
        # Viewer → status bar zoom
        self.viewer.zoom_changed.connect(self._on_zoom_changed)

        # Viewer → inspector pixel display (live on mouse move)
        self.viewer.pixel_hovered.connect(self._on_pixel_hovered)

        # Browser → open image
        self.browser.image_selected.connect(self.open_image)

        # Pipeline → re-process and update viewer + 3D
        self.pipeline_panel.pipeline_changed.connect(self._on_pipeline_changed)

        # Fusion → send composite to viewer
        self.fusion_panel.composite_ready.connect(self._on_composite_ready)

        # Comparison panel → open image in main viewer
        self.comparison_panel.open_image_requested.connect(self.open_image)

    # ═══════════════════════════════════════════════════════════════
    #  Menu
    # ═══════════════════════════════════════════════════════════════

    def _build_menu(self):
        mb = self.menuBar()

        # ── File ───────────────────────────────────────────────────
        m = mb.addMenu("&File")
        self._action(m, "Open Image…",     self._menu_open_image,  "Ctrl+O")
        self._action(m, "Open Folder…",    self._menu_open_folder, "Ctrl+Shift+O")
        m.addSeparator()
        self._action(m, "Add to Comparison", self._add_to_comparison, "Ctrl+Shift+C")
        m.addSeparator()
        self._action(m, "Save Processed Image…", self._menu_save,  "Ctrl+S")
        m.addSeparator()
        self._action(m, "Exit",            self.close,             "Alt+F4")

        # ── View ───────────────────────────────────────────────────
        m = mb.addMenu("&View")
        self._action(m, "Fit to Window",   self.viewer.fit_to_window,          "Space")
        self._action(m, "100% Zoom",       lambda: self.viewer.set_zoom(1.0),  "1")
        self._action(m, "200% Zoom",       lambda: self.viewer.set_zoom(2.0),  "2")
        self._action(m, "400% Zoom",       lambda: self.viewer.set_zoom(4.0),  "4")
        self._action(m, "50% Zoom",        lambda: self.viewer.set_zoom(0.5),  "5")
        m.addSeparator()
        self._action(m, "Toggle Focus Heatmap",  self.viewer.toggle_heatmap, "F")
        self._action(m, "Toggle Histogram",      self.inspector.toggle_histogram, "H")
        m.addSeparator()
        for name, dock in self._docks.items():
            m.addAction(dock.toggleViewAction())

        # ── Navigate ───────────────────────────────────────────────
        m = mb.addMenu("&Navigate")
        self._action(m, "Previous Image",  self._prev_image, "Left")
        self._action(m, "Next Image",      self._next_image, "Right")
        self._action(m, "First Image",     self._first_image, "Home")
        self._action(m, "Last Image",      self._last_image,  "End")

        # ── Pipeline ───────────────────────────────────────────────
        m = mb.addMenu("&Pipeline")
        self._action(m, "Clear Pipeline",   self._clear_pipeline)
        self._action(m, "Save Pipeline…",   self._save_pipeline,  "Ctrl+Shift+S")
        self._action(m, "Load Pipeline…",   self._load_pipeline,  "Ctrl+Shift+L")

        # ── Tools ──────────────────────────────────────────────────
        m = mb.addMenu("&Tools")
        self._action(m, "Illumination Fusion",
                     lambda: self._toggle_dock(self._dock_fusion), "Ctrl+F")
        self._action(m, "3D Surface View",
                     lambda: self._toggle_dock(self._dock_3d),    "Ctrl+3")
        self._action(m, "Image Comparison",
                     lambda: self._toggle_dock(self._dock_compare), "Ctrl+M")

    def _action(self, menu: QMenu, label: str, slot, shortcut: str = "") -> QAction:
        a = QAction(label, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _toggle_dock(self, dock: QDockWidget):
        dock.setVisible(not dock.isVisible())
        if dock.isVisible():
            dock.raise_()

    # ═══════════════════════════════════════════════════════════════
    #  Image Loading
    # ═══════════════════════════════════════════════════════════════

    def open_image(self, path: str):
        try:
            self.current_image = load_image(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self.config.add_recent(path)
        self.config.save()
        self.browser.highlight_path(path)

        # Display immediately — analysis runs in background
        self._display_current()
        self._run_analysis()

        self._status_main.setText(
            f"  {self.current_image.filename}   "
            f"{self.current_image.shape_str()}"
        )
        self._status_verdict.setText("")

    def _display_current(self):
        if not self.current_image.is_loaded():
            return

        # Apply pipeline (CPU — only on visible region in future tile version)
        processed = self.pipeline.process(self.current_image.raw)
        self.current_image.display = processed

        # Update OpenGL viewer — GPU upload, instant display
        self.viewer.set_image(processed)

        # Update 3D surface with processed image
        self.surface_3d.set_image(processed)

        # Update histogram in inspector
        hist_data = self.quality_engine.compute_histogram(processed)
        self.inspector.update_histogram(hist_data)

    def _run_analysis(self):
        if not self.current_image.is_loaded():
            return
        # Cancel previous worker if still running
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.quit()
            self._analysis_worker.wait()

        self._analysis_worker = AnalysisWorker(
            self.current_image.raw,
            self.focus_engine,
            self.quality_engine,
        )
        self._analysis_worker.finished.connect(self._on_analysis_done)
        self._analysis_worker.start()

    # ═══════════════════════════════════════════════════════════════
    #  Signal Handlers
    # ═══════════════════════════════════════════════════════════════

    def _on_analysis_done(self, focus_result, quality_result):
        """Called from background thread — update all panels with results."""
        # Store on image data
        self.current_image.focus_score    = focus_result.score
        self.current_image.focus_verdict  = focus_result.verdict
        self.current_image.focus_map      = focus_result.heatmap
        self.current_image.quality_score  = quality_result.overall_score
        self.current_image.quality_verdict = quality_result.verdict

        # Inspector: focus + quality metrics
        self.inspector.update_focus(focus_result)
        self.inspector.update_quality(quality_result)

        # Viewer: focus heatmap overlay
        heatmap_rgb = self.focus_engine.heatmap_to_rgb(focus_result.heatmap)
        self.viewer.set_heatmap(heatmap_rgb)

        # Status bar: color-coded verdict
        self._update_status_verdict(focus_result, quality_result)

    def _on_pipeline_changed(self):
        """Pipeline layer added/removed/changed → re-process and update all views."""
        self._display_current()

    def _on_composite_ready(self, composite: np.ndarray):
        """Fusion panel produced a composite → show in viewer and 3D."""
        self.viewer.set_image(composite)
        self.surface_3d.set_image(composite)
        hist_data = self.quality_engine.compute_histogram(composite)
        self.inspector.update_histogram(hist_data)

    def _on_pixel_hovered(self, x: int, y: int, pixel):
        """Mouse moved over viewer → update pixel inspector."""
        self.inspector.update_pixel(x, y, pixel)

    def _on_zoom_changed(self, zoom: float):
        """Viewer zoom changed → update status bar zoom label."""
        pct = zoom * 100
        if pct >= 100:
            self._status_zoom.setText(f"  {pct:.0f}%  ")
        else:
            self._status_zoom.setText(f"  {pct:.1f}%  ")

    def _update_status_verdict(self, focus_result, quality_result):
        f = focus_result
        q = quality_result
        fcolor = VERDICT_COLOR.get(f.verdict, "#888")
        qcolor = COLOR_PERFECT if q.verdict == "PASS" else COLOR_FAIL

        self._status_main.setText(
            f"  {self.current_image.filename}   "
            f"{self.current_image.shape_str()}   │   "
            f"Focus: {f.score:.0f}   │   "
            f"Quality: {q.overall_score:.0f}/100"
        )
        self._status_verdict.setText(
            f"  {f.verdict}  ·  {q.verdict}  "
        )
        self._status_verdict.setStyleSheet(
            f"color: {fcolor}; padding: 0 10px; font-weight: 700; font-size: 11px;"
        )

    # ═══════════════════════════════════════════════════════════════
    #  Navigation
    # ═══════════════════════════════════════════════════════════════

    def _prev_image(self):
        if self.folder_index > 0:
            self.folder_index -= 1
            self.open_image(self.folder_images[self.folder_index])

    def _next_image(self):
        if self.folder_index < len(self.folder_images) - 1:
            self.folder_index += 1
            self.open_image(self.folder_images[self.folder_index])

    def _first_image(self):
        if self.folder_images:
            self.folder_index = 0
            self.open_image(self.folder_images[0])

    def _last_image(self):
        if self.folder_images:
            self.folder_index = len(self.folder_images) - 1
            self.open_image(self.folder_images[-1])

    # ═══════════════════════════════════════════════════════════════
    #  Pipeline
    # ═══════════════════════════════════════════════════════════════

    def _clear_pipeline(self):
        self.pipeline.clear()
        self.pipeline_panel.refresh()
        self._display_current()

    def _save_pipeline(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "", "VyuhaAI Pipeline (*.pipeline)"
        )
        if path:
            self.pipeline.save(path)

    def _load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", "", "VyuhaAI Pipeline (*.pipeline)"
        )
        if path:
            self.pipeline.load(path)
            self.pipeline_panel.refresh()
            self._display_current()

    # ═══════════════════════════════════════════════════════════════
    #  Menu Handlers
    # ═══════════════════════════════════════════════════════════════

    def _menu_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.config.last_folder,
            "All Images (*.tiff *.tif *.png *.bmp *.jpg *.jpeg *.pgm *.ppm *.exr *.hdr);;"
            "TIFF (*.tiff *.tif);;PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)"
        )
        if not path:
            return
        folder = os.path.dirname(path)
        self.config.last_folder = folder
        self.folder_images = list_images_in_folder(folder)
        self.folder_index  = self.folder_images.index(path) if path in self.folder_images else -1
        self.browser.set_folder(folder)
        self.open_image(path)

    def _menu_open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Open Folder", self.config.last_folder
        )
        if not folder:
            return
        self.config.last_folder = folder
        self.folder_images = list_images_in_folder(folder)
        self.folder_index  = 0
        self.browser.set_folder(folder)
        if self.folder_images:
            self.open_image(self.folder_images[0])

    def _add_to_comparison(self):
        if self.current_image.path:
            self._dock_compare.setVisible(True)
            self._dock_compare.raise_()
            self.comparison_panel.add_image_from_path(self.current_image.path)

    def _menu_save(self):
        if self.current_image.display is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "",
            "PNG (*.png);;TIFF (*.tiff);;BMP (*.bmp);;JPEG (*.jpg)"
        )
        if not path:
            return
        img = self.current_image.display
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)
        self._status_main.setText(f"  Saved: {path}")

    # ═══════════════════════════════════════════════════════════════
    #  Lifecycle
    # ═══════════════════════════════════════════════════════════════

    def _restore_geometry(self):
        self.showMaximized()   # always start maximized — full screen layout
        # Dock sizes set after show so Qt has real pixel dimensions
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._set_dock_sizes)

    def _set_dock_sizes(self):
        """Set initial dock proportions after the window is fully shown."""
        h = self.height()
        w = self.width()
        bottom_h = max(260, int(h * 0.28))
        left_w   = max(200, int(w * 0.14))
        right_w  = max(260, int(w * 0.18))

        self.resizeDocks(
            [self._dock_pipeline],
            [bottom_h],
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self._dock_browser],
            [left_w],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self._dock_inspector],
            [right_w],
            Qt.Orientation.Horizontal,
        )

    def closeEvent(self, event):
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.quit()
            self._analysis_worker.wait()
        self.config.window_width     = self.width()
        self.config.window_height    = self.height()
        self.config.window_maximized = self.isMaximized()
        self.config.save()
        super().closeEvent(event)
