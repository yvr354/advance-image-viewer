"""Main application window — orchestrates all panels."""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QSplitter, QStatusBar,
    QMenuBar, QMenu, QFileDialog, QMessageBox, QWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QKeySequence, QAction

from src.core.config import Config
from src.core.image_data import ImageData
from src.core.image_loader import load_image, list_images_in_folder
from src.analysis.focus_engine import FocusEngine
from src.analysis.quality_engine import QualityEngine
from src.pipeline.pipeline import Pipeline

from src.ui.panels.viewer_panel import ViewerPanel
from src.ui.panels.inspector_panel import InspectorPanel
from src.ui.panels.pipeline_panel import PipelinePanel
from src.ui.panels.browser_panel import BrowserPanel
from src.ui.panels.fusion_panel import FusionPanel


class AnalysisWorker(QThread):
    """Run focus + quality analysis off the main thread."""
    finished = pyqtSignal(object, object)  # FocusResult, QualityResult

    def __init__(self, image_data: ImageData, focus_engine: FocusEngine, quality_engine: QualityEngine):
        super().__init__()
        self._image = image_data
        self._focus = focus_engine
        self._quality = quality_engine

    def run(self):
        if self._image.raw is None:
            return
        focus_result  = self._focus.analyze(self._image.raw)
        quality_result = self._quality.analyze(self._image.raw)
        self.finished.emit(focus_result, quality_result)


class MainWindow(QMainWindow):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.current_image: ImageData = ImageData()
        self.folder_images: list[str] = []
        self.folder_index: int = -1
        self._analysis_worker: AnalysisWorker | None = None

        self.focus_engine   = FocusEngine(metric=config.focus_metric, grid=config.focus_grid_size)
        self.quality_engine = QualityEngine(
            overexpose_threshold=config.overexpose_threshold,
            underexpose_threshold=config.underexpose_threshold,
        )
        self.pipeline = Pipeline()

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self._restore_geometry()

    # ------------------------------------------------------------------ #
    #  UI Construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        self.setWindowTitle("VyuhaAI Image Viewer")

        # Central viewer
        self.viewer = ViewerPanel(self.config)
        self.setCentralWidget(self.viewer)

        # Inspector dock (right)
        self.inspector = InspectorPanel()
        dock_inspector = QDockWidget("Inspector", self)
        dock_inspector.setWidget(self.inspector)
        dock_inspector.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_inspector)

        # Pipeline dock (bottom-right)
        self.pipeline_panel = PipelinePanel(self.pipeline)
        self.pipeline_panel.pipeline_changed.connect(self._on_pipeline_changed)
        dock_pipeline = QDockWidget("Processing Pipeline", self)
        dock_pipeline.setWidget(self.pipeline_panel)
        dock_pipeline.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_pipeline)

        # Browser dock (left)
        self.browser = BrowserPanel()
        self.browser.image_selected.connect(self.open_image)
        dock_browser = QDockWidget("File Browser", self)
        dock_browser.setWidget(self.browser)
        dock_browser.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_browser)

        # Fusion dock (bottom)
        self.fusion_panel = FusionPanel()
        self.fusion_panel.composite_ready.connect(self._on_composite_ready)
        dock_fusion = QDockWidget("Illumination Fusion", self)
        dock_fusion.setWidget(self.fusion_panel)
        dock_fusion.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_fusion)
        dock_fusion.hide()   # Hidden by default, open from menu

        # Store dock references for menu toggle
        self._docks = {
            "inspector": dock_inspector,
            "pipeline":  dock_pipeline,
            "browser":   dock_browser,
            "fusion":    dock_fusion,
        }

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("VyuhaAI Image Viewer  —  Open an image or folder  (Ctrl+O)")

    def _build_menu(self):
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        self._add_action(file_menu, "Open Image...",  self._menu_open_image,  "Ctrl+O")
        self._add_action(file_menu, "Open Folder...", self._menu_open_folder, "Ctrl+Shift+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "Save Processed Image...", self._menu_save, "Ctrl+S")
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", self.close, "Alt+F4")

        # View
        view_menu = mb.addMenu("&View")
        self._add_action(view_menu, "Fit to Window",  self.viewer.fit_to_window, "Space")
        self._add_action(view_menu, "100% Zoom",      lambda: self.viewer.set_zoom(1.0), "1")
        self._add_action(view_menu, "200% Zoom",      lambda: self.viewer.set_zoom(2.0), "2")
        self._add_action(view_menu, "50% Zoom",       lambda: self.viewer.set_zoom(0.5))
        view_menu.addSeparator()
        self._add_action(view_menu, "Toggle Focus Heatmap", self.viewer.toggle_heatmap, "F")
        self._add_action(view_menu, "Toggle Histogram",     self.inspector.toggle_histogram, "H")
        view_menu.addSeparator()
        for name, dock in self._docks.items():
            action = dock.toggleViewAction()
            view_menu.addAction(action)

        # Navigate
        nav_menu = mb.addMenu("&Navigate")
        self._add_action(nav_menu, "Previous Image", self._prev_image, "Left")
        self._add_action(nav_menu, "Next Image",     self._next_image, "Right")

        # Pipeline
        pipe_menu = mb.addMenu("&Pipeline")
        self._add_action(pipe_menu, "Clear Pipeline",  self._clear_pipeline)
        self._add_action(pipe_menu, "Save Pipeline...", self._save_pipeline)
        self._add_action(pipe_menu, "Load Pipeline...", self._load_pipeline)

        # Tools
        tools_menu = mb.addMenu("&Tools")
        self._add_action(tools_menu, "Toggle Illumination Fusion", lambda: self._docks["fusion"].setVisible(not self._docks["fusion"].isVisible()))

    def _build_shortcuts(self):
        pass  # Shortcuts handled in menu via QKeySequence

    def _add_action(self, menu: QMenu, label: str, slot, shortcut: str = ""):
        action = QAction(label, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _restore_geometry(self):
        self.resize(self.config.window_width, self.config.window_height)
        if self.config.window_maximized:
            self.showMaximized()

    # ------------------------------------------------------------------ #
    #  Image Loading
    # ------------------------------------------------------------------ #

    def open_image(self, path: str):
        try:
            self.current_image = load_image(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self.config.add_recent(path)
        self.config.save()

        self._display_current()
        self._run_analysis()
        self.status.showMessage(f"{self.current_image.filename}  —  {self.current_image.shape_str()}")

    def _display_current(self):
        if not self.current_image.is_loaded():
            return
        processed = self.pipeline.process(self.current_image.raw)
        self.current_image.display = processed
        self.viewer.set_image(processed, self.current_image)

    def _run_analysis(self):
        if not self.current_image.is_loaded():
            return
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.quit()
        self._analysis_worker = AnalysisWorker(
            self.current_image, self.focus_engine, self.quality_engine
        )
        self._analysis_worker.finished.connect(self._on_analysis_done)
        self._analysis_worker.start()

    def _on_analysis_done(self, focus_result, quality_result):
        self.current_image.focus_score   = focus_result.score
        self.current_image.focus_verdict = focus_result.verdict
        self.current_image.focus_map     = focus_result.heatmap
        self.current_image.quality_score = quality_result.overall_score
        self.current_image.quality_verdict = quality_result.verdict

        self.inspector.update_focus(focus_result)
        self.inspector.update_quality(quality_result)
        self.viewer.set_focus_map(focus_result.heatmap)

        verdict_color = {"PERFECT": "green", "GOOD": "limegreen", "SOFT": "orange", "BLURRY": "red"}
        color = verdict_color.get(focus_result.verdict, "white")
        self.status.showMessage(
            f"{self.current_image.filename}  —  "
            f"Focus: {focus_result.score:.0f} [{focus_result.verdict}]  —  "
            f"Quality: {quality_result.overall_score:.0f}/100 [{quality_result.verdict}]"
        )

    # ------------------------------------------------------------------ #
    #  Navigation
    # ------------------------------------------------------------------ #

    def _prev_image(self):
        if self.folder_index > 0:
            self.folder_index -= 1
            self.open_image(self.folder_images[self.folder_index])

    def _next_image(self):
        if self.folder_index < len(self.folder_images) - 1:
            self.folder_index += 1
            self.open_image(self.folder_images[self.folder_index])

    # ------------------------------------------------------------------ #
    #  Pipeline
    # ------------------------------------------------------------------ #

    def _on_pipeline_changed(self):
        self._display_current()

    def _clear_pipeline(self):
        self.pipeline.clear()
        self.pipeline_panel.refresh()
        self._display_current()

    def _save_pipeline(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline", "", "Pipeline Files (*.pipeline)")
        if path:
            self.pipeline.save(path)

    def _load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Pipeline", "", "Pipeline Files (*.pipeline)")
        if path:
            self.pipeline.load(path)
            self.pipeline_panel.refresh()
            self._display_current()

    # ------------------------------------------------------------------ #
    #  Fusion
    # ------------------------------------------------------------------ #

    def _on_composite_ready(self, composite):
        self.viewer.set_image(composite, self.current_image)

    # ------------------------------------------------------------------ #
    #  Menu handlers
    # ------------------------------------------------------------------ #

    def _menu_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.config.last_folder,
            "Images (*.tiff *.tif *.png *.bmp *.jpg *.jpeg *.pgm *.ppm *.exr *.hdr)"
        )
        if path:
            self.config.last_folder = os.path.dirname(path)
            folder = os.path.dirname(path)
            self.folder_images = list_images_in_folder(folder)
            self.folder_index  = self.folder_images.index(path) if path in self.folder_images else -1
            self.browser.set_folder(folder)
            self.open_image(path)

    def _menu_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder", self.config.last_folder)
        if folder:
            self.config.last_folder = folder
            self.folder_images = list_images_in_folder(folder)
            self.folder_index  = 0
            self.browser.set_folder(folder)
            if self.folder_images:
                self.open_image(self.folder_images[0])

    def _menu_save(self):
        if self.current_image.display is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;TIFF (*.tiff);;BMP (*.bmp)"
        )
        if path:
            import cv2
            img = self.current_image.display
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img)
            self.status.showMessage(f"Saved: {path}")

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        self.config.window_width  = self.width()
        self.config.window_height = self.height()
        self.config.window_maximized = self.isMaximized()
        self.config.save()
        super().closeEvent(event)
