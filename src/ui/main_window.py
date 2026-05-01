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
    QApplication,
    QMainWindow, QDockWidget, QStatusBar, QMenu,
    QFileDialog, QMessageBox, QLabel, QWidget,
    QHBoxLayout, QVBoxLayout, QSizePolicy, QPushButton,
    QButtonGroup, QStackedWidget, QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QKeySequence, QAction, QIcon, QShortcut

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
from src.ui.panels.multi_viewer     import MultiViewer
from src.ui.theme import VERDICT_COLOR, COLOR_PERFECT, COLOR_WARN, COLOR_FAIL


# ── Background workers ─────────────────────────────────────────────────────

class ImageLoadWorker(QThread):
    """Loads image file off UI thread — large TIFFs/BMPs never freeze the window."""
    loaded  = pyqtSignal(object)   # ImageData
    failed  = pyqtSignal(str)      # error message

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def run(self):
        try:
            data = load_image(self._path)
            self.loaded.emit(data)
        except Exception as e:
            self.failed.emit(str(e))


class AnalysisWorker(QThread):
    """Runs focus + quality analysis off the UI thread — never blocks display."""
    finished = pyqtSignal(object, object)   # FocusResult, QualityResult

    def __init__(self, image: np.ndarray, focus: FocusEngine, quality: QualityEngine,
                 reference=None, mask=None):
        super().__init__()
        self._image     = image
        self._focus     = focus
        self._quality   = quality
        self._reference = reference
        self._mask      = mask

    def run(self):
        focus_result   = self._focus.analyze(self._image, self._reference, self._mask)
        quality_result = self._quality.analyze(self._image, self._mask)
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
        self._load_worker:    ImageLoadWorker | None = None
        self._active_mode = "Inspect"
        self._last_focus_result = None
        self._last_quality_result = None
        self._mm_per_px: float = 0.0

        # Mask system
        from src.analysis.mask_engine import MaskData
        self._mask_data: MaskData | None = None   # current inspection mask

        # Reference system
        from src.analysis.focus_engine import FocusReference
        self._focus_reference: FocusReference | None = None   # current reference (auto or locked)
        self._session_best_raw_lap: float = 0.0               # best raw Laplacian seen this session
        self._ref_path: str = str(                            # where locked reference is saved
            config.config_dir / "focus_reference.json"
            if hasattr(config, "config_dir") else
            __import__("pathlib").Path.home() / ".vyuhaai_focus_ref.json"
        )
        self._load_locked_reference()                          # restore from previous session

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
        self.setMinimumSize(700, 450)   # hard floor — never forced bigger by docks
        # Prevent Qt from auto-resizing the main window when docks appear/disappear
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks |
            QMainWindow.DockOption.AllowTabbedDocks
        )

        # ── Central widget: OpenGL viewer ──────────────────────────
        # Central workspace: one active mode owns the main visual area.
        self._central_root = QWidget()
        central_layout = QVBoxLayout(self._central_root)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self._mode_buttons: dict[str, QPushButton] = {}
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        mode_bar = QHBoxLayout()
        mode_bar.setContentsMargins(8, 4, 8, 4)
        mode_bar.setSpacing(6)
        for mode in ["Inspect", "Focus", "Compare", "Tune", "Fusion", "3D"]:
            btn = QPushButton(mode)
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda checked, m=mode: self._set_mode(m))
            self._mode_group.addButton(btn)
            self._mode_buttons[mode] = btn
            mode_bar.addWidget(btn)
        mode_bar.addStretch()

        # ── Split layout buttons ───────────────────────────────────
        _split_style = (
            "QPushButton { background:#0C1824; color:#446688; border:1px solid #1A2A3A; "
            "              padding:2px 8px; font-size:11px; border-radius:3px; }"
            "QPushButton:checked { background:#0E2840; color:#00AAFF; border:1px solid #00AAFF; }"
            "QPushButton:hover   { color:#AACCDD; }"
        )
        self._split_group  = QButtonGroup(self)
        self._split_group.setExclusive(True)
        self._split_buttons: dict[str, QPushButton] = {}
        for key, label in [("1","▣"), ("2H","◫"), ("2V","⬒"), ("4","⊞")]:
            sb = QPushButton(label)
            sb.setCheckable(True)
            sb.setFixedSize(28, 26)
            sb.setToolTip({"1":"Single view","2H":"Split left|right",
                           "2V":"Split top/bottom","4":"4-grid"}[key])
            sb.setStyleSheet(_split_style)
            sb.clicked.connect(lambda checked, k=key: self._set_split(k))
            self._split_group.addButton(sb)
            self._split_buttons[key] = sb
            mode_bar.addWidget(sb)
        self._split_buttons["1"].setChecked(True)

        # ── Sync zoom toggle ──────────────────────────────────────
        mode_bar.addSpacing(8)
        self._btn_sync_zoom = QPushButton("⇄ Sync")
        self._btn_sync_zoom.setCheckable(True)
        self._btn_sync_zoom.setChecked(True)
        self._btn_sync_zoom.setFixedHeight(26)
        self._btn_sync_zoom.setToolTip("Sync zoom & pan across all split panels")
        self._btn_sync_zoom.setStyleSheet(
            "QPushButton { background:#0C1824; color:#446688; border:1px solid #1A2A3A; "
            "              padding:2px 8px; font-size:11px; border-radius:3px; }"
            "QPushButton:checked { background:#0E2840; color:#00AAFF; border:1px solid #00AAFF; }"
            "QPushButton:hover   { color:#AACCDD; }"
        )
        mode_bar.addWidget(self._btn_sync_zoom)

        central_layout.addLayout(mode_bar)

        self._workspace = QStackedWidget()
        central_layout.addWidget(self._workspace, stretch=1)

        self._image_page = QWidget()
        image_layout = QVBoxLayout(self._image_page)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(4)
        self._mode_banner = QLabel("")
        self._mode_banner.setFixedHeight(30)
        self._mode_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mode_banner.setStyleSheet(
            "background: #101826; color: #00D4F8; "
            "border-bottom: 1px solid #25354A; font-weight: 700;"
        )
        image_layout.addWidget(self._mode_banner)

        # ── Inspect tool toolbar (only visible in Inspect mode) ────
        self._inspect_toolbar = self._build_inspect_toolbar()
        image_layout.addWidget(self._inspect_toolbar)

        self.multi_viewer = MultiViewer()
        image_layout.addWidget(self.multi_viewer, stretch=1)
        self._workspace.addWidget(self._image_page)

        # Primary viewer alias — always points to the active cell's GLImageViewer
        self.viewer = self.multi_viewer.active_viewer

        # Per-cell image data and analysis results
        self._cell_images: list = [None, None, None, None]  # ImageData per cell
        self.multi_viewer.active_changed.connect(self._on_active_cell_changed)

        self.comparison_panel = ComparisonPanel()
        self._workspace.addWidget(self.comparison_panel)

        self.surface_3d = Surface3DPanel()
        self._workspace.addWidget(self.surface_3d)

        self.setCentralWidget(self._central_root)

        # ── Inspector dock (right) ─────────────────────────────────
        self.inspector = InspectorPanel()
        self._dock_inspector = self._make_dock(
            "📊  Inspector — Focus / Quality / Histogram",
            self.inspector,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )

        # ── Pipeline dock (bottom) ─────────────────────────────────
        self.pipeline_panel = PipelinePanel(self.pipeline)
        self._dock_pipeline = self._make_dock(
            "⚙  Processing Pipeline — Filters & Effects",
            self.pipeline_panel,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self._dock_pipeline.setMinimumHeight(200)

        # ── Browser dock (left) ────────────────────────────────────
        self.browser = BrowserPanel()
        self._dock_browser = self._make_dock(
            "📁  File Browser",
            self.browser,
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )

        self._dock_inspector.raise_()   # Inspector on top by default

        _focus_dock_widget = self._build_focus_assist_widget()
        self._dock_focus = self._make_dock(
            "Focus Assist",
            _focus_dock_widget,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.tabifyDockWidget(self._dock_inspector, self._dock_focus)

        # ── Fusion dock (bottom, tabbed with pipeline) ────────────────
        self.fusion_panel = FusionPanel()
        self._dock_fusion = self._make_dock(
            "⊕  Illumination Fusion — Multi-Light Composite",
            self.fusion_panel,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.tabifyDockWidget(self._dock_pipeline, self._dock_fusion)

        self._dock_pipeline.raise_()

        # Bottom docks: allow user to resize freely (min 80px = just the tab strip)
        for dock in [self._dock_pipeline, self._dock_fusion]:
            dock.setMinimumHeight(80)

        # ── Status bar ─────────────────────────────────────────────
        self._build_status_bar()

        # Store all docks for menu toggle
        self._docks = {
            "📊 Inspector":          self._dock_inspector,
            "🎯 Focus Assist":       self._dock_focus,
            "⚙ Processing Pipeline": self._dock_pipeline,
            "⊕ Illumination Fusion": self._dock_fusion,
            "📁 File Browser":        self._dock_browser,
        }
        self._set_mode("Inspect")

    def _make_dock(self, title: str, widget: QWidget,
                   area: Qt.DockWidgetArea,
                   floatable: bool = False) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea   |
            Qt.DockWidgetArea.RightDockWidgetArea  |
            Qt.DockWidgetArea.BottomDockWidgetArea |
            Qt.DockWidgetArea.TopDockWidgetArea
        )
        # Prevent accidental floating — docks stay inside the main window
        features = QDockWidget.DockWidgetFeature.DockWidgetMovable | \
                   QDockWidget.DockWidgetFeature.DockWidgetClosable
        if floatable:
            features |= QDockWidget.DockWidgetFeature.DockWidgetFloatable
        dock.setFeatures(features)
        self.addDockWidget(area, dock)
        return dock

    def _build_focus_assist_widget(self) -> QWidget:
        """Build the Focus Assist dock: reference buttons + text panel."""
        _BTN = (
            "QPushButton { background:#0A1828; color:#5588AA; border:1px solid #1A2A3A; "
            "              padding:3px 8px; font-size:9px; font-weight:600; }"
            "QPushButton:hover { background:#112233; color:#88CCEE; }"
            "QPushButton:disabled { color:#334455; }"
        )
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.setSpacing(4)

        # Reference management buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._btn_set_ref = QPushButton("📷 Use as Ref")
        self._btn_set_ref.setToolTip(
            "Set current image as AUTO reference for this session.\n"
            "Score = how sharp this image is relative to the current reference.\n"
            "Auto-ref is reset each session unless you Lock it."
        )
        self._btn_set_ref.setEnabled(False)
        self._btn_set_ref.clicked.connect(self._set_auto_reference)

        self._btn_lock_ref = QPushButton("🔒 Lock Ref")
        self._btn_lock_ref.setToolTip(
            "Lock current image as permanent reference (saved to disk).\n"
            "Survives restarts. Use only with your best known-good image.\n"
            "LOCKED REF = production-grade scoring."
        )
        self._btn_lock_ref.setEnabled(False)
        self._btn_lock_ref.clicked.connect(self._lock_reference)

        self._btn_clear_ref = QPushButton("✕ Clear")
        self._btn_clear_ref.setToolTip("Remove reference → return to RELATIVE mode")
        self._btn_clear_ref.setEnabled(False)
        self._btn_clear_ref.clicked.connect(self._clear_reference)

        for b in [self._btn_set_ref, self._btn_lock_ref, self._btn_clear_ref]:
            b.setStyleSheet(_BTN)
            btn_row.addWidget(b)

        vl.addLayout(btn_row)

        # Reference status label
        self._ref_status_lbl = QLabel("No reference  —  RELATIVE mode")
        self._ref_status_lbl.setStyleSheet(
            "color:#886633; font-size:9px; font-weight:600; padding:2px 2px;"
        )
        self._ref_status_lbl.setWordWrap(True)
        vl.addWidget(self._ref_status_lbl)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#1A2A3A;")
        vl.addWidget(line)

        # Analysis text
        self._focus_assist = QLabel(
            "FOCUS ASSIST\n\n"
            "Open an image to analyze focus.\n\n"
            "This panel shows the real focus verdict, score, warnings, "
            "and operator action after analysis completes."
        )
        self._focus_assist.setWordWrap(True)
        self._focus_assist.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._focus_assist.setStyleSheet(
            "background: #101826; color: #D7F7FF; border: 1px solid #25455E; "
            "padding: 10px; font-size: 11px; font-weight: 600;"
        )
        vl.addWidget(self._focus_assist, stretch=1)
        return w

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

    def _build_inspect_toolbar(self) -> QWidget:
        """Inspection tool strip — shown only in Inspect mode."""
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet(
            "QWidget { background:#0C1520; border-bottom:1px solid #1A2A3A; }"
            "QPushButton { background:#0A1828; color:#6688AA; border:1px solid #1A2A3A; "
            "              padding:3px 12px; font-size:10px; font-weight:600; }"
            "QPushButton:checked { background:#00304A; color:#00E5FF; "
            "                      border:1px solid #00B4D8; }"
            "QPushButton:hover:!checked { background:#111F2E; color:#AACCDD; }"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 3, 8, 3)
        row.setSpacing(4)

        self._tool_buttons: dict[str, QPushButton] = {}
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        tools = [
            ("navigate", "↖ Navigate", "Pan & zoom — left drag pans, right drag zooms"),
            ("roi",      "⬛ ROI",       "Draw rectangle → region stats in Inspector"),
            ("profile",  "📈 Profile",   "Draw line → intensity chart in Inspector"),
            ("annotate", "📍 Annotate",  "Click to place defect markers"),
            ("measure",  "↔ Measure",   "Drag to measure distance (px / mm)"),
            ("mask",     "⬡ Mask",      "Draw inspection region polygon — metrics computed only inside"),
        ]
        for tool_id, label, tip in tools:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _, t=tool_id: self._set_inspect_tool(t))
            self._tool_group.addButton(btn)
            self._tool_buttons[tool_id] = btn
            row.addWidget(btn)

        self._tool_buttons["navigate"].setChecked(True)

        row.addSpacing(12)

        # Mask: auto-detect reflections
        self._mask_auto_btn = QPushButton("✦ Auto-Detect")
        self._mask_auto_btn.setToolTip(
            "Automatically detect specular reflection regions\n"
            "and suggest an inspection mask that excludes them"
        )
        self._mask_auto_btn.setCheckable(False)
        self._mask_auto_btn.setStyleSheet(
            "QPushButton { background:#0A1828; color:#5588AA; border:1px solid #1A2A3A; "
            "              padding:3px 10px; font-size:10px; }"
            "QPushButton:hover { color:#AACCDD; background:#111F2E; }"
        )
        self._mask_auto_btn.clicked.connect(self._mask_auto_detect)
        row.addWidget(self._mask_auto_btn)

        # Mask: apply to all images in folder
        self._mask_apply_btn = QPushButton("📋 Apply to Folder")
        self._mask_apply_btn.setToolTip(
            "Align and apply current mask to all images in the folder.\n"
            "Phase 1 (fixed camera): copy directly.\n"
            "Phase 3 (part moves): auto-align using edge matching."
        )
        self._mask_apply_btn.setCheckable(False)
        self._mask_apply_btn.setStyleSheet(
            "QPushButton { background:#0A1828; color:#5588AA; border:1px solid #1A2A3A; "
            "              padding:3px 10px; font-size:10px; }"
            "QPushButton:hover { color:#AACCDD; background:#111F2E; }"
        )
        self._mask_apply_btn.clicked.connect(self._mask_apply_to_folder)
        row.addWidget(self._mask_apply_btn)

        # Find all similar regions button
        self._mask_find_all_btn = QPushButton("⧉ Find All Similar")
        self._mask_find_all_btn.setToolTip(
            "Draw mask around ONE example region, then click this.\n"
            "Finds all identical/similar regions in the image and masks them all."
        )
        self._mask_find_all_btn.setCheckable(False)
        self._mask_find_all_btn.setStyleSheet(
            "QPushButton { background:#0A1828; color:#5588AA; border:1px solid #1A2A3A; "
            "              padding:3px 10px; font-size:10px; }"
            "QPushButton:hover { color:#AACCDD; background:#111F2E; }"
        )
        self._mask_find_all_btn.clicked.connect(self._mask_find_all_similar)
        row.addWidget(self._mask_find_all_btn)

        # Save mask button
        self._mask_save_btn = QPushButton("💾 Save Mask")
        self._mask_save_btn.setToolTip(
            "Save the current mask.\n"
            "Choose: save mask position only, or export masked image file."
        )
        self._mask_save_btn.setCheckable(False)
        self._mask_save_btn.setStyleSheet(
            "QPushButton { background:#0A1828; color:#44AA66; border:1px solid #1A3A2A; "
            "              padding:3px 10px; font-size:10px; }"
            "QPushButton:hover { color:#88FFAA; background:#0A1E14; }"
        )
        self._mask_save_btn.clicked.connect(self._mask_save_dialog)
        row.addWidget(self._mask_save_btn)

        # Clear mask button
        self._mask_clear_btn = QPushButton("⬡ Clear Mask")
        self._mask_clear_btn.setToolTip("Remove inspection mask — analyze full image")
        self._mask_clear_btn.setCheckable(False)
        self._mask_clear_btn.setStyleSheet(
            "QPushButton { background:#0A1828; color:#664444; border:1px solid #2A1A1A; "
            "              padding:3px 10px; font-size:10px; }"
            "QPushButton:hover { color:#FF8888; background:#1A0808; }"
        )
        self._mask_clear_btn.clicked.connect(self._mask_clear)
        row.addWidget(self._mask_clear_btn)

        row.addSpacing(12)

        # Scale calibration button
        self._scale_btn = QPushButton("⚙ Set Scale…")
        self._scale_btn.setToolTip("Set mm/pixel calibration for Measure tool")
        self._scale_btn.setCheckable(False)
        self._scale_btn.setStyleSheet(
            "QPushButton { background:#0A1828; color:#6688AA; border:1px solid #1A2A3A; "
            "              padding:3px 10px; font-size:10px; }"
            "QPushButton:hover { color:#AACCDD; background:#111F2E; }"
        )
        self._scale_btn.clicked.connect(self._set_scale_calibration)
        row.addWidget(self._scale_btn)

        row.addSpacing(12)

        # Clear overlays button
        clear_btn = QPushButton("✕ Clear Overlays")
        clear_btn.setToolTip("Remove ROI, profile line, measure line, and all annotations")
        clear_btn.setCheckable(False)
        clear_btn.setStyleSheet(
            "QPushButton { background:#0A1828; color:#886666; border:1px solid #2A1A1A; "
            "              padding:3px 10px; font-size:10px; }"
            "QPushButton:hover { color:#FF8888; background:#1A1010; }"
        )
        clear_btn.clicked.connect(self._clear_inspect_overlays)
        row.addWidget(clear_btn)

        row.addStretch()

        # Save / load annotation hint
        ann_lbl = QLabel("Annotations auto-saved alongside image file")
        ann_lbl.setStyleSheet("color:#2A3A4A; font-size:9px;")
        row.addWidget(ann_lbl)

        # Wrap in scroll area — buttons scroll horizontally instead of forcing window wider
        scroll = QScrollArea()
        scroll.setWidget(bar)
        scroll.setWidgetResizable(False)
        scroll.setFixedHeight(36)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(0)
        scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        scroll.setVisible(False)
        self._inspect_toolbar_bar = bar   # keep ref for visibility toggle
        return scroll

    # ═══════════════════════════════════════════════════════════════
    #  Signal Wiring — everything connected to everything
    # ═══════════════════════════════════════════════════════════════

    def _connect_signals(self):
        # Viewer → status bar zoom
        self.viewer.zoom_changed.connect(self._on_zoom_changed)

        # Viewer → inspector pixel display (live on mouse move)
        self.viewer.pixel_hovered.connect(self._on_pixel_hovered)

        # Wire zoom + sync for ALL 4 cells (not just active)
        for i in range(4):
            v = self.multi_viewer.viewer_at(i)
            idx = i
            v.zoom_changed.connect(
                lambda zoom, _i=idx: self.multi_viewer.update_cell_zoom(_i, zoom)
            )
            v.view_state_changed.connect(
                lambda zoom, ox, oy, _i=idx: self._on_view_state_changed(_i, zoom, ox, oy)
            )

        # Browser → open image
        self.browser.image_selected.connect(self.open_image)

        # Pipeline → re-process and update viewer + 3D
        self.pipeline_panel.pipeline_changed.connect(self._on_pipeline_changed)

        # Fusion → send composite to viewer
        self.fusion_panel.composite_ready.connect(self._on_composite_ready)

        # Comparison panel → open image in main viewer
        self.comparison_panel.open_image_requested.connect(self.open_image)

        # Inspection tool signals → inspector panel updates
        self.viewer.roi_selected.connect(self._on_roi_selected)
        self.viewer.line_profile_drawn.connect(self._on_line_profile_drawn)
        self.viewer.annotation_placed.connect(self._on_annotation_placed)
        self.viewer.measure_done.connect(self._on_measure_done)

        # Inspector annotation clear button
        self.inspector._ann_clear_btn.clicked.connect(self._clear_annotations)

        # Mask signals
        self.viewer.mask_polygon_added.connect(self._on_mask_polygon_added)
        self.viewer.mask_cleared.connect(self._on_mask_cleared)

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
        self._action(m, "Save Processed Image…", self._menu_save,       "Ctrl+S")
        m.addSeparator()
        self._action(m, "Export Focus Report  (CSV)…",  self._export_csv,  "Ctrl+E")
        self._action(m, "Export Focus Report  (PDF)…",  self._export_pdf)
        m.addSeparator()
        self._action(m, "Batch Analyze Folder…",         self._batch_analyze, "Ctrl+B")
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
        self._action(m, "Toggle Focus Grid",     self._toggle_focus_grid,    "G")
        self._action(m, "Toggle Histogram",      self.inspector.toggle_histogram, "H")
        m.addSeparator()
        self._action(m, "Full Image View  (hide all panels)",
                     self._toggle_all_panels, "Tab")
        self._action(m, "Reset Default Layout",  self._reset_layout, "Ctrl+Shift+R")
        m.addSeparator()
        for name, dock in self._docks.items():
            m.addAction(dock.toggleViewAction())

        # ── Image navigation — app-wide shortcuts (work even when viewer has focus) ──
        for key, slot in [
            ("Left",  self._prev_image),
            ("Right", self._next_image),
            ("Home",  self._first_image),
            ("End",   self._last_image),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)

        # ── Pipeline ───────────────────────────────────────────────
        m = mb.addMenu("&Pipeline")
        self._action(m, "Clear Pipeline",   self._clear_pipeline)
        self._action(m, "Save Pipeline…",   self._save_pipeline,  "Ctrl+Shift+S")
        self._action(m, "Load Pipeline…",   self._load_pipeline,  "Ctrl+Shift+L")

        # ── Tools ──────────────────────────────────────────────────
        m = mb.addMenu("&Tools")
        self._action(m, "Illumination Fusion",
                     lambda: self._set_mode("Fusion"), "Ctrl+F")
        self._action(m, "3D Surface View",
                     lambda: self._set_mode("3D"),    "Ctrl+3")
        self._action(m, "Image Comparison",
                     self._open_comparison_window, "Ctrl+M")
        m.addSeparator()
        self._action(m, "Set Scale Calibration (mm/px)…", self._set_scale_calibration)
        self._action(m, "Clear Inspect Overlays",          self._clear_inspect_overlays)
        m.addSeparator()
        self._action(m, "Use Current Image as Reference",  self._set_auto_reference)
        self._action(m, "Lock Current Image as Reference (saved)", self._lock_reference)
        self._action(m, "Clear Focus Reference",           self._clear_reference)

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

    def _set_split(self, key: str):
        """Change the split layout (1 / 2H / 2V / 4)."""
        self.multi_viewer.set_layout(key)
        self._split_buttons[key].setChecked(True)
        self._rewire_active_viewer()
        # Re-display so GL context shows current image after layout change
        if self.current_image.is_loaded():
            self._display_current(preserve_view=True)

    def _on_active_cell_changed(self, idx: int):
        """User clicked a different cell — update viewer alias and restore its image."""
        self._rewire_active_viewer()
        data = self._cell_images[idx]
        if data is not None:
            self.current_image = data
            self._display_current(preserve_view=True)
            self._run_analysis()
            self._status_main.setText(
                f"  [{idx+1}] {data.filename}   {data.shape_str()}"
            )

    def _rewire_active_viewer(self):
        """Point self.viewer to the currently active GLImageViewer cell."""
        old = self.viewer
        new = self.multi_viewer.active_viewer
        if old is new:
            return
        # Re-connect signals from the new active viewer
        try:
            old.pixel_hovered.disconnect(self._on_pixel_hovered)
            old.zoom_changed.disconnect(self._on_zoom_changed)
            old.roi_selected.disconnect(self._on_roi_selected)
            old.line_profile_drawn.disconnect(self._on_line_profile_drawn)
            old.annotation_placed.disconnect(self._on_annotation_placed)
            old.measure_done.disconnect(self._on_measure_done)
            old.mask_polygon_added.disconnect(self._on_mask_polygon_added)
            old.mask_cleared.disconnect(self._on_mask_cleared)
        except Exception:
            pass
        self.viewer = new
        new.pixel_hovered.connect(self._on_pixel_hovered)
        new.zoom_changed.connect(self._on_zoom_changed)
        new.roi_selected.connect(self._on_roi_selected)
        new.line_profile_drawn.connect(self._on_line_profile_drawn)
        new.annotation_placed.connect(self._on_annotation_placed)
        new.measure_done.connect(self._on_measure_done)
        new.mask_polygon_added.connect(self._on_mask_polygon_added)
        new.mask_cleared.connect(self._on_mask_cleared)

    def _set_mode(self, mode: str):
        """Switch production workspaces without making users manage docks."""
        self._active_mode = mode
        for name, btn in self._mode_buttons.items():
            btn.setChecked(name == mode)

        self._dock_pipeline.setVisible(False)
        self._dock_fusion.setVisible(False)
        self._dock_focus.setVisible(False)

        # Inspect toolbar only in Inspect mode; switching away resets tool to navigate
        self._inspect_toolbar.setVisible(mode == "Inspect")
        if mode != "Inspect":
            self._set_inspect_tool("navigate")

        if mode == "Compare":
            self._workspace.setCurrentWidget(self.comparison_panel)
            self._dock_browser.setVisible(True)
            self._dock_inspector.setVisible(False)
        elif mode == "3D":
            self._workspace.setCurrentWidget(self.surface_3d)
            self._dock_browser.setVisible(False)
            self._dock_inspector.setVisible(False)
        else:
            self._workspace.setCurrentWidget(self._image_page)
            self._dock_browser.setVisible(mode in {"Inspect", "Focus"})
            self._dock_inspector.setVisible(mode in {"Inspect", "Focus", "Tune", "Fusion"})
            if mode == "Tune":
                self._dock_pipeline.setVisible(True)
                self._dock_pipeline.raise_()
            elif mode == "Fusion":
                self._dock_fusion.setVisible(True)
                self._dock_fusion.raise_()
            elif mode == "Focus":
                self._dock_focus.setVisible(True)
                self._dock_focus.raise_()
                self._update_ref_status()
                self._refresh_focus_assist()

        # Apply overlay settings to ALL visible viewers
        for i in range(self.multi_viewer.n_visible()):
            v = self.multi_viewer.viewer_at(i)
            v.set_focus_grid_visible(mode == "Focus")
            v.set_heatmap_visible(False)
        if mode == "Focus":
            ref = getattr(self, "_focus_reference", None)
            if ref is None:
                ref_tag = "⚠ RELATIVE mode — use 'Use as Ref' or 'Lock Ref' for absolute scoring"
            elif ref.mode == "locked":
                ref_tag = f"✓ LOCKED REF: {ref.source}"
            else:
                ref_tag = f"AUTO-REF: {ref.source}"
            self._mode_banner.setText(
                f"FOCUS MODE  ·  {ref_tag}  ·  GREEN=sharp  AMBER=soft  RED=blurry  ·  G=grid  F=heatmap"
            )
            self._mode_banner.setVisible(True)
        elif mode == "Inspect":
            self._mode_banner.setText("")
            self._mode_banner.setVisible(False)
        elif mode == "Tune":
            self._mode_banner.setText("TUNE MODE - filter controls are active, processed image stays in the center")
            self._mode_banner.setVisible(True)
        elif mode == "Fusion":
            self._mode_banner.setText("FUSION MODE - build multi-light composite, result shows in the center")
            self._mode_banner.setVisible(True)
        else:
            self._mode_banner.setVisible(False)

        self._status_main.setText(f"  {mode} mode")

    # ═══════════════════════════════════════════════════════════════
    #  Image Loading
    # ═══════════════════════════════════════════════════════════════

    def open_image(self, path: str):
        # Cancel any previous load
        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.quit()
            self._load_worker.wait()

        self._status_main.setText(f"  Loading  {os.path.basename(path)} …")
        self._status_verdict.setText("")

        self._load_worker = ImageLoadWorker(path)
        self._load_worker.loaded.connect(self._on_image_loaded)
        self._load_worker.failed.connect(self._on_image_failed)
        self._load_worker.start()

    def open_image_with_context(self, path: str):
        """Open an image and load its folder so navigation works."""
        folder = os.path.dirname(path)
        self.config.last_folder = folder
        self.folder_images = list_images_in_folder(folder)
        self.folder_index = self.folder_images.index(path) if path in self.folder_images else -1
        self.browser.set_folder(folder)
        self.open_image(path)

    def _on_image_loaded(self, data):
        self.current_image = data
        self._last_focus_result = None
        self._last_quality_result = None
        self.config.add_recent(data.path)
        self.config.save()
        self.browser.highlight_path(data.path)
        self.viewer.clear_tool_overlays()   # clear previous image's overlays
        self._load_annotations()            # restore saved annotations for this image
        self._load_mask()                   # restore saved mask for this image
        # Store in active cell slot
        self._cell_images[self.multi_viewer.active_idx] = data
        self.multi_viewer.set_header(self.multi_viewer.active_idx, data.filename)
        self._display_current()
        self._update_ref_status()
        self._run_analysis()
        self._status_main.setText(
            f"  {data.filename}   {data.shape_str()}   {self._image_position_text()}"
        )

    def _on_image_failed(self, error: str):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Load Error", error)
        self._status_main.setText("  Load failed")

    def _display_current(self, preserve_view: bool = False):
        if not self.current_image.is_loaded():
            return

        # Apply pipeline (CPU — only on visible region in future tile version)
        processed = self.pipeline.process(self.current_image.raw)
        self.current_image.display = processed

        # Update OpenGL viewer — GPU upload, instant display
        self.viewer.set_image(processed, preserve_view=preserve_view)

        # Update 3D surface with processed image
        self.surface_3d.set_image(processed)

        # Update histogram in inspector (masked if mask active)
        hist_data = self.quality_engine.compute_histogram(processed, mask=self._mask_data)
        self.inspector.update_histogram(hist_data)

    def _run_analysis(self):
        if not self.current_image.is_loaded():
            return
        self._set_focus_assist_analyzing()
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.quit()
            self._analysis_worker.wait()

        self._analysis_worker = AnalysisWorker(
            self.current_image.raw,
            self.focus_engine,
            self.quality_engine,
            self._focus_reference,
            self._mask_data,
        )
        self._analysis_worker.finished.connect(self._on_analysis_done)
        self._analysis_worker.start()

    # ═══════════════════════════════════════════════════════════════
    #  Signal Handlers
    # ═══════════════════════════════════════════════════════════════

    def _on_analysis_done(self, focus_result, quality_result):
        """Called from background thread — update all panels with results."""
        self._last_focus_result   = focus_result
        self._last_quality_result = quality_result

        # Session best tracking → builds auto-reference only from non-blurry images
        raw_lap = getattr(focus_result, "raw_lap", 0.0)
        # Minimum: image must be at least SOFT (absolute score >= 200, raw_lap >= 1000)
        # A blurry image as reference makes every cell score 100% against garbage — never promote.
        MIN_REF_LAP = self.focus_engine.SOFT_THRESHOLD * 5  # same scale as score = raw_lap/5
        if raw_lap > self._session_best_raw_lap:
            self._session_best_raw_lap = raw_lap
            if raw_lap >= MIN_REF_LAP and self.current_image.is_loaded():
                from src.analysis.focus_engine import FocusReference
                auto_ref = self.focus_engine.make_reference(
                    self.current_image.raw,
                    source=self.current_image.filename,
                    mode="auto",
                )
                # Only promote to active reference if no locked reference exists
                if self._focus_reference is None or \
                        getattr(self._focus_reference, "mode", "auto") == "auto":
                    self._focus_reference = auto_ref
                    self._update_ref_status()
                    self._reanalyze_with_reference(focus_result)

        # Store on image data
        self.current_image.focus_score    = focus_result.score
        self.current_image.focus_verdict  = focus_result.verdict
        self.current_image.focus_map      = focus_result.heatmap
        self.current_image.quality_score  = quality_result.overall_score
        self.current_image.quality_verdict = quality_result.verdict

        # Inspector: focus + quality metrics
        self.inspector.update_focus(focus_result, self.current_image.filename)
        self.inspector.update_quality(quality_result)

        # Viewer: focus heatmap + grid overlay
        heatmap_rgb = self.focus_engine.heatmap_to_rgb(focus_result.heatmap)
        self.viewer.set_heatmap(heatmap_rgb)
        self.viewer.set_focus_grid(focus_result.grid)
        # Ensure grid/heatmap visibility matches current mode for this viewer
        if self._active_mode == "Focus":
            self.viewer.set_focus_grid_visible(True)

        # Update active cell header badge with score
        verdict_sym = {"SHARP": "🟢", "SOFT": "🟡", "BLURRY": "🔴"}.get(
            focus_result.verdict, ""
        )
        badge = f"{verdict_sym} F:{focus_result.score:.0f}  Q:{quality_result.overall_score:.0f}"
        self.multi_viewer.set_header(
            self.multi_viewer.active_idx,
            self.current_image.filename,
            badge,
        )

        # Status bar: color-coded verdict
        self._update_status_verdict(focus_result, quality_result)
        self._update_focus_assist(focus_result, quality_result)

    def _on_pipeline_changed(self):
        """Pipeline layer added/removed/changed → re-process and update all views."""
        self._display_current(preserve_view=True)

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

    def _on_view_state_changed(self, source_idx: int, zoom: float, ox: float, oy: float):
        """Viewer pan/zoom changed — sync other panels if sync is enabled."""
        n = self.multi_viewer.n_visible()
        if n < 2 or not self._btn_sync_zoom.isChecked():
            return
        for i in range(n):
            if i != source_idx:
                v = self.multi_viewer.viewer_at(i)
                v.blockSignals(True)
                v.set_view_state(zoom, ox, oy)
                v.blockSignals(False)
                self.multi_viewer.update_cell_zoom(i, zoom)

    def _update_status_verdict(self, focus_result, quality_result):
        f = focus_result
        q = quality_result
        fcolor = VERDICT_COLOR.get(f.verdict, "#888")
        qcolor = COLOR_PERFECT if q.verdict == "PASS" else COLOR_FAIL

        self._status_main.setText(
            f"  {self.current_image.filename}   "
            f"{self.current_image.shape_str()}   {self._image_position_text()}   │   "
            f"Focus: {f.score:.0f}   │   "
            f"Quality: {q.overall_score:.0f}/100"
        )
        self._status_verdict.setText(
            f"  {f.verdict}  ·  {q.verdict}  "
        )
        self._status_verdict.setStyleSheet(
            f"color: {fcolor}; padding: 0 10px; font-weight: 700; font-size: 11px;"
        )

    def _image_position_text(self) -> str:
        if self.folder_index >= 0 and self.folder_images:
            return f"Image {self.folder_index + 1} / {len(self.folder_images)}"
        return ""

    def _toggle_focus_grid(self):
        self.viewer._show_focus_grid = not self.viewer._show_focus_grid
        if self.viewer._show_focus_grid:
            self.viewer._show_heatmap = False
        self.viewer.update()

    # ── Inspect tool handlers ──────────────────────────────────────

    def _set_inspect_tool(self, tool: str):
        self.viewer.set_tool(tool)
        if tool in self._tool_buttons:
            self._tool_buttons[tool].setChecked(True)
        if tool == "annotate":
            self.inspector.show_annotation_tools(True)
        self._status_main.setText(f"  Tool: {tool.upper()}  —  "
                                   + self._tool_hint(tool))

    def _tool_hint(self, tool: str) -> str:
        return {
            "navigate": "left drag = pan  |  right drag = zoom  |  scroll = zoom",
            "roi":      "left drag to draw region  |  right drag = pan",
            "profile":  "left drag to draw line  |  right drag = pan",
            "annotate": "left click to place marker  |  right drag = pan",
            "measure":  "left drag to measure distance  |  right drag = pan",
        }.get(tool, "")

    def _on_roi_selected(self, ix1: int, iy1: int, ix2: int, iy2: int):
        if self.current_image.raw is None:
            return
        self.inspector.update_roi_stats(self.current_image.raw, ix1, iy1, ix2, iy2)
        self._dock_inspector.raise_()
        self._dock_inspector.setVisible(True)
        self.inspector.scroll_to(self.inspector._roi_group)

    def _on_line_profile_drawn(self, ix1: int, iy1: int, ix2: int, iy2: int):
        if self.current_image.raw is None:
            return
        self.inspector.update_line_profile(self.current_image.raw, ix1, iy1, ix2, iy2)
        self._dock_inspector.raise_()
        self._dock_inspector.setVisible(True)
        self.inspector.scroll_to(self.inspector._profile_group)

    def _on_annotation_placed(self, ix: int, iy: int):
        labels = ["Scratch", "Pit", "Contamination", "Burr", "Crack", "OK", "Other"]
        from PyQt6.QtWidgets import QInputDialog
        label, ok = QInputDialog.getItem(
            self, "Defect Label",
            f"Label for annotation at ({ix}, {iy}):",
            labels, 0, False
        )
        if not ok:
            return
        self.viewer.add_annotation(ix, iy, label)
        self.inspector.refresh_annotations(self.viewer.get_annotations())
        self._save_annotations()
        self._dock_inspector.raise_()
        self._dock_inspector.setVisible(True)
        self.inspector.scroll_to(self.inspector._ann_group)

    def _on_measure_done(self, ix1: int, iy1: int, ix2: int, iy2: int):
        self.inspector.update_measurement(ix1, iy1, ix2, iy2, self._mm_per_px)
        self._dock_inspector.raise_()
        self._dock_inspector.setVisible(True)
        self.inspector.scroll_to(self.inspector._measure_group)

    def _set_scale_calibration(self):
        from PyQt6.QtWidgets import QInputDialog
        current = self._mm_per_px if self._mm_per_px > 0 else 0.001
        val, ok = QInputDialog.getDouble(
            self, "Set Scale Calibration",
            "Enter mm per pixel (e.g. 0.00625 means 1px = 6.25 µm):",
            current, 0.0000001, 100.0, 8
        )
        if ok and val > 0:
            self._mm_per_px = val
            self.viewer.set_calibration(val)
            self.inspector.set_calibration_label(val)
            self._status_main.setText(f"  Scale set: {val:.8f} mm/px")

    def _clear_inspect_overlays(self):
        self.viewer.clear_tool_overlays()
        self.viewer.clear_annotations()
        self.inspector.refresh_annotations([])
        self._save_annotations()
        self._status_main.setText("  Overlays and annotations cleared")

    def _clear_annotations(self):
        self.viewer.clear_annotations()
        self.inspector.refresh_annotations([])
        self._save_annotations()

    def _save_annotations(self):
        """Save annotations to JSON file alongside the image."""
        if not self.current_image.path:
            return
        import json
        ann_path = self.current_image.path + ".annotations.json"
        anns = self.viewer.get_annotations()
        try:
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(anns, f, indent=2)
        except Exception:
            pass

    def _load_annotations(self):
        """Load annotations from JSON file alongside the image, if it exists."""
        if not self.current_image.path:
            return
        import json
        ann_path = self.current_image.path + ".annotations.json"
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                anns = json.load(f)
            self.viewer.set_annotations(anns)
            self.inspector.refresh_annotations(anns)
        except FileNotFoundError:
            self.viewer.clear_annotations()
            self.inspector.refresh_annotations([])
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    #  Reference Management
    # ═══════════════════════════════════════════════════════════════

    def _load_locked_reference(self):
        """Try to restore a locked reference from the previous session."""
        try:
            ref = __import__("src.analysis.focus_engine", fromlist=["FocusReference"]).FocusReference.load(self._ref_path)
            if ref.mode == "locked":
                self._focus_reference = ref
        except FileNotFoundError:
            pass
        except Exception:
            pass   # corrupt file — ignore silently

    def _reanalyze_with_reference(self, _prev=None):
        """Re-run analysis with the current reference (called after reference changes)."""
        if self.current_image.is_loaded():
            self._run_analysis()

    def _set_auto_reference(self):
        """Use current image as the auto-reference for this session."""
        if not self.current_image.is_loaded():
            return
        ref = self.focus_engine.make_reference(
            self.current_image.raw,
            source=self.current_image.filename,
            mode="auto",
        )
        self._focus_reference = ref
        self._session_best_raw_lap = getattr(
            self._last_focus_result, "raw_lap", self._session_best_raw_lap
        ) if self._last_focus_result else self._session_best_raw_lap
        self._update_ref_status()
        self._reanalyze_with_reference()
        self._status_main.setText(
            f"  Reference set to: {self.current_image.filename} (AUTO)"
        )

    def _lock_reference(self):
        """Lock current image as permanent reference — saves to disk, survives restarts."""
        if not self.current_image.is_loaded():
            return
        ref = self.focus_engine.make_reference(
            self.current_image.raw,
            source=self.current_image.filename,
            mode="locked",
        )
        self._focus_reference = ref
        try:
            ref.save(self._ref_path)
            msg = f"  Locked reference saved: {self.current_image.filename}"
        except Exception as e:
            msg = f"  Lock saved in memory (disk write failed: {e})"
        self._update_ref_status()
        self._reanalyze_with_reference()
        self._status_main.setText(msg)

    def _clear_reference(self):
        """Remove reference — return to RELATIVE scoring mode."""
        self._focus_reference = None
        self._session_best_raw_lap = 0.0
        import os as _os
        try:
            if _os.path.exists(self._ref_path):
                _os.remove(self._ref_path)
        except Exception:
            pass
        self._update_ref_status()
        self._reanalyze_with_reference()
        self._status_main.setText("  Reference cleared — RELATIVE mode")

    def _update_ref_status(self):
        """Update the reference status label and button states in Focus Assist dock."""
        ref = self._focus_reference
        has_image = self.current_image.is_loaded()

        # Button enable states
        self._btn_set_ref.setEnabled(has_image)
        self._btn_lock_ref.setEnabled(has_image)
        self._btn_clear_ref.setEnabled(ref is not None)

        if ref is None:
            self._ref_status_lbl.setText("No reference  —  RELATIVE mode")
            self._ref_status_lbl.setStyleSheet(
                "color:#886633; font-size:9px; font-weight:600; padding:2px;"
            )
        elif ref.mode == "locked":
            ts = ref.locked_at[:10] if ref.locked_at else "?"
            self._ref_status_lbl.setText(
                f"✓ LOCKED REF  {ref.source}  ({ts})"
            )
            self._ref_status_lbl.setStyleSheet(
                "color:#00AA66; font-size:9px; font-weight:600; padding:2px;"
            )
        else:
            self._ref_status_lbl.setText(
                f"AUTO-REF  {ref.source}  (session best)"
            )
            self._ref_status_lbl.setStyleSheet(
                "color:#5599BB; font-size:9px; font-weight:600; padding:2px;"
            )

    # ═══════════════════════════════════════════════════════════════
    #  Mask System
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _polygons_to_rects(polygons: list) -> list:
        """Convert 4-corner polygon data (MaskData format) back to (ix1,iy1,ix2,iy2) tuples."""
        rects = []
        for poly in polygons:
            if not poly:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            rects.append((min(xs), min(ys), max(xs), max(ys)))
        return rects

    def _mask_path(self, image_path: str = None) -> str:
        p = image_path or (self.current_image.path or "")
        return p + ".mask.json" if p else ""

    def _save_mask(self):
        if not self.current_image.path or self._mask_data is None:
            return
        try:
            self._mask_data.save(self._mask_path())
        except Exception:
            pass

    def _load_mask(self):
        """Load mask from disk for the current image. Auto-align if from a different image."""
        if not self.current_image.path:
            return
        from src.analysis.mask_engine import MaskData
        path = self._mask_path()
        try:
            mask = MaskData.load(path)
            self._mask_data = mask
            self.viewer.set_mask_polygons(mask.polygons)
            self.inspector.update_mask_status(mask)
        except FileNotFoundError:
            # No mask file — if we have a mask from a previous image (fixed camera),
            # carry it forward as-is (same camera position assumed)
            if self._mask_data is not None and self.current_image.is_loaded():
                self.viewer.set_mask_polygons(self._mask_data.polygons)
                self.inspector.update_mask_status(self._mask_data)
            else:
                self._mask_data = None
                self.viewer.set_mask_polygons([])
                self.inspector.update_mask_status(None)
        except Exception:
            pass

    def _on_mask_polygon_added(self, polygon: list):
        """New mask polygon closed — add to MaskData and re-analyze."""
        from src.analysis.mask_engine import MaskData
        H, W = self.current_image.raw.shape[:2] if self.current_image.is_loaded() else (0, 0)
        if self._mask_data is None:
            self._mask_data = MaskData(
                polygons     = [polygon],
                image_shape  = (H, W),
                source_image = self.current_image.filename,
            )
        else:
            self._mask_data.polygons.append(polygon)
            self._mask_data.image_shape  = (H, W)
            self._mask_data.source_image = self.current_image.filename
        # auto-save removed — user must click Save Mask
        self.inspector.update_mask_status(self._mask_data)
        self._run_analysis()
        pct = self._mask_data.coverage_pct()
        self._status_main.setText(
            f"  Mask region added — {len(self._mask_data.polygons)} excluded region(s), "
            f"{pct:.1f}% of image excluded"
        )

    def _mask_save_dialog(self):
        """Ask user HOW to save the mask — position only, or also export masked image."""
        if self._mask_data is None or not self._mask_data.polygons:
            QMessageBox.information(self, "Save Mask",
                                    "No mask is active. Draw a mask first.")
            return
        if not self.current_image.is_loaded():
            QMessageBox.information(self, "Save Mask", "No image loaded.")
            return

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle("Save Mask")
        dlg.setStyleSheet("background:#0C1520; color:#AABBCC;")
        lay = QVBoxLayout(dlg)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 16, 20, 16)

        info = QLabel(
            f"Image: <b>{self.current_image.filename}</b><br>"
            f"Excluded regions: {len(self._mask_data.polygons)}<br>"
            f"Excluded area: {self._mask_data.coverage_pct():.1f}% of image"
        )
        info.setStyleSheet("color:#AABBCC; font-size:11px;")
        info.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(info)

        cb_position = QCheckBox("Save mask position  (.mask.json)")
        cb_position.setChecked(True)
        cb_position.setToolTip(
            "Saves the polygon positions alongside the image file.\n"
            "Next time you open this image the mask reloads automatically."
        )
        cb_position.setStyleSheet("color:#AACCFF; font-size:11px;")
        lay.addWidget(cb_position)

        cb_image = QCheckBox("Export masked image  (excluded area = black pixels)")
        cb_image.setChecked(False)
        cb_image.setToolTip(
            "Saves a new PNG file with the excluded region replaced by black pixels.\n"
            "Saved as:  original_name_masked.png"
        )
        cb_image.setStyleSheet("color:#AACCFF; font-size:11px;")
        lay.addWidget(cb_image)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        saved = []
        if cb_position.isChecked():
            self._save_mask()
            saved.append("mask position (.mask.json)")
        if cb_image.isChecked():
            out_path = self._mask_save_image()
            if out_path:
                saved.append(f"masked image → {os.path.basename(out_path)}")

        if saved:
            self._status_main.setText("  Saved: " + "  +  ".join(saved))
        else:
            self._status_main.setText("  Nothing saved.")

    def _mask_save_image(self) -> str:
        """Export current image with excluded (masked) pixels set to black. Returns output path."""
        if not self.current_image.is_loaded() or self._mask_data is None:
            return ""
        img = self.current_image.raw.copy()
        H, W = img.shape[:2]
        mask_arr = self._mask_data.to_array((H, W))   # 1=valid, 0=excluded

        # Black out excluded pixels
        if img.ndim == 3:
            img[mask_arr == 0] = 0
        else:
            img[mask_arr == 0] = 0

        # Build output path: same folder, _masked suffix, always PNG
        base, _ = os.path.splitext(self.current_image.path)
        out_path = base + "_masked.png"

        # If file exists, add counter
        counter = 1
        while os.path.exists(out_path):
            out_path = base + f"_masked_{counter}.png"
            counter += 1

        # Save — cv2 expects BGR
        if img.ndim == 3:
            save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            save_img = img
        cv2.imwrite(out_path, save_img)
        return out_path

    def _on_mask_cleared(self):
        self._mask_data = None
        self.inspector.update_mask_status(None)
        self._save_mask()   # removes file — intentional on clear
        self._run_analysis()

    def _mask_clear(self):
        self.viewer.clear_mask()
        # Also delete the mask file
        import os as _os
        p = self._mask_path()
        if p and _os.path.exists(p):
            try:
                _os.remove(p)
            except Exception:
                pass

    def _mask_find_all_similar(self):
        """
        Take the FIRST drawn polygon as a template and find all identical
        regions in the current image. Replaces the mask with all found regions.
        """
        if not self.current_image.is_loaded():
            return
        if self._mask_data is None or not self._mask_data.polygons:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Find All Similar",
                "Draw a mask around ONE example region first,\n"
                "then click 'Find All Similar'."
            )
            return

        from src.analysis.mask_engine import MaskEngine, MaskData
        template_poly = self._mask_data.polygons[0]   # use first drawn polygon
        self._status_main.setText("  Searching for similar regions…")

        try:
            found = MaskEngine.find_similar_regions(
                self.current_image.raw, template_poly
            )
        except Exception as e:
            self._status_main.setText(f"  Find similar failed: {e}")
            return

        H, W = self.current_image.raw.shape[:2]
        self._mask_data = MaskData(
            polygons     = found,
            image_shape  = (H, W),
            source_image = self.current_image.filename,
        )
        self.viewer.set_mask_polygons(found)
        self.inspector.update_mask_status(self._mask_data)
        # auto-save removed — user must click Save Mask
        self._run_analysis()
        self._status_main.setText(
            f"  Found {len(found)} similar region(s) — all masked"
        )

    def _mask_auto_detect(self):
        """Phase 2: Auto-detect specular reflections and suggest inspection mask."""
        if not self.current_image.is_loaded():
            return
        from src.analysis.mask_engine import MaskEngine
        self._status_main.setText("  Detecting reflections…")
        mask = MaskEngine.detect_reflections(self.current_image.raw)
        if mask is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Auto-Detect",
                "No significant specular reflections detected.\n\n"
                "The image appears clean. You can draw a manual mask if needed."
            )
            self._status_main.setText("  No reflections detected")
            return
        mask.source_image = self.current_image.filename
        self._mask_data = mask
        self.viewer.set_mask_polygons(mask.polygons)
        self.inspector.update_mask_status(mask)
        # auto-save removed — user must click Save Mask
        self._run_analysis()
        pct = mask.coverage_pct()
        self._status_main.setText(
            f"  Auto-detected inspection region: {pct:.1f}% of image "
            f"(reflections excluded)"
        )

    def _mask_auto_align(self):
        """Phase 3: Auto-align mask using ORB feature matching + RANSAC."""
        if self._mask_data is None or not self.current_image.is_loaded():
            return
        from src.analysis.mask_engine import MaskEngine
        ref_src = self._mask_data.source_image
        # Load the reference image to use for alignment
        ref_path = ""
        if self.folder_images:
            ref_path = next(
                (p for p in self.folder_images
                 if __import__("os").path.basename(p) == ref_src), ""
            )
        if not ref_path or not __import__("os").path.exists(ref_path):
            # Cannot load reference — use mask as-is (fixed camera assumption)
            self.viewer.set_mask_polygons(self._mask_data.polygons)
            self._status_main.setText(
                f"  Mask from {ref_src} applied directly (reference not found for alignment)"
            )
            return

        try:
            from src.core.image_loader import load_image
            ref_data = load_image(ref_path)
            aligned, confidence = MaskEngine.align_mask(
                ref_data.raw, self.current_image.raw, self._mask_data
            )
            aligned.source_image = self._mask_data.source_image

            conf_label = (
                "HIGH" if confidence >= 80 else
                "MEDIUM" if confidence >= 50 else "LOW"
            )

            n_inliers  = getattr(aligned, "n_match_inliers", 0)
            conf_label = ("HIGH" if confidence >= 80 else
                          "MEDIUM" if confidence >= 50 else "LOW")

            if confidence >= 50:
                self._mask_data = aligned
                self.viewer.set_mask_polygons(aligned.polygons)
                self.inspector.update_mask_status(aligned)
                # auto-save removed
                self._run_analysis()
                self._status_main.setText(
                    f"  Mask aligned — {n_inliers} feature matches, "
                    f"confidence: {confidence:.0f}% ({conf_label})"
                )
            else:
                from PyQt6.QtWidgets import QMessageBox
                btn = QMessageBox.question(
                    self, "Mask Alignment — Low Confidence",
                    f"Only {n_inliers} matching features found "
                    f"(confidence: {confidence:.0f}%).\n\n"
                    f"The images may be very different or the part moved a lot.\n\n"
                    f"Apply the aligned mask anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if btn == QMessageBox.StandardButton.Yes:
                    self._mask_data = aligned
                    self.viewer.set_mask_polygons(aligned.polygons)
                    self.inspector.update_mask_status(aligned)
                    # auto-save removed
                    self._run_analysis()
                else:
                    self._mask_data = None
                    self.viewer.set_mask_polygons([])
                    self.inspector.update_mask_status(None)
        except RuntimeError as e:
            self._status_main.setText(f"  Feature matching failed: {e}")
        except Exception as e:
            self._status_main.setText(f"  Mask alignment error: {e}")

    def _mask_apply_to_folder(self):
        """Apply current mask to all images in the folder (with auto-alignment)."""
        if self._mask_data is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Apply Mask to Folder",
                "No mask is active. Draw a mask on this image first,\n"
                "then click 'Apply to Folder'."
            )
            return
        if not self.folder_images:
            return

        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt as _Qt
        from src.analysis.mask_engine import MaskEngine
        from src.core.image_loader import load_image
        import os as _os

        curr_path = self.current_image.path
        others    = [p for p in self.folder_images if p != curr_path]
        if not others:
            QMessageBox.information(self, "Apply Mask", "No other images in folder.")
            return

        progress = QProgressDialog(
            "Aligning mask to folder images…", "Cancel", 0, len(others), self
        )
        progress.setWindowModality(_Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        results = []
        for i, path in enumerate(others):
            progress.setValue(i)
            if progress.wasCanceled():
                break
            try:
                data    = load_image(path)
                aligned, conf = MaskEngine.align_mask(
                    self.current_image.raw, data.raw, self._mask_data
                )
                aligned.source_image = self._mask_data.source_image
                aligned.save(self._mask_path(path))
                results.append((
                    _os.path.basename(path),
                    conf,
                    "HIGH" if conf >= 80 else "MEDIUM" if conf >= 50 else "LOW"
                ))
            except Exception as e:
                results.append((_os.path.basename(path), 0.0, f"ERROR: {e}"))

        progress.setValue(len(others))

        # Summary report
        lines = [f"Mask applied to {len(results)} images:\n"]
        for name, conf, label in results:
            lines.append(f"  {label:6s}  {conf:5.0f}%  {name}")
        low_conf = [r for r in results if r[2] == "LOW"]
        if low_conf:
            lines.append(f"\n⚠ {len(low_conf)} image(s) have LOW confidence — verify manually.")
        QMessageBox.information(self, "Apply Mask — Done", "\n".join(lines))
        self._status_main.setText(
            f"  Mask applied to {len(results)} images in folder"
        )

    def _refresh_focus_assist(self):
        if self._last_focus_result is not None and self._last_quality_result is not None:
            self._update_focus_assist(self._last_focus_result, self._last_quality_result)
        elif self.current_image.is_loaded():
            self._set_focus_assist_analyzing()
        else:
            self._focus_assist.setText(
                "FOCUS ASSIST\n\n"
                "Open an image to analyze focus.\n\n"
                "After loading, this panel shows the real focus verdict, score, "
                "warnings, and operator action."
            )

    def _set_focus_assist_analyzing(self):
        if not hasattr(self, "_focus_assist"):
            return
        name = self.current_image.filename or "image"
        self._focus_assist.setText(
            "FOCUS ASSIST\n\n"
            f"Analyzing {name}...\n\n"
            "Computing focus score, quality score, and heatmap.\n"
            "Large industrial images can take a few seconds."
        )

    def _update_focus_assist(self, focus_result, quality_result):
        v  = focus_result.verdict
        g  = focus_result.grid
        q  = quality_result.overall_score

        # Scoring mode header
        mode = getattr(focus_result, "scoring_mode", "RELATIVE")
        conf = getattr(focus_result, "confidence",   "LOW")
        ref_pct = getattr(focus_result, "ref_pct",   0.0)
        r_src   = getattr(focus_result, "ref_source", "")
        is_self_ref = (mode != "RELATIVE" and r_src
                       and self.current_image.filename == r_src)
        if is_self_ref:
            mode_line = (
                f"⚠ THIS IS THE REFERENCE IMAGE ({r_src})\n"
                "   Score vs itself = 100% — meaningless. Trust the Verdict + Score above."
            )
        elif mode == "LOCKED_REF":
            mode_line = f"MODE: LOCKED REF ({r_src})  ·  {ref_pct:.1f}% of reference"
        elif mode == "AUTO_REF":
            mode_line = f"MODE: AUTO-REF ({r_src})  ·  {ref_pct:.1f}% of session best"
        else:
            mode_line = "MODE: RELATIVE  ⚠  Cannot confirm absolute sharpness"

        # Grid statistics block
        grid_block = (
            f"GRID ANALYSIS  ({g.rows}×{g.cols} = {g.rows*g.cols} cells)\n"
            f"  SHARP  (≥72%)  :  {g.pct_sharp:.0f}% of cells\n"
            f"  SOFT   (38-72%):  {g.pct_soft:.0f}% of cells\n"
            f"  BLURRY (<38%)  :  {g.pct_blurry:.0f}% of cells\n"
            f"  Best  cell : row {g.best_cell[0]+1}, col {g.best_cell[1]+1}  "
            f"({g.scores[g.best_cell]:.0f}%)\n"
            f"  Worst cell : row {g.worst_cell[0]+1}, col {g.worst_cell[1]+1}  "
            f"({g.scores[g.worst_cell]:.0f}%)"
        )

        # Tilt warning
        tilt_block = ""
        if g.tilt_warn:
            tilt_block = f"\n\nTILT DETECTED\n{g.tilt_warn}\nAdjust part fixture or camera angle."

        # Verdict action
        if v in {"PERFECT", "GOOD"}:
            action = (
                "ACTION\n"
                "Focus is usable. Lock lens settings.\n"
                "Use this as reference for comparison captures."
            )
        elif v == "SOFT":
            action = (
                "ACTION\n"
                "Image is soft. Re-tune lens focus ring,\n"
                "reduce exposure time or vibration,\n"
                "or move camera closer to the sharp region."
            )
        else:
            action = (
                "ACTION\n"
                "Blurry — DO NOT use for defect inspection.\n"
                "Fix lens focus, increase lighting, reduce\n"
                "motion blur before accepting this image."
            )

        # Quality conflict
        conflict = ""
        if v in {"SOFT", "BLURRY"} and quality_result.verdict == "PASS":
            conflict = (
                "\n\nCONFLICT: Quality=PASS but Focus=WEAK\n"
                "Exposure/contrast look OK but sharpness\n"
                "is insufficient for edge/scratch detection.\n"
                "For AI defect models — trust focus score."
            )

        self._focus_assist.setText(
            f"FOCUS ASSIST\n\n"
            f"Verdict : {v}   [{conf} confidence]\n"
            f"Score   : {focus_result.score:.0f}  (Laplacian+Tenengrad fusion)\n"
            f"Quality : {q:.0f}/100  ({quality_result.verdict})\n"
            f"{mode_line}\n\n"
            f"{grid_block}"
            f"{tilt_block}\n\n"
            f"{action}"
            f"{conflict}\n\n"
            f"GRID KEY\n"
            f"GREEN  = SHARP (≥72%)  — good for inspection\n"
            f"AMBER  = SOFT  (38-72%) — marginal\n"
            f"RED    = BLURRY (<38%) — reject / refocus\n"
            f"[G] toggle grid  [F] toggle heatmap"
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
        self.open_image_with_context(path)

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

    def _open_comparison_window(self):
        """Open comparison as a center workspace."""
        self._set_mode("Compare")

    def _toggle_all_panels(self):
        """Tab — hide all panels for full image view. Tab again — restore all."""
        any_visible = any(d.isVisible() for d in self._docks.values())
        if any_visible:
            # Save which docks were visible, then hide all
            self._panels_hidden_state = {k: d.isVisible() for k, d in self._docks.items()}
            for dock in self._docks.values():
                dock.setVisible(False)
            self._status_main.setText(
                "  Full image view  —  Press Tab to restore panels"
            )
        else:
            # Restore previous visibility
            state = getattr(self, "_panels_hidden_state", {})
            for name, dock in self._docks.items():
                dock.setVisible(state.get(name, True))
            self._status_main.setText("")

    def _reset_layout(self):
        """Restore production image-first layout."""
        # Make all docks visible first
        for dock in self._docks.values():
            dock.setVisible(True)
            dock.setFloating(False)

        # Re-attach docks to their correct areas
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,   self._dock_browser)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,  self._dock_inspector)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,  self._dock_focus)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_pipeline)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_fusion)

        # Re-tab right docks and bottom docks
        self.tabifyDockWidget(self._dock_inspector, self._dock_focus)
        self.tabifyDockWidget(self._dock_pipeline, self._dock_fusion)

        # Raise default tabs
        self._dock_inspector.raise_()
        self._dock_pipeline.raise_()
        self._set_mode("Inspect")

        # Re-apply sizes after Qt processes the layout
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(50, self._set_dock_sizes)

    def _add_to_comparison(self):
        if self.current_image.path:
            self.comparison_panel.add_image_from_path(self.current_image.path)
            self._set_mode("Compare")

    def _export_csv(self):
        if not self.current_image.is_loaded() or self._last_focus_result is None:
            QMessageBox.information(self, "Export",
                "Open an image and wait for analysis to complete first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV Report", "", "CSV (*.csv)")
        if not path:
            return
        from src.export.report_exporter import ImageRecord, export_csv
        record = ImageRecord.from_analysis(
            self.current_image, self._last_focus_result, self._last_quality_result)
        export_csv([record], path)
        self._status_main.setText(f"  CSV exported: {path}")
        QMessageBox.information(self, "Exported",
            f"CSV report saved to:\n{path}\n\n"
            f"Decision: {record.overall_decision()}\n"
            f"Focus: {record.focus_verdict}  |  "
            f"Sharp: {record.pct_sharp:.0f}%  Soft: {record.pct_soft:.0f}%  "
            f"Blurry: {record.pct_blurry:.0f}%")

    def _export_pdf(self):
        if not self.current_image.is_loaded() or self._last_focus_result is None:
            QMessageBox.information(self, "Export",
                "Open an image and wait for analysis to complete first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF Report", "", "PDF (*.pdf);;HTML (*.html)")
        if not path:
            return
        from src.export.report_exporter import ImageRecord, export_pdf
        record = ImageRecord.from_analysis(
            self.current_image, self._last_focus_result, self._last_quality_result)
        result = export_pdf([record], path)
        self._status_main.setText(f"  Report saved: {result}")
        QMessageBox.information(self, "Exported", f"Report saved to:\n{result}")

    def _batch_analyze(self):
        from src.ui.dialogs.batch_dialog import BatchDialog
        dlg = BatchDialog(self, self.focus_engine, self.quality_engine)
        dlg.exec()

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
        primary = QApplication.primaryScreen()
        ag = primary.availableGeometry()   # excludes taskbar

        w = self.config.window_width
        h = self.config.window_height

        if self.config.window_x == -1:
            # First launch — centre on primary monitor at a comfortable size
            w = min(w, int(ag.width()  * 0.80))
            h = min(h, int(ag.height() * 0.80))
            x = ag.x() + (ag.width()  - w) // 2
            y = ag.y() + (ag.height() - h) // 2
        else:
            x = self.config.window_x
            y = self.config.window_y
            # Guard: make sure it's still on a visible screen
            visible = any(
                s.availableGeometry().contains(x + w // 2, y + h // 2)
                for s in QApplication.screens()
            )
            if not visible:
                x = ag.x() + (ag.width()  - w) // 2
                y = ag.y() + (ag.height() - h) // 2

        self.setGeometry(x, y, w, h)
        if self.config.window_maximized:
            self.showMaximized()
        else:
            self.show()
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._set_dock_sizes)

    def _set_dock_sizes(self):
        """Set initial dock proportions after the window is fully shown."""
        h = self.height()
        w = self.width()
        bottom_h = max(160, int(h * 0.18))   # small strip — image gets most space
        left_w   = max(180, int(w * 0.13))
        right_w  = max(240, int(w * 0.17))

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
        if not self.isMaximized():
            self.config.window_width  = self.width()
            self.config.window_height = self.height()
            self.config.window_x      = self.x()
            self.config.window_y      = self.y()
        self.config.window_maximized = self.isMaximized()
        self.config.save()
        super().closeEvent(event)
