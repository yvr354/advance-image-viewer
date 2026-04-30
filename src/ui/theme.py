"""
Professional dark instrument theme.
Designed to feel like a high-end scientific instrument — not a Windows app.
Color language: electric blue accents, traffic-light metrics, near-black backgrounds.
"""

# ── Palette ────────────────────────────────────────────────────────────────
BG_DEEP     = "#0A0A0F"   # deepest background — window, panels
BG_BASE     = "#0F0F18"   # base surface
BG_RAISED   = "#15151F"   # raised card / group box
BG_CONTROL  = "#1C1C2A"   # inputs, buttons, sliders
BG_HOVER    = "#252535"   # hovered control
BG_ACTIVE   = "#2E2E45"   # active / pressed control
BG_SELECTED = "#1A2A4A"   # selected item

BORDER_DIM    = "#252535"
BORDER_NORMAL = "#35354A"
BORDER_BRIGHT = "#4A4A6A"

ACCENT       = "#00B4D8"  # electric blue — primary accent
ACCENT_DARK  = "#0077A8"
ACCENT_GLOW  = "#00D4F8"

TEXT_PRIMARY   = "#E8E8F0"
TEXT_SECONDARY = "#8888AA"
TEXT_DIM       = "#44445A"
TEXT_ACCENT    = "#00B4D8"

# Metric colors — traffic light system
COLOR_PERFECT = "#00E676"   # green  — perfect / pass
COLOR_GOOD    = "#69F0AE"   # light green
COLOR_WARN    = "#FF6D00"   # orange — soft / warning
COLOR_FAIL    = "#FF1744"   # red    — fail / blurry
COLOR_INFO    = "#40C4FF"   # blue   — info


def get_stylesheet() -> str:
    return f"""
/* ═══════════════════════════════════════════════════════
   YVR Advanced Image Viewer — Professional Dark Theme
   ═══════════════════════════════════════════════════════ */

/* ── Base ─────────────────────────────────────────────── */
QMainWindow, QDialog, QWidget {{
    background-color: {BG_DEEP};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Inter", Arial;
    font-size: 11px;
}}

QSplitter {{
    background: {BG_DEEP};
}}
QSplitter::handle {{
    background: {BORDER_DIM};
    width: 2px;
    height: 2px;
}}
QSplitter::handle:hover {{
    background: {ACCENT};
}}

/* ── Menu bar ─────────────────────────────────────────── */
QMenuBar {{
    background-color: {BG_BASE};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {BORDER_DIM};
    padding: 2px 4px;
    spacing: 2px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: 3px;
}}
QMenuBar::item:selected, QMenuBar::item:pressed {{
    background: {BG_CONTROL};
    color: {ACCENT};
}}
QMenu {{
    background-color: {BG_RAISED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_NORMAL};
    border-radius: 4px;
    padding: 4px 0;
}}
QMenu::item {{
    padding: 6px 28px 6px 16px;
    border-radius: 2px;
}}
QMenu::item:selected {{
    background: {BG_SELECTED};
    color: {ACCENT_GLOW};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER_DIM};
    margin: 4px 8px;
}}

/* ── Dock widgets ─────────────────────────────────────── */
QDockWidget {{
    color: {TEXT_PRIMARY};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background: {BG_BASE};
    padding: 5px 8px;
    border-bottom: 1px solid {BORDER_DIM};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: {TEXT_SECONDARY};
}}
QDockWidget::float-button, QDockWidget::close-button {{
    background: transparent;
    border: none;
    padding: 0px;
}}

/* ── Group boxes ──────────────────────────────────────── */
QGroupBox {{
    background: {BG_RAISED};
    border: 1px solid {BORDER_DIM};
    border-radius: 5px;
    margin-top: 16px;
    padding: 8px 6px 6px 6px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.6px;
    color: {TEXT_SECONDARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 8px;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
}}

/* ── Scroll areas ─────────────────────────────────────── */
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: {BG_BASE};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_BRIGHT};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {BG_BASE};
    height: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_BRIGHT};
    border-radius: 3px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {ACCENT};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Buttons ──────────────────────────────────────────── */
QPushButton {{
    background: {BG_CONTROL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_NORMAL};
    border-radius: 4px;
    padding: 5px 12px;
    font-size: 11px;
}}
QPushButton:hover {{
    background: {BG_HOVER};
    border-color: {ACCENT};
    color: {ACCENT_GLOW};
}}
QPushButton:pressed {{
    background: {BG_ACTIVE};
    border-color: {ACCENT};
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    border-color: {BORDER_DIM};
}}
QPushButton[accent="true"] {{
    background: {ACCENT_DARK};
    border-color: {ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background: {ACCENT};
    color: {BG_DEEP};
}}
QPushButton[danger="true"] {{
    color: {COLOR_FAIL};
    border-color: {COLOR_FAIL}66;
}}
QPushButton[danger="true"]:hover {{
    background: {COLOR_FAIL}22;
    border-color: {COLOR_FAIL};
}}

/* ── Input widgets ────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {BG_CONTROL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_NORMAL};
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: {ACCENT_DARK};
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit::placeholder {{
    color: {TEXT_DIM};
}}

QSpinBox, QDoubleSpinBox {{
    background: {BG_CONTROL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_NORMAL};
    border-radius: 4px;
    padding: 3px 6px;
    min-width: 60px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {BG_HOVER};
    border: none;
    width: 16px;
    border-radius: 2px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {ACCENT_DARK};
}}

QComboBox {{
    background: {BG_CONTROL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_NORMAL};
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 80px;
}}
QComboBox:focus, QComboBox:hover {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {BG_RAISED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_NORMAL};
    selection-background-color: {BG_SELECTED};
    selection-color: {ACCENT_GLOW};
    outline: none;
}}

/* ── Checkboxes ───────────────────────────────────────── */
QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER_NORMAL};
    border-radius: 3px;
    background: {BG_CONTROL};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}

/* ── Sliders ──────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 3px;
    background: {BG_CONTROL};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
    border: 2px solid {BG_DEEP};
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT_GLOW};
    width: 14px;
    height: 14px;
    margin: -6px 0;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT_DARK};
    border-radius: 2px;
}}

/* ── Progress bars ────────────────────────────────────── */
QProgressBar {{
    background: {BG_CONTROL};
    border: none;
    border-radius: 2px;
    height: 5px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {COLOR_PERFECT};
    border-radius: 2px;
}}

/* ── List / tree widgets ──────────────────────────────── */
QListWidget, QTreeWidget, QTableWidget {{
    background: {BG_BASE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_DIM};
    border-radius: 4px;
    outline: none;
}}
QListWidget::item, QTreeWidget::item {{
    padding: 4px 6px;
    border-radius: 3px;
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {BG_SELECTED};
    color: {ACCENT_GLOW};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background: {BG_HOVER};
}}
QHeaderView::section {{
    background: {BG_BASE};
    color: {TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {BORDER_DIM};
    padding: 4px 8px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Tabs ─────────────────────────────────────────────── */
QTabWidget::pane {{
    background: {BG_RAISED};
    border: 1px solid {BORDER_DIM};
    border-radius: 4px;
}}
QTabBar::tab {{
    background: {BG_BASE};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER_DIM};
    border-bottom: none;
    padding: 6px 14px;
    border-radius: 4px 4px 0 0;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QTabBar::tab:selected {{
    background: {BG_RAISED};
    color: {ACCENT};
    border-color: {BORDER_NORMAL};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
    background: {BG_CONTROL};
}}

/* ── Status bar ───────────────────────────────────────── */
QStatusBar {{
    background: {BG_BASE};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER_DIM};
    font-size: 10px;
    padding: 2px 6px;
}}

/* ── Tooltips ─────────────────────────────────────────── */
QToolTip {{
    background: {BG_RAISED};
    color: {TEXT_PRIMARY};
    border: 1px solid {ACCENT_DARK};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 11px;
    opacity: 230;
}}

/* ── Labels ───────────────────────────────────────────── */
QLabel[role="metric-value"] {{
    color: {TEXT_PRIMARY};
    font-size: 13px;
    font-weight: 700;
}}
QLabel[role="metric-label"] {{
    color: {TEXT_SECONDARY};
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}
QLabel[verdict="PERFECT"] {{ color: {COLOR_PERFECT}; font-weight: 800; font-size: 14px; }}
QLabel[verdict="GOOD"]    {{ color: {COLOR_GOOD};    font-weight: 700; font-size: 14px; }}
QLabel[verdict="SOFT"]    {{ color: {COLOR_WARN};    font-weight: 700; font-size: 14px; }}
QLabel[verdict="BLURRY"]  {{ color: {COLOR_FAIL};    font-weight: 700; font-size: 14px; }}
QLabel[verdict="PASS"]    {{ color: {COLOR_PERFECT}; font-weight: 800; font-size: 14px; }}
QLabel[verdict="FAIL"]    {{ color: {COLOR_FAIL};    font-weight: 800; font-size: 14px; }}
"""


# ── Metric bar colors by verdict ───────────────────────────────────────────
VERDICT_COLOR = {
    "PERFECT": COLOR_PERFECT,
    "GOOD":    COLOR_GOOD,
    "SOFT":    COLOR_WARN,
    "BLURRY":  COLOR_FAIL,
    "PASS":    COLOR_PERFECT,
    "FAIL":    COLOR_FAIL,
}
