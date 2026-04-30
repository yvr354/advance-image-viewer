"""
OpenGL image viewer.
Renders image as GPU texture — 10K×10K images pan/zoom at 60fps.

Controls:
  Left drag (Navigate)  — pan
  Scroll wheel          — zoom toward cursor
  Right drag (Navigate) — rubber-band zoom to region
  Double-click          — fit to window
  F                     — toggle focus heatmap
  G                     — toggle focus grid

Inspection tool modes (set via set_tool()):
  navigate  — default: pan + rubber-band zoom
  roi       — left drag draws ROI rectangle; right drag pans
  profile   — left drag draws intensity-profile line; right drag pans
  annotate  — left click places annotation marker; right drag pans
  measure   — left drag draws calibrated distance line; right drag pans
"""

import math
import numpy as np
import cv2
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import (QWheelEvent, QMouseEvent, QSurfaceFormat, QFont,
                          QPainter, QColor, QPen, QBrush)

from OpenGL.GL import (
    glEnable, glDisable, glBlendFunc, glClearColor, glClear, glViewport,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glDeleteTextures,
    glMatrixMode, glLoadIdentity, glOrtho,
    glBegin, glEnd, glTexCoord2f, glVertex2f, glColor4f, glLineWidth,
    GL_TEXTURE_2D, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_COLOR_BUFFER_BIT, GL_RGBA, GL_RGB, GL_LUMINANCE,
    GL_UNSIGNED_BYTE, GL_LINEAR, GL_NEAREST,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
    GL_PROJECTION, GL_MODELVIEW, GL_QUADS, GL_LINES, GL_LINE_LOOP,
    glGenBuffers,
)

# ── Per-label annotation colors ────────────────────────────────────────────
_ANN_COLORS = {
    "Scratch":       "#FF1744",
    "Pit":           "#FF6D00",
    "Contamination": "#FFD600",
    "Burr":          "#E040FB",
    "Crack":         "#FF5252",
    "OK":            "#00E676",
    "Other":         "#00B0FF",
}


class GLImageViewer(QOpenGLWidget):
    """
    OpenGL-accelerated image viewer.
    Handles images of any size — rendering cost is screen-size, not image-size.
    """

    # ── Navigation / analysis signals ────────────────────────────────────
    pixel_hovered      = pyqtSignal(int, int, object)    # x, y, pixel value
    zoom_changed       = pyqtSignal(float)
    view_state_changed = pyqtSignal(float, float, float)  # zoom, offset_x, offset_y

    # ── Inspection tool signals (image-pixel coordinates) ─────────────────
    roi_selected       = pyqtSignal(int, int, int, int)   # ix1,iy1,ix2,iy2
    line_profile_drawn = pyqtSignal(int, int, int, int)   # ix1,iy1,ix2,iy2
    annotation_placed  = pyqtSignal(int, int)             # ix, iy
    measure_done       = pyqtSignal(int, int, int, int)   # ix1,iy1,ix2,iy2

    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)
        super().__init__(parent)

        self._image:   np.ndarray | None = None
        self._heatmap: np.ndarray | None = None

        self._tex_image:   int = 0
        self._tex_heatmap: int = 0
        self._img_w: int = 0
        self._img_h: int = 0

        self._zoom:        float   = 1.0
        self._offset:      QPointF = QPointF(0, 0)
        self._drag_start:  QPointF | None = None
        self._drag_offset: QPointF | None = None

        # Rubber-band zoom (navigate mode only)
        self._rb_start:   QPointF | None = None
        self._rb_current: QPointF | None = None

        self._show_heatmap:    bool  = False
        self._show_pixel_grid: bool  = True
        self._heatmap_opacity: float = 0.30
        self._grid_threshold:  float = 8.0
        self._syncing:         bool  = False

        # Focus grid overlay
        self._focus_grid       = None   # FocusGridData or None
        self._show_focus_grid: bool = False

        # ── Inspection tool system ────────────────────────────────────────
        self._tool: str = "navigate"         # navigate|roi|profile|annotate|measure
        self._tdrag_start: QPointF | None = None   # screen pos at tool press
        self._tdrag_now:   QPointF | None = None   # screen pos at current move

        # Finalized overlays (image pixel coords)
        self._roi_img:     tuple | None = None   # (ix1,iy1,ix2,iy2)
        self._profile_img: tuple | None = None   # (ix1,iy1,ix2,iy2)
        self._measure_img: tuple | None = None   # (ix1,iy1,ix2,iy2)

        # Annotations: list of {ix, iy, label}
        self._annotations: list = []

        # mm/px calibration for measurement tool
        self._mm_per_px: float = 0.0   # 0 = not calibrated

        # Throttle pixel hover — only emit when image coordinate changes
        self._last_hover_ix: int = -1
        self._last_hover_iy: int = -1

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)

    # ── Public API ────────────────────────────────────────────────────────

    def set_image(self, image: np.ndarray, preserve_view: bool = False):
        """Upload new image to GPU. Any size, any bit depth."""
        old_w, old_h = self._img_w, self._img_h
        self._image = image
        self._img_h, self._img_w = image.shape[:2]
        self.makeCurrent()
        self._upload_image_texture(image)
        self.doneCurrent()
        if preserve_view and old_w == self._img_w and old_h == self._img_h:
            self.update()
        else:
            self.fit_to_window()

    def set_heatmap(self, heatmap_rgb: np.ndarray | None):
        self._heatmap = heatmap_rgb
        if heatmap_rgb is not None:
            self.makeCurrent()
            self._upload_heatmap_texture(heatmap_rgb)
            self.doneCurrent()
        self.update()

    def toggle_heatmap(self):
        self._show_heatmap = not self._show_heatmap
        if self._show_heatmap:
            self._show_focus_grid = False
        self.update()

    def set_heatmap_visible(self, visible: bool):
        self._show_heatmap = visible
        self.update()

    def set_focus_grid(self, grid_data):
        self._focus_grid = grid_data
        self.update()

    def set_focus_grid_visible(self, visible: bool):
        self._show_focus_grid = visible
        self.update()

    # ── Inspection tool API ───────────────────────────────────────────────

    def set_tool(self, tool: str):
        """Switch inspection tool. 'navigate' restores normal pan/zoom behavior."""
        self._tool = tool
        self._tdrag_start = None
        self._tdrag_now   = None
        if tool == "navigate":
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif tool == "annotate":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def set_calibration(self, mm_per_px: float):
        self._mm_per_px = mm_per_px
        self.update()

    def clear_tool_overlays(self):
        """Remove ROI, profile line, and measure line."""
        self._roi_img     = None
        self._profile_img = None
        self._measure_img = None
        self._tdrag_start = None
        self._tdrag_now   = None
        self.update()

    def add_annotation(self, ix: int, iy: int, label: str):
        self._annotations.append({"ix": ix, "iy": iy, "label": label})
        self.update()

    def remove_annotation(self, index: int):
        if 0 <= index < len(self._annotations):
            self._annotations.pop(index)
            self.update()

    def clear_annotations(self):
        self._annotations.clear()
        self.update()

    def get_annotations(self) -> list:
        return list(self._annotations)

    def set_annotations(self, anns: list):
        self._annotations = list(anns)
        self.update()

    def fit_to_window(self):
        if self._img_w == 0:
            return
        sx = self.width()  / self._img_w
        sy = self.height() / self._img_h
        self._zoom = min(sx, sy) * 0.96
        self._center()
        self.update()
        self.zoom_changed.emit(self._zoom)

    def set_zoom(self, zoom: float):
        self._zoom = max(0.02, min(zoom, 64.0))
        self._center()
        self.update()
        self.zoom_changed.emit(self._zoom)

    def set_view_state(self, zoom: float, ox: float, oy: float):
        """Sync from external source — does NOT re-emit view_state_changed."""
        self._syncing = True
        self._zoom   = max(0.02, min(zoom, 64.0))
        self._offset = QPointF(ox, oy)
        self.update()
        self.zoom_changed.emit(self._zoom)
        self._syncing = False

    def pixel_at(self, screen_x: int, screen_y: int):
        ix = (screen_x - self._offset.x()) / self._zoom
        iy = (screen_y - self._offset.y()) / self._zoom
        return int(ix), int(iy)

    def _img_to_screen(self, ix: float, iy: float):
        return (self._offset.x() + ix * self._zoom,
                self._offset.y() + iy * self._zoom)

    # ── OpenGL lifecycle ──────────────────────────────────────────────────

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.039, 0.039, 0.059, 1.0)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        if self._image is not None and self._tex_image == 0:
            self._upload_image_texture(self._image)
            self._center()

        glClear(GL_COLOR_BUFFER_BIT)

        if self._tex_image == 0 or self._img_w == 0:
            self._draw_empty_hint()
            return

        self._draw_texture(self._tex_image, opacity=1.0)

        if self._show_heatmap and self._tex_heatmap != 0:
            self._draw_texture(self._tex_heatmap, opacity=self._heatmap_opacity)

        if self._show_focus_grid and self._focus_grid is not None:
            self._draw_focus_grid_gl()

        if self._zoom >= self._grid_threshold and self._show_pixel_grid:
            self._draw_pixel_grid()

        if self._rb_start and self._rb_current:
            self._draw_rubber_band()

    def paintEvent(self, event):
        """GL renders first; QPainter draws all vector overlays on top."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        if self._img_w > 0:
            if self._show_focus_grid and self._focus_grid is not None:
                self._paint_focus_grid_labels(painter)
            self._paint_roi_overlay(painter)
            self._paint_profile_overlay(painter)
            self._paint_measure_overlay(painter)
            self._paint_annotations(painter)

        painter.end()

    # ── GL draw helpers ───────────────────────────────────────────────────

    def _draw_texture(self, tex: int, opacity: float):
        x = self._offset.x()
        y = self._offset.y()
        w = self._img_w * self._zoom
        h = self._img_h * self._zoom

        glBindTexture(GL_TEXTURE_2D, tex)
        glColor4f(1, 1, 1, opacity)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x,     y    )
        glTexCoord2f(1, 0); glVertex2f(x + w, y    )
        glTexCoord2f(1, 1); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 1); glVertex2f(x,     y + h)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)

    def _draw_pixel_grid(self):
        glDisable(GL_TEXTURE_2D)
        glColor4f(0.25, 0.25, 0.4, 0.5)
        glBegin(GL_LINES)

        cell = self._zoom
        ox, oy = self._offset.x(), self._offset.y()
        W, H   = self.width(), self.height()

        x = ox % cell
        while x <= W:
            glVertex2f(x, 0); glVertex2f(x, H)
            x += cell

        y = oy % cell
        while y <= H:
            glVertex2f(0, y); glVertex2f(W, y)
            y += cell

        glEnd()
        glEnable(GL_TEXTURE_2D)

    def _draw_rubber_band(self):
        glDisable(GL_TEXTURE_2D)
        x1, y1 = self._rb_start.x(),   self._rb_start.y()
        x2, y2 = self._rb_current.x(), self._rb_current.y()

        glColor4f(0.0, 0.706, 0.847, 0.12)
        glBegin(GL_QUADS)
        glVertex2f(x1, y1); glVertex2f(x2, y1)
        glVertex2f(x2, y2); glVertex2f(x1, y2)
        glEnd()

        glColor4f(0.0, 0.706, 0.847, 1.0)
        glLineWidth(1.5)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x1, y1); glVertex2f(x2, y1)
        glVertex2f(x2, y2); glVertex2f(x1, y2)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_TEXTURE_2D)

    def _draw_focus_grid_gl(self):
        g = self._focus_grid
        R, C    = g.rows, g.cols
        ox, oy  = self._offset.x(), self._offset.y()
        zoom    = self._zoom
        cell_sw = g.img_w * zoom / C
        cell_sh = g.img_h * zoom / R

        glDisable(GL_TEXTURE_2D)

        for r in range(R):
            for c in range(C):
                score    = float(g.scores[r, c])
                is_best  = (r, c) == g.best_cell
                is_worst = (r, c) == g.worst_cell

                sx0 = ox + c       * cell_sw
                sy0 = oy + r       * cell_sh
                sx1 = ox + (c + 1) * cell_sw
                sy1 = oy + (r + 1) * cell_sh

                if is_best:
                    fr, fg_, fb = 0.0,  1.0,  0.35
                elif is_worst:
                    fr, fg_, fb = 1.0,  0.08, 0.08
                elif score >= 72:
                    fr, fg_, fb = 0.0,  0.90, 0.30
                elif score >= 38:
                    fr, fg_, fb = 1.0,  0.72, 0.0
                else:
                    fr, fg_, fb = 0.95, 0.10, 0.10

                fill_alpha = 0.45 if (is_best or is_worst) else 0.28
                glColor4f(fr, fg_, fb, fill_alpha)
                glBegin(GL_QUADS)
                glVertex2f(sx0, sy0); glVertex2f(sx1, sy0)
                glVertex2f(sx1, sy1); glVertex2f(sx0, sy1)
                glEnd()

                lw = 3.0 if (is_best or is_worst) else 1.5
                glLineWidth(lw)
                glColor4f(fr, fg_, fb, 1.0)
                glBegin(GL_LINE_LOOP)
                glVertex2f(sx0, sy0); glVertex2f(sx1, sy0)
                glVertex2f(sx1, sy1); glVertex2f(sx0, sy1)
                glEnd()

        glLineWidth(1.0)
        glEnable(GL_TEXTURE_2D)

    def _draw_empty_hint(self):
        painter = QPainter(self)
        painter.setPen(QColor(68, 68, 90))
        painter.setFont(QFont("Segoe UI", 13))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                         "Open Image   Ctrl+O\nor drag & drop here")
        painter.end()

    # ── QPainter overlay layers ───────────────────────────────────────────

    def _paint_focus_grid_labels(self, painter: QPainter):
        g = self._focus_grid
        R, C    = g.rows, g.cols
        ox, oy  = self._offset.x(), self._offset.y()
        zoom    = self._zoom
        cell_sw = g.img_w * zoom / C
        cell_sh = g.img_h * zoom / R

        if cell_sw < 30 or cell_sh < 20:
            return

        from PyQt6.QtCore import QRectF, Qt as _Qt

        font_size  = max(9, min(22, int(cell_sh * 0.38)))
        font_bold  = QFont("Segoe UI", font_size, QFont.Weight.Bold)
        font_small = QFont("Segoe UI", max(7, font_size - 3), QFont.Weight.Bold)

        for r in range(R):
            for c in range(C):
                score    = float(g.scores[r, c])
                is_best  = (r, c) == g.best_cell
                is_worst = (r, c) == g.worst_cell
                sx0 = ox + c * cell_sw
                sy0 = oy + r * cell_sh
                cell_rect = QRectF(sx0, sy0, cell_sw, cell_sh)

                if is_best:
                    txt_color = QColor(220, 255, 220)
                elif is_worst:
                    txt_color = QColor(255, 210, 210)
                elif score >= 72:
                    txt_color = QColor(200, 255, 200)
                elif score >= 38:
                    txt_color = QColor(255, 240, 160)
                else:
                    txt_color = QColor(255, 200, 200)

                score_str = f"{score:.0f}"
                painter.setFont(font_bold)

                painter.setPen(QColor(0, 0, 0, 200))
                for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    shifted = QRectF(sx0+dx, sy0+dy+cell_sh*0.15, cell_sw, cell_sh*0.55)
                    painter.drawText(shifted, _Qt.AlignmentFlag.AlignCenter, score_str)

                painter.setPen(txt_color)
                painter.drawText(QRectF(sx0, sy0+cell_sh*0.15, cell_sw, cell_sh*0.55),
                                 _Qt.AlignmentFlag.AlignCenter, score_str)

                if is_best or is_worst:
                    painter.setFont(font_small)
                    label = "◆ BEST" if is_best else "◆ WORST"
                    lcolor = QColor(180, 255, 180) if is_best else QColor(255, 180, 180)

                    painter.setPen(QColor(0, 0, 0, 180))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        lr = QRectF(sx0+dx, sy0+dy+cell_sh*0.62, cell_sw, cell_sh*0.32)
                        painter.drawText(lr, _Qt.AlignmentFlag.AlignCenter, label)

                    painter.setPen(lcolor)
                    painter.drawText(QRectF(sx0, sy0+cell_sh*0.62, cell_sw, cell_sh*0.32),
                                     _Qt.AlignmentFlag.AlignCenter, label)

    def _paint_roi_overlay(self, painter: QPainter):
        from PyQt6.QtCore import QRectF

        # Live drag preview
        if self._tdrag_start is not None and self._tdrag_now is not None \
                and self._tool == "roi":
            x1 = min(self._tdrag_start.x(), self._tdrag_now.x())
            y1 = min(self._tdrag_start.y(), self._tdrag_now.y())
            x2 = max(self._tdrag_start.x(), self._tdrag_now.x())
            y2 = max(self._tdrag_start.y(), self._tdrag_now.y())
            painter.setPen(QPen(QColor("#00E5FF"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(0, 229, 255, 25)))
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

        # Finalized ROI
        if self._roi_img is None:
            return
        ix1, iy1, ix2, iy2 = self._roi_img
        sx1, sy1 = self._img_to_screen(ix1, iy1)
        sx2, sy2 = self._img_to_screen(ix2, iy2)

        painter.setPen(QPen(QColor("#00E5FF"), 2.0))
        painter.setBrush(QBrush(QColor(0, 229, 255, 18)))
        painter.drawRect(QRectF(sx1, sy1, sx2 - sx1, sy2 - sy1))

        # Corner handles
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#00E5FF")))
        for cx, cy in [(sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)]:
            painter.drawEllipse(QPointF(cx, cy), 4, 4)

        # Size label
        w_px = ix2 - ix1;  h_px = iy2 - iy1
        lbl = f" {w_px}×{h_px} px "
        painter.setPen(QColor(0, 0, 0, 160))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.drawText(QPointF(sx1 + 2, sy1 - 3), lbl)
        painter.setPen(QColor("#00E5FF"))
        painter.drawText(QPointF(sx1 + 1, sy1 - 4), lbl)

    def _paint_profile_overlay(self, painter: QPainter):
        # Live drag preview
        if self._tdrag_start is not None and self._tdrag_now is not None \
                and self._tool == "profile":
            painter.setPen(QPen(QColor("#FFE040"), 2.0, Qt.PenStyle.DashLine))
            painter.drawLine(self._tdrag_start, self._tdrag_now)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#FFE040")))
            painter.drawEllipse(self._tdrag_start, 4, 4)
            painter.drawEllipse(self._tdrag_now, 4, 4)

        if self._profile_img is None:
            return
        ix1, iy1, ix2, iy2 = self._profile_img
        sx1, sy1 = self._img_to_screen(ix1, iy1)
        sx2, sy2 = self._img_to_screen(ix2, iy2)

        painter.setPen(QPen(QColor("#FFE040"), 2.0))
        painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#FFE040")))
        painter.drawEllipse(QPointF(sx1, sy1), 5, 5)
        painter.drawEllipse(QPointF(sx2, sy2), 5, 5)

        length = math.sqrt((ix2 - ix1)**2 + (iy2 - iy1)**2)
        mx, my = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        lbl = f" {length:.0f} px "
        painter.setPen(QColor(0, 0, 0, 160))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.drawText(QPointF(mx + 2, my - 3), lbl)
        painter.setPen(QColor("#FFE040"))
        painter.drawText(QPointF(mx + 1, my - 4), lbl)

    def _paint_measure_overlay(self, painter: QPainter):
        def _draw(sx1, sy1, sx2, sy2, ix1, iy1, ix2, iy2, dashed=False):
            style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
            painter.setPen(QPen(QColor("#FFFFFF"), 1.5, style))
            painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))
            # End tick marks
            painter.drawLine(QPointF(sx1 - 5, sy1), QPointF(sx1 + 5, sy1))
            painter.drawLine(QPointF(sx2 - 5, sy2), QPointF(sx2 + 5, sy2))
            # Distance label
            dx = ix2 - ix1;  dy = iy2 - iy1
            dist_px = math.sqrt(dx * dx + dy * dy)
            angle   = math.degrees(math.atan2(abs(dy), abs(dx)))
            parts   = [f"{dist_px:.1f} px"]
            if self._mm_per_px > 0:
                parts.append(f"{dist_px * self._mm_per_px:.3f} mm")
            parts.append(f"∠{angle:.1f}°")
            lbl = "  ".join(parts)
            mx, my = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            painter.setPen(QColor(0, 0, 0, 180))
            painter.drawText(QPointF(mx + 2, my - 3), lbl)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(QPointF(mx + 1, my - 4), lbl)

        if self._tdrag_start is not None and self._tdrag_now is not None \
                and self._tool == "measure":
            sx1, sy1 = self._tdrag_start.x(), self._tdrag_start.y()
            sx2, sy2 = self._tdrag_now.x(),   self._tdrag_now.y()
            ix1, iy1 = self.pixel_at(int(sx1), int(sy1))
            ix2, iy2 = self.pixel_at(int(sx2), int(sy2))
            _draw(sx1, sy1, sx2, sy2, ix1, iy1, ix2, iy2, dashed=True)

        if self._measure_img is not None:
            ix1, iy1, ix2, iy2 = self._measure_img
            sx1, sy1 = self._img_to_screen(ix1, iy1)
            sx2, sy2 = self._img_to_screen(ix2, iy2)
            _draw(sx1, sy1, sx2, sy2, ix1, iy1, ix2, iy2, dashed=False)

    def _paint_annotations(self, painter: QPainter):
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)

        for i, ann in enumerate(self._annotations):
            ix, iy  = ann["ix"], ann["iy"]
            label   = ann.get("label", "Other")
            sx, sy  = self._img_to_screen(ix, iy)

            cstr  = _ANN_COLORS.get(label, "#00B0FF")
            color = QColor(cstr)

            # Circle
            r = 10
            painter.setPen(QPen(color, 2.0))
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 55)))
            painter.drawEllipse(QPointF(sx, sy), r, r)

            # Crosshair
            painter.setPen(QPen(color, 1.5))
            painter.drawLine(QPointF(sx - 6, sy), QPointF(sx + 6, sy))
            painter.drawLine(QPointF(sx, sy - 6), QPointF(sx, sy + 6))

            # Label with dark shadow
            text = f"#{i + 1} {label}"
            painter.setPen(QColor(0, 0, 0, 180))
            painter.drawText(QPointF(sx + r + 3, sy + 1), text)
            painter.setPen(color)
            painter.drawText(QPointF(sx + r + 2, sy), text)

    # ── Texture upload ────────────────────────────────────────────────────

    def _upload_image_texture(self, image: np.ndarray):
        if self._tex_image:
            glDeleteTextures(1, [self._tex_image])

        self._tex_image = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex_image)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        img8 = self._to_gl_format(image)
        h, w = img8.shape[:2]

        if img8.ndim == 2:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0,
                         GL_LUMINANCE, GL_UNSIGNED_BYTE, img8.tobytes())
        elif img8.shape[2] == 3:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, img8.tobytes())
        elif img8.shape[2] == 4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, img8.tobytes())

        glBindTexture(GL_TEXTURE_2D, 0)

    def _upload_heatmap_texture(self, heatmap_rgb: np.ndarray):
        if self._tex_heatmap:
            glDeleteTextures(1, [self._tex_heatmap])

        self._tex_heatmap = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex_heatmap)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # GL_NEAREST = sharp cell boundaries (no interpolation between cells)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        h, w = heatmap_rgb.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, heatmap_rgb.tobytes())
        glBindTexture(GL_TEXTURE_2D, 0)

    # ── Mouse interaction ─────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        if self._img_w == 0:
            return
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        pos = event.position()
        img_x = (pos.x() - self._offset.x()) / self._zoom
        img_y = (pos.y() - self._offset.y()) / self._zoom
        self._zoom = max(0.02, min(self._zoom * factor, 64.0))
        self._offset = QPointF(
            pos.x() - img_x * self._zoom,
            pos.y() - img_y * self._zoom,
        )
        self.update()
        self.zoom_changed.emit(self._zoom)
        if not self._syncing:
            self.view_state_changed.emit(self._zoom, self._offset.x(), self._offset.y())

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool == "navigate":
                self._drag_start  = event.position()
                self._drag_offset = QPointF(self._offset)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self._tdrag_start = event.position()
                self._tdrag_now   = event.position()
        elif event.button() == Qt.MouseButton.RightButton:
            if self._tool == "navigate":
                self._rb_start   = event.position()
                self._rb_current = event.position()
            else:
                # Right drag = pan in all tool modes
                self._drag_start  = event.position()
                self._drag_offset = QPointF(self._offset)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        # Pan (navigate left-drag or tool right-drag)
        if self._drag_start and self._drag_offset:
            d = event.position() - self._drag_start
            self._offset = QPointF(self._drag_offset.x() + d.x(),
                                   self._drag_offset.y() + d.y())
            if not self._syncing:
                self.view_state_changed.emit(self._zoom, self._offset.x(), self._offset.y())
            self.update()

        # Rubber-band zoom preview
        if self._rb_start is not None:
            self._rb_current = event.position()
            self.update()

        # Tool drag live preview
        if self._tdrag_start is not None:
            self._tdrag_now = event.position()
            self.update()

        # Pixel hover readout
        if self._image is not None:
            ix, iy = self.pixel_at(int(event.position().x()), int(event.position().y()))
            h, w = self._image.shape[:2]
            if 0 <= ix < w and 0 <= iy < h:
                if ix != self._last_hover_ix or iy != self._last_hover_iy:
                    self._last_hover_ix = ix
                    self._last_hover_iy = iy
                    self.pixel_hovered.emit(ix, iy, self._image[iy, ix])

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool == "navigate":
                self._drag_start  = None
                self._drag_offset = None
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self._finalize_tool_action()
                self._tdrag_start = None
                self._tdrag_now   = None
                self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            if self._tool == "navigate":
                if self._rb_start is not None:
                    self._zoom_to_rubber_band()
                self._rb_start   = None
                self._rb_current = None
                self.update()
            else:
                self._drag_start  = None
                self._drag_offset = None
                self.setCursor(Qt.CursorShape.CrossCursor)

    def _finalize_tool_action(self):
        if self._tdrag_start is None or self._tdrag_now is None:
            return

        sx1, sy1 = self._tdrag_start.x(), self._tdrag_start.y()
        sx2, sy2 = self._tdrag_now.x(),   self._tdrag_now.y()
        ix1, iy1 = self.pixel_at(int(sx1), int(sy1))
        ix2, iy2 = self.pixel_at(int(sx2), int(sy2))

        # Clamp to image bounds
        if self._image is not None:
            H, W = self._image.shape[:2]
            ix1 = max(0, min(ix1, W - 1));  iy1 = max(0, min(iy1, H - 1))
            ix2 = max(0, min(ix2, W - 1));  iy2 = max(0, min(iy2, H - 1))

        move_px = math.sqrt((sx2 - sx1)**2 + (sy2 - sy1)**2)

        if self._tool == "roi":
            w_px = abs(ix2 - ix1);  h_px = abs(iy2 - iy1)
            if w_px > 4 and h_px > 4:
                self._roi_img = (min(ix1, ix2), min(iy1, iy2),
                                 max(ix1, ix2), max(iy1, iy2))
                self.roi_selected.emit(*self._roi_img)

        elif self._tool == "profile":
            dist = math.sqrt((ix2 - ix1)**2 + (iy2 - iy1)**2)
            if dist > 5:
                self._profile_img = (ix1, iy1, ix2, iy2)
                self.line_profile_drawn.emit(ix1, iy1, ix2, iy2)

        elif self._tool == "annotate":
            if move_px < 6:   # click (not drag)
                self.annotation_placed.emit(ix1, iy1)

        elif self._tool == "measure":
            dist = math.sqrt((ix2 - ix1)**2 + (iy2 - iy1)**2)
            if dist > 2:
                self._measure_img = (ix1, iy1, ix2, iy2)
                self.measure_done.emit(ix1, iy1, ix2, iy2)

    def _zoom_to_rubber_band(self):
        if self._rb_start is None or self._rb_current is None:
            return
        x1, y1 = self._rb_start.x(),   self._rb_start.y()
        x2, y2 = self._rb_current.x(), self._rb_current.y()
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return

        ix1 = (min(x1, x2) - self._offset.x()) / self._zoom
        iy1 = (min(y1, y2) - self._offset.y()) / self._zoom
        ix2 = (max(x1, x2) - self._offset.x()) / self._zoom
        iy2 = (max(y1, y2) - self._offset.y()) / self._zoom

        ix1 = max(0.0, ix1);  iy1 = max(0.0, iy1)
        ix2 = min(float(self._img_w), ix2)
        iy2 = min(float(self._img_h), iy2)

        roi_w = ix2 - ix1;  roi_h = iy2 - iy1
        if roi_w <= 0 or roi_h <= 0:
            return

        new_zoom = min(self.width() / roi_w, self.height() / roi_h) * 0.95
        new_zoom = max(0.02, min(new_zoom, 64.0))

        cx = (ix1 + ix2) / 2;  cy = (iy1 + iy2) / 2
        self._zoom   = new_zoom
        self._offset = QPointF(
            self.width()  / 2 - cx * new_zoom,
            self.height() / 2 - cy * new_zoom,
        )
        self.zoom_changed.emit(self._zoom)
        if not self._syncing:
            self.view_state_changed.emit(self._zoom, self._offset.x(), self._offset.y())
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_to_window()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._img_w > 0:
            self._center()

    # ── Drag and drop ─────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            win = self.window()
            if hasattr(win, "open_image"):
                win.open_image(path)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _center(self):
        self._offset = QPointF(
            (self.width()  - self._img_w * self._zoom) / 2,
            (self.height() - self._img_h * self._zoom) / 2,
        )

    @staticmethod
    def _to_gl_format(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return np.ascontiguousarray(image)
