"""Pixel-perfect image viewer with zoom, pan, heatmap overlay, and pixel inspector."""

import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal, QPointF
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QWheelEvent,
    QMouseEvent, QKeyEvent, QFont,
)
from src.core.config import Config
from src.core.image_data import ImageData
from src.analysis.focus_engine import FocusEngine


class ViewerPanel(QWidget):
    pixel_hovered = pyqtSignal(int, int, object)  # x, y, pixel_value

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._image: np.ndarray | None = None
        self._image_data: ImageData | None = None
        self._focus_map: np.ndarray | None = None
        self._show_heatmap = config.show_focus_heatmap
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._drag_start: QPoint | None = None
        self._drag_offset_start: QPointF | None = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(80, 60)
        self.setStyleSheet("background-color: #1a1a1a;")

        self._focus_engine = FocusEngine()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def set_image(self, image: np.ndarray, image_data: ImageData | None = None):
        self._image = image
        self._image_data = image_data
        self.update()

    def set_focus_map(self, focus_map: np.ndarray):
        self._focus_map = focus_map
        self.update()

    def toggle_heatmap(self):
        self._show_heatmap = not self._show_heatmap
        self.update()

    def fit_to_window(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        scale_x = self.width()  / w
        scale_y = self.height() / h
        self._zoom = min(scale_x, scale_y) * 0.95
        self._center_image()
        self.update()

    def set_zoom(self, zoom: float):
        self._zoom = max(0.05, min(zoom, 32.0))
        self._center_image()
        self.update()

    # ------------------------------------------------------------------ #
    #  Paint
    # ------------------------------------------------------------------ #

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        if self._image is None:
            self._draw_empty(painter)
            return

        pixmap = self._to_pixmap(self._image)
        dest_w = int(pixmap.width()  * self._zoom)
        dest_h = int(pixmap.height() * self._zoom)
        dest_x = int(self._offset.x())
        dest_y = int(self._offset.y())
        dest_rect = QRect(dest_x, dest_y, dest_w, dest_h)

        painter.drawPixmap(dest_rect, pixmap)

        if self._show_heatmap and self._focus_map is not None:
            self._draw_heatmap(painter, dest_x, dest_y, dest_w, dest_h)

        if self._zoom >= 8.0:
            self._draw_pixel_grid(painter, dest_x, dest_y, dest_w, dest_h)

        self._draw_info_overlay(painter)

    def _draw_empty(self, painter: QPainter):
        painter.setPen(QColor(80, 80, 80))
        font = QFont("Arial", 14)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                         "Open an image  (Ctrl+O)  or  drag & drop here")

    def _draw_heatmap(self, painter: QPainter, dx, dy, dw, dh):
        if self._focus_map is None:
            return
        heatmap_rgb = self._focus_engine.heatmap_to_rgb(self._focus_map)
        heatmap_pixmap = self._to_pixmap(heatmap_rgb)
        painter.setOpacity(0.4)
        painter.drawPixmap(QRect(dx, dy, dw, dh), heatmap_pixmap)
        painter.setOpacity(1.0)

    def _draw_pixel_grid(self, painter: QPainter, dx, dy, dw, dh):
        if self._image is None:
            return
        pen = QPen(QColor(60, 60, 60, 120))
        pen.setWidth(1)
        painter.setPen(pen)
        cell = self._zoom
        x = dx
        while x <= dx + dw:
            painter.drawLine(int(x), dy, int(x), dy + dh)
            x += cell
        y = dy
        while y <= dy + dh:
            painter.drawLine(dx, int(y), dx + dw, int(y))
            y += cell

    def _draw_info_overlay(self, painter: QPainter):
        if self._image_data is None:
            return
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Courier", 9)
        painter.setFont(font)
        text = f"Zoom: {self._zoom*100:.0f}%   {self._image_data.shape_str()}"
        painter.drawText(8, self.height() - 8, text)

    # ------------------------------------------------------------------ #
    #  Mouse interaction
    # ------------------------------------------------------------------ #

    def wheelEvent(self, event: QWheelEvent):
        if self._image is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        mouse_pos = event.position()
        # Zoom toward cursor
        img_x = (mouse_pos.x() - self._offset.x()) / self._zoom
        img_y = (mouse_pos.y() - self._offset.y()) / self._zoom
        self._zoom = max(0.05, min(self._zoom * factor, 32.0))
        self._offset = QPointF(
            mouse_pos.x() - img_x * self._zoom,
            mouse_pos.y() - img_y * self._zoom,
        )
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self._drag_start = event.pos()
            self._drag_offset_start = QPointF(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start and self._drag_offset_start:
            delta = event.pos() - self._drag_start
            self._offset = QPointF(
                self._drag_offset_start.x() + delta.x(),
                self._drag_offset_start.y() + delta.y(),
            )
            self.update()

        # Emit pixel value under cursor
        if self._image is not None:
            px = int((event.pos().x() - self._offset.x()) / self._zoom)
            py = int((event.pos().y() - self._offset.y()) / self._zoom)
            h, w = self._image.shape[:2]
            if 0 <= px < w and 0 <= py < h:
                pixel = self._image[py, px]
                self.pixel_hovered.emit(px, py, pixel)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_start = None
        self._drag_offset_start = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def resizeEvent(self, event):
        if self._image is not None and self._zoom == 1.0:
            self._center_image()
        super().resizeEvent(event)

    # ------------------------------------------------------------------ #
    #  Drag and drop
    # ------------------------------------------------------------------ #

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.parent().parent().open_image(path)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _center_image(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        self._offset = QPointF(
            (self.width()  - w * self._zoom) / 2,
            (self.height() - h * self._zoom) / 2,
        )

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QPixmap:
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        if image.ndim == 2:
            h, w = image.shape
            qimg = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif image.shape[2] == 3:
            h, w = image.shape[:2]
            qimg = QImage(image.data, w, h, w * 3, QImage.Format.Format_RGB888)
        elif image.shape[2] == 4:
            h, w = image.shape[:2]
            qimg = QImage(image.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        else:
            return QPixmap()
        return QPixmap.fromImage(qimg.copy())
