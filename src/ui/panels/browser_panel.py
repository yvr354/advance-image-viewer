"""Left-side file browser with thumbnail filmstrip."""

import os
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QTreeWidget, QTreeWidgetItem, QSplitter, QLineEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QPixmap, QIcon, QImage

from src.core.image_loader import list_images_in_folder, is_supported


class ThumbnailLoader(QThread):
    thumbnail_ready = Signal(str, QPixmap)

    def __init__(self, paths: list[str], size: int = 80):
        super().__init__()
        self.paths = paths
        self.size = size
        self._stop = False

    def run(self):
        for path in self.paths:
            if self._stop:
                break
            try:
                img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
                if img is None:
                    img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                scale = self.size / max(h, w)
                nh, nw = int(h * scale), int(w * scale)
                img = cv2.resize(img, (nw, nh))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QImage(img_rgb.data, nw, nh, nw * 3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg.copy())
                self.thumbnail_ready.emit(path, pixmap)
            except Exception:
                pass

    def stop(self):
        self._stop = True


class BrowserPanel(QWidget):
    image_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._folder = ""
        self._paths: list[str] = []
        self._thumb_loader: ThumbnailLoader | None = None
        self._path_to_item: dict[str, QListWidgetItem] = {}
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Search bar
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter filenames...")
        self._search.textChanged.connect(self._filter_list)
        layout.addWidget(self._search)

        # File count label
        self._count_label = QLabel("No folder open")
        self._count_label.setStyleSheet("color: gray; font-size: 9px;")
        layout.addWidget(self._count_label)

        # Thumbnail list
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(80, 80))
        self._list.setGridSize(QSize(90, 100))
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setMovement(QListWidget.Movement.Static)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        self.setMinimumWidth(180)
        self.setMaximumWidth(250)

    def set_folder(self, folder: str):
        if folder == self._folder:
            return
        self._folder = folder
        self._refresh()

    def _refresh(self):
        if self._thumb_loader:
            self._thumb_loader.stop()
            self._thumb_loader.wait()

        self._list.clear()
        self._path_to_item.clear()
        self._paths = list_images_in_folder(self._folder)
        self._count_label.setText(f"{len(self._paths)} images")

        for path in self._paths:
            name = os.path.basename(path)
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setSizeHint(QSize(90, 100))
            self._list.addItem(item)
            self._path_to_item[path] = item

        self._thumb_loader = ThumbnailLoader(self._paths, size=80)
        self._thumb_loader.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._thumb_loader.start()

    def _on_thumbnail_ready(self, path: str, pixmap: QPixmap):
        item = self._path_to_item.get(path)
        if item:
            item.setIcon(QIcon(pixmap))

    def _on_item_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.image_selected.emit(path)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.image_selected.emit(path)

    def _filter_list(self, text: str):
        text = text.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            name = os.path.basename(path).lower()
            item.setHidden(bool(text) and text not in name)

    def highlight_path(self, path: str):
        item = self._path_to_item.get(path)
        if item:
            self._list.setCurrentItem(item)
            self._list.scrollToItem(item)
