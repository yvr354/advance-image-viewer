"""
Image Converter — single file and batch folder conversion.
Opened from File menu → Image Converter…
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QSpinBox, QFileDialog, QTextEdit,
    QProgressBar, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut, QFont


# ── Supported formats ──────────────────────────────────────────────────────

FORMATS = {
    "JPG  — JPEG (lossy)":  ("jpg",  True),
    "PNG  — Lossless":       ("png",  False),
    "BMP  — Bitmap":         ("bmp",  False),
    "TIFF — Lossless":       ("tiff", False),
    "PPM  — Portable Pixmap":("ppm",  False),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
              ".ppm", ".pgm", ".exr"}

# ── Style ──────────────────────────────────────────────────────────────────

_BG     = "#0A1218"
_BG2    = "#0E1820"
_BG3    = "#111C28"
_BORDER = "#1A2A3A"
_TEXT   = "#AABBCC"
_DIM    = "#445566"
_CYAN   = "#00B4D8"
_GREEN  = "#2ECC71"
_AMBER  = "#D4840A"
_RED    = "#E05555"

_CSS = f"""
    QDialog, QWidget  {{ background:{_BG}; color:{_TEXT}; font-family:Segoe UI; font-size:11px; }}
    QLabel            {{ color:{_TEXT}; }}
    QLineEdit         {{ background:{_BG2}; border:1px solid {_BORDER}; border-radius:3px;
                         color:{_TEXT}; padding:4px 8px; }}
    QLineEdit:focus   {{ border-color:{_CYAN}; }}
    QComboBox         {{ background:{_BG2}; border:1px solid {_BORDER}; border-radius:3px;
                         color:{_TEXT}; padding:3px 8px; }}
    QComboBox QAbstractItemView {{ background:{_BG2}; color:{_TEXT}; border:1px solid {_BORDER}; }}
    QSpinBox          {{ background:{_BG2}; border:1px solid {_BORDER}; border-radius:3px;
                         color:{_TEXT}; padding:3px 8px; }}
    QTextEdit         {{ background:{_BG3}; border:1px solid {_BORDER}; border-radius:3px;
                         color:{_TEXT}; font-family:Consolas; font-size:10px; }}
    QProgressBar      {{ background:{_BG2}; border:1px solid {_BORDER}; border-radius:3px;
                         height:6px; text-align:center; }}
    QProgressBar::chunk {{ background:{_CYAN}; border-radius:3px; }}
    QPushButton       {{ background:{_BG2}; color:{_TEXT}; border:1px solid {_BORDER};
                         border-radius:3px; padding:5px 16px; }}
    QPushButton:hover {{ background:#111F2E; border-color:{_CYAN}; }}
    QPushButton:disabled {{ color:{_DIM}; border-color:#111A22; }}
    QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color:{_BORDER}; }}
"""

def _btn(text, color=None, bold=False):
    b = QPushButton(text)
    style = ""
    if color:
        style = f"background:{color}; color:white; border:none;"
    if bold:
        style += "font-weight:700;"
    if style:
        b.setStyleSheet(style + "border-radius:3px; padding:5px 16px;")
    return b

def _label(text, dim=False, bold=False):
    l = QLabel(text)
    style = f"color:{_DIM};" if dim else f"color:{_TEXT};"
    if bold:
        style += "font-weight:700;"
    l.setStyleSheet(style)
    return l

def _sep():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{_BORDER};")
    return f


# ── Worker signals ─────────────────────────────────────────────────────────

class _Signals(QObject):
    log      = pyqtSignal(str)
    progress = pyqtSignal(int, int)   # done, total
    finished = pyqtSignal(int, int)   # ok, failed


# ── Conversion logic ───────────────────────────────────────────────────────

def _output_name(src_path: Path, ext: str, quality: int) -> Path:
    stem = src_path.stem
    if ext in ("jpg", "jpeg"):
        stem = f"{stem}_{quality}"
    return src_path.with_name(stem + "." + ext)


def _convert_one(src: Path, out: Path, ext: str, quality: int) -> str | None:
    try:
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            return f"Cannot read: {src.name}"
        if ext in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        else:
            params = []
        ok = cv2.imwrite(str(out), img, params)
        if not ok:
            return f"Write failed: {out.name}"
        return None
    except Exception as e:
        return str(e)


# ── Main dialog ────────────────────────────────────────────────────────────

class ConverterDialog(QDialog):
    def __init__(self, current_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Converter")
        self.resize(620, 640)
        self.setStyleSheet(_CSS)
        self._current_path = current_path
        self._running = False
        self._build()
        QShortcut(QKeySequence("Escape"), self, self.close)

    # ── UI build ───────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        # Header
        hdr = _label("Image Converter", bold=True)
        hdr.setStyleSheet(f"color:{_CYAN}; font-size:15px; font-weight:700;")
        root.addWidget(hdr)
        root.addWidget(_sep())

        # ── Format + Quality row ───────────────────────────────────
        fq = QHBoxLayout()
        fq.setSpacing(16)

        fq.addWidget(_label("Output Format:"))
        self._fmt_combo = QComboBox()
        for label in FORMATS:
            self._fmt_combo.addItem(label)
        self._fmt_combo.setFixedWidth(220)
        self._fmt_combo.currentIndexChanged.connect(self._on_format_change)
        fq.addWidget(self._fmt_combo)

        fq.addSpacing(20)
        self._quality_label = _label("Quality %:")
        fq.addWidget(self._quality_label)
        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(1, 100)
        self._quality_spin.setValue(90)
        self._quality_spin.setFixedWidth(70)
        self._quality_spin.setSuffix(" %")
        fq.addWidget(self._quality_spin)
        fq.addStretch()
        root.addLayout(fq)

        # Quality note
        self._quality_note = _label("", dim=True)
        root.addWidget(self._quality_note)
        self._on_format_change()

        root.addWidget(_sep())

        # ── Single File ────────────────────────────────────────────
        root.addWidget(_label("Single File", bold=True))

        sf_row = QHBoxLayout()
        sf_row.setSpacing(6)
        self._single_src = QLineEdit()
        self._single_src.setPlaceholderText("Select image file…")
        if self._current_path:
            self._single_src.setText(self._current_path)
        sf_row.addWidget(self._single_src)
        btn_sf = _btn("Browse…")
        btn_sf.setFixedWidth(80)
        btn_sf.clicked.connect(self._browse_single)
        sf_row.addWidget(btn_sf)
        root.addLayout(sf_row)

        so_row = QHBoxLayout()
        so_row.setSpacing(6)
        self._single_out = QLineEdit()
        self._single_out.setPlaceholderText("Output folder (same as source if empty)…")
        so_row.addWidget(self._single_out)
        btn_so = _btn("Browse…")
        btn_so.setFixedWidth(80)
        btn_so.clicked.connect(self._browse_single_out)
        so_row.addWidget(btn_so)
        root.addLayout(so_row)

        self._btn_single = _btn("  Convert File  ", color="#00B4D8", bold=True)
        self._btn_single.setFixedWidth(160)
        self._btn_single.clicked.connect(self._run_single)
        root.addWidget(self._btn_single)

        root.addWidget(_sep())

        # ── Batch Folder ───────────────────────────────────────────
        root.addWidget(_label("Batch Folder", bold=True))

        bf_row = QHBoxLayout()
        bf_row.setSpacing(6)
        self._batch_src = QLineEdit()
        self._batch_src.setPlaceholderText("Select input folder…")
        bf_row.addWidget(self._batch_src)
        btn_bf = _btn("Browse…")
        btn_bf.setFixedWidth(80)
        btn_bf.clicked.connect(self._browse_batch)
        bf_row.addWidget(btn_bf)
        root.addLayout(bf_row)

        bo_row = QHBoxLayout()
        bo_row.setSpacing(6)
        self._batch_out = QLineEdit()
        self._batch_out.setPlaceholderText("Output folder (required — originals not touched)…")
        bo_row.addWidget(self._batch_out)
        btn_bo = _btn("Browse…")
        btn_bo.setFixedWidth(80)
        btn_bo.clicked.connect(self._browse_batch_out)
        bo_row.addWidget(btn_bo)
        root.addLayout(bo_row)

        self._btn_batch = _btn("  Convert Folder  ", color="#00B4D8", bold=True)
        self._btn_batch.setFixedWidth(160)
        self._btn_batch.clicked.connect(self._run_batch)
        root.addWidget(self._btn_batch)

        root.addWidget(_sep())

        # ── Progress + Log ─────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setFixedHeight(8)
        root.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(120)
        self._log.setFont(QFont("Consolas", 9))
        root.addWidget(self._log)

        # ── Bottom buttons ─────────────────────────────────────────
        bot = QHBoxLayout()
        bot.addStretch()
        close_btn = _btn("Close")
        close_btn.clicked.connect(self.close)
        bot.addWidget(close_btn)
        root.addLayout(bot)

    # ── Format change ──────────────────────────────────────────────────────

    def _on_format_change(self):
        label = self._fmt_combo.currentText()
        ext, has_quality = FORMATS[label]
        self._quality_spin.setEnabled(has_quality)
        self._quality_label.setStyleSheet(
            f"color:{_TEXT};" if has_quality else f"color:{_DIM};")
        if ext == "jpg":
            note = "JPG — lossy. Lower % = smaller file, more artefacts. 80% recommended for inspection."
        elif ext == "png":
            note = "PNG — always lossless, always maximum compression. No quality setting needed."
        else:
            note = "Lossless format — quality setting not used, file saved as-is."
        self._quality_note.setText(note)

    # ── Browse helpers ─────────────────────────────────────────────────────

    def _browse_single(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.ppm *.pgm *.exr)")
        if f:
            self._single_src.setText(f)

    def _browse_single_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self._single_out.setText(d)

    def _browse_batch(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if d:
            self._batch_src.setText(d)

    def _browse_batch_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self._batch_out.setText(d)

    # ── Current format / quality ───────────────────────────────────────────

    def _get_ext_quality(self):
        label = self._fmt_combo.currentText()
        ext, has_quality = FORMATS[label]
        quality = self._quality_spin.value() if has_quality else 100
        return ext, quality

    # ── Single convert ─────────────────────────────────────────────────────

    def _run_single(self):
        src_str = self._single_src.text().strip()
        if not src_str:
            self._log_line("ERROR: No input file selected.", error=True)
            return
        src = Path(src_str)
        if not src.exists():
            self._log_line(f"ERROR: File not found: {src}", error=True)
            return

        ext, quality = self._get_ext_quality()
        out_dir_str = self._single_out.text().strip()
        out_dir = Path(out_dir_str) if out_dir_str else src.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out = _output_name(src, ext, quality)
        out = out_dir / out.name

        self._log_line(f"Converting: {src.name}  →  {out.name}")
        self._progress.setValue(0)

        err = _convert_one(src, out, ext, quality)
        if err:
            self._log_line(f"FAILED: {err}", error=True)
        else:
            size_kb = out.stat().st_size / 1024
            self._log_line(
                f"✓  Saved: {out.name}  ({size_kb:.0f} KB)", ok=True)
        self._progress.setValue(100)

    # ── Batch convert ──────────────────────────────────────────────────────

    def _run_batch(self):
        if self._running:
            return
        src_str = self._batch_src.text().strip()
        out_str = self._batch_out.text().strip()

        if not src_str:
            self._log_line("ERROR: No input folder selected.", error=True)
            return
        if not out_str:
            self._log_line("ERROR: Output folder required for batch (originals not touched).", error=True)
            return

        src_dir = Path(src_str)
        out_dir = Path(out_str)

        if not src_dir.exists():
            self._log_line(f"ERROR: Input folder not found: {src_dir}", error=True)
            return

        files = [f for f in src_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not files:
            self._log_line("ERROR: No supported images found in folder.", error=True)
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        ext, quality = self._get_ext_quality()

        self._running = True
        self._btn_batch.setEnabled(False)
        self._btn_single.setEnabled(False)
        self._progress.setValue(0)
        self._log_line(f"Batch: {len(files)} images  →  {out_dir.name}/")

        sig = _Signals()
        sig.log.connect(self._log_line)
        sig.progress.connect(self._on_progress)
        sig.finished.connect(self._on_batch_done)

        def worker():
            ok = failed = 0
            total = len(files)
            for i, f in enumerate(sorted(files)):
                out = _output_name(f, ext, quality)
                out = out_dir / out.name
                err = _convert_one(f, out, ext, quality)
                if err:
                    sig.log.emit(f"  ✗ {f.name} — {err}")
                    failed += 1
                else:
                    sig.log.emit(f"  ✓ {f.name}  →  {out.name}")
                    ok += 1
                sig.progress.emit(i + 1, total)
            sig.finished.emit(ok, failed)

        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, done: int, total: int):
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_batch_done(self, ok: int, failed: int):
        self._running = False
        self._btn_batch.setEnabled(True)
        self._btn_single.setEnabled(True)
        color = "ok" if failed == 0 else "error"
        self._log_line(
            f"Done — {ok} converted, {failed} failed.", ok=(failed == 0), error=(failed > 0))

    # ── Log ────────────────────────────────────────────────────────────────

    def _log_line(self, text: str, ok: bool = False, error: bool = False):
        if ok:
            html = f'<span style="color:{_GREEN};">{text}</span>'
        elif error:
            html = f'<span style="color:{_RED};">{text}</span>'
        else:
            html = f'<span style="color:{_TEXT};">{text}</span>'
        self._log.append(html)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())
