"""
VyuhaAI — Batch Analyze Dialog
Processes an entire folder of images, runs focus + quality analysis on each,
shows a live results table, and allows CSV / PDF export of the consolidated report.
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QFrame, QApplication,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from src.core.image_loader import load_image, list_images_in_folder
from src.analysis.focus_engine import FocusEngine
from src.analysis.quality_engine import QualityEngine
from src.export.report_exporter import ImageRecord, export_csv, export_pdf


# ── Worker ─────────────────────────────────────────────────────────────────

class BatchWorker(QThread):
    """Process images one-by-one in background; emit result after each."""
    image_done    = pyqtSignal(object)   # ImageRecord
    image_failed  = pyqtSignal(str, str) # filename, error
    all_done      = pyqtSignal()

    def __init__(self, paths: list[str], focus: FocusEngine, quality: QualityEngine):
        super().__init__()
        self._paths   = paths
        self._focus   = focus
        self._quality = quality
        self._stop    = False

    def stop(self):
        self._stop = True

    def run(self):
        for path in self._paths:
            if self._stop:
                break
            fname = os.path.basename(path)
            try:
                data          = load_image(path)
                focus_result  = self._focus.analyze(data.raw)
                quality_result = self._quality.analyze(data.raw)
                record = ImageRecord.from_analysis(data, focus_result, quality_result)
                self.image_done.emit(record)
            except Exception as e:
                self.image_failed.emit(fname, str(e))
        self.all_done.emit()


# ── Dialog ─────────────────────────────────────────────────────────────────

_DEC_COLORS = {
    "ACCEPT": "#00C853",
    "REVIEW": "#FFB300",
    "REJECT": "#FF1744",
}
_VER_COLORS = {
    "PERFECT": "#00C853",
    "GOOD":    "#00B4D8",
    "SOFT":    "#FFB300",
    "BLURRY":  "#FF1744",
    "PASS":    "#00C853",
    "WARN":    "#FFB300",
    "FAIL":    "#FF1744",
}

COLS = [
    "#", "Filename", "Decision",
    "Focus", "Score", "Quality", "Q.Score",
    "Sharp%", "Soft%", "Blurry%", "Tilt",
]


class BatchDialog(QDialog):

    def __init__(self, parent, focus_engine: FocusEngine, quality_engine: QualityEngine):
        super().__init__(parent)
        self.setWindowTitle("VyuhaAI — Batch Analyze")
        self.resize(1020, 640)
        self.setMinimumSize(780, 480)
        self.setStyleSheet("""
            QDialog   { background:#0A0A1A; color:#CCCCDD; }
            QLabel    { color:#CCCCDD; }
            QTableWidget { background:#0D0D1A; gridline-color:#1A1A2A;
                           color:#CCCCDD; border:1px solid #1A1A2A; }
            QHeaderView::section { background:#141428; color:#00B4D8;
                                   border:1px solid #1A1A2A; padding:4px; font-weight:bold; }
            QProgressBar { background:#111118; border:1px solid #222233;
                           color:#00B4D8; text-align:center; }
            QProgressBar::chunk { background:#00B4D8; }
            QPushButton { background:#1A1A2A; color:#CCCCDD; border:1px solid #2A2A3A;
                          padding:6px 14px; border-radius:3px; }
            QPushButton:hover   { background:#222233; border-color:#00B4D8; }
            QPushButton:disabled { color:#555566; border-color:#111118; }
        """)

        self._focus   = focus_engine
        self._quality = quality_engine
        self._worker: BatchWorker | None = None
        self._records: list[ImageRecord] = []
        self._total   = 0

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # ── Folder selector row ──────────────────────────────────────
        row1 = QHBoxLayout()
        self._folder_lbl = QLabel("No folder selected")
        self._folder_lbl.setStyleSheet("color:#888899; font-style:italic;")
        self._folder_lbl.setWordWrap(True)
        row1.addWidget(self._folder_lbl, stretch=1)

        self._btn_pick = QPushButton("  Choose Folder…  ")
        self._btn_pick.setStyleSheet(
            "QPushButton { background:#0A2040; color:#00B4D8; border:1px solid #00B4D8; }"
            "QPushButton:hover { background:#0D2A50; }"
        )
        self._btn_pick.clicked.connect(self._pick_folder)
        row1.addWidget(self._btn_pick)
        layout.addLayout(row1)

        # ── Summary strip ────────────────────────────────────────────
        self._summary_lbl = QLabel("")
        self._summary_lbl.setStyleSheet("color:#555566; font-size:11px;")
        layout.addWidget(self._summary_lbl)

        # ── Progress ─────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setFixedHeight(18)
        layout.addWidget(self._progress)

        # ── Table ────────────────────────────────────────────────────
        self._table = QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(COLS)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setDefaultSectionSize(80)
        self._table.setColumnWidth(0, 40)
        self._table.setColumnWidth(1, 220)
        self._table.setColumnWidth(10, 200)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { alternate-background-color:#111120; }"
        )
        layout.addWidget(self._table, stretch=1)

        # ── Counts strip ─────────────────────────────────────────────
        self._counts_lbl = QLabel("")
        self._counts_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._counts_lbl.setStyleSheet("font-size:11px;")
        layout.addWidget(self._counts_lbl)

        # ── Buttons ──────────────────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#1A1A2A;")
        layout.addWidget(sep)

        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("  ▶  Start Batch  ")
        self._btn_start.setStyleSheet(
            "QPushButton { background:#00B4D8; color:#000; font-weight:bold; border:none; }"
            "QPushButton:hover { background:#00C8F0; }"
            "QPushButton:disabled { background:#1A1A2A; color:#555566; border:none; }"
        )
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._start_batch)

        self._btn_stop = QPushButton("  ■  Stop  ")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_batch)

        self._btn_csv = QPushButton("  ↓  Export CSV  ")
        self._btn_csv.setEnabled(False)
        self._btn_csv.clicked.connect(self._export_csv)

        self._btn_pdf = QPushButton("  ↓  Export PDF  ")
        self._btn_pdf.setEnabled(False)
        self._btn_pdf.clicked.connect(self._export_pdf)

        self._btn_close = QPushButton("  Close  ")
        self._btn_close.clicked.connect(self.close)

        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_csv)
        btn_row.addWidget(self._btn_pdf)
        btn_row.addWidget(self._btn_close)
        layout.addLayout(btn_row)

    # ── Folder selection ───────────────────────────────────────────────────

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        paths = list_images_in_folder(folder)
        if not paths:
            QMessageBox.information(self, "No Images",
                "No supported images found in that folder.\n"
                "Supported: .tiff .tif .bmp .png .jpg .jpeg .pgm .ppm .exr")
            return
        self._paths = paths
        self._total = len(paths)
        self._folder_lbl.setText(folder)
        self._folder_lbl.setStyleSheet("color:#CCCCDD;")
        self._summary_lbl.setText(
            f"{self._total} images found  ·  "
            f"formats: {', '.join(sorted({os.path.splitext(p)[1].lower() for p in paths}))}"
        )
        self._progress.setMaximum(self._total)
        self._progress.setValue(0)
        self._btn_start.setEnabled(True)
        self._records.clear()
        self._table.setRowCount(0)
        self._counts_lbl.setText("")
        self._btn_csv.setEnabled(False)
        self._btn_pdf.setEnabled(False)

    # ── Batch run ──────────────────────────────────────────────────────────

    def _start_batch(self):
        self._records.clear()
        self._table.setRowCount(0)
        self._counts_lbl.setText("")
        self._progress.setValue(0)
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_csv.setEnabled(False)
        self._btn_pdf.setEnabled(False)
        self._btn_pick.setEnabled(False)

        self._worker = BatchWorker(self._paths, self._focus, self._quality)
        self._worker.image_done.connect(self._on_image_done)
        self._worker.image_failed.connect(self._on_image_failed)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

    def _stop_batch(self):
        if self._worker:
            self._worker.stop()
        self._btn_stop.setEnabled(False)

    # ── Per-image result ───────────────────────────────────────────────────

    def _on_image_done(self, record: ImageRecord):
        self._records.append(record)
        row = self._table.rowCount()
        self._table.insertRow(row)
        dec = record.overall_decision()

        values = [
            str(row + 1),
            record.filename,
            dec,
            record.focus_verdict,
            f"{record.focus_score:.0f}",
            record.quality_verdict,
            f"{record.quality_score:.0f}",
            f"{record.pct_sharp:.0f}%",
            f"{record.pct_soft:.0f}%",
            f"{record.pct_blurry:.0f}%",
            record.tilt_warning or "—",
        ]
        colors = [
            None, None,
            _DEC_COLORS.get(dec),
            _VER_COLORS.get(record.focus_verdict),
            None,
            _VER_COLORS.get(record.quality_verdict),
            None, None, None, None,
            "#FFB300" if record.tilt_warning else None,
        ]

        for col, (val, color) in enumerate(zip(values, colors)):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if color:
                item.setForeground(QColor(color))
                if col == 2:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
            self._table.setItem(row, col, item)

        self._table.scrollToBottom()
        self._progress.setValue(row + 1)
        self._update_counts()

    def _on_image_failed(self, fname: str, error: str):
        row = self._table.rowCount()
        self._table.insertRow(row)
        item = QTableWidgetItem(fname)
        item.setForeground(QColor("#FF1744"))
        self._table.setItem(row, 1, item)
        err_item = QTableWidgetItem(f"ERROR: {error}")
        err_item.setForeground(QColor("#FF5566"))
        self._table.setItem(row, 2, err_item)
        self._progress.setValue(row + 1)

    def _on_all_done(self):
        self._btn_stop.setEnabled(False)
        self._btn_start.setEnabled(True)
        self._btn_pick.setEnabled(True)
        if self._records:
            self._btn_csv.setEnabled(True)
            self._btn_pdf.setEnabled(True)
        self._update_counts()
        n = len(self._records)
        self._summary_lbl.setText(
            f"Batch complete — {n} of {self._total} images analyzed"
        )

    def _update_counts(self):
        if not self._records:
            return
        accept = sum(1 for r in self._records if r.overall_decision() == "ACCEPT")
        review = sum(1 for r in self._records if r.overall_decision() == "REVIEW")
        reject = sum(1 for r in self._records if r.overall_decision() == "REJECT")
        self._counts_lbl.setText(
            f'<span style="color:#00C853">ACCEPT: {accept}</span>  '
            f'<span style="color:#FFB300">REVIEW: {review}</span>  '
            f'<span style="color:#FF1744">REJECT: {reject}</span>  '
            f'<span style="color:#888899">({len(self._records)} total)</span>'
        )
        self._counts_lbl.setTextFormat(Qt.TextFormat.RichText)

    # ── Export ─────────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._records:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Batch CSV Report", "batch_report.csv", "CSV (*.csv)")
        if not path:
            return
        export_csv(self._records, path)
        QMessageBox.information(self, "Exported",
            f"CSV report saved:\n{path}\n\n{len(self._records)} images")

    def _export_pdf(self):
        if not self._records:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Batch PDF Report", "batch_report.pdf",
            "PDF (*.pdf);;HTML (*.html)")
        if not path:
            return
        result = export_pdf(self._records, path)
        QMessageBox.information(self, "Exported",
            f"Report saved:\n{result}\n\n{len(self._records)} images")

    # ── Cleanup ────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(event)
