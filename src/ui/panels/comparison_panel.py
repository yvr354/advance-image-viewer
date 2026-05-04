"""
Professional A/B Image Comparison Panel with Defect Detection.

Layout:
  Load bar       — load A/B, swap, active indicator
  Controls bar   — mode, threshold, min-area, amplify
  Verdict bar    — big PASS/FAIL + SSIM / PSNR / diff% / defect count
  Main area      — A panel | B panel | Diff+boxes | Defect list
  Pixel loupe    — hover to inspect A vs B pixel side by side

Detection pipeline (industry standard — Cognex/Keyence pattern):
  diff_raw → gaussian blur → threshold → morph open → connected components
  → filter by area → classify by area → draw bounding boxes

Severity thresholds (area in pixels):
  COSMETIC   < 25 px²   green
  FUNCTIONAL 25–200 px² yellow
  CRITICAL   > 200 px²  red
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QSlider, QComboBox,
    QFileDialog, QScrollArea, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QDragEnterEvent, QDropEvent

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

from src.core.image_loader import load_image
from src.ui.panels.gl_viewer import GLImageViewer
from src.ui.theme import ACCENT, TEXT_PRIMARY, TEXT_SECONDARY, BG_RAISED, BG_CONTROL


# ─────────────────────────────────────────────────────────────────────────────
# Defect data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Defect:
    idx:      int
    x:        int    # bbox top-left
    y:        int
    w:        int    # bbox size
    h:        int
    cx:       int    # centroid
    cy:       int
    area:     int    # pixels²
    severity: str    # COSMETIC / FUNCTIONAL / CRITICAL

    @property
    def color_bgr(self):
        return {"COSMETIC": (50,180,50), "FUNCTIONAL": (0,170,255),
                "CRITICAL": (50,50,220)}[self.severity]

    @property
    def color_hex(self):
        return {"COSMETIC": "#44BB44", "FUNCTIONAL": "#FFAA00",
                "CRITICAL": "#FF4444"}[self.severity]


def _classify_severity(area: int) -> str:
    if area < 25:
        return "COSMETIC"
    if area < 200:
        return "FUNCTIONAL"
    return "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# Background worker — diff + detection
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisWorker(QThread):
    """Runs ALL display computation off the main thread (diff, blend, signed)."""

    done = pyqtSignal(
        np.ndarray,   # display image for diff viewer
        np.ndarray,   # annotated image (boxes drawn); same as display for non-diff
        dict,         # metrics
        list,         # defects (List[Defect])
    )

    def __init__(self, img_a, img_b, mode: str,
                 threshold: int, min_area: int, amp: int):
        super().__init__()
        self._a         = img_a
        self._b         = img_b
        self._mode      = mode
        self._threshold = threshold
        self._min_area  = min_area
        self._amp       = amp

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to8(img):
        if img.dtype == np.uint16:
            return (img >> 8).astype(np.uint8)
        if img.dtype != np.uint8:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(self):
        try:
            a = self._to8(self._a)
            b = self._to8(self._b)

            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

            # Convert to grayscale for normalize + align (shared by all modes)
            ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
            bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY) if b.ndim == 3 else b

            # ── 0a. Normalize brightness — match B mean luminance to A ────
            mean_a = float(np.mean(ag))
            mean_b = float(np.mean(bg))
            if mean_b > 1:
                scale = float(np.clip(mean_a / mean_b, 0.5, 2.0))
                bg = np.clip(bg.astype(np.float32) * scale, 0, 255).astype(np.uint8)
                b  = np.clip(b.astype(np.float32)  * scale, 0, 255).astype(np.uint8)

            # ── 0b. Auto-align B onto A using ORB feature matching ────────
            did_align = False
            try:
                orb  = cv2.ORB_create(500)
                kp1, des1 = orb.detectAndCompute(ag, None)
                kp2, des2 = orb.detectAndCompute(bg, None)
                if (des1 is not None and des2 is not None
                        and len(kp1) >= 4 and len(kp2) >= 4):
                    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = sorted(bf.match(des1, des2),
                                     key=lambda m: m.distance)[:50]
                    if len(matches) >= 4:
                        src = np.float32(
                            [kp2[m.trainIdx].pt for m in matches]
                        ).reshape(-1, 1, 2)
                        dst = np.float32(
                            [kp1[m.queryIdx].pt for m in matches]
                        ).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                        if H is not None:
                            h, w = ag.shape
                            bg = cv2.warpPerspective(bg, H, (w, h))
                            b  = cv2.warpPerspective(b,  H, (w, h))
                            did_align = True
            except Exception:
                pass

            _empty = {"pixel_diff": 0, "max_dev": 0, "mean_dev": 0,
                      "ssim": None, "psnr": None, "n_defects": 0,
                      "auto_norm": False, "aligned": did_align, "verdict": "—"}

            # ── Blend: early return ───────────────────────────────────────
            if self._mode == "Blend 50/50":
                ra  = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) if a.ndim == 2 else a
                rb  = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB) if b.ndim == 2 else b
                out = cv2.addWeighted(ra, 0.5, rb, 0.5, 0)
                self.done.emit(out, out, _empty, [])
                return

            # ── Signed ±128: early return ─────────────────────────────────
            if self._mode == "Signed ±128":
                signed = np.clip(
                    ag.astype(np.int16) - bg.astype(np.int16) + 128, 0, 255
                ).astype(np.uint8)
                out = cv2.cvtColor(
                    cv2.applyColorMap(signed, cv2.COLORMAP_COOLWARM),
                    cv2.COLOR_BGR2RGB,
                )
                self.done.emit(out, out, _empty, [])
                return

            # ── 1. Raw absolute difference ────────────────────────────────
            diff_raw = cv2.absdiff(ag.astype(np.int16),
                                   bg.astype(np.int16)).astype(np.float32)

            # ── 2. Display diff (amplified + auto-normalize) ──────────────
            amplified = diff_raw * self._amp
            amp_max   = float(np.max(amplified))
            auto_norm = False
            if 0 < amp_max < 32:
                amplified = amplified * (255.0 / amp_max)
                auto_norm = True
            amp8       = np.clip(amplified, 0, 255).astype(np.uint8)
            diff_rgb   = cv2.cvtColor(
                cv2.applyColorMap(amp8, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB
            )

            # ── 3. Detection pipeline ─────────────────────────────────────
            blurred = cv2.GaussianBlur(diff_raw, (5, 5), 1.0)
            binary  = (blurred > self._threshold).astype(np.uint8) * 255
            kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
            closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

            n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
                closed, connectivity=8
            )

            # ── 4. Filter blobs + classify ────────────────────────────────
            defects: List[Defect] = []
            for i in range(1, n_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < self._min_area or area > 200_000:
                    continue
                x  = int(stats[i, cv2.CC_STAT_LEFT])
                y  = int(stats[i, cv2.CC_STAT_TOP])
                w  = int(stats[i, cv2.CC_STAT_WIDTH])
                h  = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx = int(centroids[i][0])
                cy = int(centroids[i][1])
                defects.append(Defect(
                    idx=len(defects)+1, x=x, y=y, w=w, h=h,
                    cx=cx, cy=cy, area=area,
                    severity=_classify_severity(area),
                ))

            _order = {"CRITICAL": 0, "FUNCTIONAL": 1, "COSMETIC": 2}
            defects.sort(key=lambda d: (_order[d.severity], -d.area))
            for i, d in enumerate(defects):
                d.idx = i + 1

            # ── 5. Draw bounding boxes ────────────────────────────────────
            annot = diff_rgb.copy()
            for d in defects:
                cv2.rectangle(annot, (d.x, d.y), (d.x+d.w, d.y+d.h),
                              d.color_bgr, 2)
                label = f"#{d.idx} {d.severity[:4]}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                ly = max(d.y - 4, th + 2)
                cv2.rectangle(annot, (d.x, ly-th-2), (d.x+tw+4, ly+2),
                              d.color_bgr, -1)
                cv2.putText(annot, label, (d.x+2, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # ── 6. Metrics ────────────────────────────────────────────────
            pct_diff = float(np.count_nonzero(diff_raw > 5)) / ag.size * 100
            has_critical   = any(d.severity == "CRITICAL"   for d in defects)
            has_functional = any(d.severity == "FUNCTIONAL" for d in defects)

            metrics = {
                "pixel_diff": pct_diff,
                "max_dev":    int(np.max(diff_raw)),
                "mean_dev":   float(np.mean(diff_raw)),
                "ssim":       None,
                "psnr":       None,
                "n_defects":  len(defects),
                "auto_norm":  auto_norm,
                "aligned":    did_align,
                "verdict":    "FAIL" if (has_critical or has_functional) else "PASS",
            }

            if _SKIMAGE:
                try:
                    s = float(structural_similarity(ag, bg, data_range=255))
                    p = float(peak_signal_noise_ratio(ag, bg, data_range=255))
                    metrics["ssim"] = s
                    metrics["psnr"] = p
                    if not (has_critical or has_functional):
                        metrics["verdict"] = "PASS" if s >= 0.92 else "FAIL"
                except Exception:
                    pass

            self.done.emit(diff_rgb, annot, metrics, defects)

        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Slot panel (A or B)
# ─────────────────────────────────────────────────────────────────────────────

class SlotPanel(QWidget):
    activated      = pyqtSignal(str)         # 'A' or 'B'
    load_requested = pyqtSignal(str, str)    # slot, path (drag-drop)

    _ACTIVE_HDR   = ("background:#0E2438; color:#00AAFF; "
                     "padding:3px 8px; font-weight:700; font-size:10px;")
    _INACTIVE_HDR = ("background:#0A1520; color:#445566; "
                     "padding:3px 8px; font-size:10px;")

    def __init__(self, slot: str, parent=None):
        super().__init__(parent)
        self.slot = slot
        self._path = ""
        self._w = self._h = 0
        self.setObjectName("SlotPanel")
        self.setAcceptDrops(True)
        self._build()
        self._set_active(False)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        lbl_text = "REFERENCE" if self.slot == "A" else "TEST IMAGE"
        self._header = QLabel(f"  {self.slot} — {lbl_text}   ·   click to activate")
        self._header.setFixedHeight(24)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.mousePressEvent = lambda _: self.activated.emit(self.slot)
        lay.addWidget(self._header)

        self.viewer = GLImageViewer()
        orig = self.viewer.mousePressEvent
        def _fwd(ev, _o=orig):
            self.activated.emit(self.slot)
            _o(ev)
        self.viewer.mousePressEvent = _fwd
        lay.addWidget(self.viewer, stretch=1)

    def set_active(self, active: bool):
        self._set_active(active)
        self.setStyleSheet(
            "QWidget#SlotPanel { border: 2px solid #00AAFF; }" if active else
            "QWidget#SlotPanel { border: 2px solid #1A2535; }"
        )

    def _set_active(self, active: bool):
        self._header.setStyleSheet(
            self._ACTIVE_HDR if active else self._INACTIVE_HDR
        )

    def set_image(self, path: str, img: np.ndarray):
        self._path = path
        self._h, self._w = img.shape[:2]
        self.viewer.set_image(img)
        self._refresh_header()

    def _refresh_header(self):
        name = os.path.basename(self._path)
        if len(name) > 26:
            name = name[:13] + "…" + name[-11:]
        lbl = "REFERENCE" if self.slot == "A" else "TEST IMAGE"
        self._header.setText(
            f"  {self.slot} — {lbl}   ·   {name}   {self._w}×{self._h} px"
        )

    def update_zoom(self, zoom: float):
        if not self._path:
            return
        name = os.path.basename(self._path)
        if len(name) > 26:
            name = name[:13] + "…" + name[-11:]
        lbl = "REFERENCE" if self.slot == "A" else "TEST IMAGE"
        self._header.setText(
            f"  {self.slot} — {lbl}   ·   {name}   "
            f"{self._w}×{self._h} px   {zoom*100:.0f}%"
        )

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.activated.emit(self.slot)
                self.load_requested.emit(self.slot, path)
                break


# ─────────────────────────────────────────────────────────────────────────────
# Defect list panel
# ─────────────────────────────────────────────────────────────────────────────

class DefectListPanel(QWidget):
    """Scrollable list of detected defects with severity color coding."""

    jump_to = pyqtSignal(int, int)   # cx, cy — jump diff viewer to defect

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(170)
        self.setMaximumWidth(240)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        hdr = QLabel("  DEFECTS")
        hdr.setFixedHeight(24)
        hdr.setStyleSheet(
            "background:#0A1520; color:#445566; "
            "padding:3px 8px; font-size:10px; font-weight:700;"
        )
        root.addWidget(hdr)
        self._hdr = hdr

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:#060C12;")

        self._container = QWidget()
        self._container.setStyleSheet("background:#060C12;")
        self._list_lay = QVBoxLayout(self._container)
        self._list_lay.setContentsMargins(4, 4, 4, 4)
        self._list_lay.setSpacing(4)
        self._list_lay.addStretch()

        scroll.setWidget(self._container)
        root.addWidget(scroll, stretch=1)

    def clear(self):
        while self._list_lay.count() > 1:   # keep trailing stretch
            item = self._list_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._hdr.setText("  DEFECTS")
        self._hdr.setStyleSheet(
            "background:#0A1520; color:#445566; "
            "padding:3px 8px; font-size:10px; font-weight:700;"
        )

    def show_defects(self, defects: List[Defect]):
        self.clear()
        if not defects:
            lbl = QLabel("  No defects detected")
            lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:10px; padding:8px;")
            lbl.setWordWrap(True)
            self._list_lay.insertWidget(0, lbl)
            self._hdr.setText("  DEFECTS — 0 found")
            return

        # Count by severity
        n_crit = sum(1 for d in defects if d.severity == "CRITICAL")
        n_func = sum(1 for d in defects if d.severity == "FUNCTIONAL")
        n_cosm = sum(1 for d in defects if d.severity == "COSMETIC")
        parts  = []
        if n_crit: parts.append(f"{n_crit} critical")
        if n_func: parts.append(f"{n_func} functional")
        if n_cosm: parts.append(f"{n_cosm} cosmetic")
        self._hdr.setText(f"  DEFECTS — {', '.join(parts)}")
        self._hdr.setStyleSheet(
            "background:#1A0808; color:#FF6666; "
            "padding:3px 8px; font-size:10px; font-weight:700;"
            if n_crit else
            "background:#1A1000; color:#FFAA44; "
            "padding:3px 8px; font-size:10px; font-weight:700;"
        )

        for d in defects:
            card = self._make_card(d)
            self._list_lay.insertWidget(self._list_lay.count()-1, card)

    def _make_card(self, d: Defect) -> QWidget:
        card = QWidget()
        card.setStyleSheet(
            f"QWidget {{ background:#0C1824; border-left:3px solid {d.color_hex}; "
            f"border-radius:2px; }}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 4, 6, 4)
        lay.setSpacing(2)

        sev_lbl = QLabel(f"#{d.idx}  {d.severity}")
        sev_lbl.setStyleSheet(
            f"color:{d.color_hex}; font-size:10px; font-weight:700; border:none;"
        )
        lay.addWidget(sev_lbl)

        pos_lbl = QLabel(f"X: {d.cx}   Y: {d.cy}")
        pos_lbl.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:9px; border:none;"
        )
        lay.addWidget(pos_lbl)

        area_lbl = QLabel(f"Area: {d.area} px²   {d.w}×{d.h}")
        area_lbl.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:9px; border:none;"
        )
        lay.addWidget(area_lbl)

        jump_btn = QPushButton("→ Jump to defect")
        jump_btn.setFixedHeight(20)
        jump_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{d.color_hex}; "
            f"border:1px solid {d.color_hex}44; border-radius:2px; "
            f"font-size:9px; padding:0 4px; }}"
            f"QPushButton:hover {{ background:{d.color_hex}22; }}"
        )
        jump_btn.clicked.connect(lambda _, cx=d.cx, cy=d.cy: self.jump_to.emit(cx, cy))
        lay.addWidget(jump_btn)

        return card


# ─────────────────────────────────────────────────────────────────────────────
# Pixel loupe
# ─────────────────────────────────────────────────────────────────────────────

class PixelLoupe(QWidget):
    PATCH_PX = 16
    DISP_PX  = 80

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_a = None
        self._img_b = None
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(12)

        def _col(title):
            col = QVBoxLayout()
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:9px;")
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img = QLabel()
            img.setFixedSize(self.DISP_PX, self.DISP_PX)
            img.setStyleSheet("background:#08080F; border:1px solid #1A2535;")
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("—")
            val.setStyleSheet(
                f"color:{TEXT_PRIMARY}; font-size:9px; font-weight:600;"
            )
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(t)
            col.addWidget(img)
            col.addWidget(val)
            return col, img, val

        col_a, self._img_a_lbl, self._val_a = _col("A — pixel")
        col_b, self._img_b_lbl, self._val_b = _col("B — pixel")

        delta = QVBoxLayout()
        delta.setSpacing(2)
        dt = QLabel("Δ A−B")
        dt.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:9px;")
        dt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._delta_lbl = QLabel("—")
        self._delta_lbl.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:10px; font-weight:700;"
        )
        self._delta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._delta_lbl.setWordWrap(True)
        self._delta_lbl.setFixedWidth(70)
        delta.addWidget(dt)
        delta.addStretch()
        delta.addWidget(self._delta_lbl)
        delta.addStretch()

        self._coord = QLabel("X: —   Y: —")
        self._coord.setStyleSheet(
            f"color:{ACCENT}; font-size:10px; font-weight:600;"
        )
        self._coord.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addLayout(col_a)
        lay.addLayout(delta)
        lay.addLayout(col_b)
        lay.addStretch()
        lay.addWidget(self._coord, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.setFixedHeight(self.DISP_PX + 44)
        self.setStyleSheet(f"background:{BG_RAISED}; border-top:1px solid #1A2535;")

    def update_hover(self, ix: int, iy: int, img_a, img_b):
        self._coord.setText(f"X: {ix}   Y: {iy}")
        pa = self._render(img_a, ix, iy, self._img_a_lbl, self._val_a)
        pb = self._render(img_b, ix, iy, self._img_b_lbl, self._val_b)
        if pa and pb:
            dr, dg, db = pa[0]-pb[0], pa[1]-pb[1], pa[2]-pb[2]
            self._delta_lbl.setText(f"R:{dr:+d}\nG:{dg:+d}\nB:{db:+d}")
        else:
            self._delta_lbl.setText("—")

    def _render(self, img, cx, cy, lbl, val_lbl):
        if img is None:
            lbl.clear(); val_lbl.setText("—"); return None
        img8 = self._to8(img)
        if img8.ndim == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        h, w = img8.shape[:2]
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return None
        half = self.PATCH_PX // 2
        x1 = max(0, cx-half); y1 = max(0, cy-half)
        x2 = min(w, x1+self.PATCH_PX); y2 = min(h, y1+self.PATCH_PX)
        patch = img8[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        sc  = self.DISP_PX // self.PATCH_PX
        big = cv2.resize(patch, (patch.shape[1]*sc, patch.shape[0]*sc),
                         interpolation=cv2.INTER_NEAREST)
        qi  = QImage(big.data, big.shape[1], big.shape[0],
                     big.shape[1]*3, QImage.Format.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qi.copy()))
        px = img8[cy, cx]
        if len(px) < 3:
            val_lbl.setText(f"L:{int(px)}")
            return (int(px),)*3
        val_lbl.setText(f"R:{px[0]} G:{px[1]} B:{px[2]}")
        return (int(px[0]), int(px[1]), int(px[2]))

    @staticmethod
    def _to8(img):
        if img.dtype == np.uint16: return (img>>8).astype(np.uint8)
        if img.dtype != np.uint8:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison panel
# ─────────────────────────────────────────────────────────────────────────────

class ComparisonPanel(QWidget):
    """
    Public API:
      load_path(path)  — load into active slot (called by browser click)
      active_slot      — 'A' or 'B'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_a:   np.ndarray | None = None
        self._img_b:   np.ndarray | None = None
        self._active   = "A"
        self._worker:  AnalysisWorker | None = None
        self._flicker_timer = QTimer()
        self._flicker_timer.timeout.connect(self._flicker_tick)
        self._flicker_state = False
        # Debounce: wait 300 ms after last change before running analysis
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._run_analysis)
        self._build()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def active_slot(self) -> str:
        return self._active

    def load_path(self, path: str):
        """Load file into active slot — called from browser or drag-drop."""
        try:
            img = load_image(path).raw
        except Exception:
            return
        self._load_into(self._active, path, img)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_load_bar())
        root.addWidget(self._build_controls_bar())
        root.addWidget(self._build_verdict_bar())
        root.addWidget(self._build_main_area(), stretch=1)
        root.addWidget(self._build_loupe())

        self._activate("A")

    # ── Load bar ──────────────────────────────────────────────────────────────

    def _build_load_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet("background:#080E18; border-bottom:1px solid #1A2535;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)

        _S = ("QPushButton { background:#0A1828; color:#5588AA; "
              "border:1px solid #1A2A3A; padding:3px 10px; "
              "font-size:10px; border-radius:3px; }"
              "QPushButton:hover { background:#112233; color:#88CCEE; }")

        self._btn_a = QPushButton("📂  Load Reference  (A)")
        self._btn_a.setToolTip("Load known-good reference image into slot A")
        self._btn_a.setStyleSheet(_S)
        self._btn_a.clicked.connect(lambda: self._browse("A"))

        self._btn_b = QPushButton("📂  Load Test  (B)")
        self._btn_b.setToolTip("Load test / inspection image into slot B")
        self._btn_b.setStyleSheet(_S)
        self._btn_b.clicked.connect(lambda: self._browse("B"))

        self._swap_btn = QPushButton("⇄  Swap A↔B")
        self._swap_btn.setStyleSheet(_S)
        self._swap_btn.setToolTip("Swap reference and test images")
        self._swap_btn.clicked.connect(self._swap)

        self._active_lbl = QLabel("Active: A")
        self._active_lbl.setStyleSheet(
            f"color:{ACCENT}; font-size:10px; font-weight:700;"
        )

        lay.addWidget(self._btn_a)
        lay.addWidget(self._btn_b)
        lay.addSpacing(8)
        lay.addWidget(self._swap_btn)
        lay.addStretch()
        lay.addWidget(self._active_lbl)
        return bar

    # ── Controls bar ──────────────────────────────────────────────────────────

    def _build_controls_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet("background:#06101A; border-bottom:1px solid #1A2535;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(10)

        _LBL = f"color:{TEXT_SECONDARY}; font-size:10px;"
        _VAL = f"color:{ACCENT}; font-size:10px; font-weight:700; min-width:24px;"

        def _slider(lo, hi, val, width=80):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            s.setFixedWidth(width)
            return s

        # Mode
        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet(_LBL)
        mode_lbl.setToolTip(
            "Diff Map    — absolute difference, hot colormap\n"
            "Signed ±128 — A−B centered at 128 gray; blue=darker, red=brighter\n"
            "Blend       — 50% A + 50% B overlay\n"
            "Flicker     — alternates A and B every 600 ms"
        )
        self._mode = QComboBox()
        self._mode.addItems(["Diff Map", "Signed ±128", "Blend 50/50", "Flicker"])
        self._mode.setFixedWidth(115)
        self._mode.setStyleSheet(
            "QComboBox { background:#0A1828; color:#88AACC; "
            "border:1px solid #1A2A3A; padding:2px 6px; "
            "font-size:10px; border-radius:3px; }"
            "QComboBox::drop-down { border:none; }"
        )
        self._mode.setToolTip(
            "Diff Map    — absolute difference, hot colormap\n"
            "Signed ±128 — A−B centered at 128 gray; blue=darker, red=brighter\n"
            "Blend       — 50% A + 50% B overlay\n"
            "Flicker     — alternates A and B every 600 ms"
        )
        self._mode.currentTextChanged.connect(self._on_mode_changed)

        # Threshold
        thr_lbl = QLabel("Threshold:")
        thr_lbl.setStyleSheet(_LBL)
        thr_lbl.setToolTip(
            "Sensitivity — pixel difference above this value counts as 'different'.\n"
            "Lower = more sensitive (more noise detected too).\n"
            "Higher = less sensitive (only strong differences flagged).\n"
            "Typical production range: 20–50."
        )
        self._thr_slider = _slider(1, 128, 30)
        self._thr_slider.setToolTip(
            "Sensitivity — pixel difference above this value counts as 'different'.\n"
            "Lower = more sensitive (more noise detected too).\n"
            "Higher = less sensitive (only strong differences flagged).\n"
            "Typical production range: 20–50."
        )
        self._thr_val = QLabel("30")
        self._thr_val.setStyleSheet(_VAL)
        self._thr_slider.valueChanged.connect(
            lambda v: (self._thr_val.setText(str(v)), self._trigger())
        )

        # Min area
        ma_lbl = QLabel("Min area:")
        ma_lbl.setStyleSheet(_LBL)
        ma_lbl.setToolTip(
            "Minimum blob size in pixels to count as a defect.\n"
            "Blobs smaller than this are treated as noise and ignored.\n"
            "Typical: 10–50 px. Increase to filter out more noise."
        )
        self._ma_slider = _slider(1, 500, 25)
        self._ma_slider.setToolTip(
            "Minimum blob size in pixels to count as a defect.\n"
            "Blobs smaller than this are treated as noise and ignored.\n"
            "Typical: 10–50 px. Increase to filter out more noise."
        )
        self._ma_val = QLabel("25 px")
        self._ma_val.setStyleSheet(_VAL)
        self._ma_slider.valueChanged.connect(
            lambda v: (self._ma_val.setText(f"{v} px"), self._trigger())
        )

        # Amplify
        amp_lbl = QLabel("Amplify:")
        amp_lbl.setStyleSheet(_LBL)
        amp_lbl.setToolTip(
            "Boost subtle pixel differences so they are visible in the diff map.\n"
            "1× = raw values   ·   20× = maximum boost\n"
            "Does NOT affect detection threshold — only the visual display."
        )
        self._amp_slider = _slider(1, 20, 4, 70)
        self._amp_slider.setToolTip(
            "Boost subtle pixel differences so they are visible in the diff map.\n"
            "1× = raw values   ·   20× = maximum boost\n"
            "Does NOT affect detection threshold — only the visual display."
        )
        self._amp_val = QLabel("×4")
        self._amp_val.setStyleSheet(_VAL)
        self._amp_slider.valueChanged.connect(
            lambda v: (self._amp_val.setText(f"×{v}"), self._trigger())
        )

        for w in [mode_lbl, self._mode,
                  thr_lbl, self._thr_slider, self._thr_val,
                  ma_lbl, self._ma_slider, self._ma_val,
                  amp_lbl, self._amp_slider, self._amp_val]:
            lay.addWidget(w)
        lay.addStretch()
        return bar

    # ── Verdict bar ───────────────────────────────────────────────────────────

    def _build_verdict_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(32)
        self._verdict_bar = bar
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(0)

        self._verdict_lbl = QLabel("Load both images to begin")
        self._verdict_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lay.addWidget(self._verdict_lbl)

        self._metrics_lbl = QLabel("")
        self._metrics_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._metrics_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:10px;")
        lay.addWidget(self._metrics_lbl)

        self._set_verdict_neutral()
        return bar

    def _set_verdict_neutral(self):
        self._verdict_bar.setStyleSheet("background:#0C1420; border-bottom:1px solid #1A2535;")
        self._verdict_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px; font-weight:700;")
        self._verdict_lbl.setText("Load both images to begin")
        self._metrics_lbl.setText("")

    def _set_analysing(self):
        self._verdict_bar.setStyleSheet("background:#0D1020; border-bottom:1px solid #334466;")
        self._verdict_lbl.setStyleSheet("color:#4488BB; font-size:11px; font-weight:700;")
        self._verdict_lbl.setText("⟳  Analysing…")
        self._metrics_lbl.setText("")

    def _set_controls_enabled(self, enabled: bool):
        for w in [self._btn_a, self._btn_b, self._swap_btn,
                  self._mode, self._thr_slider, self._ma_slider, self._amp_slider]:
            w.setEnabled(enabled)

    def _set_verdict(self, verdict: str, metrics: dict, defects: list):
        is_pass = verdict == "PASS"
        bg    = "#061A08" if is_pass else "#1A0606"
        color = "#44FF88" if is_pass else "#FF4444"
        sym   = "✓  PASS" if is_pass else "✗  FAIL"

        self._verdict_bar.setStyleSheet(
            f"background:{bg}; border-bottom:1px solid {color}44;"
        )
        self._verdict_lbl.setStyleSheet(
            f"color:{color}; font-size:11px; font-weight:700; min-width:100px;"
        )
        self._verdict_lbl.setText(sym)

        parts = []
        if metrics.get("ssim") is not None:
            parts.append(f"SSIM: {metrics['ssim']:.3f}")
        if metrics.get("psnr") is not None:
            psnr = metrics['psnr']
            parts.append(f"PSNR: {psnr:.1f} dB" if psnr < 100 else "PSNR: ∞")
        parts.append(f"Diff: {metrics['pixel_diff']:.1f}%")
        parts.append(f"Max Δ: {metrics['max_dev']}")
        n = metrics['n_defects']
        parts.append(f"Defects: {n}" if n else "Defects: none")

        self._metrics_lbl.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:10px; padding-left:20px;"
        )
        self._metrics_lbl.setText("   │   ".join(parts))

    # ── Main area ─────────────────────────────────────────────────────────────

    def _build_main_area(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self._slot_a = SlotPanel("A")
        self._slot_b = SlotPanel("B")

        # Diff viewer
        diff_wrap = QWidget()
        diff_lay  = QVBoxLayout(diff_wrap)
        diff_lay.setContentsMargins(0, 0, 0, 0)
        diff_lay.setSpacing(0)
        self._diff_hdr = QLabel("  DIFF MAP   ·   load both images")
        self._diff_hdr.setFixedHeight(24)
        self._diff_hdr.setStyleSheet(
            "background:#101A10; color:#336633; padding:3px 8px; font-size:10px;"
        )
        self._diff_viewer = GLImageViewer()
        diff_lay.addWidget(self._diff_hdr)
        diff_lay.addWidget(self._diff_viewer, stretch=1)

        self._defect_list = DefectListPanel()

        splitter.addWidget(self._slot_a)
        splitter.addWidget(self._slot_b)
        splitter.addWidget(diff_wrap)
        splitter.addWidget(self._defect_list)
        splitter.setSizes([3, 3, 3, 1])

        # Activate on click
        self._slot_a.activated.connect(self._activate)
        self._slot_b.activated.connect(self._activate)

        # Drag-drop
        self._slot_a.load_requested.connect(
            lambda _s, p: self._browse_load("A", p)
        )
        self._slot_b.load_requested.connect(
            lambda _s, p: self._browse_load("B", p)
        )

        # Zoom header
        self._slot_a.viewer.zoom_changed.connect(self._slot_a.update_zoom)
        self._slot_b.viewer.zoom_changed.connect(self._slot_b.update_zoom)

        # Linked zoom/pan (A↔B↔Diff all sync)
        self._slot_a.viewer.view_state_changed.connect(self._sync_from_a)
        self._slot_b.viewer.view_state_changed.connect(self._sync_from_b)

        # Pixel loupe
        self._slot_a.viewer.pixel_hovered.connect(self._on_hover)
        self._slot_b.viewer.pixel_hovered.connect(self._on_hover)

        # Jump to defect
        self._defect_list.jump_to.connect(self._jump_diff_to)

        return splitter

    def _build_loupe(self) -> QWidget:
        self._loupe = PixelLoupe()
        return self._loupe

    # ── Activation ────────────────────────────────────────────────────────────

    def _activate(self, slot: str):
        self._active = slot
        self._slot_a.set_active(slot == "A")
        self._slot_b.set_active(slot == "B")
        self._active_lbl.setText(f"Active: {slot}")

    # ── Load ──────────────────────────────────────────────────────────────────

    def _browse(self, slot: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load image into slot {slot}", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"
        )
        if path:
            self._activate(slot)
            self._browse_load(slot, path)

    def _browse_load(self, slot: str, path: str):
        try:
            img = load_image(path).raw
        except Exception:
            return
        self._load_into(slot, path, img)

    def _load_into(self, slot: str, path: str, img: np.ndarray):
        if slot == "A":
            self._img_a = img
            self._slot_a.set_image(path, img)
        else:
            self._img_b = img
            self._slot_b.set_image(path, img)
        self._trigger()

    # ── Analysis trigger ──────────────────────────────────────────────────────

    def _trigger(self):
        """Called on every slider/mode change — debounced to avoid flooding."""
        if self._mode.currentText() == "Flicker":
            return
        if self._img_a is not None and self._img_b is not None:
            self._set_analysing()   # instant feedback — user sees it the moment they touch anything
        self._debounce.start()      # resets the 300 ms countdown each call

    def _run_analysis(self):
        """Actually start the worker — called 300 ms after last _trigger()."""
        if self._img_a is None or self._img_b is None:
            return

        if self._worker and self._worker.isRunning():
            try:
                self._worker.done.disconnect()
            except RuntimeError:
                pass
            self._worker.quit()   # no wait — let it die on its own

        mode = self._mode.currentText()
        self._worker = AnalysisWorker(
            self._img_a, self._img_b,
            mode=mode,
            threshold=self._thr_slider.value(),
            min_area=self._ma_slider.value(),
            amp=self._amp_slider.value(),
        )
        self._worker.done.connect(self._on_done)
        self._set_controls_enabled(False)
        self._worker.start()

    def _on_done(self, display_img, annot_rgb, metrics: dict, defects: list):
        self._set_controls_enabled(True)
        mode = self._mode.currentText()
        self._diff_viewer.set_image(annot_rgb)

        if mode == "Blend 50/50":
            self._diff_hdr.setText("  BLEND   ·   50% A + 50% B")
            self._diff_hdr.setStyleSheet(
                "background:#101010; color:#5588AA; padding:3px 8px; font-size:10px;"
            )
            self._set_verdict_neutral()
            self._defect_list.show_defects([])
            return

        if mode == "Signed ±128":
            self._diff_hdr.setText(
                "  SIGNED ±128   ·   blue = A darker   ·   red = A brighter   ·   gray = identical"
            )
            self._diff_hdr.setStyleSheet(
                "background:#101010; color:#5588AA; padding:3px 8px; font-size:10px;"
            )
            self._set_verdict_neutral()
            self._defect_list.show_defects([])
            return

        parts = ["  DIFF MAP"]
        if metrics.get("aligned"):
            parts.append("⇄ aligned")
        if metrics.get("auto_norm"):
            parts.append("⚡ auto-norm")
        else:
            parts.append(f"amp ×{self._amp_slider.value()}")
        n = metrics["n_defects"]
        parts.append(f"{'%d defect%s found' % (n,'s' if n!=1 else '') if n else 'no defects'}")
        self._diff_hdr.setText("   ·   ".join(parts))
        self._diff_hdr.setStyleSheet(
            "background:#101A10; color:#44AA44; padding:3px 8px; "
            "font-size:10px; font-weight:700;"
        )
        self._set_verdict(metrics["verdict"], metrics, defects)
        self._defect_list.show_defects(defects)

    def _on_mode_changed(self, mode: str):
        if mode == "Flicker":
            self._flicker_timer.start(600)
        else:
            self._flicker_timer.stop()
            self._trigger()

    def _flicker_tick(self):
        if self._img_a is None or self._img_b is None:
            return
        img   = self._img_a if self._flicker_state else self._img_b
        label = "A — REFERENCE" if self._flicker_state else "B — TEST"
        self._diff_viewer.set_image(img)
        self._diff_hdr.setText(f"  FLICKER   ·   showing {label}")
        self._flicker_state = not self._flicker_state

    # ── Sync zoom/pan ─────────────────────────────────────────────────────────

    def _sync_from_a(self, zoom, ox, oy):
        for v in [self._slot_b.viewer, self._diff_viewer]:
            v.blockSignals(True)
            v.set_view_state(zoom, ox, oy)
            v.blockSignals(False)
        self._slot_b.update_zoom(zoom)

    def _sync_from_b(self, zoom, ox, oy):
        for v in [self._slot_a.viewer, self._diff_viewer]:
            v.blockSignals(True)
            v.set_view_state(zoom, ox, oy)
            v.blockSignals(False)
        self._slot_a.update_zoom(zoom)

    # ── Pixel loupe ───────────────────────────────────────────────────────────

    def _on_hover(self, ix: int, iy: int, _pixel):
        self._loupe.update_hover(ix, iy, self._img_a, self._img_b)

    # ── Jump to defect ────────────────────────────────────────────────────────

    def _jump_diff_to(self, cx: int, cy: int):
        """Pan + zoom diff viewer so defect centroid is centred on screen."""
        v = self._diff_viewer
        if v.width() == 0 or v._img_w == 0:
            return
        zoom = min(v.width() / 300, v.height() / 300, 4.0)
        ox   = v.width()  / 2 - cx * zoom
        oy   = v.height() / 2 - cy * zoom
        for vv in [self._slot_a.viewer, self._slot_b.viewer, self._diff_viewer]:
            vv.blockSignals(True)
            vv.set_view_state(zoom, ox, oy)
            vv.blockSignals(False)

    # ── Swap ──────────────────────────────────────────────────────────────────

    def _swap(self):
        pa, pb = self._slot_a._path, self._slot_b._path
        ia, ib = self._img_a, self._img_b
        self._img_a, self._img_b = ib, ia
        if ib is not None: self._slot_a.set_image(pb, ib)
        if ia is not None: self._slot_b.set_image(pa, ia)
        self._trigger()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint16: return (img>>8).astype(np.uint8)
        if img.dtype != np.uint8:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img
