"""
VyuhaAI Image Viewer — Report Exporter
Exports focus + quality analysis results to CSV and PDF.

CSV:  one row per image — filename, size, focus verdict, score, % sharp/soft/blurry,
      best/worst cell, tilt warning, quality score, quality verdict.

PDF:  per-image page with thumbnail, focus grid visualization, all metrics,
      pass/fail verdict box, operator action.
"""

import csv
import os
import datetime
from pathlib import Path
from typing import List


# ── Data record ────────────────────────────────────────────────────────────

class ImageRecord:
    """Holds all analysis results for one image — passed to exporter."""
    def __init__(self):
        self.path:            str   = ""
        self.filename:        str   = ""
        self.width:           int   = 0
        self.height:          int   = 0
        self.focus_score:     float = 0.0
        self.focus_verdict:   str   = ""
        self.quality_score:   float = 0.0
        self.quality_verdict: str   = ""
        self.pct_sharp:       float = 0.0
        self.pct_soft:        float = 0.0
        self.pct_blurry:      float = 0.0
        self.best_cell_row:   int   = 0
        self.best_cell_col:   int   = 0
        self.best_cell_score: float = 0.0
        self.worst_cell_row:  int   = 0
        self.worst_cell_col:  int   = 0
        self.worst_cell_score:float = 0.0
        self.tilt_warning:    str   = ""
        self.grid_rows:       int   = 0
        self.grid_cols:       int   = 0
        self.timestamp:       str   = ""

    @classmethod
    def from_analysis(cls, image_data, focus_result, quality_result) -> "ImageRecord":
        r = cls()
        r.path            = image_data.path or ""
        r.filename        = image_data.filename or ""
        r.width           = image_data.raw.shape[1] if image_data.raw is not None else 0
        r.height          = image_data.raw.shape[0] if image_data.raw is not None else 0
        r.focus_score     = round(focus_result.score, 1)
        r.focus_verdict   = focus_result.verdict
        r.quality_score   = round(quality_result.overall_score, 1)
        r.quality_verdict = quality_result.verdict

        g = focus_result.grid
        r.pct_sharp        = round(g.pct_sharp,  1)
        r.pct_soft         = round(g.pct_soft,   1)
        r.pct_blurry       = round(g.pct_blurry, 1)
        r.best_cell_row    = g.best_cell[0]  + 1
        r.best_cell_col    = g.best_cell[1]  + 1
        r.best_cell_score  = round(float(g.scores[g.best_cell]),  1)
        r.worst_cell_row   = g.worst_cell[0] + 1
        r.worst_cell_col   = g.worst_cell[1] + 1
        r.worst_cell_score = round(float(g.scores[g.worst_cell]), 1)
        r.tilt_warning     = g.tilt_warn
        r.grid_rows        = g.rows
        r.grid_cols        = g.cols
        r.timestamp        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return r

    def overall_decision(self) -> str:
        if self.focus_verdict in {"PERFECT", "GOOD"} and self.quality_verdict == "PASS":
            return "ACCEPT"
        elif self.focus_verdict == "SOFT" or self.quality_verdict in {"WARN", "PASS"}:
            return "REVIEW"
        else:
            return "REJECT"


# ── CSV export ─────────────────────────────────────────────────────────────

CSV_HEADERS = [
    "Timestamp", "Filename", "Width", "Height",
    "Decision", "Focus Verdict", "Focus Score",
    "Quality Verdict", "Quality Score",
    "Sharp Cells %", "Soft Cells %", "Blurry Cells %",
    "Best Cell (row,col)", "Best Cell Score %",
    "Worst Cell (row,col)", "Worst Cell Score %",
    "Tilt Warning", "Grid Size",
]


def export_csv(records: List[ImageRecord], out_path: str):
    """Write all records to a CSV file."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        for r in records:
            writer.writerow([
                r.timestamp,
                r.filename,
                r.width,
                r.height,
                r.overall_decision(),
                r.focus_verdict,
                r.focus_score,
                r.quality_verdict,
                r.quality_score,
                r.pct_sharp,
                r.pct_soft,
                r.pct_blurry,
                f"r{r.best_cell_row} c{r.best_cell_col}",
                r.best_cell_score,
                f"r{r.worst_cell_row} c{r.worst_cell_col}",
                r.worst_cell_score,
                r.tilt_warning or "None",
                f"{r.grid_rows}x{r.grid_cols}",
            ])


# ── PDF export ─────────────────────────────────────────────────────────────

def export_pdf(records: List[ImageRecord], out_path: str,
               get_image_fn=None, get_grid_fn=None):
    """
    Write a PDF report. One page per image.
    get_image_fn(path) -> np.ndarray or None  (for thumbnail)
    get_grid_fn(path)  -> FocusGridData or None
    Uses reportlab if available, falls back to plain HTML.
    """
    try:
        _export_pdf_reportlab(records, out_path, get_image_fn, get_grid_fn)
    except ImportError:
        # reportlab not installed — generate HTML report instead
        html_path = out_path.replace(".pdf", ".html")
        _export_html(records, html_path, get_image_fn, get_grid_fn)
        return html_path
    return out_path


def _export_pdf_reportlab(records, out_path, get_image_fn, get_grid_fn):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    W, H = A4
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm,  bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()

    DARK   = colors.HexColor("#0A0A1A")
    CYAN   = colors.HexColor("#00B4D8")
    GREEN  = colors.HexColor("#00C853")
    AMBER  = colors.HexColor("#FFB300")
    RED    = colors.HexColor("#FF1744")
    LGRAY  = colors.HexColor("#1A1A2A")

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  textColor=CYAN, fontSize=16, spaceAfter=6)
    sub_style   = ParagraphStyle("sub",   parent=styles["Normal"],
                                  textColor=colors.HexColor("#888899"), fontSize=9)
    hdr_style   = ParagraphStyle("hdr",   parent=styles["Heading2"],
                                  textColor=CYAN, fontSize=11, spaceBefore=10, spaceAfter=4)
    body_style  = ParagraphStyle("body",  parent=styles["Normal"],
                                  textColor=colors.HexColor("#CCCCDD"), fontSize=9)

    story = []

    # Cover title
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    story.append(Paragraph("VyuhaAI Image Viewer — Focus & Quality Report", title_style))
    story.append(Paragraph(f"Generated: {now}   ·   {len(records)} image(s) analyzed", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=CYAN, spaceAfter=12))

    # Summary table
    accept  = sum(1 for r in records if r.overall_decision() == "ACCEPT")
    review  = sum(1 for r in records if r.overall_decision() == "REVIEW")
    reject  = sum(1 for r in records if r.overall_decision() == "REJECT")
    summary_data = [
        ["Total Images", "ACCEPT", "REVIEW", "REJECT"],
        [str(len(records)), str(accept), str(review), str(reject)],
    ]
    t = Table(summary_data, colWidths=[4*cm]*4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), LGRAY),
        ("TEXTCOLOR",  (0,0), (-1,0), CYAN),
        ("TEXTCOLOR",  (1,1), (1,1), GREEN),
        ("TEXTCOLOR",  (2,1), (2,1), AMBER),
        ("TEXTCOLOR",  (3,1), (3,1), RED),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#111118")]),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#333344")),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Per-image detail
    for i, r in enumerate(records):
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#222233"), spaceAfter=8))
        decision = r.overall_decision()
        dec_color = GREEN if decision == "ACCEPT" else (AMBER if decision == "REVIEW" else RED)
        story.append(Paragraph(
            f'<font color="#{_hex(dec_color)}">[{decision}]</font>  {r.filename}',
            hdr_style
        ))
        story.append(Paragraph(
            f"Analyzed: {r.timestamp}   ·   {r.width}×{r.height} px", sub_style))

        data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Focus Verdict",   r.focus_verdict,    "Quality Verdict",  r.quality_verdict],
            ["Focus Score",     f"{r.focus_score}",  "Quality Score",    f"{r.quality_score}/100"],
            ["Sharp Cells",     f"{r.pct_sharp}%",   "Soft Cells",       f"{r.pct_soft}%"],
            ["Blurry Cells",    f"{r.pct_blurry}%",  "Grid Size",        f"{r.grid_rows}×{r.grid_cols}"],
            ["Best Cell",       f"r{r.best_cell_row} c{r.best_cell_col} ({r.best_cell_score}%)",
             "Worst Cell",      f"r{r.worst_cell_row} c{r.worst_cell_col} ({r.worst_cell_score}%)"],
            ["Tilt Warning",    r.tilt_warning or "None", "Decision",    decision],
        ]
        tbl = Table(data, colWidths=[4*cm, 5.5*cm, 4*cm, 5.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  LGRAY),
            ("TEXTCOLOR",     (0,0), (-1,0),  CYAN),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#0D0D1A"),
                                               colors.HexColor("#111118")]),
            ("TEXTCOLOR",     (0,1), (-1,-1), colors.HexColor("#CCCCDD")),
            ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
            ("FONTNAME",      (2,1), (2,-1),  "Helvetica-Bold"),
            ("TEXTCOLOR",     (0,1), (0,-1),  colors.HexColor("#8888BB")),
            ("TEXTCOLOR",     (2,1), (2,-1),  colors.HexColor("#8888BB")),
            ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#222233")),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 8))

    doc.build(story)


def _hex(color) -> str:
    """Convert reportlab Color to 6-char hex string."""
    try:
        return f"{int(color.red*255):02X}{int(color.green*255):02X}{int(color.blue*255):02X}"
    except Exception:
        return "FFFFFF"


def _export_html(records, out_path, get_image_fn=None, get_grid_fn=None):
    """Fallback HTML report when reportlab is not installed."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = ""
    for r in records:
        dec = r.overall_decision()
        dec_color = {"ACCEPT": "#00C853", "REVIEW": "#FFB300", "REJECT": "#FF1744"}.get(dec, "#fff")
        rows += f"""
        <tr>
          <td>{r.filename}</td>
          <td style="color:{dec_color};font-weight:bold">{dec}</td>
          <td>{r.focus_verdict}</td>
          <td>{r.focus_score}</td>
          <td>{r.quality_verdict}</td>
          <td>{r.quality_score}</td>
          <td>{r.pct_sharp}%</td>
          <td>{r.pct_soft}%</td>
          <td>{r.pct_blurry}%</td>
          <td>r{r.best_cell_row} c{r.best_cell_col} ({r.best_cell_score}%)</td>
          <td>r{r.worst_cell_row} c{r.worst_cell_col} ({r.worst_cell_score}%)</td>
          <td style="color:#FFB300">{r.tilt_warning or "—"}</td>
          <td>{r.timestamp}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VyuhaAI Focus Report</title>
<style>
  body {{ background:#0A0A1A; color:#CCCCDD; font-family:Segoe UI,sans-serif; padding:20px; }}
  h1   {{ color:#00B4D8; }} h2 {{ color:#00B4D8; font-size:13px; }}
  table{{ border-collapse:collapse; width:100%; font-size:12px; }}
  th   {{ background:#1A1A2A; color:#00B4D8; padding:8px; border:1px solid #333344; }}
  td   {{ padding:6px 8px; border:1px solid #222233; }}
  tr:nth-child(even) td {{ background:#111118; }}
</style>
</head>
<body>
<h1>VyuhaAI Image Viewer — Focus &amp; Quality Report</h1>
<p style="color:#888899">Generated: {now} &nbsp;·&nbsp; {len(records)} image(s)</p>
<table>
<tr>
  <th>Filename</th><th>Decision</th><th>Focus</th><th>Focus Score</th>
  <th>Quality</th><th>Quality Score</th>
  <th>Sharp%</th><th>Soft%</th><th>Blurry%</th>
  <th>Best Cell</th><th>Worst Cell</th><th>Tilt</th><th>Timestamp</th>
</tr>
{rows}
</table>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
