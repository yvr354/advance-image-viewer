"""
In-app User Guide window.

Layout:
  Left  — navigation list (sections)
  Right — QTextBrowser showing rich HTML for the selected section

Opened from Help menu → User Guide (F1).
"""

from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QTextBrowser, QLineEdit, QLabel, QSplitter, QPushButton,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QKeySequence, QShortcut


# ── Shared style constants ─────────────────────────────────────────────────

_BG       = "#0A1218"
_BG2      = "#0E1820"
_BORDER   = "#1A2A3A"
_TEXT     = "#AABBCC"
_DIM      = "#445566"
_CYAN     = "#00B4D8"
_AMBER    = "#D4840A"
_GREEN    = "#2ECC71"
_PURPLE   = "#CC88FF"
_RED      = "#E05555"

_CSS_BASE = f"""
    QDialog      {{ background:{_BG}; color:{_TEXT}; }}
    QSplitter    {{ background:{_BG}; }}
    QListWidget  {{ background:{_BG2}; border:none; border-right:1px solid {_BORDER};
                   color:{_TEXT}; font-size:11px; outline:none; }}
    QListWidget::item            {{ padding:8px 14px; border-radius:0; }}
    QListWidget::item:selected   {{ background:#001A28; color:{_CYAN};
                                   border-left:3px solid {_CYAN}; padding-left:11px; }}
    QListWidget::item:hover:!selected {{ background:#0E1E2E; color:#CCDDEE; }}
    QTextBrowser {{ background:{_BG}; border:none; color:{_TEXT};
                    font-size:12px; font-family: Segoe UI, sans-serif; }}
    QLineEdit    {{ background:{_BG2}; border:1px solid {_BORDER}; border-radius:4px;
                   color:{_TEXT}; padding:4px 8px; font-size:11px; }}
    QLineEdit:focus {{ border-color:{_CYAN}; }}
    QPushButton  {{ background:{_BG2}; color:{_DIM}; border:1px solid {_BORDER};
                   border-radius:3px; padding:3px 10px; font-size:10px; }}
    QPushButton:hover {{ color:{_TEXT}; background:#111F2E; }}
"""

_HTML_STYLE = f"""
<style>
  body   {{ background:{_BG}; color:{_TEXT}; font-family:Segoe UI,sans-serif;
            font-size:13px; margin:20px 28px; line-height:1.6; }}
  h1     {{ color:{_CYAN}; font-size:20px; border-bottom:1px solid {_BORDER};
            padding-bottom:8px; margin-bottom:16px; }}
  h2     {{ color:{_CYAN}; font-size:14px; margin-top:24px; margin-bottom:6px; }}
  h3     {{ color:#88BBCC; font-size:12px; margin-top:16px; margin-bottom:4px; }}
  p      {{ margin:6px 0; }}
  code   {{ background:#0E1E2E; color:#88FFCC; border-radius:3px;
            padding:1px 5px; font-family:Consolas,monospace; font-size:11px; }}
  pre    {{ background:#0E1E2E; color:#88FFCC; border:1px solid {_BORDER};
            border-radius:4px; padding:10px 14px; font-family:Consolas,monospace;
            font-size:11px; white-space:pre-wrap; }}
  table  {{ border-collapse:collapse; width:100%; margin:10px 0; }}
  th     {{ background:#0E1E2E; color:{_CYAN}; padding:7px 12px;
            border:1px solid {_BORDER}; font-size:11px; text-align:left; }}
  td     {{ padding:6px 12px; border:1px solid {_BORDER}; font-size:11px;
            vertical-align:top; }}
  tr:nth-child(even) td {{ background:#0C1620; }}
  .badge {{ display:inline-block; border-radius:3px; padding:1px 7px;
            font-size:10px; font-weight:700; }}
  .cyan  {{ background:#001A28; color:{_CYAN}; border:1px solid #005A70; }}
  .green {{ background:#001A0A; color:{_GREEN}; border:1px solid #1A5030; }}
  .amber {{ background:#1A0A00; color:{_AMBER}; border:1px solid #6A3A00; }}
  .purple{{ background:#150028; color:{_PURPLE}; border:1px solid #5A3080; }}
  .red   {{ background:#1A0808; color:{_RED};   border:1px solid #5A2020; }}
  .dim   {{ color:{_DIM}; font-size:11px; }}
  .tip   {{ background:#0E1E2E; border-left:3px solid {_CYAN}; padding:8px 12px;
            margin:10px 0; border-radius:0 4px 4px 0; font-size:11px; }}
  .warn  {{ background:#1A1000; border-left:3px solid {_AMBER}; padding:8px 12px;
            margin:10px 0; border-radius:0 4px 4px 0; font-size:11px; }}
  kbd    {{ background:#1A2A3A; color:#CCDDEE; border:1px solid {_BORDER};
            border-radius:3px; padding:1px 6px; font-family:Consolas,monospace;
            font-size:10px; }}
  hr     {{ border:none; border-top:1px solid {_BORDER}; margin:18px 0; }}
  ul,ol  {{ margin:6px 0; padding-left:20px; }}
  li     {{ margin:3px 0; }}
  a      {{ color:{_CYAN}; text-decoration:none; }}
</style>
"""


# ── Section content ────────────────────────────────────────────────────────

def _page(title: str, body: str) -> str:
    return f"{_HTML_STYLE}<body><h1>{title}</h1>{body}</body>"


SECTIONS = {

"Overview": _page("Overview — What This App Does", """
<p>This is an industrial image viewer built for surface defect inspection.
It combines a high-resolution image viewer with a set of professional analysis tools
that are normally only found in dedicated machine vision systems (Halcon, Cognex, Keyence).</p>

<h2>Who it is for</h2>
<ul>
  <li>Quality engineers inspecting manufactured parts</li>
  <li>Vision system developers building inspection pipelines</li>
  <li>Researchers analyzing surface topology with multi-light imaging</li>
</ul>

<h2>What it can do</h2>
<table>
  <tr><th>Feature</th><th>What it gives you</th></tr>
  <tr><td>Inspect mode</td><td>Pan, zoom, ROI, measure, annotate, mask — full inspection toolkit</td></tr>
  <tr><td>Focus scoring</td><td>Per-cell sharpness grid — know before you inspect if the image is sharp enough</td></tr>
  <tr><td>Image Compare</td><td>Auto-align two images, compute |A−B|, classify defects, PASS/FAIL verdict</td></tr>
  <tr><td>Filter Pipeline</td><td>Stackable image processing — CLAHE, Canny, Gabor, wavelet, false color</td></tr>
  <tr><td>Illumination Fusion</td><td>Combine multiple lighting angles into one image — reveals hidden defects</td></tr>
  <tr><td>3D Surface</td><td>Real surface topology from Photometric Stereo, or visual brightness mesh</td></tr>
</table>

<h2>Window layout</h2>
<pre>
┌──────────────────────────────────────────────────────────────┐
│  File   View   Pipeline   Tools   Help     ← Menu bar        │
├──────────────────────────────────────────────────────────────┤
│  Inspect │ Focus │ Compare │ Filters │ Fusion │ 3D  ← Tabs   │
│  [context toolbar — only visible when a tool is active]      │
├──────────┬───────────────────────────────────────────────────┤
│ Left     │                                                   │
│ Panel    │   Main workspace                                  │
│          │   (viewer / compare / fusion / 3D)                │
│ Browser  │                                                   │
│ Inspector│                                                   │
│ Pipeline │                                                   │
│ Fusion   │                                                   │
├──────────┴───────────────────────────────────────────────────┤
│  Status bar — file · zoom · focus score · tool hint          │
└──────────────────────────────────────────────────────────────┘
</pre>
<p class="dim">Left panel docks can be resized, reordered, hidden, or floated as separate windows.</p>
"""),


"File Browser": _page("File Browser", """
<p>The File Browser is the left-most dock panel. It lets you navigate folders and open images.</p>

<h2>How to use</h2>
<ol>
  <li>Click any folder in the tree to open it</li>
  <li>The image list below the tree shows all supported images in that folder</li>
  <li>Click an image to open it in the main viewer</li>
  <li>Use the <strong>Filter filenames…</strong> box to search by name</li>
  <li>Use <kbd>←</kbd> / <kbd>→</kbd> arrow keys to step through images</li>
</ol>

<h2>Supported formats</h2>
<p>TIFF · TIF · PNG · BMP · JPG · PGM — including 16-bit grayscale TIFF</p>

<h2>Fusion shortcut</h2>
<p>When the Fusion panel is open, the folder list appears inside it.
Double-click any filename there to add it directly to the fusion set — no file dialog needed.</p>

<div class="tip">Tip: The app remembers the last folder you opened across sessions.</div>
"""),


"Inspect Tools": _page("Inspect Mode — Tools", """
<p>Inspect is the default mode. The main viewer shows the image at full resolution.</p>

<h2>Selecting a tool</h2>
<p>Open <strong>Tools menu</strong> and pick a tool, or use the keyboard shortcut:</p>

<table>
  <tr><th>Tool</th><th>Key</th><th>What it does</th></tr>
  <tr><td><span class="badge cyan">↖ Navigate</span></td><td><kbd>Esc</kbd></td>
      <td>Default — pan with left drag, zoom with scroll. Toolbar hidden.</td></tr>
  <tr><td><span class="badge cyan">⬛ ROI</span></td><td><kbd>R</kbd></td>
      <td>Draw rectangle → region statistics appear in Inspector</td></tr>
  <tr><td><span class="badge cyan">📈 Profile</span></td><td><kbd>P</kbd></td>
      <td>Draw line → intensity chart appears in Inspector</td></tr>
  <tr><td><span class="badge amber">📍 Annotate</span></td><td><kbd>A</kbd></td>
      <td>Click to place defect marker — choose label from popup</td></tr>
  <tr><td><span class="badge green">↔ Measure</span></td><td><kbd>M</kbd></td>
      <td>Drag to measure distance in pixels and mm</td></tr>
  <tr><td><span class="badge purple">⬡ Mask</span></td><td><kbd>K</kbd></td>
      <td>Draw polygon — metrics computed only inside masked area</td></tr>
</table>

<div class="tip">The active tool is always shown as a coloured badge on the left of the toolbar.
Navigate is dim — it is the "go back" button, not the active tool.</div>

<hr>

<h2>Navigate <kbd>Esc</kbd></h2>
<p>Default mode. Toolbar is hidden — no clutter.</p>
<ul>
  <li>Left drag → pan</li>
  <li>Right drag → zoom</li>
  <li>Scroll wheel → zoom</li>
</ul>
<p>Switch back to Navigate from any tool by pressing <kbd>Esc</kbd> or clicking <strong>↖ Navigate</strong> in the toolbar.</p>

<hr>

<h2>ROI — Region of Interest <kbd>R</kbd></h2>
<p>Draw a rectangle. The Inspector shows statistics for that exact region:</p>
<ul>
  <li>Mean, min, max, standard deviation</li>
  <li>Histogram of the region</li>
  <li>Channel breakdown for colour images</li>
</ul>
<p><strong>Use for:</strong> Comparing brightness between a defect and the surrounding good surface.
Checking if a dark spot is statistically different from background.</p>

<hr>

<h2>Profile — Line Intensity <kbd>P</kbd></h2>
<p>Draw a line across the image. The Inspector shows a brightness chart along that line.</p>
<p><strong>Use for:</strong> Measuring scratch width (distance between dips in the chart).
Checking if a coating is uniform. Verifying edge sharpness.</p>

<hr>

<h2>Annotate — Defect Markers <kbd>A</kbd></h2>
<p>Click anywhere on the image. A label popup appears — choose:</p>
<p>Scratch · Pit · Contamination · Burr · Crack · OK · Other</p>
<p>Markers are shown on the image and listed in the Inspector panel.</p>
<div class="tip">Annotations are automatically saved next to the image file as
<code>imagename.annotations.json</code> and loaded automatically next time you open that image.</div>

<hr>

<h2>Measure — Distance <kbd>M</kbd></h2>
<p>Drag from point A to point B. The Inspector shows:</p>
<ul>
  <li>Pixel distance</li>
  <li>Real-world mm distance (if scale is calibrated)</li>
  <li>Start / end coordinates and angle</li>
</ul>
<h3>Setting the scale</h3>
<ol>
  <li>Photograph a ruler at the same working distance as your parts</li>
  <li>Draw a Measure line over a known distance</li>
  <li>Open <strong>Tools → Set Scale Calibration…</strong></li>
  <li>Enter the mm/pixel ratio</li>
</ol>
<p>The scale is saved and used for all future measurements in the session.</p>

<hr>

<h2>Mask — Inspection Polygon <kbd>K</kbd></h2>
<p>Draw a polygon. All metrics (focus score, histogram, ROI stats) are computed
<em>only inside</em> the masked area. Everything outside is excluded.</p>

<h3>Mask toolbar actions</h3>
<table>
  <tr><th>Button</th><th>What it does</th></tr>
  <tr><td>✦ Auto-Detect</td><td>Detect specular reflection regions and suggest a mask that excludes them</td></tr>
  <tr><td>📋 Apply to Folder</td><td>Copy mask to all images in the folder (fixed camera = direct copy; moving part = auto-align by edge matching)</td></tr>
  <tr><td>⧉ Find All Similar</td><td>Draw mask around one example region → finds all identical regions and masks them all</td></tr>
  <tr><td>💾 Save Mask</td><td>Save mask position or export masked image file</td></tr>
  <tr><td>⬡ Clear Mask</td><td>Remove mask — return to analyzing the full image</td></tr>
</table>
<p>Clear mask is also available in <strong>Tools → Clear Mask</strong>.</p>
"""),


"Focus": _page("Focus Mode — Sharpness Scoring", """
<p>Focus mode overlays a sharpness score grid on the image.
Use it to verify that an image is sharp enough before using it for defect inspection.</p>

<h2>The grid</h2>
<p>The image is divided into an 8×8 grid (64 cells). Each cell is scored independently.</p>
<table>
  <tr><th>Colour</th><th>Score</th><th>Meaning</th></tr>
  <tr><td><span class="badge green">GREEN</span></td><td>≥ 72%</td><td>Sharp — safe to inspect</td></tr>
  <tr><td><span class="badge amber">AMBER</span></td><td>38–72%</td><td>Soft — marginal, check lens</td></tr>
  <tr><td><span class="badge red">RED</span></td><td>&lt; 38%</td><td>Blurry — reject this image</td></tr>
</table>

<h2>Scoring algorithm</h2>
<p>Two independent metrics are fused:</p>
<ul>
  <li><strong>Laplacian Variance</strong> — variance of the second derivative. Best at low noise.</li>
  <li><strong>Tenengrad</strong> — sum of squared gradients. Best noise robustness.</li>
</ul>
<p>If both agree → HIGH confidence. If they disagree by one category → MEDIUM. If they disagree by two → LOW.</p>

<h2>Scoring modes</h2>
<table>
  <tr><th>Mode</th><th>When to use</th></tr>
  <tr><td>RELATIVE</td><td>No reference — scores relative to best cell in same image. <strong>Not for production.</strong></td></tr>
  <tr><td>AUTO-REF</td><td>Tracks sharpest image seen this session. Better than relative.</td></tr>
  <tr><td>LOCKED REF ✓</td><td>You locked a known-good reference image. <strong>Use this for production.</strong></td></tr>
</table>

<h2>First-time setup</h2>
<ol>
  <li>Capture your sharpest possible image at the correct working distance</li>
  <li>Open it in the viewer</li>
  <li>Click <strong>Lock Ref</strong> in the Focus Assist panel</li>
  <li>Done — reference is saved to disk and survives restarts</li>
</ol>

<div class="tip">100% in LOCKED REF mode means "as sharp as your calibrated reference sample." 60% means the image is only 60% as sharp — check lens focus or lighting.</div>
"""),


"Compare": _page("Compare Mode — A/B Defect Detection", """
<p>Compare mode loads two images (A and B) side by side,
auto-aligns them, computes the difference, detects defects, and gives a PASS/FAIL verdict.</p>

<h2>How to use</h2>
<ol>
  <li>Switch to <strong>Compare</strong> tab</li>
  <li>Load Image A (reference / good part)</li>
  <li>Load Image B (part to inspect)</li>
  <li>The system runs automatically — alignment, diff, detection</li>
  <li>Read the PASS / FAIL verdict at the top</li>
</ol>

<h2>What happens automatically</h2>
<table>
  <tr><th>Step</th><th>What it does</th></tr>
  <tr><td>Brightness normalization</td><td>Matches overall brightness of B to A — eliminates lighting variation false defects</td></tr>
  <tr><td>Alignment (ORB + RANSAC)</td><td>Warps B to match A's position — eliminates fixture shift false defects</td></tr>
  <tr><td>|A − B| diff map</td><td>Absolute difference per pixel shown as HOT colormap</td></tr>
  <tr><td>Noise removal</td><td>Gaussian blur + threshold + morphological open/close — eliminates camera noise</td></tr>
  <tr><td>Connected components</td><td>Detects and classifies each defect blob by area</td></tr>
</table>

<h2>Defect severity</h2>
<table>
  <tr><th>Severity</th><th>Area</th><th>Action</th></tr>
  <tr><td><span class="badge green">COSMETIC</span></td><td>&lt; 25 px²</td><td>Document and accept</td></tr>
  <tr><td><span class="badge amber">FUNCTIONAL</span></td><td>25–200 px²</td><td>Engineer review required</td></tr>
  <tr><td><span class="badge red">CRITICAL</span></td><td>&gt; 200 px²</td><td>Reject — do not ship</td></tr>
</table>

<h2>Display modes</h2>
<table>
  <tr><th>Mode</th><th>What it shows</th></tr>
  <tr><td>Diff Map</td><td>|A−B| as HOT colormap — primary view</td></tr>
  <tr><td>Signed ±128</td><td>Directional diff — blue = A darker, red = A brighter</td></tr>
  <tr><td>Blend 50/50</td><td>A and B overlaid at 50% — visual alignment check</td></tr>
  <tr><td>Flicker</td><td>Alternates A and B every 600ms — human eye catches texture differences</td></tr>
</table>

<div class="tip">Click <strong>→ Jump to defect</strong> on any defect card to pan all viewers to that exact location at 4× zoom.</div>
"""),


"Filters": _page("Filter Pipeline", """
<p>The Filter Pipeline is a stackable chain of image processing steps.
Each filter runs in order (top → bottom) and passes its result to the next filter.</p>

<h2>How to use</h2>
<ol>
  <li>Switch to <strong>Filters</strong> tab</li>
  <li>Click a filter from the category list to add it to the pipeline</li>
  <li>Adjust its sliders — the image updates live (400ms debounce)</li>
  <li>Add more filters — they chain in order</li>
  <li>Toggle individual filters on/off without removing them</li>
  <li>Drag filters to reorder</li>
</ol>

<h2>Workflow steps</h2>
<table>
  <tr><th>Step</th><th>Purpose</th><th>Filters</th></tr>
  <tr><td><span class="badge cyan">① Pre-process</span></td>
      <td>Remove noise, correct exposure</td>
      <td>Gaussian Blur, Median, Bilateral, NL-Means, Normalize, Brightness/Contrast</td></tr>
  <tr><td><span class="badge amber">② Enhance</span></td>
      <td>Increase defect visibility</td>
      <td>CLAHE, Histogram EQ, Gamma, Levels, Unsharp Mask, Laplacian Sharpen, Top-Hat, Black-Hat</td></tr>
  <tr><td><span class="badge green">③ Detect</span></td>
      <td>Extract the defect signal</td>
      <td>Canny, Sobel, DoG, Morph Gradient, Gabor Bank, Wavelet, FFT Magnitude, LBP Texture</td></tr>
  <tr><td><span class="badge purple">④ Visualize</span></td>
      <td>Map results to colour</td>
      <td>False Color, Clipping Highlight, Channel Split, Channel Mixer, Invert</td></tr>
</table>

<h2>Expert presets — one click starting points</h2>
<table>
  <tr><th>Preset</th><th>Best for</th><th>Filters added</th></tr>
  <tr><td>Surface Scratch</td><td>Metal, glass, plastic scratches</td><td>Bilateral → CLAHE → Canny</td></tr>
  <tr><td>Dark Pits / Voids</td><td>Holes, voids on bright surfaces</td><td>Gaussian → Black-Hat → False Color</td></tr>
  <tr><td>Bright Protrusions</td><td>Burrs, raised material</td><td>Gaussian → Top-Hat → False Color</td></tr>
  <tr><td>PCB / Trace</td><td>PCB trace and solder inspection</td><td>NL-Means → Levels → Sobel</td></tr>
  <tr><td>Texture Defect</td><td>Woven, machined, patterned surfaces</td><td>Gaussian → Gabor Bank → Normalize</td></tr>
  <tr><td>Low Contrast Fix</td><td>Poorly lit images</td><td>CLAHE → Gamma → Unsharp Mask</td></tr>
</table>

<div class="tip">The pipeline result feeds into the 3D viewer and can be exported for batch processing.</div>
<div class="warn">Always start with a noise removal filter (Gaussian or Bilateral) before detection filters. Detection filters amplify noise as well as signal.</div>
"""),


"Fusion": _page("Illumination Fusion", """
<p>Illumination Fusion combines multiple images of the same part taken under different lighting angles.
A defect hidden under one angle often appears clearly under another.
Fusion shows everything every angle revealed — in one image.</p>

<h2>How to use</h2>
<ol>
  <li>Switch to <strong>Fusion</strong> tab (or press <kbd>Ctrl+F</kbd>)</li>
  <li>Load images from the folder list (double-click) or click <strong>＋ Add</strong></li>
  <li>Select the fusion operation</li>
  <li>Click <strong>▶ Compose</strong></li>
  <li>Result appears in the main viewer — send it to the Filter Pipeline for further processing</li>
</ol>

<h2>Operations — which to use when</h2>
<table>
  <tr><th>Operation</th><th>Best for</th><th>Needs</th></tr>
  <tr><td>Range (Max−Min)</td><td>Best automatic defect detector — no angle input needed</td><td>3+ images</td></tr>
  <tr><td>Photometric Stereo → Gradient</td><td>Scratches, cracks, raised edges — real surface normals</td><td>3+ images + angles</td></tr>
  <tr><td>Photometric Stereo → Height Map</td><td>Dents, warps, 3D topology → feeds real geometry to 3D viewer</td><td>3+ images + angles</td></tr>
  <tr><td>RGB Composite</td><td>Assign lights to R/G/B — defects become colour-visible</td><td>3 images</td></tr>
  <tr><td>Difference |A−B|</td><td>Isolate what one angle reveals vs another</td><td>2 images</td></tr>
  <tr><td>Superposition A+B+C</td><td>Defects in multiple lights accumulate brighter</td><td>2+ images</td></tr>
  <tr><td>Multiply A×B×C</td><td>Only pixels bright in every image survive — confirms real defects</td><td>2+ images</td></tr>
  <tr><td>Max</td><td>Bright defects, protrusions, high-reflectance spots</td><td>Any</td></tr>
  <tr><td>Min</td><td>Dark defects, pits, voids</td><td>Any</td></tr>
  <tr><td>Average</td><td>Noise reduction, balanced exposure</td><td>Any</td></tr>
  <tr><td>RTI Relight</td><td>Drag sliders to find the angle that reveals a hidden defect</td><td>6+ images + angles</td></tr>
</table>

<h2>Setting light angles (Photometric Stereo / RTI)</h2>
<p>Each image row shows <strong>Az</strong> (azimuth 0–360°) and <strong>El</strong> (elevation 0–90°) spinboxes.</p>
<ul>
  <li>Azimuth 0° = light from the right. 90° = top. 180° = left. 270° = bottom.</li>
  <li>Elevation 0° = grazing (side-lit, reveals texture). 90° = overhead.</li>
  <li>Ideal setup: 4 lights at 0°/90°/180°/270° azimuth, 30–45° elevation.</li>
</ul>

<div class="tip">Start with <strong>Range</strong> — it works on any images without angle input and gives the best automatic defect signal. Then switch to Photometric Stereo if you have calibrated angles.</div>
"""),


"3D Viewer": _page("3D Surface Viewer", """
<p>The 3D viewer renders the image as a surface mesh.
It has two completely different modes depending on the source.</p>

<h2>Visual mode — brightness → height</h2>
<p>Any image loaded normally is shown in this mode.
Pixel brightness is treated as Z height: bright = tall, dark = flat.</p>
<div class="warn">This geometry is <strong>not physically real.</strong>
A bright scratch appears raised even though it is actually a groove.
Use this mode for visual overview only — not for dimensional analysis.</div>
<p>The info bar reads: <code>Visual — brightness → height</code></p>

<h2>Real Geometry mode — Photometric Stereo</h2>
<p>When you compose <strong>Photometric Stereo → Height Map</strong> in the Fusion panel,
the app automatically switches to the 3D tab and loads the real surface geometry.</p>
<ul>
  <li>Bright = surface is truly raised above the baseline</li>
  <li>Dark = surface is truly recessed</li>
  <li>The mesh shape is the actual surface topology</li>
</ul>
<p>A green badge is shown: <span class="badge green">⚡ Real Surface Geometry — Photometric Stereo</span></p>
<p>The info bar reads: <code>⚡ Real Surface — Photometric Stereo</code></p>

<h2>Controls</h2>
<table>
  <tr><th>Control</th><th>Effect</th></tr>
  <tr><td>Resolution slider</td><td>Mesh density — lower = faster, higher = more detail</td></tr>
  <tr><td>Z Scale slider</td><td>Amplify height values — increase for subtle topology</td></tr>
  <tr><td>Color</td><td>Colormap: Thermal · Viridis · Plasma · Grays · Cyclic</td></tr>
  <tr><td>Smooth</td><td>Gaussian blur before mesh build — reduces jagged edges</td></tr>
  <tr><td>Reset View</td><td>Return camera to default position</td></tr>
</table>

<h2>Navigation</h2>
<ul>
  <li>Left drag → rotate</li>
  <li>Scroll wheel → zoom</li>
  <li>Right drag → pan</li>
</ul>

<div class="tip">To get real 3D: Fusion panel → load 3+ images → set light angles → Photometric Stereo → Height Map → Compose. The 3D tab opens automatically.</div>
"""),


"Tools Menu": _page("Tools Menu — Reference", """
<p>The <strong>Tools</strong> menu is in the menu bar at the top.</p>

<h2>Global tools</h2>
<table>
  <tr><th>Item</th><th>Shortcut</th><th>What it does</th></tr>
  <tr><td>Illumination Fusion</td><td><kbd>Ctrl+F</kbd></td><td>Switch to Fusion tab</td></tr>
  <tr><td>3D Surface View</td><td><kbd>Ctrl+3</kbd></td><td>Switch to 3D tab</td></tr>
  <tr><td>Image Comparison</td><td><kbd>Ctrl+M</kbd></td><td>Open comparison window</td></tr>
</table>

<h2>Inspect tools</h2>
<p>These switch the active inspect tool. A checkmark shows which tool is currently on.</p>
<table>
  <tr><th>Item</th><th>Shortcut</th><th>What it does</th></tr>
  <tr><td>↖ Navigate (pan / zoom)</td><td><kbd>Esc</kbd></td><td>Default — toolbar hidden</td></tr>
  <tr><td>⬛ ROI</td><td><kbd>R</kbd></td><td>Draw rectangle → region stats</td></tr>
  <tr><td>📈 Profile</td><td><kbd>P</kbd></td><td>Draw line → intensity chart</td></tr>
  <tr><td>📍 Annotate</td><td><kbd>A</kbd></td><td>Click to place defect marker</td></tr>
  <tr><td>↔ Measure</td><td><kbd>M</kbd></td><td>Drag to measure distance</td></tr>
  <tr><td>⬡ Mask</td><td><kbd>K</kbd></td><td>Draw polygon inspection mask</td></tr>
</table>

<h2>Utility tools</h2>
<table>
  <tr><th>Item</th><th>What it does</th></tr>
  <tr><td>Set Scale Calibration (mm/px)…</td><td>Enter mm/pixel ratio for Measure tool</td></tr>
  <tr><td>Clear Inspect Overlays</td><td>Remove ROI, profile line, measure line, annotations</td></tr>
  <tr><td>Clear Mask</td><td>Remove the inspection mask polygon</td></tr>
</table>
"""),


"Keyboard Shortcuts": _page("Keyboard Shortcuts", """
<h2>Inspect tools</h2>
<table>
  <tr><th>Key</th><th>Tool</th></tr>
  <tr><td><kbd>Esc</kbd></td><td>Navigate (pan / zoom) — returns to default</td></tr>
  <tr><td><kbd>R</kbd></td><td>ROI — draw region of interest</td></tr>
  <tr><td><kbd>P</kbd></td><td>Profile — draw intensity line</td></tr>
  <tr><td><kbd>A</kbd></td><td>Annotate — place defect marker</td></tr>
  <tr><td><kbd>M</kbd></td><td>Measure — drag distance</td></tr>
  <tr><td><kbd>K</kbd></td><td>Mask — draw inspection polygon</td></tr>
</table>

<h2>Navigation</h2>
<table>
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><kbd>←</kbd> / <kbd>→</kbd></td><td>Previous / next image in folder</td></tr>
  <tr><td>Scroll wheel</td><td>Zoom in / out on viewer</td></tr>
  <tr><td>Left drag</td><td>Pan the image</td></tr>
</table>

<h2>Global shortcuts</h2>
<table>
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><kbd>Ctrl+O</kbd></td><td>Open image file</td></tr>
  <tr><td><kbd>Ctrl+F</kbd></td><td>Switch to Fusion tab</td></tr>
  <tr><td><kbd>Ctrl+3</kbd></td><td>Switch to 3D tab</td></tr>
  <tr><td><kbd>Ctrl+M</kbd></td><td>Open Image Comparison</td></tr>
  <tr><td><kbd>Ctrl+Shift+L</kbd></td><td>Load pipeline</td></tr>
  <tr><td><kbd>F1</kbd></td><td>Open this User Guide</td></tr>
</table>
"""),


"Inspector Panel": _page("Inspector Panel", """
<p>The Inspector panel is a dock on the left side. It shows analysis results for the current image and active tool.
Scroll down inside it to see all sections.</p>

<h2>Sections</h2>
<table>
  <tr><th>Section</th><th>What it shows</th><th>Appears when</th></tr>
  <tr><td>Pixel</td><td>R, G, B (or grayscale) value under the mouse cursor</td><td>Always</td></tr>
  <tr><td>Histogram</td><td>Intensity distribution of the full image or ROI</td><td>Always</td></tr>
  <tr><td>ROI Stats</td><td>Mean, min, max, std-dev for the drawn rectangle</td><td>After drawing ROI</td></tr>
  <tr><td>Line Profile</td><td>Brightness chart along the drawn line</td><td>After drawing profile</td></tr>
  <tr><td>Measurement</td><td>Distance in px and mm, angle, coordinates</td><td>After measuring</td></tr>
  <tr><td>Annotations</td><td>All placed markers with labels and coordinates</td><td>After annotating</td></tr>
</table>

<div class="tip">The Inspector opens automatically when you finish an action (draw ROI, measure, etc.) and scrolls to the relevant section.</div>
"""),


"Status Bar": _page("Status Bar", """
<p>The status bar is the thin strip at the very bottom of the window.</p>

<h2>What each field means</h2>
<pre>[1]  19RGB.bmp   11%  (1:8.8)   ● F:14  Q:98   Tool: MASK — draw polygon</pre>

<table>
  <tr><th>Field</th><th>Meaning</th></tr>
  <tr><td><code>[1]</code></td><td>Active viewer cell number (1–4 in multi-view layout)</td></tr>
  <tr><td><code>19RGB.bmp</code></td><td>Current image filename</td></tr>
  <tr><td><code>11%</code></td><td>Current zoom level</td></tr>
  <tr><td><code>(1:8.8)</code></td><td>Pixel ratio — 1 screen pixel = 8.8 image pixels</td></tr>
  <tr><td><code>● F:14</code></td><td>Focus score (dot colour = RED/AMBER/GREEN)</td></tr>
  <tr><td><code>Q:98</code></td><td>Image quality score</td></tr>
  <tr><td>Tool hint</td><td>Current tool name and one-line usage reminder</td></tr>
</table>
"""),

}


# ── Dialog ─────────────────────────────────────────────────────────────────

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("User Guide")
        self.resize(1020, 680)
        self.setStyleSheet(_CSS_BASE)
        self._build()
        QShortcut(QKeySequence("Escape"), self, self.close)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ────────────────────────────────────────────────
        top = QWidget()
        top.setFixedHeight(38)
        top.setStyleSheet(f"background:#0C1520; border-bottom:1px solid {_BORDER};")
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(12, 4, 12, 4)
        top_row.setSpacing(8)

        title = QLabel("  User Guide")
        title.setStyleSheet(f"color:{_CYAN}; font-size:11px; font-weight:700; letter-spacing:1px;")
        top_row.addWidget(title)
        top_row.addStretch()

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search…")
        self._search.setFixedWidth(200)
        self._search.textChanged.connect(self._on_search)
        top_row.addWidget(self._search)

        close_btn = QPushButton("✕  Close")
        close_btn.clicked.connect(self.close)
        top_row.addWidget(close_btn)

        root.addWidget(top)

        # ── Splitter ───────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        root.addWidget(splitter, stretch=1)

        # Nav list
        self._nav = QListWidget()
        self._nav.setFixedWidth(195)
        self._nav.setSpacing(1)
        for name in SECTIONS:
            item = QListWidgetItem(f"  {name}")
            item.setSizeHint(QSize(195, 34))
            self._nav.addItem(item)
        self._nav.currentRowChanged.connect(self._on_nav)
        splitter.addWidget(self._nav)

        # Content browser
        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(False)
        self._browser.setFont(QFont("Segoe UI", 11))
        splitter.addWidget(self._browser)
        splitter.setSizes([195, 825])

        # Select first section
        self._nav.setCurrentRow(0)

    def _on_nav(self, row: int):
        if row < 0:
            return
        name = list(SECTIONS.keys())[row]
        self._browser.setHtml(SECTIONS[name])

    def _on_search(self, text: str):
        text = text.strip().lower()
        for i in range(self._nav.count()):
            item = self._nav.item(i)
            name = list(SECTIONS.keys())[i].lower()
            item.setHidden(bool(text) and text not in name)
