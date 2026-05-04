"""
In-app User Guide window — comprehensive reference.

Layout:
  Left  — navigation list (sections)
  Right — QTextBrowser showing rich HTML for the selected section

Opened from Help menu → User Guide (F1).
"""

from PyQt6.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
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
  h2     {{ color:{_CYAN}; font-size:14px; margin-top:26px; margin-bottom:6px; }}
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
  .math  {{ background:#070F16; border-left:3px solid {_PURPLE}; padding:8px 14px;
            margin:10px 0; border-radius:0 4px 4px 0;
            font-family:Consolas,monospace; font-size:11px; color:#CCAAFF; }}
  .ref   {{ color:#557799; font-size:10px; font-style:italic; }}
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

# ══════════════════════════════════════════════════════════════════════════
"Overview": _page("Overview — What This App Does", """
<p>This is an industrial image viewer built for surface defect inspection.
It combines a high-resolution image viewer with professional analysis tools
normally only found in dedicated machine vision systems (Halcon, Cognex, Keyence).</p>

<h2>Who it is for</h2>
<ul>
  <li>Quality engineers inspecting manufactured parts</li>
  <li>Vision system developers building inspection pipelines</li>
  <li>Researchers analyzing surface topology with multi-light imaging</li>
</ul>

<h2>Feature summary</h2>
<table>
  <tr><th>Feature</th><th>What it gives you</th></tr>
  <tr><td>Inspect mode</td><td>Pan, zoom, ROI, measure, annotate, mask — full inspection toolkit</td></tr>
  <tr><td>Focus scoring</td><td>Per-cell sharpness grid — know before you inspect if the image is sharp enough</td></tr>
  <tr><td>Compare</td><td>Auto-align two images, compute |A−B|, classify defects, PASS/FAIL verdict</td></tr>
  <tr><td>Filter Pipeline</td><td>28 stackable image processing filters with real-time preview</td></tr>
  <tr><td>Illumination Fusion</td><td>10 fusion operations — combine multiple lighting angles into one image</td></tr>
  <tr><td>3D Surface</td><td>Real surface topology from Photometric Stereo, or visual brightness mesh</td></tr>
</table>

<h2>Window layout</h2>
<pre>
┌──────────────────────────────────────────────────────────────┐
│  File   View   Pipeline   Tools   Help       ← Menu bar      │
├──────────────────────────────────────────────────────────────┤
│  Inspect │ Focus │ Compare │ Filters │ Fusion │ 3D  ← Tabs   │
│  [context toolbar — only visible when a tool is active]      │
├──────────┬───────────────────────────────────────────────────┤
│ Browser  │                                                   │
│ Inspector│   Main workspace                                  │
│ Pipeline │   (viewer / compare / fusion / 3D)                │
│ Fusion   │                                                   │
└──────────┴───────────────────────────────────────────────────┘
│  Status bar — file · zoom · focus score · tool hint          │
└──────────────────────────────────────────────────────────────┘
</pre>
"""),


# ══════════════════════════════════════════════════════════════════════════
"File Browser": _page("File Browser", """
<p>The File Browser is the left-most dock panel. Navigate folders and open images.</p>

<h2>How to use</h2>
<ol>
  <li>Click any folder in the tree to open it</li>
  <li>The image list shows all supported images in that folder</li>
  <li>Click an image to open it in the main viewer</li>
  <li>Use <strong>Filter filenames…</strong> to search by name</li>
  <li><kbd>←</kbd> / <kbd>→</kbd> arrow keys to step through images</li>
</ol>

<h2>Supported formats</h2>
<p>TIFF · TIF · PNG · BMP · JPG · JPEG · PGM · PPM · EXR — including 16-bit grayscale TIFF</p>

<h2>Fusion shortcut</h2>
<p>When the Fusion panel is open, the folder list appears inside it.
Double-click any filename there to add it directly to the fusion set — no file dialog needed.</p>

<div class="tip">The app remembers the last folder you opened across sessions.</div>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Inspect Tools": _page("Inspect Mode — Tools", """
<p>Inspect is the default mode. The main viewer shows the image at full resolution.</p>

<h2>Selecting a tool</h2>
<p>Open <strong>Tools menu</strong> and pick a tool, or press the keyboard shortcut:</p>
<table>
  <tr><th>Tool</th><th>Key</th><th>What it does</th></tr>
  <tr><td><span class="badge cyan">↖ Navigate</span></td><td><kbd>Esc</kbd></td>
      <td>Default — pan with left drag, zoom with scroll. Toolbar hidden.</td></tr>
  <tr><td><span class="badge cyan">⬛ ROI</span></td><td><kbd>R</kbd></td>
      <td>Draw rectangle → region statistics in Inspector</td></tr>
  <tr><td><span class="badge cyan">📈 Profile</span></td><td><kbd>P</kbd></td>
      <td>Draw line → intensity chart in Inspector</td></tr>
  <tr><td><span class="badge amber">📍 Annotate</span></td><td><kbd>A</kbd></td>
      <td>Click to place defect marker — choose label from popup</td></tr>
  <tr><td><span class="badge green">↔ Measure</span></td><td><kbd>M</kbd></td>
      <td>Drag to measure distance in pixels and mm</td></tr>
  <tr><td><span class="badge purple">⬡ Mask</span></td><td><kbd>K</kbd></td>
      <td>Draw polygon — metrics computed only inside masked area</td></tr>
</table>

<div class="tip">The active tool badge appears on the left of the toolbar in its colour.
Navigate is dim grey — it is the "go back" action, not an active analysis tool.</div>

<hr>
<h2>Navigate <kbd>Esc</kbd></h2>
<ul>
  <li>Left drag → pan</li>
  <li>Right drag or scroll wheel → zoom</li>
</ul>
<p>Toolbar is hidden in this mode — clean view.</p>

<hr>
<h2>ROI — Region of Interest <kbd>R</kbd></h2>
<p>Draw a rectangle. Inspector shows statistics for that region:</p>
<ul>
  <li>Mean, min, max, standard deviation</li>
  <li>Histogram of the region</li>
  <li>Channel breakdown for colour images</li>
</ul>
<p><strong>Use for:</strong> Comparing brightness between a defect and surrounding good surface.
Checking if a dark spot is statistically different from background.</p>

<hr>
<h2>Profile — Line Intensity <kbd>P</kbd></h2>
<p>Draw a line across the image. Inspector shows a brightness chart along that line.</p>
<p><strong>Use for:</strong> Measuring scratch width (distance between dips in the chart).
Checking coating uniformity. Verifying edge sharpness.</p>

<hr>
<h2>Annotate — Defect Markers <kbd>A</kbd></h2>
<p>Click anywhere on the image. A label popup appears — choose:</p>
<p>Scratch · Pit · Contamination · Burr · Crack · OK · Other</p>
<div class="tip">Annotations are automatically saved next to the image as
<code>imagename.annotations.json</code> and loaded automatically next time.</div>

<hr>
<h2>Measure — Distance <kbd>M</kbd></h2>
<p>Drag from A to B. Inspector shows pixel distance, mm distance (if calibrated), and angle.</p>
<h3>Setting the scale</h3>
<ol>
  <li>Photograph a ruler at the same working distance as your parts</li>
  <li>Draw a Measure line over a known distance on the ruler</li>
  <li>Open <strong>Tools → Set Scale Calibration…</strong></li>
  <li>Enter the mm/pixel ratio</li>
</ol>

<hr>
<h2>Mask — Inspection Polygon <kbd>K</kbd></h2>
<p>Draw a polygon. All metrics are computed <em>only inside</em> the masked area.</p>
<table>
  <tr><th>Button</th><th>What it does</th></tr>
  <tr><td>✦ Auto-Detect</td><td>Detect specular reflection regions and suggest a mask</td></tr>
  <tr><td>📋 Apply to Folder</td><td>Copy mask to all images in the folder</td></tr>
  <tr><td>⧉ Find All Similar</td><td>Find all identical regions and mask them all</td></tr>
  <tr><td>💾 Save Mask</td><td>Save mask or export masked image file</td></tr>
  <tr><td>⬡ Clear Mask</td><td>Remove mask — return to full image analysis</td></tr>
</table>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Focus": _page("Focus Mode — Sharpness Scoring", """
<p>Focus mode overlays a sharpness score grid on the image.
Use it to verify an image is sharp enough before defect inspection.</p>

<h2>The grid</h2>
<p>The image is divided into an 8×8 grid (64 cells). Each cell is scored independently.</p>
<table>
  <tr><th>Colour</th><th>Score</th><th>Meaning</th></tr>
  <tr><td><span class="badge green">GREEN</span></td><td>≥ 72%</td><td>Sharp — safe to inspect</td></tr>
  <tr><td><span class="badge amber">AMBER</span></td><td>38–72%</td><td>Soft — marginal, check lens</td></tr>
  <tr><td><span class="badge red">RED</span></td><td>&lt; 38%</td><td>Blurry — reject this image</td></tr>
</table>

<h2>Algorithm — how the score is computed</h2>
<p>Two independent metrics are fused per cell:</p>

<h3>1. Laplacian Variance (Brenner 1976)</h3>
<div class="math">
Score₁ = Var( ∇²I )  =  Var( I[x,y] − 2·I[x+1,y] + I[x+2,y] )
</div>
<p>The Laplacian (second derivative) is large where pixel values change rapidly — i.e. at sharp edges.
A blurry image has smooth gradients so its Laplacian is small → low variance → low score.
Best at low sensor noise. Fast — one convolution.</p>

<h3>2. Tenengrad (Tenenbaum 1970, revalidated IEEE 2013)</h3>
<div class="math">
Score₂ = Σ max( Gx² + Gy² − threshold, 0 )
where Gx, Gy = Sobel gradient at each pixel
</div>
<p>Sums the squared gradient magnitude across the cell.
Only gradients above a noise threshold are counted — this makes Tenengrad far more
noise-robust than Laplacian Variance.
Best metric for real camera noise conditions.</p>

<h3>Fusion rule</h3>
<ul>
  <li>Both agree → HIGH confidence, score shown directly</li>
  <li>Disagree by one category → MEDIUM confidence</li>
  <li>Disagree by two categories → LOW confidence (cell shown with dashed border)</li>
</ul>

<h2>Reference modes</h2>
<table>
  <tr><th>Mode</th><th>When to use</th></tr>
  <tr><td>RELATIVE</td><td>No reference — relative to best cell in the same image. <strong>Not for production.</strong></td></tr>
  <tr><td>AUTO-REF</td><td>Tracks sharpest image seen this session. Better than relative.</td></tr>
  <tr><td>LOCKED REF ✓</td><td>You locked a known-good reference image. <strong>Use this in production.</strong></td></tr>
</table>

<div class="tip">Lock a reference once when the lens and lighting are perfectly set.
100% = as sharp as that calibrated reference. 60% = something has shifted.</div>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Compare": _page("Compare Mode — A/B Defect Detection", """
<p>Compare mode loads two images (A = reference, B = part under test),
auto-aligns them, computes the difference, detects defects, and gives a PASS/FAIL verdict.</p>

<h2>How to use</h2>
<ol>
  <li>Switch to <strong>Compare</strong> tab</li>
  <li>Load Image A (good reference part)</li>
  <li>Load Image B (part to inspect)</li>
  <li>The system runs automatically — alignment → diff → detection</li>
  <li>Read the PASS / FAIL verdict</li>
</ol>

<h2>Processing pipeline — step by step</h2>

<h3>Step 1 — Brightness Normalization</h3>
<div class="math">
B_norm = B × ( mean(A) / mean(B) )
</div>
<p>Matches overall brightness of B to A. Eliminates false defects caused by lighting variation
between captures (different day, different illuminator warm-up state).</p>

<h3>Step 2 — Alignment: ORB + RANSAC Homography</h3>
<p><strong>ORB (Oriented FAST and Rotated BRIEF) — Rublee et al. 2011</strong></p>
<div class="math">
1. Detect keypoints in A and B using FAST corner detector
2. Compute binary BRIEF descriptors (256-bit, rotation-invariant)
3. Match descriptors using Hamming distance
4. Estimate homography H via RANSAC (Random Sample Consensus)
5. Warp B: B_aligned = H · B
</div>
<p>RANSAC eliminates outlier matches caused by defects themselves (a defect on B has no match in A).
The homography corrects translation, rotation, scale, and mild perspective shift
between captures — eliminates fixture positioning false defects.</p>

<h3>Step 3 — Absolute Difference Map</h3>
<div class="math">
D(x,y) = | A(x,y) − B_aligned(x,y) |
</div>
<p>Per-pixel absolute difference. A perfectly good part gives D≈0 everywhere.
Defects give D>0 — shown as HOT colormap (black → red → yellow → white).</p>

<h3>Step 4 — Noise Removal</h3>
<div class="math">
D_clean = MorphOpen( Threshold( GaussianBlur(D, σ=1.5), t ) )
</div>
<p>Gaussian blur removes sub-pixel camera noise. Thresholding removes residual alignment error.
Morphological opening removes isolated single-pixel hits (stuck pixels, demosaicing artifacts).
What remains are real, spatially-extended defects.</p>

<h3>Step 5 — Connected Component Analysis (CCA)</h3>
<p>Remaining bright regions in D_clean are labeled as separate blobs.
Each blob is measured for area, centroid, bounding box, and intensity.</p>

<h2>Defect severity classification</h2>
<table>
  <tr><th>Severity</th><th>Area</th><th>Action</th></tr>
  <tr><td><span class="badge green">COSMETIC</span></td><td>&lt; 25 px²</td><td>Document and accept</td></tr>
  <tr><td><span class="badge amber">FUNCTIONAL</span></td><td>25–200 px²</td><td>Engineer review required</td></tr>
  <tr><td><span class="badge red">CRITICAL</span></td><td>&gt; 200 px²</td><td>Reject — do not ship</td></tr>
</table>

<h2>Display modes</h2>
<table>
  <tr><th>Mode</th><th>What it shows</th></tr>
  <tr><td>Diff Map</td><td>|A−B| as HOT colormap — primary defect view</td></tr>
  <tr><td>Signed ±128</td><td>Directional diff — blue = A darker, red = A brighter than B</td></tr>
  <tr><td>Blend 50/50</td><td>A and B overlaid at 50% — visual alignment check</td></tr>
  <tr><td>Flicker</td><td>Alternates A and B every 600 ms — human eye catches texture differences</td></tr>
</table>

<div class="tip">Click <strong>→ Jump to defect</strong> on any defect card to pan all viewers to that location at 4× zoom.</div>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Filters — Pre-process": _page("Filter Pipeline — ① Pre-process", """
<p>Pre-process filters are applied first. Their purpose is to remove camera noise and
correct exposure so that the subsequent Enhance and Detect filters work on clean data.</p>

<div class="warn">Always start with at least one noise filter (Gaussian or Bilateral)
before any edge detection filter. Detection filters amplify noise just as well as signal.</div>

<hr>
<h2>1. Gaussian Blur</h2>
<p class="ref">Gaussian filter — standard linear smoothing, textbook signal processing</p>
<div class="math">
Output(x,y) = Σ Σ G(dx,dy,σ) · Input(x+dx, y+dy)

G(x,y,σ) = (1 / 2πσ²) · exp( −(x²+y²) / 2σ² )

kernel size k = ceil(6σ) + 1  (always odd)
</div>
<p>The Gaussian kernel weights nearby pixels by a bell-curve falloff.
Result: smooth blurring with no edge ringing (unlike box blur).
<strong>Sigma</strong> is the only parameter — controls blur radius.
σ=1 removes single-pixel hot pixels. σ=3+ removes fine texture.
Use σ=0.5–1.5 before Canny or Sobel to suppress noise without losing edge position.</p>

<hr>
<h2>2. Median Filter</h2>
<p class="ref">Median filter — Tukey (1977), optimal for impulse noise removal</p>
<div class="math">
Output(x,y) = median of all pixel values in k×k neighborhood around (x,y)
</div>
<p>Replaces each pixel with the middle value of its neighbors (not the average).
Dramatically more effective than Gaussian at removing salt-and-pepper noise (stuck pixels, dust)
because a single outlier cannot shift the median.
<strong>Unlike Gaussian, it preserves edges perfectly</strong> — the median of a neighborhood
crossing an edge is still on one side of the edge.
Kernel size 3 = removes isolated hot/cold pixels. 5–7 = removes larger noise clusters.</p>

<hr>
<h2>3. Bilateral Filter</h2>
<p class="ref">Tomasi &amp; Manduchi (1998) — edge-preserving smoother</p>
<div class="math">
Output(x,y) = Σ w_spatial(dx,dy) · w_color(I(x,y), I(x+dx,y+dy)) · I(x+dx,y+dy)
            ─────────────────────────────────────────────────────────────────────
              Σ w_spatial(dx,dy) · w_color(I(x,y), I(x+dx,y+dy))

w_spatial(dx,dy) = exp( −(dx²+dy²) / 2σ_space² )
w_color(Ic,In)   = exp( −(Ic−In)² / 2σ_color² )
</div>
<p>The spatial weight is a standard Gaussian — closer pixels contribute more.
The color weight is also Gaussian but in intensity space — pixels that differ greatly
in brightness contribute very little. The combined effect: pixels across an edge do
<em>not</em> influence each other (because their intensity differs by σ_color).
<strong>Result: noise smoothed in flat areas, edges kept sharp.</strong>
Best general pre-processor for industrial inspection.
Diameter=9, σ_color=75, σ_space=75 is a solid starting point.</p>

<hr>
<h2>4. NL-Means Denoise</h2>
<p class="ref">Buades, Coll &amp; Morel (2005) — non-local means denoising, IEEE CVPR</p>
<div class="math">
Output(x,y) = Σ w(p,q) · I(q)
              ───────────────
               Σ w(p,q)

w(p,q) = exp( −‖N(p) − N(q)‖² / h² )

N(p) = patch of template_window × template_window pixels centered at p
h = filter strength parameter
</div>
<p>Instead of only using nearby pixels (bilateral), NL-Means searches the
<em>entire image</em> within a search window for patches similar to the patch at each pixel.
Pixels in similar patches — even if far away — are used to estimate the true (noise-free) value.
<strong>Result: highest quality denoising, preserves fine texture that bilateral misses.</strong>
Price: slower (each pixel searches a 21×21 or larger area).
H=10 is balanced. H=3–5 for texture-critical surfaces. H=20+ for very noisy cameras.</p>

<hr>
<h2>5. Brightness / Contrast</h2>
<div class="math">
Output = Input × (1 + Contrast/255) + Brightness
</div>
<p>Linear exposure correction. Contrast scales the tonal range (higher = farther from midpoint).
Brightness shifts all values uniformly.
Use when the image is obviously under- or over-exposed before any other processing.</p>

<hr>
<h2>6. Normalize</h2>
<div class="math">
Output(x,y) = 255 × ( Input(x,y) − min ) / ( max − min )
</div>
<p>Stretches the image to use the full 0–255 range.
Essential when a 16-bit camera captures images where actual pixel values only span
e.g. 1000–4000 out of the 65535 range — after normalization, the full range is used
and defects become visible. No parameters — one click.</p>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Filters — Enhance": _page("Filter Pipeline — ② Enhance", """
<p>Enhance filters increase the visual contrast of defects.
Apply after pre-processing, before detection.</p>

<hr>
<h2>7. CLAHE — Contrast Limited Adaptive Histogram Equalization</h2>
<p class="ref">Zuiderveld (1994) — standard in medical and industrial machine vision</p>
<div class="math">
1. Divide image into tile_size × tile_size tiles
2. For each tile: compute histogram → clip at clip_limit → equalize (CDF-based)
3. Interpolate result from adjacent tile borders (bilinear)
4. Applied to L channel (LAB color space) for color images
</div>
<p>Global histogram equalization stretches contrast based on the whole image,
so a small dark defect on a bright surface is completely overwhelmed by the
bright background and stays dark after equalization.
CLAHE computes equalization <em>locally</em> per tile — the dark defect gets
its own local stretch and becomes visible.
The clip_limit prevents over-amplification of noise in near-flat regions.
<strong>This is the single most effective filter for low-contrast industrial defects.</strong>
Clip limit 2.0–4.0, tile size 8 is the standard setup.</p>

<hr>
<h2>8. Histogram Equalization</h2>
<div class="math">
Output(x,y) = round( CDF(Input(x,y)) × 255 )

CDF(v) = cumulative sum of histogram up to value v, normalized to [0,1]
</div>
<p>Global equalization — stretches intensity so all values are equally likely.
Fast and simple but does not work well on industrial images with large uniform
regions (the background dominates the histogram). Use CLAHE instead for inspection.
Useful for quick overview or when the whole image needs more contrast.</p>

<hr>
<h2>9. Gamma Correction</h2>
<div class="math">
Output(x,y) = 255 × ( Input(x,y) / 255 )^(1/γ)
</div>
<p>Non-linear brightness curve applied via a lookup table.
γ &lt; 1.0 lifts shadows (makes the image brighter, especially in dark regions).
γ &gt; 1.0 darkens midtones.
Useful for revealing defects in dark regions without blowing out bright areas.
γ=0.5 doubles the perceived brightness in shadows without touching highlights.</p>

<hr>
<h2>10. Levels</h2>
<div class="math">
Step 1: clip_in  = clamp( Input, in_black, in_white )
Step 2: stretch  = ( clip_in − in_black ) / ( in_white − in_black )
Step 3: gamma    = stretch^(1/midtones)
Step 4: Output   = gamma × (out_white − out_black) + out_black
</div>
<p>Full professional tonal range control — same as Photoshop Levels dialog.
In Black/In White clip the input range (sets the black and white points).
Midtones adjusts the gamma of the middle range without affecting black/white points.
Out Black/Out White limit the output range (useful for display matching or calibration).
The most precise exposure correction tool in the pipeline.</p>

<hr>
<h2>11. Unsharp Mask</h2>
<p class="ref">Industry-standard sharpening — photographic origin, 1930s darkroom technique, formalized digitally by Gonzalez &amp; Woods</p>
<div class="math">
blurred = GaussianBlur( Input, radius )
diff    = Input − blurred       ← this IS the "unsharp mask" (high-pass)
Output  = Input + strength × diff,   where |diff| ≥ threshold
</div>
<p>Creates a blurred copy of the image and subtracts it from the original to extract edges.
That edge signal (diff) is added back scaled by strength.
The threshold parameter prevents noise amplification — only edges above a minimum
contrast level are sharpened.
<strong>Radius 1.0, Strength 1.5, Threshold 0 is the default inspection setup.</strong>
For noisy images, set Threshold to 5–10 to skip noise amplification.</p>

<hr>
<h2>12. Laplacian Sharpen</h2>
<p class="ref">Second-order derivative sharpening — Marr &amp; Hildreth (1980)</p>
<div class="math">
∇²I = ∂²I/∂x² + ∂²I/∂y²         (computed via 3×3 Laplacian kernel)
Output = Input + strength × |∇²I|
</div>
<p>Uses the second derivative instead of first derivative (as in Unsharp Mask).
More sensitive to sharp discontinuities and zero-crossings — better at
revealing micro-cracks, pit edges, and surface discontinuities.
May amplify noise more aggressively than Unsharp Mask on rough surfaces.
Use on smooth machined parts where noise is low.</p>

<hr>
<h2>13. Top-Hat Transform</h2>
<p class="ref">Mathematical morphology — Serra (1983); Meyer (1979)</p>
<div class="math">
White Top-Hat(I, B) = I − Opening(I, B)
Opening(I, B)       = Dilate( Erode(I, B), B )
B = structuring element (ellipse of size k×k)
</div>
<p>Opening removes bright features smaller than the structuring element.
Subtracting the opening from the original leaves <em>only</em> those removed bright features.
<strong>Top-Hat isolates bright features smaller than the kernel on any background.</strong>
Use for bright specks, protrusions, raised burrs, high-reflectance spots.
Kernel size should be slightly larger than the defect size. Shape: ellipse for most surfaces.</p>

<hr>
<h2>14. Black-Hat Transform</h2>
<p class="ref">Mathematical morphology — Serra (1983)</p>
<div class="math">
Black Hat(I, B) = Closing(I, B) − I
Closing(I, B)   = Erode( Dilate(I, B), B )
</div>
<p>The dual of Top-Hat. Closing fills dark features smaller than the kernel.
Subtracting the original from the closing leaves only those dark features.
<strong>Black-Hat isolates dark features smaller than the kernel.</strong>
Use for pits, voids, holes, dark scratches on reflective surfaces.
Direct complement: if defects appear bright → Top-Hat; if dark → Black-Hat.</p>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Filters — Detect": _page("Filter Pipeline — ③ Detect", """
<p>Detect filters extract the defect signal from a pre-processed, enhanced image.
Their output is typically a grayscale map where bright pixels = defect signal.</p>

<hr>
<h2>15. Canny Edge Detection</h2>
<p class="ref">Canny (1986) — IEEE Transactions on Pattern Analysis and Machine Intelligence.
Considered the optimal edge detector for its balance of detection, localization, and single-response criteria.</p>
<div class="math">
Step 1: Smooth with Gaussian (built-in, aperture-controlled)
Step 2: Compute gradient magnitude M and angle θ using Sobel
        Gx = Sobel(I, x),  Gy = Sobel(I, y)
        M = sqrt(Gx²+Gy²),  θ = atan2(Gy, Gx)
Step 3: Non-maximum suppression — thin edges to 1 pixel
        Keep pixel only if M is local maximum in direction θ
Step 4: Hysteresis thresholding
        Strong edges: M ≥ High Threshold → always kept
        Weak edges:   M between Low and High → kept only if connected to strong edge
        Noise:        M &lt; Low Threshold → discarded
</div>
<p><strong>Low Threshold</strong>: too low = too much noise; too high = missed edges.
<strong>High Threshold</strong>: keep ratio High:Low = 2:1 or 3:1.
<strong>Aperture</strong>: Sobel kernel size. 3 for fine detail, 5 or 7 for broad edges on noisy images.
Output: binary 1-pixel-wide edge map.</p>

<hr>
<h2>16. Sobel Edge</h2>
<p class="ref">Sobel &amp; Feldman (1968) — first-derivative gradient edge detector</p>
<div class="math">
Kx = [−1  0  +1]   Ky = [−1  −2  −1]
     [−2  0  +2]        [ 0   0   0]
     [−1  0  +1]        [+1  +2  +1]

Gx = Kx * I,  Gy = Ky * I
Combined = sqrt(Gx² + Gy²)    (normalized 0–255)
</div>
<p>Computes image gradient — rate of change of brightness.
Edges (transitions between dark and bright) produce large gradient values.
Direction X only detects vertical edges. Direction Y only detects horizontal edges.
Combined detects all edges.
<strong>Faster than Canny but produces thicker edges.</strong>
Use when edge precision is less important than speed or when edges need to be wide
enough to close into regions for area measurement.</p>

<hr>
<h2>17. Difference of Gaussians (DoG)</h2>
<p class="ref">Marr &amp; Hildreth (1980) — approximates Laplacian of Gaussian (LoG), models retinal ganglion cells</p>
<div class="math">
DoG(I, σ₁, σ₂) = GaussianBlur(I, σ₁) − GaussianBlur(I, σ₂)

where σ₁ &lt; σ₂ (σ₂/σ₁ ≈ 1.6 mimics human visual system — Marr-Hildreth)
</div>
<p>DoG is a band-pass filter in the spatial frequency domain — it responds to features
at one specific size scale and suppresses everything else.
Features smaller than σ₁ and larger than σ₂ are both suppressed.
<strong>Use to target defects of one specific size</strong> while ignoring smaller noise and larger structures.
Increase the gap between σ₁ and σ₂ to widen the frequency band (catch more sizes).</p>

<hr>
<h2>18. Morphological Gradient</h2>
<p class="ref">Mathematical morphology — Serra (1983)</p>
<div class="math">
MorphGradient(I, B) = Dilation(I, B) − Erosion(I, B)
</div>
<p>Dilation expands bright regions; erosion shrinks them.
Their difference is the boundary ring around every object.
<strong>Produces thick, connected edge outlines</strong> — useful when edges must form closed regions
for area measurement by connected-component analysis.
Unlike Canny (1-pixel-thin edges), morphological gradient gives fat boundaries
that are easier to threshold into closed blobs.</p>

<hr>
<h2>19. Gabor Filter Bank</h2>
<p class="ref">Daugman (1985) — minimum uncertainty in joint space-frequency representation (Heisenberg limit).
Validated for industrial surface defect detection in IEEE/JIM (2025).</p>
<div class="math">
Gabor(x,y,θ,f,σx,σy) = exp(−x'²/2σx² − y'²/2σy²) · cos(2π·f·x')

x' =  x·cos(θ) + y·sin(θ)    ← rotated coordinates
y' = −x·sin(θ) + y·cos(θ)

For N orientations: θ_i = i·π/N,  i = 0..N−1
For S scales: f_s = f · 0.7^s
Response = |filter2D(I, GaborKernel(θ_i, f_s))|

Final = max / mean / sum / energy over all orientation×scale responses
</div>
<p>A Gabor filter is a Gaussian-windowed sinusoid. It is tuned to a specific orientation
and spatial frequency — it responds strongly only to texture running in direction θ at frequency f.
A bank of N orientations at S scales covers all directions simultaneously.
<strong>Best filter for directional surface defects: machining marks, weave, scratches at any angle.</strong>
The "max" combine mode keeps the strongest signal from any direction — one number captures
whether a scratch exists regardless of its angle.
Frequency 0.15 and 6 orientations is the standard starting point.</p>

<hr>
<h2>20. Wavelet Decomposition</h2>
<p class="ref">Mallat (1989) — multiresolution analysis; Haar (1909) wavelet basis.
Applied to foil/surface defect detection in ACM Proceedings (2025).</p>
<div class="math">
Haar decomposition at level L:
  LL (approximation) = low-pass in both x and y → coarse shape
  LH (horizontal)    = low-pass x, high-pass y  → horizontal edges
  HL (vertical)      = high-pass x, low-pass y  → vertical edges
  HH (diagonal)      = high-pass in both        → diagonal detail

Each level L halves spatial resolution and doubles the feature scale.
Level 1 = finest detail (1–2 px features)
Level 2 = 2–4 px features
Level 3 = 4–8 px features, etc.

detail_all = max(LH, HL, HH) at selected level → broadest sensitivity
all_levels = max across all levels → entire frequency range
</div>
<p>Wavelet decomposition separates the image into a hierarchy of frequency bands.
Each band reveals defects at one specific spatial scale.
<strong>Use Level 1–2 for fine surface defects; Level 3–4 for cracks or large flaws.</strong>
Enable "Enhance" to apply histogram equalization to the output band — makes subtle detail visible.</p>

<hr>
<h2>21. FFT Analysis</h2>
<p class="ref">Cooley &amp; Tukey (1965) — Fast Fourier Transform algorithm. Standard signal/image processing.</p>
<div class="math">
F(u,v) = Σx Σy I(x,y) · exp(−2πi(ux/W + vy/H))

|F(u,v)| = magnitude spectrum (spatial frequency content)
∠F(u,v) = phase spectrum

Periodic defects → bright off-center spots in |F(u,v)|
DC component (mean brightness) → central bright spot

Filtering in frequency domain:
  Highpass: zero all F(u,v) where dist(u,v,center) &lt; cutoff_low
  Lowpass:  zero all F(u,v) where dist(u,v,center) &gt; cutoff_high
  Bandpass: keep only the ring between cutoff_low and cutoff_high
  Reconstruct: I_filtered = ifft2( ifftshift( F_masked ) )
</div>
<p>The FFT transforms the image from spatial domain to frequency domain.
<strong>Magnitude mode:</strong> periodic defects (banding, moire, regular machining marks) appear as
distinct bright spots away from the center — easy to identify and notch-filter out.
<strong>Highpass mode:</strong> removes low-frequency illumination gradients, sharpens fine detail.
<strong>Bandpass mode:</strong> selects one spatial frequency band — targets defects of one specific period.</p>

<hr>
<h2>22. LBP Texture</h2>
<p class="ref">Ojala, Pietikäinen &amp; Harwood (1994, 1996, 2002) — Local Binary Pattern, IEEE TPAMI</p>
<div class="math">
LBP(x,y) = Σ s(I(p) − I(c)) · 2^i
            i=0..P−1

I(c) = intensity at center pixel (x,y)
I(p) = intensity at point p on circle of radius R, P equally-spaced points
s(x) = 1 if x ≥ 0,  else 0

Result: P-bit binary code (0..2^P−1) — the local texture pattern
</div>
<p>Each pixel becomes a binary code describing how its neighbors compare to itself.
<strong>LBP is illumination-invariant</strong> — if all pixels shift equally in brightness
(different exposure), the comparison signs (s) stay the same → same LBP code.
Defective regions have different texture patterns than the surrounding good surface
→ they produce a different LBP histogram.
Radius 1–2 = micro-texture (surface finish). Radius 3–5 = meso-texture (weave, pattern).
Uniform mode: only keep codes with ≤ 2 bit transitions (90% of real textures).</p>

<hr>
<h2>23. Morphological Open</h2>
<div class="math">Open(I, B) = Dilate( Erode(I, B), B )</div>
<p>Erosion removes bright features smaller than B. Dilation restores larger ones that survived.
<strong>Use after thresholding</strong> to eliminate isolated false-positive pixels before blob counting.
Set kernel size to the minimum defect size — smaller noise is removed, real defects survive.</p>

<hr>
<h2>24. Morphological Close</h2>
<div class="math">Close(I, B) = Erode( Dilate(I, B), B )</div>
<p>Dilation fills small dark holes inside bright objects. Erosion restores their boundaries.
<strong>Use after thresholding</strong> to join fragmented defect blobs.
A scratch with small gaps will be joined into a single blob for correct area measurement.</p>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Filters — Visualize": _page("Filter Pipeline — ④ Visualize", """
<p>Visualize filters map the filter pipeline result to a colour representation.
Apply last in the chain.</p>

<hr>
<h2>25. False Color (LUT)</h2>
<p class="ref">Scientific colormaps — Matplotlib / OpenCV colormaps. VIRIDIS: Nathaniel Smith &amp; Stéfan van der Walt (2015) — perceptually uniform.</p>
<div class="math">
For each pixel: gray_clipped = clamp(gray, range_min, range_max)
               gray_scaled  = (gray_clipped − range_min) / (range_max − range_min) × 255
               RGB = colormap_LUT[ gray_scaled ]
</div>
<p>Maps grayscale intensity to a scientific colour scale using a lookup table.</p>
<table>
  <tr><th>Map</th><th>Use when</th></tr>
  <tr><td>JET</td><td>Blue→green→red. Maximum colour spread for engineering reports. Most commonly recognized.</td></tr>
  <tr><td>HOT</td><td>Black→red→orange→yellow→white. Thermal imaging style. Defect intensity = heat.</td></tr>
  <tr><td>VIRIDIS</td><td>Perceptually uniform — equal steps in brightness = equal steps in value. Best for quantitative measurement and publications. Also readable by colour-blind viewers.</td></tr>
  <tr><td>INFERNO</td><td>Perceptually uniform, high contrast in dark range. Good for dark defects on surfaces.</td></tr>
  <tr><td>PLASMA</td><td>Perceptually uniform, vivid. Good for continuous surface topology maps.</td></tr>
  <tr><td>BONE/OCEAN/PINK</td><td>Low contrast subtle toning. Use when only slight colour guidance is needed.</td></tr>
</table>
<p>Use <strong>Input Min / Max</strong> to zoom the colour range onto a specific intensity band — effectively a colour-space zoom into the region of interest.</p>

<hr>
<h2>26. Clipping Highlight</h2>
<div class="math">
pixel RED  if gray ≥ over_threshold   (overexposed — saturated, no data)
pixel BLUE if gray ≤ under_threshold  (underexposed — crushed, no data)
</div>
<p><strong>Overexposed pixels (RED):</strong> the sensor is saturated — all information in that region
is destroyed. A defect inside a bright specular reflection cannot be detected.
Adjust lighting to eliminate saturation before inspection.
<strong>Underexposed pixels (BLUE):</strong> deep shadow regions have no detail.
A crack hidden in a crushed shadow will not be detected.
Use this filter to verify camera exposure before running your inspection pipeline.</p>

<hr>
<h2>27. Channel Split</h2>
<p class="ref">Bayer mosaic pattern — camera sensor architecture</p>
<div class="math">
Red   channel = I[:,:,0]
Green channel = I[:,:,1]   ← 2× more Bayer pixels than R or B
Blue  channel = I[:,:,2]
Gray          = 0.299R + 0.587G + 0.114B  (ITU-R BT.601)
</div>
<p>Isolates a single colour channel for analysis.
<strong>Green channel has the highest SNR</strong> on Bayer cameras because there are
2× more green photosites per pixel than red or blue.
Red channel: shows red-absorbing coatings, fluorescent markers, blood contamination.
Blue channel: UV-fluorescent defects appear bright in blue.</p>

<hr>
<h2>28. Channel Mixer</h2>
<div class="math">
Output_R = rr·R + rg·G + rb·B
Output_G = gr·R + gg·G + gb·B
Output_B = br·R + bg·G + bb·B
(each clipped to 0–255)
</div>
<p>Full 3×3 colour matrix — custom weighted combination of RGB channels.
Use to emphasize a specific spectral signature that appears in only one channel.
Example: a red-tinted surface contamination appears only in R → set rg=0.5 (add R into G output)
to make the contamination visible in an output channel that would otherwise miss it.
Identity matrix (diagonal = 1.0, rest = 0.0) = no change.</p>

<hr>
<h2>29. Invert</h2>
<div class="math">
Output(x,y) = 255 − Input(x,y)    (uint8)
Output(x,y) = 65535 − Input(x,y)  (uint16)
</div>
<p>Flips bright↔dark. Use when defects are dark on a bright background and you want to
apply Top-Hat (which targets bright features) — invert first so defects become bright,
then run Top-Hat, then optionally invert back.</p>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Fusion": _page("Illumination Fusion — All Operations", """
<p>Illumination Fusion combines multiple images of the same part taken under different lighting angles.
A defect hidden under one angle often appears clearly under another.
Fusion reveals everything — in one image.</p>

<h2>How to use</h2>
<ol>
  <li>Switch to <strong>Fusion</strong> tab</li>
  <li>Load images: double-click in folder list, or click <strong>＋ Add</strong></li>
  <li>Select the fusion operation</li>
  <li>Click <strong>▶ Compose</strong></li>
</ol>

<hr>
<h2>Operation 1 — Max Fusion</h2>
<div class="math">Output(x,y) = max( I₁(x,y), I₂(x,y), ..., Iₙ(x,y) )</div>
<p>Takes the brightest value at each pixel across all images.
A bright defect that is visible in any one light angle will survive into the output.
<strong>Best for: bright protrusions, high-reflectance defects, specular hits.</strong>
Works with any number of images, no angle input needed.</p>

<hr>
<h2>Operation 2 — Min Fusion</h2>
<div class="math">Output(x,y) = min( I₁(x,y), I₂(x,y), ..., Iₙ(x,y) )</div>
<p>Takes the darkest value at each pixel across all images.
A dark defect that is visible in any one light angle will survive.
<strong>Best for: pits, voids, dark contamination.</strong></p>

<hr>
<h2>Operation 3 — Average Fusion</h2>
<div class="math">Output(x,y) = (1/N) · Σ Iᵢ(x,y)</div>
<p>Simple mean across all images.
Reduces random per-image noise (by factor √N). Balances exposure across angles.
<strong>Best for: noise reduction, balanced overview when no specific defect type is targeted.</strong>
Does not enhance defects — visible features in all images appear average.
Defects visible in only one image are diluted by N−1 clean images.</p>

<hr>
<h2>Operation 4 — Range Fusion (Max − Min)</h2>
<div class="math">Output(x,y) = max(Iᵢ) − min(Iᵢ)  across all i at each pixel (x,y)</div>
<p>Measures how much each pixel <em>changes</em> between different lighting angles.
A perfectly flat uniform surface reflects the same way regardless of light direction →
I₁≈I₂≈...≈Iₙ → max≈min → output ≈ 0 (black, no defect).
A scratch or pit scatters light differently at different angles →
large variation between images → large Max−Min → bright white.
<strong>Best overall automatic defect detector — no angle calibration needed, works with
any 3+ images, requires no A/B selection.</strong>
Simply load all your lighting images and compose.</p>

<hr>
<h2>Operation 5 — Superposition (Sum)</h2>
<div class="math">Output(x,y) = normalize( Σ Iᵢ(x,y) )
A defect visible in K lights accumulates K times stronger than a defect in 1 light.
</div>
<p>Any pixel bright in ANY image accumulates — the more lights it appears under, the brighter.
Good surface pixels are bright in all lights → appear bright in output.
<strong>Accumulates and reinforces real defects.</strong> Works with 2, 3, 4+ images.
After normalization, the relative contrast between good surface and defects is preserved.</p>

<hr>
<h2>Operation 6 — Multiply (AND logic)</h2>
<div class="math">Output(x,y) = normalize( I₁(x,y)/255 × I₂(x,y)/255 × ... × Iₙ(x,y)/255 )
Any image where pixel is dark → product → zero for that pixel.
</div>
<p>Only pixels bright in <em>every</em> image survive.
<strong>Extremely selective — confirms real defects that appear under every lighting angle.</strong>
A noise spike in one image (dark in all others) = killed.
A true defect visible in all lights = survives.
Use to eliminate false positives when you have 4+ well-calibrated lights.</p>

<hr>
<h2>Operation 7 — Difference |A − B|</h2>
<div class="math">Output(x,y) = |I_A(x,y) − I_B(x,y)|
Flat uniform surface: I_A ≈ I_B → |A−B| ≈ 0 (black)
Defect: reacts differently to each light → |A−B| is large (bright)
</div>
<p>Isolates what one lighting angle reveals vs another.
<strong>Best for directional defects:</strong> a scratch running left-right is invisible under
left-side lighting but bright under right-side lighting → large |A−B|.
Select A = left light image, B = right light image.
You must choose two images manually.</p>

<hr>
<h2>Operation 8 — RGB Composite</h2>
<div class="math">Output_R = weighted average of images assigned to R channel
Output_G = weighted average of images assigned to G channel
Output_B = weighted average of images assigned to B channel
→ assembled into RGB colour image
</div>
<p>Assign each image to a colour channel (R, G, or B).
The output is a colour image where each channel comes from a different lighting angle.
<strong>Defects become colour-visible:</strong> a scratch that appears only under one angle
will be bright in that colour channel and create a colour cast.
Works with exactly 3 images (one per channel). Weights allow mixing images into one channel.</p>

<hr>
<h2>Operation 9 — Photometric Stereo → Gradient Map</h2>
<p class="ref">Woodham (1980) — Photometric Stereo, Optical Engineering. Requires calibrated light directions.</p>
<div class="math">
Lambertian reflectance model:
  I_k = ρ · n · l_k     (dot product of surface normal n and light direction l_k)

Light direction vector:
  l_k = ( cos(el)·cos(az),  cos(el)·sin(az),  sin(el) )
  where el=elevation, az=azimuth

Stack N images: I = (I₁..Iₙ)ᵀ,  L = (l₁..lₙ)ᵀ
Solve: g = L† · I     (pseudo-inverse least squares, one shot for all pixels)
  g = ρ·n   →   ρ = ‖g‖  (albedo),   n = g/ρ  (unit normal)

Gradient magnitude output = ‖ (−nx/nz, −ny/nz) ‖
= slope of surface in X and Y, combined magnitude
</div>
<p>Photometric Stereo computes the surface normal at every pixel from multiple images
taken under different known light directions. The gradient magnitude output shows
where the surface slope is steep — scratches, cracks, steps, and edges appear bright.
<strong>Far more sensitive to surface relief than any single-image edge detector</strong>
because it uses actual geometry, not just brightness.
Minimum 3 images, recommend 4 images at 0°/90°/180°/270° azimuth, 30–45° elevation.</p>

<hr>
<h2>Operation 10 — Photometric Stereo → Height Map + 3D Viewer</h2>
<p class="ref">Woodham (1980) normals + Frankot &amp; Chellappa (1988) FFT integration, IEEE TPAMI</p>
<div class="math">
From Photometric Stereo: surface gradients  gx = −nx/nz,  gy = −ny/nz

Frankot-Chellappa FFT integration (enforces integrability):
  Z(u,v) = −i·( fx·FFT(gx) + fy·FFT(gy) ) / ( fx² + fy² + ε )
  height map z(x,y) = Re( IFFT( Z(u,v) ) )

  fx = frequency coordinates,  ε = 1e-8  (avoid division by zero)
  DC term Z(0,0) = 0  (height is relative, not absolute)
</div>
<p>The Frankot-Chellappa algorithm integrates the surface gradients into a height field.
Integration in the frequency domain enforces that the height field is consistent
(the curl of the gradient field is zero — a physical constraint of real surfaces).
<strong>Result: a real, physically-correct 3D surface map.</strong>
The app automatically routes this to the 3D Viewer and shows a green badge.
Bright = surface is raised. Dark = surface is recessed.</p>

<hr>
<h2>Operation 11 — RTI Relight (Polynomial Texture Mapping)</h2>
<p class="ref">Malzbender, Gelb &amp; Wolters (2001) — PTM, ACM SIGGRAPH. Standard in cultural heritage digitization.</p>
<div class="math">
Per-pixel polynomial model (biquadratic):
  I(u,v) = c₀u² + c₁v² + c₂uv + c₃u + c₄v + c₅

u,v = normalized XY components of light direction
c₀..c₅ = 6 coefficients fit per pixel from N≥6 images

Fit (one time): A = [u²,v²,uv,u,v,1] for each image → solve A·C = I (least-squares)
Relight (interactive): evaluate polynomial at any (u,v) → new image in milliseconds
</div>
<p>RTI fits a polynomial function of light direction per pixel, capturing how each surface
point reflects under any lighting angle. After fitting, you can relight interactively by
dragging azimuth/elevation sliders — the result appears in real time.
<strong>Use to find the lighting angle that makes a specific hidden defect most visible.</strong>
Requires 6+ images with calibrated angles. Fit takes a few seconds. Relighting is instant.</p>
"""),


# ══════════════════════════════════════════════════════════════════════════
"3D Viewer": _page("3D Surface Viewer", """
<p>The 3D viewer renders the image as a surface mesh. Two completely different modes.</p>

<h2>Visual mode — brightness → height</h2>
<div class="math">Z(x,y) = pixel brightness at (x,y),  scaled by Z-scale slider</div>
<p>Any image loaded normally is shown in this mode.
Pixel brightness becomes height: bright = tall, dark = flat.
<strong>This geometry is not physically real</strong> — a bright scratch appears raised even
though it is a groove. Use for visual overview only, not for dimensional analysis.</p>
<div class="warn">The info bar reads: <code>Visual — brightness → height</code></div>

<h2>Real Geometry mode — Photometric Stereo</h2>
<p>When you compose <strong>Photometric Stereo → Height Map</strong> in Fusion,
the app automatically switches to the 3D tab and loads real surface geometry.</p>
<ul>
  <li>Bright = surface is truly raised above baseline</li>
  <li>Dark = surface is truly recessed</li>
  <li>Mesh shape = actual surface topology from Frankot-Chellappa integration</li>
</ul>
<p>A green badge is shown: <span class="badge green">⚡ Real Surface Geometry — Photometric Stereo</span></p>
<p>Info bar reads: <code>⚡ Real Surface — Photometric Stereo</code></p>

<h2>Mesh controls</h2>
<table>
  <tr><th>Control</th><th>Effect</th></tr>
  <tr><td>Resolution</td><td>Mesh density — lower = faster render, higher = more geometric detail</td></tr>
  <tr><td>Z Scale</td><td>Amplify height values — increase for subtle topology (auto-set to 15 for real geometry)</td></tr>
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

<div class="tip">To get real 3D geometry: Fusion → load 3+ images → set light angles → Photometric Stereo → Height Map → Compose. 3D tab opens automatically.</div>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Tools Menu": _page("Tools Menu — Reference", """
<h2>Global tools</h2>
<table>
  <tr><th>Item</th><th>Shortcut</th><th>What it does</th></tr>
  <tr><td>Illumination Fusion</td><td><kbd>Ctrl+F</kbd></td><td>Switch to Fusion tab</td></tr>
  <tr><td>3D Surface View</td><td><kbd>Ctrl+3</kbd></td><td>Switch to 3D tab</td></tr>
  <tr><td>Image Comparison</td><td><kbd>Ctrl+M</kbd></td><td>Open comparison window</td></tr>
</table>

<h2>Inspect tools (checkmark shows active tool)</h2>
<table>
  <tr><th>Item</th><th>Key</th><th>Effect</th></tr>
  <tr><td>↖ Navigate</td><td><kbd>Esc</kbd></td><td>Pan/zoom — toolbar hidden</td></tr>
  <tr><td>⬛ ROI</td><td><kbd>R</kbd></td><td>Draw rectangle → region stats</td></tr>
  <tr><td>📈 Profile</td><td><kbd>P</kbd></td><td>Draw line → intensity chart</td></tr>
  <tr><td>📍 Annotate</td><td><kbd>A</kbd></td><td>Click to place defect marker</td></tr>
  <tr><td>↔ Measure</td><td><kbd>M</kbd></td><td>Drag to measure distance</td></tr>
  <tr><td>⬡ Mask</td><td><kbd>K</kbd></td><td>Draw inspection polygon</td></tr>
</table>

<h2>Utility tools</h2>
<table>
  <tr><th>Item</th><th>Effect</th></tr>
  <tr><td>Set Scale Calibration…</td><td>Enter mm/pixel ratio for the Measure tool</td></tr>
  <tr><td>Clear Inspect Overlays</td><td>Remove ROI, profile, measure, annotations</td></tr>
  <tr><td>Clear Mask</td><td>Remove the inspection mask polygon</td></tr>
</table>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Keyboard Shortcuts": _page("Keyboard Shortcuts", """
<h2>Inspect tools</h2>
<table>
  <tr><th>Key</th><th>Tool</th></tr>
  <tr><td><kbd>Esc</kbd></td><td>Navigate (pan / zoom)</td></tr>
  <tr><td><kbd>R</kbd></td><td>ROI — draw region of interest</td></tr>
  <tr><td><kbd>P</kbd></td><td>Profile — draw intensity line</td></tr>
  <tr><td><kbd>A</kbd></td><td>Annotate — place defect marker</td></tr>
  <tr><td><kbd>M</kbd></td><td>Measure — drag distance</td></tr>
  <tr><td><kbd>K</kbd></td><td>Mask — draw inspection polygon</td></tr>
</table>

<h2>Navigation</h2>
<table>
  <tr><th>Key / Action</th><th>Result</th></tr>
  <tr><td><kbd>←</kbd> / <kbd>→</kbd></td><td>Previous / next image in folder</td></tr>
  <tr><td>Scroll wheel</td><td>Zoom in / out</td></tr>
  <tr><td>Left drag</td><td>Pan the image</td></tr>
</table>

<h2>Global</h2>
<table>
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><kbd>Ctrl+O</kbd></td><td>Open image file</td></tr>
  <tr><td><kbd>Ctrl+F</kbd></td><td>Switch to Fusion tab</td></tr>
  <tr><td><kbd>Ctrl+3</kbd></td><td>Switch to 3D tab</td></tr>
  <tr><td><kbd>Ctrl+M</kbd></td><td>Open Image Comparison</td></tr>
  <tr><td><kbd>F1</kbd></td><td>Open this User Guide</td></tr>
</table>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Inspector Panel": _page("Inspector Panel — All Metrics Explained", """
<p>The Inspector dock shows objective numerical measurements of the image.
No PASS/FAIL verdicts — the expert reads the numbers and decides.</p>

<hr>
<h2>Focus &amp; Sharpness</h2>
<p>Three independent sharpness algorithms run on every image and are fused into one score.</p>

<h3>Score (0–1000+)</h3>
<p>Composite sharpness index. Higher = sharper. No threshold verdict — read the raw number.</p>

<h3>Lap (Laplacian Variance)</h3>
<p class="ref">Brenner et al. (1976) — first published focus measure for autofocus systems</p>
<div class="math">
Lap = Var( ∇²I )  =  Var( I[x] − 2·I[x+1] + I[x+2] )

∇²I = second derivative (Laplacian) of the image
Var = variance across all pixels in the cell
</div>
<p>A perfectly sharp image has strong, high-contrast edges → large second derivatives → high Lap variance.
A blurry image has smooth gradients → small second derivatives → low Lap variance.
<strong>Best in low-noise conditions. Very fast.</strong></p>

<h3>Ten (Tenengrad)</h3>
<p class="ref">Tenenbaum (1970) — autofocus criterion; revalidated IEEE Trans. Circuits Syst. (2013)</p>
<div class="math">
Ten = Σ max( Gx(x,y)² + Gy(x,y)² − threshold,  0 )

Gx, Gy = Sobel gradient (first derivative) in X and Y directions
threshold = noise floor — only gradients above this are counted
</div>
<p>Sums squared gradient magnitude across the image cell.
The noise threshold makes Tenengrad far more robust than Lap on real camera images with sensor noise.
<strong>Best all-round metric — recommended for production.</strong></p>

<h3>Bren (Brenner)</h3>
<p class="ref">Brenner et al. (1976) — fastest focus measure, still used in embedded systems</p>
<div class="math">
Bren = Σ ( I[x+2,y] − I[x,y] )²    (sum of squared 2-pixel differences)
</div>
<p>Measures local contrast by comparing pixels 2 steps apart.
Fastest computation of the three. Good correlation with perceived sharpness.
Used as a cross-check — if Bren and Ten agree, confidence is HIGH.</p>

<h3>Confidence (HIGH / MEDIUM / LOW)</h3>
<ul>
  <li><strong>HIGH</strong> — all three metrics agree on the same sharpness category</li>
  <li><strong>MEDIUM</strong> — two agree, one disagrees by one level</li>
  <li><strong>LOW</strong> — metrics disagree significantly (unusual texture, high noise, or damaged sensor)</li>
</ul>

<h3>Scoring mode</h3>
<table>
  <tr><th>Mode</th><th>Meaning</th></tr>
  <tr><td>RELATIVE</td><td>Score relative to the sharpest cell in the same image. Cannot confirm absolute sharpness.</td></tr>
  <tr><td>AUTO-REF</td><td>Score relative to the sharpest image seen this session.</td></tr>
  <tr><td>LOCKED REF ✓</td><td>Score relative to a locked known-good reference. Most reliable — use in production.</td></tr>
</table>

<hr>
<h2>Image Quality Metrics</h2>
<p>All metrics are objective measurements. The expert interprets them — no automatic verdict.</p>

<h3>Brightness  (mean pixel value, 0–255)</h3>
<div class="math">
Brightness = mean( gray(x,y) )  over all valid pixels

gray = 0.299·R + 0.587·G + 0.114·B   (ITU-R BT.601 luminance)
</div>
<table>
  <tr><th>Value</th><th>Label</th><th>Meaning</th></tr>
  <tr><td>&lt; 30 / 255</td><td>DARK</td><td>Image too dark — defects in shadow regions invisible. Fix lighting or exposure.</td></tr>
  <tr><td>30–80 / 255</td><td>LOW</td><td>Underlit — may miss low-contrast defects. Increase illumination.</td></tr>
  <tr><td>&gt; 80 / 255</td><td>OK</td><td>Sufficient brightness for inspection.</td></tr>
</table>

<h3>Exposure</h3>
<div class="math">
Overexposed  = pixels where gray ≥ 250  (saturated — no information)
Underexposed = pixels where gray ≤ 5    (crushed — no information)
OK = both percentages &lt; 1% of total pixels
</div>
<p>Clipped pixels contain no information — defects in those regions cannot be detected.
If overexposed: reduce light intensity or camera gain.
If underexposed: increase light or exposure time.</p>

<h3>Contrast  (RMS Contrast %)</h3>
<p class="ref">Michelson (1927) — contrast definition; RMS contrast standard in machine vision</p>
<div class="math">
RMS Contrast = std(gray) / 255 × 100   (%)

Also computed: Michelson = (max − min) / (max + min)
Dynamic Range = log₂(max / min)  in stops
</div>
<p>RMS contrast measures the spread of intensity values.
Low contrast (&lt;5%) means the image is flat — defects have similar brightness to background, hard to detect.
Increase lighting angle or use CLAHE filter to improve contrast before inspection.</p>

<h3>Noise</h3>
<p class="ref">Laplacian noise estimator — Immerkær (1996), fast and accurate without requiring flat regions</p>
<div class="math">
Kernel K = [ 1  −2   1 ]
           [−2   4  −2 ]
           [ 1  −2   1 ]

filtered = K * image
Noise = sqrt( mean(filtered²) ) / 2
</div>
<p>The Laplacian kernel removes signal and leaves only noise.
Low noise (&lt;5) = clean sensor, good for fine defect detection.
High noise (&gt;20) = noisy sensor or high ISO — apply NL-Means or Bilateral filter first.</p>

<h3>SNR  (Signal-to-Noise Ratio, dB)</h3>
<div class="math">
SNR = 20 · log₁₀( mean_brightness / noise_level )   dB

Higher dB = cleaner image.
Every +6 dB = noise halved relative to signal.
</div>
<table>
  <tr><th>SNR</th><th>Quality</th></tr>
  <tr><td>&gt; 40 dB</td><td>Excellent — fine defect detection reliable</td></tr>
  <tr><td>30–40 dB</td><td>Good — suitable for most inspection tasks</td></tr>
  <tr><td>20–30 dB</td><td>Marginal — use denoising filter before detection</td></tr>
  <tr><td>&lt; 20 dB</td><td>Poor — noise dominates, false detections likely</td></tr>
</table>

<hr>
<h2>Other Inspector sections</h2>
<table>
  <tr><th>Section</th><th>What it shows</th><th>When</th></tr>
  <tr><td>Histogram</td><td>Intensity distribution — full image or ROI</td><td>Always</td></tr>
  <tr><td>Pixel</td><td>R, G, B values under mouse cursor</td><td>Always</td></tr>
  <tr><td>ROI Stats</td><td>Mean, min, max, std-dev for drawn rectangle</td><td>After drawing ROI</td></tr>
  <tr><td>Line Profile</td><td>Brightness chart along drawn line</td><td>After drawing profile</td></tr>
  <tr><td>Measurement</td><td>Distance in px and mm, angle, coordinates</td><td>After measuring</td></tr>
  <tr><td>Annotations</td><td>All markers with labels and coordinates</td><td>After annotating</td></tr>
</table>
"""),


# ══════════════════════════════════════════════════════════════════════════
"Status Bar": _page("Status Bar", """
<p>The thin strip at the very bottom of the window.</p>
<pre>[1]  19RGB.bmp   11%  (1:8.8)   ● F:14  Q:98   Tool: MASK — draw polygon</pre>
<table>
  <tr><th>Field</th><th>Meaning</th></tr>
  <tr><td><code>[1]</code></td><td>Active viewer cell (1–4 in multi-view layout)</td></tr>
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
        self.resize(1080, 720)
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
        self._search.setPlaceholderText("Search sections…")
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
        self._nav.setFixedWidth(210)
        self._nav.setSpacing(1)
        for name in SECTIONS:
            item = QListWidgetItem(f"  {name}")
            item.setSizeHint(QSize(210, 34))
            self._nav.addItem(item)
        self._nav.currentRowChanged.connect(self._on_nav)
        splitter.addWidget(self._nav)

        # Content browser
        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(False)
        self._browser.setFont(QFont("Segoe UI", 11))
        splitter.addWidget(self._browser)
        splitter.setSizes([210, 870])

        self._nav.setCurrentRow(0)

    def _on_nav(self, row: int):
        if row < 0:
            return
        name = list(SECTIONS.keys())[row]
        self._browser.setHtml(SECTIONS[name])
        self._browser.verticalScrollBar().setValue(0)

    def _on_search(self, text: str):
        text = text.strip().lower()
        for i in range(self._nav.count()):
            item = self._nav.item(i)
            name = list(SECTIONS.keys())[i].lower()
            item.setHidden(bool(text) and text not in name)
