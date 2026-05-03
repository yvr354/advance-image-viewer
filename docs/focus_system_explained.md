# VyuhaAI Focus System — How It Works

This document explains the focus scoring system used in VyuhaAI Image Viewer.
Written so anyone — engineer, manager, or customer — can understand the design choices.

---

## The Problem We Solved

Most focus tools do this:

> "Find the sharpest cell in this image. Call that 100%. Score every other cell relative to it."

This is **wrong for industrial inspection**. Here is why:

If you give the tool a completely blurry image, the sharpest cell in that image becomes 100%.
The tool shows a green grid — all passing — even though the image is garbage.
An expert looks at this and immediately loses trust in the tool. **One mistake, tool is gone.**

We needed scoring that tells the truth: *is this image actually sharp enough for defect detection?*

---

## The Science Behind It

We based the focus metrics on peer-reviewed research:

**Pertuz, S. et al. (2013). "Analysis of focus measure operators for shape-from-focus."
Pattern Recognition, 46(5), 1415–1432.**

This paper benchmarked 36 different focus measurement algorithms on real images.
Three came out on top:

| Metric | Formula | Why We Use It |
|--------|---------|---------------|
| **Laplacian Variance** | Variance of ∇²I | Best at low noise. Most used in industrial cameras. #2 overall in Pertuz. |
| **Tenengrad** | Σ(Gx² + Gy²) | Best noise robustness. #1 overall in Pertuz. |
| **Brenner Gradient** | Σ(I[x+2]−I[x])² | Fastest. Good cross-check on high-SNR cameras. |

We use **Laplacian + Tenengrad fused** (55% / 45%) for the main score.
Brenner is shown as a third verification number.

Why two metrics? Because Laplacian and Tenengrad have different sensitivity profiles.
If they **agree** on the verdict → HIGH confidence.
If they **disagree** by one category → MEDIUM confidence.
If they disagree by two or more → LOW confidence. Expert must verify manually.

---

## Three Scoring Modes

### 1. RELATIVE mode (⚠ warning shown)

No reference image. Score = this cell ÷ best cell in this same image × 100%.

- Good for: seeing where the lens is soft vs sharp within one image (field curvature, tilt)
- Bad for: pass/fail decisions. A blurry image scores 100% in its best cell.
- **Do not use this for production inspection.**

### 2. AUTO-REF mode

The app automatically tracks the sharpest image you have opened this session.
That image becomes the reference. Score = this cell ÷ same cell in session-best × 100%.

- Better than RELATIVE — at least you're comparing images to each other
- Still session-dependent — each time you restart, reference resets
- Only a non-blurry image can become AUTO-REF (minimum quality threshold enforced)

### 3. LOCKED REF mode (✓ shown — production grade)

You deliberately capture or select your best known-good image and click **Lock Ref**.
The app saves this reference to disk. It survives restarts.

Score = this cell ÷ same cell in locked reference × 100%.

- 100% means "as sharp as your calibrated golden sample"
- 60% means "only 60% as sharp as your reference — check lens or lighting"
- This is how professional machine vision tools work (Cognex, Keyence, Halcon)
- **This is the mode to use for production inspection**

---

## The Grid

The image is divided into an 8×8 grid (64 cells).
Each cell is scored independently.

Color coding:
- **GREEN** — ≥ 72% — sharp, good for defect detection
- **AMBER** — 38–72% — soft, marginal, review lens settings
- **RED** — < 38% — blurry, reject this image, do not use for inspection

Why per-cell? Because a lens may be sharp in the center but soft at the edges.
This is called **field curvature** — a real optical problem common in industrial lenses.
The grid catches it immediately. The Focus Assist panel also shows tilt direction
(e.g. "sharper on LEFT", "sharper at BOTTOM").

---

## Confidence Levels

| Level | When | What It Means |
|-------|------|---------------|
| **HIGH** | LOCKED REF + Laplacian and Tenengrad agree on verdict | Fully trustworthy. Use for automated pass/fail. |
| **MEDIUM** | AUTO-REF, or metrics are one category apart | Reliable for manual review. Not for unattended automation. |
| **LOW** | RELATIVE mode only | Spatial variation only. Cannot confirm absolute sharpness. |

---

## Raw Numbers — Always Visible

The inspector always shows three raw numbers:

```
Lap: 1,840  |  Ten: 45,230  |  Bren: 12.1
```

These are the unscaled metric values directly from the algorithms.
An expert can verify or challenge any verdict by looking at these directly.
We never hide the math.

---

## How to Use It — Step by Step

### First time setup (one-time, 5 minutes)

1. Mount your camera and part at the correct working distance
2. Capture the sharpest image you can — turn the lens focus ring until the image is crisp
3. Open that image in the app
4. Click **Lock Ref** in the Focus Assist panel (top right when in Focus mode)
5. The app saves this reference. Done. You never need to do this again unless the setup changes.

### Every inspection session

1. Open the image (or folder of images)
2. Switch to **Focus mode** (button at top)
3. The grid shows each cell as % of your locked reference
4. GREEN cells → sharp, proceed to defect inspection
5. AMBER/RED cells → fix lens focus, lighting, or vibration before inspecting
6. The Focus Assist panel shows verdict, confidence, and specific action to take

### When the reference image is shown

If you open the exact same image that was locked as reference, the app shows:

> ⚠ THIS IS THE REFERENCE IMAGE — comparing to itself — 100% is meaningless

This is intentional. The 100% grid is correct math but useless information.
The absolute Score and Verdict at the top are what matter for that image.

---

## What the Verdicts Mean

| Verdict | Absolute Score | Meaning |
|---------|---------------|---------|
| **PERFECT** | ≥ 700 | Excellent. Lock this as reference if you don't have one. |
| **GOOD** | 400–699 | Usable for all defect types including fine scratches. |
| **SOFT** | 200–399 | Marginal. OK for large defects. Risk of missing hairline cracks. |
| **BLURRY** | < 200 | Reject. Do not use for inspection. Fix focus before proceeding. |

The absolute score uses Laplacian Variance ÷ 5, calibrated to a 0–1000 scale.
These thresholds are configurable in the app config file if your camera requires adjustment.

---

## Why This Design

The guiding rule during development:

> *"This tool will be used by experts who know everything. Any mistake they can find immediately. One mistake — the tool is gone."*

Every design decision follows from that:
- Never normalize away a bad image to look good
- Always show raw numbers — expert can verify
- Always show confidence — expert knows when to trust vs verify manually
- Always show scoring mode — expert knows what kind of comparison is being made
- Self-reference warning — expert is not misled by trivial 100% scores
- Blurry images blocked from becoming reference — garbage in, garbage out prevented

---
---

# Image Comparison System — How It Works

This section explains the A/B comparison and defect detection system.
Same audience: engineer, manager, or customer.

---

## The Problem We Solve

A human inspector comparing two parts:
- Gets tired after 200 parts
- Misses defects smaller than 2mm consistently
- Is inconsistent — passes a part in the morning, rejects the same part in the afternoon
- Cannot produce a traceable record of what was inspected

Our system finds differences **automatically, every time, in under 2 seconds**, with a full record.

---

## Why Naive Pixel Comparison Fails

The obvious approach — subtract image B from image A — produces thousands of false defects on a perfect part. Three reasons:

**1. Lighting changes between shots.**
The LED dims 3% as it warms up. The camera sees the part as slightly darker.
A naive diff shows the entire image as "different" — even though nothing is wrong.

**2. The part shifts slightly.**
Even a 1mm shift in the fixture makes every edge in the image register as a difference.
A screw hole that moved 3 pixels looks like a 3-pixel-wide defect running around its full circumference.

**3. Camera sensor noise.**
Every camera produces random ±2–3 pixel value noise on every shot.
Without noise removal, a clean part would show tens of thousands of 1-pixel "defects".

Professional tools (Cognex PatMax, Keyence XG-X, Halcon) solve all three before running any comparison. We do the same.

---

## The Pipeline — Six Steps

### Step 1 — Brightness Normalization

Match the overall brightness of image B to image A before comparing.

```
scale = mean(A) / mean(B)
B_corrected = clip(B × scale, 0, 255)
```

Scale is clamped between 0.5× and 2.0× to prevent correction of images that are genuinely very different (e.g. completely different exposure).

> **Result:** A 10% lighting change between shots no longer produces false defects.

---

### Step 2 — Alignment (ORB Feature Matching)

Automatically warp image B to match image A's position and orientation.

**How it works:**
1. Detect up to 500 distinctive keypoints in each image (corners, edges, texture boundaries)
2. Match keypoints between A and B using Hamming distance
3. Take the best 50 matches, discard outliers using RANSAC
4. Calculate a homography matrix (2D perspective transform)
5. Warp B through that transform so keypoints line up with A

This is the same family of algorithms used in panorama stitching, augmented reality, and satellite image registration.

**RANSAC** (Random Sample Consensus) is the key quality step — it automatically rejects wrong matches before computing the transform, making alignment robust even when some keypoints are misleading.

> **Result:** A part that shifted 2mm in the fixture produces zero false edge defects.

---

### Step 3 — Absolute Difference Map

Subtract the aligned, normalized images pixel by pixel.

```
diff = |A − B|   (per pixel, grayscale)
```

Each pixel in the diff image represents **how different that spot is**, from 0 (identical) to 255 (completely different).

**Display:** The diff is passed through the HOT colormap (black → dark red → orange → yellow → white). Small differences appear as scattered dark red. Large differences appear as bright orange or white.

**Auto-normalize:** If the maximum difference in the entire image is less than 32 (very similar images, e.g. BMP vs JPEG of the same photo), the diff is stretched to fill the full 0–255 range so differences are visible. The header shows `⚡ auto-norm` when this is active. This only affects the **display** — not the detection threshold.

---

### Step 4 — Noise Removal

Two-stage morphological filtering removes camera noise and compression artifacts.

**Stage A — Gaussian Blur (5×5, σ=1.0)**
Smooths single-pixel random noise before thresholding.
A real defect spans many pixels and survives smoothing. Random noise does not.

**Stage B — Threshold**
Every pixel where diff > threshold is marked as "different". Default threshold = 30.
Below threshold = noise. Above threshold = potential defect.

**Stage C — Morphological Opening (5×5 ellipse kernel)**
Erosion followed by dilation. Removes isolated blobs smaller than the kernel.
Any blob that cannot survive erosion was too small to be a real defect.

**Stage D — Morphological Closing**
Dilation followed by erosion. Closes small gaps inside larger blobs.
A real defect is a solid region. Closing prevents a scratch from being counted as 20 separate dots.

```
opened = erode(binary, kernel)   → removes isolated noise dots
closed = dilate(opened, kernel)  → fills gaps within real defects
```

> **Result:** Camera sensor noise, JPEG block artifacts, and fine surface texture variation are eliminated before detection runs.

---

### Step 5 — Defect Detection (Connected Components)

The filtered binary image is scanned for connected regions of "different" pixels.

`cv2.connectedComponentsWithStats(closed, connectivity=8)`

This returns every separate blob with:
- Bounding box (x, y, width, height)
- Centroid (exact center point)
- Area (pixel count)

Blobs are then filtered:
- **Below min area** (default 25 px²) → discarded as residual noise
- **Above 200,000 px²** → discarded as image-wide artifact (e.g. wrong image loaded)
- Everything in between → classified as a defect

**Severity classification:**

| Severity | Area | Real-world meaning |
|---|---|---|
| COSMETIC | < 25 px² | Surface mark. No structural or functional impact. Document and accept. |
| FUNCTIONAL | 25–200 px² | May affect fit, seal, or electrical contact. Engineer review required. |
| CRITICAL | > 200 px² | Structural defect. Reject. Do not ship. |

These thresholds match published industry guidelines for machine vision inspection (ISO 13379, IPC-A-610 visual inspection standards).

---

### Step 6 — PASS / FAIL Verdict

```
FAIL  if any CRITICAL defect found
FAIL  if any FUNCTIONAL defect found
PASS  if defects are COSMETIC only, or no defects
```

**Optional SSIM verification** (when scikit-image is installed):

SSIM (Structural Similarity Index) is an IEEE-standard image quality metric developed at the University of Texas (Wang et al., 2004). It measures luminance, contrast, and structure similarity simultaneously. Range: 0 (completely different) to 1.0 (identical).

```
SSIM ≥ 0.92  →  images are structurally identical  →  PASS confirmed
SSIM < 0.92  →  structural difference detected      →  FAIL
```

SSIM is a second opinion. It does not override the blob detection verdict — both are shown.

**PSNR** (Peak Signal-to-Noise Ratio) is also reported in decibels. Above 40 dB = visually lossless. Provided for completeness and traceability.

---

## What the Operator Sees

```
✓ PASS   |   SSIM: 0.998   |   PSNR: 52.3 dB   |   Diff: 0.0%   |   Defects: none
```

```
✗ FAIL   |   SSIM: 0.871   |   PSNR: 31.1 dB   |   Diff: 3.2%   |   Defects: 2 critical, 1 functional
```

Each defect appears as a colored bounding box on the diff map, and as a card in the defect list with exact pixel coordinates, area, and severity. The operator can click **→ Jump to defect** to pan all three viewers (A, B, Diff) to that exact location at 4× zoom.

---

## Display Modes

| Mode | What it shows | When to use |
|---|---|---|
| **Diff Map** | Absolute difference, HOT colormap | Primary mode. Always start here. |
| **Signed ±128** | Directional difference, COOLWARM colormap. Gray = same. Blue = A darker. Red = A brighter. | Diagnosing systematic shifts — exposure change, coating thickness change. |
| **Blend 50/50** | A and B overlaid at 50% opacity each | Visual alignment check before trusting the diff. |
| **Flicker** | Alternates A and B every 600ms | Human eye catches differences that computers miss in complex textures. |

---

## Why This Design

The same guiding rule applies here as in the focus system:

> *"This tool will be used by experts who know everything. Any mistake they can find immediately. One mistake — the tool is gone."*

Every design decision follows from that:
- Alignment and normalization happen automatically — expert does not need to remember to do it
- `⇄ aligned` shown in header — expert knows alignment ran and can verify
- `⚡ auto-norm` shown in header — expert knows the display was stretched and why
- Raw metrics always visible (SSIM, PSNR, Diff %, Max Δ) — expert can challenge any verdict
- Defect coordinates are exact pixel values — expert can verify on the original image
- Both images shown side by side at synchronized zoom/pan — expert can visually confirm any flagged defect

---
---

# Illumination Fusion System — How It Works

This section explains the multi-illumination fusion engine built into the viewer.
Same audience: engineer, manager, or customer.

---

## The Problem We Solve

A scratch on a metal part may be:
- **Invisible** under overhead lighting — the light hits both sides of the scratch at the same angle
- **Clearly visible** under side (grazing) lighting — the light rakes across the surface and the scratch casts a shadow

The same scratch. Same camera. Same part. **Different light angle — completely different result.**

An inspector using one fixed light will miss defects that only appear under a specific angle.
But capturing 4 or 8 different lighting angles and manually comparing them is slow and error-prone.

**Illumination Fusion solves this by mathematically combining all lighting angles into one image** — one that shows everything every single angle revealed, presented as a single clear result.

This is the same technology used in:
- **Cognex In-Sight** for structured-light surface inspection
- **Halcon** photometric stereo pipelines
- **Keyence IV series** multi-angle lighting controllers
- **Academic RTI (Reflectance Transformation Imaging)** for museum artifacts and coin inspection

---

## The Core Idea — Same Surface, Different Lights

Imagine you photograph the same scratched metal part four times, each time with the light coming from a different direction:

```
Light from RIGHT  →  sees scratches running left-right clearly,
                     misses scratches running up-down

Light from TOP    →  sees scratches running up-down clearly,
                     misses scratches running left-right

Light from LEFT   →  different shadows than RIGHT

Light from BOTTOM →  completes the set
```

Each image captures the truth from one angle.
No single image tells the whole story.
**Fusion combines all four into one image that tells the whole story.**

---

## Eight Fusion Operations

The system has eight operations. They are organized into four groups.

---

### Group 1 — Colour-Based Fusion

#### RGB Composite

**What it does:**
Assigns one grayscale image to the Red channel, one to Green, one to Blue — then merges them into a colour image.

**The physics:**
The red channel is bright wherever the right-side light caused a bright reflection.
The green channel is bright wherever the top light caused a bright reflection.
The blue channel is bright wherever the left light caused a bright reflection.

A flat, smooth surface reflects each light at the same angle → same brightness in all three channels → the merged colour is **neutral grey**.

A scratch or pit scatters one light differently from the other two → one channel is brighter than the others → the merged colour is **red, green, or blue** — colour-visible against a grey background.

**When to use:**
- When you have exactly 3 images from 3 different light positions
- For a fast, visually striking result that shows where the surface is non-uniform
- For presentation — colour maps are easier for non-experts to read than grey maps

**What to set:**
- Assign image 1 to R, image 2 to G, image 3 to B using the R/G/B checkboxes
- Weight (0–2): scale how strongly each image contributes. Default 1.0 — do not change unless one image is over/under-exposed.

---

### Group 2 — Physics-Based Fusion (require light angle input)

These two operations need you to enter the azimuth and elevation angle for each light source.
The math uses the angles to calculate real surface geometry, not just a visual blend.

---

#### Photometric Stereo (Woodham 1980 algorithm)

**What it does:**
From 3 or more images with known light angles, it calculates the **real surface normal** (the direction the surface faces) at every single pixel.

**The physics — in plain language:**
If you know the direction a light is coming from, and you know how bright the surface appears,
you can calculate the direction that surface must be facing.
With 3 lights, you have 3 equations and 3 unknowns (surface normal x, y, z).
With more lights, you get a more stable answer (least-squares fit, same principle as GPS or any over-determined system).

This is called **Woodham's method** (Robert Woodham, 1980, University of British Columbia).
It is the foundational algorithm for all photometric stereo in industrial machine vision.

**Four outputs — pick one:**

| Output | What it shows | Best for |
|---|---|---|
| **Gradient Magnitude** | How sharply the surface slope changes at each pixel. Flat = black. Steep edge = bright white. | Scratches, cracks, stamp edges, machined features |
| **Normal Map** | Surface orientation as colour. Red = slopes right, Green = slopes up, Blue = faces camera. | Visualizing 3D topology, verifying light calibration |
| **Albedo** | True material reflectivity with lighting effects removed. | Stains, coatings, discolouration, logo verification |
| **Height Map** | Reconstructed 3D depth. Bright = raised, Dark = recessed. | Dents, bumps, warps, engraved features |

**Requirements:**
- At minimum 3 images, each from a different light direction
- Enter the azimuth (0–360°) and elevation (0–90°) for each image in the image list
- Lights should be spread out — 4 lights at 0°/90°/180°/270° azimuth is ideal

**When to use:**
- When you need to know the actual surface geometry, not just where things look different
- When defects are low-contrast under every single light individually
- Gradient Magnitude is the recommended default output for defect detection

---

#### RTI Relight (Reflectance Transformation Imaging)

**What it does:**
Fits a mathematical model to every pixel, then lets you **virtually move the light in real time** — no physical light moved, no new image taken.

**The math — in plain language:**
For each pixel, the system fits a 6-term polynomial:

```
Brightness = c₀·u² + c₁·v² + c₂·uv + c₃·u + c₄·v + c₅
```

where u and v are the horizontal and vertical light direction components.
This polynomial describes how that specific pixel's brightness changes as the light moves.

After fitting (a one-time 5–30 second computation), evaluating the polynomial for any (u, v) takes microseconds.
You drag the Azimuth and Elevation sliders — the image updates instantly.

**Why this is powerful:**
Sometimes a defect only appears under one very specific lighting angle — perhaps 27° azimuth, 12° elevation.
No one would physically set up that exact angle in a lab.
RTI lets you sweep through every possible angle by dragging a slider until the defect "pops" into visibility.

**Requirements:**
- Minimum 6 images (the polynomial has 6 unknowns — need at least 6 equations to solve it)
- More images = more stable fit. 12–16 images is ideal.
- Enter azimuth and elevation for each
- Click **Fit RTI Polynomials** once (slow). Then drag sliders (instant).

**When to use:**
- For subtle surface features that are only visible at one specific light angle
- For archaeological objects, coins, embossed text, and engraved surfaces
- When you want to interactively explore the surface rather than run a fixed algorithm

---

### Group 3 — Pixel Statistics (no angle input needed)

These operations do not use light angles. They treat the images as a stack of numbers and compute statistics.
Fast. No calibration needed. Works with any images.

---

#### Max

Every output pixel = the brightest value that pixel had across all input images.

**In plain language:** If a bright reflection appears in ANY one lighting angle, it shows up in the result.

**Use for:** Finding bright defects — protrusions, burrs, raised solder, reflective contamination.

---

#### Min

Every output pixel = the darkest value that pixel had across all input images.

**In plain language:** If a dark shadow appears in ANY one lighting angle, it shows up in the result.

**Use for:** Finding dark defects — pits, voids, holes, missing material, dark contamination.

---

#### Average

Every output pixel = the mean value across all images.

**In plain language:** Reduces noise. Averages out lighting-direction-specific reflections. Shows the general surface brightness.

**Use for:** Noise reduction, baseline comparison, reducing the effect of any single outlier image.

---

### Group 4 — Multi-Image Arithmetic (the most powerful tools)

These three operations are where multi-image fusion becomes significantly better than any two-image comparison.

---

#### Difference  |A − B|

Every output pixel = the absolute difference between image A and image B.

**In plain language:**
- Flat surface → same brightness under both lights → |A − B| = 0 → **black**
- Scratch or pit → reacts differently to each light → |A − B| is large → **bright white**

The background disappears. Only what changed between the two angles is visible.

**When to use:**
- When you want to isolate exactly what one lighting angle reveals that another does not
- When the background is complex and you want it suppressed

**Limitation:** Only compares two specific images. If the defect appears under light 3 but not lights 1 or 2, Difference misses it.

---

#### Superposition   A + B + C + …

Every output pixel = the sum of that pixel across ALL images, then normalised to 0–255.

**In plain language:**
- A defect visible in 3 different lighting angles accumulates 3× brighter than a defect visible in only 1 angle.
- A surface that is dark everywhere = dark in the result.
- A scratch that catches light from multiple angles = very bright in the result.

**Why this is useful:**
Your colleague who uses "+ of pixels" for RGB enhancement is using this principle.
When 3 separate grayscale images each show part of the defect, the sum shows all of it at once.
Defects that appear under many angles are emphasized. Noise (random, only in one image) is damped.

**When to use:**
- To confirm defects that appear under multiple lighting angles
- As a starting point when you have 3+ images and want a fast first look
- Before RGB Composite — Superposition works with any number of images, not just 3

---

#### Multiply   A × B × C × …

Every output pixel = the product of all images (each divided by 255 first, then multiplied together).

**In plain language:**
- If a pixel is bright (close to 1.0) in EVERY image → the product is still close to 1.0 → **bright**
- If a pixel is dark (close to 0) in ANY ONE image → the product collapses toward 0 → **black**

This is extremely selective. Only pixels that are bright in every single lighting angle survive.

**Why this is useful:**
Your colleague who uses "× of pixels" is using this operation.
It asks: "Which defects are visible no matter what angle you look from?"
A real structural defect (deep scratch, genuine void) is likely to affect reflectance from multiple angles.
A surface texture, dust particle, or specular highlight usually only appears under one angle and is killed by multiplication.

**When to use:**
- To confirm defects that are real regardless of lighting direction
- To filter out lighting-angle-specific reflections (glare, specular spots)
- Combined with Superposition: run both, compare — what appears in Superposition but not Multiply is angle-specific; what appears in both is a confirmed defect

---

#### Range   Max − Min

Every output pixel = the maximum value minus the minimum value across all images.

**In plain language:**
- Flat, uniform surface → same brightness in every lighting angle → Max ≈ Min → **black** (no change = no defect)
- Scratch or pit → very bright under one angle, very dark under another → Max − Min is large → **bright white**

**This is the most powerful automatic defect detector in the entire system.**

You do not need to choose which two images to compare.
You do not need to know which angle reveals which defect.
You load all your images, click Range, and the result is bright wherever the surface behaves differently across lighting angles — regardless of which specific angles reveal it.

**Why Range beats Difference:**
Difference compares two images: A and B. If the defect is in image C, you miss it.
Range uses all N images simultaneously. No defect is missed because you chose the wrong pair.

**When to use:**
- When you have 3, 4, 5, 6, or more images and want the single most informative result
- For automated inspection where you cannot manually select the best pair
- As the primary operation in a multi-light illumination dome setup
- After Range, feed the result into the Filter Pipeline → CLAHE + Canny for further analysis

---

## Workflow — How an Expert Uses This System

### Setup (one time per part type)

1. Mount the part in the fixture
2. Capture images under each lighting angle — use a consistent naming convention:
   - `part_001_az000_el45.tif` (azimuth 0°, elevation 45°)
   - `part_001_az090_el45.tif` (azimuth 90°, elevation 45°)
   - etc.
3. In the viewer, open the Fusion panel
4. Load all images from the current folder using **From Current Folder → Add Selected to Fusion**
5. Select the operation that best matches the defect type (see table below)

### Operation selection guide

| Defect type | Recommended operation |
|---|---|
| Scratches (any direction) | **Range** or **Photometric Stereo → Gradient Magnitude** |
| Dark pits / voids | **Min** or **Range** |
| Bright protrusions / burrs | **Max** or **Superposition** |
| Stains / coating defects | **Photometric Stereo → Albedo** |
| Unknown defect type | **Range** first, then explore |
| 3D topology (dents, warps) | **Photometric Stereo → Height Map** |
| Interactive exploration | **RTI Relight** |
| Colour-mapped result for report | **RGB Composite** |

### Production inspection

For automated pass/fail, the recommended pipeline is:

```
1. Capture 4 images: lights at 0°, 90°, 180°, 270° azimuth, 30° elevation
2. Load all 4 into Fusion
3. Run Range  →  get the defect map
4. Feed into Filter Pipeline:  CLAHE → Canny Edge
5. Feed into Compare Panel  →  compare to a known-good reference part
6. Read PASS / FAIL verdict
```

This combines the physics of illumination fusion with the statistical power of the comparison engine.

---

## What the Compose Button Does

Clicking **▶ Compose** runs the selected algorithm on a background thread (the UI never freezes) and sends the result to the main viewer as a new image.

The fused result can then be:
- Panned and zoomed in the main viewer
- Processed through the Filter Pipeline (CLAHE, Canny, False Color, etc.)
- Compared against a reference in the Compare Panel
- Zoomed in Inspect mode for measurement and annotation

The fusion result is a full-resolution image. Nothing is downsampled. All precision is preserved.

---

## Design Principles

**No black box results.**
Every operation has a documented formula. The hover tooltip on every button explains exactly what it computes and when to use it.

**No hidden normalization that hides information.**
Range and Superposition normalize to 0–255 for display. The raw numerical range is preserved in the tooltip description so experts know what they are looking at.

**Works with any number of images.**
Max, Min, Average, Superposition, Multiply, Range all work with 2, 3, 4, 5, or more images automatically. You never have to choose a pair.

**Progressive complexity.**
A beginner can run Range and get a useful result in 30 seconds.
An expert can run Photometric Stereo with calibrated angles and extract real surface normals.
Both use the same panel.

**Fails clearly.**
If you run Photometric Stereo with fewer than 3 images, it returns nothing and shows a status message.
If you run RTI Relight before fitting, it shows "Fit RTI Polynomials first."
The system never silently returns a wrong result.

---
---

# 3D Surface Viewer — How It Works

This section explains the 3D surface viewer and the difference between its two modes.

---

## The Problem With "Fake" 3D

The obvious way to make a 3D surface from a 2D image is to treat brightness as height:
- Bright pixel → treat as raised
- Dark pixel → treat as recessed

This works as a **visual effect**. It is useful for getting a dramatic overview of contrast distribution.

But it is **physically wrong**. A scratch on a metal part appears bright under side-lighting — the light glances off one edge of the groove. That scratch is actually recessed into the surface. Brightness-as-height would show that scratch as a mountain sticking upward, which is the opposite of reality.

**This is not a niche edge case. This is the normal situation in industrial inspection.** Defects are visible precisely because they change how light reflects. Their brightness has no reliable relationship to their physical height.

---

## Two Modes — Visual vs Real

The 3D viewer operates in two clearly labelled modes:

### Visual Mode  (brightness → height)

**When active:** Any image loaded directly — camera photo, pipeline result, fusion composite (except Height Map).

**What it shows:** Pixel brightness mapped to Z height. Bright = tall, dark = flat.

**Use for:** Quick visual overview. Checking contrast distribution across the image. Presentation.

**Limitation:** The geometry is not physically real. A bright defect appears raised even if it is actually a pit. Do not use this mode for dimensional measurement or topology analysis.

The info bar reads: `Visual — brightness → height`

---

### Real Geometry Mode  (Photometric Stereo height map)

**When active:** Automatically, when you run Photometric Stereo → Height Map in the Fusion panel.

**What it shows:** The actual reconstructed surface depth at every pixel, computed by the Woodham (1980) photometric stereo algorithm and integrated by the Frankot-Chellappa (1988) FFT method.

- Bright = surface is truly raised above the baseline
- Dark = surface is truly recessed below the baseline
- The shape of the mesh is the shape of the actual surface

**Use for:** Detecting dents, bumps, warps, engraved text, solder height variation, any defect that has physical depth. This is how Halcon and Cognex implement 3D surface inspection from camera-only setups.

**Green badge shown in panel:** `⚡ Real Surface Geometry — Photometric Stereo — Bright = raised · Dark = recessed`

The info bar reads: `⚡ Real Surface — Photometric Stereo`

---

## How to Get Real Geometry — Step by Step

1. Open the Fusion panel (left sidebar)
2. Load ≥ 3 images of the same part, each taken with the light from a different angle
3. Set azimuth and elevation for each image in the image list
4. Select mode: **Photometric Stereo**
5. Select output: **Height Map**
6. Click **▶ Compose**

The app automatically:
- Runs the Woodham least-squares algorithm on a background thread (UI never freezes)
- Sends the result to the main viewer as a greyscale image
- Sends the same result to the 3D viewer as real geometry
- Switches to the 3D tab

You see the surface topology in 3D immediately. Drag to rotate. Scroll to zoom. Use the Z Scale slider to amplify subtle height variation.

---

## The Algorithm Behind It

### Step 1 — Photometric Stereo (Woodham 1980)

Given N images with known light directions L₁, L₂, … Lₙ, solve for the surface gradient at every pixel:

```
g = pinv(L) × I
```

where `I` is the stack of pixel intensities (N × pixels) and `L` is the light direction matrix (N × 3).

This gives a pseudo-normal vector `g = ρ·n` at every pixel, where ρ is the surface albedo and n is the unit normal direction.

### Step 2 — Frankot-Chellappa Integration (1988)

Convert the surface normals (gradient field) to a height field using FFT-based integration:

```
Z_fft = (−i·fx·FFT(gx) − i·fy·FFT(gy)) / (fx² + fy²)
Z = real(IFFT(Z_fft))
```

This is mathematically equivalent to integrating the gradient over the entire surface simultaneously, which is much more stable than row-by-row or column-by-column integration.

**Result:** A floating-point height map where every pixel's Z value is the reconstructed surface depth. This is normalized to 0–255 for display, preserving all relative height relationships.

---

## Controls

| Control | Effect |
|---|---|
| **Resolution slider** | Downsamples the mesh for speed. 8× default. Drop to 4× or 2× for more detail (slower). |
| **Z Scale slider** | Amplifies the height values vertically. Real Geometry mode defaults to 15 (gentler). Increase for subtle topology. |
| **Color** | Colormap applied to the height values. Thermal = classic. Viridis = perceptually uniform. |
| **Smooth** | Gaussian blur applied before mesh construction. Reduces jagged edges at the cost of fine detail. |
| **Reset View** | Returns camera to default position (elevation 24°, azimuth 42°, distance 145). |

---

## Accuracy and Limitations

**Photometric Stereo height maps are approximate reconstructions**, not calibrated measurements.

The accuracy depends on:
- How many images are used (more = more stable, fewer noise effects)
- How well-distributed the light angles are (4 lights at 0°/90°/180°/270° is ideal)
- How accurately you entered the azimuth and elevation angles
- Whether the surface is Lambertian (diffuse reflector) — specular / mirror surfaces violate the PS assumption and will produce artifacts

For absolute height measurement in micrometres, a laser profilometer or calibrated structured light system is needed. Photometric Stereo gives accurate **relative topology** — the shape is correct, the scale is approximate.

For detecting that a dent exists, where it is, and roughly how deep it is relative to the surrounding surface: Photometric Stereo is entirely sufficient and is the standard method used in academic and industrial research without dedicated 3D hardware.

---

## Design Principle

The two modes are explicitly labelled at all times. The viewer never shows a 3D surface without telling you whether the geometry is real or visual. An expert looking at the panel immediately knows which kind of data they are seeing and can calibrate their interpretation accordingly.

---
---

# UI Reference — Complete Guide to the Interface

This section is the baseline reference for the entire application UI.
Written for: new engineers joining the project, QA testers, customers, and managers.
If you want to understand what every button, panel, menu, and toolbar does — this is the document.

---

## Overall Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  File   View   Pipeline   Tools          ← Menu bar             │
├──────────────────────────────────────────────────────────────────┤
│  Inspect │ Focus │ Compare │ Filters │ Fusion │ 3D    ← Mode tabs│
│  [context toolbar — appears only when a tool is active]          │
├───────────┬──────────────────────────────────────────────────────┤
│           │                                                      │
│  Left     │   Main workspace                                     │
│  Panel    │   (image viewer / comparison / fusion / 3D)          │
│  (docked) │                                                      │
│           │                                                      │
├───────────┴──────────────────────────────────────────────────────┤
│  Status bar — filename · zoom · focus score · tool hint          │
└──────────────────────────────────────────────────────────────────┘
```

The left panel is a dockable sidebar. It contains:
- **File Browser** — folder tree + image list
- **Inspector** — pixel values, ROI stats, histogram, line profile, measurements, annotations
- **Pipeline** — filter chain (stackable image processing steps)
- **Fusion** — illumination fusion controls (multi-light image combining)

Each dock panel can be shown/hidden from the **View** menu or by dragging. They can be resized, stacked, or floated as separate windows.

---

## Mode Tabs

The six tabs at the top switch the main workspace between completely different views.

### Inspect (default)

The primary image viewer. Shows the loaded image with full zoom/pan controls.
The left panel shows the Inspector and Browser docks.

**What you do here:**
- Browse and open images
- Zoom in on defects
- Draw ROI, measure distances, place annotations, draw inspection masks
- Run the filter pipeline and see the result
- Check focus scores (if Focus mode data is available)

---

### Focus

Same image viewer, but overlays a focus score grid on the image.

- The image is divided into an 8×8 grid (64 cells)
- Each cell is scored GREEN (sharp), AMBER (soft), or RED (blurry)
- Score is computed using Laplacian Variance and Tenengrad — two independent metrics
- The Focus Assist panel shows the overall verdict, confidence level, and what to do
- See the **Focus System** section of this document for full algorithm details

**Important:** Focus mode does not change the image. It only overlays the score grid. All other tools still work.

---

### Compare

Side-by-side comparison of two images (A and B).

- Load image A and image B using the two load buttons in the panel
- The system automatically aligns B to A (ORB feature matching + homography) and normalizes brightness
- The diff map shows `|A − B|` per pixel using the HOT colormap (black = same, white = very different)
- Defects are detected, classified (Cosmetic / Functional / Critical), and listed
- PASS / FAIL verdict shown at the top
- See the **Image Comparison System** section for full algorithm details

**Display modes:** Diff Map · Signed ±128 · Blend 50/50 · Flicker (alternates A/B)

---

### Filters

The filter pipeline. A chain of image processing steps applied in order, top to bottom.

- Add filters from the category list on the left
- Each filter has sliders for its parameters — changes apply in real time (debounced 400ms)
- Drag filters to reorder; use the toggle to enable/disable without removing
- The pipeline result updates the main viewer and the 3D viewer simultaneously
- Save/load pipeline configurations for reuse across sessions
- Expert presets available: Surface Scratch, Dark Pits, Bright Protrusions, PCB Trace, Texture Defect, Low Contrast Fix

**Workflow steps:** ① Pre-process → ② Enhance → ③ Detect → ④ Visualize

---

### Fusion

Illumination fusion. Combines multiple images of the same part taken under different lighting angles.

- Load images from the current folder or from disk
- Choose a fusion operation (see below)
- Click **▶ Compose** — result appears in the main viewer
- The result can be sent to the Filter Pipeline for further processing

**Operations:**

| Operation | What it does |
|---|---|
| RGB Composite | Assign 3 grayscale images to R, G, B channels — defects become colour-visible |
| Photometric Stereo | Woodham algorithm — computes real surface normals from ≥3 images with known light angles |
| RTI Relight | Fits biquadratic polynomial per pixel — drag sliders to relight surface in real time |
| Max | Brightest pixel wins across all images |
| Min | Darkest pixel wins — reveals pits and voids |
| Average | Mean of all images — reduces noise |
| Difference \|A−B\| | Absolute difference between two selected images — flat = black, defect = white |
| Superposition | A+B+C sum — defects visible in multiple lights accumulate brighter |
| Multiply | A×B×C product — only pixels bright in every image survive |
| Range | Max−Min — best automatic defect detector, works with any number of images |

See the **Illumination Fusion System** section for full details.

---

### 3D

3D surface mesh viewer. Renders the current image as a 3D surface.

**Two modes:**

| Mode | When | Geometry |
|---|---|---|
| Visual | Any image loaded normally | Brightness → Z height. Visual only, not physically real. |
| Real Geometry | Photometric Stereo → Height Map composed in Fusion | Actual reconstructed surface depth. Bright = raised, Dark = recessed. |

When PS Height Map is composed, the app automatically switches to the 3D tab and shows the real geometry with a green badge: `⚡ Real Surface Geometry — Photometric Stereo`.

**Controls:** Resolution (mesh density) · Z Scale (height amplification) · Colormap · Smooth
**Navigation:** Drag = rotate · Scroll = zoom · Reset View button

---

## The Tools Menu

The **Tools** menu contains two groups of items.

### Global tools (top section)

| Item | Shortcut | What it does |
|---|---|---|
| Illumination Fusion | Ctrl+F | Switch to Fusion tab |
| 3D Surface View | Ctrl+3 | Switch to 3D tab |
| Image Comparison | Ctrl+M | Open comparison window |

### Inspect Tools (middle section)

These select which inspect tool is active. Only visible/relevant when in Inspect mode.

| Tool | Shortcut | What it does |
|---|---|---|
| ↖ Navigate (pan / zoom) | Esc | Default mode — left drag pans, scroll zooms. Toolbar hidden. |
| ⬛ ROI | R | Draw a rectangle → region statistics appear in Inspector panel |
| 📈 Profile | P | Draw a line → intensity chart appears in Inspector panel |
| 📍 Annotate | A | Click to place a defect marker — choose label from popup |
| ↔ Measure | M | Drag to measure distance — shown in px and mm (if scale is set) |
| ⬡ Mask | K | Draw a polygon — metrics are only computed inside the masked area |

**The checkmark** in the menu shows which tool is currently active.

### Utility tools (bottom section)

| Item | What it does |
|---|---|
| Set Scale Calibration (mm/px)… | Enter the mm-per-pixel ratio for Measure tool accuracy |
| Clear Inspect Overlays | Remove ROI rectangle, profile line, measure line, all annotations |
| Clear Mask | Remove the inspection mask polygon |

---

## The Context Toolbar

The toolbar row below the mode tabs is **context-sensitive** — it only appears when a non-navigate inspect tool is active.

**When Navigate is active:** toolbar is hidden. The interface is clean.

**When any other tool is active:** toolbar appears showing:
1. A **coloured active tool badge** on the left — immediately shows which tool is on
   - `⬛ ROI` — cyan
   - `📈 Profile` — cyan
   - `📍 Annotate` — amber
   - `↔ Measure` — green
   - `⬡ Mask` — purple
2. **Tool-specific action buttons** in the middle
3. A dim **↖ Navigate** button on the right — click it to return to navigate mode (same as pressing Esc)

### Toolbar contents per tool

**ROI**
`[⬛ ROI]  |  [✕ Clear]  ···  [↖ Navigate]`

**Profile**
`[📈 Profile]  |  [✕ Clear]  ···  [↖ Navigate]`

**Annotate**
`[📍 Annotate]  |  [✕ Clear]   Annotations auto-saved alongside image file  ···  [↖ Navigate]`

**Measure**
`[↔ Measure]  |  [⚙ Set Scale…]  [✕ Clear]  ···  [↖ Navigate]`

**Mask**
`[⬡ Mask]  |  [✦ Auto-Detect]  [📋 Apply to Folder]  [⧉ Find All Similar]  |  [💾 Save Mask]  [⬡ Clear Mask]  ···  [↖ Navigate]`

---

## Inspect Tools — Detailed Behaviour

### Navigate (default, Esc)

Default state. No tool active — just pan and zoom.
- Left drag → pan the image
- Right drag → zoom
- Scroll wheel → zoom
- Toolbar hidden — interface is clean

Switch to Navigate from any tool by pressing **Esc** or clicking **↖ Navigate** in the toolbar.
Switching to any non-Inspect mode also resets to Navigate automatically.

---

### ROI — Region of Interest (R)

Draw a rectangle on the image. The Inspector panel immediately shows statistics for that region:
- Mean, min, max, standard deviation of pixel values
- Histogram of the region
- Channel breakdown for colour images

**How to use:**
1. Press R (or Tools → ROI)
2. Left-drag a rectangle over the area of interest
3. Inspector panel opens automatically showing stats for that region
4. Draw a new rectangle to replace the ROI
5. Press Esc to return to Navigate

**Use for:** Comparing brightness between defect regions and good surface regions. Checking if a dark spot is statistically different from the background.

---

### Profile — Line Profile (P)

Draw a line on the image. The Inspector panel shows an intensity chart along that line.

**How to use:**
1. Press P (or Tools → Profile)
2. Left-drag a line across the area of interest
3. Inspector panel shows the brightness values along the line as a chart
4. Peaks and dips in the chart correspond to edges, scratches, and surface features

**Use for:** Measuring the width of a scratch (distance between the two intensity dips). Verifying that a coating is uniform (flat line = uniform, bumps = variation). Checking edge sharpness.

---

### Annotate — Place Defect Markers (A)

Click anywhere on the image to place a defect annotation marker.

**How to use:**
1. Press A (or Tools → Annotate)
2. Left-click on a defect location
3. A label popup appears — choose the defect type (Scratch, Pit, Contamination, Burr, Crack, OK, Other)
4. Marker appears on the image with the label
5. All annotations are listed in the Inspector panel
6. Annotations are automatically saved to a sidecar file alongside the image

**Saved file:** `imagename.annotations.json` — stored next to the image file. Survives app restarts. Loaded automatically when the image is opened next time.

**Use for:** Creating a traceable inspection record. Marking defect locations for review. Generating a defect map for a report.

---

### Measure — Distance Measurement (M)

Drag a line between two points to measure the distance.

**How to use:**
1. Press M (or Tools → Measure)
2. Left-drag from point A to point B
3. Inspector shows: pixel distance, real-world distance in mm (if scale is set), start/end coordinates, angle
4. To get mm values: click **⚙ Set Scale…** and enter the mm-per-pixel ratio for your lens/distance

**Setting the scale:**
- Photograph a ruler or calibration target at the same working distance as your parts
- Measure a known distance in pixels (draw a Measure line over a known feature)
- Divide known mm by pixel count → mm/px ratio
- Enter this in Set Scale… — it is saved and used for all future measurements in this session

**Use for:** Measuring scratch length, pit diameter, gap between features, verifying dimensions against drawing tolerances.

---

### Mask — Inspection Polygon (K)

Draw a polygon to define the inspection region. All metrics (focus score, histogram, ROI stats) are computed only inside the mask.

**How to use:**
1. Press K (or Tools → Mask)
2. Left-click to place polygon vertices on the image
3. Right-click or close the polygon to finish
4. Masked area is highlighted — everything outside is excluded from analysis

**Mask toolbar actions:**

| Button | What it does |
|---|---|
| ✦ Auto-Detect | Automatically detect specular reflection regions and suggest a mask that excludes them |
| 📋 Apply to Folder | Copy the current mask to all images in the folder (fixed camera: direct copy; moving part: auto-align by edge matching) |
| ⧉ Find All Similar | Draw mask around one example region, then click — finds all identical/similar regions and masks them all |
| 💾 Save Mask | Save the mask to disk — choose position-only or export masked image file |
| ⬡ Clear Mask | Remove the mask — return to analyzing the full image |

**Use for:**
- Excluding fixture, background, label, or edge regions from the inspection area
- Focusing the focus score computation on the part only, not the background
- Masking reflective spots that confuse defect detection

---

## Status Bar

The status bar at the bottom always shows:

```
[1] 19RGB.bmp   11%  (1:8.8)   ● F:14  Q:98    Tool: MASK — draw polygon inspection region
```

| Field | Meaning |
|---|---|
| `[1]` | Active viewer cell (1–4 in multi-view) |
| `19RGB.bmp` | Current image filename |
| `11%` | Current zoom level |
| `(1:8.8)` | Pixel ratio — 1 screen pixel = 8.8 image pixels |
| `● F:14` | Focus score (red/amber/green dot) |
| `Q:98` | Image quality score |
| Tool hint | Current tool name and one-line usage hint |

---

## Keyboard Shortcuts Summary

| Key | Action |
|---|---|
| Esc | Return to Navigate (pan / zoom) |
| R | ROI tool |
| P | Profile tool |
| A | Annotate tool |
| M | Measure tool |
| K | Mask tool |
| Ctrl+F | Switch to Fusion tab |
| Ctrl+3 | Switch to 3D tab |
| Ctrl+M | Open Image Comparison |
| Ctrl+O | Open image file |
| Ctrl+Shift+L | Load pipeline |
| Left / Right arrow | Next / previous image in folder |

---

## Left Panel Docks

### File Browser

- Folder tree on top — click to navigate directories
- Image list below — click to open an image in the viewer
- Filter bar — type to filter filenames
- Supports TIFF, TIF, PNG, BMP, JPG, PGM
- Double-click any image in the Fusion panel's folder list to add it to the fusion set

### Inspector

The Inspector panel shows analysis results for the current image and tool state.

Sections (scroll down to see all):
- **Pixel** — value under mouse cursor (R, G, B, or grayscale)
- **Histogram** — intensity distribution of the full image or ROI
- **ROI Stats** — mean, min, max, std-dev for the drawn rectangle
- **Line Profile** — intensity chart along the drawn profile line
- **Measurement** — distance result from the Measure tool
- **Annotations** — list of all placed markers with labels and coordinates

### Pipeline

Filter chain — see the **Filters** tab section above.

### Fusion

Illumination fusion controls — see the **Fusion** tab section above.

---

## Design Philosophy

**Every state is visible.**
The active tool is always shown in the toolbar badge. The current mode is shown in the mode tabs. The scoring mode (relative / auto-ref / locked) is always labelled. Nothing is hidden.

**No tool shows controls that do not apply.**
The toolbar is empty by default. Mask controls only appear when Mask is active. Profile controls only appear when Profile is active. Clutter causes mistakes — experts notice clutter immediately.

**The expert can always verify the math.**
Raw metric values (Laplacian, Tenengrad, Brenner), raw SSIM/PSNR numbers, exact pixel coordinates, real-world distances — all shown in the Inspector. The tool never just says "pass" without showing the number behind it.

**Fails clearly, never silently.**
If Photometric Stereo has fewer than 3 images, it says so. If RTI has not been fitted yet, it says so. If a pipeline filter fails, the error is shown in the status area. The app never returns a wrong result silently.
