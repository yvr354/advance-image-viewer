# YVR Advanced Image Viewer — Complete Specification
**Version:** 1.0  
**Date:** 2026-04-30  
**Purpose:** Professional-grade image acquisition, visualization, and defect analysis tool for industrial vision engineers  

---

## 1. VISION & GOALS

This application is built for AI vision engineers who design and validate defect detection systems. The core belief:

> **"Good image quality = Good AI results"**

The tool replaces human-eye judgment with **objective, numerical metrics** at every step. It is not an AI inference tool — it is the instrument that ensures the data fed into AI models is optimal.

---

## 2. CORE PRINCIPLES

| Principle | Implementation |
|-----------|---------------|
| Human eye is unreliable | Every quality metric is numerical and objective |
| Camera-agnostic | Works with any camera by reading saved image files |
| Basler native | pypylon SDK for live view and acquisition |
| Non-destructive | All processing is view-only; original files never modified |
| Best-of-best quality | If image is bad, AI results will be bad — catch it here |

---

## 3. COMPLETE FEATURE LIST

### 3.1 — IMAGE ACQUISITION & INPUT

| # | Feature | Detail |
|---|---------|--------|
| A1 | Folder watcher | Auto-loads new images as camera saves them |
| A2 | Multi-format support | TIFF (8/16-bit), BMP, PNG, JPEG, RAW, PGM, PFM, EXR |
| A3 | 16-bit support | Full 16-bit grayscale display and processing |
| A4 | Basler live view | pypylon SDK — GigE and USB3 Basler cameras |
| A5 | Camera parameter control | Exposure, gain, gamma, white balance, trigger mode |
| A6 | Snapshot capture | Capture single frames from live view to file |
| A7 | Burst capture | Capture N frames, auto-select sharpest |
| A8 | Drag and drop | Drop image files or folders onto the app |
| A9 | Recent files | Quick access to last 50 images |
| A10 | Image sequence loader | Load numbered sequences as a filmstrip |

---

### 3.2 — PIXEL-PERFECT VIEWER

| # | Feature | Detail |
|---|---------|--------|
| V1 | 1:1 pixel zoom | True pixel-perfect, zero interpolation |
| V2 | Zoom levels | 6.25% → 25% → 50% → 100% → 200% → 400% → 800% → 1600% |
| V3 | Fit-to-window | Fit entire image in view |
| V4 | Pan | Click-drag or middle-click pan |
| V5 | Zoom to cursor | Zoom centered on mouse position |
| V6 | Mini-map | Overview thumbnail showing current view region |
| V7 | Grid overlay | Pixel grid visible at high zoom |
| V8 | Crosshair cursor | Precise pixel targeting |
| V9 | Ruler overlay | Pixel distance ruler (H and V) |
| V10 | Multi-screen support | Undock viewer to secondary monitor |

---

### 3.3 — PIXEL INSPECTOR

| # | Feature | Detail |
|---|---------|--------|
| P1 | Pixel value display | R, G, B, Alpha, Grayscale under cursor |
| P2 | 16-bit value | Raw 16-bit value displayed, not normalized |
| P3 | Coordinates | X, Y pixel coordinates shown at all times |
| P4 | Neighborhood values | 3×3 and 5×5 pixel neighborhood grid |
| P5 | Profile line | Draw a line, see intensity profile graph |
| P6 | Area statistics | Select rectangle, show Min/Max/Mean/Std/Median |
| P7 | Color picker | Pick pixel color, show HEX / RGB / HSV / Lab |
| P8 | Comparison cursor | Show pixel values from two images simultaneously |

---

### 3.4 — FOCUS & SHARPNESS ANALYSIS

> Based on research: Variance of Laplacian, Tenengrad, Brenner Function are the standard metrics used in industrial machine vision.

| # | Feature | Detail |
|---|---------|--------|
| F1 | Global focus score | Single number 0–1000, higher = sharper |
| F2 | Focus verdict | PERFECT / GOOD / SOFT / BLURRY with color indicator |
| F3 | Focus heatmap overlay | Grid-based map, green=sharp, red=blurry |
| F4 | Grid resolution | User sets grid: 4×4, 8×8, 16×16, 32×32 |
| F5 | Per-region score | Click any grid cell to see its exact score |
| F6 | Sharpest region highlight | Auto-highlight the sharpest zone in the image |
| F7 | Focus trend graph | In live view: focus score over time (last 60 frames) |
| F8 | Multi-metric | Laplacian variance, Tenengrad, Brenner — user selects |
| F9 | ROI focus | Draw custom ROI, measure focus only within it |
| F10 | Focus threshold config | User sets GOOD/SOFT/BLURRY thresholds |
| F11 | Best-frame selector | Capture 10 frames, auto-return sharpest |
| F12 | Focus map export | Save heatmap as overlay PNG |

**Score thresholds (configurable):**
```
> 700    PERFECT   ██████████  Green
400-700  GOOD      ████████░░  Yellow-green
200-400  SOFT      █████░░░░░  Orange
< 200    BLURRY    ██░░░░░░░░  Red
```

---

### 3.5 — IMAGE QUALITY METRICS

| # | Feature | Detail |
|---|---------|--------|
| Q1 | Histogram | Full RGB + luminance histogram with live update |
| Q2 | Exposure analysis | Detect overexposed / underexposed regions (highlight clipping) |
| Q3 | Dynamic range | Measured DR in stops |
| Q4 | SNR measurement | Signal-to-noise ratio in flat regions |
| Q5 | Noise map | Spatial noise distribution overlay |
| Q6 | Contrast score | RMS contrast, Michelson contrast, Weber contrast |
| Q7 | MTF curve | Modulation Transfer Function (edge sharpness vs frequency) |
| Q8 | Banding detection | Detect horizontal/vertical banding artifacts |
| Q9 | Vignette map | Detect corner darkening from lens |
| Q10 | Distortion grid | Detect barrel/pincushion lens distortion |
| Q11 | Overall quality score | Composite score: Focus + Exposure + SNR + Contrast |
| Q12 | Pass/Fail stamp | User-defined thresholds → PASS ✓ or FAIL ✗ on image |

---

### 3.6 — ADVANCED IMAGE PROCESSING FILTERS

> All filters are stackable, reorderable, non-destructive, with real-time preview.  
> Based on research: CLAHE, multi-illumination fusion, structured lighting analysis are proven best practices for surface defect visualization.

#### 3.6.1 — Contrast & Tone

| # | Filter | Controls |
|---|--------|----------|
| T1 | Brightness / Contrast | −255 to +255 each |
| T2 | Levels | Black point, white point, gamma (like Photoshop Levels) |
| T3 | Curves | Full tone curve editor with anchor points |
| T4 | CLAHE | Clip limit (0.1–10.0), tile grid size (4×4 to 32×32) |
| T5 | Histogram equalization | Global equalization |
| T6 | Adaptive threshold | Local thresholding for binarization |
| T7 | Gamma correction | 0.1 – 5.0 |
| T8 | HDR tone mapping | Reinhard / Drago / Mantiuk operators |

#### 3.6.2 — False Color & LUT

| # | Filter | Controls |
|---|--------|----------|
| C1 | False color LUT | JET, HOT, COOL, VIRIDIS, INFERNO, PLASMA, RAINBOW, custom |
| C2 | LUT range clamp | Set min/max input range for LUT |
| C3 | Isoline overlay | Draw contour lines at equal intensity levels |
| C4 | Highlight clipping | Overexposed=Red, Underexposed=Blue overlay |
| C5 | Custom LUT import | Load .cube or .lut file |
| C6 | Threshold colorize | Below threshold=Color A, above=Color B |

#### 3.6.3 — Sharpening & Edge Enhancement

| # | Filter | Controls |
|---|--------|----------|
| E1 | Unsharp mask | Radius, strength, threshold |
| E2 | Laplacian sharpen | Kernel strength |
| E3 | Canny edge | Low threshold, high threshold, aperture |
| E4 | Sobel edge | X/Y/combined, kernel size |
| E5 | Prewitt edge | Direction selection |
| E6 | DoG (Difference of Gaussians) | Sigma 1, sigma 2 — reveals surface texture |
| E7 | LoG (Laplacian of Gaussian) | Sigma, threshold |
| E8 | Morphological gradient | Dilation minus erosion = edge ring |
| E9 | Frequency sharpen | Boost high-frequency components in FFT |

#### 3.6.4 — Smoothing & Noise Reduction

| # | Filter | Controls |
|---|--------|----------|
| N1 | Gaussian blur | Sigma 0.5 – 20.0 |
| N2 | Median filter | Kernel 3×3 to 21×21 |
| N3 | Bilateral filter | Diameter, sigma color, sigma space |
| N4 | NLMeans denoising | H strength, template, search window |
| N5 | Anisotropic diffusion | Iterations, kappa, lambda |

#### 3.6.5 — Morphological Operations

| # | Filter | Controls |
|---|--------|----------|
| M1 | Erode | Kernel shape (rect/ellipse/cross), size |
| M2 | Dilate | Kernel shape, size |
| M3 | Open (erode+dilate) | Removes small bright spots |
| M4 | Close (dilate+erode) | Fills small dark holes |
| M5 | Top-hat | Reveals bright features smaller than kernel |
| M6 | Black-hat | Reveals dark features smaller than kernel |
| M7 | Morphological gradient | Highlights edges |

#### 3.6.6 — Frequency Domain

| # | Filter | Controls |
|---|--------|----------|
| FR1 | FFT magnitude view | Visualize frequency spectrum |
| FR2 | FFT phase view | Phase spectrum visualization |
| FR3 | Low-pass filter | Cutoff frequency slider |
| FR4 | High-pass filter | Cutoff frequency slider |
| FR5 | Band-pass filter | Low and high cutoff |
| FR6 | Notch filter | Remove periodic noise (fix frequency) |
| FR7 | Inverse FFT | Apply frequency-domain modifications |

#### 3.6.7 — Channel Operations

| # | Filter | Controls |
|---|--------|----------|
| CH1 | Channel split | View R, G, B, or grayscale individually |
| CH2 | Channel mixer | R=f(R,G,B), G=f(R,G,B), B=f(R,G,B) with weights |
| CH3 | Grayscale conversion | Luminosity, average, max, min, custom weights |
| CH4 | Invert | Invert all or per-channel |
| CH5 | Normalize | Stretch to full 0–255 range |
| CH6 | Convert to Lab | L, a, b channel separation |
| CH7 | Convert to HSV | H, S, V channel separation |

---

### 3.7 — MULTI-ILLUMINATION FUSION (KEY FEATURE)

> Based on research: "Fusion of multi-light source illuminated images for effective defect inspection on highly reflective surfaces" (ScienceDirect 2022, still state-of-the-art approach)

| # | Feature | Detail |
|---|---------|--------|
| I1 | Load image set | Load 2–8 grayscale images of same scene, different lighting |
| I2 | Channel assignment | Assign any image to R, G, B (or multiple channels) |
| I3 | Per-channel weight | 0.0 – 2.0 weight slider per image per channel |
| I4 | Live composite | Real-time preview as you adjust assignments and weights |
| I5 | Auto-align | Automatic image registration if images are slightly misaligned |
| I6 | Difference image | Subtract any two images to see what changed |
| I7 | Min/Max fusion | Pixel-wise minimum or maximum across all images |
| I8 | Average fusion | Weighted average of all input images |
| I9 | PCA fusion | Principal Component Analysis to extract maximum variance |
| I10 | Save composite | Export fused result as TIFF/PNG |
| I11 | Preset save | Save current channel mix as named preset |
| I12 | Batch apply | Apply saved fusion preset to entire folder |

**Workflow:**
```
Load: Light_1_BrightField.tiff
      Light_2_DarkFieldLeft.tiff
      Light_3_DarkFieldRight.tiff
      Light_4_CoaxialLight.tiff

Assign: R = Light_2 (weight 0.8)
        G = Light_3 (weight 0.6)
        B = Light_1 (weight 0.4)
        
Result: RGB image where defects are color-visible
        Surface scratches = cyan shift
        Particles = bright spots in all channels
        Dents = shadow in dark-field channels
```

---

### 3.8 — PROCESSING PIPELINE (STACKABLE LAYERS)

| # | Feature | Detail |
|---|---------|--------|
| PL1 | Layer stack | Add, remove, reorder processing layers |
| PL2 | Per-layer enable/disable | Toggle any layer on/off instantly |
| PL3 | Per-layer opacity | Blend processed result with previous layer |
| PL4 | Per-layer blend mode | Normal, Multiply, Screen, Overlay, Difference |
| PL5 | Real-time preview | All layers computed and displayed live |
| PL6 | Before/After split | Drag a line to see before vs after |
| PL7 | Save pipeline | Save full stack as named preset (.pipeline file) |
| PL8 | Load pipeline | Load and apply saved pipeline |
| PL9 | Pipeline library | Built-in presets: Surface Scratch, Particle, Crack, Corrosion |
| PL10 | Export pipeline | Share pipeline file with colleagues |
| PL11 | Batch process | Apply pipeline to entire folder, save results |

---

### 3.9 — COMPARISON & ANALYSIS TOOLS

| # | Feature | Detail |
|---|---------|--------|
| CM1 | Side-by-side view | Two images in synchronized panels |
| CM2 | Overlay compare | Blend two images with opacity slider |
| CM3 | Difference image | Absolute difference between two images |
| CM4 | Flicker compare | Alt between two images rapidly (spot differences) |
| CM5 | Reference image | Lock one image as reference, compare all others to it |
| CM6 | Synchronized zoom/pan | Both panels move together |
| CM7 | Statistics comparison | Show quality metrics for both images side by side |
| CM8 | Annotation copy | Copy annotations from reference to current image |

---

### 3.10 — ANNOTATION & MEASUREMENT

| # | Feature | Detail |
|---|---------|--------|
| AN1 | Rectangle ROI | Draw, resize, label |
| AN2 | Ellipse ROI | For circular defect regions |
| AN3 | Polygon ROI | Freeform region selection |
| AN4 | Point marker | Mark specific pixels with label |
| AN5 | Line measurement | Measure distance in pixels |
| AN6 | Angle measurement | Measure angle between two lines |
| AN7 | Arrow annotation | Directional arrows with text labels |
| AN8 | Defect tag | Tag: Scratch / Particle / Crack / Void / Unknown |
| AN9 | Save annotations | Export as JSON or XML alongside image |
| AN10 | Annotation overlay | Show/hide all annotations toggle |

---

### 3.11 — FILE MANAGEMENT & BROWSER

| # | Feature | Detail |
|---|---------|--------|
| FB1 | Thumbnail filmstrip | Horizontal strip with all images in folder |
| FB2 | Thumbnail size | Adjustable thumbnail size |
| FB3 | Sort options | By name, date, size, focus score, quality score |
| FB4 | Filter by quality | Show only images above focus threshold |
| FB5 | Filter by verdict | Show only PASS or FAIL images |
| FB6 | Folder tree | Navigate folder hierarchy |
| FB7 | Quick preview | Hover thumbnail for instant full preview |
| FB8 | Batch rename | Rename files with pattern (e.g., PART_001_GOOD) |
| FB9 | Export selected | Copy selected images to output folder |
| FB10 | Metadata view | EXIF, capture time, camera settings |

---

### 3.12 — EXPORT & REPORTING

| # | Feature | Detail |
|---|---------|--------|
| EX1 | Export processed image | Save current view with all filters applied |
| EX2 | Export at original resolution | No downsampling on export |
| EX3 | Export focus report | PDF/HTML with focus heatmap per image |
| EX4 | Export quality report | PDF/HTML with all quality metrics |
| EX5 | Export comparison | Side-by-side comparison as single image |
| EX6 | Batch export | Process and export entire folder |
| EX7 | CSV export | Quality metrics for all images in spreadsheet |
| EX8 | Screenshot | Save exact current view (including UI overlays) |

---

### 3.13 — BASLER LIVE VIEW (pypylon)

| # | Feature | Detail |
|---|---------|--------|
| BL1 | Camera discovery | Auto-detect all connected Basler cameras |
| BL2 | Camera selector | Switch between multiple cameras |
| BL3 | Live stream | Display live feed at full camera resolution |
| BL4 | FPS display | Show actual frames per second |
| BL5 | Exposure control | Slider + numeric input, microseconds |
| BL6 | Gain control | dB value, 0 – max gain |
| BL7 | Trigger mode | Freerun / Software trigger / Hardware trigger |
| BL8 | Pixel format | Mono8, Mono12, Mono16, BayerRG8, RGB8 |
| BL9 | ROI on camera | Set hardware ROI (offset X/Y, width/height) |
| BL10 | Frame rate limit | Cap FPS to reduce CPU load |
| BL11 | Live focus score | Real-time focus meter on live feed |
| BL12 | Live histogram | Real-time histogram on live feed |
| BL13 | Freeze frame | Freeze live view for inspection |
| BL14 | Record sequence | Record N frames to disk |
| BL15 | White balance | One-click auto white balance |

---

## 4. USER INTERFACE LAYOUT

```
┌──────────────────────────────────────────────────────────────────────┐
│ MENU BAR: File | View | Camera | Filters | Tools | Pipeline | Help   │
├──────────────┬─────────────────────────────────┬────────────────────┤
│ FILE BROWSER │         MAIN VIEWER              │   INSPECTOR PANEL  │
│              │  ┌───────────────────────────┐  │                    │
│ [Folder tree]│  │                           │  │ Focus: 847 GOOD    │
│              │  │    IMAGE / LIVE VIEW      │  │ ████████░░         │
│ [Thumbnails] │  │                           │  │                    │
│              │  │  [Heatmap overlay]        │  │ Exposure: OK       │
│              │  │                           │  │ Noise: 2.1%        │
│              │  └───────────────────────────┘  │ Contrast: 88       │
│              │  Zoom: 100% | X:1024 Y:512       │                    │
│              │  Pixel: R:142 G:89 B:201         │ HISTOGRAM          │
│              │                                  │ [live histogram]   │
│              ├──────────────────────────────────│                    │
│              │  PROCESSING PIPELINE             │ PIXEL INSPECTOR    │
│              │  ┌──────────────────────────┐   │ X:1024 Y:512       │
│              │  │ Layer 1: CLAHE    [on][✕]│   │ R:142 G:89 B:201   │
│              │  │ Layer 2: False Color [on]│   │                    │
│              │  │ Layer 3: Edge     [off]  │   │ ANNOTATIONS        │
│              │  │ [+ Add Layer]            │   │ [list]             │
│              │  └──────────────────────────┘   │                    │
│              │                                  │                    │
│ [Filmstrip]  │  ILLUMINATION MIXER              │                    │
│ ┌──┬──┬──┐  │  Img A→R [slider] Img B→G [sl]  │                    │
│ │  │  │  │  │                                  │                    │
└──────────────┴──────────────────────────────────┴────────────────────┘
```

---

## 5. TECHNOLOGY STACK

| Component | Technology | Why |
|-----------|-----------|-----|
| UI Framework | Python 3.11 + PyQt6 | Fast, native, excellent for image display |
| Image Processing | OpenCV 4.9 + NumPy | Industry standard, all algorithms available |
| Basler Camera | pypylon 3.x | Official Basler Python SDK |
| Histogram display | PyQtGraph | GPU-accelerated real-time plots |
| Packaging | PyInstaller | Single .exe, no Python install needed |
| Image I/O | tifffile, imageio | 16-bit TIFF, RAW, EXR support |
| FFT | NumPy / SciPy | Frequency domain operations |

---

## 6. BUILD PHASES

### Phase 1 — Core Viewer (Week 1)
- Image loading (all formats, 16-bit)
- Pixel-perfect viewer (zoom, pan, minimap)
- Pixel inspector
- Basic histogram

### Phase 2 — Quality Metrics (Week 2)
- Focus score engine (Laplacian, Tenengrad, Brenner)
- Focus heatmap overlay
- Exposure analysis
- Overall quality score + Pass/Fail

### Phase 3 — Filter Pipeline (Week 3)
- Stackable filter layers
- All contrast/tone filters
- False color LUTs
- Edge detection filters
- Before/after split view

### Phase 4 — Multi-illumination Fusion (Week 4)
- Image set loader
- Channel assignment UI
- Weight sliders
- Live composite preview
- Save/load presets

### Phase 5 — Basler Live View (Week 5)
- pypylon integration
- Camera discovery and control
- Live focus score overlay
- Burst capture + best frame selection

### Phase 6 — Advanced Tools + Polish (Week 6)
- Annotation tools
- Comparison view
- Batch processing
- Reporting / export
- Full UI polish
- PyInstaller .exe packaging

---

## 7. RESEARCH-BACKED TECHNIQUES

| Technique | Source | Why it helps |
|-----------|--------|-------------|
| Variance of Laplacian | Roboflow Camera Focus Guide 2025 | Best single-metric focus measurement |
| Tenengrad / Sobel energy | DXOMARK Industrial Vision | Robust to noise, good for textured surfaces |
| CLAHE | Standard OpenCV | Best local contrast enhancement for defect visibility |
| Multi-light fusion to RGB | ScienceDirect 2022 (Fusion of multi-light source) | Defects invisible in single light become color-visible |
| DoG filter | Industrial inspection standard | Reveals surface texture anomalies |
| Top-hat morphology | Industrial inspection standard | Detects bright defects smaller than background features |
| MTF measurement | Imatest / DXOMARK | Objective lens+camera system sharpness |

---

## 8. CONFIGURATION FILE

All settings saved in `config.json`:
- Window layout and panel sizes
- Last opened folder
- Focus score thresholds
- Default pipeline preset
- Camera settings per camera serial number
- LUT preferences
- Keyboard shortcuts (all remappable)

---

## 9. KEYBOARD SHORTCUTS

| Key | Action |
|-----|--------|
| Space | Fit to window |
| 1 | 100% zoom |
| 2 | 200% zoom |
| 0.5 | 50% zoom |
| F | Toggle focus heatmap |
| H | Toggle histogram |
| P | Toggle processing pipeline |
| L | Toggle live view |
| S | Snapshot (capture from live) |
| Left/Right | Previous/next image |
| D | Toggle difference image |
| B | Before/after split |
| Ctrl+S | Save processed image |
| Ctrl+E | Export report |

---

*Sources consulted:*
- [AI-enabled defect detection survey — ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S1474034625009607)
- [Fusion of multi-light source illuminated images — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S088832702200276X)
- [Camera Focus Guide — Roboflow 2025](https://blog.roboflow.com/computer-vision-camera-focus-guide/)
- [Sharpness metrics — Imatest](https://www.imatest.com/imaging/sharpness/)
- [Industrial Vision — DXOMARK](https://corp.dxomark.com/industrial-vision/)
- [Machine Vision System Design Guide 2026 — FJW Optical](https://fjwoptical.com/blogs/blog/the-complete-guide-to-machine-vision-system-design-2026-industrial-edition)
