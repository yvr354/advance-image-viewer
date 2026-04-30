# VyuhaAI Image Viewer

Professional-grade image acquisition and defect visualization tool for industrial vision engineers.

Built for AI vision engineers who design and validate defect detection systems.

> **Core belief: Good image quality = Good AI results**

---

## What This Is

This is **not** an AI inference tool. It is the instrument that ensures every image entering your AI training dataset is optimal — before training begins.

- Camera-agnostic (reads saved image files from any camera)
- Basler camera live view via pypylon
- Multi-illumination fusion (assign different-lighting images to R/G/B channels)
- Pixel-perfect OpenGL viewer (10,000×10,000 images at 60fps)
- 3D surface visualization from 2D images
- Research-backed filters: Gabor, Wavelet, FFT, CLAHE, Top-Hat, DoG

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | Python 3.11 + PyQt6 |
| Image Processing | OpenCV 4.9 + NumPy |
| GPU Rendering | OpenGL via PyQt6 |
| 3D Visualization | PyQtGraph GLViewWidget |
| Basler Camera | pypylon |
| Packaging | PyInstaller → .exe |

---

## Quick Start

```bash
# 1. Install dependencies
setup.bat

# 2. Run
run.bat
```

---

## Features

- **OpenGL Viewer** — GPU texture rendering, 10K×10K images pan/zoom at 60fps
- **Focus Scoring** — Laplacian / Tenengrad / Brenner metrics with spatial heatmap
- **Image Quality** — SNR, exposure, contrast, noise — objective numerical metrics
- **Filter Pipeline** — Stackable non-destructive layers: CLAHE, False Color, Edge, Gabor, Wavelet, FFT
- **Multi-illumination Fusion** — Assign different-lighting images to RGB channels, defects become color-visible
- **3D Surface View** — Image intensity as height map, rotatable 3D mesh, no hardware needed
- **Expert Tooltips** — Every parameter explained with IEEE references and optimal ranges
- **Basler Live View** — pypylon SDK, real-time focus score on live feed

---

## Research Basis

| Feature | Source |
|---------|--------|
| Gabor Filter Bank | IEEE/JIM 2025 — optimal for directional texture defects |
| Wavelet Decomposition | ACM 2025 — multi-scale defect analysis |
| Multi-illumination Fusion | ScienceDirect 2022 — best practice for reflective surfaces |
| Focus Metrics | Imatest / DXOMARK Industrial Vision 2025 |

---

## Build .exe

```bash
build_exe.bat
# Output: dist/YVR_ImageViewer.exe
```
