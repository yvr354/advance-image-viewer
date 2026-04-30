# VyuhaAI Image Viewer - Production Task Tracker

Last updated: 2026-04-30

Goal: build a best-in-class industrial image viewer/acquisition/inspection tool. The app must be image-first, smooth, understandable, and useful for real camera/focus/defect engineering.

Core rule: the center of the window is always the main work area. Every feature must show its result large in the center, with only supporting controls around it.

---

## Current Baseline

| # | Area | Status |
|---|------|--------|
| 1 | Project structure, PyQt6 app shell, resources | Done |
| 2 | Image loading for common formats, including 16-bit files | Done |
| 3 | OpenGL image viewer with zoom/pan | Done |
| 4 | Focus engine: Laplacian, Tenengrad, Brenner, heatmap | Done |
| 5 | Quality engine: exposure, contrast, noise, SNR, pass/fail | Done |
| 6 | Filter pipeline engine and filter registry | Done |
| 7 | Multi-illumination fusion engine | Done |
| 8 | 3D surface viewer prototype | Done |
| 9 | Comparison panel prototype | Done |
| 10 | Dark instrument-style theme | Done |

---

## Priority 0 - Do Not Randomly Add Features

| # | Task | Status |
|---|------|--------|
| P0.1 | Stop treating every feature as another always-visible panel | Pending |
| P0.2 | Redesign around modes: Inspect, Focus, Compare, Tune, Acquire, Review | Pending |
| P0.3 | Keep image/result large in the center for every workflow | Pending |
| P0.4 | Make side/bottom panels contextual and collapsible | Pending |
| P0.5 | Remove wasted empty space from default layout | Pending |
| P0.6 | Create a simple production workflow before adding more tools | Pending |

---

## Priority 1 - Critical Usability Bugs

| # | Task | Status |
|---|------|--------|
| 1.1 | Opening one image with the app must load same-folder image sequence for Next/Previous navigation | Done |
| 1.2 | Left/Right keys must work after double-click/open-with image launch | Done |
| 1.3 | File browser must automatically show and highlight the opened image folder | Done |
| 1.4 | Status bar must show current image index, e.g. Image 19 / 84 | Done |
| 1.5 | Center image must not be pushed small by empty pipeline/compare panels | In Progress |
| 1.6 | Filter parameter changes must not reset zoom/pan while inspecting a defect | Pending |
| 1.7 | Histogram toggle must actually hide/show histogram | Done |
| 1.8 | 16-bit pixel values, histogram, and display swatches must be handled correctly | Done |
| 1.9 | Focus heatmap must not break on small images or unusual image sizes | Done |
| 1.10 | App must explain conflicts like Focus = BLURRY but Quality = PASS | Pending |

---

## Priority 2 - Main Layout: Image-First Production UI

| # | Task | Status |
|---|------|--------|
| 2.1 | Replace default all-panels-visible layout with image-first Inspect Mode | In Progress |
| 2.2 | Add top mode switcher: Inspect, Focus, Compare, Tune, Acquire, Review | In Progress |
| 2.3 | Make pipeline a compact drawer/strip, not a huge empty bottom area | Pending |
| 2.4 | Make inspector a compact verdict panel, not a long technical dashboard by default | Pending |
| 2.5 | Move advanced details into expandable sections | Pending |
| 2.6 | Add a clear first-run/empty-state workflow: Open Image, Open Folder, Connect Camera | Pending |
| 2.7 | Keep only essential numbers always visible: focus, exposure, zoom, pixel, verdict | Pending |
| 2.8 | Add Reset Layout that returns to production layout, not developer layout | In Progress |

---

## Priority 3 - Focus Mode: First-Class Industrial Feature

| # | Task | Status |
|---|------|--------|
| 3.1 | Create dedicated Focus Mode with large center image | Pending |
| 3.2 | Overlay focus heatmap directly on the center image | Pending |
| 3.3 | Show sharpest region and softest region markers | Pending |
| 3.4 | Add grid selector: 4x4, 8x8, 16x16, 32x32 | Pending |
| 3.5 | Clicking a grid cell shows exact local focus score | Pending |
| 3.6 | Add ROI focus so user can measure only product/defect area | Pending |
| 3.7 | Add focus verdict explanation: why sharp/soft/failed | Pending |
| 3.8 | Add presets: metal, plastic, printed label, PCB, textile, custom | Pending |
| 3.9 | Add live focus trend graph for camera tuning | Pending |
| 3.10 | Add peak-hold focus score while user adjusts lens | Pending |
| 3.11 | Add burst best-frame selector: capture N frames and choose sharpest | Pending |
| 3.12 | Add warning when exposure/gain may be corrupting focus score | Pending |

---

## Priority 4 - Compare Mode: Real A/B Engineering Compare

| # | Task | Status |
|---|------|--------|
| 4.1 | Make Compare Mode use the center workspace, not a cramped dock | In Progress |
| 4.2 | Show two large synchronized viewers: A Reference and B Candidate | Pending |
| 4.3 | Make A/B selection obvious; remove confusing generic blue selection state | Pending |
| 4.4 | Support Side-by-side, Difference, Overlay, Flicker modes | Pending |
| 4.5 | Show focus score above each viewer | Pending |
| 4.6 | Show focus heatmap for A and B | Pending |
| 4.7 | Add focus difference heatmap: where B is sharper/softer than A | Pending |
| 4.8 | Add winner badge: A sharper, B sharper, or no meaningful difference | Pending |
| 4.9 | Add small metrics strip: Focus delta, SNR delta, Exposure delta, SSIM/PSNR, Pixel delta | Pending |
| 4.10 | Move batch ranking cards/thumbnails into collapsible left drawer | Pending |
| 4.11 | Add "Compare current image with reference" workflow | Pending |
| 4.12 | Add "Find best image in batch" workflow | Pending |

---

## Priority 5 - Tune/Filter Mode

| # | Task | Status |
|---|------|--------|
| 5.1 | Filter output must always show large in center viewer | Pending |
| 5.2 | Pipeline controls should be compact and contextual | Pending |
| 5.3 | Add before/after split view in center | Pending |
| 5.4 | Add one-click filter presets: scratches, particles, cracks, corrosion, texture | Pending |
| 5.5 | Preserve zoom/pan while tuning filter parameters | Pending |
| 5.6 | Show active filter count and quick disable/enable all | Pending |
| 5.7 | Warn if a filter converts 16-bit image to 8-bit for preview/export | Pending |
| 5.8 | Add filter help text in plain language, not only technical names | Pending |

---

## Priority 6 - Multi-Illumination Fusion Mode

| # | Task | Status |
|---|------|--------|
| 6.1 | Fusion result must show large in the center | Pending |
| 6.2 | Input lighting images should appear as small selectable thumbnails | Pending |
| 6.3 | RGB channel assignment must be visual and easy to understand | Pending |
| 6.4 | Add presets for brightfield, darkfield left/right, coaxial, mixed defect reveal | Pending |
| 6.5 | Add live difference/min/max/average/PCA fusion preview | Pending |
| 6.6 | Add alignment check and warning for misaligned lighting images | Pending |
| 6.7 | Add export fused result at original resolution | Pending |

---

## Priority 7 - Acquisition / Basler Camera Mode

| # | Task | Status |
|---|------|--------|
| 7.1 | Add Basler camera discovery using pypylon | Pending |
| 7.2 | Add live image in center with high FPS display | Pending |
| 7.3 | Add exposure, gain, gamma, trigger, pixel format controls | Pending |
| 7.4 | Add live histogram and clipping warning | Pending |
| 7.5 | Add live focus score and focus trend | Pending |
| 7.6 | Add ROI focus during live camera view | Pending |
| 7.7 | Add snapshot capture | Pending |
| 7.8 | Add burst capture and auto-select sharpest frame | Pending |
| 7.9 | Add camera setting presets per camera serial number | Pending |
| 7.10 | Add folder watcher for camera-saved images | Pending |

---

## Priority 8 - Measurement / Annotation Mode

| # | Task | Status |
|---|------|--------|
| 8.1 | Add rectangle ROI | Pending |
| 8.2 | Add line measurement in pixels | Pending |
| 8.3 | Add point marker with pixel value | Pending |
| 8.4 | Add defect tags: scratch, particle, crack, dent, stain, unknown | Pending |
| 8.5 | Add area statistics for ROI: min, max, mean, std, median | Pending |
| 8.6 | Add intensity profile along line | Pending |
| 8.7 | Save annotations as sidecar JSON | Pending |
| 8.8 | Toggle annotation overlay on/off | Pending |

---

## Priority 9 - Review, Batch, Export

| # | Task | Status |
|---|------|--------|
| 9.1 | Review Mode with image table, thumbnails, focus, quality, verdict | Pending |
| 9.2 | Sort/filter by focus, quality, filename, date, verdict | Pending |
| 9.3 | Batch process folder with selected pipeline | Pending |
| 9.4 | Export processed images at original resolution | Pending |
| 9.5 | Export CSV report of metrics | Pending |
| 9.6 | Export PDF/HTML report with focus heatmap and key images | Pending |
| 9.7 | Export comparison screenshot/report | Pending |
| 9.8 | Add clear progress and cancellation for long batch jobs | Pending |

---

## Priority 10 - Performance / Reliability

| # | Task | Status |
|---|------|--------|
| 10.1 | Profile large 10K image open, zoom, pan, and filter performance | Pending |
| 10.2 | Avoid reprocessing full image on every tiny UI change when possible | Pending |
| 10.3 | Add background workers for heavy filters and batch jobs | Pending |
| 10.4 | Add cancellation for image loading/analysis if user opens another image | Pending |
| 10.5 | Keep UI responsive during thumbnail loading and analysis | Pending |
| 10.6 | Add visible error messages instead of silent exception swallowing | Pending |
| 10.7 | Add real test images for 8-bit, 16-bit, RGB, grayscale, tiny, huge | Pending |
| 10.8 | Add smoke tests for open image, next/previous, focus, compare, save | Pending |

---

## Priority 11 - Packaging

| # | Task | Status |
|---|------|--------|
| 11.1 | Build Windows exe with PyInstaller | Pending |
| 11.2 | Build installer with app icon and file associations | Pending |
| 11.3 | Ensure opening image from Windows Explorer loads folder context | Pending |
| 11.4 | Add sample images and quick-start guide | Pending |
| 11.5 | Test on clean Windows machine without dev environment | Pending |

---

## Immediate Next Work Order

Do these one by one. Do not jump ahead.

| Order | Task |
|---|------|
| 1 | Fix open-with/single-image launch so folder navigation works |
| 2 | Redesign default layout so center image is large and pipeline does not waste space |
| 3 | Build mode switcher and Inspect Mode baseline |
| 4 | Convert Compare into true center Compare Mode |
| 5 | Build Focus Mode with large heatmap overlay and ROI focus |
| 6 | Clean Tune/Filter Mode so filter output stays large and smooth |
| 7 | Add Acquisition/Basler Mode after image/focus/compare workflow is solid |
