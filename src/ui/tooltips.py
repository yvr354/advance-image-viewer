"""
Expert-level tooltips for every control.
Written for experienced vision system engineers — not beginners.
Each tooltip explains: what it does, optimal range, and why it matters.
"""

TIP = {

    # ── Focus & Sharpness ─────────────────────────────────────────────────

    "focus_score": (
        "Focus Score (Variance of Laplacian)\n\n"
        "Measures high-frequency energy in the image.\n"
        "Sharp image = many strong edges = high variance.\n\n"
        "Scale: 0 – 1000\n"
        "  > 700   PERFECT  — use for AI training\n"
        "  400–700 GOOD     — acceptable for most tasks\n"
        "  200–400 SOFT     — borderline, check critical regions\n"
        "  < 200   BLURRY   — reject, refocus camera\n\n"
        "IEEE ref: Variance of Laplacian (Pertuz et al. 2013)\n"
        "is the most reliable single-metric focus estimator for\n"
        "industrial machine vision with textured surfaces."
    ),

    "focus_metric_laplacian": (
        "Laplacian Variance\n\n"
        "Applies a Laplacian kernel (edge detector) then measures\n"
        "the variance of the response.\n\n"
        "Best for: General surfaces, mixed textures.\n"
        "Sensitive to: Noise — use on clean images.\n"
        "Speed: Fast — suitable for live view."
    ),

    "focus_metric_tenengrad": (
        "Tenengrad (Sobel Energy)\n\n"
        "Computes Sobel gradient magnitude squared, then sums.\n"
        "More robust to noise than Laplacian variance.\n\n"
        "Best for: Textured surfaces (machined metal, fabric, PCB).\n"
        "More robust than Laplacian on noisy images.\n"
        "Speed: Moderate."
    ),

    "focus_metric_brenner": (
        "Brenner Function\n\n"
        "Measures intensity difference between pixels 2 steps apart.\n"
        "Simple and fast — original autofocus algorithm.\n\n"
        "Best for: Smooth surfaces, flat backgrounds.\n"
        "Not recommended for: Textured surfaces.\n"
        "Speed: Fastest of all three."
    ),

    "focus_grid": (
        "Focus Heatmap Grid Resolution\n\n"
        "Divides image into N×N cells, computes focus score per cell.\n"
        "Higher grid = more spatial detail in heatmap.\n\n"
        "4×4   — quick overview, lowest CPU\n"
        "8×8   — recommended for most use cases\n"
        "16×16 — high spatial resolution, moderate CPU\n"
        "32×32 — maximum detail, use on stopped image only\n\n"
        "Color: Green=sharp, Yellow=soft, Red=blurry"
    ),

    "focus_threshold_perfect": (
        "PERFECT threshold\n\n"
        "Images scoring above this value are considered perfect focus.\n"
        "Default: 700\n\n"
        "Calibrate by:\n"
        "1. Set camera to known perfect focus\n"
        "2. Capture 10 images\n"
        "3. Average score = your PERFECT baseline\n"
        "4. Set threshold to 90% of baseline"
    ),

    # ── Image Quality ─────────────────────────────────────────────────────

    "quality_overall": (
        "Overall Quality Score (0–100)\n\n"
        "Composite metric: Focus + Exposure + SNR + Contrast.\n"
        "Weighted combination — focus is highest weight.\n\n"
        "> 80  PASS — suitable for AI training dataset\n"
        "60–80 PASS — acceptable\n"
        "< 60  FAIL — do not use for AI training\n\n"
        "Rule: 'Good image quality = good AI results'\n"
        "A model trained on blurry or overexposed images\n"
        "will fail on good images and vice versa."
    ),

    "snr": (
        "Signal-to-Noise Ratio (dB)\n\n"
        "Ratio of signal strength to noise floor.\n\n"
        "> 40 dB  Excellent — sensor or gain settings optimal\n"
        "30–40 dB Good\n"
        "20–30 dB Acceptable\n"
        "< 20 dB  Poor — reduce gain, increase exposure\n\n"
        "Common cause of low SNR: too much analog gain.\n"
        "Fix: increase exposure time before increasing gain."
    ),

    "exposure_over": (
        "Overexposed Pixel Percentage\n\n"
        "Pixels at or above saturation threshold (default: 250/255).\n"
        "Overexposed pixels have no information — they are clipped.\n\n"
        "< 0.1%  Acceptable\n"
        "> 1%    Warning — reduce exposure or gain\n"
        "> 5%    Critical — image unusable for defect detection\n\n"
        "Shown as RED overlay in Clipping Highlight filter.\n"
        "Note: Even 1 overexposed pixel on a defect region\n"
        "can cause false negatives in AI detection."
    ),

    "exposure_under": (
        "Underexposed Pixel Percentage\n\n"
        "Pixels at or below the dark threshold (default: 5/255).\n\n"
        "< 0.5%  Acceptable (background shadow)\n"
        "> 2%    Warning — increase exposure\n\n"
        "Underexposed regions hide dark defects.\n"
        "Shown as BLUE overlay in Clipping Highlight filter."
    ),

    "rms_contrast": (
        "RMS Contrast\n\n"
        "Root-mean-square of pixel intensity deviations from mean.\n"
        "Normalized to 0–100%.\n\n"
        "> 15%  Good contrast — defects will be distinguishable\n"
        "8–15%  Moderate — consider CLAHE\n"
        "< 8%   Low — apply CLAHE or increase lighting contrast\n\n"
        "Low contrast is the most common reason AI models miss defects."
    ),

    # ── Filters ───────────────────────────────────────────────────────────

    "clahe_clip": (
        "CLAHE Clip Limit\n\n"
        "Controls maximum slope of the cumulative distribution function.\n"
        "Limits over-amplification of noise in flat regions.\n\n"
        "0.5–1.0  Subtle enhancement — preserves gradients\n"
        "2.0–3.0  Standard — best for defect visualization\n"
        "5.0–8.0  Strong — use for very low contrast images\n"
        "> 8.0    Creates halos and amplifies sensor noise\n\n"
        "IEEE rec: 2.0–4.0 for industrial surface inspection.\n"
        "Start at 2.0 and increase until defect is visible."
    ),

    "clahe_tile": (
        "CLAHE Tile Grid Size\n\n"
        "Divides image into tiles, applies histogram equalization\n"
        "locally within each tile.\n\n"
        "Smaller tile = more local adaptation = more contrast\n"
        "Larger tile = more global = less noise amplification\n\n"
        "4×4   — strong local contrast, risk of halos\n"
        "8×8   — recommended starting point\n"
        "16×16 — gentle, global-like enhancement\n\n"
        "Rule: tile size should be larger than the largest defect."
    ),

    "dog_sigma1": (
        "DoG — Fine Sigma (σ₁)\n\n"
        "Controls the fine-scale Gaussian blur.\n"
        "Difference of Gaussians = bandpass filter in frequency domain.\n\n"
        "Smaller σ₁ = reveals finer surface texture details.\n"
        "Typical range: 0.5 – 2.0 for most surfaces.\n\n"
        "Use DoG to reveal:\n"
        "  - Machining marks and tool lines\n"
        "  - Surface roughness variations\n"
        "  - Micro-scratches invisible to edge detectors"
    ),

    "dog_sigma2": (
        "DoG — Coarse Sigma (σ₂)\n\n"
        "Controls the coarse-scale Gaussian blur.\n"
        "Must always be larger than σ₁.\n\n"
        "Ratio σ₂/σ₁ = bandpass width:\n"
        "  1.5× — narrow band, specific defect size\n"
        "  2×   — standard DoG (σ₂ = 2×σ₁)\n"
        "  4×+  — wide band, broad defect visualization\n\n"
        "IEEE: ratio of 1.6 is perceptually optimal (Marr-Hildreth)."
    ),

    "tophat_ksize": (
        "Top-Hat Kernel Size\n\n"
        "Top-hat transform = image minus morphological opening.\n"
        "Extracts bright features SMALLER than the kernel.\n\n"
        "Set kernel size LARGER than the background texture\n"
        "but SMALLER than the defect you want to ignore.\n\n"
        "Typical use:\n"
        "  Particle detection: kernel = 3–7px (larger than particle)\n"
        "  Scratch detection: elongated kernel along scratch direction\n\n"
        "Black-hat detects DARK features (pits, voids, inclusions)."
    ),

    "gabor_frequency": (
        "Gabor Filter — Frequency\n\n"
        "Spatial frequency of the sinusoidal carrier wave.\n"
        "Controls which feature size the filter responds to.\n\n"
        "0.05 — large features (coarse texture, large defects)\n"
        "0.15 — medium features (standard surface inspection)\n"
        "0.30 — fine features (micro-scratches, fine texture)\n\n"
        "IEEE: Gabor filters are optimal for texture analysis because\n"
        "they achieve the theoretical minimum time-frequency uncertainty\n"
        "(Daugman 1985). Better than Sobel for periodic defects."
    ),

    "gabor_theta": (
        "Gabor Filter — Orientation (θ)\n\n"
        "Angle of the filter in degrees. The filter responds\n"
        "maximally to edges and textures perpendicular to this angle.\n\n"
        "0°    — detects vertical features\n"
        "45°   — detects diagonal features\n"
        "90°   — detects horizontal features\n\n"
        "Use Gabor Bank (all 6 orientations) to detect\n"
        "defects at any orientation without knowing direction.\n\n"
        "Critical for scratches: orientation-specific illumination\n"
        "plus orientation-specific Gabor filter = maximum detection."
    ),

    "wavelet_level": (
        "Wavelet Decomposition Level\n\n"
        "Number of decomposition stages. Each level halves\n"
        "the spatial frequency range analyzed.\n\n"
        "Level 1 — finest detail (scratches, micro-cracks)\n"
        "Level 2 — medium features (most surface defects)\n"
        "Level 3 — coarse features (large dents, blobs)\n"
        "Level 4 — very coarse (background variations)\n\n"
        "IEEE: Level 2–3 captures most industrial surface defects.\n"
        "Combine multiple levels for multi-scale defect analysis."
    ),

    # ── Fusion ────────────────────────────────────────────────────────────

    "fusion_weight": (
        "Channel Weight\n\n"
        "Contribution of this illumination image to its assigned channel.\n\n"
        "1.0  — normal contribution\n"
        "0.5  — half weight (useful when one light is brighter)\n"
        "2.0  — double emphasis (amplify specific defect signal)\n"
        "0.0  — disabled\n\n"
        "Use different weights to compensate for illumination\n"
        "intensity differences between light sources.\n"
        "For equal-power lights, keep all weights at 1.0."
    ),

    "fusion_mode_rgb": (
        "RGB Composite Mode\n\n"
        "Assigns each illumination image to a color channel (R, G, B).\n"
        "Defects that reflect differently under different lights\n"
        "appear as COLOR SHIFTS in the composite.\n\n"
        "Example — scratch detection:\n"
        "  R = Bright field (baseline)\n"
        "  G = Dark field left\n"
        "  B = Dark field right\n"
        "  → Scratch appears cyan (G+B) or magenta (R+B)\n"
        "  depending on orientation.\n\n"
        "Based on: 'Fusion of multi-light source illuminated images\n"
        "for effective defect inspection' — ScienceDirect 2022."
    ),

    # ── 3D Surface ────────────────────────────────────────────────────────

    "3d_z_scale": (
        "Z Scale (Height Exaggeration)\n\n"
        "Multiplies image intensity differences in the Z axis.\n"
        "Exaggerates subtle height variations.\n\n"
        "1×   — true proportion (usually too flat to see)\n"
        "5×   — subtle variations visible\n"
        "20×  — good starting point for surface inspection\n"
        "50×+ — maximum exaggeration for micro-defects\n\n"
        "Note: This is a visualization tool — not a true 3D measurement.\n"
        "For calibrated 3D: use photometric stereo (Phase 3)."
    ),

    "3d_downsample": (
        "Surface Resolution (Downsample Factor)\n\n"
        "Divides image dimensions by this factor for 3D mesh.\n"
        "Controls performance vs. detail tradeoff.\n\n"
        "1×  — full resolution (can be slow for large images)\n"
        "4×  — recommended — good detail, smooth interaction\n"
        "8×  — fast — use for initial exploration\n"
        "16× — overview only\n\n"
        "A 10,000×10,000 image at 4× = 2,500×2,500 = 6.25M vertices.\n"
        "Rendered in real-time on any modern GPU."
    ),

    "3d_colormap": (
        "3D Surface Color Map\n\n"
        "Maps height (pixel intensity) to color.\n\n"
        "Thermal — hot=high, cold=low. Best for defect visualization.\n"
        "Viridis — perceptually uniform, good for publications.\n"
        "Plasma  — high contrast, reveals subtle variations.\n"
        "Grays   — grayscale surface, closest to original image.\n"
        "Cyclic  — repeating colors, reveals periodic patterns."
    ),

    "3d_smooth": (
        "Surface Smoothing\n\n"
        "Applies light Gaussian smoothing before building mesh.\n\n"
        "OFF — raw pixel values as height (shows sensor noise)\n"
        "ON  — smoothed surface (cleaner visualization)\n\n"
        "Recommendation: Turn OFF to inspect noise patterns.\n"
        "Turn ON to see true surface topology."
    ),
}
