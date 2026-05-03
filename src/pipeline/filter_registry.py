"""Central registry — all filters, ordered workflow categories, and expert presets."""

from src.filters.contrast_filters import (
    BrightnessContrastFilter, GammaFilter, CLAHEFilter,
    HistogramEqualizationFilter, LevelsFilter, NormalizeFilter, InvertFilter,
)
from src.filters.edge_filters import (
    UnsharpMaskFilter, CannyEdgeFilter, SobelEdgeFilter,
    LaplacianSharpenFilter, DoGFilter, MorphGradientFilter,
)
from src.filters.color_filters import (
    FalseColorFilter, ClippingHighlightFilter,
    ChannelSplitFilter, ChannelMixerFilter,
)
from src.filters.noise_filters import (
    GaussianBlurFilter, MedianFilter, BilateralFilter, NLMeansFilter,
    TopHatFilter, BlackHatFilter, MorphOpenFilter, MorphCloseFilter,
)
from src.filters.advanced_filters import (
    GaborBankFilter, WaveletFilter, FFTMagnitudeFilter, LBPTextureFilter,
)

# ── Class name → class (for pipeline serialization/deserialization) ─────────
FILTER_REGISTRY = {
    cls.__name__: cls for cls in [
        BrightnessContrastFilter, GammaFilter, CLAHEFilter,
        HistogramEqualizationFilter, LevelsFilter, NormalizeFilter, InvertFilter,
        UnsharpMaskFilter, CannyEdgeFilter, SobelEdgeFilter,
        LaplacianSharpenFilter, DoGFilter, MorphGradientFilter,
        FalseColorFilter, ClippingHighlightFilter,
        ChannelSplitFilter, ChannelMixerFilter,
        GaussianBlurFilter, MedianFilter, BilateralFilter, NLMeansFilter,
        TopHatFilter, BlackHatFilter, MorphOpenFilter, MorphCloseFilter,
        GaborBankFilter, WaveletFilter, FFTMagnitudeFilter, LBPTextureFilter,
    ]
}

# ── Workflow steps — display order and description ──────────────────────────
WORKFLOW_STEPS = [
    ("① Pre-process", "Remove noise and correct exposure. Always start here."),
    ("② Enhance",     "Increase defect visibility: contrast, sharpening, morphological enhancement."),
    ("③ Detect",      "Extract the defect signal: edges, texture, frequency analysis."),
    ("④ Visualize",   "Map results to color and isolate channels for clear presentation."),
]

# Step color (left-border accent on each filter card)
STEP_COLORS = {
    "① Pre-process": "#1E6BA8",   # blue
    "② Enhance":     "#D4840A",   # amber
    "③ Detect":      "#1E8A4C",   # green
    "④ Visualize":   "#8A1E8A",   # purple
}

# ── Ordered categories (manually placed for correct step order) ─────────────
_ALL = list(FILTER_REGISTRY.values())

FILTER_CATEGORIES = {
    "① Pre-process": [
        GaussianBlurFilter, MedianFilter, BilateralFilter, NLMeansFilter,
        NormalizeFilter, BrightnessContrastFilter,
    ],
    "② Enhance": [
        CLAHEFilter, HistogramEqualizationFilter, GammaFilter, LevelsFilter,
        UnsharpMaskFilter, LaplacianSharpenFilter,
        TopHatFilter, BlackHatFilter,
    ],
    "③ Detect": [
        CannyEdgeFilter, SobelEdgeFilter, DoGFilter, MorphGradientFilter,
        MorphOpenFilter, MorphCloseFilter,
        GaborBankFilter, WaveletFilter, FFTMagnitudeFilter, LBPTextureFilter,
    ],
    "④ Visualize": [
        FalseColorFilter, ClippingHighlightFilter,
        ChannelSplitFilter, ChannelMixerFilter, InvertFilter,
    ],
}

# ── Expert presets — one-click starting points for common inspection tasks ──
# Each preset adds its filters in order (top → bottom of pipeline).
PRESETS = {
    "Surface Scratch": {
        "icon":    "🔧",
        "tip":     (
            "Detect scratches on metal, glass, or plastic surfaces.\n"
            "Adds: Bilateral Filter → CLAHE → Canny Edge\n\n"
            "Bilateral removes noise while keeping edges sharp.\n"
            "CLAHE boosts local contrast to reveal low-contrast scratches.\n"
            "Canny finds the precise scratch boundaries."
        ),
        "filters": ["Bilateral Filter", "CLAHE", "Canny Edge"],
    },
    "Dark Pits / Voids": {
        "icon":    "⬛",
        "tip":     (
            "Find dark pits, voids, or holes on bright surfaces.\n"
            "Adds: Gaussian Blur → Black-Hat → False Color (LUT)\n\n"
            "Gaussian suppresses noise before morphological processing.\n"
            "Black-Hat isolates features darker than the background.\n"
            "False Color maps intensity to HOT colormap for visual clarity."
        ),
        "filters": ["Gaussian Blur", "Black-Hat", "False Color (LUT)"],
    },
    "Bright Protrusions": {
        "icon":    "⬜",
        "tip":     (
            "Find bright spots, burrs, or raised material on dark surfaces.\n"
            "Adds: Gaussian Blur → Top-Hat → False Color (LUT)\n\n"
            "Gaussian suppresses noise before morphological processing.\n"
            "Top-Hat isolates features brighter than the background.\n"
            "False Color maps intensity to HOT colormap for visual clarity."
        ),
        "filters": ["Gaussian Blur", "Top-Hat", "False Color (LUT)"],
    },
    "PCB / Trace": {
        "icon":    "📋",
        "tip":     (
            "Inspect PCB traces, solder joints, and component placement.\n"
            "Adds: NL-Means Denoise → Levels → Sobel Edge\n\n"
            "NL-Means gives the highest quality denoising for fine PCB detail.\n"
            "Levels corrects exposure and clips dark/bright extremes.\n"
            "Sobel Edge reveals trace boundaries and missing connections."
        ),
        "filters": ["NL-Means Denoise", "Levels", "Sobel Edge"],
    },
    "Texture Defect": {
        "icon":    "⧉",
        "tip":     (
            "Detect texture defects in woven, machined, or patterned surfaces.\n"
            "Adds: Gaussian Blur → Gabor Filter Bank → Normalize\n\n"
            "Gaussian reduces noise that would confuse frequency analysis.\n"
            "Gabor Bank detects directional texture anomalies at all angles.\n"
            "Normalize stretches the response for maximum visual contrast."
        ),
        "filters": ["Gaussian Blur", "Gabor Filter Bank", "Normalize"],
    },
    "Low Contrast Fix": {
        "icon":    "☀",
        "tip":     (
            "Enhance poorly-lit or low-contrast inspection images.\n"
            "Adds: CLAHE → Gamma Correction → Unsharp Mask\n\n"
            "CLAHE boosts local contrast adaptively without over-enhancing noise.\n"
            "Gamma lifts shadow detail without blowing out highlights.\n"
            "Unsharp Mask sharpens edges after the contrast enhancement."
        ),
        "filters": ["CLAHE", "Gamma Correction", "Unsharp Mask"],
    },
}
