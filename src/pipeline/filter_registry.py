"""Central registry of all available filters."""

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

# Map class name → class (used for deserialization)
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

# Organized by category for the UI "Add Filter" menu
FILTER_CATEGORIES = {}
for cls in FILTER_REGISTRY.values():
    cat = cls.CATEGORY
    if cat not in FILTER_CATEGORIES:
        FILTER_CATEGORIES[cat] = []
    FILTER_CATEGORIES[cat].append(cls)
