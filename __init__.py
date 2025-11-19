# ============================================
# critifusion/__init__.py
# Convenience re-exports for the CritiFusion package
# ============================================

from .diffusion.sdxl import (
    init_sdxl,
    SDXL_base,
    SDXL_i2i,
    DEFAULT_NEG,
    base_sample_latent,
    img2img_latent,
    strength_for_last_k,
    decode_image_sdxl,
    DEVICE,
    DTYPE,
)
from .critique.criticore import CritiCore
from .fusion.specfusion import SpecFusion, frequency_fusion
from .pipeline.pipeline_variants import (
    generate_variants,
    VARIANT_LABELS,
    VARIANT_ORDER,
    VARIANT_LEGEND,
    print_variant_legend,
)
from .viz.visualize import visualize_variants

__all__ = [
    "init_sdxl",
    "SDXL_base",
    "SDXL_i2i",
    "DEFAULT_NEG",
    "base_sample_latent",
    "img2img_latent",
    "strength_for_last_k",
    "decode_image_sdxl",
    "DEVICE",
    "DTYPE",
    "CritiCore",
    "SpecFusion",
    "frequency_fusion",
    "generate_variants",
    "VARIANT_LABELS",
    "VARIANT_ORDER",
    "VARIANT_LEGEND",
    "print_variant_legend",
    "visualize_variants",
]
