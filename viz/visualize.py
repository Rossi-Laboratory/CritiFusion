# ============================================
# critifusion/viz/visualize.py
# Visualization utilities for variant grids
# (Equivalent to notebook visualize_variants)
# ============================================

from __future__ import annotations
from typing import Dict, Optional

import matplotlib.pyplot as plt
from PIL import Image

from critifusion.pipeline.pipeline_variants import VARIANT_ORDER


def visualize_variants(
    results: Dict[str, Dict[int, Image.Image]],
    dpi: int = 140,
    title: Optional[str] = None,  # intentionally unused as in notebook
):
    """
    Render all variants in a single row; each variant displays one image at key=0.
    The figure's suptitle is intentionally not set with the prompt.
    """
    row_names = [name for name in VARIANT_ORDER if name in results]
    if not row_names:
        print("[visualize_variants] no images to show")
        return

    ncols = len(row_names)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.4), dpi=dpi)

    if ncols == 1:
        axes = [axes]

    for c, vname in enumerate(row_names):
        ax = axes[c]
        im = results[vname].get(0, None)
        if im is not None:
            ax.imshow(im)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.axis("off")
        ax.set_title(vname)

    fig.tight_layout()
    plt.show()


__all__ = [
    "visualize_variants",
]
