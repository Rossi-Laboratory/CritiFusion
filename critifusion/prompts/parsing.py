# ============================================
# critifusion/prompts/parsing.py
# Tag splitting, ordering, cleanup utilities
# (Fully equivalent to notebook Cell 1)
# ============================================

from __future__ import annotations
from typing import List, Tuple
import re

# ----------------------------------------------
# Basic splitting
# ----------------------------------------------

def split_tags(s: str) -> List[str]:
    """
    Exact equivalent of your notebook _split_tags().
    Split by comma or newline.
    """
    return [p.strip() for p in re.split(r",|\n", s.strip()) if p.strip()]


def split_tags_relaxed(s: str) -> List[str]:
    """
    Equivalent to your notebook _split_tags_relaxed().
    Allows repeated delimiters.
    """
    if not s:
        return []
    return [p.strip() for p in re.split(r"[,\n]+", s) if p.strip()]

# ----------------------------------------------
# Deduplicate while preserving order
# ----------------------------------------------

def dedup_keep_order(items: List[str]) -> List[str]:
    """
    Same logic as your notebook _dedup_keep_order().
    Key = normalized lowercase strip.
    """
    seen = set()
    out = []

    for t in items:
        key = re.sub(r"\s+", " ", t.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())

    return out

# ----------------------------------------------
# Tag category ordering (subject → style → comp → lighting → color → detail → other)
# ----------------------------------------------

def order_tags(subject_first: List[str], rest: List[str]) -> List[str]:
    """
    Direct translation of your notebook logic.
    Tag buckets are determined by keyword heuristics.
    """

    buckets = {
        "subject": [],
        "style": [],
        "composition": [],
        "lighting": [],
        "color": [],
        "detail": [],
        "other": [],
    }

    style_kw = (
        "style", "painterly", "illustration",
        "photorealistic", "neon", "poster",
        "matte painting", "watercolor", "cyberpunk"
    )

    comp_kw = (
        "composition", "rule of thirds", "centered",
        "symmetry", "balanced composition"
    )

    light_kw = (
        "lighting", "light", "glow", "glowing", "aura",
        "rim", "sunset", "sunrise", "golden hour",
        "global illumination", "cinematic"
    )

    color_kw = (
        "color", "palette", "vibrant", "muted",
        "monochrome", "pastel", "warm", "cool",
        "balanced contrast"
    )

    detail_kw = (
        "detailed", "hyperdetailed", "texture",
        "textured", "intricate", "high detail",
        "highly detailed", "sharp focus",
        "uhd", "8k"
    )

    # 1. Insert subjects exactly as your code did
    for t in subject_first:
        buckets["subject"].append(t)

    # 2. Categorize the rest
    for t in rest:
        lt = t.lower()
        if any(k in lt for k in style_kw):
            buckets["style"].append(t)
        elif any(k in lt for k in comp_kw):
            buckets["composition"].append(t)
        elif any(k in lt for k in light_kw):
            buckets["lighting"].append(t)
        elif any(k in lt for k in color_kw):
            buckets["color"].append(t)
        elif any(k in lt for k in detail_kw):
            buckets["detail"].append(t)
        else:
            buckets["other"].append(t)

    # 3. Preserve the exact output order same as notebook
    return (
        buckets["subject"]
        + buckets["style"]
        + buckets["composition"]
        + buckets["lighting"]
        + buckets["color"]
        + buckets["detail"]
        + buckets["other"]
    )

# ----------------------------------------------
# Cleaning commas
# ----------------------------------------------

def cleanup_commas(s: str) -> str:
    """
    Exact match to your notebook _cleanup_commas().
    """
    s = re.sub(r"\s*,\s*", ", ", s.strip())
    s = re.sub(r"(,\s*){2,}", ", ", s)
    return s.strip(" ,")

# ----------------------------------------------
# Exports
# ----------------------------------------------

__all__ = [
    "split_tags",
    "split_tags_relaxed",
    "dedup_keep_order",
    "order_tags",
    "cleanup_commas",
]
