# ============================================
# critifusion/prompts/style_hints.py
# Lightweight automatic style hints
# (Equivalent to notebook Cell 1 STYLE_RULES / _auto_style_hints)
# ============================================

from __future__ import annotations
from typing import List, Tuple
import re

# (pattern, [tags...])
STYLE_RULES: List[Tuple[str, List[str]]] = [
    (r"\bphoto(graph|real| realistic)?\b|\bportrait\b|\brealistic\b",
     ["photorealistic", "dslr look"]),
    (r"\bpainting|oil|canvas|brush|painterly\b",
     ["painterly", "brush strokes"]),
    (r"\bposter|cinematic|film|movie\b",
     ["cinematic composition", "matte painting"]),
    (r"\banime|manga|chibi|kawaii\b",
     ["clean line art", "illustration"]),
    (r"\bwatercolor\b",
     ["watercolor"]),
    (r"\bcyberpunk|neon\b",
     ["neon lights", "cyberpunk"]),
]


def auto_style_hints(user_prompt: str) -> List[str]:
    """
    Same behavior as your notebook _auto_style_hints():
    return at most 2 soft style hints based on regex matches.
    """
    hints: List[str] = []
    up = user_prompt.lower()
    for pattern, tags in STYLE_RULES:
        if re.search(pattern, up):
            for t in tags:
                if len(hints) < 2 and t not in hints:
                    hints.append(t)
    return hints


# For backward compatibility with notebook naming
_auto_style_hints = auto_style_hints

__all__ = [
    "STYLE_RULES",
    "auto_style_hints",
    "_auto_style_hints",
]
