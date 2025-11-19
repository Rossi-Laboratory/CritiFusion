# ============================================
# critifusion/prompts/tokenizer.py
# CLIP-77 safe truncation + token counting
# (Fully equivalent to your notebook Cell 0/1)
# ============================================

import re
from typing import Optional

# We attempt to load CLIPTokenizerFast (same as your notebook).
try:
    from transformers import CLIPTokenizerFast
    _clip_tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

    def count_tokens(text: str) -> int:
        """
        Exact token-count logic from notebook:
        uses CLIPTokenizerFast when available.
        """
        return len(_clip_tok(
            text,
            add_special_tokens=True,
            truncation=False
        )["input_ids"])

except Exception:
    # Fallback approximation (your notebook's regex-based fallback)
    _clip_tok = None

    def count_tokens(text: str) -> int:
        """
        Fallback approximate token count.
        Mirrors your notebook behavior.
        """
        return int(len(re.findall(r"\w+", text)) * 1.3)

# ---------------------------------------------------------
# Utility cleanup
# ---------------------------------------------------------
def cleanup_commas(s: str) -> str:
    """
    Normalizes comma spacing, identical to your notebook version.
    """
    s = re.sub(r"\s*,\s*", ", ", s.strip())
    s = re.sub(r",\s*,+", ", ", s)
    return s.strip(" ,")

# ---------------------------------------------------------
# Truncate to CLIP-77 tokens
# ---------------------------------------------------------
def clip77_strict(text: str, max_tok: int = 77) -> str:
    """
    Fully equivalent to your notebook implementation:
    binary-search truncation until token count <= 77.
    """
    if count_tokens(text) <= max_tok:
        return text.strip()

    words = text.strip().split()
    lo, hi, best = 0, len(words), ""

    while lo <= hi:
        mid = (lo + hi) // 2
        cand = " ".join(words[:mid]) if mid > 0 else ""
        if count_tokens(cand) <= max_tok:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1

    return best.strip()

# ---------------------------------------------------------
# Exports
# ---------------------------------------------------------
__all__ = [
    "clip77_strict",
    "count_tokens",
    "cleanup_commas",
]
