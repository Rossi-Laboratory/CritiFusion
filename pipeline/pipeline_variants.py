# ============================================
# critifusion/pipeline/pipeline_variants.py
# Variant generation (base + CritiCore + SpecFusion)
# (Equivalent to notebook bottom part)
# ============================================

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable, Set

from pathlib import Path
from PIL import Image
import asyncio

from critifusion.diffusion.sdxl import (
    base_sample_latent,
    img2img_latent,
    strength_for_last_k,
    DEFAULT_NEG,
    SDXL_i2i,
    DEVICE,
)
from critifusion.critique.criticore import CritiCore
from critifusion.fusion.specfusion import frequency_fusion
from critifusion.prompts.tokenizer import clip77_strict
from critifusion.diffusion.sdxl import decode_image_sdxl

# (pref_score is external; we keep a stub so code runs and falls back to default)
try:
    from critifusion.evaluation.preference_scorer import pref_score  # optional
except Exception:
    def pref_score(prompt, image):
        raise RuntimeError("pref_score not available.")


# ───────── Variant names / display order / legend ─────────
VARIANT_LABELS = {
    "base_multi_llm":                     "1_base_multi_llm",
    "criticore_on_multi_llm__specfusion": "2_criticore_on_multi_llm__specfusion",
    "base_original":                      "3_base_original",
    "criticore_on_original__specfusion":  "4_criticore_on_original__specfusion",
}

VARIANT_ORDER = [
    "1_base_multi_llm",
    "2_criticore_on_multi_llm__specfusion",
    "3_base_original",
    "4_criticore_on_original__specfusion",
]

VARIANT_LEGEND = {
    "1_base_multi_llm": "Base with Multi-LLM tags only (no CritiCore / SpecFusion)",
    "2_criticore_on_multi_llm__specfusion": "CritiCore refine on Multi-LLM → img2img_latent → SpecFusion",
    "3_base_original": "Base with original prompt only (no CritiCore / SpecFusion)",
    "4_criticore_on_original__specfusion": "CritiCore refine on original → img2img_latent → SpecFusion",
}


def print_variant_legend():
    print("\n[Variant Legend]")
    for key in VARIANT_ORDER:
        print(f"  {key}: {VARIANT_LEGEND[key]}")


# ───────── Helpers ─────────
RHO_T_DEFAULT = 0.85  # fixed rho_t as in your code


def _normalize_enabled(enabled_variants: Optional[Iterable[str]]) -> Set[str]:
    default = set(VARIANT_LABELS.keys())
    if enabled_variants is None:
        return default
    return set(enabled_variants)


def _guidance_for_k(k: int) -> float:
    """Simple fallback guidance rule based on last_k."""
    if k >= 20:
        return 12.0
    if k >= 10:
        return 7.5
    return 5.2


def _align_score_or_default(prompt: str, image: Image.Image) -> float:
    """Kept for compatibility; not used after removing CADR inside this file."""
    try:
        return float(pref_score(prompt, image) * 100.0)
    except Exception:
        return 60.0


def _decode_to_pil(x, pipe):
    """Compat decoder using decode_image_sdxl()."""
    out = decode_image_sdxl(x, pipe)
    return out.images[0] if hasattr(out, "images") else out


# ───────── Shared materials ─────────
async def _shared_materials(
    user_prompt: str,
    seed: int,
    H: int,
    W: int,
    preset: str,
):
    """
    Returns:
      - pos_tags_77 / neg_tags from Multi-LLM
      - components for VLM
      - base latents/images for original prompt and Multi-LLM tags
      - vlm_agg_77: CritiCore's single refined (≤77) on the Multi-LLM baseline image
    """
    critic = CritiCore(preset=preset)

    pos_tags_77, neg_tags = await critic.make_tags(user_prompt, clip77=True)
    comps = await critic.decompose_components(user_prompt)

    # Base (original prompt)
    z0_og, base_og = base_sample_latent(
        user_prompt,
        seed=seed,
        H=H,
        W=W,
        neg=DEFAULT_NEG,
    )

    # Base (Multi-LLM tags prompt)
    z0_enh, base_enh = base_sample_latent(
        pos_tags_77,
        seed=seed,
        H=H,
        W=W,
        neg=neg_tags,
    )

    # CritiCore on Multi-LLM baseline image
    vlm_out = await critic.vlm_refine(base_enh, pos_tags_77, comps or [])
    vlm_agg_77 = vlm_out.get("refined") or pos_tags_77

    return dict(
        pos_tags_77=pos_tags_77,
        neg_tags=neg_tags,
        comps=comps,
        z0_og=z0_og,
        base_og=base_og,
        z0_enh=z0_enh,
        base_enh=base_enh,
        vlm_agg_77=vlm_agg_77,
        critic=critic,
    )


# ───────── Variants (explicit img2img_latent + frequency_fusion) ─────────
async def generate_variants(
    user_prompt: str,
    seed: int,
    H: int,
    W: int,
    total_steps_refine: int,
    last_k_list: Iterable[int],
    guidance_list: Optional[List[float]] = None,
    preset: str = "hq_preference",
    out_dir: Optional[Path] = None,
    enabled_variants: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[int, Image.Image]]:
    """
    Returns: { variant_name: {0: PIL.Image} }.

    Equivalent to your notebook generate_variants(), but placed in a module.
    """
    enabled = _normalize_enabled(enabled_variants)

    if isinstance(last_k_list, int):
        lk = int(last_k_list)
    else:
        lk = int(list(last_k_list)[-1]) if last_k_list else 36

    shared = await _shared_materials(user_prompt, seed, H, W, preset)

    pos_tags_77 = shared["pos_tags_77"]
    comps = shared["comps"]
    z0_og, base_og = shared["z0_og"], shared["base_og"]
    z0_enh, base_enh = shared["z0_enh"], shared["base_enh"]
    vlm_agg_77 = shared["vlm_agg_77"]
    critic: CritiCore = shared["critic"]

    out: Dict[str, Dict[int, Image.Image]] = {}

    def _save(im: Image.Image, vname: str, k: int = 0):
        if out_dir is None:
            return
        sub = out_dir / f"var_{vname}"
        sub.mkdir(parents=True, exist_ok=True)
        im.save(sub / f"{vname}_k{k}.png")

    print_variant_legend()
    print(
        f"\n[SpecFusion] using last_k={lk}, total_steps_refine={total_steps_refine}, rho_t={RHO_T_DEFAULT}"
    )

    # 1) base_multi_llm
    if "base_multi_llm" in enabled:
        v = VARIANT_LABELS["base_multi_llm"]
        out[v] = {0: base_enh}
        _save(base_enh, v, 0)

    # 2) CritiCore on Multi-LLM → img2img_latent → SpecFusion
    if "criticore_on_multi_llm__specfusion" in enabled:
        v = VARIANT_LABELS["criticore_on_multi_llm__specfusion"]
        out[v] = {}

        refined_on_enh = CritiCore.merge_vlm_multi_text(
            vlm_agg_77,
            pos_tags_77,
        )

        print("\n[Prompts][criticore_on_multi_llm__specfusion]")
        print("Original (Multi-LLM) :", pos_tags_77)
        print("Refined              :", refined_on_enh)

        strength = float(strength_for_last_k(lk, total_steps_refine))
        guidance = float(
            guidance_list[-1]
        ) if guidance_list else float(_guidance_for_k(lk))
        steps = int(total_steps_refine)

        z_ref = img2img_latent(
            refined_on_enh,
            z0_enh,
            strength=strength,
            guidance=guidance,
            steps=steps,
            seed=seed + 2100 + lk,
        )

        fused_lat = frequency_fusion(
            x_hi_latent=z_ref,
            x_lo_latent=z0_enh,
            base_c=0.5,
            rho_t=RHO_T_DEFAULT,
            device=DEVICE,
        )
        img_sf = _decode_to_pil(fused_lat, SDXL_i2i)

        print(
            f"[SpecFusion][multi_llm] last_k={lk} strength={strength:.3f} guidance={guidance:.2f} rho_t={RHO_T_DEFAULT:.2f}"
        )
        out[v][0] = img_sf
        _save(img_sf, v, 0)

    # 3) base_original
    if "base_original" in enabled:
        v = VARIANT_LABELS["base_original"]
        out[v] = {0: base_og}
        _save(base_og, v, 0)

    # 4) CritiCore on Original → img2img_latent → SpecFusion
    if "criticore_on_original__specfusion" in enabled:
        v = VARIANT_LABELS["criticore_on_original__specfusion"]
        out[v] = {}

        vlm_on_og = await critic.vlm_refine(base_og, user_prompt, comps or [])
        refined_og_77 = clip77_strict(
            vlm_on_og.get("refined") or user_prompt, 77
        )
        refined_merge = CritiCore.merge_vlm_multi_text(
            refined_og_77,
            pos_tags_77,
        )

        print("\n[Prompts][criticore_on_original__specfusion]")
        print("Original (user) :", user_prompt)
        print("Refined         :", refined_merge)

        strength = float(strength_for_last_k(lk, total_steps_refine))
        guidance = float(
            guidance_list[-1]
        ) if guidance_list else float(_guidance_for_k(lk))
        steps = int(total_steps_refine)

        z_ref = img2img_latent(
            refined_merge,
            z0_og,
            strength=strength,
            guidance=guidance,
            steps=steps,
            seed=seed + 2400 + lk,
        )

        fused_lat = frequency_fusion(
            x_hi_latent=z_ref,
            x_lo_latent=z0_og,
            base_c=0.5,
            rho_t=RHO_T_DEFAULT,
            device=DEVICE,
        )
        img_sf = _decode_to_pil(fused_lat, SDXL_i2i)

        print(
            f"[SpecFusion][original] last_k={lk} strength={strength:.3f} guidance={guidance:.2f} rho_t={RHO_T_DEFAULT:.2f}"
        )
        out[v][0] = img_sf
        _save(img_sf, v, 0)

    printable = [name for name in VARIANT_ORDER if name in out]
    print("\n[Variants] done:", ", ".join(printable))

    return out


__all__ = [
    "generate_variants",
    "VARIANT_LABELS",
    "VARIANT_ORDER",
    "VARIANT_LEGEND",
    "print_variant_legend",
]
