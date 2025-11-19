# ============================================
# critifusion/diffusion/sdxl.py
# SDXL initialization + latent sampling utilities
# (Fully equivalent to your notebook Cell 0)
# ============================================

from pathlib import Path
import os, math, re, json
from typing import List, Dict, Optional, Tuple

import torch
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

# -----------------------------
# Device / dtype setup
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Global pipelines
SDXL_base = None
SDXL_i2i = None

# Default negative prompt
DEFAULT_NEG = (
    "blurry, low quality, artifacts, watermark, extra fingers, missing limbs, "
    "over-sharpened, harsh lighting, oversaturated"
)

# ---------------------------------------------------------
# Initialize SDXL
# ---------------------------------------------------------
def init_sdxl():
    """
    Initialize SDXL base and img2img pipelines (global variables).
    Same behavior as notebook Cell 0.
    """
    global SDXL_base, SDXL_i2i

    print(f"[Init] DEVICE={DEVICE} DTYPE={DTYPE} SDXL_ID={SDXL_ID}")

    SDXL_base = StableDiffusionXLPipeline.from_pretrained(
        SDXL_ID, torch_dtype=DTYPE
    ).to(DEVICE)

    SDXL_i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_ID, torch_dtype=DTYPE
    ).to(DEVICE)

    for p in (SDXL_base, SDXL_i2i):
        try:
            p.enable_vae_slicing()
            p.enable_attention_slicing()
        except Exception:
            pass
        p.scheduler = DPMSolverMultistepScheduler.from_config(
            p.scheduler.config,
            use_karras_sigmas=True
        )

    return SDXL_base, SDXL_i2i

# ---------------------------------------------------------
# Decode with VAE
# ---------------------------------------------------------
@torch.no_grad()
def decode_image_sdxl(latents: torch.Tensor, pipe: StableDiffusionXLImg2ImgPipeline, output_type="pil"):
    """
    Exact copy of your notebook implementation, with VAE upcast handling.
    """
    vae = pipe.vae
    needs_upcast = (
        (vae.dtype in (torch.float16, torch.bfloat16))
        and bool(getattr(vae.config, "force_upcast", False))
    )

    if needs_upcast:
        try:
            pipe.upcast_vae()
        except Exception:
            pipe.vae = pipe.vae.to(torch.float32)
        vae = pipe.vae

    lat = latents.to(device=vae.device, dtype=(next(vae.post_quant_conv.parameters()).dtype))
    lat = lat / vae.config.scaling_factor
    out = vae.decode(lat)

    x = (
        out[0]
        if isinstance(out, (list, tuple))
        else (out.sample if hasattr(out, "sample") else out)
    )

    if getattr(pipe, "watermark", None) is not None:
        x = pipe.watermark.apply_watermark(x)

    return pipe.image_processor.postprocess(x.detach(), output_type=output_type)[0]

# ---------------------------------------------------------
# Base SDXL latent sampling
# ---------------------------------------------------------
@torch.no_grad()
def base_sample_latent(
    prompt: str,
    seed: int = 2025,
    H: int = 1024,
    W: int = 1024,
    neg: str = ""
):
    """
    Exact replica of notebook behavior:
    SDXL_base(prompt → latent) + decode via SDXL_i2i.
    """
    assert SDXL_base is not None and SDXL_i2i is not None, \
        "You must call init_sdxl() before using sampling functions."

    g = torch.Generator(device=DEVICE).manual_seed(int(seed))

    out = SDXL_base(
        prompt=prompt,
        negative_prompt=neg,
        height=H,
        width=W,
        guidance_scale=4.5,
        num_inference_steps=50,
        generator=g,
        output_type="latent"
    )

    z0 = out.images
    x0 = decode_image_sdxl(z0, SDXL_i2i)
    return z0, x0

# ---------------------------------------------------------
# Latent-space img2img
# ---------------------------------------------------------
@torch.no_grad()
def img2img_latent(prompt: str, image_or_latent, strength: float,
                   guidance: float, steps: int, seed: int):
    """
    Fully equivalent to notebook Cell 0 img2img_latent().
    """
    assert SDXL_i2i is not None, "Call init_sdxl() first."

    g = torch.Generator(device=DEVICE).manual_seed(int(seed))

    out = SDXL_i2i(
        prompt=prompt,
        image=image_or_latent,
        strength=float(strength),
        guidance_scale=float(guidance),
        num_inference_steps=int(steps),
        generator=g,
        output_type="latent",
        negative_prompt=DEFAULT_NEG
    )
    return out.images

# ---------------------------------------------------------
# Convert last_k into img2img strength
# ---------------------------------------------------------
def strength_for_last_k(k: int, total_steps: int) -> float:
    """
    Same implementation as notebook.
    """
    k = max(1, int(k))
    return min(0.95, max(0.01, float(k) / float(max(1, total_steps))))


# === Export symbols ===
__all__ = [
    "init_sdxl",
    "SDXL_base",
    "SDXL_i2i",
    "DEFAULT_NEG",
    "decode_image_sdxl",
    "base_sample_latent",
    "img2img_latent",
    "strength_for_last_k",
    "DEVICE",
    "DTYPE"
]
