# ============================================
# critifusion/fusion/specfusion.py
# SpecFusion: CADR mapping + frequency fusion + final_touch
# (Equivalent to notebook Cell 3/5)
# ============================================

from __future__ import annotations
from typing import Tuple, Dict, Optional

import torch
from PIL import Image

from critifusion.diffusion.sdxl import (
    SDXL_i2i,
    decode_image_sdxl,
    DEVICE,
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class SpecFusion:
    """
    CADR-based parameter mapping + frequency-domain fusion + final refinement.
    Fully equivalent to your notebook class SpecFusion.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE

    @staticmethod
    def cadr_from_alignment(align_score: float) -> Tuple[float, float, int, float]:
        """
        alignment: 0..100
        Lower scores → stronger correction (larger strength/guidance/steps; higher rho_t).
        Returns: (strength, guidance, steps, rho_t)
        """
        s = _clamp01(align_score / 100.0)
        mis = 1.0 - s

        strength = _lerp(0.12, 0.30, mis)
        guidance = _lerp(3.6, 5.0, mis)
        steps = int(round(_lerp(16, 30, mis)))
        rho_t = _lerp(0.60, 0.85, mis)
        return strength, guidance, steps, rho_t

    @torch.no_grad()
    def frequency_fusion(
        self,
        x_hi_latent: torch.Tensor,
        x_lo_latent: torch.Tensor,
        base_c: float = 0.5,
        rho_t: float = 0.8,
    ) -> torch.Tensor:
        """
        Low freq from x_lo, high freq from x_hi, with central rectangle scaled by rho_t.
        Exactly mirrors your notebook implementation.
        """
        B, C, H, W = x_hi_latent.shape

        x_h = x_hi_latent.to(torch.float32)
        x_l = x_lo_latent.to(torch.float32)

        Xh = torch.fft.fftshift(torch.fft.fftn(x_h, dim=(-2, -1)), dim=(-2, -1))
        Xl = torch.fft.fftshift(torch.fft.fftn(x_l, dim=(-2, -1)), dim=(-2, -1))

        tau_h = int(H * base_c * (1 - rho_t))
        tau_w = int(W * base_c * (1 - rho_t))

        mask = torch.ones((B, C, H, W), device=self.device, dtype=torch.float32)
        cy, cx = H // 2, W // 2
        mask[..., cy - tau_h:cy + tau_h, cx - tau_w:cx + tau_w] = rho_t

        Xf = Xh * mask + Xl * (1 - mask)
        x = torch.fft.ifftn(torch.fft.ifftshift(Xf, dim=(-2, -1)), dim=(-2, -1)).real
        x += torch.randn_like(x) * 0.001

        return x.to(dtype=x_hi_latent.dtype)

    @torch.no_grad()
    def final_touch(
        self,
        enhanced_prompt: str,
        base_latent: torch.Tensor,
        align_score: Optional[float] = None,
        seed: int = 2025,
    ) -> Tuple[torch.Tensor, Image.Image, Dict[str, float]]:
        """
        Apply CADR mapping → SDXL img2img (latent) → frequency fusion → decode.
        As in your notebook, align_score must be provided externally.
        """
        if align_score is None:
            raise ValueError(
                "SpecFusion.final_touch requires align_score "
                "(recommended from pref_score or external alignment)."
            )

        strength, guidance, steps, rho_t = self.cadr_from_alignment(float(align_score))

        g = torch.Generator(device=self.device).manual_seed(seed)

        out = SDXL_i2i(
            prompt=enhanced_prompt,
            image=base_latent,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g,
            output_type="latent",
        )
        lat_refined = out.images

        fused = self.frequency_fusion(
            x_hi_latent=lat_refined,
            x_lo_latent=base_latent,
            base_c=0.5,
            rho_t=rho_t,
        )

        decoded = decode_image_sdxl(fused, SDXL_i2i)
        img = decoded.images[0] if hasattr(decoded, "images") else decoded

        return fused, img, dict(
            strength=strength,
            guidance=guidance,
            steps=steps,
            rho_t=rho_t,
        )


# Convenience global wrapper to match notebook comments
@torch.no_grad()
def frequency_fusion(
    x_hi_latent: torch.Tensor,
    x_lo_latent: torch.Tensor,
    base_c: float = 0.5,
    rho_t: float = 0.8,
    device=None,
) -> torch.Tensor:
    """
    Wrapper for SpecFusion().frequency_fusion(...) so that code which calls
    a free function `frequency_fusion(...)` (as in your variants cell) still works.
    """
    sf = SpecFusion(device=device or DEVICE)
    return sf.frequency_fusion(
        x_hi_latent=x_hi_latent,
        x_lo_latent=x_lo_latent,
        base_c=base_c,
        rho_t=rho_t,
    )


__all__ = [
    "SpecFusion",
    "frequency_fusion",
]
