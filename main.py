# ============================================
# main.py
# Minimal entrypoint equivalent to notebook run_all()
# ============================================

from pathlib import Path
import asyncio

from critifusion import (
    init_sdxl,
    generate_variants,
    visualize_variants,
)

OUT_DIR = Path("./variants_demo4")
ENABLED_VARIANTS = [
    "base_original",
    "base_multi_llm",
    "criticore_on_original__specfusion",
    "criticore_on_multi_llm__specfusion",
]
SEED = 2026
H, W = 1024, 1024
PRESET = "hq_preference"

PROMPTS = [
    "A fluffy orange cat lying on a window ledge, front-facing, stylized in 3D Pixar look, soft indoor lighting",
]


async def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(PROMPTS, 1):
        dst = OUT_DIR / f"prompt_{i:02d}"
        print(f"\n===== Running Prompt {i} / {len(PROMPTS)} =====")
        print("Prompt:", prompt)

        results = await generate_variants(
            user_prompt=prompt,
            seed=SEED,
            H=H,
            W=W,
            total_steps_refine=50,
            last_k_list=(37,),
            guidance_list=None,
            preset=PRESET,
            out_dir=dst,
            enabled_variants=ENABLED_VARIANTS,
        )
        visualize_variants(results, dpi=140, title=None)

    print("\nAll prompts finished. Outputs saved under:", OUT_DIR.resolve())


if __name__ == "__main__":
    init_sdxl()
    asyncio.run(run_all())
