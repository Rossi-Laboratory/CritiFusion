# ============================================
# critifusion/critique/criticore.py
# Multi-LLM + VLM CritiCore module
# (Equivalent to notebook Cell 2/5)
# ============================================

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import os, re, json, asyncio, inspect, base64, io

from PIL import Image
from together import AsyncTogether
from together.error import InvalidRequestError, RateLimitError

from critifusion.prompts.parsing import (
    split_tags as _split_tags,
    dedup_keep_order as _dedup_keep_order,
    order_tags as _order_tags,
    cleanup_commas as _cleanup_commas,
)
from critifusion.prompts.tokenizer import (
    clip77_strict,
    count_tokens as _count_tokens,
)
from critifusion.prompts.style_hints import _auto_style_hints


# ----------------------------------------------
# Shared small utilities (from notebook Cell 1)
# ----------------------------------------------

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _maybe_close_async_together(client) -> None:
    try:
        if hasattr(client, "aclose") and inspect.iscoroutinefunction(client.aclose):
            await client.aclose()
        elif hasattr(client, "close"):
            fn = client.close
            if inspect.iscoroutinefunction(fn):
                await fn()
            else:
                try:
                    fn()
                except Exception:
                    pass
    except Exception:
        pass


# ----------------------------------------------
# Config from notebook
# ----------------------------------------------

AGGREGATOR_MODEL = os.environ.get(
    "AGGREGATOR_MODEL",
    "Qwen/Qwen2.5-72B-Instruct-Turbo"
)

LLM_MULTI_CANDIDATES: List[str] = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
]

_env_list = [
    s.strip()
    for s in os.environ.get("VLM_MOA_CANDIDATES", "").split(",")
    if s.strip()
]
VLM_CANDIDATES: List[str] = _env_list or [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
]

TAG_PRESETS = {
    "hq_preference": {
        "seed_pos": [
            "balanced composition",
            "natural color palette", "vibrant colors", "balanced contrast",
            "high detail", "highly detailed", "hyperdetailed", "sharp focus",
            "UHD", "8k",
        ],
        "seed_neg": [
            "low quality", "blurry", "watermark", "jpeg artifacts",
            "overexposed", "underexposed", "color banding",
            "extra fingers", "missing limbs", "disfigured", "mutated hands",
        ],
    }
}

_DECOMP_SYS = (
    "Decompose the user's visual instruction into 3-6 concrete, checkable visual components "
    "(entities + interactions + spatial relations). Return ONLY JSON: "
    '{"components":["..."]}'
)

_TXT_SYS = (
    "Expand a VERY SHORT visual idea into a COMMA-SEPARATED TAG LIST for SDXL.\n"
    "Constraints:\n"
    "- Start with the subject phrase first.\n"
    "- Prioritize composition, lighting, color, and detail over style.\n"
    "- Use at most TWO style tags if any.\n"
    "- 16–26 concise tags total. Commas only, no sentences, no 'and'. No trailing period.\n"
    "- Prefer human-preference aesthetics; keep 'high detailed', 'sharp focus', '8k', 'UHD'."
)

def _TAG_RE(tag: str):
    return re.compile(rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>", re.S | re.I)

def _extract_tag(text: str, tag: str, fallback: str = "") -> str:
    s = (text or "").strip()
    r = _TAG_RE(tag)
    m = r.search(s)
    if m:
        return m.group(1).strip()
    s2 = s.replace("&lt;", "<").replace("&gt;", ">")
    m2 = r.search(s2)
    return m2.group(1).strip() if m2 else fallback.strip()

def _summarize_issues_lines(text: str, max_lines: int = 5) -> str:
    if not text:
        return ""
    parts = [p.strip(" -•\t") for p in re.split(r"[\n;]+", text) if p.strip()]
    parts = parts[:max_lines]
    return "\n".join(f"- {p}" for p in parts)


# ----------------------------------------------
# CritiCore main class
# ----------------------------------------------

class CritiCore:
    """
    Multi-LLM + VLM Critique Core, equivalent to notebook Cell 2.
    Depends on: clip77_strict, AsyncTogether (TOGETHER_API_KEY).
    """

    def __init__(self, preset: str = "hq_preference", aggregator_model: str = AGGREGATOR_MODEL):
        if not os.environ.get("TOGETHER_API_KEY"):
            raise RuntimeError("Missing TOGETHER_API_KEY.")
        self.preset = preset
        self.aggregator = aggregator_model

    # ---------- Multi-LLM: component decomposition ----------

    async def decompose_components(self, user_prompt: str) -> List[str]:
        client = AsyncTogether(api_key=os.environ["TOGETHER_API_KEY"])
        try:
            tasks = [
                client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": _DECOMP_SYS},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                    max_tokens=256,
                )
                for m in LLM_MULTI_CANDIDATES
            ]

            rs = await asyncio.gather(*tasks, return_exceptions=True)
            texts = []
            for r in rs:
                try:
                    texts.append(r.choices[0].message.content)
                except Exception:
                    pass

            if not texts:
                return []

            joined = "\n\n---\n\n".join(texts)
            merged = await client.chat.completions.create(
                model=self.aggregator,
                messages=[
                    {
                        "role": "system",
                        "content": "Merge JSON candidates and return ONLY {'components':[...]}."
                    },
                    {"role": "user", "content": joined},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            txt = merged.choices[0].message.content
            try:
                obj = json.loads(txt)
            except Exception:
                s, e = txt.find("{"), txt.rfind("}")
                obj = json.loads(txt[s:e + 1]) if (s != -1 and e != -1) else {"components": []}

            comps = [
                c.strip()
                for c in obj.get("components", [])
                if isinstance(c, str) and c.strip()
            ]
            return comps[:6]
        finally:
            await _maybe_close_async_together(client)

    # ---------- Multi-LLM: positive / negative tag generation ----------

    async def make_tags(self, user_prompt: str, clip77: bool = True) -> Tuple[str, str]:
        client = AsyncTogether(api_key=os.environ["TOGETHER_API_KEY"])

        seed = TAG_PRESETS.get(self.preset, TAG_PRESETS["hq_preference"])
        auto_style = _auto_style_hints(user_prompt)

        seed_pos = _dedup_keep_order(seed["seed_pos"] + auto_style)
        seed_neg = seed["seed_neg"]

        try:
            tasks = [
                client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": _TXT_SYS},
                        {
                            "role": "user",
                            "content": (
                                f"Short idea: {user_prompt}\n"
                                f"Seed: {', '.join(seed_pos)}\n"
                                "Output: a single comma-separated tag list."
                            ),
                        },
                    ],
                    temperature=0.7,
                    max_tokens=220,
                )
                for m in LLM_MULTI_CANDIDATES
            ]

            rs = await asyncio.gather(*tasks, return_exceptions=True)
            props = []
            for r in rs:
                try:
                    props.append(r.choices[0].message.content)
                except Exception:
                    pass

            if not props:
                pos = ", ".join([user_prompt.strip()] + seed_pos)
            else:
                joined = "\n---\n".join(props)
                merged = await client.chat.completions.create(
                    model=self.aggregator,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Merge candidate tag lists into ONE comma list (16–26 tags). "
                                "Subject first; at most TWO style tags; "
                                "keep high detailed/sharp focus/8k/UHD."
                            ),
                        },
                        {"role": "user", "content": joined},
                    ],
                    temperature=0.2,
                    max_tokens=240,
                )
                raw = merged.choices[0].message.content
                tags = _dedup_keep_order(_split_tags(raw))

                subject = user_prompt.strip().rstrip(",.")
                if subject and not any(subject.lower() == t.lower() for t in tags):
                    tags = [subject] + tags

                ordered = _order_tags([tags[0]], tags[1:])
                pos = ", ".join(_dedup_keep_order(ordered))

            QUALITY_FLOOR = ["high detailed", "sharp focus", "8k", "UHD"]
            have = {t.lower() for t in _split_tags(pos)}
            for q in QUALITY_FLOOR:
                if q.lower() not in have:
                    pos += ", " + q

            pos = _cleanup_commas(pos)
            if clip77 and _count_tokens(pos) > 77:
                pos = clip77_strict(pos, 77)

            neg = ", ".join(seed_neg)
            return pos, neg

        finally:
            await _maybe_close_async_together(client)

    # ---------- VLM refinement ----------

    async def vlm_refine(
        self,
        image: Image.Image,
        original_prompt: str,
        components: List[str],
    ) -> Dict[str, Any]:
        client = AsyncTogether(api_key=os.environ["TOGETHER_API_KEY"])
        b64 = pil_to_base64(image, "PNG")

        def _user_prompt_text() -> str:
            return (
                "You are a precise image-grounded critic.\n"
                "1) List concrete visual problems and brief corrections.\n"
                "2) Provide a refined prompt that keeps the original intent.\n\n"
                f'Original prompt: "{original_prompt}"\n'
                f"Key components to check: {components}\n"
                "Output EXACTLY two tags:\n"
                "<issues>...</issues>\n<refined>...</refined>"
            )

        try:
            tasks = []
            for m in VLM_CANDIDATES:
                msgs = [
                    {
                        "role": "system",
                        "content": "Return ONLY <issues> and <refined>. No extra text.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _user_prompt_text()},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    },
                ]
                tasks.append(
                    client.chat.completions.create(
                        model=m,
                        messages=msgs,
                        temperature=0.2,
                        max_tokens=420,
                    )
                )

            rs = await asyncio.gather(*tasks, return_exceptions=True)
            ok = []
            for m, r in zip(VLM_CANDIDATES, rs):
                try:
                    raw = r.choices[0].message.content
                    ok.append((m, raw))
                except Exception:
                    pass

            if not ok:
                return {
                    "refined": original_prompt,
                    "used_models": [],
                    "per_vlm_refined_77": {},
                    "per_vlm_issues": {},
                    "issues_merged": "",
                }

            refined_items: List[Tuple[str, str]] = []
            per_vlm_issues: Dict[str, str] = {}

            for m, raw in ok:
                issues = _extract_tag(raw, "issues", "")
                refined = _extract_tag(raw, "refined", original_prompt)
                if refined.strip():
                    refined_items.append((m, refined.strip()))
                if issues.strip():
                    per_vlm_issues[m] = _summarize_issues_lines(issues, 5)

            per_vlm_refined_77: Dict[str, str] = {}
            for m, txt in refined_items:
                try:
                    per_vlm_refined_77[m] = clip77_strict(txt, 77)
                except Exception:
                    per_vlm_refined_77[m] = original_prompt

            joined_issues = "\n".join(
                f"[{m}] {t}" for m, t in per_vlm_issues.items()
            )
            joined_refined = (
                "\n".join(f"[{m}] {t}" for m, t in refined_items)
                if refined_items
                else original_prompt
            )

            merged = await client.chat.completions.create(
                model=self.aggregator,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Merge multiple critics. Output ONLY <issues> (≤5 bullets) "
                            "and <refined> (≤70 words, no camera/gear/resolution jargon)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{joined_issues}\n\n----\n\n{joined_refined}",
                    },
                ],
                temperature=0.2,
                max_tokens=420,
            )
            final_raw = merged.choices[0].message.content
            final_refined = _extract_tag(final_raw, "refined", original_prompt)
            final_refined = clip77_strict(final_refined, 77)

            issues_merged = _summarize_issues_lines(
                _extract_tag(final_raw, "issues", ""),
                5,
            )

            return {
                "refined": final_refined,
                "used_models": [m for m, _ in refined_items],
                "per_vlm_refined_77": per_vlm_refined_77,
                "per_vlm_issues": per_vlm_issues,
                "issues_merged": issues_merged,
            }

        finally:
            await _maybe_close_async_together(client)

    # ---------- Merge VLM refined text with Multi-LLM tags ----------

    @staticmethod
    def merge_vlm_multi_text(vlm_refined_77: str, tags_77: str) -> str:
        vlm_tags = _split_tags(vlm_refined_77)
        moa_tags = _split_tags(tags_77)

        merged = _dedup_keep_order(
            _order_tags(
                [vlm_tags[0] if vlm_tags else ""],
                (vlm_tags[1:] + moa_tags),
            )
        )
        merged = [t for t in merged if t]

        from critifusion.prompts.parsing import cleanup_commas as _clean
        text = _clean(", ".join(merged))
        if _count_tokens(text) > 77:
            text = clip77_strict(text, 77)
        return text


__all__ = [
    "CritiCore",
    "TAG_PRESETS",
    "LLM_MULTI_CANDIDATES",
    "VLM_CANDIDATES",
    "AGGREGATOR_MODEL",
]
