# CritiFusion: Semantic and Spectral Refinement for Text-to-Image Diffusion
<div align="center">
  
</div>

CritiFusion is a training-free, inference-time refinement framework that enhances semantic alignment and visual fidelity in text-to-image diffusion models.  
It operates as a modular plug-in on top of SDXL pipelines, requiring **no finetuning**, **no reward optimization**, and **no model modification**.

<p align="center">
  <img src="img/fig1.png" width="100%">
</p>

---

# Overview

Modern diffusion models achieve strong image quality yet struggle with complex or compositional prompts:

- Missing or hallucinated objects  
- Incorrect attributes (color, count, orientation)  
- Loss of global structure when correcting local details  

CritiFusion introduces **two fully training-free modules** to address these issues:

- **CritiCore** — a multimodal semantic critic built on an LLM ensemble plus a VLM.  
  It rewrites or augments the user prompt with structured, CLIP-friendly semantic hints.

- **SpecFusion** — a spectral latent fusion module for SDXL.  
  It preserves low-frequency structure while injecting high-frequency detail in one img2img step.

Together, these modules provide a drop-in refinement layer that consistently boosts semantic accuracy and perceptual quality.

---

# Method

<p align="center">
  <img src="img/fig2-1.png" width="85%">
</p>

## CritiCore: Multimodal Semantic Critique

CritiCore receives the original prompt and optionally a base image.

It then:

1. Uses a **multi-LLM ensemble** to decompose the prompt into semantic clauses.  
2. Aggregates these outputs into ordered positive and negative tags (CLIP-77 safe).  
3. Invokes a **vision-language model** to critique image–text mismatch.  
4. Merges both signals into a concise, alignment-optimized refined prompt.

The refined prompt strengthens the conditioning of the diffusion model without altering the user’s intent.

## SpecFusion: Spectral Latent Refinement

SpecFusion refines SDXL latents through:

- A **CADR-style mapping** from alignment scores to sampling parameters.  
- A **frequency-domain fusion** between:
  - a structural latent preserving geometry  
  - a detail latent enhancing texture  

The result is a stable, single-step img2img refinement improving sharpness, consistency, and adherence to content.

---

# Gallery

<p align="center">
  <img src="img/fig 4-1.png" width="32%">
  <img src="img/fig 4-2.png" width="32%">
  <img src="img/fig 4-3.png" width="32%">
</p>

---

# Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Set your Together API key
Required for LLM/VLM critique.

Linux / macOS:

```bash
export TOGETHER_API_KEY=your_key_here
```
Windows PowerShell:
```bash
$env:TOGETHER_API_KEY = "your_key_here"
```

---
## Models and Configuration
CritiFusion uses Together-hosted LLMs and VLMs. Model availability may change.

Edit the configuration inside the repository (e.g., configs/criticore.yaml) or modify the Python files:

```bash

LLM_MULTI_CANDIDATES = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
]

AGGREGATOR_MODEL = "Qwen/Qwen2.5-72B-Instruct-Turbo"

VLM_CANDIDATES = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
]

```
---

## 🔥 Quick Start (Minimal Example)
After installation, you can run CritiFusion in three lines:
```python
from critifusion.main import run_critifusion

img = run_critifusion(
    prompt="a knight riding a dragon made of crystal",
    seed=2025
)

img.save("output.png")
```
This will:

  1. Generate a base SDXL image
  
  2. Run CritiCore (LLM + VLM refinement)
  
  3. Run SpecFusion (frequency-domain refinement)
  
  4. Return the final image
---

## 🔧 CLI Usage

This repository also includes ready-to-run scripts for quick experimentation.
**Run end-to-end generation:**
```bash
python scripts/run_generate.py \
    --prompt "a futuristic city made of glass" \
    --seed 123 \
    --output out.png
```
**Run only CritiCore (prompt refinement):**
```bash
python scripts/run_generate.py \
    --prompt "crowded night market with neon signs" \
    --refine_only \
    --output refined_prompt.txt

```
**Run only SpecFusion:**
```bash
python scripts/run_generate.py \
    --prompt "golden forest at sunrise" \
    --apply_specfusion_only
```

---

## Evaluation Metrics Explained
We evaluate each method using four human-aligned metrics:

- **PickScore↑**<br>
  Global human preference predictor that reflects overall image quality and prompt consistency.

- **HPSv2 ↑**<br>
  Measures semantic fidelity and compositional correctness relative to the text prompt.

- **ImageReward ↑**<br>
  Reward-model-based assessment of text–image alignment and realism.

- **Aesthetic ↑**<br>
  Predicts visual appeal independently of the prompt, focusing on composition, lighting, and style.

Higher is better for all four metrics.

---

## Results
<p align="center"> <img src="img/tab1.png" width="45%"> <img src="img/tab2.png" width="45%"> </p>
Across multiple backbones and prompt sets, CritiFusion delivers consistent gains in PickScore, HPSv2, ImageReward, and Aesthetic Score.
The framework closes much of the gap to training-based alignment methods while preserving the simplicity of a training-free, plug-in design.

---

## Limitations and Notes
- CritiFusion depends on external LLM and VLM APIs, which introduces latency and requires network access.

- The effectiveness of CritiCore depends on the quality and diversity of the underlying language models.

- SpecFusion is currently tailored to SDXL latent geometry; extending to other architectures may require additional tuning.
