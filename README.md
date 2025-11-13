# CritiFusion: Semantic and Spectral Refinement for Text-to-Image Diffusion
paper ID 9907 cvpr 2025

CritiFusion is an inference‑time refinement framework designed to enhance semantic alignment and visual quality in text‑to‑image diffusion models. It operates as a plug‑in stage that works with existing SDXL pipelines without any additional training.
<p align="center">
  <img src="img/fig1.png" width="100%">
</p>

## Overview

Recent diffusion models achieve impressive photorealism but often fail on complex textual instructions. CritiFusion addresses this gap with two key modules:

- **CritiCore** — A multimodal semantic critic using multiple LLMs and a VLM to rewrite and enhance the prompt while preserving intent.
- **SpecFusion** — A frequency‑domain latent fusion module that preserves global structure and injects high‑frequency refinements for improved detail.

Together, these components yield stronger semantic alignment and more realistic images, matching or surpassing reward‑optimized baselines while remaining fully training‑free.

## Method

<p align="center">
  <img src="img/fig2-1.png" width="85%">
</p>

## Gallery
<p align="center">
  <img src="img/fig 4-1.png" width="32%">
  <img src="img/fig 4-2.png" width="32%">
  <img src="img/fig 4-3.png" width="32%">
</p>

## Installation

```bash
pip install -r requirements.txt
```

🔗 Set your [Together API Key](https://api.together.xyz/settings/api-keys)
```bash
export TOGETHER_API_KEY=your_key_here
```

## Important Note on Model Availability

CritiFusion relies on Together AI hosted models. Their availability may change over time.  
If an LLM or VLM returns an API error, check the latest supported models:



Update **Cell 2** of the notebook where model lists are defined:

```python
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

## Running CritiFusion

Open the notebook:

```
critifusion.ipynb
```

The notebook will guide you through:

- Generating base images  
- Multi‑LLM concept decomposition  
- VLM critique and prompt refinement  
- Frequency‑domain fusion  
- Final image synthesis  


##  Evaluation Metrics Explained

We evaluate each method using four human-aligned metrics:

- **PickScore↑**: Measures overall human preference for image quality and relevance.
- **HPSv2↑**: Captures prompt adherence based on semantic fidelity.
- **ImageReward↑**: Assesses text-image alignment with a learned reward model.
- **Aesthetic↑**: Predicts visual appeal independently of the prompt.
cs v2).

## Results

<p align="center">
  <img src="img/tab1.png" width="45%">
  <img src="img/tab2.png" width="45%">
</p>

CritiFusion demonstrates consistent improvements on alignment and aesthetic metrics such as PickScore, HPSv2, ImageReward, and Aesthetic Score.

---
