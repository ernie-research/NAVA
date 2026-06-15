---
license: apache-2.0
language:
  - en
  - zh
tags:
  - text-to-video
  - text-to-audio-video
  - audio-video-generation
  - mmdit
  - flow-matching
  - wan2.2
pipeline_tag: text-to-video
library_name: custom
base_model: Wan-AI/Wan2.2-TI2V-5B
---

<p align="center">
  <img src="assets/logo.png" alt="NAVA" width="160">
</p>

<h1 align="center">NAVA — Native Audio-Visual Alignment for Generation</h1>

<p align="center">
  <em>State-of-the-art audio-visual synchronization with only <b>6.3 B</b> parameters.</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.30073"><img alt="arXiv" src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a>
  <a href="https://github.com/ernie-research/NAVA"><img alt="Code" src="https://img.shields.io/badge/Code-GitHub-181717.svg"></a>
  <a href="https://ernie-research.github.io/NAVA/"><img alt="Project Page" src="https://img.shields.io/badge/Project_Page-online-2c8ebb.svg"></a>
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green.svg">
  <img alt="Params" src="https://img.shields.io/badge/params-6.3B-orange.svg">
  <img alt="Base model" src="https://img.shields.io/badge/base-Wan2.2--TI2V--5B-7c5cff.svg">
</p>

<p align="center">
  <b>ERNIE Team</b> · Baidu Inc. · arXiv 2026
</p>

<p align="center">
  ⭐ <b>If you find this model useful, please consider giving our <a href="https://github.com/ernie-research/NAVA">GitHub repo</a> a star!</b> ⭐
</p>

<p align="center">
  📖 <a href="https://huggingface.co/baidu/NAVA/blob/main/README_zh.md"><b>中文版 README</b></a>
</p>

---

## TL;DR

NAVA is a **6.3 B-parameter joint audio-video generator** that synthesizes synchronized video **and** audio from a single prompt — including multi-speaker speech with reference-timbre control and image-conditioned continuations.

Instead of post-hoc-aligned dual towers or fully unified tri-modal stacks, NAVA uses an **Align-then-Fuse MMDiT**: a dedicated alignment space first establishes audio-video correspondence, then context (text, speaker embeddings) is fused via cross-attention. On Verse-Bench it sets new SOTA on Sync-C / Sync-D / video quality / audio WER while using **2× to 5× fewer parameters** than open-source baselines.

> **Highlights**
> - **720p 1-min Fast Generation** — 720p synchronized audio-video in ~1 minute via 8-GPU Ulysses sequence parallel.
> - **Dual-Channel Audio** — stereo audio (scene + speech) jointly denoised with video, no post-hoc vocoder alignment.
> - **Precise Multi-Timbre Control** — reference WAVs bound to `<S>...<E>` speech spans for per-speaker voice identity.
> - **Language-Described Camera Control** — shot composition, motion, and pacing directly from the prompt.
> - **Multi-Resolution** — landscape / portrait / square aspect ratios from the same checkpoint.

---

## Model Details

### Quick Facts

| | |
|---|---|
| **Architecture** | Align-then-Fuse MMDiT (Wan2.2 backbone) |
| **Parameters** | **6.3 B** (backbone, joint AV) |
| **Modality** | Joint audio + video, text-conditioned |
| **Resolution** | 1280×704 (recommended) · 960×960 also supported |
| **Frames / FPS** | 37 frames @ 24 fps ≈ 6 s · 55–61 frames ≈ 9–10 s |
| **Audio** | 25 latent tokens / sec, ≤ 10 s |
| **Sampling** | Flow matching · UniPC scheduler · 50 default steps |
| **Precision** | bf16 |
| **Parallelism** | Single-GPU **or** Ulysses sequence parallel (up to 8 GPUs) |
| **Base model** | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |

### Architecture

<p align="center">
  <img src="assets/arch.png" alt="NAVA Architecture" width="900">
</p>

NAVA instantiates *Native Audio-Visual Alignment* as an **Align-then-Fuse MMDiT** stack:

- **Hierarchical Alignment Layers — 10 double-stream blocks.** Video and audio keep separate QKV projections and FFNs but share a joint self-attention over concatenated `[video_tokens; audio_tokens]`, plus dedicated cross-attention to text. This builds an alignment space where AV correspondence is learned without semantic context interference.
- **Unified Fusion Layers — 20 single-stream blocks.** Video and audio share QKV/FFN; a unified joint attention treats all tokens as one stream, with a single text cross-attention path. This is where context-conditioned denoising happens.
- **Backbone hyperparameters.** `dim=3072`, `ffn_dim=14336`, 24 attention heads, 30 layers (10 double + 20 single), `text_len=512`, patch size `(1, 2, 2)`. RMSNorm on QK; cross-attention norm; ε = 1e-6.
- **Positional encoding.** 3D RoPE for video (temporal + height + width), 1D RoPE for audio, applied jointly inside the joint-attention path.
- **Timbre-in-Context Conditioning.** Reference-WAV speaker embeddings (ReDimNet, 192-d) are injected through the context pathway and bound to `<S>...<E>` speech spans, enabling per-speaker timbre control in multi-speaker scenes.
- **3D cross-modal CFG.** Independent classifier-free guidance scales for video, audio, and the cross-modal alignment direction (`video_align_guidance_scale`, `audio_align_guidance_scale`) keep AV synchronization tight at inference.

---

## Evaluation

### Table 1 — VerseBench (general AV capability)

NAVA achieves the **best** AV synchronization (Sync-C / Sync-D), video quality, and audio WER, with the smallest parameter budget.

| Model | Params | Resolution | Sync-C ↑ | Sync-D ↓ | IB ↑ | Video Quality ↑ | WER ↓ | PQ ↑ | FD ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Ovi 1.1 | 10 B | 720p | <u>7.4839</u> | 7.9791 | 0.199 | <u>0.636</u> | 0.102 | 5.8432 | 0.9418 |
| MOVA | A18B (32 B) | 720p | 7.2888 | 7.808 | 0.269 | 0.603 | 0.126 | **7.2331** | 0.9222 |
| Davinci | 15 B | 540p | 7.1487 | 7.8158 | 0.269 | 0.600 | 0.151 | 5.9559 | 0.9307 |
| LTX 2.3 | 19 B | 512p | 7.2476 | <u>7.6902</u> | **0.337** | 0.576 | 0.106 | <u>6.9459</u> | **0.8287** |
| **NAVA (ours)** | **6.3 B** | 720p | **7.7914** | **7.5655** | <u>0.313</u> | **0.659** | **0.099** | 6.8609 | <u>0.8328</u> |

<sub>↑ higher is better · ↓ lower is better · **bold** = best · <u>underline</u> = 2nd best.</sub>

### Table 2 — Seed-TTS-eval (speech quality)

Among joint AV models, NAVA delivers speech quality close to dedicated audio-only systems. Audio-only rows are listed *for reference*; they are not directly comparable.

| Category | Model | WER ↓ | Speaker Similarity ↑ |
|---|---|---|---|
| Audio-Only *(reference)* | CosyVoice | 4.29 | 60.9 |
| Audio-Only *(reference)* | Qwen2.5-Omni | 2.72 | 63.2 |
| Audio-Video Joint | DreamID-Omni | 33.44 | 34.1 |
| Audio-Video Joint | **NAVA (ours)** | **5.81** | **62.4** |

---

## How to Use

> **TL;DR command.** After §1 setup is complete:
> ```bash
> bash scripts/inference.sh           # General T2AV
> bash scripts/inference_timbre.sh    # I2AV + timbre control
> ```
> Outputs land under `eval_results/`.

### 1 · Setup (once)

```bash
git clone https://github.com/ernie-research/NAVA && cd NAVA
conda create -n nava python=3.10 -y && conda activate nava

# 1. PyTorch — install first, matching your CUDA. Do NOT skip and let
#    `pip install -e .` resolve it; pip will pick a CPU wheel or clobber
#    your existing CUDA build.
#    cu128 covers RTX 40 / 50 series, H100/H800. Use cu121 for older cards.
pip install torch==2.8.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. NAVA itself + core inference deps (editable install).
pip install -e .

# 3. flash-attn — must be a separate step with --no-build-isolation, otherwise
#    pip builds it in a sandbox that can't see your installed torch.
pip install flash-attn --no-build-isolation

# 4. Optional extras — pick what you need:
pip install -e ".[sp]"       # multi-GPU sequence parallel (--use_sp, xfuser)
pip install -e ".[demo]"     # gradio_demo/ Web UI
pip install -e ".[rewrite]"  # pe_src/ vLLM batch prompt rewriter
pip install -e ".[all]"      # everything above

# 5. All weights in one shot — main checkpoint + Wan2.2 VAE + T5 + LTX audio VAE
huggingface-cli download <NAVA-repo-id> --local-dir .
```

<details>
<summary><b>Why split into four pip steps?</b></summary>

`torch`, `flash-attn`, and `xfuser` each compile against a specific CUDA / PyTorch ABI. Listing them inside `pyproject.toml`'s base `dependencies` causes pip to re-resolve and replace your working CUDA wheels with whatever happens to be latest on PyPI — the #1 source of "env is impossible to install" reports. We deliberately keep `torch` / `triton` out of base deps and `flash-attn` out of every dep list; install order is the contract.

If you must reinstall in-place, **never** run `pip install -r requirements.txt --force-reinstall` — go through the four numbered steps above.

</details>

<details>
<summary><b>GPU compatibility (single-GPU FP8 path)</b></summary>

| GPU | VRAM | Status | Notes |
|---|---|---|---|
| H100 / H800 / A100 80G | 80 GB | ✅ Default | bf16 SP=8, FP8 single-GPU — both work |
| RTX 5090 | 32 GB | ✅ Default | `bash scripts/inference_fp8.sh` runs as-is. flash-attn ≥ 2.8 needed for sm_120 |
| RTX 4090 / 4090D | 24 GB | ✅ With group offload | Use `inference_timbre_fp8.sh`-style flags: `--t5_offload --group_offload --vae_tiling` with `OFFLOAD_GROUP_SIZE=2~3`, `VAE_TILE_H/W=16/30`. Peak ~18 GB |
| RTX 3090 / 4080 / A6000 | 24 GB | ✅ With group offload | Same as 4090 |
| RTX 4070 Ti / 5070 Ti | 16 GB | ⚠️ Tight | Requires `OFFLOAD_GROUP_SIZE=1` + `VAE_TILE_H/W=14/26`. Peak ~14 GB, slow |
| RTX 3060 / 4060 / 5060 | 12 GB | ❌ Not supported at default 704×1280×37 | Even with maximum offload, peak stays >12 GB. Possible only if you drop resolution to 480×832 and frames to 17–25, with degraded quality |

FP8 weight-only is **dequant-to-bf16** at compute time (`NAVA_FP8/` Phase 1) — it saves VRAM but does **not** unlock FP8 tensor cores yet, so 5090 / 4090 / H100 see no compute speedup over bf16.

</details>

<details>
<summary><b>Expected on-disk layout</b></summary>

```
NAVA/
├── NAVA.ckpt                                                    # main checkpoint (24 GB)
├── Wan2.2-TI2V-5B/
│   ├── Wan2.2_VAE.pth                                           # 2.7 GB
│   ├── models_t5_umt5-xxl-enc-bf16.pth                          # 11 GB
│   └── google/umt5-xxl/{spiece.model, tokenizer.json}
├── params/
│   └── LTX2/
│       ├── ltx-2.3-22b-dev_audio_vae.safetensors                # 348 MB
│       └── LICENSE                                              # LTX-2 Community License
└── configs/                                                     # inference YAMLs
```

The LTX audio-VAE Python code is vendored under `nava_src/vendor/ltx_core/` (see its `NOTICE.md`), so no separate clone of the LTX-Video repo is needed. ReDimNet is fetched via `torch.hub` on first run.
</details>

### 2 · One-command inference (recommended, 8 GPU SP)

The repo ships two end-to-end scripts that build a JSONL inline and launch SP=8 inference:

```bash
# General T2AV (text-only)
bash scripts/inference.sh

# I2AV + Timbre Control (first-frame image + reference voice)
bash scripts/inference_timbre.sh
```

Override defaults via env vars:

```bash
CKPT=/path/to/NAVA.ckpt OUT_DIR=eval_results/run1 bash scripts/inference.sh
TIMBRE_SCALE=3.0 SPK_WAV=/path/to/spk.wav    bash scripts/inference_timbre.sh
```

### 2.5 · End-to-end workflows with prompt rewrite

NAVA was trained on long, structured captions, so short user prompts (and any I2AV input where the image carries scene info) benefit from an inline rewrite step. Two ready-to-run scripts cover the two common cases:

```bash
# Text-only: rewrite short prompts → FP8 generate
bash scripts/inference_fp8_rewrite.sh

# Image + text: VL caption the image → compose with user prompt → rewrite → FP8 generate
bash scripts/inference_fp8_vl_rewrite.sh
```

| Script | Pipeline (per sample, rank 0) | When to use |
|---|---|---|
| `inference_fp8_rewrite.sh` | `user_prompt → Rewriter → broadcast → FP8 DiT → VAE` | Text-only T2AV. Short / English / casual prompts that lack the structured detail NAVA expects |
| `inference_fp8_vl_rewrite.sh` | `image → VL caption → compose(scene, user_prompt) → Rewriter → broadcast → FP8 DiT → VAE`. Samples without `image_path` fall back to the plain rewrite path | I2AV (first-frame image given) — the VL captioner extracts scene description from the image, which is then merged with the user's text before rewrite |

#### Choosing the rewriter model

The default `REWRITE_MODEL` points at `Qwen3-4B-Instruct-2507`, but you can swap in `Qwen3-4B-Thinking-2507` by overriding the env var:

```bash
REWRITE_MODEL=/path/to/Qwen3-4B-Thinking-2507 bash scripts/inference_fp8_rewrite.sh
```

| Model | Latency | Reliability | Notes |
|---|---|---|---|
| **Qwen3-4B-Instruct-2507** *(default)* | Fast (~1–3 s/prompt) | Less stable — occasionally returns malformed output and triggers the rewriter's retry path | Use when throughput matters and you can tolerate a few retries |
| **Qwen3-4B-Thinking-2507** | Slow (~10–20 s/prompt — emits internal `<thinking>` tokens) | Very stable — virtually no retries needed | Use for batch / overnight runs where you want every sample to land cleanly on the first try |

VL captioner (only used by `inference_fp8_vl_rewrite.sh`) is `VL_MODEL=Qwen3-VL-4B-Instruct` by default; same override pattern applies.

### 3 · Custom batches — write your own JSONL

The two FP8 rewrite scripts above point at preset JSONL files by default — use them as templates when writing your own:

| Default `DATA_FILE` | Used by | Contents |
|---|---|---|
| `infer_cases/general/prompts_simple.jsonl` | `inference_fp8_rewrite.sh` | Short text-only prompts (T2AV) |
| `infer_cases/general/prompts_simple_i2v.jsonl` | `inference_fp8_vl_rewrite.sh` | Prompts + `image_path` (+ optional `spk_wavs`) for I2AV |

Each line is one prompt:

```jsonl
{"prompt": "一位男子在海边奔跑，镜头跟随。背景是海浪声和风声。"}
{"prompt": "两人对话<S>Hello<E><S>Hi there<E>", "spk_wavs": ["spk1.wav", "spk2.wav"]}
{"prompt": "镜头跟随主体...", "image_path": "infer_cases/timbre/wolverine.png"}
```

| Field | Required | Description |
|---|---|---|
| `prompt` | yes | Text caption (also accepts legacy `text` field name) |
| `image_path` | no | Path to first-frame image — absolute, or relative to the repo root. Auto-enables I2V for this sample |
| `spk_wavs` | no | Speaker reference WAVs (max 2), absolute or repo-root-relative paths |

Then launch:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib torchrun \
    --nnodes=1 --nproc_per_node=8 \
    --master_addr=127.0.0.1 --master_port=29507 \
    inference_nava.py \
    --config configs/baseline_t2av_demo_mmdit_no_split_ltx_control_unipc.yaml \
    --ckpt NAVA.ckpt \
    --out_dir ./outputs \
    --data_format json --data_file my_prompts.jsonl \
    --width 1280 --height 704 --frames 37 --fps 24 \
    --steps 50 --save_sample --gen_turn 1 --use_sp
```

Outputs land at `outputs/{save_path}-{gen_turn}_av.mp4`. For timbre-controlled samples, also pass `--timbre_cfg --timbre_align_guidance_scale 3.0`.

#### Mode cheatsheet

| Goal | JSONL fields | Extra flags |
|---|---|---|
| Text → AV | `prompt` | — |
| Image → AV | `prompt` + `image_path` | (auto-detected) |
| Timbre-controlled speech | `prompt` + `spk_wavs` | `--timbre_cfg --timbre_align_guidance_scale 3.0` |
| 9-second video | any | `--frames 55` |
| Single-GPU (slower) | any | omit `--use_sp` |

### 4 · Prompt rewriting (recommended for short / English inputs)

NAVA is trained on Chinese dense captions; short or English prompts benefit substantially from rewriting before inference. Three pathways are provided, all sharing the same system prompt and sampling profile (so output style stays consistent), with `<S>...<E>` speech spans preserved verbatim.

| Pathway | Backend | Speed | Best for |
|---|---|---|---|
| **vLLM batch server** (`pe_src/`) | Qwen3-4B-Thinking-2507 served via vLLM, async HTTP | **< 2 s** / prompt | Offline batches |
| **Local transformers, single** (`gradio_demo/rewrite_single.py`) | Same model, in-process | 40–80 s / prompt | One-off CLI |
| **Gradio "Rewrite" button** | Same as above, hosted in Gradio | 40–80 s / prompt | Interactive UI |

```bash
# Batch path: start vLLM server, then rewrite a txt of prompts
bash pe_src/start_server.sh --gpu 0 --low-footprint
python pe_src/rewrite.py -i prompts.txt -o prompts_rewritten.txt
```

### 5 · Gradio Web UI

Interactive demo with click-to-rewrite (Qwen3-4B-Thinking by default), image upload, and reference-WAV upload:

```bash
bash gradio_demo/start_gradio.sh \
    --config configs/baseline_t2av_demo_mmdit_no_split_ltx_control_unipc.yaml \
    --ckpt NAVA.ckpt \
    --rewrite_model pe_src/Qwen3-4B-Thinking-2507 \
    --port 8000 --nproc 8
```

> **Speed tip:** the default `Qwen3-4B-Thinking-2507` runs an internal `<think>...</think>` block and takes ~40–80 s per click. For interactive testing where latency matters more than rewrite quality, swap in `pe_src/Qwen3-4B-Instruct-2507` via `--rewrite_model` — same SYSTEM_PROMPT, same sampling profile, no thinking pass (~2–3× faster).

<details>
<summary><b>Debug mode (no models, UI only)</b></summary>

```bash
python gradio_demo/gradio_server.py --debug --port 8000
```
</details>

---

## Bias, Safety, and Misuse

NAVA can synthesize video and speech conditioned on a reference image (`image_path`) and reference voice (`spk_wavs`). Using it to depict real persons without consent — including face-likeness or voice-likeness reproduction — is prohibited by the license and may also be illegal in your jurisdiction. We recommend:

1. Only use **consent-approved** reference media.
2. **Label generated content as synthetic.**
3. Apply **provenance / watermarking** before redistribution.

---

## Citation

```bibtex
@article{nava2026,
  title   = {NAVA: Native Audio-Visual Alignment for Joint Audio-Video Generation},
  author  = {ERNIE Team},
  journal = {arXiv preprint},
  year    = {2026},
}
```

## Acknowledgements

NAVA builds on excellent upstream work: **Wan2.2-TI2V-5B** (video backbone & VAE), **LTX 2.3** (audio VAE + built-in vocoder), **umt5-xxl** (text encoder), and **ReDimNet** (speaker embedding). We also thank the open-source AV-generation community — Ovi, MOVA, Davinci, LTX — for releasing strong baselines that made fair benchmarking possible.

## License & Contact

Released under **Apache-2.0**. For research / commercial inquiries, contact the **ERNIE team at Baidu Inc.**
