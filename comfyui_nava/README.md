# NAVA ComfyUI Nodes

## Installation

```bash
# Set NAVA_ROOT to your NAVA checkout, then symlink into ComfyUI.
export NAVA_ROOT=/path/to/NAVA
export COMFY_ROOT=/path/to/ComfyUI

# Symlink the custom nodes
ln -s "$NAVA_ROOT/comfyui_nava" "$COMFY_ROOT/custom_nodes/"

# Symlink model assets into ComfyUI root so the nodes' relative paths resolve
cd "$COMFY_ROOT"
ln -s "$NAVA_ROOT/nava_src" .
ln -s "$NAVA_ROOT/configs" .
ln -s "$NAVA_ROOT/NAVA_FP8" .
ln -s "$NAVA_ROOT/NAVA_fp8.safetensors" .
ln -s "$NAVA_ROOT/pe_src" .
ln -s "$NAVA_ROOT/Wan2.2-TI2V-5B" .

# Restart ComfyUI
```

Alternatively, launch ComfyUI from the NAVA root so all relative paths resolve
without symlinks:
```bash
cd "$NAVA_ROOT"
ln -s "$NAVA_ROOT/comfyui_nava" "$COMFY_ROOT/custom_nodes/"
python "$COMFY_ROOT/main.py"
```

---

## Nodes

### NAVA Model Loader
Loads the NAVA checkpoint. Results are cached — re-running with the same paths skips reloading.

| Parameter | Description | Recommended |
|---|---|---|
| ckpt_path | Checkpoint file | `NAVA_fp8.safetensors` |
| config_path | Config file | `configs/nava.yaml` |
| t5_offload | Move T5 encoder to CPU after encoding (~32 GB freed) | `true` |
| group_offload | Page DiT blocks CPU↔GPU (~6 GB more, slower steps) | Enable if VRAM < 48 GB |
| weight_dtype | Weight precision | `fp8_e4m3fn` with fp8 checkpoint |

---

### NAVA Image Captioner (optional)
Describes an image using Qwen3-VL-4B. Output feeds into Prompt Compose.

| Parameter | Description |
|---|---|
| model_path | `pe_src/Qwen3-VL-4B-Instruct` |
| offload_after | Free VRAM after captioning — recommended |

---

### NAVA Prompt Compose
Assembles scene description and dialogue into the prompt format NAVA expects.

**mode:**

| mode | When to use | What to fill |
|---|---|---|
| `single_speaker` (default) | One person speaks | **speech** box — write role description + `<S>...<E>` yourself |
| `multi_speaker` | Two or more people speak | **dialogue** box — write all utterances with `<S>...<E>` |
| `silent` | No speech, environment audio only | Leave everything empty |

**single_speaker speech example:**
```
张三抬起头说<S>我不去。<E>
```
or in English:
```
The man leans forward and says<S>Don't move.<E>
```

**multi_speaker dialogue example:**
```
Character A leans in and says<S>Drop the weapon. Now.<E> Character B smirks<S>You really think this ends here?<E>
```
The role description before each `<S>...<E>` (position, expression, action) helps the model place voice in the soundfield.

---

### NAVA Prompt Rewriter
Expands a short prompt into the long Chinese style NAVA was trained on using Qwen3-4B-Thinking-2507 (default). Strongly recommended, especially for English or short inputs. The model is unloaded after each run to free VRAM.

> **Speed tip:** the default `Qwen3-4B-Thinking-2507` runs a `<think>...</think>` block before the rewrite, which spends ~1000–2000 extra tokens per call. If you care more about latency than quality, point `model_path` to `pe_src/Qwen3-4B-Instruct-2507` (no thinking, ~2–3× faster, slightly lower long-prompt quality). The Captioner / Sampler / RNG plumbing is unchanged.

---

### NAVA Sampler
Core inference node.

| Parameter | Description |
|---|---|
| model | Connect from Model Loader |
| prompt | Connect from Prompt Rewriter |
| image (optional) | Connect a LoadImage node to enable I2V mode |
| spk_wav_1/2 (optional) | Connect LoadAudio nodes for speaker timbre control |
| duration_sec | Video length in seconds |
| steps | Diffusion steps — 50 recommended |
| video_cfg_scale / audio_cfg_scale | CFG guidance strength — default 3.0 / 2.0 |

**Speaker binding:** `spk_wav_1` controls the 1st `<S>...<E>` span, `spk_wav_2` the 2nd.

---

### NAVA Save Video
Muxes frames + audio into MP4. Install [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for inline preview.

---

### NAVA Show Text
Displays any STRING in the terminal log (full, not truncated). Connect after Prompt Rewriter to inspect the final prompt sent to the Sampler.

---

## Example Workflows

Drag any JSON from `examples/` into the ComfyUI canvas to load a pre-wired graph.

### workflow_t2av.json — Text to audio-video
```
NAVAModelLoader ──────────────────────────────→ NAVASampler → NAVASaveVideo
NAVAPromptRewriter (type prompt directly) ───→ NAVASampler
                                                     ↓
                                              NAVAShowText
```

### workflow_i2av.json — Image to audio-video
```
LoadImage → NAVAImageCaptioner → NAVAPromptCompose → NAVAPromptRewriter → NAVASampler → NAVASaveVideo
LoadImage ────────────────────────────────────────────────────────────→ NAVASampler (I2V first frame)
```
Steps: swap LoadImage for your image, fill **speech** in PromptCompose (or set mode to `silent`), Queue.

### workflow_i2av_single_speaker.json — Image to audio-video, single-speaker timbre control
Same as `workflow_i2av.json` plus one LoadAudio node connected to `spk_wav_1`. PromptCompose mode is `single_speaker`; write the role's line with `<S>...<E>` in the **speech** box.

### workflow_i2av_multi_speaker.json — Image to audio-video, two-speaker timbre control
Two LoadAudio nodes wired to `spk_wav_1` and `spk_wav_2`. PromptCompose mode is `multi_speaker`; write both speakers' lines with `<S>...<E>` in the **dialogue** box.

---

## Troubleshooting

**Out of VRAM** — enable `t5_offload` → `group_offload` → reduce `duration_sec` or resolution.

**Poor quality** — verify Prompt Rewriter `enabled=true` and check `[NAVA-Rewriter] OUT` in the terminal for a long Chinese prompt.

**Audio/video out of sync** — check that every `<S>` has a matching `<E>` and tags are not nested.
