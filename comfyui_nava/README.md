# NAVA ComfyUI Nodes

## Installation

```bash
# Symlink into ComfyUI custom_nodes
cd <ComfyUI-root>/custom_nodes
ln -s /root/paddlejob/workspace/env_run/NAVA/comfyui_nava .

# Symlink model assets into ComfyUI root
cd <ComfyUI-root>
ln -s /root/paddlejob/workspace/env_run/NAVA/nava_src .
ln -s /root/paddlejob/workspace/env_run/NAVA/hf_space_nava/configs .
ln -s /root/paddlejob/workspace/env_run/NAVA/NAVA_FP8 .
ln -s /root/paddlejob/workspace/env_run/NAVA/NAVA_fp8.safetensors .
ln -s /root/paddlejob/workspace/env_run/NAVA/pe_src .
ln -s /root/paddlejob/workspace/env_run/NAVA/huggingface_upload/Wan2.2-TI2V-5B .

# Restart ComfyUI
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
| `single_speaker` (default) | One person speaks | **speech** box — auto-wrapped in `<S>...<E>` |
| `multi_speaker` | Two or more people speak | **dialogue** box — write `<S>...<E>` yourself per utterance |
| `silent` | No speech, environment audio only | Leave everything empty |

**multi_speaker dialogue example:**
```
Character A leans in and says<S>Drop the weapon. Now.<E> Character B smirks<S>You really think this ends here?<E>
```
The role description before each `<S>...<E>` (position, expression, action) helps the model place voice in the soundfield.

---

### NAVA Prompt Rewriter
Expands a short prompt into the long Chinese style NAVA was trained on using Qwen3-4B-Thinking. Strongly recommended, especially for English or short inputs. The model is unloaded after each run to free VRAM.

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

### workflow_i2av_speaker.json — Image to audio-video with timbre control
Same as above plus a LoadAudio node connected to `spk_wav_1`.

**Two speakers:**
1. Add a second LoadAudio node → connect to `spk_wav_2`
2. Set PromptCompose mode to `multi_speaker`
3. Write both lines with `<S>...<E>` in the **dialogue** box

---

## Troubleshooting

**Out of VRAM** — enable `t5_offload` → `group_offload` → reduce `duration_sec` or resolution.

**Poor quality** — verify Prompt Rewriter `enabled=true` and check `[NAVA-Rewriter] OUT` in the terminal for a long Chinese prompt.

**Audio/video out of sync** — check that every `<S>` has a matching `<E>` and tags are not nested.
