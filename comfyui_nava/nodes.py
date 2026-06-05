"""
ComfyUI custom node definitions for NAVA audio-video generation.

Nodes:
    NAVAModelLoader     — load checkpoint + config, cache engine in memory
    NAVAPromptRewriter  — (optional) rewrite a short / English prompt into the
                          long Chinese description NAVA was trained on
    NAVASampler         — run inference, return video frames + audio
    NAVASaveVideo       — merge frames + audio → MP4, save to disk

Typical graph:
    NAVAModelLoader ─┐
                     ├─→ NAVASampler → NAVASaveVideo
    [text] → (NAVAPromptRewriter) ─┘
                           ↑
               (optional) IMAGE input for I2V
               (optional) STRING paths for speaker WAVs
"""

import os
import tempfile
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Aspect-ratio presets exposed as a dropdown in NAVASampler
# ---------------------------------------------------------------------------
_ASPECT_RATIOS = {
    "16:9  1280×704":  (704, 1280),
    "9:16  704×1280":  (1280, 704),
    "1:1   960×960":   (960, 960),
    "custom":          None,          # use the explicit height / width fields
}

_ASPECT_RATIO_KEYS = list(_ASPECT_RATIOS.keys())


# ---------------------------------------------------------------------------
# Small helper: ComfyUI IMAGE tensor [B,H,W,C] float 0-1 → temp PNG file path.
# Used to feed a first-frame image into the VAE encoder (which expects a path).
# ---------------------------------------------------------------------------
def _image_tensor_to_tmp_png(image: torch.Tensor) -> str:
    frame = image[0].cpu().float().numpy()
    frame = (frame * 255).clip(0, 255).astype(np.uint8)
    from PIL import Image
    img = Image.fromarray(frame)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Node 1: NAVAModelLoader
# ---------------------------------------------------------------------------

class NAVAModelLoader:
    """
    Load a NAVA checkpoint and config, return a cached engine handle.

    Caches the loaded model in memory — rerunning with the same paths will
    return the cached instance without reloading weights.
    """

    CATEGORY = "NAVA"
    RETURN_TYPES = ("NAVA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": (
                    "STRING",
                    {"default": "NAVA_fp8.safetensors", "multiline": False,
                     "tooltip": "Path to checkpoint. Default points to the fp8 ckpt "
                                "(NAVA_fp8.safetensors); switch to NAVA.safetensors and set "
                                "weight_dtype=bf16 for full precision."},
                ),
                "config_path": (
                    "STRING",
                    {"default": "configs/nava.yaml", "multiline": False,
                     "tooltip": "Path to nava.yaml config file"},
                ),
            },
            "optional": {
                "t5_offload": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Move T5 text encoder to CPU after encoding (~32 GB freed; "
                                "recommended for single-GPU setups with < 80 GB VRAM)"},
                ),
                "group_offload": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Page DiT backbone blocks one group at a time (CPU↔GPU); "
                                "saves an additional ~6 GB at the cost of ~3× slower steps"},
                ),
                "offload_group_size": (
                    "INT",
                    {"default": 10, "min": 1, "max": 30,
                     "tooltip": "Number of DiT blocks per offload group (only used when "
                                "group_offload=True; smaller = less VRAM, more transfers)"},
                ),
                "weight_dtype": (
                    ["fp8_e4m3fn", "auto", "bf16"],
                    {"default": "fp8_e4m3fn",
                     "tooltip": "Weight format. fp8_e4m3fn (default) halves DiT VRAM "
                                "(~12 GB → ~6.1 GB) via NAVA_FP8 weight-only quantization "
                                "— requires an fp8 checkpoint (use NAVA_FP8/convert_to_fp8.py "
                                "to convert). 'auto' detects from ckpt; 'bf16' forces full precision."},
                ),
            },
        }

    def load(
        self,
        ckpt_path: str,
        config_path: str,
        t5_offload: bool = True,
        group_offload: bool = False,
        offload_group_size: int = 10,
        weight_dtype: str = "fp8_e4m3fn",
    ):
        from .engine import get_or_load_engine
        engine = get_or_load_engine(
            ckpt_path=ckpt_path,
            config_path=config_path,
            t5_offload=t5_offload,
            group_offload=group_offload,
            offload_group_size=offload_group_size,
            weight_dtype=weight_dtype,
        )
        return (engine,)


# ---------------------------------------------------------------------------
# Node 2: NAVAPromptRewriter (optional, can be bypassed)
# ---------------------------------------------------------------------------

class NAVAPromptRewriter:
    """
    Optional Qwen3-based prompt rewriter.

    NAVA is trained on long, descriptive Chinese prompts. Short or English
    prompts often produce weaker results. This node uses NAVA's bundled
    Qwen3-4B-Thinking-2507 (or any HF chat model you point it at) plus the
    SYSTEM_PROMPT from pe_src/rewrite_single.py to expand a short prompt
    into the format NAVA expects.

    Toggle `enabled` off to bypass rewriting entirely — when disabled the
    model is NOT loaded and the input prompt is returned as-is.
    """

    CATEGORY = "NAVA"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "rewrite_prompt"
    OUTPUT_NODE = True   # so the rewritten prompt is shown inline in the node UI

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "Original prompt. Can be short, English, or just keywords."},
                ),
                "enabled": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Off = pass the prompt through unchanged (no model loaded). "
                                "On = run Qwen3 to expand into NAVA's long-Chinese style."},
                ),
            },
            "optional": {
                "model_path": (
                    "STRING",
                    {"default": "pe_src/Qwen3-4B-Thinking-2507", "multiline": False,
                     "tooltip": "Path to a local Qwen3 chat model (relative to NAVA root or "
                                "absolute), or a HuggingFace repo id. Default Qwen3-4B-Thinking-2507 "
                                "is the bundled NAVA rewriter; thinking is always on for this variant."},
                ),
                "use_4bit": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Load the rewriter in 4-bit (bitsandbytes nf4). Default ON for "
                                "speed and to leave VRAM for NAVA. Turn off for marginally better "
                                "quality at ~2x the memory."},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 4096, "min": 128, "max": 16384, "step": 128,
                     "tooltip": "Max tokens for the rewritten prompt. 4096 leaves comfortable "
                                "headroom for the <think> block plus the long-Chinese answer."},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Sampling temperature (matches pe_src/rewrite_single.py default)"},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "top_k": (
                    "INT",
                    {"default": 20, "min": 0, "max": 200},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**31 - 1,
                     "tooltip": "Sampling seed for the rewriter (independent of NAVA's seed)"},
                ),
                "disable_thinking": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Skip the model's <think>...</think> block (faster). Effective "
                                "only on Qwen3 standard variants (Qwen3-8B etc.); the default "
                                "Qwen3-*-Thinking-2507 is hard-wired to think and ignores this."},
                ),
            },
        }

    def rewrite_prompt(
        self,
        prompt: str,
        enabled: bool = True,
        model_path: str = "pe_src/Qwen3-4B-Thinking-2507",
        use_4bit: bool = True,
        max_new_tokens: int = 4096,
        temperature: float = 0.3,
        top_p: float = 0.75,
        top_k: int = 20,
        seed: int = 42,
        disable_thinking: bool = False,
    ):
        if not enabled or not prompt or not prompt.strip():
            return {"ui": {"text": [prompt or ""]}, "result": (prompt,)}

        from .rewriter import rewrite as _do_rewrite
        rewritten = _do_rewrite(
            prompt=prompt,
            model_path=model_path,
            use_4bit=use_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            disable_thinking=disable_thinking,
        )
        print(f"[NAVA-Rewriter] OUT ({len(rewritten)} chars):\n{rewritten}")

        from .rewriter import offload_all_to_cpu as _offload_rewriter
        _offload_rewriter()

        return {"ui": {"text": [rewritten]}, "result": (rewritten,)}


# ---------------------------------------------------------------------------
# Node 3: NAVASampler
# ---------------------------------------------------------------------------

class NAVASampler:
    """
    Run NAVA inference for a single prompt.

    Required input  : model (from NAVAModelLoader), prompt text.
    Optional inputs : IMAGE for I2V first-frame conditioning;
                      STRING paths to speaker-reference WAVs for timbre control.

    Returns video frames as a ComfyUI IMAGE batch [T,H,W,C] and audio as a
    ComfyUI AUDIO dict {waveform: [1,C,L], sample_rate: int}.
    """

    CATEGORY = "NAVA"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("NAVA_MODEL",),
                "prompt": (
                    "STRING",
                    {"multiline": True,
                     "default": "",
                     "tooltip": "Text prompt. Use <S>...<E> to wrap speech spans; "
                                "add spk_wav paths below to bind timbre per speaker."},
                ),
                "aspect_ratio": (
                    _ASPECT_RATIO_KEYS,
                    {"default": "16:9  1280×704",
                     "tooltip": "Output aspect ratio preset. Select 'custom' to use "
                                "the height / width fields below instead."},
                ),
                "height": (
                    "INT",
                    {"default": 704, "min": 64, "max": 2048, "step": 16,
                     "tooltip": "Output height in pixels (used only when aspect_ratio=custom; "
                                "ignored otherwise)."},
                ),
                "width": (
                    "INT",
                    {"default": 1280, "min": 64, "max": 2048, "step": 16,
                     "tooltip": "Output width in pixels (used only when aspect_ratio=custom; "
                                "ignored otherwise)."},
                ),
                "duration_sec": (
                    "INT",
                    {"default": 6, "min": 1, "max": 10, "step": 1,
                     "tooltip": "Video duration in seconds. Converted internally to latent frames "
                                "via latent_frames = duration_sec × 6 + 1 "
                                "(e.g. 6 s → 37 latent frames → 145 decoded frames @ 24 fps)."},
                ),
                "steps": (
                    "INT",
                    {"default": 50, "min": 1, "max": 100,
                     "tooltip": "Number of diffusion steps (50 recommended)"},
                ),
                "video_cfg_scale": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": "Video classifier-free guidance scale"},
                ),
                "audio_cfg_scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": "Audio classifier-free guidance scale"},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**31 - 1,
                     "tooltip": "Random seed for reproducible generation"},
                ),
                "vae_tiling": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Tile VAE decode spatially to reduce peak VRAM during decode"},
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "First-frame image for image-to-video (I2V) mode. "
                                "Leave unconnected for text-to-video."},
                ),
                "spk_wav_1": (
                    "AUDIO",
                    {"tooltip": "Speaker-1 reference audio for timbre control. "
                                "Connect a LoadAudio node. Binds to the 1st <S>...<E> span."},
                ),
                "spk_wav_2": (
                    "AUDIO",
                    {"tooltip": "Speaker-2 reference audio for timbre control. "
                                "Connect a LoadAudio node. Binds to the 2nd <S>...<E> span."},
                ),
                "align_3d_cfg": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Enable 3D-aligned CFG (audio-visual mutual masking branch). "
                                "On = sharper AV alignment; off = a touch faster, weaker sync. "
                                "Default matches configs/nava.yaml."},
                ),
                "video_align_cfg_scale": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": "Video align-CFG scale. Used only when align_3d_cfg is on. "
                                "Higher = stronger pull toward audio-conditioned video branch."},
                ),
                "audio_align_cfg_scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": "Audio align-CFG scale. Used only when align_3d_cfg is on."},
                ),
                "image_cfg_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": "Image-mode CFG scale (T2I path). Used only when no video frames "
                                "are produced (vanishingly rare in this node — keep default)."},
                ),
                "image_align_cfg_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": "Image align-CFG scale (T2I path). See image_cfg_scale."},
                ),
                "timbre_cfg": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Enable timbre-conditional CFG. Only effective when at least one "
                                "spk_wav_* is provided; ignored otherwise. Default matches "
                                "configs/nava.yaml."},
                ),
                "timbre_align_cfg_scale": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": "Timbre align-CFG scale. Used only when timbre_cfg is on AND "
                                "a speaker reference is provided."},
                ),
            },
        }

    def sample(
        self,
        model,
        prompt: str,
        aspect_ratio: str,
        height: int,
        width: int,
        duration_sec: int,
        steps: int = 50,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 2.0,
        seed: int = 42,
        vae_tiling: bool = False,
        image: Optional[torch.Tensor] = None,
        spk_wav_1: Optional[dict] = None,
        spk_wav_2: Optional[dict] = None,
        align_3d_cfg: bool = True,
        video_align_cfg_scale: float = 3.0,
        audio_align_cfg_scale: float = 2.0,
        image_cfg_scale: float = 5.0,
        image_align_cfg_scale: float = 5.0,
        timbre_cfg: bool = True,
        timbre_align_cfg_scale: float = 3.0,
    ):
        # Resolve resolution: aspect_ratio preset wins; "custom" falls through
        # to the height / width fields.
        preset = _ASPECT_RATIOS.get(aspect_ratio)
        if preset is not None:
            height, width = preset

        latent_frames = duration_sec * 6 + 1

        # I2V: save IMAGE tensor to temp PNG so the VAE encoder can read it as a file
        image_path = None
        _tmp_png = None
        is_i2v = image is not None
        if is_i2v:
            _tmp_png = _image_tensor_to_tmp_png(image)
            image_path = _tmp_png

        # Speaker reference: save AUDIO dicts to temp WAV files
        import torchaudio as _torchaudio
        _tmp_wavs = []
        spk_wav_paths = []
        for wav_dict in [spk_wav_1, spk_wav_2]:
            if wav_dict is not None:
                waveform = wav_dict["waveform"]
                if waveform.dim() == 3:
                    waveform = waveform[0]  # [C, L]
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                _torchaudio.save(tmp.name, waveform.cpu().float(), int(wav_dict["sample_rate"]))
                tmp.close()
                _tmp_wavs.append(tmp.name)
                spk_wav_paths.append(tmp.name)
        spk_wav_paths = spk_wav_paths if spk_wav_paths else None

        try:
            frames_f32, audio_out = model.generate(
                prompt=prompt,
                height=height,
                width=width,
                latent_frames=latent_frames,
                image_path=image_path,
                spk_wav_paths=spk_wav_paths,
                steps=steps,
                is_i2v=is_i2v,
                video_cfg=video_cfg_scale,
                audio_cfg=audio_cfg_scale,
                image_cfg=image_cfg_scale,
                align_3d_cfg=align_3d_cfg,
                video_align_cfg=video_align_cfg_scale,
                audio_align_cfg=audio_align_cfg_scale,
                image_align_cfg=image_align_cfg_scale,
                timbre_cfg=timbre_cfg,
                timbre_align_cfg=timbre_align_cfg_scale,
                seed=seed,
                vae_tiling=vae_tiling,
            )
        finally:
            # Clean up temporary PNG and WAV files regardless of success/failure
            if _tmp_png and os.path.exists(_tmp_png):
                os.remove(_tmp_png)
            for p in _tmp_wavs:
                if os.path.exists(p):
                    os.remove(p)

        return (frames_f32, audio_out)


# ---------------------------------------------------------------------------
# Node 4: NAVASaveVideo
# ---------------------------------------------------------------------------

class NAVASaveVideo:
    """
    Merge video frames and audio into an MP4 file and save to disk.

    Input  : frames (IMAGE batch [T,H,W,C]) + audio (AUDIO dict)
    Output : STRING file path of the saved MP4

    When running inside ComfyUI the file is written to ComfyUI's output
    directory by default (via folder_paths). Falls back to ./nava_outputs/
    when folder_paths is unavailable (standalone use).
    """

    CATEGORY = "NAVA"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "audio":  ("AUDIO",),
            },
            "optional": {
                "fps": (
                    "INT",
                    {"default": 24, "min": 1, "max": 60,
                     "tooltip": "Output video frame rate"},
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "nava",
                     "tooltip": "Prefix for the output filename (a timestamp is appended)"},
                ),
                "output_dir": (
                    "STRING",
                    {"default": "",
                     "tooltip": "Output directory. Leave empty to use ComfyUI's default "
                                "output folder (or ./nava_outputs/ outside ComfyUI)."},
                ),
                "video_quality_crf": (
                    "INT",
                    {"default": 18, "min": 0, "max": 51,
                     "tooltip": "H.264 CRF value: lower = better quality, larger file (0=lossless)"},
                ),
            },
        }

    def save(
        self,
        frames: torch.Tensor,
        audio: dict,
        fps: int = 24,
        filename_prefix: str = "nava",
        output_dir: str = "",
        video_quality_crf: int = 18,
    ):
        import time
        from torchvision.io import write_video

        # Resolve output directory. We want it to be inside ComfyUI's output
        # folder when possible — that's the only path the web UI can serve via
        # /view?filename=...&type=output for in-node preview.
        in_comfy_output = False
        subfolder = ""
        if not output_dir:
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
                in_comfy_output = True
            except ImportError:
                output_dir = os.path.join(os.getcwd(), "nava_outputs")
        else:
            # User supplied a path — check whether it lives under ComfyUI's
            # output dir so we can still serve a preview.
            try:
                import folder_paths
                comfy_out = os.path.abspath(folder_paths.get_output_directory())
                abs_user = os.path.abspath(output_dir)
                if abs_user == comfy_out:
                    in_comfy_output = True
                elif abs_user.startswith(comfy_out + os.sep):
                    in_comfy_output = True
                    subfolder = os.path.relpath(abs_user, comfy_out)
            except ImportError:
                pass
        os.makedirs(output_dir, exist_ok=True)

        timestamp = int(time.time() * 1000)
        filename = f"{filename_prefix}_{timestamp}.mp4"
        output_path = os.path.join(output_dir, filename)

        # frames: [T, H, W, C] float32 0-1  →  uint8
        video_uint8 = (frames.cpu().float() * 255).clamp(0, 255).to(torch.uint8)
        T, H, W, C = video_uint8.shape

        # audio: ComfyUI AUDIO is {"waveform": [B, C, L], "sample_rate": int}
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.dim() == 3:
            waveform = waveform[0]   # [C, L]
        waveform = waveform.cpu().float().contiguous()

        print(f"[NAVA] Writing video: {T} frames {W}x{H} @ {fps}fps, "
              f"audio {tuple(waveform.shape)} @ {sample_rate}Hz")

        # Write video + audio. If the muxed write fails (codec/sample-rate
        # quirks with pyav/AAC on certain rates), retry without audio rather
        # than producing a 0-byte file.
        try:
            write_video(
                output_path,
                video_uint8,              # [T, H, W, C] uint8
                fps=fps,
                video_codec="h264",
                audio_array=waveform,     # [C, L] float32
                audio_fps=sample_rate,
                audio_codec="aac",
                options={"crf": str(video_quality_crf)},
            )
        except Exception as e:
            print(f"[NAVA] Muxed write failed ({e}); writing video-only and "
                  f"a sidecar WAV instead.")
            write_video(
                output_path, video_uint8, fps=fps, video_codec="h264",
                options={"crf": str(video_quality_crf)},
            )
            # Save a sidecar WAV next to the MP4 so audio isn't lost.
            try:
                import torchaudio
                wav_path = os.path.splitext(output_path)[0] + ".wav"
                torchaudio.save(wav_path, waveform, sample_rate)
                print(f"[NAVA] Sidecar audio: {wav_path}")
            except Exception as e2:
                print(f"[NAVA] Sidecar WAV write also failed: {e2}")

        size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        print(f"[NAVA] Saved: {output_path}  ({size/1e6:.2f} MB)")
        if size == 0:
            print("[NAVA] WARNING: output file is 0 bytes — encoder likely "
                  "failed silently. Check ffmpeg/pyav installation.")

        # ------------------------------------------------------------------
        # UI preview payload.
        #
        # Vanilla ComfyUI's frontend only renders these `ui` keys natively:
        #   - "images"   (list of image refs, will display as <img>)
        #   - "text"
        # Video previews are provided by ComfyUI-VideoHelperSuite (VHS) which
        # consumes  ui.gifs = [{filename, subfolder, type, format, frame_rate}]
        # We emit BOTH:
        #   - ui.gifs  : VHS picks this up and shows an inline player
        #   - ui.images: a single-frame PNG poster so something shows up
        #                even when VHS isn't installed
        # The file itself is served by ComfyUI's /view endpoint when type=output
        # and the path lives under folder_paths.get_output_directory().
        # ------------------------------------------------------------------
        ui = {}
        if in_comfy_output:
            ui["gifs"] = [{
                "filename": filename,
                "subfolder": subfolder,
                "type": "output",
                "format": "video/h264-mp4",
                "frame_rate": fps,
            }]
            # Drop a poster PNG so the node shows something without VHS.
            try:
                from PIL import Image
                poster_name = f"{filename_prefix}_{timestamp}_poster.png"
                poster_path = os.path.join(output_dir, poster_name)
                Image.fromarray(video_uint8[0].numpy()).save(poster_path)
                ui["images"] = [{
                    "filename": poster_name,
                    "subfolder": subfolder,
                    "type": "output",
                }]
            except Exception:
                pass
        else:
            # File is outside ComfyUI's output dir — UI cannot fetch it via
            # /view, so we just surface the path as text.
            ui["text"] = [output_path]

        return {"ui": ui, "result": (output_path,)}


# ---------------------------------------------------------------------------
# Node 5: NAVAImageCaptioner
# ---------------------------------------------------------------------------

class NAVAImageCaptioner:
    """
    Caption a ComfyUI IMAGE with Qwen3-VL-4B-Instruct.

    Produces a short Chinese visual description (subject / scene / framing /
    mood) suitable as the "scene" half of a NAVA T2AV prompt. Pair with
    NAVAPromptCompose to splice in user-provided speech wrapped in <S>...<E>,
    then feed into NAVAPromptRewriter.
    """

    CATEGORY = "NAVA"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "model_path": (
                    "STRING",
                    {"default": "pe_src/Qwen3-VL-4B-Instruct", "multiline": False,
                     "tooltip": "Path to Qwen3-VL weights (relative to NAVA root or absolute), "
                                "or a HuggingFace repo id."},
                ),
                "use_4bit": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Load in 4-bit (bitsandbytes nf4) to save VRAM."},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 256, "min": 32, "max": 1024, "step": 16},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "0 = greedy decoding."},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**31 - 1},
                ),
                "offload_after": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Push the VL model to CPU after captioning to free VRAM "
                                "for the downstream rewriter / NAVA sampler."},
                ),
            },
        }

    def caption_image(
        self,
        image,
        model_path: str = "pe_src/Qwen3-VL-4B-Instruct",
        use_4bit: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        seed: int = 42,
        offload_after: bool = True,
    ):
        from .captioner import caption as _do_caption, offload_all_to_cpu, reload_to_gpu

        reload_to_gpu(model_path, use_4bit)
        text = _do_caption(
            image=image,
            model_path=model_path,
            use_4bit=use_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
        print(f"[NAVA-Captioner] OUT ({len(text)} chars):\n{text}")
        if offload_after:
            offload_all_to_cpu()
        return (text,)


# ---------------------------------------------------------------------------
# Node 6: NAVAPromptCompose
# ---------------------------------------------------------------------------

class NAVAPromptCompose:
    """
    Compose a NAVA T2AV prompt from a visual caption + spoken text.

    Modes:
      - single_speaker  (default): wraps `speech` in one <S>...<E>, joins with caption
      - multi_speaker            : parses `dialogue` (one line per utterance,
                                   `角色描述||台词`) into multiple <S>...<E> pairs
      - silent                   : ignores speech, appends "No dialogue or voiceover throughout."
      - caption_only             : returns the caption alone

    NAVA's convention: each utterance gets its OWN <S>...<E>; same speaker's
    continuous run stays in one pair. Multi-speaker prompts also need explicit
    role descriptions before each `<S>...<E>` ("男青年压低声音说<S>...<E>").
    """

    CATEGORY = "NAVA"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "compose"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": (
                    "STRING",
                    {"multiline": True, "default": "", "forceInput": True,
                     "tooltip": "Scene description from NAVAImageCaptioner (or type manually)."},
                ),
                "mode": (
                    ["single_speaker", "multi_speaker", "silent"],
                    {"default": "single_speaker",
                     "tooltip": "single_speaker: one person speaks — fill speech below.\n"
                                "multi_speaker: two or more people — fill dialogue below.\n"
                                "silent: no speech, environment audio only."},
                ),
                "speech": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "[single_speaker] The spoken line — just the words, no tags.\n"
                                "Example: When I snap my finger, half of the people in this universe dies.\n"
                                "<S>...<E> is added automatically."},
                ),
                "dialogue": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "[multi_speaker] Write each line with role description + <S>...<E>.\n"
                                "Example:\n"
                                "Character A leans in and says<S>Drop the weapon. Now.<E> "
                                "Character B smirks<S>You really think this ends here?<E>"},
                ),
            },
        }

    @staticmethod
    def _normalize_dialogue(dialogue: str) -> str:
        if not dialogue:
            return ""
        parts = [ln.strip() for ln in dialogue.splitlines() if ln.strip()]
        return " ".join(parts)

    def compose(
        self,
        caption: str,
        mode: str = "single_speaker",
        speech: str = "",
        dialogue: str = "",
    ):
        cap = (caption or "").strip()

        if mode == "silent":
            out = f"{cap} No dialogue or voiceover throughout." if cap else "No dialogue or voiceover throughout."
        elif mode == "multi_speaker":
            dlg = self._normalize_dialogue(dialogue)
            out = f"{cap} {dlg}".strip() if cap else dlg
        else:  # single_speaker
            spk = (speech or "").strip()
            out = f"{cap} <S>{spk}<E>".strip() if spk else cap

        out = out.strip()
        print(f"[NAVA-Compose] mode={mode} OUT:\n{out}")
        return (out,)


# ---------------------------------------------------------------------------
# Node 7: NAVAShowText  — display any STRING in the node body
# ---------------------------------------------------------------------------

class NAVAShowText:
    CATEGORY = "NAVA"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
            "optional": {
                "display": ("STRING", {"multiline": True, "default": "（结果显示在这里）"}),
            },
        }

    def show(self, text: str, unique_id=None, extra_pnginfo=None, display=None):
        # Write the result back into the workflow node's widget so it persists
        # in the canvas even without a ui.text-capable frontend extension.
        if extra_pnginfo and isinstance(extra_pnginfo, dict):
            workflow = extra_pnginfo.get("workflow", {})
            for node in workflow.get("nodes", []):
                if str(node.get("id")) == str(unique_id):
                    node.setdefault("widgets_values", [])
                    if node["widgets_values"]:
                        node["widgets_values"][0] = text
                    else:
                        node["widgets_values"].append(text)
        return {"ui": {"text": [text]}, "result": (text,)}

    @classmethod
    def IS_CHANGED(cls, text, **_):
        return text


# ---------------------------------------------------------------------------
# Registration map — imported by __init__.py
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NAVAModelLoader":    NAVAModelLoader,
    "NAVAPromptRewriter": NAVAPromptRewriter,
    "NAVAImageCaptioner": NAVAImageCaptioner,
    "NAVAPromptCompose":  NAVAPromptCompose,
    "NAVASampler":        NAVASampler,
    "NAVASaveVideo":      NAVASaveVideo,
    "NAVAShowText":       NAVAShowText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NAVAModelLoader":    "NAVA Model Loader",
    "NAVAPromptRewriter": "NAVA Prompt Rewriter",
    "NAVAImageCaptioner": "NAVA Image Captioner (Qwen3-VL)",
    "NAVAPromptCompose":  "NAVA Prompt Compose",
    "NAVASampler":        "NAVA Sampler",
    "NAVASaveVideo":      "NAVA Save Video",
    "NAVAShowText":       "NAVA Show Text",
}
