"""
Single-GPU NAVA inference engine for ComfyUI custom nodes.

Design notes:
  - No torchrun / distributed required; runs in-process inside ComfyUI.
  - Follows inference_nava.py for offload logic (t5_offload, group_offload, vae_tiling).
  - apply_group_offload is copied from inference_nava.py — no changes to existing code.
  - Model instances are cached by (ckpt_path, config_path, offload options).
    Resolution / latent_frames are NOT part of the cache key — they are picked
    at sampling time, not at model load.
"""

import math
import os
import sys
import importlib
from typing import Optional

import torch
import yaml

# ---------------------------------------------------------------------------
# Ensure NAVA repo root is importable when this package lives inside it.
# Layout assumed:  <nava_root>/comfyui_nava/engine.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.realpath(__file__))
NAVA_ROOT = os.path.dirname(_HERE)
if NAVA_ROOT not in sys.path:
    sys.path.insert(0, NAVA_ROOT)

# ---------------------------------------------------------------------------
# Module-level model cache: avoids reloading 24 GB weights on every queue run.
# Key: tuple of (abs_ckpt_path, abs_config_path,
#                t5_offload, group_offload, offload_group_size)
# Resolution and latent_frames are sampler-time parameters and intentionally
# NOT part of the cache key.
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _to01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x.float() + 1.0) / 2.0, 0.0, 1.0)


def _toWav(x: torch.Tensor) -> torch.Tensor:
    peak = x.abs().max().clamp(min=1e-12)
    return (x * (0.95 / peak)).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# apply_group_offload — copied verbatim from inference_nava.py so this package
# requires no modifications to the existing codebase.
# Pipelined CPU↔GPU offload for DiT backbone blocks via pinned memory + async
# CUDA stream; see inline comments in inference_nava.py for full explanation.
# ---------------------------------------------------------------------------

def apply_group_offload(backbone, group_size: int, device):
    all_blocks = (
        list(backbone.double_blocks)
        + list(backbone.single_blocks)
        + list(backbone.double_final_blocks)
    )
    groups = [all_blocks[i : i + group_size] for i in range(0, len(all_blocks), group_size)]
    n_groups = len(groups)
    blk_idx = {id(b): i for i, b in enumerate(all_blocks)}

    for blk in all_blocks:
        blk.to("cpu")
    cpu_bufs: list = []
    for blk in all_blocks:
        d: dict = {}
        for name, p in blk.named_parameters(recurse=True):
            d[name] = p.data.pin_memory()
            p.data = d[name]
        cpu_bufs.append(d)
    torch.cuda.empty_cache()

    _param_cache = [list(blk.named_parameters(recurse=True)) for blk in all_blocks]
    xfer_stream = torch.cuda.Stream(device=device)

    def _restore_pinned(gi: int):
        for b in groups[gi]:
            idx = blk_idx[id(b)]
            for name, p in _param_cache[idx]:
                if not p.data.is_cuda:
                    p.data = cpu_bufs[idx][name]

    def _load(gi: int):
        with torch.cuda.stream(xfer_stream):
            for b in groups[gi]:
                b.to(device, non_blocking=True)

    def _store(gi: int):
        for b in groups[gi]:
            idx = blk_idx[id(b)]
            for name, p in _param_cache[idx]:
                if p.data.is_cuda:
                    p.data = cpu_bufs[idx][name]

    _load(0)
    torch.cuda.current_stream().wait_stream(xfer_stream)

    handles = []
    for gi, group in enumerate(groups):
        prev_gi = (gi - 1 + n_groups) % n_groups
        nxt_gi = (gi + 1) % n_groups

        def make_pre(cur_gi: int, p_gi: int, n_gi: int):
            def pre(module, args):
                first_param = next(groups[cur_gi][0].parameters(), None)
                if first_param is not None and not first_param.data.is_cuda:
                    _restore_pinned(cur_gi)
                    _load(cur_gi)
                    torch.cuda.current_stream().wait_stream(xfer_stream)
                else:
                    torch.cuda.current_stream().wait_stream(xfer_stream)
                _store(p_gi)
                _load(n_gi)
                return args
            return pre

        handles.append(group[0].register_forward_pre_hook(make_pre(gi, prev_gi, nxt_gi)))

    return handles


# ---------------------------------------------------------------------------
# NAVAComfyEngine — single-GPU inference engine
# ---------------------------------------------------------------------------

class NAVAComfyEngine:
    """
    Single-GPU NAVA inference engine for ComfyUI nodes.

    Supports T5 offload, DiT group offload, and VAE tiling.
    Does NOT use torchrun / sequence parallel — runs in a single process.
    """

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        t5_offload: bool = True,
        group_offload: bool = False,
        offload_group_size: int = 10,
        weight_dtype: str = "auto",
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.t5_offload = t5_offload
        self.group_offload = group_offload
        self.weight_dtype = weight_dtype

        print(f"[NAVA] Loading config: {config_path}")
        self.cfg = yaml.safe_load(open(config_path, "r"))
        self.modality = self.cfg.get("modality", "audio_video")

        from nava_src.utils.common import set_seed
        set_seed(self.cfg.get("seed", 42))

        # Build pipeline
        module_path, class_name = self.cfg["pipeline"].rsplit(".", 1)
        PipelineClass = getattr(importlib.import_module(module_path), class_name)
        if "video" in self.modality and "audio" in self.modality:
            self.cfg["init_from_meta"] = True

        print(f"[NAVA] Creating pipeline: {class_name}")
        self.pipe = PipelineClass.create(
            model_id=self.cfg["model_id"],
            use_bf16=self.cfg["use_bf16"],
            audio_latent_ch=self.cfg["audio_latent_ch"],
            video_latent_ch=self.cfg["video_latent_ch"],
            lambda_ddpm=self.cfg["lambda_ddpm"],
            cfg=self.cfg,
            device=device,
        )

        # Load checkpoint; fall back from .safetensors → .ckpt if needed
        if not os.path.exists(ckpt_path):
            fallback = os.path.splitext(ckpt_path)[0] + ".ckpt"
            if os.path.exists(fallback):
                print(f"[NAVA] {ckpt_path} not found, using {fallback}")
                ckpt_path = fallback
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (also tried {fallback})")

        print(f"[NAVA] Loading checkpoint: {ckpt_path}")
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file as _sf_load
            state_dict = _sf_load(ckpt_path, device="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True)["state_dict"]

        # ----- fp8 detection / patching (mirrors inference_nava.py) -----
        # If the checkpoint contains float8_e4m3fn tensors, swap every block-Linear
        # in the freshly-built bf16 model with FP8Linear so load_state_dict can
        # populate `weight` (fp8) and `weight_scale` (bf16) buffers correctly.
        is_fp8_ckpt = any(
            isinstance(v, torch.Tensor) and v.dtype == torch.float8_e4m3fn
            for v in state_dict.values()
        )
        if weight_dtype == "fp8_e4m3fn":
            use_fp8 = True
        elif weight_dtype == "bf16":
            use_fp8 = False
        else:  # auto
            use_fp8 = is_fp8_ckpt

        if use_fp8 and not is_fp8_ckpt:
            print("[NAVA][WARN] weight_dtype=fp8_e4m3fn but checkpoint contains no fp8 "
                  "tensors. Patching anyway; load will likely report missing *_scale keys.")
        if not use_fp8 and is_fp8_ckpt:
            print("[NAVA][WARN] Checkpoint is fp8 but weight_dtype=bf16 was requested. "
                  "Skipping the fp8 patch — outputs will be wrong. Use 'auto' or 'fp8_e4m3fn'.")

        if use_fp8:
            from NAVA_FP8 import patch_model_to_fp8
            n_patched = patch_model_to_fp8(self.pipe.model)
            n_fp8_keys = sum(
                1 for v in state_dict.values()
                if isinstance(v, torch.Tensor) and v.dtype == torch.float8_e4m3fn
            )
            print(f"[NAVA] fp8 mode: patched {n_patched} Linear modules; "
                  f"checkpoint has {n_fp8_keys} fp8 tensors")
        self._fp8_active = use_fp8

        missing, unexpected = self.pipe.model.load_state_dict(state_dict, strict=False)
        print(f"[NAVA] Checkpoint loaded. missing={len(missing)}, unexpected={len(unexpected)}")

        self.pipe = self.pipe.to(device)
        self.pipe.model.eval()
        self.pipe.model.backbone.set_rope_params()

        # Mirror the attribute flags that pipeline.sample() reads internally
        self.pipe._t5_offload = t5_offload
        self.pipe._group_offload = group_offload

        # T5 offload: move text encoder to CPU; pipeline reloads it during encoding
        if t5_offload:
            self.pipe.text_model.model.to("cpu")
            torch.cuda.empty_cache()
            print("[NAVA] T5 text encoder offloaded to CPU (~32 GB freed)")

        # Group offload: install paged CPU↔GPU hooks on all DiT backbone blocks
        if group_offload:
            apply_group_offload(self.pipe.model.backbone, offload_group_size, device)
            total = (
                len(self.pipe.model.backbone.double_blocks)
                + len(self.pipe.model.backbone.single_blocks)
                + len(self.pipe.model.backbone.double_final_blocks)
            )
            print(f"[NAVA] DiT group offload enabled: {total} blocks, group_size={offload_group_size}")

        # Inference-time constants derived from config
        self.fps = self.cfg["data"].get("video_fps", 24)
        self.audio_tokens_per_sec = self.cfg["data"].get("audio_tokens_per_sec", 25)
        self.patch_size = self.cfg.get("spatial_downsample", 16)
        self.dtype = torch.bfloat16 if self.cfg["use_bf16"] else torch.float16

        print(
            f"[NAVA] Engine ready. modality={self.modality}, device={device}. "
            f"Resolution / latent_frames are picked at sample time."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_batch(
        self,
        prompt: str,
        image_path: Optional[str],
        spk_wav_paths: Optional[list],
        is_i2v: bool,
        height: int,
        width: int,
        latent_frames: int,
    ) -> dict:
        h = height // self.patch_size
        w = width // self.patch_size

        # latent_frames is the temporal LATENT dim that goes into the DiT.
        # Actual decoded video frames = (latent_frames - 1) * 4 + 1
        # because the video VAE temporally downsamples by 4 (with the +1 for
        # the anchor frame).
        actual_frames = (latent_frames - 1) * 4 + 1

        # Audio length must cover the full video duration
        video_duration = actual_frames / self.fps
        audio_len = math.ceil(video_duration * self.audio_tokens_per_sec)

        video_latents = torch.randn((latent_frames, h, w, 48))
        img_latents = None

        if is_i2v and image_path and os.path.exists(image_path):
            img_latents = self.pipe.video_vae.encode(
                image_path, target_height=height, target_width=width
            ).latent_dist.sample()
            # Update spatial dims to match VAE-encoded image
            video_latents = torch.randn(
                (latent_frames, img_latents.shape[1], img_latents.shape[2], 48)
            )

        audio_latents = torch.randn((audio_len, 48))

        spk_embs = None
        if spk_wav_paths:
            valid = [p for p in spk_wav_paths if p and os.path.exists(p)]
            if valid:
                spk_embs = []
                for wav_path in valid:
                    query = {"data_path": wav_path, "use_spk_emb": True}
                    result = self.pipe.audio_vae.encode(query).latent_dist.sample()
                    spk_embs.append(result["spk_embs"])

        # Insert <extra_id_2> marker after <S> tags for speaker-position detection
        # (mirrors T2AVDataset preprocessing)
        prompt = prompt.replace("<S>", "<S><extra_id_2>")

        return {
            "idx": 0,
            "video_latents": video_latents,
            "first_frames": img_latents,
            "audio_latents": audio_latents,
            "save_path": "comfyui_output.mp4",
            "captions": prompt,
            "spk_embs": spk_embs,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        latent_frames: int,
        image_path: Optional[str] = None,
        spk_wav_paths: Optional[list] = None,
        steps: int = 50,
        is_i2v: bool = False,
        video_cfg: float = 3.0,
        audio_cfg: float = 2.0,
        image_cfg: Optional[float] = None,
        align_3d_cfg: Optional[bool] = None,
        video_align_cfg: Optional[float] = None,
        audio_align_cfg: Optional[float] = None,
        image_align_cfg: Optional[float] = None,
        timbre_cfg: Optional[bool] = None,
        timbre_align_cfg: Optional[float] = None,
        seed: int = 42,
        vae_tiling: bool = False,
        vae_tile_size: tuple = (22, 40),
        vae_tile_stride: tuple = (14, 26),
    ):
        """
        Run single-GPU inference and return decoded outputs.

        Args:
            height, width   : pixel resolution. Must be divisible by patch_size.
            latent_frames   : number of LATENT temporal frames fed to the DiT.
                              Decoded output frame count is
                                  actual_frames = (latent_frames - 1) * 4 + 1
                              e.g. latent_frames=37 → 145 frames ≈ 6 s @ 24 fps.

        Returns:
            frames_f32  : torch.Tensor [T, H, W, C] float32 0-1
                          (ComfyUI IMAGE batch — T frames treated as batch dim)
            audio_out   : dict {"waveform": Tensor[1, C, L] float32,
                                "sample_rate": int}
                          (ComfyUI native AUDIO format)
        """
        from nava_src.utils.common import set_seed
        set_seed(seed)

        sample = self._build_batch(
            prompt, image_path, spk_wav_paths, is_i2v,
            height, width, latent_frames,
        )

        from nava_src.data.t2v import collate_fn
        batch = collate_fn([sample])
        batch = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            try:
                from comfy.utils import ProgressBar
                _pbar = ProgressBar(steps)
                _progress_cb = lambda done, total: _pbar.update(1)
            except Exception:
                _progress_cb = None

            # Resolve cfg params: explicit kwargs override config defaults.
            _align3d = align_3d_cfg if align_3d_cfg is not None else self.cfg.get("align_3d_cfg", True)
            _vid_align = video_align_cfg if video_align_cfg is not None else self.cfg.get("video_align_guidance_scale", 3.0)
            _aud_align = audio_align_cfg if audio_align_cfg is not None else self.cfg.get("audio_align_guidance_scale", 2.0)
            _img_align = image_align_cfg if image_align_cfg is not None else self.cfg.get("image_align_guidance_scale", 5.0)
            _img_cfg = image_cfg if image_cfg is not None else self.cfg.get("image_guidance_scale", 5.0)
            _timbre = timbre_cfg if timbre_cfg is not None else self.cfg.get("timbre_cfg", False)
            _timbre_align = timbre_align_cfg if timbre_align_cfg is not None else self.cfg.get("timbre_align_guidance_scale", 3.0)

            gen_vid, gen_aud = self.pipe.sample(
                batch,
                num_steps=steps,
                video_guidance_scale=video_cfg,
                audio_guidance_scale=audio_cfg,
                image_guidance_scale=_img_cfg,
                align_3d_cfg=_align3d,
                video_align_guidance_scale=_vid_align,
                audio_align_guidance_scale=_aud_align,
                image_align_guidance_scale=_img_align,
                save_vid_latent=False,
                is_i2v=is_i2v,
                timbre_cfg=_timbre,
                timbre_align_guidance_scale=_timbre_align,
                offload_backbone=self.t5_offload or self.group_offload,
                tiled_vae=vae_tiling,
                vae_tile_size=vae_tile_size,
                vae_tile_stride=vae_tile_stride,
                progress_callback=_progress_cb,
            )

        # Video: [T, H, W, C] float32 0-1 (ComfyUI IMAGE batch over T frames)
        frames_f32 = _to01(gen_vid).float()[0].permute(0, 2, 3, 1).cpu()

        # Audio: ComfyUI AUDIO format requires waveform shape [B, C, L]
        raw_aud = gen_aud[0]
        waveform = _toWav(raw_aud["waveform"])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)       # [C, L]
        waveform = waveform.unsqueeze(0).cpu().float()  # [1, C, L]
        audio_out = {"waveform": waveform, "sample_rate": raw_aud["sample_rate"]}

        return frames_f32, audio_out


# ---------------------------------------------------------------------------
# Factory: return cached engine or load a new one
# ---------------------------------------------------------------------------

def get_or_load_engine(
    ckpt_path: str,
    config_path: str,
    t5_offload: bool = True,
    group_offload: bool = False,
    offload_group_size: int = 10,
    weight_dtype: str = "auto",
) -> NAVAComfyEngine:
    """Return a cached NAVAComfyEngine or instantiate a new one.

    Resolution and latent_frames are sampler-time parameters and are NOT used
    here — the same loaded engine handles any resolution / frame count.
    """
    key = (
        os.path.abspath(ckpt_path),
        os.path.abspath(config_path),
        t5_offload, group_offload, offload_group_size,
        weight_dtype,
    )
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = NAVAComfyEngine(
            config_path=config_path,
            ckpt_path=ckpt_path,
            t5_offload=t5_offload,
            group_offload=group_offload,
            offload_group_size=offload_group_size,
            weight_dtype=weight_dtype,
        )
    return _MODEL_CACHE[key]
