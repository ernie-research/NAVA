"""
NAVA inference engine wrapper for Gradio demo.
Handles pipeline init, checkpoint loading, SP patching, and single-sample generation.
Supports: text-to-AV, image-to-AV (i2v), up to 2 speaker reference WAVs.
"""

import os
import math
import subprocess
import importlib
import torch
import torch.distributed as dist
import yaml
import torchaudio
from torchvision.io import write_video

from nava_src.utils.common import set_seed
from nava_src.models.nava.utils.model_loading_utils import load_fusion_checkpoint


def _to01(x):
    return torch.clamp((x.float() + 1.0) / 2.0, 0.0, 1.0)


def _toWav(x):
    peak = x.abs().max().clamp(min=1e-12)
    x = x * (0.95 / peak)
    return x.clamp(-1.0, 1.0)


def _convert_backbone_to_sp(backbone):
    from nava_src.models.nava.modules.model_mm_sp import (
        WanDoubleStreamSelfAttentionSP,
        WanSelfAttentionSP,
        _swap_self_attn,
    )
    for blk in list(backbone.double_blocks) + list(backbone.double_final_blocks):
        _swap_self_attn(blk, WanDoubleStreamSelfAttentionSP)
    for blk in backbone.single_blocks:
        _swap_self_attn(blk, WanSelfAttentionSP)


def apply_group_offload(backbone, group_size: int, device):
    """Pipelined CPU↔GPU offload for DiT backbone blocks.

    Mirrors inference_nava.apply_group_offload — see that file's docstring for
    the design rationale (pinned host memory, dedicated xfer stream, async
    pre-hook prefetch, self-heal between samples).
    """
    all_blocks = (
        list(backbone.double_blocks) +
        list(backbone.single_blocks) +
        list(backbone.double_final_blocks)
    )
    groups = [all_blocks[i:i + group_size] for i in range(0, len(all_blocks), group_size)]
    n_groups = len(groups)
    blk_idx = {id(b): i for i, b in enumerate(all_blocks)}

    for blk in all_blocks:
        blk.to("cpu")
    cpu_bufs: list[dict] = []
    for blk in all_blocks:
        d: dict = {}
        for name, p in blk.named_parameters(recurse=True):
            d[name] = p.data.pin_memory()
            p.data = d[name]
        cpu_bufs.append(d)
    torch.cuda.empty_cache()

    _param_cache = [
        list(blk.named_parameters(recurse=True)) for blk in all_blocks
    ]

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
        nxt_gi  = (gi + 1) % n_groups

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


class NAVAEngine:
    def __init__(self, config_path: str, ckpt_path: str, device: torch.device,
                 rank: int, world_size: int, use_sp: bool = True,
                 height: int = 704, width: int = 1280, frames: int = 37,
                 weight_dtype: str = "auto",
                 t5_offload: bool = False,
                 group_offload: bool = False,
                 offload_group_size: int = 1,
                 vae_tiling: bool = False,
                 vae_tile_size: tuple = (22, 40),
                 vae_tile_stride: tuple = (14, 26)):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.use_sp = use_sp

        # Load config
        self.cfg = yaml.safe_load(open(config_path, "r"))
        self.modality = self.cfg.get("modality", "audio_video")

        set_seed(self.cfg.get("seed", 42))

        # SP init
        if use_sp:
            from nava_src.models.nava.distributed_comms.parallel_states import (
                initialize_sequence_parallel_state,
            )
            initialize_sequence_parallel_state(world_size)
            if rank == 0:
                print(f"[SP] Sequence parallel enabled, sp_size={world_size}")

        # Load pipeline
        module_path, class_name = self.cfg["pipeline"].rsplit(".", 1)
        PipelineClass = getattr(importlib.import_module(module_path), class_name)
        if "video" in self.modality and "audio" in self.modality:
            self.cfg["init_from_meta"] = True

        self.pipe = PipelineClass.create(
            model_id=self.cfg["model_id"],
            use_bf16=self.cfg["use_bf16"],
            audio_latent_ch=self.cfg["audio_latent_ch"],
            video_latent_ch=self.cfg["video_latent_ch"],
            lambda_ddpm=self.cfg["lambda_ddpm"],
            cfg=self.cfg,
            device=device,
        )

        # Load checkpoint — prefer .safetensors, fall back to .ckpt
        if not os.path.exists(ckpt_path):
            ckpt_fallback = os.path.splitext(ckpt_path)[0] + ".ckpt"
            if os.path.exists(ckpt_fallback):
                if rank == 0:
                    print(f"[Engine] {ckpt_path} not found, falling back to {ckpt_fallback}")
                ckpt_path = ckpt_fallback
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (also tried {ckpt_fallback})")

        if "video" in self.modality and "audio" in self.modality and not self.cfg.get("use_mmdit_model", False):
            load_fusion_checkpoint(self.pipe.model, checkpoint_path=ckpt_path, from_meta=True)
        else:
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file as _sf_load
                state_dict = _sf_load(ckpt_path, device="cpu")
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

            # ----- fp8 detection / patching (mirrors inference_nava.py) -----
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

            if use_fp8 and not is_fp8_ckpt and rank == 0:
                print("[Engine] WARN: weight_dtype=fp8_e4m3fn but ckpt has no fp8 tensors. "
                      "Patching anyway; load will likely report missing *_scale keys.")
            if not use_fp8 and is_fp8_ckpt and rank == 0:
                print("[Engine] WARN: ckpt is fp8 but weight_dtype=bf16 was requested. "
                      "Skipping fp8 patch — outputs will be wrong. Did you mean 'auto'?")

            if use_fp8:
                from NAVA_FP8 import patch_model_to_fp8
                n_patched = patch_model_to_fp8(self.pipe.model)
                if rank == 0:
                    n_fp8_keys = sum(
                        1 for v in state_dict.values()
                        if isinstance(v, torch.Tensor) and v.dtype == torch.float8_e4m3fn
                    )
                    print(f"[Engine] fp8 mode: patched {n_patched} Linear modules; "
                          f"ckpt has {n_fp8_keys} fp8 tensors")

            missing, unexpected = self.pipe.model.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print(f"[Engine] missing: {missing}, unexpected: {unexpected}")

        self.pipe = self.pipe.to(device)
        self.pipe.model.eval()
        self.pipe.model.backbone.set_rope_params()

        # SP patching
        if use_sp:
            _convert_backbone_to_sp(self.pipe.model.backbone)
            if rank == 0:
                print(f"[SP] Patched backbone blocks to SP-aware self-attn.")

        # Inference params (can be overridden per-call)
        self.fps = self.cfg["data"].get("video_fps", 24)
        self.audio_tokens_per_sec = self.cfg["data"].get("audio_tokens_per_sec", 25)
        self.video_latent_ch = self.cfg["video_latent_ch"]
        self.height = height
        self.width = width
        self.frames = frames
        self.patch_size = self.cfg.get("spatial_downsample", 16)
        self.resolution = self.pipe.video_vae.resolution if hasattr(self.pipe.video_vae, 'resolution') else 960

        self.dtype = torch.bfloat16 if self.cfg["use_bf16"] else torch.float16

        # Save VAE tiling knobs for use in generate()
        self._vae_tiling = vae_tiling
        self._vae_tile_size = tuple(vae_tile_size)
        self._vae_tile_stride = tuple(vae_tile_stride)
        # Pipeline-internal flags also read by pipe.sample()
        self.pipe._t5_offload = t5_offload
        self.pipe._group_offload = group_offload

        # T5 offload: encoder moves to GPU only during text encoding
        if t5_offload:
            self.pipe.text_model.model.to("cpu")
            torch.cuda.empty_cache()
            if rank == 0:
                print("[Offload] T5 CPU offload enabled")

        # DiT group offload: each block group page-cycles CPU↔GPU. Once enabled,
        # group hooks own backbone placement — skip the coarse-grained
        # self.pipe.model.backbone.to("cpu") path below, otherwise the hook's
        # pinned-buf pointers fight with the bulk move.
        self._group_offload = group_offload
        if group_offload:
            apply_group_offload(self.pipe.model.backbone, offload_group_size, device)
            if rank == 0:
                total = (len(self.pipe.model.backbone.double_blocks) +
                         len(self.pipe.model.backbone.single_blocks) +
                         len(self.pipe.model.backbone.double_final_blocks))
                print(f"[Offload] DiT group offload enabled: {total} blocks, group_size={offload_group_size}")

        # Coarse-grained backbone offload — only when group_offload is OFF.
        # When enabled, every reload_backbone()/offload_backbone() call moves
        # the whole backbone in one shot; mutually exclusive with group_offload.
        if not group_offload:
            self.pipe.model.backbone.to("cpu")
            torch.cuda.empty_cache()
            self._backbone_on_gpu = False
        else:
            self._backbone_on_gpu = True  # group hooks manage placement

        if rank == 0:
            print(f"[Engine] Ready. modality={self.modality}, "
                  f"resolution={self.width}x{self.height}, frames={self.frames}")
            if not group_offload:
                print(f"[Engine] Backbone offloaded to CPU (will reload to GPU on generate)")

    def reload_backbone(self):
        """Move backbone to GPU for diffusion sampling.
        No-op when group_offload owns placement (its forward hooks page blocks
        in/out per-group; bulk moves would clobber the pinned-buf pointers)."""
        if self._group_offload:
            return
        if not self._backbone_on_gpu:
            self.pipe.model.backbone.to(self.device)
            self._backbone_on_gpu = True

    def offload_backbone(self):
        """Move backbone to CPU to free GPU memory.
        No-op when group_offload owns placement."""
        if self._group_offload:
            return
        if self._backbone_on_gpu:
            self.pipe.model.backbone.to("cpu")
            torch.cuda.empty_cache()
            self._backbone_on_gpu = False

    def _get_spk_embs(self, spk_wav_paths: list) -> list:
        """
        Get speaker embeddings from local WAV files via ReDimNet speaker model.
        Returns list of Tensor(1, 192), same format as T2AVDataset.
        """
        spk_embs_list = []
        for wav_path in spk_wav_paths:
            if not wav_path or not os.path.exists(wav_path):
                spk_embs_list.append(torch.zeros((1, 192), dtype=torch.float32))
                continue

            # LocalAudioVAEAdapter.encode accepts local path via "data_path" key
            query = {
                "data_path": wav_path,
                "use_spk_emb": True,
            }
            result = self.pipe.audio_vae.encode(query).latent_dist.sample()
            spk_embs = result["spk_embs"]  # Tensor(1, 192)
            spk_embs_list.append(spk_embs)

        return spk_embs_list

    def _get_first_frame(self, image_path: str, target_height: int = None, target_width: int = None):
        """
        Encode first frame image via local video VAE.
        Returns img_latents tensor [1, h_latent, w_latent, z_dim].
        """
        img_latents = self.pipe.video_vae.encode(
            image_path, target_height=target_height, target_width=target_width
        ).latent_dist.sample()
        return img_latents

    def _build_batch(self, prompt: str, image_path: str = None,
                     spk_wav_paths: list = None, is_i2v: bool = False,
                     height: int = None, width: int = None):
        """Build a single-sample batch dict from raw inputs."""
        # Use per-call h/w or fall back to engine defaults
        height = height or self.height
        width = width or self.width
        h = height // self.patch_size
        w = width // self.patch_size
        frames = self.frames

        # Audio length based on video duration
        video_duration = ((frames - 1) * 4 + 1) / self.fps
        audio_len = math.ceil(video_duration * self.audio_tokens_per_sec)

        # Default video latents (random noise, shape determines output size)
        video_latents = torch.randn((frames, h, w, 48))

        # Handle first frame (i2v)
        img_latents = None
        if is_i2v and image_path and os.path.exists(image_path):
            img_latents = self._get_first_frame(image_path, target_height=height, target_width=width)
            # Update video_latents shape to match encoded image dimensions
            video_latents = torch.randn((frames, img_latents.shape[1], img_latents.shape[2], 48))

        audio_latents = torch.randn((audio_len, 48))

        # Handle speaker embeddings (0-2 speakers)
        spk_embs = None
        if spk_wav_paths:
            valid_paths = [p for p in spk_wav_paths if p and os.path.exists(p)]
            if valid_paths:
                spk_embs = self._get_spk_embs(valid_paths)

        # Insert <extra_id_2> after <S> for spk_pos detection (align with T2AVDataset).
        # Idempotent: strip any pre-existing markers first, so callers that already
        # injected (e.g. gradio rewrite_fn shows the marked form to the user) and
        # callers that didn't (raw user prompt, hand-edited textbox) both end up at
        # exactly one <extra_id_2> per <S> here.
        prompt = prompt.replace("<extra_id_2>", "").replace("<S>", "<S><extra_id_2>")

        batch = {
            "idx": 0,
            "video_latents": video_latents,
            "first_frames": img_latents,
            "audio_latents": audio_latents,
            "save_path": "gradio_output.mp4",
            "captions": prompt,
            "spk_embs": spk_embs,
        }

        return batch

    def _collate_single(self, sample: dict) -> dict:
        """Collate a single sample into batch format (mimics collate_fn for bs=1)."""
        from nava_src.data.t2v import collate_fn
        return collate_fn([sample])

    @torch.no_grad()
    def generate(self, prompt: str, image_path: str = None, spk_wav_paths: list = None,
                 steps: int = 50, output_dir: str = "/tmp/nava_outputs",
                 is_i2v: bool = False, height: int = None, width: int = None,
                 frames: int = None,
                 video_cfg: float = None, audio_cfg: float = None,
                 video_align_cfg: float = None, audio_align_cfg: float = None,
                 align_3d_cfg: bool = None, timbre_cfg: bool = None,
                 timbre_align_cfg: float = None) -> str:
        """
        Run single inference. All ranks must call this together in SP mode.
        Returns: output video path (only meaningful on rank 0).
        """
        # Pick a fresh random seed every request. In SP mode all ranks must use
        # the SAME seed so the per-step noise lines up — rank 0 generates it and
        # broadcasts; rank 1-7 receive and apply.
        if self.use_sp:
            seed_t = torch.empty(1, dtype=torch.long, device=self.device)
            if self.rank == 0:
                seed_t.fill_(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            dist.broadcast(seed_t, src=0)
            seed = int(seed_t.item())
        else:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        if self.rank == 0:
            print(f"[Engine] Random seed for this request: {seed}")
        set_seed(seed)
        # Sync all ranks before inference to ensure clean CUDA state
        if self.use_sp:
            torch.cuda.empty_cache()
            dist.barrier()

        # Per-request frames override
        orig_frames = self.frames
        if frames is not None:
            self.frames = frames

        os.makedirs(output_dir, exist_ok=True)

        sample = self._build_batch(prompt, image_path, spk_wav_paths, is_i2v,
                                    height=height, width=width)
        batch = self._collate_single(sample)
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        amp_ctx = torch.autocast(device_type="cuda", dtype=self.dtype)

        # Reload backbone to GPU for diffusion sampling
        self.reload_backbone()

        with amp_ctx:
            gen_vid_out, gen_aud_out = self.pipe.sample(
                batch,
                num_steps=steps,
                audio_guidance_scale=audio_cfg if audio_cfg is not None else self.cfg.get("audio_guidance_scale", 2.0),
                video_guidance_scale=video_cfg if video_cfg is not None else self.cfg.get("video_guidance_scale", 3.0),
                align_3d_cfg=align_3d_cfg if align_3d_cfg is not None else self.cfg.get("align_3d_cfg", True),
                audio_align_guidance_scale=audio_align_cfg if audio_align_cfg is not None else self.cfg.get("audio_align_guidance_scale", 2.0),
                video_align_guidance_scale=video_align_cfg if video_align_cfg is not None else self.cfg.get("video_align_guidance_scale", 3.0),
                save_vid_latent=False,
                is_i2v=is_i2v,
                timbre_cfg=timbre_cfg if timbre_cfg is not None else self.cfg.get("timbre_cfg", False),
                timbre_align_guidance_scale=timbre_align_cfg if timbre_align_cfg is not None else self.cfg.get("timbre_align_guidance_scale", 3.0),
                offload_backbone=True,
                vae_cpu_offload=False,
                tiled_vae=self._vae_tiling,
                vae_tile_size=self._vae_tile_size,
                vae_tile_stride=self._vae_tile_stride,
                decode=(self.rank == 0),
            )

        # State after pipe.sample (with decode-only-on-rank-0):
        #   - Rank 0: backbone was offloaded → decoded → reloaded to GPU
        #   - Rank 1-7: backbone never moved, still on GPU from sampling
        # Either way, every rank ends with backbone on GPU and ready for the
        # next sample. Mark accordingly.
        self._backbone_on_gpu = True

        # Barrier so rank 1-7 don't race ahead into the next request before
        # rank 0 finishes its VAE decode + save. (Strictly redundant with
        # gradio_server's broadcast loop, but cheap insurance.)
        if self.use_sp:
            dist.barrier()

        # Restore original frames setting
        self.frames = orig_frames

        # Only rank 0 saves
        if self.rank != 0:
            return ""

        # Post-process: merge video + audio → mp4
        import time
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")

        gen_vids = _to01(gen_vid_out).float()
        video_tensor = (gen_vids[0] * 255).clamp(0, 255).to(torch.uint8)
        video_tensor = video_tensor.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]

        aud = gen_aud_out[0]
        waveform = _toWav(aud["waveform"])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        sample_rate = aud["sample_rate"]

        write_video(
            output_path,
            video_tensor,
            fps=self.fps,
            video_codec="h264",
            audio_array=waveform.cpu().float().contiguous(),
            audio_fps=sample_rate,
            audio_codec="aac",
            options={"crf": "18"},
        )

        print(f"[Engine] Saved: {output_path}")
        return output_path
