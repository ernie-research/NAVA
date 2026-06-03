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


class NAVAEngine:
    def __init__(self, config_path: str, ckpt_path: str, device: torch.device,
                 rank: int, world_size: int, use_sp: bool = True,
                 height: int = 704, width: int = 1280, frames: int = 37):
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

        # Offload backbone to CPU after init — only load to GPU when generating
        self.pipe.model.backbone.to("cpu")
        torch.cuda.empty_cache()
        self._backbone_on_gpu = False

        if rank == 0:
            print(f"[Engine] Ready. modality={self.modality}, "
                  f"resolution={self.width}x{self.height}, frames={self.frames}")
            print(f"[Engine] Backbone offloaded to CPU (will reload to GPU on generate)")

    def reload_backbone(self):
        """Move backbone to GPU for diffusion sampling."""
        if not self._backbone_on_gpu:
            self.pipe.model.backbone.to(self.device)
            self._backbone_on_gpu = True

    def offload_backbone(self):
        """Move backbone to CPU to free GPU memory."""
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

        # Insert <extra_id_2> after <S> for spk_pos detection (align with T2AVDataset)
        prompt = prompt.replace("<S>", "<S><extra_id_2>")

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
                 frames: int = None) -> str:
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
                audio_guidance_scale=self.cfg.get("audio_guidance_scale", 2.0),
                video_guidance_scale=self.cfg.get("video_guidance_scale", 3.0),
                align_3d_cfg=self.cfg.get("align_3d_cfg", True),
                audio_align_guidance_scale=self.cfg.get("audio_align_guidance_scale", 2.0),
                video_align_guidance_scale=self.cfg.get("video_align_guidance_scale", 3.0),
                save_vid_latent=False,
                is_i2v=is_i2v,
                timbre_cfg=self.cfg.get("timbre_cfg", False),
                timbre_align_guidance_scale=self.cfg.get("timbre_align_guidance_scale", 3.0),
                offload_backbone=True,
                vae_cpu_offload=False,
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
