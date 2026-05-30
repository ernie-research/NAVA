"""
NAVA inference engine wrapper.
Encapsulates pipeline init, checkpoint loading, SP patching, and single-sample generation.
"""

import os
import math
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
    from nava_src.models.nava.modules.model_mm import (
        WanAttentionBlock,
        WanDoubleStreamAttentionBlock,
    )
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
                 rank: int, world_size: int, use_sp: bool = True):
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

        # Inference params from config
        self.fps = self.cfg["data"].get("video_fps", 24)
        self.audio_tokens_per_sec = self.cfg["data"].get("audio_tokens_per_sec", 25)
        self.video_latent_ch = self.cfg["video_latent_ch"]
        self.height = self.cfg.get("log_height", 480)
        self.width = self.cfg.get("log_width", 832)
        self.frames = self.cfg["data"].get("video_tgt_frames", 121)
        self.patch_size = self.cfg.get("patch_size", 2)

        self.dtype = torch.bfloat16 if self.cfg["use_bf16"] else torch.float16

        if rank == 0:
            print(f"[Engine] Ready. modality={self.modality}, "
                  f"resolution={self.width}x{self.height}, frames={self.frames}")

    def _build_batch(self, prompt: str, image_path: str = None, spk_wav_paths: list = None):
        """Build a single-sample batch dict from raw inputs."""
        # Compute latent dimensions
        h = self.height // self.patch_size
        w = self.width // self.patch_size
        frames = self.frames

        # Audio length based on video duration
        video_duration = ((frames - 1) * 4 + 1) / self.fps
        audio_len = math.ceil(video_duration * self.audio_tokens_per_sec)

        batch = {
            "captions": [prompt],
            "video_latents": torch.randn(1, frames * h * w, self.video_latent_ch),
            "audio_latents": [torch.randn(audio_len, self.video_latent_ch)],
            "t_h_w_list": [(frames, h, w)],
            "first_frames": None,
            "spk_embs": None,
            "save_path": ["gradio_output.mp4"],
        }

        # Handle i2v (first frame image)
        if image_path and os.path.exists(image_path):
            from torchvision import transforms
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            img = img.resize((self.width, self.height))
            img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]
            # Encode through video VAE
            img_tensor = img_tensor.to(self.device, dtype=self.dtype)
            with torch.no_grad():
                first_frame_latent = self.pipe.video_vae.encode(img_tensor)
            batch["first_frames"] = first_frame_latent

        # Handle speaker embeddings
        if spk_wav_paths:
            spk_embs_list = []
            for wav_path in spk_wav_paths:
                if os.path.exists(wav_path):
                    waveform, sr = torchaudio.load(wav_path)
                    with torch.no_grad():
                        spk_emb = self.pipe.audio_vae.get_speaker_embedding(
                            waveform.to(self.device), sr
                        )
                    spk_embs_list.append(spk_emb)
            if spk_embs_list:
                batch["spk_embs"] = [spk_embs_list]

        return batch

    @torch.no_grad()
    def generate(self, prompt: str, image_path: str = None, spk_wav_paths: list = None,
                 steps: int = 25, output_dir: str = "/tmp/nava_outputs",
                 is_i2v: bool = False) -> str:
        """
        Run single inference. All ranks must call this together in SP mode.
        Returns: output video path (only meaningful on rank 0).
        """
        os.makedirs(output_dir, exist_ok=True)

        batch = self._build_batch(prompt, image_path, spk_wav_paths)
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        amp_ctx = torch.autocast(device_type="cuda", dtype=self.dtype)

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
            )

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
