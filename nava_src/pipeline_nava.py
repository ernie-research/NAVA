import math
import torch
from diffusers import DiffusionPipeline, DDPMScheduler, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from .scheduler.flow_match import FlowMatchScheduler
from transformers import AutoTokenizer
from .model_nava import NAVA
from .vae.vae import AutoencoderKL, DiffusersVAEAdapter
from .models.nava.modules.t5 import T5EncoderModel
from .models.nava.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import numpy as np
import os


class AudioVideoPipeline(DiffusionPipeline):
    # ✅ 只接“组件”，不要接 model_id / model_path 这类构建参数
    def __init__(self, model, audio_vae, video_vae, image_vae, scheduler):
        super().__init__()
        self.register_modules(
            model=model,
            audio_vae=audio_vae,
            video_vae=video_vae,
            image_vae=image_vae,
            scheduler=scheduler,
        )
        self.audio_latent_ch = None
        self.video_latent_ch = None
        self.text_model = None

    @classmethod
    def create(cls, model_id: str, use_bf16: bool = True,
                 audio_latent_ch: int = 20, video_latent_ch: int = 48,
                 lambda_ddpm: float = 5.0, cfg=None, device="cpu"):
        audio_vae, video_vae, image_vae = None, None, None
        tgt_dtype = torch.bfloat16 if use_bf16 else torch.float16

        if cfg["data"].get("use_server", False):
            raise RuntimeError(
                "cfg.data.use_server=True but the remote VAE-server path was removed. "
                "Set use_server: false in the config and use local VAEs."
            )
        ckpt_dir = cfg["model"].get("ckpt_dir", "./")
        if "video" in cfg["modality"]:
            from nava_src.models.nava.utils.model_loading_utils import init_wan_vae_2_2
            from nava_src.vae.local_video_vae import LocalVideoVAEAdapter
            wan_vae = init_wan_vae_2_2(ckpt_dir, rank=device)
            wan_vae.model.requires_grad_(False).eval()
            wan_vae.model = wan_vae.model.to(torch.bfloat16)
            video_vae = LocalVideoVAEAdapter(wan_vae, resolution=cfg["image_size"])
        if "audio" in cfg["modality"]:
            from nava_src.vae.local_audio_vae import LocalAudioVAEAdapter, init_ltx_vae
            audio_vae_ckpt_dir = cfg["model"].get("audio_vae_ckpt_dir", "./params")
            ltx_vae = init_ltx_vae(audio_vae_ckpt_dir, device=device)
            spk_model = None
            try:
                spk_model = torch.hub.load(
                    'IDRnD/ReDimNet', 'ReDimNet',
                    model_name='M', train_type='ft_mix', dataset='vb2+vox2+cnc'
                ).eval().to(device)
                print(f"[AudioVAE] ReDimNet speaker model loaded successfully on {device}")
            except Exception as e:
                print(f"[AudioVAE] WARNING: Failed to load ReDimNet speaker model: {e}")
                spk_model = None
            audio_vae = LocalAudioVAEAdapter(ltx_vae, spk_model=spk_model, sample_rate=16000)

        shift = cfg["model"].get('shift', 5.0)
        shift_audio = cfg["model"].get('shift_audio', 5.0)
        num_train_timesteps = cfg["model"].get("num_train_timesteps", 1000)
        scheduler_personalized = cfg.get('scheduler_personalized', False)
        scheduler_unipc = cfg.get('scheduler_unipc', False)
        if scheduler_personalized:
            scheduler = FlowMatchScheduler(
                num_train_timesteps=num_train_timesteps,
                shift=shift,
                extra_one_step=True 
            )
            scheduler_audio = FlowMatchScheduler(
                num_train_timesteps=num_train_timesteps,
                shift=shift_audio,
                extra_one_step=True 
            )

            if scheduler_unipc:
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=1000,
                    shift=shift,
                    use_dynamic_shifting=False)
                sample_scheduler_audio = FlowUniPCMultistepScheduler(
                    num_train_timesteps=1000,
                    shift=shift_audio,
                    use_dynamic_shifting=False)
            else:
                sample_scheduler = scheduler
                sample_scheduler_audio = scheduler_audio
        else:
            if os.path.exists(f"{model_id}/scheduler"):
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            else:
                scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=num_train_timesteps,
                    shift=shift
                )

        if cfg.get('model_type') == "NAVA":
            ckpt_dir = cfg["model"].get("ckpt_dir", "nava_src/models/nava/ckpts")
            wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
            text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
            text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")
            text_model = T5EncoderModel(
                text_len=512,
                dtype=tgt_dtype,
                device=device,
                checkpoint_path=text_encoder_path,
                tokenizer_path=text_tokenizer_path,
                cpu_offload=cfg.get("cpu_offload", False),
                shard_fn=None)
            text_encoder = text_model.model
            text_encoder.requires_grad_(False)
            text_encoder = text_encoder.to(torch.bfloat16).eval()
            text_encoder = torch.compile(text_encoder)   # 只 compile encoder
            text_model.model = text_encoder
            masking_modality_prob = cfg.get('masking_modality_prob', 0.0)
            i2v_mode_prob = cfg.get('i2v_mode_prob', 0.0)

            model = NAVA(
                lambda_ddpm=lambda_ddpm,
                target_dtype=tgt_dtype,
                config=cfg,
            )
        else:
            raise ValueError("model type {} not supported".format(cfg['model_type']))

        pipe = cls(
            model=model,
            audio_vae=audio_vae,
            video_vae=video_vae,
            image_vae=image_vae,
            scheduler=scheduler,
        )
        pipe.cfg = cfg
        pipe.scheduler_personalized = scheduler_personalized
        pipe.scheduler_unipc = scheduler_unipc
        pipe.scheduler_audio = scheduler_audio
        pipe.sample_scheduler = sample_scheduler
        pipe.sample_scheduler_audio = sample_scheduler_audio
        pipe.tgt_dtype = tgt_dtype
        pipe.audio_latent_ch = audio_latent_ch
        pipe.video_latent_ch = video_latent_ch
        pipe.text_model = text_model
        pipe.masking_modality_prob = masking_modality_prob
        pipe.i2v_mode_prob = i2v_mode_prob

        return pipe
    
    def switch_training_mode(self):
        """
        switch training mode
        """
        if self.scheduler_personalized:
            # 训练模式强制inference-steps为1000，初始化reweighter
            self.scheduler.set_timesteps(1000, training=True)
            self.scheduler_audio.set_timesteps(1000, training=True)

    def forward(self, batch, global_step = None):
        """
        训练端调用：batch 包含 audio_latents / video_latents
        """
        device = self._get_device()
        self.text_model.model.to(device)

        audio_latents = batch["audio_latents"]
        video_latents = batch["video_latents"]
        image_latents = batch["image_latents"]
        spk_embs = batch["spk_embs"]

        has_audio = audio_latents is not None
        has_video = video_latents is not None
        has_image = image_latents is not None
        if has_video and has_audio:
            # 生成一个 0-1 的随机数
            masking_modality = (torch.rand(1, device=device).item() < self.masking_modality_prob)
        else:
            masking_modality = False
        
        if has_video:
            is_i2v = (torch.rand(1, device=device).item() < self.i2v_mode_prob)
        else:
            is_i2v = False

        batch_size = len(batch["captions"])
        assert batch_size > 0, "batch size must be greater than 0"
        audio_input, video_input, image_input = None, None, None
        audio_target, video_target, image_target = None, None, None
        audio_context = [] if has_audio else None
        vid_context = [] if has_video else None
        image_context = [] if has_image else None
        text_lens = []
        t_h_w_list = batch.get("t_h_w_list", None)

        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,))
        t = self.scheduler.timesteps[timestep_id].to(dtype=torch.float32, device=device)
        loss_reweight = self.scheduler.training_weight(t).to(device=device)
        
        drop_probs = torch.rand(batch_size, device=device)
        drop_indices = drop_probs < 0.1
        if drop_indices.all():
            drop_indices[0] = False

        with torch.no_grad():
            text_list, text_lens, spk_pos = self.text_model(batch["captions"], device, return_seqlens=True, return_spk_pos=True)   # [B, L, C], [B, L]
        cur_context = [
            torch.zeros_like(ctx) if drop_indices[i] else ctx
            for i, ctx in enumerate(text_list)
        ]
        spk_pos = [
            pos if not drop_indices[i] else []
            for i, pos in enumerate(spk_pos)
        ]
        if has_audio:
            spk_embs = [
                emb
                for i, emb_list in enumerate(spk_embs)
                if not drop_indices[i]
                for emb in emb_list
            ]
            if len(spk_embs) > 0:
                spk_embs = torch.cat(spk_embs, dim=0)
            else:
                spk_embs = None
        else:
            spk_embs = None

        if has_audio:
            audio_context = cur_context
        if has_video:
            vid_context = cur_context
        if has_image:
            image_context = cur_context
    
        # add text lens for speed loggin
        batch["text_lens"] = text_lens

        if audio_latents is not None:
            audio_scaling_factor = self.audio_vae.config.scaling_factor
            audio_shift_factor = self.audio_vae.config.shift_factor
            if isinstance(audio_latents, list):
                audio_latents = [
                    x.to(device).to(self.tgt_dtype) 
                    for x in audio_latents
                ]
                audio_z0 = [
                    (x - audio_shift_factor) * audio_scaling_factor 
                    for x in audio_latents
                ]
                audio_noise = [
                    torch.randn_like(x) for x in audio_z0
                ]
            else:
                audio_latents = audio_latents.to(device).to(self.tgt_dtype)
                audio_z0 = (audio_latents - audio_shift_factor) * audio_scaling_factor
                audio_noise = torch.randn_like(audio_z0)
            audio_input = self.scheduler_audio.add_noise_batch(audio_z0, audio_noise, t)
            audio_target = self.scheduler_audio.training_target(audio_z0, audio_noise, t)

        if spk_embs is not None:
            spk_embs = spk_embs.to(device).to(self.tgt_dtype)

        if video_latents is not None:
            video_scaling_factor = self.video_vae.config.scaling_factor
            video_shift_factor = self.video_vae.config.shift_factor
            if isinstance(video_latents, list):
                video_latents = [
                    x.to(device).to(self.tgt_dtype) 
                    for x in video_latents
                ]
                video_z0 = [
                    (x - video_shift_factor) * video_scaling_factor 
                    for x in video_latents
                ]
                video_noise = [
                    torch.randn_like(x) for x in video_z0
                ]
            else:
                video_latents = video_latents.to(device).to(self.tgt_dtype)
                video_z0 = (video_latents - video_shift_factor) * video_scaling_factor
                video_noise = torch.randn_like(video_z0)
            video_input = self.scheduler.add_noise_batch(video_z0, video_noise, t)
            if is_i2v:
                for b_idx in range(batch_size):
                    video_input[b_idx][:1] = video_z0[b_idx][:1]
            video_target = self.scheduler.training_target(video_z0, video_noise, t)

        if image_latents is not None:
            image_scaling_factor = self.image_vae.config.scaling_factor
            image_shift_factor = self.image_vae.config.shift_factor
            if isinstance(image_latents, list):
                image_latents = [
                    x.to(device).to(self.tgt_dtype) 
                    for x in image_latents
                ]
                image_z0 = [
                    (x - image_shift_factor) * image_scaling_factor 
                    for x in image_latents
                ]
                image_noise = [
                    torch.randn_like(x) for x in image_z0
                ]
            else:
                image_latents = image_latents.to(device).to(self.tgt_dtype)
                image_z0 = (image_latents - image_shift_factor) * image_scaling_factor
                image_noise = torch.randn_like(image_z0)
            image_input = self.scheduler.add_noise_batch(image_z0, image_noise, t)
            image_target = self.scheduler.training_target(image_z0, image_noise, t)

        loss, logs = self.model(
            audio_context=audio_context,
            image_context=image_context,
            vid_context=vid_context,
            audio_input=audio_input,
            image_input=image_input,
            video_input=video_input,
            audio_target=audio_target,
            image_target=image_target,
            video_target=video_target,
            spk_embed=spk_embs,
            spk_pos=spk_pos,
            timesteps=t,
            t_h_w_list=t_h_w_list,
            loss_reweight=loss_reweight,
            masking_modality=masking_modality,
            is_i2v=is_i2v,
        )
    
        return loss, logs

    @torch.no_grad()
    def sample(
        self,
        batch,
        num_steps: int = 25,
        audio_guidance_scale: float = 4.0,
        image_guidance_scale: float = 5.0,
        video_guidance_scale: float = 5.0,
        num_samples: int = None,
        negative_prompt_mode: bool = True,
        align_3d_cfg: bool = False,
        audio_align_guidance_scale: float = 4.0,
        image_align_guidance_scale: float = 5.0,
        video_align_guidance_scale: float = 5.0,
        is_i2v: bool = False,
        save_vid_latent: bool = False,
        timbre_cfg: bool = False,
        timbre_align_guidance_scale: float = 2.0,
        offload_backbone: bool = False,
        tiled_vae: bool = False,
        vae_tile_size: tuple = (44, 80),
        vae_tile_stride: tuple = (28, 52),
    ):
        # num_steps = 1000
        """
        从 batch 条件生成图像。
        - height/width 不传时，会尝试从 batch["images"] 或 VAE config 里推断。
        返回 [-1, 1] 的 [B, 3, H, W]。
        """

        device = next(self._unwrap_model().parameters()).device
        dtype  = next(self._unwrap_model().parameters()).dtype

        modaility = self.cfg.get("modality", "audio")
        has_audio = "audio" in modaility and batch.get("audio_latents", None) is not None
        has_image = "image" in modaility and batch.get("image_latents", None) is not None
        has_video = "video" in modaility and batch.get("video_latents", None) is not None
        has_vision = has_image or has_video

        b = len(batch["captions"])

        audio_pos_context, vision_pos_context = [], []
        audio_neg_context, vision_neg_context = [], []
        spk_pos = []

        audio_negative_prompt = "机械音、闷糊、回音、失真、电流声、爆音、杂音"
        if self.cfg.get("use_mmdit_model", False):
            video_negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, 音频带有机械音、闷糊、回音、失真、电流声、爆音、杂音"
        else:
            video_negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"        # text_embeddings_audio_neg, text_embeddings_image_neg, text_embeddings_video_neg = \

        # T5 offload: move encoder to GPU only for the duration of text encoding
        if getattr(self, '_t5_offload', False):
            self.text_model.model.to(device)

        for caption in batch["captions"]:
            with torch.no_grad():
                text_embeddings_cond, cur_spk_pos = self.text_model([caption], device, return_spk_pos=True)
                text_embeddings_cond = text_embeddings_cond[0]
                cur_spk_pos = cur_spk_pos[0]
            # use padding zeros as uncondition input
            if not negative_prompt_mode:
                text_embeddings_uncond = torch.zeros_like(text_embeddings_cond).to(device)
                audio_neg_context.append(text_embeddings_uncond)
                vision_neg_context.append(text_embeddings_uncond)
            else:
                text_embeddings_video_neg, text_embeddings_audio_neg = self.text_model([video_negative_prompt, audio_negative_prompt], device)
                audio_neg_context.append(text_embeddings_audio_neg)
                vision_neg_context.append(text_embeddings_video_neg)
            audio_pos_context.append(text_embeddings_cond)
            vision_pos_context.append(text_embeddings_cond)
            spk_pos.append(cur_spk_pos)

        if getattr(self, '_t5_offload', False):
            self.text_model.model.to("cpu")
            torch.cuda.empty_cache()

        audio_vae_dim = self.audio_latent_ch
        video_vae_dim = self.video_latent_ch

        t_h_w_list = batch.get("t_h_w_list", None)
        first_frames = batch.get("first_frames", None)
        if t_h_w_list is not None:
            all_image_len = torch.tensor(t_h_w_list).prod(dim=1).sum()
            latents_vision = torch.randn((all_image_len, video_vae_dim), device=device, dtype=torch.float32)
            if is_i2v:
                assert first_frames is not None
                

        audio_len_list = None
        audio_latents = batch.get("audio_latents", None)
        spk_embs = batch.get("spk_embs", None)
        if spk_embs is not None:
            spk_embs = [
                emb
                for emb_list in spk_embs
                for emb in emb_list
            ]
            if len(spk_embs) > 0:
                spk_embs = torch.cat(spk_embs, dim=0)
            else:
                spk_embs = None
        if audio_latents is not None:
            audio_len_list = torch.tensor([len(_) for _ in audio_latents], 
                                          dtype=torch.int, device=device).unsqueeze(1)
            all_audio_len = audio_len_list.sum()
            latents_audio = torch.randn((all_audio_len, audio_vae_dim), device=device, dtype=torch.float32)
        
        if spk_embs is not None:
            spk_embs = spk_embs.to(device).to(torch.float32)

        if not has_vision:
            latents_vision = None
            vision_pos_context, vision_neg_context = None, None

        if not has_audio:
            latents_audio = None
            audio_pos_context, audio_neg_context = None, None
        # latents = latents * self.scheduler.init_noise_sigma

        # 4) Flow matching 采样
        self.sample_scheduler.set_timesteps(num_steps)
        self.sample_scheduler_audio.set_timesteps(num_steps)
        timesteps = self.sample_scheduler.timesteps
        effective_timbre = timbre_cfg and spk_embs is not None

        from tqdm.auto import tqdm
        for t_v, t_a in tqdm(zip(timesteps, timesteps)):
            t_v = t_v.to(device=device)
            t_a = t_a.to(device=device)
            if audio_guidance_scale == 1.0 or video_guidance_scale == 1.0:
                # 不使用cfg
                assert False
            else:
                vis_pos_context = vision_pos_context if has_vision else None
                vis_neg_context = vision_neg_context if has_vision else None
                eps_cond_vid, eps_cond_audio = self._unwrap_model().predict_eps(
                    audio_context=audio_pos_context,
                    vid_context=vis_pos_context,
                    latents_audio=latents_audio,
                    latents_vid=latents_vision,
                    timesteps=t_v.unsqueeze(0),
                    audio_len_list=audio_len_list,
                    spk_embs=spk_embs,
                    spk_pos=spk_pos,
                    t_h_w_list=t_h_w_list,
                    masking_modality=False,
                    is_i2v=is_i2v,
                    first_frames=first_frames,
                )
                eps_uncond_vision, eps_uncond_audio = self._unwrap_model().predict_eps(
                    vid_context=vis_neg_context,
                    audio_context=audio_neg_context,
                    latents_vid=latents_vision,
                    latents_audio=latents_audio,
                    timesteps=t_v.unsqueeze(0),
                    t_h_w_list=t_h_w_list,
                    audio_len_list=audio_len_list,
                    spk_embs=None,
                    slg_layer=11,
                    masking_modality=False,
                    is_i2v=is_i2v,
                    first_frames=first_frames,
                )
                if align_3d_cfg:
                    eps_mmask_cond_vid, eps_mmask_cond_audio = self._unwrap_model().predict_eps(
                        audio_context=audio_pos_context,
                        vid_context=vis_pos_context,
                        latents_audio=latents_audio,
                        latents_vid=latents_vision,
                        timesteps=t_v.unsqueeze(0),
                        audio_len_list=audio_len_list,
                        spk_embs=spk_embs,
                        t_h_w_list=t_h_w_list,
                        masking_modality=True,
                        is_i2v=is_i2v,
                        first_frames=first_frames,
                    )
                if effective_timbre:
                    eps_timbre_uncond_vid, eps_timbre_uncond_audio = self._unwrap_model().predict_eps(
                        audio_context=audio_pos_context,
                        vid_context=vis_pos_context,
                        latents_audio=latents_audio,
                        latents_vid=latents_vision,
                        timesteps=t_v.unsqueeze(0),
                        audio_len_list=audio_len_list,
                        spk_embs=None,
                        spk_pos=spk_pos,
                        t_h_w_list=t_h_w_list,
                        masking_modality=False,
                        is_i2v=is_i2v,
                        first_frames=first_frames,
                    )

                if has_vision:
                    vision_guidance_scale = video_guidance_scale if has_video else image_guidance_scale
                    vision_align_guidance_scale = video_align_guidance_scale if has_video else image_align_guidance_scale
                    # 加入timebre control的 if else 
                    # eps_vision = eps_cond_vid + vision_guidance_scale * (eps_cond_vid - eps_uncond_vision) + vision_align_guidance_scale * (eps_cond_vid - eps_mmask_cond_vid) + timbre_align_guidance_scale * (eps_cond_vid - eps_timbre_uncond_vid)
                    if not align_3d_cfg:
                        eps_vision = eps_uncond_vision + vision_guidance_scale * (eps_cond_vid - eps_uncond_vision)
                    else:
                        eps_vision = eps_cond_vid + vision_guidance_scale * (eps_cond_vid - eps_uncond_vision) + vision_align_guidance_scale * (eps_cond_vid - eps_mmask_cond_vid)
                    latents_vision = self.sample_scheduler.step(eps_vision.to(torch.float32), t_v, latents_vision.to(torch.float32), return_dict=False)
                    latents_vision = latents_vision[0] if self.scheduler_unipc else latents_vision
                if has_audio:
                    if not align_3d_cfg and not effective_timbre:
                        eps_audio = eps_uncond_audio + audio_guidance_scale * (eps_cond_audio - eps_uncond_audio)
                    elif align_3d_cfg and not effective_timbre:
                        eps_audio = eps_cond_audio + audio_guidance_scale * (eps_cond_audio - eps_uncond_audio) + audio_align_guidance_scale * (eps_cond_audio - eps_mmask_cond_audio)
                    elif effective_timbre and not align_3d_cfg:
                        eps_audio = eps_cond_audio + audio_guidance_scale * (eps_cond_audio - eps_uncond_audio) + timbre_align_guidance_scale * (eps_cond_audio - eps_timbre_uncond_audio)
                    else:  # align_3d_cfg and effective_timbre
                        eps_audio = eps_cond_audio + audio_guidance_scale * (eps_cond_audio - eps_uncond_audio) + audio_align_guidance_scale * (eps_cond_audio - eps_mmask_cond_audio) + timbre_align_guidance_scale * (eps_cond_audio - eps_timbre_uncond_audio)
                    latents_audio = self.sample_scheduler_audio.step(eps_audio.to(torch.float32), t_a, latents_audio.to(torch.float32), return_dict=False)
                    latents_audio = latents_audio[0] if self.scheduler_unipc else latents_audio

        # Offload backbone to CPU before VAE decode to free GPU memory
        if offload_backbone:
            if getattr(self, '_group_offload', False):
                # Blocks are in pinned CPU buffers managed by hooks.
                # Only move non-block modules (heads, embedders, etc.) to CPU.
                for name, module in self.model.backbone.named_children():
                    if name not in ('double_blocks', 'single_blocks', 'double_final_blocks'):
                        module.to("cpu")
                # Also ensure any block that is still on GPU gets moved to CPU
                # (the last loaded group may still have GPU tensors).
                for blk in (list(self.model.backbone.double_blocks) +
                            list(self.model.backbone.single_blocks) +
                            list(self.model.backbone.double_final_blocks)):
                    for p in blk.parameters():
                        if p.data.is_cuda:
                            p.data = p.data.cpu()
            else:
                self.model.backbone.to("cpu")
            torch.cuda.empty_cache()

        # 5) decode 成图
        start_idx = 0
        imgs, audio_list = None, None
        if has_vision:
            latents = latents_vision
            vision_vae = self.image_vae if has_image else self.video_vae
            latents = (latents / vision_vae.config.scaling_factor) + vision_vae.config.shift_factor
            img_list = []
            t_h_w_list = t_h_w_list[:num_samples] if num_samples else t_h_w_list
            min_frames = (min([t for t, _, _ in t_h_w_list]) - 1) * 4 + 1
            if not save_vid_latent:
                for t, h, w in t_h_w_list:
                    img_latent = latents[start_idx: start_idx + t * h * w].view(t, h, w, video_vae_dim)
                    dec = vision_vae.decode(img_latent, cpu_offload=offload_backbone,
                                            tiled=tiled_vae, tile_size=vae_tile_size,
                                            tile_stride=vae_tile_stride)
                    img = dec.sample if hasattr(dec, "sample") else dec#(1,3,h,w)
                    while img.shape[1] == 1 and img.shape[2] == 1:
                        import time
                        time.sleep(0.2)
                        print("retry decoding for generation cases")
                        dec = vision_vae.decode(img_latent, tiled=tiled_vae,
                                                tile_size=vae_tile_size,
                                                tile_stride=vae_tile_stride)
                        img = dec.sample if hasattr(dec, "sample") else dec#(1,3,h,w)
                    # img=img.permute(0, 3, 1, 2)
                    if img.shape[0] == 1:
                        # single image: keep original resolution
                        img_list.append(img)
                    else:
                        # video: keep original resolution, no resize
                        vid = img[:min_frames].unsqueeze(0)
                        img_list.append(vid)

                    start_idx += t * h * w
                imgs = torch.cat(img_list, dim=0) #(B,3,256,256) or (B,T,3,256,256)
            else:
                for t, h, w in t_h_w_list:
                    img_latent = latents[start_idx: start_idx + t * h * w].view(t, h, w, video_vae_dim)
                    img_list.append(img_latent.unsqueeze(0))
                imgs = torch.cat(img_list, dim=0) #(B,3,256,256) or (B,T,3,256,256)
        if has_audio:
            audio_list = []
            start_idx = 0
            latents_audio = (latents_audio / self.audio_vae.config.scaling_factor) + self.audio_vae.config.shift_factor
            audio_len_list = audio_len_list[:num_samples] if num_samples else audio_len_list
            for audio_len in audio_len_list:
                audio_latent_tensor = latents_audio[start_idx: start_idx + audio_len].permute(1, 0).unsqueeze(0)
                decode_dict = self.audio_vae.decode(audio_latent_tensor)
                decode_dict = decode_dict.sample if hasattr(decode_dict, "sample") else decode_dict
                audio_list.append(decode_dict)
                start_idx += audio_len

        # Reload backbone for next sample after decode
        if offload_backbone:
            if getattr(self, '_group_offload', False):
                # blocks are managed by hooks; only reload non-block static parts to GPU
                for name, module in self.model.backbone.named_children():
                    if name not in ('double_blocks', 'single_blocks', 'double_final_blocks'):
                        module.to(device)
            else:
                self.model.backbone.to(device)

        return imgs, audio_list

    def _unwrap_model(self):
        # 多卡时 self.model 是 DDP，单卡时就是原始模型
        m = self.model
        return m.module if hasattr(m, "module") else m

    def _get_device(self):
        # diffusers 推荐：优先用 _execution_device；没有就从模型参数上推断
        try:
            return self._execution_device
        except AttributeError:
            return next(self._unwrap_model().parameters()).device