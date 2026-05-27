import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import math
from einops import rearrange

# from diffusers import AutoencoderKL
from .vae.vae import DiffusersVAEAdapter

from .utils.patchify import LinearPatchProjector
from .utils.mask import make_transfusion_attention_mask

import numpy as np
import os
import json

def save_bias_to_txt(bias: torch.Tensor, tag: str = "bias_debug"):
    """
    把 attention bias 存成 txt，方便排查。

    支持的输入形状：
      - [B, L, L]
      - [B, 1, L, L]
      - [B, H, L, L]

    最终写入文件的是若干个 [L, L] 的矩阵（默认取 head=0）。
    """
    save_dir = "./"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"bias_{tag}.txt")

    with open(save_path, "w") as f:
        f.write("=== attention bias debug ===\n")

        if bias is None:
            f.write("bias is None\n")
            return

        # 移到 CPU，转 numpy
        bias_np = bias.detach().cpu().numpy()
        f.write(f"raw shape: {bias_np.shape}\n")

        ndim = bias_np.ndim

        # 统一转换成 [B, L, L] 的视角
        if ndim == 4:
            # 形状类似 [B, H, L, L] 或 [B, 1, L, L]
            B, H, L, L2 = bias_np.shape
            assert L == L2, f"Last two dims should be square, got {(L, L2)}"
            # 只看第 0 个 head，变成 [B, L, L]
            bias_np = bias_np[:, 0, :, :]
            f.write(f"view as [B, L, L], after taking head 0: {bias_np.shape}\n")

        elif ndim == 3:
            # 已经是 [B, L, L] 形状
            B, L, L2 = bias_np.shape
            assert L == L2, f"Last two dims should be square, got {(L, L2)}"
            f.write(f"view as [B, L, L]: {bias_np.shape}\n")

        elif ndim == 2:
            # 单样本 [L, L]，包一层 batch 维
            L, L2 = bias_np.shape
            assert L == L2, f"Last two dims should be square, got {(L, L2)}"
            bias_np = bias_np[None, ...]  # -> [1, L, L]
            B = 1
            f.write(f"view as [1, L, L]: {bias_np.shape}\n")

        else:
            f.write(f"Unsupported ndim={ndim}, skip saving.\n")
            return

        # 不要省略打印（只是为了调试方便）
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        B = bias_np.shape[0]
        for b in range(B):
            f.write(f"\n[batch {b}]\n")
            mat = bias_np[b]  # [L, L]
            np.savetxt(f, mat, fmt="%.2f")

    print(f"[DEBUG] bias saved to {save_path}")


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.
    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.
    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[..., None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer=nn.GELU,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True, **factory_kwargs
            ),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period
        ).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


from .models.nava.modules.t5 import T5EncoderModel
from .models.nava.modules.fusion import FusionModel
from .models.nava.modules.model_mm import WanAVModel
from .models.nava.utils.model_loading_utils import load_fusion_checkpoint
class NAVA(nn.Module):
    def __init__(
            self, 
            lambda_ddpm: float = 1.0,
            target_dtype=torch.bfloat16,
            config: dict = None,
        ):
        super().__init__()
        self.config = config
        self.lambda_ddpm = lambda_ddpm
        self.target_dtype = target_dtype
        self.use_mmdit_model = config.get("use_mmdit_model", False)
        self.use_loss_reweight = config.get("use_loss_reweight", False)
        self.patch_size = config.get("patch_size", 2)
        self.audio_loss_coff = config.get("audio_loss_coff", 1)
        self.vision_loss_coff = config.get("vision_loss_coff", 1)

        audio_config, video_config = None, None
        audio_latent_ch, video_latent_ch =  None, None
        modaility = config["modality"]
        if self.use_mmdit_model:
            joint_config = config["model"]["joint_config"]
            with open(joint_config) as f:
                joint_config = json.load(f)
            video_latent_ch = joint_config["vid_in_dim"]
            audio_latent_ch = joint_config["audio_in_dim"]
        else:
            if "audio" in modaility:
                audio_config = config["model"]["audio_config"]
                with open(audio_config) as f:
                    audio_config = json.load(f)
                audio_latent_ch = audio_config["in_dim"]
            if "video" in modaility or "image" in modaility:
                video_config = config["model"]["video_config"]
                with open(video_config) as f:
                    video_config = json.load(f)
                video_latent_ch = video_config["in_dim"]
        from_meta = config.get("init_from_meta", False)
        
        backbone = WanAVModel(
            **joint_config,
            gradient_checkpointing=config["model"].get("gradient_checkpointing", False),
            gradient_checkpointing_offload=config["model"].get("gradient_checkpointing_offload", False),
            gradient_checkpoint_every_n=config["model"].get("gradient_checkpoint_every_n", 1),
            add_spk_emb=config["data"].get("add_spk_emb", False),
            no_split_norm_ffn=config.get("no_split_norm_ffn", False))
        params_all = sum(p.numel() for p in backbone.parameters())
        print(f"Score model (MMDIT) all parameters:{params_all}")

        checkpoint_path = config["model"].get("checkpoint_path", None)
        if checkpoint_path:
            load_fusion_checkpoint(backbone, checkpoint_path)
        backbone.set_rope_params()

        self.backbone = backbone
        self.audio_latent_ch = audio_latent_ch
        self.video_latent_ch = video_latent_ch

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        audio_context,
        image_context,
        vid_context,
        audio_input,
        image_input,
        video_input,
        audio_target,
        image_target,
        video_target,
        timesteps=None,
        t_h_w_list=None,
        loss_reweight=None,
        is_i2v=False,
        spk_embed=None,
        spk_pos=None,
        masking_modality=False,
        **kwargs,
    ):
        losses = {}
        loss_audio, loss_image, loss_video, total_loss = 0.0, 0.0, 0.0, 0.0

        has_audio = audio_input is not None
        has_image = image_input is not None
        has_video = video_input is not None
        has_vision = has_image or has_video

        vision_input = image_input if has_image else video_input
        vision_context = image_context if has_image else vid_context
        vision_target = image_target if has_image else video_target

        max_seq_len_vision, max_seq_len_audio = None, None
        batch_size = len(vision_context) if has_vision else len(audio_context)
        if has_audio:
            max_seq_len_audio = max([len(_) for _ in audio_input])
        if has_image or has_video:
            temp_vision_input = []
            temp_vision_target = []
            if not self.use_mmdit_model:
                _patch_size_h, _patch_size_w = self.backbone.video_model.patch_size[1], self.backbone.video_model.patch_size[2]
            else:
                _patch_size_h, _patch_size_w = self.backbone.patch_size[1], self.backbone.patch_size[2]
            max_seq_len_vision = max(
                int((t * math.ceil(h / self.patch_size) * self.patch_size * math.ceil(w / self.patch_size) * self.patch_size) \
                    // (_patch_size_h*_patch_size_w)) for (t, h, w) in t_h_w_list
            )
            
            for i in range(batch_size):
                t, h, w = int(t_h_w_list[i][0]), int(t_h_w_list[i][1]), int(t_h_w_list[i][2])
                vid_sample = vision_input[i].permute(3, 0, 1, 2) # [t, h, w, c] -> [c, t, h, w]
                vid_target = vision_target[i].permute(3, 0, 1, 2)
                pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
                pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
                if pad_h > 0 or pad_w > 0:
                    vid_sample = nn.functional.pad(
                        vid_sample, (0, pad_w, 0, pad_h), mode="constant", value=0
                    )
                temp_vision_input.append(vid_sample)
                temp_vision_target.append(vid_target)
            vision_input = temp_vision_input
            vision_target = temp_vision_target

        pred_vision_list, pred_audio_list = self.backbone(
            vid=vision_input,
            audio=audio_input,
            t=timesteps,
            vid_context=vision_context,
            audio_context=audio_context,
            vid_seq_len=max_seq_len_vision,
            audio_seq_len=max_seq_len_audio,
            spk_embed=spk_embed,
            spk_pos=spk_pos,
            masking_modality=masking_modality,
            first_frame_is_clean=is_i2v,
        )

        loss_batch_audio, loss_batch_image, loss_batch_vid = [], [], []
        ori_loss_batch_audio, ori_loss_batch_image, ori_loss_batch_vid = [], [], []

        if self.use_loss_reweight:
            assert loss_reweight is not None, f"loss_reweight is None when use_loss_reweight"

        if has_vision:
            for idx, (pred_vis, tgt_vis) in enumerate(
                zip(pred_vision_list, vision_target)
            ):
                sample_reweight = loss_reweight[idx] if self.use_loss_reweight else 1.0
                pred_vis = pred_vis[:, :, :t_h_w_list[idx][1], :t_h_w_list[idx][2]]
                vision_loss = nn.functional.mse_loss(pred_vis.float(), tgt_vis.float())
                if has_image:
                    ori_loss_batch_image.append(vision_loss)
                    loss_batch_image.append(vision_loss * sample_reweight)
                else:
                    ori_loss_batch_vid.append(vision_loss)
                    loss_batch_vid.append(vision_loss * sample_reweight * self.vision_loss_coff)

        if has_audio:
            for idx, (pred_audio, tgt_audio) in enumerate(
                zip(pred_audio_list, audio_target)
            ):
                sample_reweight = loss_reweight[idx] if self.use_loss_reweight else 1.0
                audio_loss = nn.functional.mse_loss(pred_audio.float(), tgt_audio.float())
                ori_loss_batch_audio.append(audio_loss)
                loss_batch_audio.append(audio_loss * sample_reweight * self.audio_loss_coff)

        if has_audio:
            loss_audio = (sum(loss_batch_audio) / batch_size) * self.lambda_ddpm
            total_loss_noreweight_audio = sum(ori_loss_batch_audio) / batch_size
            total_loss += loss_audio
        
        if has_image:
            loss_image = (sum(loss_batch_image) / batch_size) * self.lambda_ddpm
            total_loss_noreweight_image = sum(ori_loss_batch_image) / batch_size
            total_loss += loss_image * self.vision_loss_coff

        if has_video:
            loss_video = (sum(loss_batch_vid) / batch_size) * self.lambda_ddpm
            total_loss_noreweight_vid = sum(ori_loss_batch_vid) / batch_size
            total_loss += loss_video * self.vision_loss_coff

        losses["ddpm_audio"] = loss_audio.detach().clone() if has_audio else torch.zeros(())
        losses["ddpm_audio_noreweight"] = total_loss_noreweight_audio.detach().clone() if has_audio else torch.zeros(())

        losses["ddpm_image"] = loss_image.detach().clone() if has_image else torch.zeros(())
        losses["ddpm_image_noreweight"] = total_loss_noreweight_image.detach().clone() if has_image else torch.zeros(())

        losses["ddpm_vid"] = loss_video.detach().clone() if has_video else torch.zeros(())
        losses["ddpm_vid_noreweight"] = total_loss_noreweight_vid.detach().clone() if has_video else torch.zeros(())

        losses["ddpm"] = losses["ddpm_vid"] + losses["ddpm_audio"] + losses["ddpm_image"]
        losses["ddpm_noreweight"] = losses["ddpm_vid_noreweight"] + losses["ddpm_audio_noreweight"] + losses["ddpm_image_noreweight"]

        return total_loss, losses

    def dispersive_loss(self, pred_list):
        # video item shape of model_preds_list: (channel, 1, h, w)
        z_list = []
        for pred in pred_list:
            z = pred.mean(dim=[1, 2, 3])
            z_list.append(z)
        z_pooled = torch.stack(z_list)
        diff = torch.nn.functional.pdist(z_pooled).pow(2) / z_pooled.shape[1]
        dis_loss = torch.log(torch.exp(-diff).mean())
        return dis_loss

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def predict_eps(
        self,
        vid_context,
        audio_context,
        latents_vid,
        latents_audio,
        timesteps,
        spk_embs=None,
        spk_pos=None,
        t_h_w_list=None,
        audio_len_list=None,
        is_i2v=False,
        slg_layer=False,
        masking_modality=False,
        first_frames=None,
        **kwargs,
    ):
        has_video = latents_vid is not None
        has_audio = latents_audio is not None
        max_seq_len_audio, max_seq_len_video = None, None
        batch_size = len(vid_context) if has_video else len(audio_context)

        if has_audio:
            max_seq_len_audio = audio_len_list.max()
        if has_video:
            if not self.use_mmdit_model:
                _patch_size_h, _patch_size_w = self.backbone.video_model.patch_size[1], self.backbone.video_model.patch_size[2]
            else:
                _patch_size_h, _patch_size_w = self.backbone.patch_size[1], self.backbone.patch_size[2]
            # max_seq_len_video = t_h_w_list[0:1, :].prod(dim=1).sum() // (_patch_size_h*_patch_size_w) # f * h * w from [1, c, f, h, w]
            max_seq_len_video = max(
                int((t * math.ceil(h / self.patch_size) * self.patch_size * math.ceil(w / self.patch_size) * self.patch_size) \
                    // (_patch_size_h*_patch_size_w))
                for (t, h, w) in t_h_w_list
            )
        
        xt_list_vid = [] if has_video else None
        xt_list_audio = [] if has_audio else None
        original_sizes = []
        offset_vid, offset_audio = 0, 0  # 指针，用于在 2D latents 中切分

        for i in range(batch_size):
            if has_video:
                patch_size = self.patch_size
                t, h, w = int(t_h_w_list[i][0]), int(t_h_w_list[i][1]), int(t_h_w_list[i][2])
                original_sizes.append((t, h, w))
                valid_len = t * h * w

                # [🔥 FIX] 手动切片 2D 张量
                # latents 是 [Total_Pixels, C]
                z_item = latents_vid[offset_vid : offset_vid + valid_len, :]
                offset_vid += valid_len  # 移动指针

                # Reshape [L, C] -> [C, 1, H, W]
                xt_reshaped = z_item.transpose(0, 1).view(self.video_latent_ch, t, h, w) # c t h w
                if is_i2v:
                    # ti2v mode, directly replace first frame
                    xt_reshaped[:, :1, :, :] = first_frames[i].permute(3, 0, 1, 2) # t h w c -> c t h w

                # Padding
                pad_h = (patch_size - h % patch_size) % patch_size
                pad_w = (patch_size - w % patch_size) % patch_size
                if pad_h > 0 or pad_w > 0:
                    xt_reshaped = nn.functional.pad(
                        xt_reshaped, (0, pad_w, 0, pad_h), mode="constant", value=0
                    )

                xt_list_vid.append(xt_reshaped)

            if has_audio:
                audio_len = int(audio_len_list[i])

                # [🔥 FIX] 手动切片 2D 张量
                # latents 是 [Total_Pixels, C]
                z_item_audio = latents_audio[offset_audio : offset_audio + audio_len, :]
                offset_audio += audio_len  # 移动指针

                # Reshape [L, C] -> [C, 1, H, W]
                # xt_reshaped_audio = z_item_audio.view(audio_len, self.latent_ch_audio)

                xt_list_audio.append(z_item_audio)

        # Backbone Forward
        pred_video_list, pred_audio_list = self.backbone(
            vid=xt_list_vid,
            audio=xt_list_audio,
            t=timesteps,
            vid_context=vid_context,
            audio_context=audio_context,
            vid_seq_len=max_seq_len_video,
            audio_seq_len=max_seq_len_audio,
            spk_embed=spk_embs,
            spk_pos=spk_pos,
            first_frame_is_clean=is_i2v,
            slg_layer=slg_layer,
            masking_modality=masking_modality,
        )
        
        # [🔥 FIX] 拼回 2D Flattened Tensor
        # 结果必须也是 [Total_Pixels, C]
        velocity_pred_vid = torch.zeros_like(latents_vid) if has_video else None
        velocity_pred_audio = torch.zeros_like(latents_audio) if has_audio else None
        offset_vid, offset_audio = 0, 0

        if has_video:
            for i, pred in enumerate(pred_video_list):
                t, h, w = t_h_w_list[i]

                # Unpad
                pred = pred[:, :t, :h, :w]
                # [C, 1, H, W] -> [H*W, C]
                flat_pred = pred.permute(1, 2, 3, 0).flatten(0, 2)

                valid_len = flat_pred.shape[0]
                velocity_pred_vid[offset_vid : offset_vid + valid_len, :] = flat_pred
                offset_vid += valid_len

        if has_audio:
            for i, pred in enumerate(pred_audio_list):
                audio_len = audio_len_list[i]

                # Unpad
                flat_pred = pred[:audio_len, :] # L, C
                # # [C, L] -> [L, C]
                # flat_pred = pred.permute(1, 0)

                valid_len = flat_pred.shape[0]
                velocity_pred_audio[offset_audio : offset_audio + valid_len, :] = flat_pred
                offset_audio += valid_len

        return velocity_pred_vid, velocity_pred_audio

    @property
    def dtype(self):
        return next(self.backbone.parameters()).dtype
