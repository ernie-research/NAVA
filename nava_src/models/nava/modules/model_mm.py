# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .attention import flash_attention
from torch.utils.checkpoint import checkpoint
from nava_src.models.nava.distributed_comms.communications import all_gather, all_to_all_4D
from nava_src.models.nava.distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state
from nava_src.gradient import gradient_checkpoint_forward

def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000, freqs_scaling=1.0):
    assert dim % 2 == 0
    pos =  torch.arange(max_seq_len)
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    freqs = freqs_scaling * freqs
    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@amp.autocast('cuda', enabled=False)
def rope_apply_joint(x, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, vid_seq_len):
    x_vid = x[:, :vid_seq_len, :, :]
    x_audio = x[:, vid_seq_len:, :, :]
    # print(x_vid.shape, x_audio.shape, 88888)
    x_video_rope = rope_apply_3d(x_vid, grid_sizes_vid, freqs_vid)
    x_audio_rope = rope_apply_1d(x_audio, grid_sizes_audio, freqs_audio)
    x_rope = torch.cat([x_video_rope, x_audio_rope], dim=1)
    return x_rope

@amp.autocast('cuda', enabled=False)
def rope_apply_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2 ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"
    
    # loop over samples
    output = []
    for i, (l, ) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2)) # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()

@amp.autocast('cuda', enabled=False)
def rope_apply_3d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()

@amp.autocast('cuda', enabled=False)
def rope_apply_3d_to_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"
    
    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()

@amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, cross_1d_rope=False):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return rope_apply_3d(x, grid_sizes, freqs) if not cross_1d_rope else rope_apply_3d_to_1d(x, grid_sizes, freqs)
    else:
        return rope_apply_1d(x, grid_sizes, freqs)

class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim,
                                    dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w3 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.bfloat16()).type_as(x) * self.weight.bfloat16()

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.bfloat16()).type_as(x)


class WanDoubleStreamSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 joint_attention=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.joint_attention = joint_attention

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # optional sequence parallelism
        self.q_audio = nn.Linear(dim, dim)
        self.k_audio = nn.Linear(dim, dim)
        self.v_audio = nn.Linear(dim, dim)
        self.o_audio = nn.Linear(dim, dim)
        self.norm_q_audio = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k_audio = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # self.world_size = get_world_size()
        self.use_sp = get_sequence_parallel_state()
        if self.use_sp:
            self.sp_size = nccl_info.sp_size
            self.sp_rank = nccl_info.rank_within_group
            assert self.num_heads % self.sp_size == 0, \
                f"Num heads {self.num_heads} must be divisible by sp_size {self.sp_size}"
    # query, key, value function
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v
    
    def qkv_fn_audio(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q_audio(self.q_audio(x)).view(b, s, n, d)
        k = self.norm_k_audio(self.k_audio(x)).view(b, s, n, d)
        v = self.v_audio(x).view(b, s, n, d)
        return q, k, v
    
    def single_forward(self, x, seq_lens, grid_sizes, freqs, is_audio=False):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self.qkv_fn(x) if not is_audio else self.qkv_fn_audio(x)
        
        if self.use_sp:
            # print(f"[DEBUG SP] Doing all to all to shard head")
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        if self.use_sp: 
            # print(f"[DEBUG SP] Doing all to all to shard sequence")
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
        # output
        x = x.flatten(2)
        x = self.o(x) if not is_audio else self.o_audio(x)

        # ---------------- 补充 Fake Tensor 逻辑 (防 DDP Hang) ----------------
        if self.training:
            # 取极小 Tensor 降低冗余计算开销
            dummy_x = x[:1, :1, :].detach() 
            if not is_audio: # 当前是纯视频，假跑音频层
                dummy_q, dummy_k, dummy_v = self.qkv_fn_audio(dummy_x)
                # q/k/v 均参与加法，保证梯度流经所有投影层
                dummy_out = self.o_audio((dummy_q + dummy_k + dummy_v).flatten(2))
            else:            # 当前是纯音频，假跑视频层
                dummy_q, dummy_k, dummy_v = self.qkv_fn(dummy_x)
                dummy_out = self.o((dummy_q + dummy_k + dummy_v).flatten(2))
            
            # 将 Dummy 梯度锚点挂载到主输出上
            x = x + dummy_out.sum() * 0.0
        # --------------------------------------------------------------------
        
        return x

    
    def forward(self, x_vid, x_audio, seq_lens_vid, seq_lens_audio, grid_sizes_vid, grid_sizes_audio=None, freqs_vid=None, freqs_audio=None, max_seq_len_vid=None, max_seq_len_audio=None, use_joint_attention=True):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if x_vid is not None and x_audio is None:
            return self.single_forward(x_vid, seq_lens_vid, grid_sizes_vid, freqs_vid, is_audio=False), None
        elif x_audio is not None and x_vid is None:
            return None, self.single_forward(x_audio, seq_lens_audio, grid_sizes_audio, freqs_audio, is_audio=True)
        else:
            B = x_vid.shape[0]
            L = x_vid.shape[1] + x_audio.shape[1]
            q_vid, k_vid, v_vid = self.qkv_fn(x_vid)
            q_audio, k_audio, v_audio = self.qkv_fn_audio(x_audio)
            # concat for joint pre-precessing
            q = torch.cat([q_vid, q_audio], dim=1)
            k = torch.cat([k_vid, k_audio], dim=1)
            v = torch.cat([v_vid, v_audio], dim=1)

            pos = torch.arange(L).unsqueeze(0).expand(B, L)
            
            if use_joint_attention:
                # print("joint attention apply")
                # 判断是否是视频/音频的有效 token
                is_vid_valid = (pos < max_seq_len_vid) & (pos < seq_lens_vid.unsqueeze(1))
                is_aud_valid = (pos >= max_seq_len_vid) & ((pos - max_seq_len_vid) < seq_lens_audio.unsqueeze(1))

                # 联合有效掩码
                is_valid = is_vid_valid | is_aud_valid
                sort_keys = (~is_valid).int() 
                gather_indices = torch.argsort(sort_keys, dim=1, stable=True).to(x_vid.device) # 形状: [B, L]

                if self.use_sp:
                    # print(f"[DEBUG SP] Doing all to all to shard head")
                    q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
                    k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
                    v = all_to_all_4D(v, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]

                q_rope = rope_apply_joint(q, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid) 
                k_rope = rope_apply_joint(k, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid)

                # 把索引扩展到 4D [B, L, H, D]，匹配 QKV 的形状
                gather_indices_expanded = gather_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, q_rope.size(2), q_rope.size(3))
                
                q_shifted = torch.gather(q_rope, dim=1, index=gather_indices_expanded)
                k_shifted = torch.gather(k_rope, dim=1, index=gather_indices_expanded)
                v_shifted = torch.gather(v,      dim=1, index=gather_indices_expanded)
                x_shifted = flash_attention(
                    q=q_shifted,
                    k=k_shifted,
                    v=v_shifted,
                    k_lens=(seq_lens_vid + seq_lens_audio),
                    window_size=self.window_size)
                scatter_indices = torch.argsort(gather_indices, dim=1)
                scatter_indices_expanded = scatter_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_shifted.size(2), x_shifted.size(3))

                # 把算完的结果完美填回原位
                x = torch.gather(x_shifted, dim=1, index=scatter_indices_expanded)
            else:
                x_vid = flash_attention(
                    q=rope_apply(q_vid, grid_sizes_vid, freqs_vid),
                    k=rope_apply(k_vid, grid_sizes_vid, freqs_vid),
                    v=v_vid,
                    k_lens=seq_lens_vid,
                    window_size=self.window_size)
                x_audio = flash_attention(
                    q=rope_apply(q_audio, grid_sizes_audio, freqs_audio),
                    k=rope_apply(k_audio, grid_sizes_audio, freqs_audio),
                    v=v_audio,
                    k_lens=seq_lens_audio,
                    window_size=self.window_size)
                x = torch.cat([x_vid, x_audio], dim=1)
            if self.use_sp: 
                # print(f"[DEBUG SP] Doing all to all to shard sequence")
                x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
            # output
            x = x.flatten(2)
            x_vid = self.o(x[:, :max_seq_len_vid, :])
            x_audio = self.o_audio(x[:, max_seq_len_vid:, :])
            return x_vid, x_audio

class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # optional sequence parallelism
        # self.world_size = get_world_size()
        self.use_sp = get_sequence_parallel_state()
        if self.use_sp:
            self.sp_size = nccl_info.sp_size
            self.sp_rank = nccl_info.rank_within_group
            assert self.num_heads % self.sp_size == 0, \
                f"Num heads {self.num_heads} must be divisible by sp_size {self.sp_size}"
    # query, key, value function
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v
    
    def single_forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self.qkv_fn(x)
        if self.use_sp:
            # print(f"[DEBUG SP] Doing all to all to shard head")
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        if self.use_sp: 
            # print(f"[DEBUG SP] Doing all to all to shard sequence")
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


    def forward(self, x, seq_lens_vid, seq_lens_audio, grid_sizes_vid, grid_sizes_audio=None, freqs_vid=None, freqs_audio=None, max_seq_len_vid=None, max_seq_len_audio=None, use_joint_attention=True):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if max_seq_len_vid > 0 and max_seq_len_audio == 0:
            return self.single_forward(x, seq_lens_vid, grid_sizes_vid, freqs_vid)
        elif max_seq_len_vid == 0 and max_seq_len_audio > 0:
            return self.single_forward(x, seq_lens_audio, grid_sizes_audio, freqs_audio)
        else:
            B, L = x.shape[0], x.shape[1]
            pos = torch.arange(L).unsqueeze(0).expand(B, L)
            q, k, v = self.qkv_fn(x)
            if self.use_sp:
                # print(f"[DEBUG SP] Doing all to all to shard head")
                q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
                k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
                v = all_to_all_4D(v, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
            if use_joint_attention:
                # print("joint attention apply")
                is_vid_valid = (pos < max_seq_len_vid) & (pos < seq_lens_vid.unsqueeze(1))
                is_aud_valid = (pos >= max_seq_len_vid) & ((pos - max_seq_len_vid) < seq_lens_audio.unsqueeze(1))

                # 联合有效掩码
                is_valid = is_vid_valid | is_aud_valid
                sort_keys = (~is_valid).int() 
                gather_indices = torch.argsort(sort_keys, dim=1, stable=True).to(x.device) # 形状: [B, L]

                q_rope = rope_apply_joint(q, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid) 
                k_rope = rope_apply_joint(k, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid)

                # 把索引扩展到 4D [B, L, H, D]，匹配 QKV 的形状
                gather_indices_expanded = gather_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, q_rope.size(2), q_rope.size(3))
                
                q_shifted = torch.gather(q_rope, dim=1, index=gather_indices_expanded)
                k_shifted = torch.gather(k_rope, dim=1, index=gather_indices_expanded)
                v_shifted = torch.gather(v,      dim=1, index=gather_indices_expanded)

                x_shifted = flash_attention(
                    q=q_shifted,
                    k=k_shifted,
                    v=v_shifted,
                    k_lens=(seq_lens_vid + seq_lens_audio),
                    window_size=self.window_size)
                scatter_indices = torch.argsort(gather_indices, dim=1)
                scatter_indices_expanded = scatter_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_shifted.size(2), x_shifted.size(3))

                # 把算完的结果完美填回原位
                x = torch.gather(x_shifted, dim=1, index=scatter_indices_expanded)
            else:
                q_vid, k_vid, v_vid = q[:, :max_seq_len_vid, :], k[:, :max_seq_len_vid, :], v[:, :max_seq_len_vid, :]
                q_audio, k_audio, v_audio = q[:, max_seq_len_vid:, :], k[:, max_seq_len_vid:, :], v[:, max_seq_len_vid:, :]
                x_vid = flash_attention(
                    q=rope_apply(q_vid, grid_sizes_vid, freqs_vid),
                    k=rope_apply(k_vid, grid_sizes_vid, freqs_vid),
                    v=v_vid,
                    k_lens=seq_lens_vid,
                    window_size=self.window_size)
                x_audio = flash_attention(
                    q=rope_apply(q_audio, grid_sizes_audio, freqs_audio),
                    k=rope_apply(k_audio, grid_sizes_audio, freqs_audio),
                    v=v_audio,
                    k_lens=seq_lens_audio,
                    window_size=self.window_size)
                x = torch.cat([x_vid, x_audio], dim=1)
            if self.use_sp: 
                # print(f"[DEBUG SP] Doing all to all to shard sequence")
                x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
            # output
            x = x.flatten(2)
            x = self.o(x)
            return x


class WanT2VCrossAttention(WanSelfAttention):
    def qkv_fn(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        return q, k, v

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v = self.qkv_fn(x, context)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
    
class WanT2VDoubleStreamCrossAttention(WanDoubleStreamSelfAttention):
    def qkv_fn_audio(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q_audio(self.q_audio(x)).view(b, -1, n, d)
        k = self.norm_k_audio(self.k_audio(context)).view(b, -1, n, d)
        v = self.v_audio(context).view(b, -1, n, d)

        return q, k, v
    
    def qkv_fn(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        return q, k, v
    
    def single_forward(self, x, context, context_lens, is_audio=False):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v = self.qkv_fn(x, context) if not is_audio else self.qkv_fn_audio(x, context)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x) if not is_audio else self.o_audio(x)

        # ---------------- 补充 Fake Tensor 逻辑 (防 DDP Hang) ----------------
        if self.training:
            dummy_x = x[:1, :1, :].detach()
            dummy_ctx = context[:1, :1, :].detach()
            
            if not is_audio:
                dummy_q, dummy_k, dummy_v = self.qkv_fn_audio(dummy_x, dummy_ctx)
                dummy_out = self.o_audio((dummy_q + dummy_k + dummy_v).flatten(2))
            else:
                dummy_q, dummy_k, dummy_v = self.qkv_fn(dummy_x, dummy_ctx)
                dummy_out = self.o((dummy_q + dummy_k + dummy_v).flatten(2))
                
            x = x + dummy_out.sum() * 0.0
        # --------------------------------------------------------------------
        return x

    def forward(self, x_vid, x_audio, context, context_lens, vid_seq_len=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        if x_vid is not None and x_audio is not None:
            q, k, v = self.qkv_fn(x_vid, context)
            q_audio, k_audio, v_audio = self.qkv_fn_audio(x_audio, context)

            # compute attention
            x_vid = flash_attention(q, k, v, k_lens=context_lens)
            x_audio = flash_attention(q_audio, k_audio, v_audio, k_lens=context_lens)

            # output
            x_vid = x_vid.flatten(2)
            x_audio = x_audio.flatten(2)
            x_vid = self.o(x_vid)
            x_audio = self.o_audio(x_audio)
            return x_vid, x_audio
        elif x_vid is not None:
            return self.single_forward(x_vid, context, context_lens, is_audio=False), None
        else:
            return None, self.single_forward(x_audio, context, context_lens, is_audio=True)

class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 additional_emb_length=None):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.additional_emb_length = additional_emb_length

    def qkv_fn(self, x, context):
        context_img = context[:, : self.additional_emb_length]
        context = context[:, self.additional_emb_length :]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        return q, k, v, k_img, v_img


    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v, k_img, v_img = self.qkv_fn(x, context)

        if self.use_sp:
            # print(f"[DEBUG SP] Doing all to all to shard head")
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)  
            k = torch.chunk(k, self.sp_size, dim=2)[self.sp_rank]
            v = torch.chunk(v, self.sp_size, dim=2)[self.sp_rank]
            k_img = torch.chunk(k_img, self.sp_size, dim=2)[self.sp_rank]
            v_img = torch.chunk(v_img, self.sp_size, dim=2)[self.sp_rank]
            
        # [B, L, H/P, C/H]
        # k_img: [B, L, H, C/H]
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)
        if self.use_sp: 
            # print(f"[DEBUG SP] Doing all to all to shard sequence")
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
            
        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

class ModulationAdd(nn.Module):
    def __init__(self, dim, num):
        super().__init__()
        self.modulation = nn.Parameter(torch.randn(1, num, dim) / dim**0.5)

    def forward(self, e):
        return self.modulation.bfloat16() + e.bfloat16()
    
class WanDoubleStreamAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 additional_emb_length=None,
                 no_split_norm_ffn=False):
        """初始化跨模态注意力模块"""
        super().__init__()
        # 基础参数
        self.dim = dim                     # 输入维度
        self.ffn_dim = ffn_dim             # FFN中间层维度
        self.num_heads = num_heads         # 注意力头数
        self.window_size = window_size     # 注意力窗口大小(-1表示无窗口)
        self.qk_norm = qk_norm             # 是否对QK做归一化
        self.cross_attn_norm = cross_attn_norm  # 是否对交叉注意力做归一化
        self.eps = eps                     # 归一化的小常数
        self.no_split_norm_ffn = no_split_norm_ffn  # 是否不分离norm/ffn

        # 网络层定义
        self.norm1 = WanLayerNorm(dim, eps)  # 自注意力前归一化
        if not no_split_norm_ffn:
            self.norm1_audio = WanLayerNorm(dim, eps)  # 自注意力前归一化
        self.self_attn = WanDoubleStreamSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)  # 自注意力层
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()  # 交叉注意力前归一化(可选)
        if not no_split_norm_ffn:
            self.norm3_audio = WanLayerNorm(
                dim, eps,
                elementwise_affine=True) if cross_attn_norm else nn.Identity()  # 交叉注意力前归一化(可选)
            
        # 根据类型初始化不同的交叉注意力层
        if cross_attn_type == 'i2v_cross_attn':
            assert False, "Not support i2v_cross_attn for mmdit mode"
            assert additional_emb_length is not None, "additional_emb_length should be specified for i2v_cross_attn"
            self.cross_attn = WanI2VCrossAttention(dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps, 
                                                additional_emb_length)  # 图像到视频交叉注意力
        else:
            assert additional_emb_length is None, "additional_emb_length should be None for t2v_cross_attn"
            self.cross_attn = WanT2VDoubleStreamCrossAttention(dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps, )  # 文本到视频交叉注意力
                                                
        self.norm2 = WanLayerNorm(dim, eps)  # FFN前归一化
        if not no_split_norm_ffn:
            self.norm2_audio = WanLayerNorm(dim, eps)  # FFN前归一化
        self.ffn = nn.Sequential(          # 前馈网络
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        if not no_split_norm_ffn:
            self.ffn_audio = nn.Sequential(          # 前馈网络
                nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim))

        # 调制参数
        self.modulation = ModulationAdd(dim, 6)  # 6通道的调制加法层
        self.modulation_audio = ModulationAdd(dim, 6)  # 6通道的调制加法层


    def forward(
        self,
        x,
        e_vid,
        e_audio,
        freqs_vid,
        freqs_audio,
        context,
        context_lens,
        seq_lens_vid=None,
        seq_lens_audio=None,
        grid_sizes_vid=None,
        grid_sizes_audio=None,
        max_seq_len_vid=None,
        max_seq_len_audio=None,
        masking_modality=False,
        **kwargs
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # has video input
        x_vid, x_audio = None, None
        dummy_val = None

        # ---------------- 补充 Block 级别 Fake Tensor 逻辑 ----------------
        if self.training:
            if max_seq_len_vid > 0 and max_seq_len_audio == 0:
                # 伪造跑一遍 Audio 专属的 Norm / FFN / Modulation
                dummy_e = e_vid[:1, :1].detach()  # 借用一下视频的 e 作为输入特征
                dummy_x = x[:1, :1].detach()  # 借用一下视频的 x

                dummy_e_out = self.modulation_audio(dummy_e)
                if self.no_split_norm_ffn:
                    dummy_val = dummy_e_out.sum() * 0.0
                else:
                    dummy_n1 = self.norm1_audio(dummy_x)
                    dummy_n3 = self.norm3_audio(dummy_x)
                    dummy_n2 = self.norm2_audio(dummy_x)
                    dummy_ffn = self.ffn_audio(dummy_x)
                    dummy_val = (dummy_e_out.sum() + dummy_n1.sum() + dummy_n3.sum() + dummy_n2.sum() + dummy_ffn.sum()) * 0.0

            elif max_seq_len_audio > 0 and max_seq_len_vid == 0:
                # 伪造跑一遍 Video 专属的 Norm / FFN / Modulation
                dummy_e = e_audio[:1, :1].detach()
                dummy_x = x[:1, :1].detach()

                dummy_e_out = self.modulation(dummy_e)
                dummy_n1 = self.norm1(dummy_x)
                dummy_n3 = self.norm3(dummy_x)
                dummy_n2 = self.norm2(dummy_x)
                dummy_ffn = self.ffn(dummy_x)

                dummy_val = (dummy_e_out.sum() + dummy_n1.sum() + dummy_n3.sum() + dummy_n2.sum() + dummy_ffn.sum()) * 0.0
        # --------------------------------------------------------------------

        if max_seq_len_vid > 0:
            x_vid = x[:, :max_seq_len_vid]
            assert e_vid.dtype == torch.bfloat16
            assert len(e_vid.shape) == 4 and e_vid.size(2) == 6 and e_vid.shape[1] == x_vid.shape[1], f"{e_vid.shape}, {x_vid.shape}"
            with amp.autocast('cuda', dtype=torch.bfloat16):
                e_vid = self.modulation(e_vid).chunk(6, dim=2)
            assert e_vid[0].dtype == torch.bfloat16
        if max_seq_len_audio > 0:
            x_audio = x[:, max_seq_len_vid:]
            assert e_audio.dtype == torch.bfloat16
            assert len(e_audio.shape) == 4 and e_audio.size(2) == 6 and e_audio.shape[1] == x_audio.shape[1], f"{e_audio.shape}, {x_audio.shape}"
            with amp.autocast('cuda', dtype=torch.bfloat16):
                e_audio = self.modulation_audio(e_audio).chunk(6, dim=2)
            assert e_audio[0].dtype == torch.bfloat16

        # joint attention begin
        x_vid_norm, x_audio_norm = None, None
        if x_vid is not None:
            x_vid_norm = self.norm1(x_vid).bfloat16() * (1 + e_vid[1].squeeze(2)) + e_vid[0].squeeze(2)
        if x_audio is not None:
            x_audio_norm = (self.norm1 if self.no_split_norm_ffn else self.norm1_audio)(x_audio).bfloat16() * (1 + e_audio[1].squeeze(2)) + e_audio[0].squeeze(2)

        y_vid_attn, y_audio_attn = self.self_attn(
            x_vid_norm, x_audio_norm, seq_lens_vid, seq_lens_audio, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid, max_seq_len_audio, use_joint_attention=(not masking_modality)
            )
        with amp.autocast('cuda', dtype=torch.bfloat16):
            if x_vid is not None:
                x_vid = x_vid + y_vid_attn * e_vid[2].squeeze(2)
            if x_audio is not None:
                x_audio = x_audio + y_audio_attn * e_audio[2].squeeze(2)
        
        def cross_attn_ffn_doublestream(x_vid, x_audio, context, context_lens, e_vid, e_audio):
            x_vid_norm, x_audio_norm = None, None
            if x_vid is not None:
                x_vid_norm = self.norm3(x_vid)
            if x_audio is not None:
                x_audio_norm = (self.norm3 if self.no_split_norm_ffn else self.norm3_audio)(x_audio)
            x_vid_attn, x_audio_attn = self.cross_attn(x_vid_norm, x_audio_norm, context, context_lens, max_seq_len_vid)

            if x_vid is not None:
                x_vid = x_vid + x_vid_attn
                y_vid = self.ffn(
                    self.norm2(x_vid).bfloat16() * (1 + e_vid[4].squeeze(2)) + e_vid[3].squeeze(2))
                with amp.autocast('cuda', dtype=torch.bfloat16):
                    x_vid = x_vid + y_vid * e_vid[5].squeeze(2)
            if x_audio is not None:
                x_audio = x_audio + x_audio_attn
                _norm2 = self.norm2 if self.no_split_norm_ffn else self.norm2_audio
                _ffn = self.ffn if self.no_split_norm_ffn else self.ffn_audio
                y_audio = _ffn(
                    _norm2(x_audio).bfloat16() * (1 + e_audio[4].squeeze(2)) + e_audio[3].squeeze(2))
                with amp.autocast('cuda', dtype=torch.bfloat16):
                    x_audio = x_audio + y_audio * e_audio[5].squeeze(2)
            return x_vid, x_audio

        x_vid, x_audio = cross_attn_ffn_doublestream(x_vid, x_audio, context, context_lens, e_vid, e_audio)
        if x_vid is not None and x_audio is not None:
            x = torch.cat([x_vid, x_audio], dim=1)
        elif x_vid is not None:
            x = x_vid + dummy_val if dummy_val is not None else x_vid
        elif x_audio is not None:
            x = x_audio + dummy_val if dummy_val is not None else x_audio

        return x

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 additional_emb_length=None,
                 split_av_qk_norm_modulation=False):
        """初始化跨模态注意力模块"""
        super().__init__()
        # 基础参数
        self.dim = dim                     # 输入维度
        self.ffn_dim = ffn_dim             # FFN中间层维度
        self.num_heads = num_heads         # 注意力头数
        self.window_size = window_size     # 注意力窗口大小(-1表示无窗口)
        self.qk_norm = qk_norm             # 是否对QK做归一化
        self.cross_attn_norm = cross_attn_norm  # 是否对交叉注意力做归一化
        self.eps = eps                     # 归一化的小常数
        self.split_av_qk_norm_modulation = split_av_qk_norm_modulation

        # 网络层定义
        self.norm1 = WanLayerNorm(dim, eps)  # 自注意力前归一化
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)  # 自注意力层
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()  # 交叉注意力前归一化(可选)
            
        # 根据类型初始化不同的交叉注意力层
        if cross_attn_type == 'i2v_cross_attn':
            assert additional_emb_length is not None, "additional_emb_length should be specified for i2v_cross_attn"
            self.cross_attn = WanI2VCrossAttention(dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps, 
                                                additional_emb_length)  # 图像到视频交叉注意力
        else:
            assert additional_emb_length is None, "additional_emb_length should be None for t2v_cross_attn"
            self.cross_attn = WanT2VCrossAttention(dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps, )  # 文本到视频交叉注意力
                                                
        self.norm2 = WanLayerNorm(dim, eps)  # FFN前归一化
        self.ffn = nn.Sequential(          # 前馈网络
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # 调制参数
        self.modulation = ModulationAdd(dim, 6)  # 6通道的调制加法层
        if split_av_qk_norm_modulation:
            self.modulation_audio = ModulationAdd(dim, 6)


    def forward(
        self,
        x,
        e_vid,
        e_audio,
        freqs_vid,
        freqs_audio,
        context,
        context_lens,
        seq_lens_vid=None,
        seq_lens_audio=None,
        grid_sizes_vid=None,
        grid_sizes_audio=None,
        max_seq_len_vid=None,
        max_seq_len_audio=None,
        masking_modality=False,
        **kwargs
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if not self.split_av_qk_norm_modulation:
            if max_seq_len_vid > 0 and max_seq_len_audio > 0:
                # print(e_vid.shape, e_audio.shape, 9999)
                e = torch.cat([e_vid, e_audio], dim=1)
            elif max_seq_len_vid > 0:
                e = e_vid
            elif max_seq_len_audio > 0:
                e = e_audio
            
            assert e.dtype == torch.bfloat16
            assert len(e.shape) == 4 and e.size(2) == 6 and e.shape[1] == x.shape[1], f"{e.shape}, {x.shape}"
            with amp.autocast('cuda', dtype=torch.bfloat16):
                e = self.modulation(e).chunk(6, dim=2)
            assert e[0].dtype == torch.bfloat16
        else:
            if max_seq_len_vid > 0:
                x_vid = x[:, :max_seq_len_vid]
                assert e_vid.dtype == torch.bfloat16
                assert len(e_vid.shape) == 4 and e_vid.size(2) == 6 and e_vid.shape[1] == x_vid.shape[1], f"{e_vid.shape}, {x_vid.shape}"
                with amp.autocast('cuda', dtype=torch.bfloat16):
                    e_vid = self.modulation(e_vid).chunk(6, dim=2)
                assert e_vid[0].dtype == torch.bfloat16
            if max_seq_len_audio > 0:
                x_audio = x[:, max_seq_len_vid:]
                assert e_audio.dtype == torch.bfloat16
                assert len(e_audio.shape) == 4 and e_audio.size(2) == 6 and e_audio.shape[1] == x_audio.shape[1], f"{e_audio.shape}, {x_audio.shape}"
                with amp.autocast('cuda', dtype=torch.bfloat16):
                    e_audio = self.modulation_audio(e_audio).chunk(6, dim=2)
                assert e_audio[0].dtype == torch.bfloat16

            if max_seq_len_vid > 0 and max_seq_len_audio > 0:
                # e = tuple(torch.cat([e_v, e_a] for e_v, e_a in zip(e_vid, e_audio)))
                e = tuple(torch.cat([e_v, e_a], dim=1) for e_v, e_a in zip(e_vid, e_audio))
            elif max_seq_len_vid > 0:
                e = e_vid
            elif max_seq_len_audio > 0:
                e = e_audio

        # self-attention
        y = self.self_attn(
            self.norm1(x).bfloat16() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens_vid, seq_lens_audio, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid, max_seq_len_audio, use_joint_attention=(not masking_modality)
        )
        with amp.autocast('cuda', dtype=torch.bfloat16):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).bfloat16() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with amp.autocast('cuda', dtype=torch.bfloat16):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L, C]
        """
        assert e.dtype == torch.bfloat16
        with amp.autocast('cuda', dtype=torch.bfloat16):
            e = (self.modulation.bfloat16().unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2) # 1 1 2 D, B L 1 D -> B L 2 D -> 2 * (B L 1 D)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x



class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class SpkToken(nn.Module):
    def __init__(self, spk_dim=192, dim=1024, eps=1e-6):
        super().__init__()
        self.spk_dim = spk_dim
        self.eps = eps
        self.net = nn.Sequential(
            nn.LayerNorm(spk_dim),
            nn.Linear(spk_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)
        # learnable global speaker embedding
        self.null_token = nn.Parameter(torch.zeros(1, dim))

    def forward(self, spk_emb):  # spk_emb: [B, 192], fake spk_emb contains all zeros
        assert spk_emb.shape[-1] == self.spk_dim, f"{spk_emb.shape}"
        B = spk_emb.shape[0]
        fake_pos = (spk_emb.float().pow(2).sum(dim=-1) <= self.eps)   # [B] bool
        spk_embeds = self.out_norm(self.net(spk_emb))            # [B,dim]
        null = self.null_token.expand(B, -1)                       # [B,dim]
        m = (~fake_pos).to(spk_embeds.dtype).view(B, 1)             # 有效speaker=1，无=0
        spk_embeds = null * (1 - m) + spk_embeds * m
        return spk_embeds


class WanAVModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video, text-to-audio.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 vid_in_dim=16,
                 audio_in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 additional_emb_dim=None,
                 additional_emb_length=None,
                 vid_out_dim=16,
                 audio_out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 num_double_layers=8,
                 num_single_layers=24,
                 num_double_final_layers=0,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 gradient_checkpointing = False,
                 gradient_checkpointing_offload = False,
                 gradient_checkpoint_every_n = 1,
                 temporal_rope_scaling_factor=1.0,
                 eps=1e-6,
                 add_spk_emb=False,
                 cross_1d_rope=False,
                 no_split_norm_ffn=False,
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 't2a', 'tt2a', 'ti2v'], model_type ## tt2a means text transcript + text description to audio (to support both TTS and T2A
        self.model_type = model_type
        is_audio_type = "a" in self.model_type
        is_video_type = "v" in self.model_type
        assert is_audio_type ^ is_video_type, "Either audio or video model should be specified"
        if is_audio_type:
            ## audio model
            assert len(patch_size) == 1 and patch_size[0] == 1, "Audio model should only accept 1 dimensional input, and we dont do patchify"

        self.patch_size = patch_size
        self.text_len = text_len
        self.vid_in_dim = vid_in_dim
        self.audio_in_dim = audio_in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.vid_out_dim = vid_out_dim
        self.audio_out_dim = audio_out_dim
        self.num_heads = num_heads
        # self.num_layers = num_layers
        assert num_double_layers + num_single_layers + num_double_final_layers == num_layers, (num_double_layers, num_single_layers, num_double_final_layers, num_layers)
        self.num_double_layers = num_double_layers
        self.num_single_layers = num_single_layers
        self.num_double_final_layers = num_double_final_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.temporal_rope_scaling_factor = temporal_rope_scaling_factor
        print(f"temporal_scaling: {temporal_rope_scaling_factor} !!!!!")
        self.is_audio_type = is_audio_type
        self.is_video_type = is_video_type
        self.add_spk_emb = add_spk_emb
        self.cross_1d_rope = cross_1d_rope
            
        self.patch_embedding = nn.Conv3d(
                vid_in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_audio = nn.Sequential(
                ChannelLastConv1d(audio_in_dim, dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(dim, dim * 4, kernel_size=7, padding=3),
            )
        if add_spk_emb:
            self.speaker_embedding = SpkToken(spk_dim=192, dim=dim)
            
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.use_sp = get_sequence_parallel_state() # seq parallel
        if self.use_sp:
            self.sp_size = nccl_info.sp_size
            self.sp_rank = nccl_info.rank_within_group
            assert self.num_heads % self.sp_size == 0, \
                f"Num heads {self.num_heads} must be divisible by sp_size {self.sp_size}"
        # blocks
        ## so i2v and tt2a share the same cross attention while t2v and t2a share the same cross attention
        cross_attn_type = 't2v_cross_attn' if model_type in ['t2v', 't2a', 'ti2v'] else 'i2v_cross_attn'

        if cross_attn_type == 't2v_cross_attn':
            assert additional_emb_dim is None and additional_emb_length is None, "additional_emb_length should be None for t2v and t2a model"
        else:
            assert additional_emb_dim is not None and additional_emb_length is not None, "additional_emb_length should be specified for i2v and tt2a model"

        self.double_blocks = nn.ModuleList([
            WanDoubleStreamAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, additional_emb_length, no_split_norm_ffn=no_split_norm_ffn)
            for _ in range(num_double_layers)
        ])

        self.single_blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, additional_emb_length)
            for _ in range(num_single_layers)
        ])

        self.double_final_blocks = nn.ModuleList([
            WanDoubleStreamAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, additional_emb_length, no_split_norm_ffn=no_split_norm_ffn)
            for _ in range(num_double_final_layers)
        ])

        # head
        self.head = Head(dim, vid_out_dim, patch_size, eps)
        self.head_audio = Head(dim, audio_out_dim, patch_size=[1], eps=eps)

        self.set_rope_params()

        if model_type in ['i2v', 'tt2a']:
            self.img_emb = MLPProj(additional_emb_dim, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_offload = gradient_checkpointing_offload
        self.gradient_checkpoint_every_n = gradient_checkpoint_every_n

    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        if vid_kwargs is None:
            vid_kwargs = dict(
                e=None,
                seq_lens=0,
                max_seq_len=0,
                grid_sizes=None,
                freqs=self.freqs,
                context=None,
                context_lens=None,
            )
        if audio_kwargs is None:
            audio_kwargs = dict(
                e=None,
                seq_lens=0,
                max_seq_len=0,
                grid_sizes=None,
                freqs=self.freqs_audio,
                context=None,
                context_lens=None,
            )
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"{key}_vid"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"{key}_audio"] = audio_kwargs[key]
        return merged_kwargs

    def set_rope_params(self):
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        ## to be determined
        # self.freqs = rope_params(1024, d, freqs_scaling=temporal_rope_scaling_factor)
        self.freqs_audio = rope_params(1024, d - 4 * (d // 6), freqs_scaling=self.temporal_rope_scaling_factor)
    
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

    def prepare_transformer_block_kwargs(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        first_frame_is_clean=False,
        spk_embed=None,
        spk_pos=None,
        is_audio_type=False,
    ):

        # params
        ## need to change!
        device = next(self.patch_embedding.parameters()).device

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
            self.freqs_audio = self.freqs_audio.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        if not is_audio_type:
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x] ## x is list of [B L D] or [B C F H W]
        else:
            x = [self.patch_embedding_audio(u.unsqueeze(0)) for u in x] ## x is list of [B L D] or [B C F H W]
        if is_audio_type:
            # [B, 1]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[1:2], dtype=torch.long) for u in x]
            )
        else:
            # [B, 3]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x] # [B C F H W] -> [B (F H W) C] -> [B L C]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ]) # single [B, L, C]

        # time embeddings
        if t.dim() == 1:
            if first_frame_is_clean:
                t = torch.ones((t.size(0), seq_len), device=t.device, dtype=t.dtype) * t.unsqueeze(1)
                _first_images_seq_len = grid_sizes[:, 1:].prod(-1)
                for i in range(t.size(0)):
                    t[i, :_first_images_seq_len[i]] = 0
                # print(f"zeroing out first {_first_images_seq_len} from t: {t.shape}, {t}")
            else:
                t = t.unsqueeze(1).expand(t.size(0), seq_len)
        with amp.autocast('cuda', dtype=torch.bfloat16):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).bfloat16())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # [1, 26784, 6, 3072] - B, seq_len, 6, dim
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16

        
        if self.use_sp:
            current_len = x.shape[1]
            # we will pad up to the next multiple of sp_size: eg. [157] -> [160]
            pad_size = (-current_len ) % self.sp_size  

            if pad_size > 0:
                padding = torch.zeros(
                    x.shape[0], pad_size, x.shape[2],
                    device=x.device,
                    dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=1)
                e_padding = torch.zeros(
                    e.shape[0], pad_size, e.shape[2],
                    device=e.device,
                    dtype=e.dtype
                )
                e = torch.cat([e, e_padding], dim=1)
                e0_padding = torch.zeros(
                    e0.shape[0], pad_size, e0.shape[2], e0.shape[3],
                    device=e0.device,
                    dtype=e0.dtype
                )
                e0 = torch.cat([e0, e0_padding], dim=1)

            x = torch.chunk(x, self.sp_size, dim=1)[self.sp_rank]
            e = torch.chunk(e, self.sp_size, dim=1)[self.sp_rank]
            e0 = torch.chunk(e0, self.sp_size, dim=1)[self.sp_rank] 
            
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if self.add_spk_emb and spk_embed is not None:
            spk_embeds = self.speaker_embedding(spk_embed)  # [total_spk, dim]
            B, L, D = context.shape

            if spk_pos is not None:
                indices = [b * L + pos for b, pos_list in enumerate(spk_pos) for pos in pos_list]
                if indices:  # 确保有 spk token
                    indices = torch.tensor(indices, device=context.device)
                    if spk_embeds.shape[0] != len(indices):
                        print(f"Warning not matched pos list and spk_embeds: {spk_embeds.shape}, {indices} !!!")
                        context.view(-1, D)[indices] = spk_embeds[:len(indices)].to(context.dtype)
                    else:
                        context.view(-1, D)[indices] = spk_embeds.to(context.dtype)
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            max_seq_len=seq_len,
            grid_sizes=grid_sizes,
            freqs=self.freqs if not is_audio_type else self.freqs_audio,
            context=context,
            context_lens=context_lens,
            )

        return x, e, kwargs

    def post_transformer_block_out_doublestream(self, x_vid, x_audio, grid_sizes, grid_sizes_audio, e_vid, e_audio):
        # head
        return self.post_transformer_block_out(x_vid, grid_sizes, e_vid, is_audio=False), \
               self.post_transformer_block_out(x_audio, grid_sizes_audio, e_audio, is_audio=True)
        
    def post_transformer_block_out(self, x, grid_sizes, e, is_audio=False):
        # head
        if x is None:
            return None
        if not is_audio:
            x = self.head(x, e)
        else:
            x = self.head_audio(x, e)
        if self.use_sp: 
            x = all_gather(x, dim=1)
        # unpatchify
        if is_audio:
            ## grid_sizes is [B 1] where 1 is L, 
            # converting grid_sizes from [B 1] -> [B]
            grid_sizes = [gs[0] for gs in grid_sizes]
            assert len(x) == len(grid_sizes)
            x = [u[:gs] for u, gs in zip(x, grid_sizes)]
        else:
            ## grid_sizes is [B 3] where 3 is F H w
            x = self.unpatchify(x, grid_sizes)

        return [u.bfloat16() for u in x]


    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        y=None,
        spk_embed=None,
        spk_pos=None,
        masking_modality=False,
        first_frame_is_clean=False,
        **kwargs
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
                OR 
                List of input audio tensors, each with shape [L, C_in]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
                OR
                List of denoised audio tensors with original input shapes [L, C_in]
        """
        x_vid, e_vid, kwargs_vid = None, None, None
        x_audio, e_audio, kwargs_audio = None, None, None
        if vid is not None:
            x_vid, e_vid, kwargs_vid = self.prepare_transformer_block_kwargs(
                x=vid,
                t=t,
                context=vid_context,
                seq_len=vid_seq_len,
                clip_fea=clip_fea,
                y=y,
                first_frame_is_clean=first_frame_is_clean,
            )
        if audio is not None:
            x_audio, e_audio, kwargs_audio = self.prepare_transformer_block_kwargs(
                x=audio,
                t=t,
                context=audio_context,
                seq_len=audio_seq_len,
                clip_fea=clip_fea,
                y=y,
                first_frame_is_clean=False,
                spk_embed=spk_embed,
                spk_pos=spk_pos,
                is_audio_type=True
            )
        kwargs = self.merge_kwargs(kwargs_vid, kwargs_audio)
        # kwargs["context"] = kwargs["context_vid"]
        # kwargs["context_lens"] = kwargs["context_lens_vid"]
        kwargs["context"] = kwargs["context_vid"] if (kwargs["context_vid"] is not None and spk_embed is None) else kwargs["context_audio"]
        kwargs["context_lens"] = kwargs["context_lens_vid"] if kwargs["context_lens_vid"] is not None else kwargs["context_lens_audio"]
        kwargs["masking_modality"] = masking_modality

        # Under SP, x_vid/x_audio are already chunked to L/P length by prepare_transformer_block_kwargs.
        # Adjust max_seq_len_vid/audio to match the actual chunked lengths so blocks split correctly.
        if self.use_sp:
            kwargs["max_seq_len_vid"] = x_vid.shape[1] if x_vid is not None else 0
            kwargs["max_seq_len_audio"] = x_audio.shape[1] if x_audio is not None else 0

        if x_vid is not None and x_audio is not None:
            x = torch.cat([x_vid, x_audio], dim=1)
        elif x_vid is not None:
            x = x_vid
        elif x_audio is not None:
            x = x_audio

        for i, block in enumerate(self.double_blocks):
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=(self.gradient_checkpointing and i % self.gradient_checkpoint_every_n == 0),
                use_gradient_checkpointing_offload=self.gradient_checkpointing_offload,
                x=x,
                **kwargs
            )

        for i, block in enumerate(self.single_blocks):
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=(self.gradient_checkpointing and i % self.gradient_checkpoint_every_n == 0),
                use_gradient_checkpointing_offload=self.gradient_checkpointing_offload,
                x=x,
                **kwargs
            )
        for i, block in enumerate(self.double_final_blocks):
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=(self.gradient_checkpointing and i % self.gradient_checkpoint_every_n == 0),
                use_gradient_checkpointing_offload=self.gradient_checkpointing_offload,
                x=x,
                **kwargs
            )
        if vid is not None:
            x_vid = x[:, :kwargs["max_seq_len_vid"]]
        if audio is not None:
            x_audio = x[:, kwargs["max_seq_len_vid"]:]

        return self.post_transformer_block_out_doublestream(x_vid, x_audio, kwargs['grid_sizes_vid'], kwargs['grid_sizes_audio'], e_vid, e_audio)

    def unpatchify(self, x, grid_sizes, is_audio=False):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.vid_out_dim if not is_audio else self.audio_out_dim
        patch_size = self.patch_size if not is_audio else [1]
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # v is [F H w] F * H * 80, 100, it was right padded by 20. 
            u = u[:math.prod(v)].view(*v, *patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        # out is list of [C F H W]
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        if self.is_video_type:
            assert isinstance(self.patch_embedding, nn.Conv3d), f"Patch embedding for video should be a Conv3d layer, got {type(self.patch_embedding)}"
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)