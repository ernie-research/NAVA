"""SP-aware overrides for ``model_mm.py``.

This module does **not** modify any existing file. It provides:
  * ``_sp_regroup_joint`` / ``_sp_ungroup_joint`` token reorder helpers
  * ``WanDoubleStreamSelfAttentionSP`` / ``WanSelfAttentionSP``
        subclasses that fix the joint-attention path under sequence parallel
  * ``WanAVModelSP`` that swaps the self-attn instances inside every block to
        the SP-aware versions while preserving weights.

Why the joint branch needs fixing under SP:
  ``prepare_transformer_block_kwargs`` chunks vid and audio independently along
  dim=1, so each rank holds ``[vid_chunk_k] + [audio_chunk_k]``. After
  ``cat([q_vid, q_audio], dim=1)`` and ``all_to_all_4D(scatter=2, gather=1)``,
  the locally reconstructed full sequence is interleaved as
  ``[vid_0, audio_0, vid_1, audio_1, ..., vid_{P-1}, audio_{P-1}]`` instead of
  ``[full_vid, full_audio]``. This breaks ``rope_apply_joint``'s vid/audio split
  point and the validity mask. We rearrange to ``[full_vid, full_audio]`` with
  pure view+cat ops, run the original RoPE/mask/attention on a globally-ordered
  tensor, and rearrange back before ``all_to_all_4D(scatter=1, gather=2)``.
"""

import torch
import torch.nn as nn

from nava_src.models.nava.distributed_comms.communications import all_to_all_4D
from nava_src.models.nava.distributed_comms.parallel_states import (
    get_sequence_parallel_state,
    nccl_info,
)

from .attention import flash_attention
from .model_mm import (
    WanAttentionBlock,
    WanAVModel,
    WanDoubleStreamAttentionBlock,
    WanDoubleStreamSelfAttention,
    WanSelfAttention,
    rope_apply,
    rope_apply_joint,
)


def _sp_regroup_joint(t: torch.Tensor, sp: int, L_v_local: int, L_a_local: int) -> torch.Tensor:
    """Reorder ``[vid_k, audio_k]_k`` -> ``[full_vid][full_audio]``.

    Input shape : ``[B, sp * (L_v_local + L_a_local), H_p, D]`` (post all-to-all)
    Output shape: ``[B, sp*L_v_local + sp*L_a_local, H_p, D]`` with vid first.
    """
    B = t.shape[0]
    H_p = t.shape[2]
    D = t.shape[3]
    L_local = L_v_local + L_a_local
    assert t.shape[1] == sp * L_local, (t.shape, sp, L_local)
    t = t.view(B, sp, L_local, H_p, D)
    t_vid = t[:, :, :L_v_local].reshape(B, sp * L_v_local, H_p, D)
    t_aud = t[:, :, L_v_local:].reshape(B, sp * L_a_local, H_p, D)
    return torch.cat([t_vid, t_aud], dim=1)


def _sp_ungroup_joint(t: torch.Tensor, sp: int, L_v_local: int, L_a_local: int) -> torch.Tensor:
    """Inverse of :func:`_sp_regroup_joint`.

    Input shape : ``[B, sp*L_v_local + sp*L_a_local, H_p, D]``
    Output shape: ``[B, sp * (L_v_local + L_a_local), H_p, D]`` interleaved.
    """
    B = t.shape[0]
    H_p = t.shape[2]
    D = t.shape[3]
    L_v = sp * L_v_local
    L_a = sp * L_a_local
    assert t.shape[1] == L_v + L_a, (t.shape, L_v, L_a)
    t_vid = t[:, :L_v].reshape(B, sp, L_v_local, H_p, D)
    t_aud = t[:, L_v:].reshape(B, sp, L_a_local, H_p, D)
    out = torch.cat([t_vid, t_aud], dim=2)
    return out.reshape(B, sp * (L_v_local + L_a_local), H_p, D)


class WanDoubleStreamSelfAttentionSP(WanDoubleStreamSelfAttention):
    """SP-aware joint-attention override.

    Single-modality path delegates to :meth:`single_forward` from the base
    class (already correct under SP). Only the joint branch is rewritten.
    """

    def forward(
        self,
        x_vid,
        x_audio,
        seq_lens_vid,
        seq_lens_audio,
        grid_sizes_vid,
        grid_sizes_audio=None,
        freqs_vid=None,
        freqs_audio=None,
        max_seq_len_vid=None,
        max_seq_len_audio=None,
        use_joint_attention=True,
    ):
        if x_vid is not None and x_audio is None:
            return (
                self.single_forward(x_vid, seq_lens_vid, grid_sizes_vid, freqs_vid, is_audio=False),
                None,
            )
        if x_audio is not None and x_vid is None:
            return (
                None,
                self.single_forward(x_audio, seq_lens_audio, grid_sizes_audio, freqs_audio, is_audio=True),
            )

        # joint path
        B = x_vid.shape[0]
        L_v_local = x_vid.shape[1]
        L_a_local = x_audio.shape[1]
        sp = self.sp_size if self.use_sp else 1
        L_v = L_v_local * sp
        L_a = L_a_local * sp
        L = L_v + L_a

        q_vid, k_vid, v_vid = self.qkv_fn(x_vid)
        q_audio, k_audio, v_audio = self.qkv_fn_audio(x_audio)
        q = torch.cat([q_vid, q_audio], dim=1)
        k = torch.cat([k_vid, k_audio], dim=1)
        v = torch.cat([v_vid, v_audio], dim=1)

        if self.use_sp:
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)
            # local order after all-to-all is [vid_k, audio_k]_k. Regroup to
            # [full_vid, full_audio] so the original RoPE/mask logic applies
            # untouched on a globally-ordered tensor.
            q = _sp_regroup_joint(q, sp, L_v_local, L_a_local)
            k = _sp_regroup_joint(k, sp, L_v_local, L_a_local)
            v = _sp_regroup_joint(v, sp, L_v_local, L_a_local)

        if use_joint_attention:
            pos = torch.arange(L).unsqueeze(0).expand(B, L)
            is_vid_valid = (pos < L_v) & (pos < seq_lens_vid.unsqueeze(1))
            is_aud_valid = (pos >= L_v) & ((pos - L_v) < seq_lens_audio.unsqueeze(1))

            is_valid = is_vid_valid | is_aud_valid
            sort_keys = (~is_valid).int()
            gather_indices = torch.argsort(sort_keys, dim=1, stable=True).to(x_vid.device)

            q_rope = rope_apply_joint(q, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, L_v)
            k_rope = rope_apply_joint(k, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, L_v)

            gather_indices_expanded = gather_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, q_rope.size(2), q_rope.size(3)
            )
            q_shifted = torch.gather(q_rope, dim=1, index=gather_indices_expanded)
            k_shifted = torch.gather(k_rope, dim=1, index=gather_indices_expanded)
            v_shifted = torch.gather(v, dim=1, index=gather_indices_expanded)
            x_shifted = flash_attention(
                q=q_shifted,
                k=k_shifted,
                v=v_shifted,
                k_lens=(seq_lens_vid + seq_lens_audio),
                window_size=self.window_size,
            )
            scatter_indices = torch.argsort(gather_indices, dim=1)
            scatter_indices_expanded = scatter_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, x_shifted.size(2), x_shifted.size(3)
            )
            x = torch.gather(x_shifted, dim=1, index=scatter_indices_expanded)
        else:
            q_v_g, k_v_g, v_v_g = q[:, :L_v], k[:, :L_v], v[:, :L_v]
            q_a_g, k_a_g, v_a_g = q[:, L_v:], k[:, L_v:], v[:, L_v:]
            x_vid_attn = flash_attention(
                q=rope_apply(q_v_g, grid_sizes_vid, freqs_vid),
                k=rope_apply(k_v_g, grid_sizes_vid, freqs_vid),
                v=v_v_g,
                k_lens=seq_lens_vid,
                window_size=self.window_size,
            )
            x_audio_attn = flash_attention(
                q=rope_apply(q_a_g, grid_sizes_audio, freqs_audio),
                k=rope_apply(k_a_g, grid_sizes_audio, freqs_audio),
                v=v_a_g,
                k_lens=seq_lens_audio,
                window_size=self.window_size,
            )
            x = torch.cat([x_vid_attn, x_audio_attn], dim=1)

        if self.use_sp:
            x = _sp_ungroup_joint(x, sp, L_v_local, L_a_local)
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)

        x = x.flatten(2)
        x_vid_out = self.o(x[:, :L_v_local, :])
        x_audio_out = self.o_audio(x[:, L_v_local:, :])
        return x_vid_out, x_audio_out


class WanSelfAttentionSP(WanSelfAttention):
    """SP-aware joint-attention override for the single-stream block."""

    def forward(
        self,
        x,
        seq_lens_vid,
        seq_lens_audio,
        grid_sizes_vid,
        grid_sizes_audio=None,
        freqs_vid=None,
        freqs_audio=None,
        max_seq_len_vid=None,
        max_seq_len_audio=None,
        use_joint_attention=True,
    ):
        if max_seq_len_vid > 0 and max_seq_len_audio == 0:
            return self.single_forward(x, seq_lens_vid, grid_sizes_vid, freqs_vid)
        if max_seq_len_vid == 0 and max_seq_len_audio > 0:
            return self.single_forward(x, seq_lens_audio, grid_sizes_audio, freqs_audio)

        # joint path. ``max_seq_len_vid`` / ``max_seq_len_audio`` here are the
        # *local* chunk lengths because ``WanAVModel.forward`` overrode them
        # under SP. The stitched local sequence is exactly
        # ``[vid_chunk_k] + [audio_chunk_k]``.
        B = x.shape[0]
        L_v_local = max_seq_len_vid
        L_a_local = max_seq_len_audio
        L_local = L_v_local + L_a_local
        assert x.shape[1] == L_local, (x.shape, L_local)
        sp = self.sp_size if self.use_sp else 1
        L_v = L_v_local * sp
        L_a = L_a_local * sp
        L = L_v + L_a

        q, k, v = self.qkv_fn(x)
        if self.use_sp:
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)
            q = _sp_regroup_joint(q, sp, L_v_local, L_a_local)
            k = _sp_regroup_joint(k, sp, L_v_local, L_a_local)
            v = _sp_regroup_joint(v, sp, L_v_local, L_a_local)

        if use_joint_attention:
            pos = torch.arange(L).unsqueeze(0).expand(B, L)
            is_vid_valid = (pos < L_v) & (pos < seq_lens_vid.unsqueeze(1))
            is_aud_valid = (pos >= L_v) & ((pos - L_v) < seq_lens_audio.unsqueeze(1))

            is_valid = is_vid_valid | is_aud_valid
            sort_keys = (~is_valid).int()
            gather_indices = torch.argsort(sort_keys, dim=1, stable=True).to(x.device)

            q_rope = rope_apply_joint(q, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, L_v)
            k_rope = rope_apply_joint(k, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, L_v)

            gather_indices_expanded = gather_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, q_rope.size(2), q_rope.size(3)
            )
            q_shifted = torch.gather(q_rope, dim=1, index=gather_indices_expanded)
            k_shifted = torch.gather(k_rope, dim=1, index=gather_indices_expanded)
            v_shifted = torch.gather(v, dim=1, index=gather_indices_expanded)

            x_shifted = flash_attention(
                q=q_shifted,
                k=k_shifted,
                v=v_shifted,
                k_lens=(seq_lens_vid + seq_lens_audio),
                window_size=self.window_size,
            )
            scatter_indices = torch.argsort(gather_indices, dim=1)
            scatter_indices_expanded = scatter_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, x_shifted.size(2), x_shifted.size(3)
            )
            x = torch.gather(x_shifted, dim=1, index=scatter_indices_expanded)
        else:
            q_v_g, k_v_g, v_v_g = q[:, :L_v], k[:, :L_v], v[:, :L_v]
            q_a_g, k_a_g, v_a_g = q[:, L_v:], k[:, L_v:], v[:, L_v:]
            x_vid = flash_attention(
                q=rope_apply(q_v_g, grid_sizes_vid, freqs_vid),
                k=rope_apply(k_v_g, grid_sizes_vid, freqs_vid),
                v=v_v_g,
                k_lens=seq_lens_vid,
                window_size=self.window_size,
            )
            x_audio = flash_attention(
                q=rope_apply(q_a_g, grid_sizes_audio, freqs_audio),
                k=rope_apply(k_a_g, grid_sizes_audio, freqs_audio),
                v=v_a_g,
                k_lens=seq_lens_audio,
                window_size=self.window_size,
            )
            x = torch.cat([x_vid, x_audio], dim=1)

        if self.use_sp:
            x = _sp_ungroup_joint(x, sp, L_v_local, L_a_local)
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)

        x = x.flatten(2)
        x = self.o(x)
        return x


def _swap_self_attn(block: nn.Module, new_cls: type) -> None:
    """Replace ``block.self_attn`` with ``new_cls`` while preserving weights."""
    old = block.self_attn
    new = new_cls(
        dim=old.dim,
        num_heads=old.num_heads,
        window_size=old.window_size,
        qk_norm=old.qk_norm,
        eps=old.eps,
    )
    # If the source has FP8Linear children, patch the freshly-built SP module
    # the same way so it can absorb `*_scale` keys from old.state_dict().
    # Lazy import keeps this file decoupled from NAVA_FP8 in pure bf16 runs.
    try:
        from NAVA_FP8.fp8_linear import FP8Linear
        from NAVA_FP8.patching import patch_model_to_fp8
        has_fp8 = any(isinstance(m, FP8Linear) for m in old.modules())
    except ImportError:
        has_fp8 = False
    if has_fp8:
        # `new` is an isolated self_attn module — the default whitelist regex
        # in patch_model_to_fp8 expects the full backbone.*_blocks.<i>... path,
        # which doesn't apply here. Patch every nn.Linear inside instead.
        patch_model_to_fp8(
            new,
            should_patch=lambda path, mod: isinstance(mod, nn.Linear),
        )

    new.load_state_dict(old.state_dict())
    if has_fp8:
        # Mixed-dtype module (fp8 weights + bf16 scales/norms/bias) — only
        # change the device, leave per-param dtypes alone.
        new.to(next(old.parameters()).device)
    else:
        new.to(next(old.parameters()).device, dtype=next(old.parameters()).dtype)
    block.self_attn = new


class WanAVModelSP(WanAVModel):
    """``WanAVModel`` with SP-aware self-attention modules.

    Inherits the full architecture / weight layout from :class:`WanAVModel`,
    then walks every transformer block and swaps the ``self_attn`` instance to
    its SP-aware subclass. Cross-attn, modulation, FFN, and the rest of the
    block are unchanged because they already operate on local chunked sequences
    correctly under SP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patch_self_attn_for_sp()

    def _patch_self_attn_for_sp(self) -> None:
        for blk in list(self.double_blocks) + list(self.double_final_blocks):
            assert isinstance(blk, WanDoubleStreamAttentionBlock), type(blk)
            _swap_self_attn(blk, WanDoubleStreamSelfAttentionSP)
        for blk in self.single_blocks:
            assert isinstance(blk, WanAttentionBlock), type(blk)
            _swap_self_attn(blk, WanSelfAttentionSP)
