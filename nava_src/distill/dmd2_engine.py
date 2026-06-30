"""
DMD2 + CFG distillation engine for NAVA.

Holds three NAVA-shaped models:
  - generator   : trainable, 4-step student (one-forward, no internal cfg)
  - teacher     : frozen bf16, runs full multi-path CFG to synthesize ε_target
  - fake_score  : trainable, models the generator's current distribution

Exposed entry points:
  - generator_step(batch)  -> loss tensor + logs, for generator backward
  - fake_score_step(batch) -> loss tensor + logs, for fake_score backward
  - sample(batch, num_steps=4) -> one-shot 4-step inference for eval/log

The engine deliberately does NOT own optimizers or DDP/FSDP wrapping —
the outer training script handles those via accelerator.prepare(). The
engine only manages the math: timestep sampling, cfg composition, the
DMD2 grad projection trick.

Reused from the existing codebase (do NOT reimplement):
  - FlowMatchScheduler.add_noise_batch / training_target / training_weight
  - NAVA.predict_eps  (teacher + fake_score eval; generator uses a thin
    grad-enabled clone added in this file)
  - pipeline_nava.AudioVideoPipeline.text_model / audio_vae / video_vae
    (for text encoding + final decoding during sample())
"""
from __future__ import annotations

import math
import random
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _to_device(x, device, dtype=None):
    """Recursively move tensors / lists of tensors to ``device``."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype) if dtype is not None else x.to(device)
    if isinstance(x, list):
        return [_to_device(_, device, dtype) for _ in x]
    if isinstance(x, tuple):
        return tuple(_to_device(_, device, dtype) for _ in x)
    return x


# ---------------------------------------------------------------------------
# CFG composition — verbatim port of the rules used inside
# pipeline_nava.AudioVideoPipeline.sample (around L527-548). Centralized here
# so teacher target synthesis and 50-step reference inference share code.
# ---------------------------------------------------------------------------


def compose_cfg_video(
    eps_pos,
    eps_neg,
    eps_align=None,
    *,
    video_guidance: float,
    video_align_guidance: float,
    use_align_3d: bool,
):
    """Combine teacher's per-cfg video velocity predictions into one target.

    Matches pipeline_nava.sample's video branch:
      align off : eps_neg + g * (eps_pos - eps_neg)
      align on  : eps_pos + g * (eps_pos - eps_neg) + g_a * (eps_pos - eps_align)
    """
    if not use_align_3d:
        return eps_neg + video_guidance * (eps_pos - eps_neg)
    assert eps_align is not None, "align_3d_cfg=True requires eps_align"
    return (
        eps_pos
        + video_guidance * (eps_pos - eps_neg)
        + video_align_guidance * (eps_pos - eps_align)
    )


def compose_cfg_audio(
    eps_pos,
    eps_neg,
    eps_align=None,
    eps_timbre=None,
    *,
    audio_guidance: float,
    audio_align_guidance: float,
    timbre_align_guidance: float,
    use_align_3d: bool,
    use_timbre: bool,
):
    """Audio cfg composition — mirrors the four-way branching in
    pipeline_nava.sample (align_3d × timbre_cfg = 4 cases)."""
    if not use_align_3d and not use_timbre:
        return eps_neg + audio_guidance * (eps_pos - eps_neg)
    if use_align_3d and not use_timbre:
        return (
            eps_pos
            + audio_guidance * (eps_pos - eps_neg)
            + audio_align_guidance * (eps_pos - eps_align)
        )
    if use_timbre and not use_align_3d:
        return (
            eps_pos
            + audio_guidance * (eps_pos - eps_neg)
            + timbre_align_guidance * (eps_pos - eps_timbre)
        )
    # both align_3d and timbre
    return (
        eps_pos
        + audio_guidance * (eps_pos - eps_neg)
        + audio_align_guidance * (eps_pos - eps_align)
        + timbre_align_guidance * (eps_pos - eps_timbre)
    )


# ---------------------------------------------------------------------------
# Grad-enabled predict_eps
#
# ``NAVA.predict_eps`` is decorated with @torch.no_grad — perfect for the
# teacher and for fake_score eval, but the generator and fake_score *training*
# branches need gradients. We replicate the body verbatim minus the decorator.
# Keeping it here (instead of patching model_nava.py) localizes the diff to
# the distill package, so production inference is untouched.
# ---------------------------------------------------------------------------


def predict_eps_with_grad(
    model,
    *,
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
):
    """Mirror of ``NAVA.predict_eps`` with gradients enabled.

    ``model`` must be an ``NAVA`` instance (or FSDP-wrapped one — see
    ``DMD2DistillEngine._unwrap``). Returns ``(velocity_pred_vid,
    velocity_pred_audio)`` shaped as flat [Total_Pixels, C] tensors,
    matching pipeline_nava's contract.
    """
    has_video = latents_vid is not None
    has_audio = latents_audio is not None
    max_seq_len_audio, max_seq_len_video = None, None
    batch_size = len(vid_context) if has_video else len(audio_context)

    if has_audio:
        max_seq_len_audio = audio_len_list.max()
    if has_video:
        if not model.use_mmdit_model:
            ph, pw = (
                model.backbone.video_model.patch_size[1],
                model.backbone.video_model.patch_size[2],
            )
        else:
            ph, pw = model.backbone.patch_size[1], model.backbone.patch_size[2]
        max_seq_len_video = max(
            int(
                (
                    t
                    * math.ceil(h / model.patch_size) * model.patch_size
                    * math.ceil(w / model.patch_size) * model.patch_size
                )
                // (ph * pw)
            )
            for (t, h, w) in t_h_w_list
        )

    xt_list_vid: Optional[list] = [] if has_video else None
    xt_list_audio: Optional[list] = [] if has_audio else None
    offset_vid, offset_audio = 0, 0

    for i in range(batch_size):
        if has_video:
            ps = model.patch_size
            t, h, w = (
                int(t_h_w_list[i][0]),
                int(t_h_w_list[i][1]),
                int(t_h_w_list[i][2]),
            )
            valid_len = t * h * w
            z = latents_vid[offset_vid : offset_vid + valid_len, :]
            offset_vid += valid_len

            xt_reshaped = z.transpose(0, 1).view(model.video_latent_ch, t, h, w)
            if is_i2v:
                xt_reshaped[:, :1, :, :] = first_frames[i].permute(3, 0, 1, 2)

            pad_h = (ps - h % ps) % ps
            pad_w = (ps - w % ps) % ps
            if pad_h > 0 or pad_w > 0:
                xt_reshaped = nn.functional.pad(
                    xt_reshaped, (0, pad_w, 0, pad_h), mode="constant", value=0
                )
            xt_list_vid.append(xt_reshaped)

        if has_audio:
            audio_len = int(audio_len_list[i])
            z_audio = latents_audio[offset_audio : offset_audio + audio_len, :]
            offset_audio += audio_len
            xt_list_audio.append(z_audio)

    pred_video_list, pred_audio_list = model.backbone(
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

    velocity_pred_vid = torch.zeros_like(latents_vid) if has_video else None
    velocity_pred_audio = torch.zeros_like(latents_audio) if has_audio else None
    offset_vid, offset_audio = 0, 0

    if has_video:
        for i, pred in enumerate(pred_video_list):
            t, h, w = t_h_w_list[i]
            pred = pred[:, :t, :h, :w]
            flat_pred = pred.permute(1, 2, 3, 0).flatten(0, 2)
            valid_len = flat_pred.shape[0]
            velocity_pred_vid[offset_vid : offset_vid + valid_len, :] = flat_pred
            offset_vid += valid_len

    if has_audio:
        for i, pred in enumerate(pred_audio_list):
            audio_len = audio_len_list[i]
            flat_pred = pred[:audio_len, :]
            valid_len = flat_pred.shape[0]
            velocity_pred_audio[
                offset_audio : offset_audio + valid_len, :
            ] = flat_pred
            offset_audio += valid_len

    return velocity_pred_vid, velocity_pred_audio


# ---------------------------------------------------------------------------
# DMD2DistillEngine
# ---------------------------------------------------------------------------


class DMD2DistillEngine:
    """Owns the three NAVA models and exposes train_step / sample.

    Lifecycle (constructed once at start of training):
      engine = DMD2DistillEngine(
          generator_pipe, teacher_pipe, fake_score_pipe,
          cfg=cfg, device=device,
      )

    During training the outer loop calls in alternation:
      loss_g, logs_g = engine.generator_step(batch)   # backward → opt_g
      loss_f, logs_f = engine.fake_score_step(batch)  # backward → opt_f

    The engine intentionally does not call optimizer.step / zero_grad. The
    outer script (which wraps each NAVA with its own accelerator.prepare)
    drives those.

    Parameters
    ----------
    generator_pipe, teacher_pipe, fake_score_pipe
        Three ``AudioVideoPipeline`` instances. They share the same VAE +
        text encoder references (instantiated once in the outer script and
        attached to all three pipes) — only ``.model`` differs.
    cfg
        Full yaml config dict. Reads the ``distill:`` sub-block.
    device
        torch.device for the active rank.
    """

    def __init__(self, generator_pipe, teacher_pipe, fake_score_pipe, cfg, device):
        self.generator_pipe = generator_pipe
        self.teacher_pipe = teacher_pipe
        self.fake_score_pipe = fake_score_pipe
        self.cfg = cfg
        self.device = device

        dcfg = cfg.get("distill", {})
        self.train_sigmas = list(dcfg.get("train_sigmas", [1.0, 0.75, 0.5, 0.25]))
        self.num_inference_steps = int(dcfg.get("num_inference_steps", 4))
        self.use_loss_reweight = bool(dcfg.get("use_loss_reweight", True))

        # CFG composition weights, used for teacher target synthesis.
        self.cfg_video = float(dcfg.get("teacher_video_cfg", 3.0))
        self.cfg_audio = float(dcfg.get("teacher_audio_cfg", 2.0))
        self.cfg_video_align = float(dcfg.get("teacher_video_align_cfg", 3.0))
        self.cfg_audio_align = float(dcfg.get("teacher_audio_align_cfg", 2.0))
        self.cfg_timbre_align = float(dcfg.get("teacher_timbre_align_cfg", 3.0))
        self.use_align_3d = bool(dcfg.get("teacher_align_3d_cfg", True))
        self.use_timbre_cfg = bool(dcfg.get("teacher_timbre_cfg", True))

        # DMD2 mechanics
        self.dmd2_loss_weight = float(dcfg.get("dmd2_loss_weight", 1.0))
        self.dmd2_use_x0_grad = bool(dcfg.get("dmd2_use_x0_grad", True))

        # Build the discrete training-sigma → timestep table, with the same
        # shift transform as nava.yaml's ``model.shift``.
        sched = generator_pipe.scheduler   # FlowMatchScheduler, video branch
        shift = float(cfg["model"].get("shift", 5.0))
        # σ' = shift * σ / (1 + (shift-1) * σ)   (verbatim from
        # FlowMatchScheduler.set_timesteps when exponential_shift is False).
        sigmas_t = torch.tensor(self.train_sigmas, dtype=torch.float64)
        shifted = shift * sigmas_t / (1.0 + (shift - 1.0) * sigmas_t)
        self.train_sigma_table = shifted.float().to(device)
        self.train_timestep_table = (
            self.train_sigma_table * sched.num_train_timesteps
        )

        # Frozen teacher: never train, always eval, never tracked by autograd.
        for p in self.teacher_pipe.model.parameters():
            p.requires_grad_(False)
        self.teacher_pipe.model.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unwrap(self, pipe):
        """Reach into FSDP/DDP/EMA to get the raw NAVA module."""
        m = pipe.model
        # accelerator-prepared FSDP exposes ``.module``; orig_params=True
        # versions keep parameters reachable via the wrapper itself.
        while hasattr(m, "module"):
            m = m.module
        return m

    def _pick_sigma_idx(self, n: int = 1, device=None):
        """Sample ``n`` indices into the 4-sigma training table."""
        device = device or self.device
        return torch.randint(0, len(self.train_sigma_table), (n,), device=device)

    def _sigma_to_timestep(self, sigma_value):
        """Map a single sigma value back to a timestep (B,) tensor."""
        return sigma_value * self.generator_pipe.scheduler.num_train_timesteps

    def _add_noise(self, x0_list, noise_list, sigma_value):
        """List-aware (1-σ)·x0 + σ·ε."""
        out = []
        for x, n in zip(x0_list, noise_list):
            out.append((1.0 - sigma_value) * x + sigma_value * n)
        return out

    # ------------------------------------------------------------------
    # Text encoding (shared across all three models — text encoder lives on
    # generator_pipe; teacher / fake_score reach the same module).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_text(self, captions, drop_prob: float = 0.1):
        """Run T5 once, return (pos_context, neg_context, spk_pos_list,
        drop_indices) for all captions.

        Mirrors AudioVideoPipeline.forward (drop_prob=0.1 by default) but
        also produces the *negative* text embedding used by teacher cfg.
        """
        device = self.device
        text_model = self.generator_pipe.text_model
        text_model.model.to(device)

        pos_ctx, spk_pos_list = text_model(
            captions, device, return_spk_pos=True
        )

        # NAVA uses two distinct uncond strings — keep them in sync with
        # pipeline_nava.sample. Negative cond is encoded once per batch
        # since it's the same for all samples.
        video_neg = (
            "画质模糊，多人同时说话，倒着走, 色调艳丽，过曝，静态，细节模糊不清，字幕，"
            "风格，作品，画作，画面，静止，整体发，JPEG压缩残留，丑陋的，残缺的，多余的"
            "手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融"
            "合，静止不动的画面，杂乱的背景，三条腿，背景人很多，音频带有机械音、闷糊、回"
            "音、失真、电流声、爆音、杂音"
        )
        audio_neg = "机械音、闷糊、回音、失真、电流声、爆音、杂音"
        neg_video, neg_audio = text_model([video_neg, audio_neg], device)

        # cond drop for video / audio CFG-distill: zero out a fraction of
        # samples' positive context. Keep the bookkeeping aligned with how
        # AudioVideoPipeline.forward drops.
        b = len(captions)
        drop_probs = torch.rand(b, device=device)
        drop_indices = drop_probs < drop_prob
        if drop_indices.all():
            drop_indices[0] = False
        pos_dropped = [
            torch.zeros_like(ctx) if drop_indices[i] else ctx
            for i, ctx in enumerate(pos_ctx)
        ]
        spk_pos_dropped = [
            pos if not drop_indices[i] else []
            for i, pos in enumerate(spk_pos_list)
        ]
        return {
            "pos_ctx": pos_dropped,
            "neg_video": [neg_video for _ in range(b)],
            "neg_audio": [neg_audio for _ in range(b)],
            "spk_pos": spk_pos_dropped,
            "drop_indices": drop_indices,
        }

    # ------------------------------------------------------------------
    # Generator one-step solve (used by both train and sample paths).
    # ------------------------------------------------------------------

    def _solve_x0(self, xt_flat, v_pred_flat, sigma_value):
        """Given xt and predicted velocity, return analytic x0 estimate.

        Flow-matching velocity convention used throughout NAVA:
            v = (noise - x0)        (see scheduler.training_target)
            xt = (1-σ) · x0 + σ · noise
        Solve for x0:
            x0 = xt - σ · v
        """
        return xt_flat - sigma_value * v_pred_flat

    def _generator_one_step(
        self,
        *,
        latents_audio_init,
        latents_video_init,
        ctx_text,
        sigma_value,
        t_h_w_list,
        audio_len_list,
        spk_embs,
        is_i2v: bool,
        first_frames,
        with_grad: bool,
    ):
        """Run ONE forward of the generator with the chosen sigma anchor.

        Returns (x0_video_flat, x0_audio_flat) — both ``[N, C]`` — the
        student's denoised estimates. Either may be None when that
        modality is absent in this batch.
        """
        gen = self._unwrap(self.generator_pipe)
        ctx = nullcontext() if with_grad else torch.no_grad()
        timestep = self._sigma_to_timestep(sigma_value).reshape(1)

        with ctx:
            v_vid, v_audio = predict_eps_with_grad(
                gen,
                vid_context=ctx_text["pos_ctx"] if latents_video_init is not None else None,
                audio_context=ctx_text["pos_ctx"] if latents_audio_init is not None else None,
                latents_vid=latents_video_init,
                latents_audio=latents_audio_init,
                timesteps=timestep,
                spk_embs=spk_embs,
                spk_pos=ctx_text["spk_pos"],
                t_h_w_list=t_h_w_list,
                audio_len_list=audio_len_list,
                is_i2v=is_i2v,
                first_frames=first_frames,
            )

        x0_vid = (
            self._solve_x0(latents_video_init, v_vid, sigma_value)
            if v_vid is not None else None
        )
        x0_audio = (
            self._solve_x0(latents_audio_init, v_audio, sigma_value)
            if v_audio is not None else None
        )
        return x0_vid, x0_audio

    # ------------------------------------------------------------------
    # Teacher target (multi-path CFG composed velocity).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _teacher_target(
        self,
        *,
        xt_video_flat,
        xt_audio_flat,
        ctx_text,
        timestep,
        spk_embs,
        t_h_w_list,
        audio_len_list,
        is_i2v: bool,
        first_frames,
    ):
        """Run teacher's multi-path CFG and return composed (v_video, v_audio).

        Teacher's frozen ``predict_eps`` is called up to 4 times:
          1. positive cond
          2. negative cond
          3. align_3d cond (masking_modality=True)
          4. timbre uncond (spk_embs=None) — only if use_timbre and spk_embs

        Then composed via ``compose_cfg_video`` / ``compose_cfg_audio``.
        """
        teacher = self._unwrap(self.teacher_pipe)
        has_video = xt_video_flat is not None
        has_audio = xt_audio_flat is not None
        effective_timbre = self.use_timbre_cfg and spk_embs is not None

        kwargs_common = dict(
            latents_vid=xt_video_flat,
            latents_audio=xt_audio_flat,
            timesteps=timestep,
            t_h_w_list=t_h_w_list,
            audio_len_list=audio_len_list,
            is_i2v=is_i2v,
            first_frames=first_frames,
        )

        # 1) cond positive
        v_pos_vid, v_pos_aud = teacher.predict_eps(
            vid_context=ctx_text["pos_ctx"] if has_video else None,
            audio_context=ctx_text["pos_ctx"] if has_audio else None,
            spk_embs=spk_embs,
            spk_pos=ctx_text["spk_pos"],
            masking_modality=False,
            **kwargs_common,
        )
        # 2) cond negative — uses video_neg / audio_neg text embeddings
        v_neg_vid, v_neg_aud = teacher.predict_eps(
            vid_context=ctx_text["neg_video"] if has_video else None,
            audio_context=ctx_text["neg_audio"] if has_audio else None,
            spk_embs=None,
            slg_layer=11,
            masking_modality=False,
            **kwargs_common,
        )
        # 3) align_3d branch (masking_modality=True)
        v_align_vid = v_align_aud = None
        if self.use_align_3d:
            v_align_vid, v_align_aud = teacher.predict_eps(
                vid_context=ctx_text["pos_ctx"] if has_video else None,
                audio_context=ctx_text["pos_ctx"] if has_audio else None,
                spk_embs=spk_embs,
                masking_modality=True,
                **kwargs_common,
            )
        # 4) timbre uncond (spk_embs=None, but otherwise positive)
        v_timbre_vid = v_timbre_aud = None
        if effective_timbre:
            v_timbre_vid, v_timbre_aud = teacher.predict_eps(
                vid_context=ctx_text["pos_ctx"] if has_video else None,
                audio_context=ctx_text["pos_ctx"] if has_audio else None,
                spk_embs=None,
                spk_pos=ctx_text["spk_pos"],
                masking_modality=False,
                **kwargs_common,
            )

        v_target_vid = v_target_aud = None
        if has_video:
            v_target_vid = compose_cfg_video(
                v_pos_vid, v_neg_vid, v_align_vid,
                video_guidance=self.cfg_video,
                video_align_guidance=self.cfg_video_align,
                use_align_3d=self.use_align_3d,
            )
        if has_audio:
            v_target_aud = compose_cfg_audio(
                v_pos_aud, v_neg_aud, v_align_aud, v_timbre_aud,
                audio_guidance=self.cfg_audio,
                audio_align_guidance=self.cfg_audio_align,
                timbre_align_guidance=self.cfg_timbre_align,
                use_align_3d=self.use_align_3d,
                use_timbre=effective_timbre,
            )
        return v_target_vid, v_target_aud

    # ------------------------------------------------------------------
    # Batch unpacking — common to both train_step entries.
    # ------------------------------------------------------------------

    def _prepare_modality_inputs(self, batch, is_i2v: bool = False):
        """Return a dict with audio/video latents pre-noised at the chosen
        sigma, mirroring AudioVideoPipeline.forward's preparation but
        producing flat [N, C] tensors as predict_eps expects.
        """
        device = self.device
        dtype = self._unwrap(self.generator_pipe).dtype

        audio_latents = batch.get("audio_latents")
        video_latents = batch.get("video_latents")
        spk_embs = batch.get("spk_embs")
        t_h_w_list = batch.get("t_h_w_list", None)
        first_frames = batch.get("first_frames", None)
        captions = batch["captions"]

        has_audio = audio_latents is not None
        has_video = video_latents is not None

        out: Dict[str, Any] = {
            "captions": captions,
            "has_audio": has_audio,
            "has_video": has_video,
            "is_i2v": is_i2v and has_video,
            "first_frames": first_frames,
            "t_h_w_list": t_h_w_list,
        }

        # Flatten lists of per-sample latents into the [Total, C] layout
        # predict_eps consumes, and prepare clean references for x0.
        if has_video:
            sf = self.generator_pipe.video_vae.config.scaling_factor
            shf = self.generator_pipe.video_vae.config.shift_factor
            if isinstance(video_latents, list):
                video_z0 = [
                    ((x.to(device).to(dtype)) - shf) * sf for x in video_latents
                ]
            else:
                video_z0 = (video_latents.to(device).to(dtype) - shf) * sf
            out["video_z0_list"] = video_z0  # list of [t,h,w,c] per sample
            # build flat [Total, C]
            flat = torch.cat(
                [v.reshape(-1, v.shape[-1]) for v in video_z0], dim=0
            )
            out["video_z0_flat"] = flat
        if has_audio:
            sf = self.generator_pipe.audio_vae.config.scaling_factor
            shf = self.generator_pipe.audio_vae.config.shift_factor
            if isinstance(audio_latents, list):
                audio_z0 = [
                    ((x.to(device).to(dtype)) - shf) * sf for x in audio_latents
                ]
            else:
                audio_z0 = (audio_latents.to(device).to(dtype) - shf) * sf
            out["audio_z0_list"] = audio_z0
            flat = torch.cat([a.reshape(-1, a.shape[-1]) for a in audio_z0], dim=0)
            out["audio_z0_flat"] = flat
            audio_len_list = torch.tensor(
                [len(a) for a in audio_z0], dtype=torch.int, device=device
            ).unsqueeze(1)
            out["audio_len_list"] = audio_len_list
        else:
            out["audio_len_list"] = None

        # Speaker embeddings: same flattening as pipeline_nava.forward.
        if has_audio and spk_embs is not None:
            flat_spk = [
                emb
                for emb_list in spk_embs
                for emb in emb_list
            ]
            spk_embs = (
                torch.cat(flat_spk, dim=0).to(device).to(dtype)
                if flat_spk else None
            )
        else:
            spk_embs = None
        out["spk_embs"] = spk_embs

        return out

    # ------------------------------------------------------------------
    # generator_step — the core DMD2 + cfg-distill objective.
    # ------------------------------------------------------------------

    def generator_step(self, batch, *, is_i2v_prob: float = 0.0):
        """Compute the DMD2 + cfg-distill loss for the generator.

        Returns ``(loss, logs)`` ready for ``accelerator.backward(loss)``.
        Caller drives ``opt_g.step()`` afterwards.
        """
        prep = self._prepare_modality_inputs(batch, is_i2v=False)
        if not (prep["has_video"] or prep["has_audio"]):
            return None, {}

        ctx_text = self._encode_text(prep["captions"])

        # ---- Step 1: pick a training sigma and form xt for generator forward
        idx_g = self._pick_sigma_idx(1).item()
        sigma_g = self.train_sigma_table[idx_g]
        timestep_g = self.train_timestep_table[idx_g].reshape(1)

        # initial pure-noise → student denoises in one shot
        if prep["has_video"]:
            noise_v0 = torch.randn_like(prep["video_z0_flat"])
            xt_video = (1.0 - sigma_g) * prep["video_z0_flat"] + sigma_g * noise_v0
        else:
            xt_video = None
        if prep["has_audio"]:
            noise_a0 = torch.randn_like(prep["audio_z0_flat"])
            xt_audio = (1.0 - sigma_g) * prep["audio_z0_flat"] + sigma_g * noise_a0
        else:
            xt_audio = None

        # Generator predicts velocity → analytic x0
        x0_vid, x0_aud = self._generator_one_step(
            latents_audio_init=xt_audio,
            latents_video_init=xt_video,
            ctx_text=ctx_text,
            sigma_value=sigma_g,
            t_h_w_list=prep["t_h_w_list"],
            audio_len_list=prep["audio_len_list"],
            spk_embs=prep["spk_embs"],
            is_i2v=prep["is_i2v"],
            first_frames=prep["first_frames"],
            with_grad=True,
        )

        # ---- Step 2: re-noise x0_g at a *new* sigma for distillation pair
        idx_kd = self._pick_sigma_idx(1).item()
        sigma_kd = self.train_sigma_table[idx_kd]
        timestep_kd = self.train_timestep_table[idx_kd].reshape(1)

        with torch.no_grad():
            if prep["has_video"]:
                eps_kd_v = torch.randn_like(x0_vid)
                xt_kd_v = (1.0 - sigma_kd) * x0_vid.detach() + sigma_kd * eps_kd_v
            else:
                xt_kd_v = None
            if prep["has_audio"]:
                eps_kd_a = torch.randn_like(x0_aud)
                xt_kd_a = (1.0 - sigma_kd) * x0_aud.detach() + sigma_kd * eps_kd_a
            else:
                xt_kd_a = None

            # Teacher composed cfg target
            v_teach_vid, v_teach_aud = self._teacher_target(
                xt_video_flat=xt_kd_v,
                xt_audio_flat=xt_kd_a,
                ctx_text=ctx_text,
                timestep=timestep_kd,
                spk_embs=prep["spk_embs"],
                t_h_w_list=prep["t_h_w_list"],
                audio_len_list=prep["audio_len_list"],
                is_i2v=prep["is_i2v"],
                first_frames=prep["first_frames"],
            )

            # Fake-score predicts velocity on the same noisy generator sample.
            fake = self._unwrap(self.fake_score_pipe)
            v_fake_vid, v_fake_aud = fake.predict_eps(
                vid_context=ctx_text["pos_ctx"] if prep["has_video"] else None,
                audio_context=ctx_text["pos_ctx"] if prep["has_audio"] else None,
                latents_vid=xt_kd_v,
                latents_audio=xt_kd_a,
                timesteps=timestep_kd,
                spk_embs=prep["spk_embs"],
                spk_pos=ctx_text["spk_pos"],
                t_h_w_list=prep["t_h_w_list"],
                audio_len_list=prep["audio_len_list"],
                is_i2v=prep["is_i2v"],
                first_frames=prep["first_frames"],
                masking_modality=False,
            )

            w_kd = self._loss_reweight(timestep_kd).reshape(())  # scalar
            grad_vid = w_kd * (v_teach_vid - v_fake_vid) if v_teach_vid is not None else None
            grad_aud = w_kd * (v_teach_aud - v_fake_aud) if v_teach_aud is not None else None

        # ---- Step 3: DMD2 grad-projection — only x0_g carries the graph.
        # loss_dmd = 0.5 * MSE(x0_g, (x0_g - grad_signal).detach())
        loss_terms = []
        logs: Dict[str, torch.Tensor] = {}
        if grad_vid is not None:
            tgt_v = (x0_vid - grad_vid).detach()
            l_v = 0.5 * F.mse_loss(x0_vid.float(), tgt_v.float())
            loss_terms.append(l_v)
            logs["dmd2_video"] = l_v.detach().clone()
        if grad_aud is not None:
            tgt_a = (x0_aud - grad_aud).detach()
            l_a = 0.5 * F.mse_loss(x0_aud.float(), tgt_a.float())
            loss_terms.append(l_a)
            logs["dmd2_audio"] = l_a.detach().clone()

        loss = self.dmd2_loss_weight * sum(loss_terms)
        logs["dmd2"] = loss.detach().clone()
        logs["sigma_g"] = sigma_g.detach().clone()
        logs["sigma_kd"] = sigma_kd.detach().clone()
        return loss, logs

    # ------------------------------------------------------------------
    # fake_score_step — denoising MSE on the generator's current samples.
    # ------------------------------------------------------------------

    def fake_score_step(self, batch):
        """Compute the fake-score denoising loss.

        The fake-score net is trained to denoise samples produced by the
        *current* generator (a moving target). At convergence its score
        matches the generator's distribution, which closes the DMD2 loop.
        """
        prep = self._prepare_modality_inputs(batch, is_i2v=False)
        if not (prep["has_video"] or prep["has_audio"]):
            return None, {}

        ctx_text = self._encode_text(prep["captions"])

        # Pick generator sigma + freeze x0_g (no grad through generator).
        idx_g = self._pick_sigma_idx(1).item()
        sigma_g = self.train_sigma_table[idx_g]

        with torch.no_grad():
            if prep["has_video"]:
                noise_v = torch.randn_like(prep["video_z0_flat"])
                xt_v = (1.0 - sigma_g) * prep["video_z0_flat"] + sigma_g * noise_v
            else:
                xt_v = None
            if prep["has_audio"]:
                noise_a = torch.randn_like(prep["audio_z0_flat"])
                xt_a = (1.0 - sigma_g) * prep["audio_z0_flat"] + sigma_g * noise_a
            else:
                xt_a = None

            x0_vid, x0_aud = self._generator_one_step(
                latents_audio_init=xt_a,
                latents_video_init=xt_v,
                ctx_text=ctx_text,
                sigma_value=sigma_g,
                t_h_w_list=prep["t_h_w_list"],
                audio_len_list=prep["audio_len_list"],
                spk_embs=prep["spk_embs"],
                is_i2v=prep["is_i2v"],
                first_frames=prep["first_frames"],
                with_grad=False,
            )
            x0_vid = x0_vid.detach() if x0_vid is not None else None
            x0_aud = x0_aud.detach() if x0_aud is not None else None

        # Now noise x0_g at a fresh sigma and have fake_score predict velocity.
        idx_f = self._pick_sigma_idx(1).item()
        sigma_f = self.train_sigma_table[idx_f]
        timestep_f = self.train_timestep_table[idx_f].reshape(1)

        if x0_vid is not None:
            eps_v = torch.randn_like(x0_vid)
            xt_kd_v = (1.0 - sigma_f) * x0_vid + sigma_f * eps_v
            target_v = eps_v - x0_vid          # = noise - x0  (training target)
        else:
            xt_kd_v = target_v = None
        if x0_aud is not None:
            eps_a = torch.randn_like(x0_aud)
            xt_kd_a = (1.0 - sigma_f) * x0_aud + sigma_f * eps_a
            target_a = eps_a - x0_aud
        else:
            xt_kd_a = target_a = None

        fake = self._unwrap(self.fake_score_pipe)
        v_pred_v, v_pred_a = predict_eps_with_grad(
            fake,
            vid_context=ctx_text["pos_ctx"] if x0_vid is not None else None,
            audio_context=ctx_text["pos_ctx"] if x0_aud is not None else None,
            latents_vid=xt_kd_v,
            latents_audio=xt_kd_a,
            timesteps=timestep_f,
            spk_embs=prep["spk_embs"],
            spk_pos=ctx_text["spk_pos"],
            t_h_w_list=prep["t_h_w_list"],
            audio_len_list=prep["audio_len_list"],
            is_i2v=prep["is_i2v"],
            first_frames=prep["first_frames"],
        )

        w = self._loss_reweight(timestep_f).reshape(()) if self.use_loss_reweight else 1.0
        loss_terms = []
        logs: Dict[str, torch.Tensor] = {}
        if v_pred_v is not None:
            l_v = w * F.mse_loss(v_pred_v.float(), target_v.float())
            loss_terms.append(l_v)
            logs["fake_score_video"] = l_v.detach().clone()
        if v_pred_a is not None:
            l_a = w * F.mse_loss(v_pred_a.float(), target_a.float())
            loss_terms.append(l_a)
            logs["fake_score_audio"] = l_a.detach().clone()

        loss = sum(loss_terms)
        logs["fake_score"] = loss.detach().clone()
        return loss, logs

    # ------------------------------------------------------------------
    # 4-step inference (used by eval / sample logging).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch, num_steps: int = None, decode: bool = True):
        """Run the trained 4-step student over ``num_steps`` sigma anchors.

        Mirrors the structure of pipeline_nava.sample but skips ALL cfg
        forwards — the generator has cfg distilled in, one forward / step.

        Returns whatever pipeline_nava.sample returns when called with the
        same ``batch`` (i.e. a tuple of decoded outputs and the latent
        history). For minimal surface area we delegate the post-loop VAE
        decoding step to the existing pipeline.
        """
        num_steps = num_steps or self.num_inference_steps
        # Sigma schedule for inference: just the configured train_sigmas
        # (already shifted), in descending order.
        sigmas = self.train_sigma_table.sort(descending=True).values[:num_steps]
        device = self.device

        prep = self._prepare_modality_inputs(batch, is_i2v=batch.get("is_i2v", False))
        ctx_text = self._encode_text(prep["captions"], drop_prob=0.0)

        # Initialize from pure noise.
        if prep["has_video"]:
            xt_v = torch.randn_like(prep["video_z0_flat"])
        else:
            xt_v = None
        if prep["has_audio"]:
            xt_a = torch.randn_like(prep["audio_z0_flat"])
        else:
            xt_a = None

        gen = self._unwrap(self.generator_pipe)
        for i, sigma in enumerate(sigmas):
            timestep = self._sigma_to_timestep(sigma).reshape(1)
            v_v, v_a = gen.predict_eps(
                vid_context=ctx_text["pos_ctx"] if xt_v is not None else None,
                audio_context=ctx_text["pos_ctx"] if xt_a is not None else None,
                latents_vid=xt_v,
                latents_audio=xt_a,
                timesteps=timestep,
                spk_embs=prep["spk_embs"],
                spk_pos=ctx_text["spk_pos"],
                t_h_w_list=prep["t_h_w_list"],
                audio_len_list=prep["audio_len_list"],
                is_i2v=prep["is_i2v"],
                first_frames=prep["first_frames"],
                masking_modality=False,
            )
            # Move to next sigma using analytic Euler step on the flow:
            #   x_{t+1} = x_t + (sigma_next - sigma) * v
            sigma_next = sigmas[i + 1] if (i + 1) < len(sigmas) else torch.zeros_like(sigma)
            if v_v is not None:
                xt_v = xt_v + (sigma_next - sigma) * v_v
            if v_a is not None:
                xt_a = xt_a + (sigma_next - sigma) * v_a

        # Hand the final latents back. Caller can decode via
        # generator_pipe.audio_vae / video_vae if desired.
        return {
            "video_latents_flat": xt_v,
            "audio_latents_flat": xt_a,
            "t_h_w_list": prep["t_h_w_list"],
            "audio_len_list": prep["audio_len_list"],
        }

