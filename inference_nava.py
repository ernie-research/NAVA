#!/usr/bin/env python3
import os
# 【修复】自动解决 PyTorch 2.x 与 setuptools 的冲突 (Triton 报错)
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import importlib
import sys, time, yaml, argparse, math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import save_image
import torch.nn.functional as F
from scipy import linalg
from functools import partial
from torchvision.io import write_video
import torchaudio
import json

# === 项目依赖 ===
# 请确保这些路径在你的 PYTHONPATH 中
from nava_src.utils.common import set_seed
from nava_src.models.nava.utils.model_loading_utils import init_fusion_score_model_ovi, init_text_model, init_wan_vae_2_2, load_fusion_checkpoint

# Reuse pe_src as the single source of truth for rewrite system prompt + output
# cleanup. Mirrors gradio_server.py / comfyui_nava / vLLM batch path.
_PE_SRC = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pe_src")
if _PE_SRC not in sys.path:
    sys.path.insert(0, _PE_SRC)
from rewrite_single import SYSTEM_PROMPT as REWRITE_SYSTEM_PROMPT
from rewrite import extract_rewrite as _extract_rewrite

import re as _re

# Count completed <S>...<E> pairs in a rewriter output. Mirrors gradio_demo
# gradio_server.py:_count_speech_tags. Used to retry rewrites whose pair count
# drifts from the user's input — Qwen3-Thinking sometimes drops or duplicates
# tags despite the SYSTEM_PROMPT spelling out "preserve speech verbatim".
_SE_PAIR_RE = _re.compile(r"<S>.*?<E>", _re.DOTALL)


def _count_se_pairs(text: str) -> int:
    return len(_SE_PAIR_RE.findall(text or ""))


# ---------------------------------------------------------------
# Broadcast helpers (gradio_server.py 同款 NCCL uint8 string broadcast)
# Used so rank0 改写后所有 rank 拿到同一段 caption — SP 模式下 T5 latent
# 才能跨 rank 一致，否则 sequence-parallel all-gather 会拼出脏数据。
# ---------------------------------------------------------------
def _broadcast_string(s: str, src: int = 0) -> str:
    if dist.get_rank() == src:
        data = s.encode("utf-8")
        length = torch.tensor([len(data)], dtype=torch.long, device="cuda")
    else:
        data = b""
        length = torch.tensor([0], dtype=torch.long, device="cuda")
    dist.broadcast(length, src=src)
    n = int(length.item())
    if dist.get_rank() == src:
        tensor = torch.tensor(list(data), dtype=torch.uint8, device="cuda")
    else:
        tensor = torch.empty(n, dtype=torch.uint8, device="cuda")
    dist.broadcast(tensor, src=src)
    if dist.get_rank() != src:
        s = bytes(tensor.cpu().tolist()).decode("utf-8")
    return s


# -----------------------------
# Prompt Rewriter (onload/offload)
# -----------------------------
class PromptRewriter:
    """Rewriter that loads to GPU on demand and offloads after use.

    Behavior aligned with pe_src/gradio_server.py + comfyui_nava: SYSTEM_PROMPT
    from rewrite_single, output cleanup via rewrite.extract_rewrite (handles
    Qwen3-Thinking <think>...</think> blocks plus 3 fallback leak cases).
    """

    def __init__(self, model_path: str, device: str = "cuda:0"):
        print(f"[Rewriter] Loading {model_path} to CPU...")
        t0 = time.time()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self._device = device
        self._on_gpu = False
        print(f"[Rewriter] Loaded in {time.time() - t0:.1f}s (on CPU)")

    def onload(self):
        """Move model to GPU for rewriting."""
        if not self._on_gpu:
            self.model.to(self._device)
            self._on_gpu = True
            print(f"[Rewriter] Onloaded to {self._device}")

    def offload(self):
        """Move model to CPU to free GPU memory for NAVA inference."""
        if self._on_gpu:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self._on_gpu = False
            print("[Rewriter] Offloaded to CPU")

    def rewrite(self, text: str, expected_se_pairs: int = None,
                max_retries: int = 5) -> str:
        """Rewrite a single prompt. Handles onload/offload automatically.

        If expected_se_pairs is given, retry up to max_retries times until the
        rewritten output's <S>...<E> pair count matches. On persistent
        mismatch, fall back to the last attempt with a WARN log (single bad
        sample shouldn't crash the whole batch).
        """
        self.onload()
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self._device)
        print(f"[Rewriter] IN  ({len(text)} chars): {text}", flush=True)
        if expected_se_pairs is not None:
            print(f"[Rewriter] target <S><E> pairs: {expected_se_pairs}", flush=True)

        last_result = ""
        for attempt in range(max_retries):
            print(f"[Rewriter] Generating attempt {attempt+1}/{max_retries} "
                  f"(input tokens: {inputs['input_ids'].shape[1]})...", flush=True)
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=4096,
                    temperature=0.3, top_p=0.75, top_k=20,
                    do_sample=True, repetition_penalty=1.05,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            result = _extract_rewrite(raw)
            n_pairs = _count_se_pairs(result)
            elapsed = time.time() - t0
            print(f"[Rewriter] Done in {elapsed:.1f}s "
                  f"({len(new_tokens)} tokens, <S><E>={n_pairs})", flush=True)
            last_result = result
            if expected_se_pairs is None or n_pairs == expected_se_pairs:
                print(f"[Rewriter] OUT ({len(result)} chars): {result}", flush=True)
                self.offload()
                return result
            print(f"[Rewriter] <S><E> mismatch: got {n_pairs}, want "
                  f"{expected_se_pairs} — retrying", flush=True)

        print(f"[Rewriter] WARN: <S><E> mismatch persisted after {max_retries} "
              f"retries (last got {_count_se_pairs(last_result)}, "
              f"want {expected_se_pairs}) — using last result", flush=True)
        print(f"[Rewriter] OUT ({len(last_result)} chars): {last_result}", flush=True)
        self.offload()
        return last_result


# -----------------------------
# Image Captioner (onload/offload, rank0 only)
# -----------------------------
class ImageCaptioner:
    """Qwen3-VL captioner. Loaded to CPU, onloaded to GPU per-call.

    SYSTEM_PROMPT mirrors comfyui_nava/captioner.py verbatim — keeps the
    image-side caption style consistent across NAVA's three frontends.
    """

    SYSTEM_PROMPT = (
        "你是一个视频生成提示词助手。用一段流畅的中文描述图片中的场景：人物外貌、"
        "动作、服装、背景环境、光线与色调、整体氛围。不要使用markdown格式、不要分条列举、"
        "不要说\"这张图\"或\"这是一张图片\"，直接描述画面内容，像在描述一段正在发生的"
        "视频场景。输出一段话，不超过150字。"
    )
    USER_INSTRUCTION = "请描述这张图片的视频场景。"

    def __init__(self, model_path: str, device: str = "cuda:0"):
        print(f"[Captioner] Loading {model_path} to CPU...")
        t0 = time.time()
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _Auto
        except ImportError:
            from transformers import AutoModelForCausalLM as _Auto
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = _Auto.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self._device = device
        self._on_gpu = False
        print(f"[Captioner] Loaded in {time.time() - t0:.1f}s (on CPU)")

    def onload(self):
        if not self._on_gpu:
            self.model.to(self._device)
            self._on_gpu = True
            print(f"[Captioner] Onloaded to {self._device}")

    def offload(self):
        if self._on_gpu:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self._on_gpu = False
            print("[Captioner] Offloaded to CPU")

    def caption(self, image_path: str) -> str:
        self.onload()
        from PIL import Image
        pil = Image.open(image_path).convert("RGB")
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": self.USER_INSTRUCTION},
            ]},
        ]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=[pil], return_tensors="pt").to(self._device)
        print(f"[Captioner] IN  image: {image_path}", flush=True)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=256,
                do_sample=True, temperature=0.3, top_p=0.9,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        elapsed = time.time() - t0
        print(f"[Captioner] Done in {elapsed:.1f}s ({len(new_tokens)} tokens)", flush=True)
        print(f"[Captioner] OUT ({len(result)} chars): {result}", flush=True)
        self.offload()
        return result


def _compose_t2av_prompt(scene_caption: str, user_prompt: str) -> str:
    """Glue VL scene caption + user prompt for the rewriter.
    Caption first, user prompt last so any <S>...<E> stays at the tail.
    Mirrors comfyui_nava/nodes.py NAVAPromptCompose single_speaker mode."""
    cap = (scene_caption or "").strip()
    spk = (user_prompt or "").strip()
    if not cap:
        return spk
    if not spk:
        return cap
    return f"{cap} {spk}"

# -----------------------------
# 分布式工具函数
# -----------------------------
def apply_group_offload(backbone, group_size: int, device):
    """Pipelined CPU↔GPU offload for DiT backbone blocks.

    Uses pinned host memory + a dedicated CUDA stream so transfers overlap
    with GPU compute:
      - pre-hook of group N: wait for N's prefetch; async store(N-1) +
        prefetch(N+1) — both run while N computes.

    Key performance choices:
      - _load uses b.to(device, non_blocking=True): one C++ call per block
        instead of ~400 individual tensor copies → ~400x fewer CUDA API calls.
      - _store (inference only): just re-points p.data to the pinned cpu_buf;
        no GPU→CPU copy needed because weights never change during inference.
      - _param_cache: pre-computed named_parameters lists avoid repeated
        Python generator overhead in the hook hot-path.

    Self-heals between samples: if offload_backbone has moved params off the
    pinned bufs, the pre-hook detects this and reloads the current group before
    allowing the forward to proceed.
    """
    all_blocks = (
        list(backbone.double_blocks) +
        list(backbone.single_blocks) +
        list(backbone.double_final_blocks)
    )
    groups = [all_blocks[i:i + group_size] for i in range(0, len(all_blocks), group_size)]
    n_groups = len(groups)
    blk_idx = {id(b): i for i, b in enumerate(all_blocks)}

    # Move all blocks to CPU then pin every parameter tensor.
    # Pinned (page-locked) memory enables DMA at ~12 GB/s vs ~2 GB/s pageable.
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

    # Pre-cache named_parameters lists — avoids repeated Python generator
    # construction in the hot-path (hook fires n_groups × n_cfg_passes / step).
    _param_cache = [
        list(blk.named_parameters(recurse=True)) for blk in all_blocks
    ]

    xfer_stream = torch.cuda.Stream(device=device)

    def _restore_pinned(gi: int):
        """Re-point p.data to cpu_bufs after offload_backbone breaks the links."""
        for b in groups[gi]:
            idx = blk_idx[id(b)]
            for name, p in _param_cache[idx]:
                if not p.data.is_cuda:
                    p.data = cpu_bufs[idx][name]

    def _load(gi: int):
        """Async pinned-CPU → GPU for all blocks in group gi.

        b.to(device, non_blocking=True) is a single C++ Module.to() call that
        moves all parameters at once using the pinned source for DMA.
        """
        with torch.cuda.stream(xfer_stream):
            for b in groups[gi]:
                b.to(device, non_blocking=True)

    def _store(gi: int):
        """Return group gi params to pinned CPU bufs.

        Inference-only optimisation: weights are read-only, so we skip the
        GPU→CPU copy and just re-point p.data to the still-valid cpu_buf.
        The old GPU tensor is freed by the CUDA allocator after the stream
        that last used it completes.
        """
        for b in groups[gi]:
            idx = blk_idx[id(b)]
            for name, p in _param_cache[idx]:
                if p.data.is_cuda:
                    p.data = cpu_bufs[idx][name]

    # Pre-load first group synchronously so group 0 is ready before any hook fires.
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
                    # Self-heal: cur group ended up on CPU (e.g. offload_backbone
                    # ran between samples).  Restore pinned buf pointers so
                    # b.to(device) can use DMA, then reload synchronously.
                    _restore_pinned(cur_gi)
                    _load(cur_gi)
                    torch.cuda.current_stream().wait_stream(xfer_stream)
                else:
                    # Normal path: wait for the async prefetch issued by the
                    # previous group's pre-hook.
                    torch.cuda.current_stream().wait_stream(xfer_stream)
                # While cur_gi computes: store prev group and prefetch next —
                # both overlap with GPU compute on xfer_stream.
                _store(p_gi)
                _load(n_gi)
                return args
            return pre

        handles.append(group[0].register_forward_pre_hook(make_pre(gi, prev_gi, nxt_gi)))

    return handles


def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank
    else:
        return False, 0, 1, 0

def cleanup_dist(is_ddp: bool):
    if is_ddp:
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.barrier(device_ids=[local_rank])
        except Exception as e:
            print(f"[cleanup_dist] barrier failed (non-fatal): {e}")
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[cleanup_dist] destroy_process_group failed (non-fatal): {e}")

def is_main(rank: int) -> bool:
    return rank == 0


def _convert_backbone_to_sp(backbone):
    """In-place swap every block.self_attn to its SP-aware subclass.

    Weights are preserved via load_state_dict. ``initialize_sequence_parallel_state``
    must already have been called so the new modules pick up ``use_sp`` correctly.
    """
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
        assert isinstance(blk, WanDoubleStreamAttentionBlock), type(blk)
        _swap_self_attn(blk, WanDoubleStreamSelfAttentionSP)
    for blk in backbone.single_blocks:
        assert isinstance(blk, WanAttentionBlock), type(blk)
        _swap_self_attn(blk, WanSelfAttentionSP)

def _to01(x):
    return torch.clamp((x.float() + 1.0) / 2.0, 0.0, 1.0)

def _toWav(x):
    peak = x.abs().max().clamp(min=1e-12)
    x = x * (0.95 / peak)
    x = x.clamp(-1.0, 1.0)
    return x

def makedir_subfolders(root, data_file):
    dimensions = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            dimension = data["dimension"][0]
            if dimension not in dimensions:
                dimensions.append(dimension)
    for dimension in dimensions:
        folder = os.path.join(root, dimension)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

# -----------------------------
# 主流程
# -----------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="inference_results")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    
    # 【修改 1】: num_samples 默认 -1 (跑完全部)，save_images 开关
    parser.add_argument("--num_samples", type=int, default=-1, help="推理样本数。设为 -1 则推理整个数据集")
    parser.add_argument("--save_sample", action="store_true", help="是否将生成的图片保存到硬盘")

    parser.add_argument("--data_format", type=str, required=True)
    # parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seedtts_mode", action="store_true", help="是否为 SeedTTS benchmark 模式")
    parser.add_argument("--seed", type=int, default=100,
                        help="随机种子，覆盖 yaml 里的 seed 字段。默认 100。"
                             "use_sp 模式下所有 rank 共用同一 seed；非 SP 模式自动 +rank。")
    parser.add_argument("--gen_turn", type=int, default=2)
    parser.add_argument("--save_vid_latent", action="store_true", help="是否保存视频的latent")
    parser.add_argument("--is_i2v", action="store_true", help="是否开启i2v模式")
    parser.add_argument("--video_guidance_scale", type=float, default=None,
                        help="视频 CFG scale。设了就覆盖 yaml 里的 video_guidance_scale。"
                             "I2V 推荐 2.0，T2V 推荐 5.0。")
    parser.add_argument("--audio_guidance_scale", type=float, default=None,
                        help="音频 CFG scale。设了就覆盖 yaml 里的 audio_guidance_scale。")
    parser.add_argument("--align_3d_cfg", choices=["auto", "on", "off"], default="auto",
                        help="3D-aligned CFG 开关。auto = 用 yaml 里的 align_3d_cfg，"
                             "on/off 强制覆盖。")
    parser.add_argument("--video_align_guidance_scale", type=float, default=None,
                        help="视频 align-CFG scale（align_3d_cfg=on 时生效）。")
    parser.add_argument("--audio_align_guidance_scale", type=float, default=None,
                        help="音频 align-CFG scale（align_3d_cfg=on 时生效）。")
    parser.add_argument("--timbre_cfg", action="store_true", help="是否开启音色 CFG 控制（需 spk_embs 非空）")
    parser.add_argument("--timbre_align_guidance_scale", type=float, default=1.0, help="音色 CFG 引导强度")
    parser.add_argument("--use_sp", action="store_true",
                        help="启用 Ulysses 序列并行推理：sp_size 自动取自 WORLD_SIZE。"
                             "所有 rank 处理相同样本，仅 rank0 落盘。")
    parser.add_argument("--rewrite", action="store_true", default=False,
                        help="启用 prompt rewriter（默认关闭）")
    parser.add_argument("--rewrite_model", type=str,
                        default=os.path.join(_PE_SRC, "Qwen3-4B-Instruct-2507"),
                        help="Rewriter 模型路径（默认 pe_src/Qwen3-4B-Instruct-2507）")
    parser.add_argument("--vl_rewrite", action="store_true", default=False,
                        help="对带 image_path 的样本：先用 VL 模型对图像打 caption → 与原 prompt compose → 再 rewrite。"
                             "需要同时开启 --rewrite。无 image_path 的样本退化为普通 rewrite。")
    parser.add_argument("--vl_model", type=str,
                        default=os.path.join(_PE_SRC, "Qwen3-VL-4B-Instruct"),
                        help="VL caption 模型路径（默认 pe_src/Qwen3-VL-4B-Instruct）")
    parser.add_argument("--t5_offload", action="store_true",
                        help="T5 文本编码完成后移回 CPU，释放显存供 DiT 使用")
    parser.add_argument("--group_offload", action="store_true",
                        help="DiT backbone 逐组 block CPU↔GPU offload（去噪期间节省显存）")
    parser.add_argument("--offload_group_size", type=int, default=1,
                        help="每次转移的 transformer block 数量（默认 1，越小越省显存但越慢）")
    parser.add_argument("--vae_tiling", action="store_true",
                        help="VAE decode 空间分块（tiled decode），降低 decode 峰值显存")
    parser.add_argument("--vae_tile_size", type=int, nargs=2, default=[22, 40],
                        metavar=("H", "W"), help="Latent tile 大小（默认 22 40，对应 latent 44×80）")
    parser.add_argument("--vae_tile_stride", type=int, nargs=2, default=[14, 26],
                        metavar=("H", "W"), help="Latent tile stride（默认 14 26）")
    parser.add_argument("--weight_dtype", type=str, default="auto",
                        choices=["auto", "bf16", "fp8_e4m3fn"],
                        help="Checkpoint weight format. 'auto' detects fp8 by scanning the "
                             "state-dict; 'fp8_e4m3fn' forces the fp8 patch path; 'bf16' "
                             "is the original behavior (no patching).")

    args = parser.parse_args()
    use_rewrite = args.rewrite
    use_vl_rewrite = args.vl_rewrite
    if use_vl_rewrite and not use_rewrite:
        raise ValueError("--vl_rewrite 必须配合 --rewrite 使用")

    # --- Setup ---
    is_ddp, rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # --- Rewriter / Captioner (rank 0 持有模型；其他 rank 通过 broadcast 接收改写结果) ---
    rewriter = None
    captioner = None
    if use_rewrite and (not is_ddp or rank == 0):
        rewriter = PromptRewriter(model_path=args.rewrite_model, device=f"cuda:{local_rank}")
        print(f"[Rewriter] Enabled. Model: {args.rewrite_model}")
        if use_vl_rewrite:
            captioner = ImageCaptioner(model_path=args.vl_model, device=f"cuda:{local_rank}")
            print(f"[Captioner] Enabled. Model: {args.vl_model}")
    elif not use_rewrite and (not is_ddp or rank == 0):
        print("[Rewriter] Disabled (pass --rewrite to enable)")

    # --- Sequence parallel ---
    if args.use_sp:
        if not is_ddp or world_size < 2:
            raise RuntimeError(
                "--use_sp requires torchrun with WORLD_SIZE >= 2 "
                "(detected WORLD_SIZE={})".format(world_size)
            )
        from nava_src.models.nava.distributed_comms.parallel_states import (
            initialize_sequence_parallel_state,
        )
        initialize_sequence_parallel_state(world_size)
        if is_main(rank):
            print(f"[SP] Sequence parallel enabled, sp_size={world_size}")

    cfg = yaml.safe_load(open(args.config, "r"))
    modality = cfg.get("modality", "audio")
    # In SP mode every rank must share the same noise / sampler state.
    set_seed(args.seed + (0 if args.use_sp else rank))

    # if args.save_sample:
    #     if is_main(local_rank):
    #         os.makedirs(args.out_dir, exist_ok=True)
    #         print(f"[Info] Output dir: {args.out_dir}")
    #         if args.data_file.endswith(".json"):
    #             makedir_subfolders(args.out_dir, args.data_file)

    # --- Model ---
    module_path, class_name = cfg["pipeline"].rsplit(".", 1)
    PipelineClass = getattr(importlib.import_module(module_path), class_name)
    if "video" in modality and "audio" in modality:
        cfg["init_from_meta"] = True
    pipe = PipelineClass.create(
        model_id=cfg.get("model_id", ""),
        use_bf16=cfg["use_bf16"],
        audio_latent_ch=cfg["audio_latent_ch"],
        video_latent_ch=cfg["video_latent_ch"],
        lambda_ddpm=cfg["lambda_ddpm"],
        cfg=cfg,
        device=device,
    )

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        ckpt_fallback = os.path.splitext(ckpt_path)[0] + ".ckpt"
        if os.path.exists(ckpt_fallback):
            if is_main(rank):
                print(f"[INFO] {ckpt_path} not found, falling back to {ckpt_fallback}")
            ckpt_path = ckpt_fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (also tried {ckpt_fallback})")

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path, device="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
        state_dict = ckpt["state_dict"]

    # ----- fp8 detection / patching -----
    # If the checkpoint contains float8_e4m3fn tensors, swap every block-Linear
    # in the freshly-built bf16 model with FP8Linear so load_state_dict can
    # populate `weight` (fp8) and `weight_scale` (bf16) buffers correctly.
    is_fp8_ckpt = any(
        isinstance(v, torch.Tensor) and v.dtype == torch.float8_e4m3fn
        for v in state_dict.values()
    )
    if args.weight_dtype == "fp8_e4m3fn":
        use_fp8 = True
    elif args.weight_dtype == "bf16":
        use_fp8 = False
    else:  # auto
        use_fp8 = is_fp8_ckpt

    if use_fp8 and not is_fp8_ckpt and is_main(rank):
        print("[WARN] --weight_dtype=fp8_e4m3fn but checkpoint contains no fp8 tensors. "
              "Patching anyway; load will likely report missing *_scale keys.")
    if not use_fp8 and is_fp8_ckpt and is_main(rank):
        print("[WARN] Checkpoint is fp8 but --weight_dtype=bf16 was requested. "
              "Skipping the fp8 patch — outputs will be wrong. Did you mean 'auto'?")

    if use_fp8:
        from NAVA_FP8 import patch_model_to_fp8
        n_patched = patch_model_to_fp8(pipe.model)
        if is_main(rank):
            n_fp8_keys = sum(
                1 for v in state_dict.values()
                if isinstance(v, torch.Tensor) and v.dtype == torch.float8_e4m3fn
            )
            print(f"[INFO] fp8 mode: patched {n_patched} Linear modules; "
                  f"checkpoint has {n_fp8_keys} fp8 tensors")

    missing, unexpected = pipe.model.load_state_dict(state_dict, strict=False)
    if is_main(rank):
        print(f"missing: {missing}, unexpected: {unexpected}")
        
    pipe = pipe.to(device)
    pipe.model.eval()
    pipe.model.backbone.set_rope_params()

    if args.use_sp:
        _convert_backbone_to_sp(pipe.model.backbone)
        if is_main(rank):
            print(f"[SP] Patched {len(pipe.model.backbone.double_blocks)} double + "
                  f"{len(pipe.model.backbone.single_blocks)} single + "
                  f"{len(pipe.model.backbone.double_final_blocks)} double_final blocks "
                  "to SP-aware self-attn.")

    pipe._t5_offload = args.t5_offload
    pipe._group_offload = args.group_offload
    if args.t5_offload:
        # Move T5 to CPU *after* pipe.to(device) and torch.compile so the compiled
        # graph targets GPU. It will be moved back to GPU only during text encoding.
        pipe.text_model.model.to("cpu")
        torch.cuda.empty_cache()
        if is_main(rank):
            print("[Offload] T5 CPU offload enabled: encoder moves to GPU only during text encoding")

    if args.group_offload:
        apply_group_offload(pipe.model.backbone, args.offload_group_size, device)
        if is_main(rank):
            total = (len(pipe.model.backbone.double_blocks) +
                     len(pipe.model.backbone.single_blocks) +
                     len(pipe.model.backbone.double_final_blocks))
            print(f"[Offload] DiT group offload enabled: {total} blocks, group_size={args.offload_group_size}")

    # --- Dataset (Normal Map-Style) ---
    if modality == "video":
        from nava_src.data.t2v import T2VDataset
        from nava_src.data.t2v import collate_fn
        ds = T2VDataset(
            data_file=args.data_file,
            format=args.data_format,
            height=args.height,
            width=args.width,
            frames=args.frames,
            patch_size=cfg.get("spatial_downsample", 16), 
            video_vae=pipe.video_vae
            # resolution=args.resolution,
            # image_path=args.image_path,
        )
    elif modality == "audio":
        if args.seedtts_mode:
            from nava_src.data.t2a_seedtts import SeedTTSDatasetWithVAE, collate_fn

            language = "en" if "/en/" in args.data_file else "zh"
            if is_main(rank):
                print(f"[Info] SeedTTS mode: language={language}, meta_file={args.data_file}")

            ds = SeedTTSDatasetWithVAE(
                meta_file=args.data_file,
                language=language,
                audio_vae=pipe.audio_vae,
                audio_tokens_per_sec=cfg["data"].get("audio_tokens_per_sec", 31.25),
                audio_latent_ch=cfg.get("audio_latent_ch", 20),
                use_speech_special_token=cfg["data"].get("use_speech_special_token", False),
                use_avgen_format=cfg.get("use_avgen_format", False)
            )
        else:
            from nava_src.data.t2a import T2ADataset, collate_fn

            ds = T2ADataset(
                data_file=args.data_file,
                format=args.data_format,
                duration=args.duration,
                audio_tokens_per_sec=cfg["data"].get("audio_tokens_per_sec", 31.25),
                audio_latent_ch=cfg.get("audio_latent_ch", 20),
                audio_vae=pipe.audio_vae,
                use_speech_special_token=cfg["data"].get("use_speech_special_token", False),
            )
    elif modality == "audio_video":
        from nava_src.data.t2v import T2AVDataset
        from nava_src.data.t2v import collate_fn
        ds = T2AVDataset(
            data_file=args.data_file,
            format=args.data_format,
            height=args.height,
            width=args.width,
            frames=args.frames,
            patch_size=cfg.get("spatial_downsample", 16), 
            fps=cfg["data"].get("video_fps", 24),
            audio_tokens_per_sec=cfg["data"].get("audio_tokens_per_sec", 31.25),
            audio_vae=pipe.audio_vae,
            use_speech_special_token=cfg["data"].get("use_speech_special_token", False),
            video_vae=pipe.video_vae
            # resolution=args.resolution,
            # image_path=args.image_path,
        )
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    # 使用 DistributedSampler，shuffle=False 保证顺序一致且不重复
    # SP 模式下所有 rank 协同处理同一条样本，使用顺序采样器让每个 rank 拿到完全相同的 batch。
    if args.use_sp:
        from torch.utils.data import SequentialSampler
        sampler = SequentialSampler(ds)
    else:
        sampler = DistributedSampler(ds, shuffle=False, drop_last=False)
    dl = DataLoader(
        ds, 
        batch_size=1, 
        sampler=sampler,
        num_workers=0, #cfg.get("num_workers", 4), 
        collate_fn=partial(collate_fn), 
        drop_last=False,
        pin_memory=False
    )

    if is_main(rank):
        print(f"Total dataset size: {len(ds)}")
        print(f"Batches per GPU: {len(dl)}")

    # --- Variables ---
    real_features_list = []
    fake_features_list = []
    local_clip_score_sum = 0.0
    local_clip_count = 0
    generated_count = 0 
    save_vid_latent = args.save_vid_latent
    is_i2v = args.is_i2v
    
    dtype = torch.bfloat16 if cfg["use_bf16"] else torch.float16
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype)
    # SP 模式下所有 rank 拿到相同输出，仅 rank0 写盘；DDP 非 SP 模式每个 rank 写各自 batch。
    is_writer = (not args.use_sp) or is_main(rank)

    # --- Loop ---
    from tqdm import tqdm
    for gen_turn in range(args.gen_turn):
        generated_count = 0 
        for i, batch in enumerate(tqdm(dl)):
            # print(batch["save_path"])
            # 如果指定了 num_samples (且 >0)，则进行截断
            if args.num_samples > 0:
                # SP 模式下所有 rank 协同处理同一条样本，配额按全局计；DDP 非 SP 模式按 rank 平均切分。
                samples_per_gpu = args.num_samples if args.use_sp else math.ceil(args.num_samples / world_size)
                if generated_count >= samples_per_gpu:
                    break
            
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            save_paths = batch['save_path']

            # ── Resume：推理前计算目标路径，已存在则跳过 ──
            if "video" in modality and "audio" in modality and not save_vid_latent:
                base_name = save_paths[0].rsplit('.', 1)[0]
                final_save_path = os.path.join(args.out_dir, f"{base_name}-av-{gen_turn}.mp4")
            elif "video" in modality and save_vid_latent:
                final_save_path = os.path.join(args.out_dir, save_paths[0] + f"-{gen_turn}.pt")
            elif "video" in modality:
                final_save_path = os.path.join(args.out_dir, save_paths[0][:-4] + f"-{gen_turn}.mp4")
            elif "audio" in modality and args.seedtts_mode:
                final_save_path = os.path.join(args.out_dir, save_paths[0])
            else:
                final_save_path = os.path.join(args.out_dir, save_paths[0] + f"-{gen_turn}.wav")

            if os.path.exists(final_save_path):
                print(f"[Resume] skip existing: {final_save_path}")
                generated_count += 1
                continue

            if True:
                # --- Prompt Rewrite (rank0 改写 + broadcast 到所有 rank) ---
                # SP 模式所有 rank 共用同一条样本，T5 latent 必须跨 rank 一致；
                # 单卡 / 非 SP DDP 也保持同一通路（rank0-only + broadcast），
                # 行为对齐 pe_src/gradio_server.py。
                #
                # 关键：rank0 跑 rewriter.generate(do_sample=True) 会消耗 rank0
                # 的 CPU/CUDA RNG state（每条 caption ~1000+ token 的 multinomial
                # 采样），rank1-7 在 _broadcast_string 上阻塞、RNG 完全没动。
                # 紧接着 pipe.sample 里的 torch.randn 会让初始 noise 跨 rank
                # 不一致 → SP all-gather 拼出脏 latent → 第一帧之外全糊。
                # 解决：rank0 在 rewrite 前 snapshot RNG，rewrite 完恢复，
                # 让 rewrite 对外完全无副作用，跨 rank RNG 维持同步。
                if use_rewrite:
                    captions = batch.get("captions", None)
                    if captions is not None:
                        is_str_input = isinstance(captions, str)
                        caps_list = [captions] if is_str_input else list(captions)
                        # Per-sample image_path comes through the dataset (None when no i2v).
                        # collate_fn passes a list[str|None] for non-tensor heterogeneous keys.
                        img_paths = batch.get("image_path", None)
                        if img_paths is None:
                            img_paths_list = [None] * len(caps_list)
                        elif isinstance(img_paths, str):
                            img_paths_list = [img_paths]
                        else:
                            img_paths_list = list(img_paths)
                        for i in range(len(caps_list)):
                            if rank == 0 and rewriter is not None:
                                # Dataset injects <extra_id_2> right after every <S> as a
                                # T5 speech-segment sentinel (see t2v.py:391). The rewriter
                                # doesn't know about that token — strip it before rewrite,
                                # re-inject after, so the rewritten prompt carries the
                                # sentinel exactly where its (possibly rewritten) <S> tags
                                # land.
                                cap_in = caps_list[i].replace("<extra_id_2>", "")
                                _cpu_rng = torch.get_rng_state()
                                _cuda_rng = torch.cuda.get_rng_state(device)
                                # Optional VL caption + compose: only fires when --vl_rewrite
                                # is on AND this sample has an image_path. The composed
                                # prompt feeds the rewriter; samples without image_path fall
                                # through to plain rewrite.
                                if captioner is not None and img_paths_list[i]:
                                    scene = captioner.caption(img_paths_list[i])
                                    cap_in = _compose_t2av_prompt(scene, cap_in)
                                    print(f"[Compose] OUT ({len(cap_in)} chars): {cap_in}", flush=True)
                                # Rewriter sometimes drops/dups <S>...<E>; gate on the input
                                # pair count and let rewrite() retry until they match.
                                expected_pairs = _count_se_pairs(cap_in)
                                cap_out = rewriter.rewrite(
                                    cap_in, expected_se_pairs=expected_pairs,
                                )
                                torch.set_rng_state(_cpu_rng)
                                torch.cuda.set_rng_state(_cuda_rng, device)
                                caps_list[i] = cap_out.replace("<S>", "<S><extra_id_2>")
                            if is_ddp:
                                caps_list[i] = _broadcast_string(
                                    caps_list[i] if rank == 0 else "", src=0,
                                )
                        batch["captions"] = caps_list[0] if is_str_input else caps_list

                # Per-sample is_i2v from batch (unified json), fallback to global args.is_i2v
                sample_is_i2v = batch.get("is_i2v", is_i2v)
                if isinstance(sample_is_i2v, torch.Tensor):
                    sample_is_i2v = sample_is_i2v.item()
                elif isinstance(sample_is_i2v, list):
                    sample_is_i2v = sample_is_i2v[0] if sample_is_i2v else is_i2v

                with amp_ctx:
                    _align_3d = (cfg.get("align_3d_cfg", False) if args.align_3d_cfg == "auto"
                                 else (args.align_3d_cfg == "on"))
                    gen_vid_out, gen_aud_out = pipe.sample(
                        batch,
                        num_steps=args.steps,
                        audio_guidance_scale=(args.audio_guidance_scale
                                              if args.audio_guidance_scale is not None
                                              else cfg.get("audio_guidance_scale", 4.0)),
                        video_guidance_scale=(args.video_guidance_scale
                                              if args.video_guidance_scale is not None
                                              else cfg.get("video_guidance_scale", 5.0)),
                        align_3d_cfg=_align_3d,
                        audio_align_guidance_scale=(args.audio_align_guidance_scale
                                                    if args.audio_align_guidance_scale is not None
                                                    else cfg.get("audio_align_guidance_scale", 4.0)),
                        video_align_guidance_scale=(args.video_align_guidance_scale
                                                    if args.video_align_guidance_scale is not None
                                                    else cfg.get("video_align_guidance_scale", 5.0)),
                        save_vid_latent=save_vid_latent,
                        is_i2v=sample_is_i2v,
                        timbre_cfg=args.timbre_cfg or cfg.get("timbre_cfg", False),
                        timbre_align_guidance_scale=args.timbre_align_guidance_scale if args.timbre_cfg else cfg.get("timbre_align_guidance_scale", 3.0),
                        offload_backbone=args.t5_offload or args.group_offload,
                        tiled_vae=args.vae_tiling,
                        vae_tile_size=tuple(args.vae_tile_size),
                        vae_tile_stride=tuple(args.vae_tile_stride),
                    )

                current_batch_size = 0
                if "video" in modality and "audio" in modality and not save_vid_latent:
                    # 1. 视频预处理：转换为 [T, H, W, C] 格式的 uint8
                    gen_vids = _to01(gen_vid_out).float()
                    current_batch_size = gen_vids.shape[0]
                    
                    for idx in range(gen_vids.shape[0]):
                        # 视频帧处理
                        video_tensor = (gen_vids[idx] * 255).clamp(0, 255).to(torch.uint8)
                        video_tensor = video_tensor.permute(0, 2, 3, 1) # [T, C, H, W] -> [T, H, W, C]

                        # 2. 音频预处理：确保是 [C, L] 格式
                        aud = gen_aud_out[idx]
                        waveform = _toWav(aud["waveform"])
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0) # [1, L]
                        
                        # 采样率
                        sample_rate = aud["sample_rate"]
                        
                        # 3. 构造保存路径
                        # 去掉原后缀（如 .mp4 或 .wav），加上标识
                        base_name = save_paths[idx].rsplit('.', 1)[0]
                        save_path = os.path.join(args.out_dir, f"{base_name}-av-{gen_turn}.mp4")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        # 4. 同时写入视频和音频
                        # 注意：audio_array 必须在 CPU 上
                        if is_writer:
                            write_video(
                                save_path,
                                video_tensor,
                                fps=args.fps,
                                video_codec="h264",
                                audio_array=waveform.cpu().float().contiguous(), # 音频数据
                                audio_fps=sample_rate,      # 音频采样率
                                audio_codec="aac",          # 音频编码格式
                                options={"crf": "18"}       # 视频质量参数
                            )
                            print(f"Successfully saved AV merged video: {save_path}")
                else:
                    if "video" in modality and not save_vid_latent:
                        gen_vids = _to01(gen_vid_out).float() # [0, 1] RGB
                        # 数量截断逻辑 (仅当设置 num_samples 时生效)
                        current_batch_size = gen_vids.shape[0]

                        # 【修改 2】: 只有当参数开启时才保存图片
                        if args.save_sample and is_writer:
                            for idx, vid in enumerate(gen_vids):
                                video = (vid * 255).clamp(0, 255).to(torch.uint8) # t c h w
                                print(video.shape, 888888)
                                video = video.permute(0, 2, 3, 1)
                                write_video(
                                    os.path.join(args.out_dir, save_paths[idx][:-4]+f"-{gen_turn}.mp4"),
                                    video,        # T H W C  uint8
                                    fps=args.fps,
                                    video_codec="h264",
                                    options={"crf": "18"}    # 高质量
                                )
                    elif "video" in modality and save_vid_latent:
                        current_batch_size = len(gen_vid_out)
                        if is_writer:
                            for idx, vid in enumerate(gen_vid_out):
                                print(vid.shape, 6666)
                                latent = vid
                                save_path = os.path.join(args.out_dir, save_paths[idx] + f"-{gen_turn}.pt")
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                torch.save(latent, save_path)

                    if "audio" in modality:
                        current_batch_size = len(gen_aud_out)
                        if is_writer:
                            for idx, aud in enumerate(gen_aud_out):
                                waveform = aud["waveform"]
                                sample_rate = aud["sample_rate"]
                                waveform = _toWav(waveform)
                                if waveform.dim() == 1:
                                    waveform = waveform.unsqueeze(0)

                                if args.seedtts_mode:
                                    # SeedTTS 模式：直接使用 batch 中的 save_path（已包含语言和文件名）
                                    save_path = os.path.join(args.out_dir, save_paths[idx])
                                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                    torchaudio.save(
                                        save_path,
                                        waveform.cpu().float(),
                                        sample_rate,
                                    )
                                else:
                                    # 正常 T2A 模式：添加 gen_turn 后缀
                                    save_path = os.path.join(args.out_dir, save_paths[idx] + f"-{gen_turn}.wav")
                                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                    torchaudio.save(
                                        save_path,
                                        waveform.cpu().float(),
                                        sample_rate,
                                    )

            generated_count += current_batch_size
            
            if is_main(rank) and i % 10 == 0:
                print(f"Processed batch {i}/{len(dl)}. Count: {generated_count}")

    if is_ddp:
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            dist.barrier(device_ids=[local_rank])
        except Exception as e:
            print(f"[barrier] failed (non-fatal): {e}")
    if is_main(rank): print("Inference loop finished. Gathering metrics...")

    cleanup_dist(is_ddp)

if __name__ == "__main__":
    main()