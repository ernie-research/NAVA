#!/usr/bin/env python3
import importlib
import os, sys, time, math, yaml, argparse, types
from collections import defaultdict
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
from torchvision.utils import make_grid
from functools import partial 
from contextlib import nullcontext, contextmanager
from copy import deepcopy
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from nava_src.models.nava.modules.model    import WanAttentionBlock
from nava_src.models.nava.modules.fusion   import WanFusionBlock
from nava_src.models.nava.modules.model_mm import WanDoubleStreamAttentionBlock
from nava_src.models.nava.modules.model_mm import WanAttentionBlock as WanSingleStreamAttentionBlock
from torch.distributed.fsdp import ShardingStrategy

from nava_src.data.dataset_train import (
    AudioVideoDataset,
    collate_fn,
    collate_fn_batch,
    DistInfo,
)
from nava_src.utils.scheduler import WarmupCosineAnnealingLR

import torch.nn.functional as F
from torchvision.utils import save_image
from types import SimpleNamespace

# 初始化 logger
logger = get_logger(__name__)


class TrainPerfLogger:
    """Performance logger for training loop diagnostics."""

    def __init__(self, print_every=20, rank=0, enabled=True):
        self.print_every = print_every
        self.rank = rank
        self.enabled = enabled
        self._stats = defaultdict(lambda: {"count": 0, "total": 0.0, "max": 0.0})
        self._step_count = 0

    def record(self, key, elapsed):
        if not self.enabled:
            return
        s = self._stats[key]
        s["count"] += 1
        s["total"] += elapsed
        if elapsed > s["max"]:
            s["max"] = elapsed

    def step_done(self):
        if not self.enabled:
            return
        self._step_count += 1
        if self._step_count % self.print_every == 0:
            lines = [f"[TRAIN_PERF R{self.rank}] === Step {self._step_count} Summary (last {self.print_every} steps) ==="]
            for key in sorted(self._stats.keys()):
                s = self._stats[key]
                cnt = s["count"]
                avg = s["total"] / cnt if cnt > 0 else 0
                lines.append(
                    f"  {key}: avg={avg:.4f}s max={s['max']:.4f}s total={s['total']:.2f}s cnt={cnt}"
                )
            print("\n".join(lines), flush=True)
            self._stats.clear()

# -----------------------------
# EMA 工具类 (新增)
# -----------------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """初始化时记录模型当前参数的副本作为EMA初始状态"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """在每次 optimizer.step() 后调用，更新 EMA 权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    @contextmanager
    def average_parameters(self):
        """
        上下文管理器：进入时将模型参数替换为 EMA 参数，
        退出时恢复为原始训练参数。用于评估/采样。
        """
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
        try:
            yield
        finally:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = self.backup[name]
            self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        """
        加载 EMA 状态，并自动将参数移动到模型所在的 device。
        解决 resume 时 map_location='cpu' 导致的设备不一致报错。
        """
        # 1. 获取模型当前参数的 device 映射表
        param_device_map = {name: param.device for name, param in self.model.named_parameters()}
        
        # 2. 遍历加载进来的 state_dict，搬运到对应 device
        for name, tensor in state_dict.items():
            if name in param_device_map:
                target_device = param_device_map[name]
                state_dict[name] = tensor.to(target_device)
        
        # 3. 赋值
        self.shadow = state_dict

# -----------------------------
# 分布式初始化 & 工具函数
# -----------------------------
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
        dist.barrier()
        dist.destroy_process_group()

def is_main(rank: int) -> bool:
    return rank == 0

def fmt_mem():
    if not torch.cuda.is_available():
        return "cpu"
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved() / 1024**2
    return f"{a:.0f}MB/{r:.0f}MB"

def get_lr(optim):
    for pg in optim.param_groups:
        return pg["lr"]

def format_config_for_log(config):
    """
    将配置字典中的复杂类型（List, Dict, None）转换为字符串，
    以满足 TensorBoard add_hparams 的要求。
    """
    new_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            # 递归处理或者直接转字符串，简单起见直接扁平化 key
            for sub_k, sub_v in v.items():
                new_key = f"{k}.{sub_k}"
                # 如果子项还是复杂类型，直接转 str
                if not isinstance(sub_v, (int, float, str, bool, torch.Tensor)):
                    new_config[new_key] = str(sub_v)
                else:
                    new_config[new_key] = sub_v
        elif not isinstance(v, (int, float, str, bool, torch.Tensor)):
            # 遇到 List, None 等，转为字符串
            new_config[k] = str(v)
        else:
            new_config[k] = v
    return new_config

def _to01(x):
    return torch.clamp((x.float() + 1.0) / 2.0, 0.0, 1.0)

def _toWav(x):
    peak = x.abs().max().clamp(min=1e-12)
    x = x * (0.95 / peak)
    x = x.clamp(-1.0, 1.0)
    return x

@torch.no_grad()
def evaluate(pipe, cfg_dict, accelerator: Accelerator, step=None, writer=None):
    pass


def make_video_grid(videos, nrow):
    """
    videos: [B, T, C, H, W] in [0,1]
    return: [T, C, H_grid, W_grid]
    """
    print("make_video_grid", videos.shape)
    B, T, C, H, W = videos.shape
    grids = []
    for t in range(T):
        grid_t = make_grid(videos[:, t], nrow=nrow, normalize=False)
        grids.append(grid_t)
    return torch.stack(grids, dim=0)


@torch.no_grad()
def log_audios_triplet(writer, pipe, batch, device, step, N=8, tag_prefix="samples",
                       sample_steps=25, guidance=4.0, cfg=None):
    dtype = next(pipe.model.parameters()).dtype
    audio_len_list = batch.get("audio_len_list", None)
    
    if "audio_latents" not in batch or batch["audio_latents"] is None:
        print(f"[warn] no audio latents found in batch, skip logging audios at step {step}")
        return 
    N = min(len(batch["audio_latents"]), N)
    audio_latents_list = batch["audio_latents"][:N]
    if audio_len_list is not None:
        audio_len_list = audio_len_list[:N]
    
    pipe.model.eval()
    try:
        for i in range(N):
            decode_latent = audio_latents_list[i].permute(1, 0).unsqueeze(0).tolist()
            decode_data = {"data": decode_latent}
            decode_dict = pipe.audio_vae.decode(decode_data)
            decode_dict = decode_dict.sample if hasattr(decode_dict, "sample") else decode_dict
            waveform, sample_rate = decode_dict["waveform"], decode_dict["sample_rate"]
            # save wavefrom to writer tensorboard
            waveform = _toWav(waveform)
            writer.add_audio(
                tag=f"{tag_prefix}/recon_audio_{i}",
                snd_tensor=waveform,
                sample_rate=sample_rate,
                global_step=step,
            )

        gen_imgs_list, gen_audio_list = pipe.sample(batch, num_steps=sample_steps, audio_guidance_scale=guidance)
        for i, gen_audio in enumerate(gen_audio_list):
            waveform = gen_audio["waveform"]
            sample_rate = gen_audio["sample_rate"]
            waveform = _toWav(waveform)
            writer.add_audio(
                tag=f"{tag_prefix}/gen_audio_{i}",
                snd_tensor=waveform,
                sample_rate=sample_rate,
                global_step=step,
            )

    except Exception as e:
        print(f"[warn] audio logging failed at step {step}: {e}")
    
    finally:
        pipe.model.train()


@torch.no_grad()
def log_images_triplet(writer, pipe, batch, device, step, N=4, tag_prefix="samples",
                       sample_steps=25, guidance=1.0, cfg=None):
    dtype = next(pipe.model.parameters()).dtype
    t_h_w_list = batch.get("t_h_w_list", None)
    if "image_latents" not in batch or batch["image_latents"] is None:
        return
    if t_h_w_list is not None:
        h_w_list = torch.tensor(t_h_w_list)[:, 1:]
    
    # TODO @hujiahao03 
    # 1. packing 推理时样本量少
    # 2. packing 推理疑似存在 bug
    if h_w_list is not None: 
        N = min(len(h_w_list), N)

    image_latents = batch["image_latents"]
    if isinstance(image_latents, torch.Tensor):
        image_latents = image_latents[:N].to(device, dtype=dtype)
    else:
        image_latents = [sample.to(device, dtype=dtype) for sample in batch["image_latents"][:N]]
    # gt = _to01(x)
    rec = None
    gen = None
    
    # TODO @hujiahao03 解决 eval, 适配 packing
    pipe.model.eval()
    try:
        # 1) 重建（VAE）
        if hasattr(pipe, "image_vae") and hasattr(pipe.image_vae, "encode") and hasattr(pipe.image_vae, "decode"):
            try:
                rec_img_list = []
                for i in range(N):
                    s, e = h_w_list[i]
                    if isinstance(image_latents, torch.Tensor):
                        C = image_latents.shape[-1]
                        latent_i_1 = image_latents[i, :s * e, :]
                    else: # packing 模式
                        C = image_latents[i].shape[-1]
                        latent_i_1 = image_latents[i].reshape(-1, C)[:s * e, :]
                    latent_i_2 = latent_i_1.view(s, e, C)
                    latent_i = None
                    if cfg["data"].get("use_local_vae", False) or cfg["data"].get("use_jit", False):
                        latent_i = latent_i_2.unsqueeze(0)
                    else :
                        latent_i = latent_i_2.permute(2, 0, 1).unsqueeze(0)
                    # if cfg["data"]["use_server"] is False :
                    #     posterior = pipe.image_vae.encode(latent_i.float()).latent_dist
                    #     z = posterior.sample() if hasattr(posterior, "sample") else posterior.latent_dist.sample()
                    # else :
                    z = latent_i
                        
                    dec = pipe.image_vae.decode(z)
                    img = dec.sample if hasattr(dec, "sample") else dec
                    
                    img = F.interpolate(
                        img,
                        size=(pipe.image_vae.resolution, pipe.image_vae.resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                    rec_img_list.append(img)
                
                if rec_img_list:
                    rec_img = torch.cat(rec_img_list, dim=0)
                    rec = _to01(rec_img)
                # print("DEBUG vae重建")
                # from IPython import embed
                # embed()
            except Exception as e:
                print(f"[warn] recon logging failed at step {step}: {e}")
                rec = None

        # 2) 生成
        try:
            out, _ = pipe.sample(batch, num_steps=sample_steps, image_guidance_scale=guidance)
            gen = _to01(out[:N].to(device, dtype=torch.float32))
        except Exception as e:
            print(f"[warn] gen logging failed at step {step}: {e}")
            import traceback; traceback.print_exc()
            gen = None
    finally:
        pipe.model.train()

    if rec is not None:
        grid_rec = make_grid(rec.cpu(), nrow=N, normalize=False)
        writer.add_image(f"{tag_prefix}/recon", grid_rec, step)

    if gen is not None:
        grid_gen = make_grid(gen.cpu(), nrow=N, normalize=False)
        writer.add_image(f"{tag_prefix}/gen", grid_gen, step)


@torch.no_grad()
def log_videos_triplet(writer, pipe, batch, device, step, N=4, tag_prefix="samples",
                       sample_steps=25, guidance=1.0, cfg=None):
    """
    log video samples to tensorboard
    """
    dtype = next(pipe.model.parameters()).dtype
    t_h_w_list = batch.get("t_h_w_list", None)
    fps = cfg["data"].get("video_fps", 8)
    log_width = cfg["data"].get("log_width", 336)
    log_height = cfg["data"].get("log_height", 192)
    
    if "video_latents" not in batch or batch["video_latents"] is None:
        return
    # TODO @hujiahao03 
    # 1. packing 推理时样本量少
    # 2. packing 推理疑似存在 bug
    if t_h_w_list is not None: 
        N = min(len(t_h_w_list), N)
    min_frames = (min([t for t, _, _ in t_h_w_list]) - 1) * 4 + 1

    if isinstance(batch["video_latents"], torch.Tensor):
        x = batch["video_latents"][:N].to(device, dtype=dtype)
    else:
        x = [video_latents.to(device, dtype=dtype) for video_latents in batch["video_latents"][:N]]
    # gt = _to01(x)
    rec = None
    gen = None
    
    # TODO @hujiahao03 解决 eval, 适配 packing
    pipe.model.eval()
    try:
        # 1) 重建（VAE）
        if hasattr(pipe, "video_vae") and hasattr(pipe.video_vae, "encode") and hasattr(pipe.video_vae, "decode"):
            try:
                rec_vid_list = []
                for i in range(N):
                    t, s, e = t_h_w_list[i]
                    if isinstance(x, torch.Tensor):
                        C = x.shape[-1]
                        latent_i_1 = x[i, :t * s * e, :]
                    else: # packing 模式
                        C = x[i].shape[-1]
                        latent_i_1 = x[i].reshape(-1, C)[:t * s * e, :]
                    latent_i_2 = latent_i_1.view(t, s, e, C)
                    latent_i = None
                    if cfg["data"].get("use_local_vae", False) or cfg["data"].get("use_jit", False):
                        latent_i = latent_i_2
                    else :
                        latent_i = latent_i_2.permute(0, 3, 1, 2)
                    # if cfg["data"]["use_server"] is False :
                    #     posterior = pipe.video_vae.encode(latent_i.float()).latent_dist
                    #     z = posterior.sample() if hasattr(posterior, "sample") else posterior.latent_dist.sample()
                    # else :
                    z = latent_i
                        
                    dec = pipe.video_vae.decode(z)
                    vid = dec.sample if hasattr(dec, "sample") else dec
                    while vid.shape[1] == 1 and vid.shape[2] == 1:
                        time.sleep(0.2)
                        print("retry decoding for failure cases")
                        dec = pipe.video_vae.decode(z)
                        vid = dec.sample if hasattr(dec, "sample") else dec
                    
                    vid = F.interpolate(
                        vid,
                        size=(log_height, log_width),
                        mode="bilinear",
                        align_corners=False,
                    )[:min_frames].unsqueeze(0)
                    rec_vid_list.append(vid)
                
                if rec_vid_list:
                    rec_vid = torch.cat(rec_vid_list, dim=0) # b, t, c, h, w
                    rec = _to01(rec_vid)
            except Exception as e:
                print(f"[warn] recon logging failed at step {step}: {e}")
                rec = None

        # 2) 生成
        # try:
        out, _ = pipe.sample(batch, num_steps=sample_steps, video_guidance_scale=guidance, num_samples=N)
        gen = _to01(out[:N].to(device, dtype=torch.float32))
        # except Exception as e:
        #     print(f"[warn] gen logging failed at step {step}: {e}")
        #     import traceback; traceback.print_exc()
        #     gen = None
    finally:
        pipe.model.train()

    # print("DEBUG log_images_triplet")
    # from IPython import embed
    # embed()
    # if cfg["data"]["use_server"] is False:
    #     grid_gt = make_grid(gt.cpu(), nrow=N, normalize=False)
    #     writer.add_image(f"{tag_prefix}/gt", grid_gt, step)

    if rec is not None and writer is not None:
        grid_rec = make_video_grid(rec.cpu(), nrow=N)
        # writer.add_image(f"{tag_prefix}/recon", grid_rec, step)
        writer.add_video(f"{tag_prefix}/recon_vid", grid_rec.unsqueeze(0), step, fps=fps)

    if gen is not None and writer is not None:
        grid_gen = make_video_grid(gen.cpu(), nrow=N)
        writer.add_video(f"{tag_prefix}/gen_vid", grid_gen.unsqueeze(0), step, fps=fps)
        # writer.add_image(f"{tag_prefix}/gen", grid_gen, step)

    # rows = []
    # if rec is not None: rows.append(grid_rec)
    # if gen is not None: rows.append(grid_gen)

    # if len(rows) > 1:
    #     panel = torch.cat(rows, dim=2)
    #     writer.add_video(f"{tag_prefix}/panel_gt_recon_gen_vid", panel, step, fps=fps)


def merge_state_across_gpus(local_state, world_size):
    # gather all local states
    gather_list = [
        torch.zeros_like(local_state) for _ in range(world_size)
    ]
    dist.all_gather(gather_list, local_state)

    # final merged state
    final_state = torch.zeros_like(local_state)

    # 合并所有 rank：非零的覆盖零
    for t in gather_list:
        mask = (t != 0)
        final_state = torch.where(mask, t, final_state)

    return final_state

# -----------------------------
# 训练主函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None, help="从 ckpt 恢复")
    parser.add_argument("--load_ckpt_only", action='store_true')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    # -------------------------
    # 1. 初始化 Accelerator
    # -------------------------
    # gradient_accumulation_steps: 让 accelerate 自动管理梯度累积
    # mixed_precision: 让 accelerate 管理 (bf16/fp16/no)，通常在 launch config 中指定
    project_config = ProjectConfiguration(project_dir=cfg["out_dir"], logging_dir=os.path.join(cfg["out_dir"], "tb"))

    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))

    use_mmdit_model = cfg.get("use_mmdit_model", False)
    if not use_mmdit_model:
        wan_auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={WanFusionBlock},
        )
    else:
        wan_auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={WanDoubleStreamAttentionBlock, WanSingleStreamAttentionBlock},
        )
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy=wan_auto_wrap_policy,
    )

    accelerator = Accelerator(
        log_with="tensorboard",
        project_config=project_config,
        gradient_accumulation_steps=grad_accum_steps,
        fsdp_plugin=fsdp_plugin,
    )

    # 设置随机种子
    set_seed(cfg.get("seed", 42))

    if accelerator.is_main_process:
        os.makedirs(cfg["out_dir"], exist_ok=True)
        print(f"[Accelerator] Mixed Precision: {accelerator.mixed_precision}")
        print(f"[Accelerator] Process: {accelerator.process_index} / {accelerator.num_processes}")
        print(f"[Accelerator] Gradient Accumulation Steps: {accelerator.gradient_accumulation_steps}")

    device = accelerator.device
    local_rank = accelerator.local_process_index

    # -------------------------
    # 模型初始化
    # -------------------------
    module_path, class_name = cfg["pipeline"].rsplit(".", 1)
    PipelineClass = getattr(importlib.import_module(module_path), class_name)
    pipe = PipelineClass.create(
        model_id=cfg["model_id"],
        use_bf16=cfg["use_bf16"],
        audio_latent_ch=cfg["audio_latent_ch"],
        video_latent_ch=cfg["video_latent_ch"],
        lambda_ddpm=cfg["lambda_ddpm"],
        cfg=cfg,
        device=device,
    )

    # FSDP 要求入参模型必须 Dtype 统一
    pipe.model.to(torch.float16)
    pipe.switch_training_mode()
    
    # -------------------------
    # Attention-only冻结（新增）
    # -------------------------
    if cfg.get("attention_only", False):
        print("🔧 应用Attention-only冻结，只训练self-attention和cross-attention参数...")

        attention_patterns = ["attn", "attention", "q", "k", "v", "proj", "out", "cross_modulation"]
        if cfg.get("audio_full_sft", False):
            attention_patterns.extend(["audio_block", "audio_model"])
        elif cfg.get("video_full_sft", False):
            attention_patterns.extend(["vid_block", "video_model"])

        def freeze_attention_only(model):
            total, trainable = 0, 0

            # 1. 冻结全部
            for p in model.parameters():
                p.requires_grad = False
                total += p.numel()

            # 2. 只解冻 attention 相关模块
            for module in model.modules():
                if isinstance(module, WanAttentionBlock):
                    for name, p in module.named_parameters(recurse=True):
                        # 只允许 attention 相关
                        if any(k in name.lower() for k in attention_patterns):
                            p.requires_grad = True
                            trainable += p.numel()

            print(f"[Attention-only]")
            print(f"  Trainable: {trainable:,}")
            print(f"  Frozen:    {total - trainable:,}")
            print(f"  Ratio:     {100 * trainable / total:.2f}%")
        
        freeze_attention_only(pipe.model)
        
        # 打印详细的参数信息
        if accelerator.is_main_process:
            print("\\n📊 详细参数状态:")
            for name, param in pipe.model.named_parameters():
                status = "🟢" if param.requires_grad else "🔴"
                print(f"  {status} {name}: {param.numel():,} params")
        
        freeze_attention_only(pipe.model)
        
        # 打印详细的参数信息
        if accelerator.is_main_process:
            print("\\n📊 详细参数状态:")
            for name, param in pipe.model.named_parameters():
                status = "🟢" if param.requires_grad else "🔴"
                print(f"  {status} {name}: {param.numel():,} params")

    # -------------------------
    # 初始化 EMA
    # -------------------------
    use_ema = cfg.get("use_ema", False)  # 默认为 False，需要在 yaml 里显式开启
    ema = None
    if use_ema:
        ema_decay = cfg.get("ema_decay", 0.9999)
        ema = EMA(pipe.model, decay=ema_decay)
        if accelerator.is_main_process:
            print(f"[EMA] Initialized with decay={ema_decay}")

    # -------------------------
    # 加载 Resume 状态 (Model 权重)
    # -------------------------
    resume_data_state = None
    global_step = 0
    if args.resume:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            ckpt_fallback = os.path.splitext(resume_path)[0] + ".ckpt"
            if os.path.exists(ckpt_fallback):
                if accelerator.is_main_process:
                    print(f"[resume] {resume_path} not found, falling back to {ckpt_fallback}")
                resume_path = ckpt_fallback
            else:
                raise FileNotFoundError(f"Checkpoint not found: {resume_path} (also tried {ckpt_fallback})")

        if resume_path.endswith(".safetensors"):
            from safetensors.torch import load_file as _sf_load
            # safetensors 只含模型权重，无 data_state / global_step
            ckpt = {"state_dict": _sf_load(resume_path, device="cpu")}
            if not args.load_ckpt_only and accelerator.is_main_process:
                print(f"[resume] safetensors format: data_state and global_step not available, starting from step 0")
        else:
            ckpt = torch.load(resume_path, map_location="cpu")

        if not args.load_ckpt_only and "data_state" in ckpt:
            resume_data_state = ckpt.pop("data_state").to(torch.int64).tolist()
            # 支持断点续训时机器数量改变，每个源均使用跑的最快的sharding
            # TODO：注意此处有bug，如果有源正处于cycling，数据会重复训练一部分
            num_shards = max(1, cfg.get("num_workers", 4)) * accelerator.num_processes
            if len(resume_data_state) != num_shards:
                old_data_state = np.array(resume_data_state)
                new_data_state = old_data_state.max(axis=0, keepdims=True).repeat(num_shards)
                resume_data_state = new_data_state.tolist()
            global_step = ckpt.pop("global_step", 0)
        missing, unexpected = pipe.model.load_state_dict(ckpt["state_dict"], strict=False)

        # 加载 EMA 状态
        if ema is not None and "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            if accelerator.is_main_process:
                print(f"[resume] EMA state loaded.")
        elif ema is not None:
            ema.load_state_dict(ckpt["state_dict"])
            if accelerator.is_main_process:
                print(f"[resume] not EMA state found. load EMA state from NO-EMA state.")

        if accelerator.is_main_process:
            print(f"[resume] loaded: {resume_path}\n missing={missing}\n unexpected={unexpected}")
    
    
    # num_nodes = int(os.environ.get("NNODES", os.environ.get("WORLD_SIZE_IN_NODES", "1")))
    # node_rank = int(os.environ.get("NODE_RANK", "0"))
    dist_info = DistInfo(
        world_rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        # node_rank=node_rank,
        # num_nodes=num_nodes,
    )
    if accelerator.is_main_process:
        print(f"world_rank={accelerator.process_index}, world_size={accelerator.num_processes}")
    # -------------------------
    # 数据集
    # -------------------------
    if "data_filelist" in cfg["data"]:
        data = []
        with open(cfg["data"]["data_filelist"], "r") as f:
            for item in f.read().split('\n'):
                if not item: continue
                if len(item.split('\t')) == 3:
                    idx, name, path = item.split('\t')
                    data.append([name, path])
                elif len(item.split('\t')) == 2:
                    idx, path = item.split('\t')
                    data.append([path])
                else:
                    assert False

        src_id2ratios = {}
        with open(cfg["data"]["data_weights"], "r") as f:
            for item in f.read().split('\n'):
                if not item: continue
                if len(item.split('\t')) == 3:
                    key, value, modal = item.split('\t')
                else:
                    key, value = item.split('\t')
                    modal = 'text_to_audio'
                src_id2ratios[key] = [float(value), modal]
    else:
        data = cfg["data"]["jsonl"]
        src_id2ratios = None
    
    audio_vae_server, image_vae_server, video_vae_server = None, None, None
    if cfg["data"].get("use_local_vae", False):
        audio_vae_server = pipe.audio_vae
        video_vae_server = pipe.video_vae
        image_vae_server = pipe.image_vae
    else:
        assert cfg["data"].get("use_precomputed", False), "Must use precomputed vae without online server"
        from nava_src.vae.precomputed_vae_adapter import PrecomputedVideoVAE, PrecomputedAudioVAE
        latent_dir = cfg["data"]["latent_dir"]
        audio_vae_server = PrecomputedAudioVAE(latent_dir=latent_dir)
        video_vae_server = PrecomputedVideoVAE(latent_dir=latent_dir)


    if cfg["data"].get("modal_prob", None) is None:
        cfg["data"]["modal_prob"] = {
            "text_to_audio": 1.0,
            "text_to_video": 0.0
        }

    ds = AudioVideoDataset(
        batch_size=cfg['batch_size'],
        queue_size=cfg["data"].get("queue_size", 5),
        io_workers=cfg["data"].get("io_workers", 16),
        jsonl_or_src_list=data,
        src_id2ratios=src_id2ratios,
        modal_prob=cfg["data"]["modal_prob"],
        dist_info=dist_info,
        num_shards=max(1, cfg.get("num_workers", 4)) * accelerator.num_processes,
        audio_vae_server=audio_vae_server,
        image_vae_server=image_vae_server,
        video_vae_server=video_vae_server,
        workers2history_dict=resume_data_state,
        use_aspect_ratio_buckets=cfg["data"].get("use_aspect_ratio_buckets", False),
        use_length_buckets=cfg["data"].get("use_length_buckets", False),
        num_length_buckets=cfg["data"].get("num_length_buckets", 10),
        enable_ddp_bucket_sync=cfg["data"].get("enable_ddp_bucket_sync", False),
        is_packing=cfg["data"].get("is_packing", False),
        audio_tokens_per_sec=cfg["data"].get("audio_tokens_per_sec", 31.25),
        min_audio_duration=cfg["data"].get("min_audio_duration", 0.5),
        max_audio_duration=cfg["data"].get("max_audio_duration", 10.0),
        tgt_audio_duration=cfg["data"].get("tgt_audio_duration", -1),
        video_min_frames=cfg["data"].get("video_min_frames", 17),
        video_max_frames=cfg["data"].get("video_max_frames", 129),
        video_tgt_frames=cfg["data"].get("video_tgt_frames", 65),
        video_fps=cfg["data"].get("video_fps", 16),
        add_spk_emb=cfg["data"].get("add_spk_emb", False),
        spk_emb_prob=cfg["data"].get("spk_emb_prob", 0.9),
        use_speech_special_token=cfg["data"].get("use_speech_special_token", False),
        data_file_divisor=cfg["data"].get("data_file_divisor", 1),
    )

    # DDP 分桶同步需要 __iter__ 在主进程运行（才能调 dist.broadcast），强制 num_workers=0
    _ddp_bucket_sync = cfg["data"].get("enable_ddp_bucket_sync", False)
    _num_workers = 0 if _ddp_bucket_sync else cfg.get("num_workers", 4)
    if _ddp_bucket_sync and accelerator.is_main_process:
        print("[DATA] enable_ddp_bucket_sync=True, forcing num_workers=0 (IO handled by internal threads)")

    dl = DataLoader(
        ds,
        batch_size=1, # NOTE: 由于使用了分桶策略，此处为 1
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=False,
        collate_fn=collate_fn_batch,
        drop_last=False,
        persistent_workers=(_num_workers > 0),
        prefetch_factor=4 if _num_workers > 0 else None,
    )

    global_state = torch.zeros(
        max(8, cfg.get("num_workers", 4) * accelerator.num_processes), len(ds.src_ids) * 2
        ).to(device)


    # -------------------------
    # 优化器 & 调度器
    # -------------------------
    # 统计参数量
    trainable_numel = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
    total_numel = sum(p.numel() for p in pipe.model.parameters())
    
    # 转换为十亿 (B) 或百万 (M) 单位
    def format_params(num):
        if num >= 1e9:
            return f"{num / 1e9:.2f} B"
        else:
            return f"{num / 1e6:.2f} M"

    if accelerator.is_main_process:
        print("-" * 50)
        print(f"[*] Trainable Parameters: {format_params(trainable_numel)}")
        print(f"[*] Total Parameters:     {format_params(total_numel)}")
        print(f"[*] Trainable Ratio:      {trainable_numel / total_numel:.2%}")
        print("-" * 50)

    # 原有逻辑
    trainable_params = [p for p in pipe.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = WarmupCosineAnnealingLR(
        opt, warmup_steps=cfg["warmup_steps"], max_steps=cfg["max_steps"], eta_min=cfg["lr"] * 0.05
    )

    # -------------------------
    # Accelerate Prepare
    # -------------------------
    # 让 Accelerate 接管 Model, Optimizer, DataLoader, Scheduler
    # 这步会自动处理 DeepSpeed 初始化、DDP 包装、Device 放置
    # -------------------------
    pipe.model, opt, sched = accelerator.prepare(
        pipe.model, opt, sched
    )

    # -------------------------
    # 日志器初始化
    # -------------------------
    if accelerator.is_main_process:
        tb_dir = os.path.join(cfg["out_dir"], "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[Info] TensorBoard logging to: {tb_dir}")
        writer.add_text("config", str(cfg), 0)
    else:
        writer = None

    # -------------------------
    # 训练循环
    # -------------------------
    log_every   = int(cfg.get("log_every", 20))
    case_every   = int(cfg.get("log_cases_every", 0))
    save_every  = int(cfg.get("save_every", 1000))
    max_norm    = float(cfg.get("max_grad_norm", 1.0))
    max_steps   = int(cfg["max_steps"])
    eval_every  = int(cfg.get("eval_every", 50000))
    eval_online = bool(cfg.get("eval_online", False))

    pipe.model.train()

    micro_step = 0
    last_log_step = global_step
    tokens_seen = 0
    last_tokens_seen = 0
    last_audio_batch = None
    last_image_batch = None
    last_video_batch = None
    t0 = time.time()

    acc_loss = 0.0
    acc_lm = 0.0
    acc_ddpm = 0.0
    acc_ddpm_audio = 0.0
    acc_ddpm_image = 0.0
    acc_ddpm_video = 0.0
    acc_ddpm_ori = 0.0
    acc_ddpm_ori_audio = 0.0
    acc_ddpm_ori_image = 0.0
    acc_ddpm_ori_video = 0.0
    acc_dispersive = 0.0
    acc_repa = 0.0
    if cfg.get("log_timestep_loss", False):
        # 将 1000 steps 分为 10 个区间记录 loss
        timestep_interval = cfg.get("timestep_interval", 10)
        acc_timestep_loss = [[0.0] for _ in range(timestep_interval)]
    
    if cfg.get("log_aspect_ratio_loss", False):
        aspect_ratio_loss = {}

    # 无限循环 epoch，依靠 max_steps 退出
    train_perf = TrainPerfLogger(print_every=log_every, rank=accelerator.process_index, enabled=cfg.get("enable_perf_log", False))
    data_load_t0 = time.time()
    for epoch in range(10**9):
            
        for batch in dl:
            train_perf.record("data_load", time.time() - data_load_t0)
            micro_step += 1
            if isinstance(batch, list) and len(batch) == 1: 
                batch = batch[0]
            else:
                raise ValueError(f"[batch] batch should be a dict or a list of dict, but got {len(batch)}")

            if batch.get("image_latents", None):
                last_image_batch = batch
            
            if batch.get("audio_latents", None):
                last_audio_batch = batch
            
            if batch.get("video_latents", None):
                last_video_batch = batch
                
            with accelerator.accumulate(pipe.model):
                if accelerator.sync_gradients:
                    should_log_case = case_every > 0 and ((global_step + 1) % case_every == 0)
                    if should_log_case:
                        sample_steps = int(cfg.get("log_sample_steps", 25))
                        audio_guidance_scale = float(cfg.get("audio_guidance_scale", 4.0))
                        image_guidance_scale = float(cfg.get("image_guidance_scale", 5.0))
                        video_guidance_scale = float(cfg.get("video_guidance_scale", 5.0))

                        # EMA Context
                        context = nullcontext()
                        if ema is not None:
                            # 只有主进程打印 log，避免刷屏
                            if accelerator.is_main_process:
                                print("[info] logging images with EMA weights...")
                            context = ema.average_parameters()
                        
                        # FSDP 上下文放在最外层，所有 Rank 必须同时进入！
                        # rank0_only=True: 建议开启，这样只有 Rank0 占用完整显存，其他卡只负责发数据
                        torch.cuda.empty_cache()
                        with FSDP.summon_full_params(pipe.model, writeback=False, rank0_only=True):
                            with context:
                                # 真正的画图逻辑（包括写入 TensorBoard）只在 Rank 0 执行
                                if accelerator.is_main_process: # TODO @hujiahao03 适配此处 packing 推理
                                    # 此时 pipe.model 已经是完整的了
                                    pipe.model.to(torch.bfloat16)

                                    eval_audio_batch = last_audio_batch if last_audio_batch is not None else batch
                                    log_audios_triplet(
                                        writer=writer, pipe=pipe, batch=eval_audio_batch, device=device,
                                        step=global_step, N=8, tag_prefix="audio_samples",
                                        sample_steps=sample_steps, guidance=audio_guidance_scale, cfg=cfg,
                                    )
                                    
                                    eval_image_batch = last_image_batch if last_image_batch is not None else batch
                                    log_images_triplet(
                                        writer=writer, pipe=pipe, batch=eval_image_batch, device=device,
                                        step=global_step, N=8, tag_prefix="image_samples",
                                        sample_steps=sample_steps, guidance=image_guidance_scale, cfg=cfg,
                                    )

                                    eval_video_batch = last_video_batch if last_video_batch is not None else batch
                                    log_videos_triplet(
                                        writer=writer, pipe=pipe, batch=eval_video_batch, device=device,
                                        step=global_step, N=4, tag_prefix="video_samples",
                                        sample_steps=sample_steps, guidance=video_guidance_scale, cfg=cfg,
                                    )

                        torch.cuda.empty_cache()
                        pipe.switch_training_mode()
                                
                    accelerator.wait_for_everyone()

                torch.cuda.synchronize()
                _t_fwd_start = time.time()
                loss, logs = pipe.forward(batch, global_step=global_step)
                torch.cuda.synchronize()
                _t_fwd_end = time.time()
                train_perf.record("forward", _t_fwd_end - _t_fwd_start)

                _t_bwd_start = time.time()
                accelerator.backward(loss)
                torch.cuda.synchronize()
                train_perf.record("backward", time.time() - _t_bwd_start)
                # Gradient Clipping
                if accelerator.sync_gradients:
                    _t_clip_start = time.time()
                    accelerator.clip_grad_norm_(pipe.model.parameters(), max_norm)
                    torch.cuda.synchronize()
                    train_perf.record("grad_sync_clip", time.time() - _t_clip_start)

                # Optimizer Step
                _t_opt_start = time.time()
                opt.step()
                torch.cuda.synchronize()
                train_perf.record("opt_step", time.time() - _t_opt_start)
                #sched.step()
                opt.zero_grad()
            
                # EMA Update
                if accelerator.sync_gradients and ema is not None:
                    ema.update()
                # -----------------------------
                # 日志与保存 (使用平均值)
                # -----------------------------
                # 【Change 3】 计算平均值
                # TODO @hujiahao03, 此处 loss 记录非 global loss，添加聚合 global loss
                # TODO @hujiahao03 log 均需要适配 transfusion + Z-Image
                acc_loss += loss.item()
                if logs.get("lm") is not None:
                    acc_lm += logs["lm"].item()
                if logs.get("ddpm") is not None:
                    acc_ddpm += logs["ddpm"].item()
                if logs.get("ddpm_audio") is not None:
                    acc_ddpm_audio += logs["ddpm_audio"].item()
                if logs.get("ddpm_image") is not None:
                    acc_ddpm_image += logs["ddpm_image"].item()
                if logs.get("ddpm_vid") is not None:
                    acc_ddpm_video += logs["ddpm_vid"].item()
                if logs.get("ddpm_noreweight") is not None:
                    acc_ddpm_ori += logs["ddpm_noreweight"].item()
                if logs.get("ddpm_audio_noreweight") is not None:
                    acc_ddpm_ori_audio += logs["ddpm_audio_noreweight"].item()
                if logs.get("ddpm_image_noreweight") is not None:
                    acc_ddpm_ori_image += logs["ddpm_image_noreweight"].item()
                if logs.get("ddpm_vid_noreweight") is not None:
                    acc_ddpm_ori_video += logs["ddpm_vid_noreweight"].item()
                if logs.get("dispersive") is not None:
                    acc_dispersive += logs["dispersive"].item()
                if logs.get("repa") is not None:
                    acc_repa += logs["repa"].item()
                if logs.get("ddpm") is not None and logs.get("timesteps") is not None and cfg.get("log_timestep_loss", False):
                    for i, ts in enumerate(logs["timesteps"]):
                        acc_timestep_loss[int((ts - 1) // (1000 / timestep_interval))].append(logs["timesteps_loss"][i].item()) 
                if logs.get("ddpm") is not None and logs.get("aspect_ratio") is not None and cfg.get("log_aspect_ratio_loss", False) and cfg["data"].get("use_aspect_ratio_buckets", False):
                    if not f"{logs['aspect_ratio']}" in aspect_ratio_loss:
                        aspect_ratio_loss[f"{logs['aspect_ratio']}"] = []
                    aspect_ratio_loss[f"{logs['aspect_ratio']}"].append(logs["ddpm"].item())

                # --- Step Boundary ---
                if accelerator.sync_gradients:
                    global_step += 1
                    metrics = torch.tensor([
                            acc_loss, acc_lm, acc_dispersive, acc_repa,
                            acc_ddpm, acc_ddpm_audio, acc_ddpm_image, acc_ddpm_video,
                            acc_ddpm_ori, acc_ddpm_ori_audio, acc_ddpm_ori_image, acc_ddpm_ori_video,
                        ],
                        device=device
                    )
                    metrics = accelerator.reduce(metrics, reduction="mean")
                    if accelerator.is_main_process and (global_step % log_every == 0):
                        avg_loss, avg_lm, avg_dispersive, avg_repa, \
                        avg_ddpm, avg_ddpm_audio, avg_ddpm_image, avg_ddpm_video, \
                        avg_ddpm_ori, avg_ddpm_ori_audio, avg_ddpm_ori_image, avg_ddpm_ori_video = (
                            metrics / accelerator.gradient_accumulation_steps
                        )

                        t_now = time.time()
                        dt = max(t_now - t0, 1e-9)
                        d_steps = global_step - last_log_step

                        # compute tokens seen
                        if "text_lens" in batch:
                            text_lens = int(sum(batch["text_lens"]))
                            tokens_seen += text_lens
                        if "audio_latents" in batch and batch["audio_latents"] is not None:
                            for feature in batch["audio_latents"]:
                                tokens_seen += feature.shape[0]
                        if "video_latents" in batch and batch["video_latents"] is not None:
                            for feature in batch["video_latents"]:
                                tokens_seen += feature.shape[0] * feature.shape[1] * feature.shape[2]
                        if "image_latents" in batch and batch["image_latents"] is not None:
                            for feature in batch["image_latents"]:
                                tokens_seen += feature.shape[0] * feature.shape[1] * feature.shape[2]

                        d_tokens = tokens_seen - last_tokens_seen
                        it_per_s = d_steps / dt
                        toks_per_s = d_tokens / dt
                        # 重置基准点
                        t0 = t_now
                        last_log_step = global_step
                        last_tokens_seen = tokens_seen
                        print(
                            f"[{global_step:>6}] "
                            f"loss={avg_loss:.4f} "
                            f"lm={avg_lm:.4f} "
                            f"ddpm={avg_ddpm:.4f}  "
                            f"ddpm_audio={avg_ddpm_audio:.4f}  "
                            f"ddpm_image={avg_ddpm_image:.4f}  "
                            f"ddpm_video={avg_ddpm_video:.4f}  "
                            f"ddpm_ori={avg_ddpm_ori:.4f}  "
                            f"ddpm_ori_audio={avg_ddpm_ori_audio:.4f}  "
                            f"ddpm_ori_image={avg_ddpm_ori_image:.4f}  "
                            f"ddpm_ori_video={avg_ddpm_ori_video:.4f}  "
                            f"dispersive={avg_dispersive:.4f}  "
                            f"repa={avg_repa:.4f}  "
                            f"lr={get_lr(opt):.3e}  "
                            f"it/s={it_per_s:.2f} tok/s={toks_per_s/1e3:.2f}k  "
                            f"mem={fmt_mem()}"
                        )

                        if accelerator.is_main_process and writer is not None:
                            # 【Change 4】 TensorBoard 记录平均值
                            writer.add_scalar("train/loss", avg_loss, global_step)
                            writer.add_scalar("train/lm", avg_lm, global_step)
                            writer.add_scalar("train/ddpm", avg_ddpm, global_step)
                            writer.add_scalar("train/ddpm_audio", avg_ddpm_audio, global_step)
                            writer.add_scalar("train/ddpm_image", avg_ddpm_image, global_step)
                            writer.add_scalar("train/ddpm_video", avg_ddpm_video, global_step)
                            writer.add_scalar("train/ddpm_ori", avg_ddpm_ori, global_step)
                            writer.add_scalar("train/ddpm_ori_audio", avg_ddpm_ori_audio, global_step)
                            writer.add_scalar("train/ddpm_ori_image", avg_ddpm_ori_image, global_step)
                            writer.add_scalar("train/ddpm_ori_video", avg_ddpm_ori_video, global_step)
                            writer.add_scalar("train/dispersive", avg_dispersive, global_step)
                            writer.add_scalar("train/repa", avg_repa, global_step)
                            writer.add_scalar("train/lr", get_lr(opt), global_step)
                            if cfg.get("log_timestep_loss", False):
                                for idx in range(len(acc_timestep_loss)):
                                    if len(acc_timestep_loss[idx]) > 1:
                                        writer.add_scalar(
                                            f"train/timestep_loss/{idx*(1000//timestep_interval)}_{(idx+1)*(1000//timestep_interval)}", 
                                            sum(acc_timestep_loss[idx]) / accelerator.gradient_accumulation_steps / len(acc_timestep_loss[idx][1:]), 
                                            global_step
                                        )

                            if cfg.get("log_aspect_ratio_loss", False):
                                for ar, loss in aspect_ratio_loss.items():
                                    writer.add_scalar(
                                        f"train/aspect_ratio_loss_{ar}", 
                                        sum(loss) / accelerator.gradient_accumulation_steps / len(loss), 
                                        global_step
                                    )

                            if "input_ids" in batch:
                                non_pad = (batch["input_ids"] != pipe.tokenizer.pad_token_id).float().mean().item()
                                writer.add_scalar("data/nonpad_ratio", non_pad, global_step)
                        
                    acc_loss, acc_lm, acc_ddpm, acc_ddpm_ori, acc_dispersive, acc_repa = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    acc_ddpm_audio, acc_ddpm_image, acc_ddpm_video = 0.0, 0.0, 0.0
                    acc_ddpm_ori_audio, acc_ddpm_ori_image, acc_ddpm_ori_video = 0.0, 0.0, 0.0
                    if cfg.get("log_timestep_loss", False):
                        acc_timestep_loss = [[0.0] for _ in range(timestep_interval)]
                    if cfg.get("log_aspect_ratio_loss", False):
                        aspect_ratio_loss = {}

                    # -----------------------------
                    # 图像生成日志
                    # -----------------------------
                    # 所有 Rank 都必须知道 "这个 step 需要 summon 参数了"
                    

                    acc_loss, acc_lm, acc_ddpm, acc_dispersive = 0.0, 0.0, 0.0, 0.0
                    if cfg.get("log_timestep_loss", False):
                        acc_timestep_loss = [[0.0] for _ in range(timestep_interval)]
                    if cfg.get("log_aspect_ratio_loss", False):
                        aspect_ratio_loss = {}


                    # -----------------------------
                    # 保存 Checkpoint (修正版)
                    # -----------------------------

                    # 1. 收集 Data State (需要在每个iter都更新)
                    current_data_state = batch['data_state']
                    if isinstance(current_data_state[0], list):
                        current_data_state = current_data_state[0]

                    # 2. 解析数据
                    local_info = current_data_state.to(device) 

                    if local_info.dim() > 1:
                        local_info = local_info[-1]
                    worker_id = int(local_info[0].item())

                    # 3. 更新本地记录
                    state_data = local_info[1:]
                    global_state[worker_id] = state_data.float()

                    if global_step % save_every == 0:
                        # 1. 此时 global_state 里存的是“本GPU”辖区内所有 Worker 的最新状态
                        #    但是其他 GPU 的 Worker 状态在这里是 0
                        #    所以需要 Gather 合并
                        gathered_states = accelerator.gather(global_state.unsqueeze(0))
                        if accelerator.is_main_process:
                            # 合并逻辑：取最大值
                            final_state = gathered_states.sum(dim=0)

                        # 2. 收集 Model State (解决死锁)
                        # accelerator.get_state_dict 会自动处理 FSDP 的参数聚合
                        # 必须在 if is_main_process 之外调用！
                        model_state = accelerator.get_state_dict(pipe.model) 

                        # 3. 只在主进程保存
                        if accelerator.is_main_process:
                            ckpt_path = os.path.join(cfg["out_dir"], f"step{global_step}.ckpt")
                            full_ckpt = {
                                "state_dict": model_state,
                                "data_state": final_state.cpu(),
                                "global_step": global_step
                            }
                            
                            if ema is not None:
                                full_ckpt["ema_state"] = ema.state_dict()

                            torch.save(full_ckpt, ckpt_path)
                            print(f"saved: {ckpt_path}")
                            
                            # 释放内存
                            del full_ckpt
                        del model_state
                        torch.cuda.empty_cache()

                    if eval_online and (global_step % eval_every == 0):
                        torch.cuda.empty_cache()
                        opt.zero_grad(set_to_none=True)
                        # EMA Context
                        context = nullcontext()
                        if ema is not None:
                            # 只有主进程打印 log，避免刷屏
                            if accelerator.is_main_process:
                                print("[info] evaluate images with EMA weights...")
                            context = ema.average_parameters()
                        with FSDP.summon_full_params(pipe.model, writeback=False, rank0_only=False):
                            with context, torch.no_grad():
                                pipe.model.to(torch.bfloat16)
                                pipe.model.eval()
                                try:
                                    evaluate(pipe=pipe, cfg_dict=cfg, accelerator=accelerator, 
                                            step=global_step, writer=writer)
                                except Exception as e:
                                    print("evaluate failed", e)
                                pipe.model.train()
                        torch.cuda.empty_cache()
                        
                    if global_step >= max_steps:
                        accelerator.end_training()
                        return

                    train_perf.step_done()

            data_load_t0 = time.time()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        torch.distributed.destroy_process_group()
