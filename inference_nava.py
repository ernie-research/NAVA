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


# -----------------------------
# Prompt Rewriter (onload/offload)
# -----------------------------
class PromptRewriter:
    """Rewriter that loads to GPU on demand and offloads after use."""

    SYSTEM_PROMPT = """你是一个中文音视频生成 prompt rewriter。你的任务是把用户输入的简短描述、关键词或普通 prompt，改写成一个适合音视频生成模型使用的高质量中文长 prompt。最终只输出改写后的 prompt，不要解释，不要分析，不要输出标题，不要输出 JSON，不要换行，必须是单段中文文本。

你必须保留用户输入中的核心意图，包括主体、动作、速度、情绪、场景、台词和镜头要求。不能把用户指定的动作改成相反含义，不能删除关键主体，不能新增与用户意图冲突的剧情。用户没有明确说明的信息，可以根据画面和常识合理补全，例如背景、光线、镜头、动作细节、环境反馈和音效。

改写后的 prompt 必须具有电影化、具体、连续、可执行的风格。整体结构按以下顺序自然组织：第一，描述视频风格、核心氛围和主体所在场景；第二，描述主体的外观、服装、材质、表情、姿态、位置和整体气质；第三，描述背景环境、远景元素、光线、色调和整体氛围；第四，描述动作过程，必须使用清晰的时间线，包含"视频开始时……随后……随着动作持续……视频结束时……"这类表达；第五，描述镜头语言，包括景别、机位、角度、镜头运动、稳定性、是否切镜，以及镜头重点捕捉的细节；第六，描述对白或无对白；第七，描述音频设计，包括主体动作声、环境声、细节声、空间混响和整体听感。

开头优先使用类似句式："这是一段充满【风格/情绪】与【核心氛围】的视频，画面中【主体】正位于【场景】中……"。如果是写实人物或日常场景，可以使用"这段写实电影风格的视频记录了一个……场景……"。如果是动漫人物，可以使用"画面呈现高质量动漫电影质感……"。如果是运动场景，可以突出阳光、速度感、运动张力和真实临场感。如果是机甲、巨龙、怪兽、赛博人物等场景，可以突出史诗感、压迫感、力量感、未来感或毁灭感。

只要用户提供了台词，必须保留台词内容，不能做任何翻译，必须保留英文原文，必须用 <S> 和 <E> 包裹每句台词，用户给的所有连贯的speech只需要一对<S><E>，不允许在其中插入新的。有多个说话人时，要说明谁先说、谁回应、各自的位置、音色、情绪和声场；如果某个角色不说话，也要明确"全程不说话"。对话类音频要强调清晰近场人声、口型同步、环境底噪、声场定位和混音干净。

如果用户没有明确提供台词，必须写："画面中没有人物对白，也没有任何旁白。" 然后进入纯音效设计。音频设计必须具体，不能只写"有声音"或"有环境音"。纯音效场景要写清楚主体动作声、接触摩擦声、环境声、细节声和空间回响。例如海浪翻卷声、冲浪板切水声、风切声、水花拍打声、发动机轰鸣声、轮胎摩擦声、液压装置声、金属关节摩擦声、火焰喷射声、冰晶碰撞声、低频咆哮声、脚步声、衣料摩擦声、室内混响等。默认不要加入明显背景音乐，除非用户明确要求。结尾必须用类似句式总结："整体听感【听感关键词】，突出【核心体验】。" 或 "整体氛围【氛围关键词】，营造出【目标效果】。"

动作描写必须是视频过程，而不是静态描述。要写清楚主体从什么状态开始，接着如何运动，动作速度如何，动作对环境产生什么影响，最后停留在什么状态。例如，快速动作要体现"迅速、猛烈、强烈、连续、背景快速后掠、浪花炸开、灰尘扬起、装甲联动加快"等细节；慢速动作要体现"缓慢、平稳、克制、柔和、细微调整、节奏舒展、环境变化轻柔"等细节。动作和环境反馈要匹配，例如冲浪要有水花和浪声，机甲要有金属关节和脚步震动，巨龙喷火要有火焰、热浪和火星，吐冰要有冰雾、冰晶和寒风，人物说话要有口型同步和近场人声。

镜头语言要具体。默认使用稳定镜头，不要频繁切镜。根据动作选择合理镜头：高速运动使用低角度侧前方跟拍或稳定跟随；慢速运动使用平稳跟拍并保持固定距离；正面凝视使用中景到中近景、轻微仰视或平视、稳定凝视和轻微推进；喷火、吐冰、大吼使用正面中近景、低角度、锁定嘴部和面部；双人对话使用固定中近景，两人同时入画；日常说话使用近景或中近景，强调口型同步和表情。镜头段落中要使用类似句式："镜头采用稳定的【景别/角度】构图……全程……不切镜、不摇移……细腻捕捉……突出……"。

输出要求：只输出最终改写后的 prompt；必须保留原始speech部分不能忽略 !；必须是中文；必须保留原始speech部分不能忽略；必须是单段；不要换行；不要列表；不要解释；不要加标题；不要输出 JSON；不要使用 markdown；不要出现"根据用户输入""改写如下"等说明性文字。"""

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

    def rewrite(self, text: str) -> str:
        """Rewrite a single prompt. Handles onload/offload automatically."""
        self.onload()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self._device)
        print(f"[Rewriter] Generating (input tokens: {inputs['input_ids'].shape[1]})...")
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=2048,
                temperature=0.3, top_p=0.75, top_k=20,
                do_sample=True, repetition_penalty=1.05,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if "</think>" in result:
            result = result.split("</think>", 1)[-1].strip()
        elapsed = time.time() - t0
        print(f"[Rewriter] Done in {elapsed:.1f}s ({len(new_tokens)} tokens)")
        self.offload()
        return result

# -----------------------------
# 分布式工具函数
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
    parser.add_argument("--gen_turn", type=int, default=2)
    parser.add_argument("--save_vid_latent", action="store_true", help="是否保存视频的latent")
    parser.add_argument("--is_i2v", action="store_true", help="是否开启i2v模式")
    parser.add_argument("--timbre_cfg", action="store_true", help="是否开启音色 CFG 控制（需 spk_embs 非空）")
    parser.add_argument("--timbre_align_guidance_scale", type=float, default=1.0, help="音色 CFG 引导强度")
    parser.add_argument("--use_sp", action="store_true",
                        help="启用 Ulysses 序列并行推理：sp_size 自动取自 WORLD_SIZE。"
                             "所有 rank 处理相同样本，仅 rank0 落盘。")
    parser.add_argument("--rewrite", action="store_true", default=False,
                        help="启用 prompt rewriter（默认关闭）")
    parser.add_argument("--rewrite_model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                        help="Rewriter 模型路径")

    args = parser.parse_args()
    use_rewrite = args.rewrite

    # --- Setup ---
    is_ddp, rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # --- Rewriter (rank 0 only) ---
    rewriter = None
    if use_rewrite and (not is_ddp or rank == 0):
        rewriter = PromptRewriter(model_path=args.rewrite_model, device=f"cuda:{local_rank}")
        print(f"[Rewriter] Enabled. Model: {args.rewrite_model}")
    elif not use_rewrite:
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
    set_seed(cfg.get("seed", 42) + (0 if args.use_sp else rank))

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
        model_id=cfg["model_id"],
        use_bf16=cfg["use_bf16"],
        audio_latent_ch=cfg["audio_latent_ch"],
        video_latent_ch=cfg["video_latent_ch"],
        lambda_ddpm=cfg["lambda_ddpm"],
        cfg=cfg,
        device=device,
    )

    if "video" in modality and "audio" in modality and not cfg.get("use_mmdit_model", False):
        load_fusion_checkpoint(pipe.model, checkpoint_path=args.ckpt, from_meta=True)
    else:
        ckpt = torch.load(args.ckpt, map_location="cpu", mmap=True)
        missing, unexpected = pipe.model.load_state_dict(ckpt['state_dict'], strict=False)
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
                # --- Prompt Rewrite ---
                if rewriter is not None:
                    captions = batch.get("captions", None)
                    if captions is not None:
                        if isinstance(captions, list):
                            batch["captions"] = [rewriter.rewrite(c) for c in captions]
                        elif isinstance(captions, str):
                            batch["captions"] = rewriter.rewrite(captions)

                # Per-sample is_i2v from batch (unified json), fallback to global args.is_i2v
                sample_is_i2v = batch.get("is_i2v", is_i2v)
                if isinstance(sample_is_i2v, torch.Tensor):
                    sample_is_i2v = sample_is_i2v.item()
                elif isinstance(sample_is_i2v, list):
                    sample_is_i2v = sample_is_i2v[0] if sample_is_i2v else is_i2v

                with amp_ctx:
                    gen_vid_out, gen_aud_out = pipe.sample(
                        batch,
                        num_steps=args.steps,
                        audio_guidance_scale=cfg.get("audio_guidance_scale", 4.0),
                        video_guidance_scale=cfg.get("video_guidance_scale", 5.0),
                        align_3d_cfg=cfg.get("align_3d_cfg", False),
                        audio_align_guidance_scale=cfg.get("audio_align_guidance_scale", 4.0),
                        video_align_guidance_scale=cfg.get("video_align_guidance_scale", 5.0),
                        save_vid_latent=save_vid_latent,
                        is_i2v=sample_is_i2v,
                        timbre_cfg=args.timbre_cfg or cfg.get("timbre_cfg", False),
                        timbre_align_guidance_scale=args.timbre_align_guidance_scale if args.timbre_cfg else cfg.get("timbre_align_guidance_scale", 3.0),
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