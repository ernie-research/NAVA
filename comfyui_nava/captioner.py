"""
Image captioner engine for the NAVAImageCaptioner ComfyUI node.

Loads Qwen3-VL-4B-Instruct and produces a short visual description suitable
for feeding into NAVA's prompt rewriter as the "scene" half of a T2AV prompt.

The output style is video-oriented: subject + scene + camera framing + mood,
in one sentence, no meta-talk like "this is an image of ...".

Heavy weights are cached in a module-level dict so flipping options in a
workflow doesn't trigger reloads.
"""

import os
import sys
import time
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Reach into NAVA's pe_src for the bundled Qwen3-VL weights.
# Layout: <nava_root>/comfyui_nava/captioner.py  AND  <nava_root>/pe_src/Qwen3-VL-4B-Instruct/
# Use realpath so we follow symlinks when this package is symlinked into
# ComfyUI/custom_nodes.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.realpath(__file__))
_NAVA_ROOT = os.path.dirname(_HERE)
_PE_SRC = os.path.join(_NAVA_ROOT, "pe_src")


# ---------------------------------------------------------------------------
# Fixed caption instruction. Kept simple per user request — Qwen3-VL gets
# this system prompt and is asked to describe the image.
# ---------------------------------------------------------------------------
CAPTION_SYSTEM_PROMPT = (
    "你是一个视频生成提示词助手。用一段流畅的中文描述图片中的场景：人物外貌、动作、服装、背景环境、光线与色调、整体氛围。"
    "不要使用markdown格式、不要分条列举、不要说"这张图"或"这是一张图片"，直接描述画面内容，像在描述一段正在发生的视频场景。"
    "输出一段话，不超过150字。"
)

USER_INSTRUCTION = "请描述这张图片的视频场景。"


# ---------------------------------------------------------------------------
# Model cache: key = (resolved_model_path, use_4bit)
# ---------------------------------------------------------------------------
_CAPTIONER_CACHE: dict = {}


def _resolve_model_path(model_path: str) -> str:
    """Accept absolute, relative-to-cwd, relative-to-NAVA-root, or HF repo id."""
    if not model_path:
        return os.path.join(_PE_SRC, "Qwen3-VL-4B-Instruct")
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    cand1 = os.path.abspath(model_path)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(_NAVA_ROOT, model_path)
    if os.path.exists(cand2):
        return cand2
    cand3 = os.path.join(_PE_SRC, model_path)
    if os.path.exists(cand3):
        return cand3
    return model_path


def _load(model_path: str, use_4bit: bool):
    from transformers import AutoProcessor

    print(f"[NAVA-Captioner] Loading {model_path} ({'4bit' if use_4bit else 'bf16'})")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs = {"trust_remote_code": True, "device_map": "cuda:0"}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    # Qwen3-VL exposes itself as image-text-to-text. Fall back to AutoModelForCausalLM
    # for older transformers that haven't registered the new auto class yet.
    try:
        from transformers import AutoModelForImageTextToText as _Auto
    except ImportError:
        from transformers import AutoModelForCausalLM as _Auto
    model = _Auto.from_pretrained(model_path, **load_kwargs).eval()
    print(f"[NAVA-Captioner] Loaded in {time.time() - t0:.1f}s")
    return model, processor


def get_or_load(model_path: str, use_4bit: bool):
    resolved = _resolve_model_path(model_path)
    key = (os.path.abspath(resolved) if os.path.exists(resolved) else resolved, bool(use_4bit))
    if key not in _CAPTIONER_CACHE:
        _CAPTIONER_CACHE[key] = _load(resolved, use_4bit)
    return _CAPTIONER_CACHE[key]


def _image_tensor_to_pil(image: torch.Tensor):
    """ComfyUI IMAGE [B,H,W,C] float [0,1] → PIL.Image (first frame only)."""
    from PIL import Image
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got shape {tuple(image.shape)}")
    arr = (image[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr)


@torch.no_grad()
def caption(
    image: torch.Tensor,
    model_path: str = "",
    use_4bit: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    seed: int = 42,
) -> str:
    """Generate a caption for one ComfyUI IMAGE tensor."""
    pil = _image_tensor_to_pil(image)
    system_prompt = CAPTION_SYSTEM_PROMPT

    model, processor = get_or_load(model_path, use_4bit)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": USER_INSTRUCTION},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil], return_tensors="pt").to(model.device)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    t0 = time.time()
    do_sample = temperature > 0
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True)
    elapsed = time.time() - t0
    print(f"[NAVA-Captioner] Generated {len(new_tokens)} tokens in {elapsed:.1f}s")
    return raw.strip()


def offload_all_to_cpu():
    for model, _ in _CAPTIONER_CACHE.values():
        model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reload_to_gpu(model_path: str = "", use_4bit: bool = False):
    resolved = _resolve_model_path(model_path)
    key = (os.path.abspath(resolved) if os.path.exists(resolved) else resolved, bool(use_4bit))
    if key in _CAPTIONER_CACHE:
        model, _ = _CAPTIONER_CACHE[key]
        model.to("cuda:0")
