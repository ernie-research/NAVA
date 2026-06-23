"""
Gradio server for NAVA inference with prompt rewrite.

- rank 0: runs Gradio UI + Qwen3 rewrite + coordinates inference
- rank 1-7: wait for broadcast signals, participate in SP inference

Supports:
  - Text prompt (with optional auto-rewrite)
  - Image input for I2V mode
  - Up to 2 speaker reference WAVs for timbre control

Launch with: torchrun --nproc_per_node=8 gradio_server.py --config ... --ckpt ...
"""

import os
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
import time
import datetime
import torch
import torch.distributed as dist

from nava_engine import NAVAEngine


# ============================================================
# Aspect ratio presets
# ============================================================
ASPECT_RATIO_MAP = {
    "16:9 (1280×704)": (704, 1280),
    "9:16 (704×1280)": (1280, 704),
    "1:1 (960×960)": (960, 960),
}


# ============================================================
# Inter-rank communication protocol
# ============================================================
CMD_INFER = 1
CMD_EXIT = 0


def broadcast_string(s: str, src: int = 0):
    """Broadcast a string from src rank to all ranks."""
    if dist.get_rank() == src:
        data = s.encode("utf-8")
        length = torch.tensor([len(data)], dtype=torch.long, device="cuda")
    else:
        length = torch.tensor([0], dtype=torch.long, device="cuda")

    dist.broadcast(length, src=src)
    n = length.item()

    if n == 0:
        return ""

    if dist.get_rank() == src:
        tensor = torch.tensor(list(data), dtype=torch.uint8, device="cuda")
    else:
        tensor = torch.empty(n, dtype=torch.uint8, device="cuda")

    dist.broadcast(tensor, src=src)

    if dist.get_rank() != src:
        s = bytes(tensor.cpu().tolist()).decode("utf-8")
    return s


def broadcast_cmd(cmd: int, src: int = 0):
    """Broadcast a command integer from src to all ranks."""
    t = torch.tensor([cmd], dtype=torch.long, device="cuda")
    dist.broadcast(t, src=src)
    return t.item()


def broadcast_int(val: int, src: int = 0):
    """Broadcast a single integer."""
    t = torch.tensor([val], dtype=torch.long, device="cuda")
    dist.broadcast(t, src=src)
    return t.item()


def broadcast_float(val: float, src: int = 0):
    """Broadcast a single float."""
    t = torch.tensor([val], dtype=torch.float32, device="cuda")
    dist.broadcast(t, src=src)
    return t.item()


# ============================================================
# Rewrite model (rank 0 only, GPU + offload)
# ============================================================
class PromptRewriter:
    def __init__(self, model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"):
        print(f"[Rewriter] Loading {model_path} to CPU...")
        t0 = time.time()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        # Start on CPU, move to cuda:0 only when rewriting
        self.model.eval()
        self._on_gpu = False
        print(f"[Rewriter] Loaded in {time.time() - t0:.1f}s (on CPU)")

        from rewrite_single import SYSTEM_PROMPT
        self.system_prompt = SYSTEM_PROMPT

        # Reach into pe_src for the same output cleaner used by the vLLM batch
        # path and inference_nava — handles 4 leak cases (with/without <think>
        # markers, in-band thinking dumps, post-rewrite meta drift). gradio_demo
        # is a sibling of pe_src/ in the repo layout.
        import sys
        from pathlib import Path
        _PE_SRC = str(Path(__file__).resolve().parent.parent / "pe_src")
        if _PE_SRC not in sys.path:
            sys.path.insert(0, _PE_SRC)
        from rewrite import extract_rewrite as _extract_rewrite
        self._extract_rewrite = _extract_rewrite

    def offload(self):
        """Move rewriter model to CPU to free GPU memory for inference."""
        if self._on_gpu:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self._on_gpu = False
            print("[Rewriter] Offloaded to CPU")

    def reload(self):
        """Move rewriter model to cuda:0 for rewriting."""
        if not self._on_gpu:
            self.model.to("cuda:0")
            self._on_gpu = True
            print("[Rewriter] Reloaded to cuda:0")

    @staticmethod
    def _count_speech_tags(text: str) -> int:
        """Count number of <S>...<E> pairs in text."""
        import re
        return len(re.findall(r"<S>.*?<E>", text, re.DOTALL))

    def rewrite(self, user_input: str, max_retries: int = 5) -> tuple:
        """Rewrite prompt with automatic retry on <S><E> pair-count mismatch.

        Qwen3 occasionally drops or duplicates speech tags despite the
        SYSTEM_PROMPT spelling out "preserve speech verbatim". Each retry
        re-samples (do_sample=True advances cuda RNG, so attempts diverge).
        On persistent mismatch we return the last attempt with a warning —
        a single bad rewrite shouldn't block the user.
        Returns (result, warning) tuple."""
        self.reload()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_count = self._count_speech_tags(user_input)
        print(f"[Rewriter] target <S><E> pairs: {input_count}")

        last_result = ""
        last_count = -1
        for attempt in range(max_retries):
            print(f"[Rewriter] Generating attempt {attempt+1}/{max_retries} "
                  f"(input tokens: {inputs['input_ids'].shape[1]})...")
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.3,
                    top_p=0.75,
                    top_k=20,
                    do_sample=True,
                    repetition_penalty=1.05,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            result = self._extract_rewrite(raw)
            output_count = self._count_speech_tags(result)
            elapsed = time.time() - t0
            print(f"[Rewriter] Done in {elapsed:.1f}s "
                  f"({len(new_tokens)} tokens, <S><E>={output_count})")
            last_result = result
            last_count = output_count
            # Skip-check path: input has no speech tags → no constraint to satisfy.
            if input_count == 0 or output_count == input_count:
                return result, ""
            print(f"[Rewriter] <S><E> mismatch: got {output_count}, want "
                  f"{input_count} — retrying")

        warning = (f"⚠️ Speech 标签数量不匹配（已自动重试 {max_retries} 次）！"
                   f"输入有 {input_count} 对 <S><E>，输出有 {last_count} 对。请重新点击 Rewrite。")
        print(f"[Rewriter] WARNING: {warning}")
        return last_result, warning


# ============================================================
# Image Captioner (rank 0 only) — Qwen3-VL describes the uploaded image
# so its scene description can be composed with the user's text prompt
# before rewrite. SYSTEM_PROMPT mirrors comfyui_nava/captioner.py and
# inference_nava.ImageCaptioner verbatim.
# ============================================================
class ImageCaptioner:
    SYSTEM_PROMPT = (
        "你是一个视频生成提示词助手。用一段流畅的中文描述图片中的场景：人物外貌、"
        "动作、服装、背景环境、光线与色调、整体氛围。不要使用markdown格式、不要分条列举、"
        "不要说\"这张图\"或\"这是一张图片\"，直接描述画面内容，像在描述一段正在发生的"
        "视频场景。输出一段话，不超过150字。"
    )
    USER_INSTRUCTION = "请描述这张图片的视频场景。"

    def __init__(self, model_path: str):
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
        self._on_gpu = False
        print(f"[Captioner] Loaded in {time.time() - t0:.1f}s (on CPU)")

    def offload(self):
        if self._on_gpu:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self._on_gpu = False
            print("[Captioner] Offloaded to CPU")

    def reload(self):
        if not self._on_gpu:
            self.model.to("cuda:0")
            self._on_gpu = True
            print("[Captioner] Reloaded to cuda:0")

    @torch.no_grad()
    def caption(self, image_path: str) -> str:
        self.reload()
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
        inputs = self.processor(text=[text], images=[pil], return_tensors="pt").to(self.model.device)
        print(f"[Captioner] IN  image: {image_path}")
        t0 = time.time()
        out = self.model.generate(
            **inputs, max_new_tokens=256,
            do_sample=True, temperature=0.3, top_p=0.9,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        elapsed = time.time() - t0
        print(f"[Captioner] Done in {elapsed:.1f}s ({len(new_tokens)} tokens)")
        print(f"[Captioner] OUT ({len(result)} chars): {result}")
        return result


def _compose_t2av_prompt(scene_caption: str, user_prompt: str) -> str:
    """Glue VL scene caption + user prompt for the rewriter.
    Caption first, user prompt last so any <S>...<E> stays at the tail."""
    cap = (scene_caption or "").strip()
    spk = (user_prompt or "").strip()
    if not cap:
        return spk
    if not spk:
        return cap
    return f"{cap} {spk}"


# ============================================================
# Worker loop (rank 1-7)
# ============================================================
def worker_loop(engine: NAVAEngine):
    """Non-rank-0 processes wait for commands and execute inference."""
    rank = dist.get_rank()
    print(f"[Rank {rank}] Entering worker loop, waiting for commands...")

    while True:
        cmd = broadcast_cmd(0, src=0)

        if cmd == CMD_EXIT:
            print(f"[Rank {rank}] Received EXIT command. Shutting down.")
            break
        elif cmd == CMD_INFER:
            # Receive all params from rank 0
            prompt = broadcast_string("", src=0)
            image_path = broadcast_string("", src=0)
            spk_wav_1 = broadcast_string("", src=0)
            spk_wav_2 = broadcast_string("", src=0)
            steps = broadcast_int(0, src=0)
            is_i2v = bool(broadcast_int(0, src=0))
            height = broadcast_int(0, src=0)
            width = broadcast_int(0, src=0)
            frames = broadcast_int(0, src=0)
            video_cfg = broadcast_float(0, src=0)
            audio_cfg = broadcast_float(0, src=0)
            video_align_cfg = broadcast_float(0, src=0)
            audio_align_cfg = broadcast_float(0, src=0)
            align_3d_cfg = bool(broadcast_int(0, src=0))
            timbre_cfg = bool(broadcast_int(0, src=0))
            timbre_align_cfg = broadcast_float(0, src=0)

            # Build spk_wav_paths
            spk_wav_paths = []
            if spk_wav_1:
                spk_wav_paths.append(spk_wav_1)
            if spk_wav_2:
                spk_wav_paths.append(spk_wav_2)

            # Run inference (result discarded on non-rank-0)
            engine.generate(
                prompt=prompt,
                image_path=image_path if image_path else None,
                spk_wav_paths=spk_wav_paths if spk_wav_paths else None,
                steps=steps,
                is_i2v=is_i2v,
                height=height,
                width=width,
                frames=frames,
                video_cfg=video_cfg,
                audio_cfg=audio_cfg,
                video_align_cfg=video_align_cfg,
                audio_align_cfg=audio_align_cfg,
                align_3d_cfg=align_3d_cfg,
                timbre_cfg=timbre_cfg,
                timbre_align_cfg=timbre_align_cfg,
            )


# ============================================================
# Gradio UI (rank 0 only)
# ============================================================
def run_gradio(engine: NAVAEngine, rewriter: PromptRewriter, captioner: "ImageCaptioner", args):
    import gradio as gr

    def rewrite_fn(user_prompt: str, image_file: str):
        """Rewrite prompt; if an image is uploaded, VL-caption it and compose
        the scene description with the user's prompt before rewriting.
        Returns (rewritten_with_extra_id_2, speech_warning, vl_caption).
        """
        if not user_prompt.strip():
            return "", "", ""

        # Strip stale <extra_id_2> markers so the rewriter sees clean speech tags;
        # we'll re-inject after rewrite below.
        cap_in = user_prompt.replace("<extra_id_2>", "")

        scene_caption = ""
        if image_file and os.path.exists(image_file):
            scene_caption = captioner.caption(image_file)
            captioner.offload()  # free VRAM before rewriter onloads
            cap_in = _compose_t2av_prompt(scene_caption, cap_in)
            print(f"[Gradio] composed ({len(cap_in)} chars): {cap_in[:200]}...")

        rewritten, warning = rewriter.rewrite(cap_in)
        # Inject <extra_id_2> after every <S> so the user sees the final form
        # in the textbox. nava_engine._build_batch normalizes idempotently
        # before T5 encoding, so this is safe regardless of further user edits.
        rewritten = rewritten.replace("<S>", "<S><extra_id_2>")
        print(f"[Gradio] Rewritten prompt:\n{rewritten[:200]}...")
        return rewritten, warning, scene_caption

    def infer_fn(user_prompt: str, rewritten_prompt: str, image_file: str,
                 spk_wav_1: str, spk_wav_2: str,
                 steps: int, duration_sec: int, aspect_ratio: str,
                 video_cfg: float, audio_cfg: float,
                 video_align_cfg: float, audio_align_cfg: float,
                 align_3d_cfg: bool, timbre_cfg: bool, timbre_align_cfg: float):
        """Main inference function triggered by Generate button.
        Uses rewritten_prompt if available, otherwise falls back to user_prompt.
        """
        # Convert duration (seconds) to frames: frames = 6 * seconds + 1
        frames = int(duration_sec) * 6 + 1

        # Use rewritten prompt if it exists, otherwise use raw input
        final_prompt = rewritten_prompt.strip() if rewritten_prompt.strip() else user_prompt.strip()

        # Resolve aspect ratio to height/width
        height, width = ASPECT_RATIO_MAP.get(aspect_ratio, (704, 1280))

        # I2V mode is automatically enabled when an image is provided
        is_i2v = bool(image_file)

        # Offload rewriter to free GPU memory
        rewriter.offload()

        # Broadcast to all ranks
        broadcast_cmd(CMD_INFER, src=0)
        broadcast_string(final_prompt, src=0)
        broadcast_string(image_file or "", src=0)
        broadcast_string(spk_wav_1 or "", src=0)
        broadcast_string(spk_wav_2 or "", src=0)
        broadcast_int(steps, src=0)
        broadcast_int(int(is_i2v), src=0)
        broadcast_int(height, src=0)
        broadcast_int(width, src=0)
        broadcast_int(frames, src=0)
        broadcast_float(video_cfg, src=0)
        broadcast_float(audio_cfg, src=0)
        broadcast_float(video_align_cfg, src=0)
        broadcast_float(audio_align_cfg, src=0)
        broadcast_int(int(align_3d_cfg), src=0)
        broadcast_int(int(timbre_cfg), src=0)
        broadcast_float(timbre_align_cfg, src=0)

        # Build spk_wav_paths
        spk_wav_paths = []
        if spk_wav_1 and os.path.exists(spk_wav_1):
            spk_wav_paths.append(spk_wav_1)
        if spk_wav_2 and os.path.exists(spk_wav_2):
            spk_wav_paths.append(spk_wav_2)

        # Run inference on rank 0 (all ranks run in parallel via SP)
        output_path = engine.generate(
            prompt=final_prompt,
            image_path=image_file if image_file else None,
            spk_wav_paths=spk_wav_paths if spk_wav_paths else None,
            steps=steps,
            is_i2v=is_i2v,
            height=height,
            width=width,
            frames=frames,
            video_cfg=video_cfg,
            audio_cfg=audio_cfg,
            video_align_cfg=video_align_cfg,
            audio_align_cfg=audio_align_cfg,
            align_3d_cfg=align_3d_cfg,
            timbre_cfg=timbre_cfg,
            timbre_align_cfg=timbre_align_cfg,
        )

        # Reload rewriter back to GPU
        rewriter.reload()

        return output_path

    # Build Gradio interface
    with gr.Blocks(title="NAVA Audio-Video Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# NAVA Audio-Video Generator\nSP=8 inference with prompt rewrite")

        with gr.Row():
            # ---- Left: Inputs ----
            with gr.Column(scale=2):
                gr.Markdown(
                    "> **⚡ Recommendation:** For optimal generation quality, we strongly recommend using the **Rewrite** function — "
                    "especially if your prompt is in English or relatively brief. "
                    "NAVA is primarily trained on high-quality Chinese dense captions; "
                    "the rewriter will transform your input into the format that best activates the model's full potential."
                )

                prompt_input = gr.Textbox(
                    label="Prompt (原始输入)",
                    placeholder="输入短描述或详细 prompt\n例如：一只巨龙在城市上空喷火",
                    lines=4,
                )

                rewrite_btn = gr.Button("Rewrite Prompt", variant="secondary")

                vl_caption_box = gr.Textbox(
                    label="VL Caption (上传图片时自动生成；纯文本时为空)",
                    lines=3,
                    interactive=False,
                    visible=True,
                )

                rewritten_prompt = gr.Textbox(
                    label="Rewritten Prompt (点击 Rewrite 按钮生成，不点则使用原始输入)",
                    lines=8,
                    interactive=True,
                )

                speech_warning = gr.Textbox(
                    label="Speech 检查",
                    interactive=False,
                    visible=True,
                )

                gr.Markdown("### Image (可选，上传后自动启用 I2V 模式)")
                image_input = gr.Image(
                    label="First Frame Image",
                    type="filepath",
                )

                gr.Markdown("### Speaker Reference (可选，最多2个)")
                with gr.Row():
                    spk_wav_1_input = gr.Audio(
                        label="Speaker 1 WAV",
                        type="filepath",
                    )
                    spk_wav_2_input = gr.Audio(
                        label="Speaker 2 WAV",
                        type="filepath",
                    )

                steps_input = gr.Slider(
                    minimum=10, maximum=100, value=50,
                    step=5, label="Inference Steps"
                )

                duration_input = gr.Slider(
                    minimum=2, maximum=10, value=6,
                    step=1, label="Duration (seconds) — 6s = 37 frames"
                )

                aspect_ratio_input = gr.Dropdown(
                    choices=list(ASPECT_RATIO_MAP.keys()),
                    value="16:9 (1280×704)",
                    label="Aspect Ratio",
                )

                gr.Markdown("### CFG Parameters")
                with gr.Row():
                    video_cfg_input = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.0, step=0.5, label="Video CFG")
                    audio_cfg_input = gr.Slider(
                        minimum=1.0, maximum=10.0, value=2.0, step=0.5, label="Audio CFG")
                with gr.Row():
                    video_align_cfg_input = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.0, step=0.5, label="Video Align CFG")
                    audio_align_cfg_input = gr.Slider(
                        minimum=1.0, maximum=10.0, value=2.0, step=0.5, label="Audio Align CFG")
                with gr.Row():
                    align_3d_cfg_input = gr.Checkbox(value=True, label="Align 3D CFG")
                    timbre_cfg_input = gr.Checkbox(value=True, label="Timbre CFG")
                timbre_align_cfg_input = gr.Slider(
                    minimum=1.0, maximum=10.0, value=3.0, step=0.5, label="Timbre Align CFG")

                submit_btn = gr.Button("Generate", variant="primary", size="lg")

            # ---- Right: Outputs ----
            with gr.Column(scale=2):
                video_output = gr.Video(label="Generated Video (with Audio)")

        # Duration slider: update label to show frames.
        # IMPORTANT: gr.update only carries the keys you pass, so we must include
        # minimum/maximum/step here — otherwise they get reset to None and the
        # next submit fails preprocess with `5 < None` TypeError.
        duration_input.change(
            fn=lambda s: gr.update(
                label=f"Duration (seconds) — {int(s)}s = {int(s)*6+1} frames",
                minimum=2, maximum=10, step=1,
            ),
            inputs=[duration_input],
            outputs=[duration_input],
        )

        # Auto-pick Aspect Ratio when the user uploads an image. Avoids the
        # default-landscape-squashing-portrait failure mode; user can still
        # change the dropdown afterwards to manually override.
        def autodetect_aspect_ratio(image_path: str):
            if not image_path or not os.path.exists(image_path):
                return gr.update()
            from PIL import Image
            try:
                w, h = Image.open(image_path).size
            except Exception as e:
                print(f"[Gradio] aspect autodetect: failed to read {image_path}: {e}")
                return gr.update()
            if w > h * 1.2:
                picked = "16:9 (1280×704)"
            elif h > w * 1.2:
                picked = "9:16 (704×1280)"
            else:
                picked = "1:1 (960×960)"
            print(f"[Gradio] aspect autodetect: image {w}x{h} → {picked}")
            return picked

        image_input.change(
            fn=autodetect_aspect_ratio,
            inputs=[image_input],
            outputs=[aspect_ratio_input],
        )

        # Rewrite button: only rewrites, does not generate
        rewrite_btn.click(
            fn=rewrite_fn,
            inputs=[prompt_input, image_input],
            outputs=[rewritten_prompt, speech_warning, vl_caption_box],
        )

        # Generate button: uses rewritten prompt if available
        submit_btn.click(
            fn=infer_fn,
            inputs=[prompt_input, rewritten_prompt, image_input,
                    spk_wav_1_input, spk_wav_2_input,
                    steps_input, duration_input, aspect_ratio_input,
                    video_cfg_input, audio_cfg_input,
                    video_align_cfg_input, audio_align_cfg_input,
                    align_3d_cfg_input, timbre_cfg_input, timbre_align_cfg_input],
            outputs=[video_output],
        )

    # Single-GPU NAVA inference; one job at a time but allow a deeper queue so
    # multiple users can submit without WebSocket drops or 'queue full' errors
    # during long (10-20 min) inference runs.
    demo.queue(max_size=32, default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        max_threads=40,
    )


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="NAVA Gradio Demo with SP inference")
    parser.add_argument("--config", type=str, default="",
                        help="NAVA config yaml path")
    parser.add_argument("--ckpt", type=str, default="",
                        help="NAVA checkpoint path")
    parser.add_argument("--rewrite_model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                        help="Rewrite model path")
    parser.add_argument("--vl_model", type=str, default="pe_src/Qwen3-VL-4B-Instruct",
                        help="VL caption 模型路径；上传图片时 Rewrite 会先 caption + compose 再 rewrite。"
                             "支持本地相对/绝对路径或 HuggingFace repo id。")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=37)
    # ---- inference acceleration knobs (mirror inference_nava.py) ----
    parser.add_argument("--weight_dtype", type=str, default="auto",
                        choices=["auto", "bf16", "fp8_e4m3fn"],
                        help="auto: detect from ckpt; fp8_e4m3fn: force fp8 patch; bf16: skip fp8")
    parser.add_argument("--t5_offload", action="store_true",
                        help="T5 文本编码完成后移回 CPU，释放显存供 DiT 使用")
    parser.add_argument("--group_offload", action="store_true",
                        help="DiT backbone 逐组 block CPU↔GPU offload（去噪期间节省显存）")
    parser.add_argument("--offload_group_size", type=int, default=1,
                        help="每次转移的 transformer block 数量（默认 1，越小越省显存但越慢）")
    parser.add_argument("--vae_tiling", action="store_true",
                        help="VAE decode 分块解码以降低峰值显存")
    parser.add_argument("--vae_tile_size", type=int, nargs=2, default=[22, 40],
                        help="VAE tile size (H W) in latent space")
    parser.add_argument("--vae_tile_stride", type=int, nargs=2, default=[14, 26],
                        help="VAE tile stride (H W) in latent space")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: skip all model loading, only launch Gradio UI")
    args = parser.parse_args()

    # ---- Debug mode: no models, no distributed, just UI ----
    if args.debug:
        import gradio as gr

        def dummy_rewrite(user_prompt):
            return f"[DEBUG] Rewritten: {user_prompt}"

        def dummy_infer(user_prompt, rewritten_prompt, image_file,
                        spk_wav_1, spk_wav_2, steps, duration_sec, aspect_ratio):
            final = rewritten_prompt.strip() if rewritten_prompt.strip() else user_prompt
            height, width = ASPECT_RATIO_MAP.get(aspect_ratio, (704, 1280))
            frames = int(duration_sec) * 6 + 1
            is_i2v = bool(image_file)
            print(f"[DEBUG] Would generate with prompt: {final[:100]}...")
            print(f"[DEBUG] image={image_file}, spk1={spk_wav_1}, spk2={spk_wav_2}")
            print(f"[DEBUG] steps={steps}, frames={frames}, is_i2v={is_i2v}, {width}x{height}")
            return None

        with gr.Blocks(title="NAVA Debug") as demo:
            gr.Markdown("# NAVA Audio-Video Generator (DEBUG MODE)\nNo models loaded, UI only")

            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(label="Prompt (原始输入)", lines=4)
                    rewrite_btn = gr.Button("Rewrite Prompt", variant="secondary")
                    rewritten_prompt = gr.Textbox(
                        label="Rewritten Prompt", lines=8, interactive=True)

                    gr.Markdown("### Image (可选，上传后自动启用 I2V 模式)")
                    image_input = gr.Image(label="First Frame Image", type="filepath")

                    gr.Markdown("### Speaker Reference (可选，最多2个)")
                    with gr.Row():
                        spk_wav_1_input = gr.Audio(label="Speaker 1 WAV", type="filepath")
                        spk_wav_2_input = gr.Audio(label="Speaker 2 WAV", type="filepath")

                    steps_input = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="Steps")
                    duration_input = gr.Slider(
                        minimum=2, maximum=10, value=6,
                        step=1, label="Duration (seconds) — 6s = 37 frames"
                    )
                    aspect_ratio_input = gr.Dropdown(
                        choices=list(ASPECT_RATIO_MAP.keys()),
                        value="16:9 (1280×704)",
                        label="Aspect Ratio",
                    )
                    submit_btn = gr.Button("Generate", variant="primary", size="lg")

                with gr.Column(scale=2):
                    video_output = gr.Video(label="Generated Video")

            rewrite_btn.click(fn=dummy_rewrite, inputs=[prompt_input], outputs=[rewritten_prompt])
            submit_btn.click(
                fn=dummy_infer,
                inputs=[prompt_input, rewritten_prompt, image_input,
                        spk_wav_1_input, spk_wav_2_input, steps_input,
                        duration_input, aspect_ratio_input],
                outputs=[video_output],
            )

        demo.queue(max_size=32, default_concurrency_limit=1)
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            max_threads=40,
        )
        return

    # ---- Normal mode: full model loading + distributed ----
    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(hours=24),
    )
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] Initialized. device={device}, world_size={world_size}")

    # Init NAVA engine (all ranks)
    engine = NAVAEngine(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device=device,
        rank=rank,
        world_size=world_size,
        use_sp=True,
        height=args.height,
        width=args.width,
        frames=args.frames,
        weight_dtype=args.weight_dtype,
        t5_offload=args.t5_offload,
        group_offload=args.group_offload,
        offload_group_size=args.offload_group_size,
        vae_tiling=args.vae_tiling,
        vae_tile_size=tuple(args.vae_tile_size),
        vae_tile_stride=tuple(args.vae_tile_stride),
    )

    # Barrier to initialize NCCL communicator while all ranks are synchronized.
    # This must happen before rank 0 diverges to load the rewriter / launch Gradio.
    dist.barrier()

    if rank == 0:
        # Rank 0: load rewriter + VL captioner + launch Gradio
        rewriter = PromptRewriter(model_path=args.rewrite_model)
        captioner = ImageCaptioner(model_path=args.vl_model)
        run_gradio(engine, rewriter, captioner, args)

        # When Gradio exits, tell workers to stop
        broadcast_cmd(CMD_EXIT, src=0)
    else:
        # Rank 1-7: worker loop
        worker_loop(engine)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
