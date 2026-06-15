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

    def rewrite(self, user_input: str) -> tuple:
        """Rewrite prompt. Returns (result, warning) tuple."""
        # Ensure model is on GPU before generating
        self.reload()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        print(f"[Rewriter] Generating (input tokens: {inputs['input_ids'].shape[1]})...")
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

        elapsed = time.time() - t0
        print(f"[Rewriter] Done in {elapsed:.1f}s ({len(new_tokens)} tokens)")

        # Check <S><E> pair count
        input_count = self._count_speech_tags(user_input)
        output_count = self._count_speech_tags(result)
        warning = ""
        if input_count > 0 and output_count != input_count:
            warning = f"⚠️ Speech 标签数量不匹配！输入有 {input_count} 对 <S><E>，输出有 {output_count} 对。请重新点击 Rewrite。"
            print(f"[Rewriter] WARNING: {warning}")

        return result, warning


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
def run_gradio(engine: NAVAEngine, rewriter: PromptRewriter, args):
    import gradio as gr

    def rewrite_fn(user_prompt: str):
        """Rewrite prompt only, triggered by Rewrite button."""
        if not user_prompt.strip():
            return "", ""
        rewritten, warning = rewriter.rewrite(user_prompt)
        print(f"[Gradio] Rewritten prompt:\n{rewritten[:200]}...")
        return rewritten, warning

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

        # Rewrite button: only rewrites, does not generate
        rewrite_btn.click(
            fn=rewrite_fn,
            inputs=[prompt_input],
            outputs=[rewritten_prompt, speech_warning],
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
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=37)
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
    )

    # Barrier to initialize NCCL communicator while all ranks are synchronized.
    # This must happen before rank 0 diverges to load the rewriter / launch Gradio.
    dist.barrier()

    if rank == 0:
        # Rank 0: load rewriter + launch Gradio
        rewriter = PromptRewriter(model_path=args.rewrite_model)
        run_gradio(engine, rewriter, args)

        # When Gradio exits, tell workers to stop
        broadcast_cmd(CMD_EXIT, src=0)
    else:
        # Rank 1-7: worker loop
        worker_loop(engine)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
