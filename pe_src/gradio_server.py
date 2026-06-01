"""
Gradio server for NAVA inference with prompt rewrite.

- rank 0: runs Gradio UI + Qwen3 rewrite + coordinates inference
- rank 1-7: wait for broadcast signals, participate in SP inference

Launch with: torchrun --nproc_per_node=8 gradio_server.py --config ... --ckpt ...
"""

import os
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
import time
import torch
import torch.distributed as dist

from nava_engine import NAVAEngine


# ============================================================
# Inter-rank communication protocol
# ============================================================
CMD_INFER = 1
CMD_EXIT = 0
MAX_PROMPT_LEN = 4096  # max chars for broadcast


def broadcast_string(s: str, src: int = 0):
    """Broadcast a string from src rank to all ranks."""
    if dist.get_rank() == src:
        data = s.encode("utf-8")
        length = torch.tensor([len(data)], dtype=torch.long, device="cuda")
    else:
        length = torch.tensor([0], dtype=torch.long, device="cuda")

    dist.broadcast(length, src=src)
    n = length.item()

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


# ============================================================
# Rewrite model (rank 0 only, GPU + offload)
# ============================================================
class PromptRewriter:
    def __init__(self, model_path: str = "Qwen/Qwen3.5-9B"):
        print(f"[Rewriter] Loading {model_path} with device_map=auto (GPU+offload)...")
        t0 = time.time()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print(f"[Rewriter] Loaded in {time.time() - t0:.1f}s")

        # Import system prompt from rewrite_single
        from rewrite_single import SYSTEM_PROMPT
        self.system_prompt = SYSTEM_PROMPT

    def rewrite(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        elapsed = time.time() - t0
        print(f"[Rewriter] Done in {elapsed:.1f}s ({len(new_tokens)} tokens)")
        return result


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
            # Receive prompt and params
            prompt = broadcast_string("", src=0)
            image_path = broadcast_string("", src=0)
            steps_t = torch.tensor([0], dtype=torch.long, device="cuda")
            dist.broadcast(steps_t, src=0)
            is_i2v_t = torch.tensor([0], dtype=torch.long, device="cuda")
            dist.broadcast(is_i2v_t, src=0)

            # Run inference (result discarded on non-rank-0)
            engine.generate(
                prompt=prompt,
                image_path=image_path if image_path else None,
                steps=steps_t.item(),
                is_i2v=bool(is_i2v_t.item()),
            )


# ============================================================
# Gradio UI (rank 0 only)
# ============================================================
def run_gradio(engine: NAVAEngine, rewriter: PromptRewriter, args):
    import gradio as gr

    def infer_fn(user_prompt: str, image_path: str, spk_wav: str,
                 steps: int, do_rewrite: bool, is_i2v: bool):
        """Main inference function triggered by Gradio."""
        # Step 1: Rewrite
        if do_rewrite and user_prompt.strip():
            rewritten = rewriter.rewrite(user_prompt)
            print(f"[Gradio] Rewritten prompt:\n{rewritten[:200]}...")
        else:
            rewritten = user_prompt

        # Step 2: Broadcast to all ranks
        broadcast_cmd(CMD_INFER, src=0)
        broadcast_string(rewritten, src=0)
        broadcast_string(image_path or "", src=0)

        steps_t = torch.tensor([steps], dtype=torch.long, device="cuda")
        dist.broadcast(steps_t, src=0)
        is_i2v_t = torch.tensor([int(is_i2v)], dtype=torch.long, device="cuda")
        dist.broadcast(is_i2v_t, src=0)

        # Step 3: Run inference on rank 0 (all ranks run in parallel via SP)
        spk_paths = [spk_wav] if spk_wav and os.path.exists(spk_wav) else None
        output_path = engine.generate(
            prompt=rewritten,
            image_path=image_path if image_path else None,
            spk_wav_paths=spk_paths,
            steps=steps,
            is_i2v=is_i2v,
        )

        return output_path, rewritten

    # Build Gradio interface
    with gr.Blocks(title="NAVA Audio-Video Generator") as demo:
        gr.Markdown("# NAVA Audio-Video Generator\nSP=8 inference with prompt rewrite")

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt (短描述即可，会自动 rewrite)",
                    placeholder="例如：一只巨龙在城市上空喷火",
                    lines=3,
                )
                do_rewrite = gr.Checkbox(label="Enable Prompt Rewrite", value=True)
                image_input = gr.Image(label="First Frame (可选，用于 I2V)", type="filepath")
                spk_wav_input = gr.Audio(label="Speaker WAV (可选，用于音色控制)", type="filepath")

                with gr.Row():
                    steps_input = gr.Slider(minimum=10, maximum=50, value=25,
                                            step=5, label="Inference Steps")
                    is_i2v_input = gr.Checkbox(label="I2V Mode", value=False)

                submit_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                video_output = gr.Video(label="Generated Video")
                rewritten_output = gr.Textbox(label="Rewritten Prompt", lines=5)

        submit_btn.click(
            fn=infer_fn,
            inputs=[prompt_input, image_input, spk_wav_input,
                    steps_input, do_rewrite, is_i2v_input],
            outputs=[video_output, rewritten_output],
        )

    demo.queue(max_size=1)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="NAVA config yaml path")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="NAVA checkpoint path")
    parser.add_argument("--rewrite_model", type=str, default="Qwen/Qwen3.5-9B",
                        help="Rewrite model path (Qwen3.5-9B)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link")
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
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
    )

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
