"""
Single prompt rewrite using Qwen3-4B/8B locally.

Usage:
    python rewrite_single.py "你的短prompt"
    python rewrite_single.py --input prompt.txt --output result.txt
    python rewrite_single.py "你的短prompt" --model Qwen/Qwen3-8B
    python rewrite_single.py "你的短prompt" --4bit    # 节省显存
"""

import argparse
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (Path(__file__).resolve().parent.parent / "pe_src" / "prompts" / "rewrite_template.txt").read_text(encoding="utf-8").rstrip()


def load_model(model_path: str, use_4bit: bool = False):
    """Load model and tokenizer."""
    print(f"[INFO] Loading model: {model_path} ({'4bit' if use_4bit else 'fp16/bf16'})")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    print(f"[INFO] Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def rewrite(model, tokenizer, user_input: str, max_new_tokens: int = 4096) -> str:
    """Run single rewrite inference."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"[INFO] Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"[INFO] Generating...")
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.75,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.05,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)

    elapsed = time.time() - t0
    n_tokens = len(new_tokens)
    print(f"[INFO] Generated {n_tokens} tokens in {elapsed:.1f}s ({n_tokens/elapsed:.1f} tokens/s)")

    return result.strip()


def main():
    parser = argparse.ArgumentParser(description="Rewrite prompt using Qwen3")
    parser.add_argument("prompt", nargs="?", default=None, help="Input prompt text")
    parser.add_argument("--input", "-i", default=None, help="Read prompt from file")
    parser.add_argument("--output", "-o", default=None, help="Write result to file")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3.5-9B",
                        help="Model path (default: Qwen/Qwen3.5-9B)")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true",
                        help="Use 4-bit quantization (saves ~50%% VRAM)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max output tokens (default: 4096)")
    args = parser.parse_args()

    # Get input prompt
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            user_input = f.read().strip()
    elif args.prompt:
        user_input = args.prompt
    else:
        print("Error: provide prompt as argument or via --input file")
        return

    print(f"[INFO] User input: {user_input[:100]}...")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model(args.model, args.use_4bit)

    # Generate
    result = rewrite(model, tokenizer, user_input, args.max_tokens)

    print(f"{'='*60}")
    print(f"[RESULT]:\n{result}")
    print(f"{'='*60}")

    # Save if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[INFO] Saved to {args.output}")


if __name__ == "__main__":
    main()
