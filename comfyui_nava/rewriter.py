"""
Prompt-rewriter engine for the NAVAPromptRewriter ComfyUI node.

Loads a Qwen3 chat model (default: NAVA's bundled Qwen3-4B-Thinking-2507)
and runs the same SYSTEM_PROMPT used by `pe_src/rewrite_single.py` /
`pe_src/rewrite.py`. We re-import those modules instead of copy-pasting so
SYSTEM_PROMPT and extract_rewrite stay the single source of truth.

Heavy weights are cached in a module-level dict so flipping `enabled` off/on
in a workflow doesn't reload them.
"""

import os
import sys
import time
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Reach into NAVA's pe_src to reuse SYSTEM_PROMPT + extract_rewrite.
# Layout: <nava_root>/comfyui_nava/rewriter.py  AND  <nava_root>/pe_src/*.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.realpath(__file__))
_NAVA_ROOT = os.path.dirname(_HERE)
_PE_SRC = os.path.join(_NAVA_ROOT, "pe_src")
if _PE_SRC not in sys.path:
    sys.path.insert(0, _PE_SRC)

# Import lazily inside the rewrite call so that simply importing this module
# (e.g. when ComfyUI scans nodes at startup) doesn't require pe_src deps.

# ---------------------------------------------------------------------------
# Model cache: key = (resolved_model_path, use_4bit)
# ---------------------------------------------------------------------------
_REWRITER_CACHE: dict = {}


def _resolve_model_path(model_path: str) -> str:
    """Accept absolute, relative-to-cwd, relative-to-NAVA-root, or HF repo id."""
    if not model_path:
        return os.path.join(_PE_SRC, "Qwen3-4B-Thinking-2507")
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
    # Treat as HF repo id; transformers will download on first use.
    return model_path


def _load(model_path: str, use_4bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[NAVA-Rewriter] Loading {model_path} ({'4bit' if use_4bit else 'bf16'})")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    print(f"[NAVA-Rewriter] Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def get_or_load(model_path: str, use_4bit: bool):
    resolved = _resolve_model_path(model_path)
    key = (os.path.abspath(resolved) if os.path.exists(resolved) else resolved, bool(use_4bit))
    if key not in _REWRITER_CACHE:
        _REWRITER_CACHE[key] = _load(resolved, use_4bit)
    return _REWRITER_CACHE[key]


@torch.no_grad()
def rewrite(
    prompt: str,
    model_path: str = "",
    use_4bit: bool = False,
    max_new_tokens: int = 4096,
    temperature: float = 0.3,
    top_p: float = 0.75,
    top_k: int = 20,
    repetition_penalty: float = 1.05,
    seed: int = 42,
    disable_thinking: bool = False,
) -> str:
    """Run a single rewrite. Returns cleaned Chinese long prompt."""
    # Pull SYSTEM_PROMPT and extract_rewrite from NAVA's pe_src so they stay in sync.
    # ComfyUI ships its own top-level `utils` package, which shadows pe_src/utils.py
    # when pe_src/rewrite.py does `from utils import ...`. Temporarily swap sys.modules
    # entries while we import, then restore so we don't disturb ComfyUI.
    import importlib.util as _ilu
    _saved = {k: sys.modules.get(k) for k in ("utils", "rewrite", "rewrite_single")}
    try:
        spec = _ilu.spec_from_file_location("utils", os.path.join(_PE_SRC, "utils.py"))
        pe_utils = _ilu.module_from_spec(spec)
        sys.modules["utils"] = pe_utils
        spec.loader.exec_module(pe_utils)
        # Force fresh imports so they bind to our pe_src `utils`, not ComfyUI's.
        sys.modules.pop("rewrite", None)
        sys.modules.pop("rewrite_single", None)
        from rewrite_single import SYSTEM_PROMPT
        from rewrite import extract_rewrite
    finally:
        for k, v in _saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    model, tokenizer = get_or_load(model_path, use_4bit)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    # Try to honor enable_thinking on Qwen3-style chat templates; fall back if
    # the model's template doesn't expose the kwarg (e.g. Qwen3-*-Thinking-2507
    # is hard-wired to think and silently ignores it).
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        chat_kwargs["enable_thinking"] = False
    try:
        text = tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(messages, **chat_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        repetition_penalty=repetition_penalty,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    elapsed = time.time() - t0
    print(f"[NAVA-Rewriter] Generated {len(new_tokens)} tokens in {elapsed:.1f}s "
          f"({len(new_tokens) / max(elapsed, 1e-3):.1f} tok/s)")

    cleaned = extract_rewrite(raw)
    return cleaned.strip()


def offload_all_to_cpu():
    """No-op for device_map=auto models — accelerate manages placement."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
