"""
Batch rewrite prompts using Qwen3-4B via vLLM API.

Usage:
    python rewrite.py --input prompts.txt --output rewritten.txt
    python rewrite.py --input data.json --output rewritten.jsonl --format jsonl
    python rewrite.py --input prompts.txt --output rewritten.txt --concurrency 64
"""

import argparse
import asyncio
import logging
import re
import time
from pathlib import Path

import aiohttp

from utils import load_config, load_system_prompt, read_prompts, setup_logging, write_results

logger = logging.getLogger(__name__)


def extract_rewrite(raw: str) -> str:
    """Strip Qwen3-Thinking <think>...</think> blocks and keep only the rewrite.

    Handles three cases:

    1. vLLM with ``--enable-reasoning --reasoning-parser`` strips ``<think>``
       on its side — content is already clean. We just normalize whitespace.
    2. Older vLLM / no reasoning parser: thinking arrives in-band wrapped in
       literal ``<think>...</think>`` text. Mirror gradio's logic: keep what
       follows the LAST ``</think>``.
    3. Worst case: thinking arrives as plain text WITHOUT any ``<think>``
       marker (vLLM may have stripped the special tokens). Then we fall back
       to a heuristic: the actual rewrite is required (by the system prompt)
       to be a single Chinese paragraph that opens with "画面" / "这段" /
       "这是一段". Find the LAST occurrence of one of these openers and take
       from there to the end, collapsing any internal newlines.
    """
    s = raw.strip()

    # Case 1/2: explicit <think>...</think> markers
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[-1].strip()
    if "<think>" in s:
        s = s.split("<think>", 1)[0].strip()

    # Case 3: residual thinking dump with no markers. The final rewrite is a
    # single paragraph; if we still see newlines or known thinking-prefix
    # words, look for the last rewrite opener and slice from there.
    rewrite_openers = ("画面呈现", "这是一段", "这段写实", "画面中")
    looks_like_thinking = (
        "\n" in s
        or "首先" in s[:200]
        or "分析" in s[:200]
        or "完整输出" in s
        or "改写草稿" in s
        or "最终输出" in s
        or "最终 prompt" in s
    )
    if looks_like_thinking:
        last_pos = -1
        for opener in rewrite_openers:
            pos = s.rfind(opener)
            if pos > last_pos:
                last_pos = pos
        if last_pos > 0:
            s = s[last_pos:].strip()

    # Case 4: model emitted a clean rewrite, then drifted back into meta
    # commentary. Two-stage strategy:
    #
    #   (a) Anchor on legitimate endings. The system prompt requires the
    #       rewrite to close with "整体听感…，突出…。" or "整体氛围…，营造
    #       出…。". If we find one, keep up to the first 。/！/？ AFTER it
    #       and drop everything that follows — that's the most reliable
    #       "rewrite is done here" signal we have.
    #   (b) Otherwise, fall back to high-confidence meta markers, and only
    #       count an occurrence if it's at a sentence boundary (preceded by
    #       。！？ or whitespace, or at the very start). This avoids cutting
    #       on words like "调整" / "应该" / "改成" when they appear inside a
    #       legitimate clause.
    end_anchors = ("整体听感", "整体氛围")
    anchor_pos = -1
    for a in end_anchors:
        p = s.rfind(a)
        if p > anchor_pos:
            anchor_pos = p
    if anchor_pos >= 0:
        # take the first sentence terminator after the anchor
        tail = s[anchor_pos:]
        terminators = [tail.find(t) for t in ("。", "！", "？")]
        terminators = [t for t in terminators if t >= 0]
        if terminators:
            s = s[: anchor_pos + min(terminators) + 1].strip()
    else:
        # High-confidence meta markers only. "调整 / 应该 / 改成" removed
        # because they legitimately appear in rewrites.
        strict_markers = (
            "注意：", "注意:", "用户说", "用户的输入", "用户没",
            "改写草稿", "最终输出", "最终 prompt", "最终prompt",
            "为了准确", "为了符合要求", "我应该", "我需要",
        )
        sentence_breaks = "。！？\n "
        earliest = len(s)
        for m in strict_markers:
            start = 0
            while True:
                p = s.find(m, start)
                if p < 0:
                    break
                # Sentence boundary check: at start, or right after 。！？\n / space
                if p == 0 or s[p - 1] in sentence_breaks:
                    if p < earliest:
                        earliest = p
                    break
                start = p + 1
        if earliest < len(s):
            head = s[:earliest]
            cut = max(head.rfind("。"), head.rfind("！"), head.rfind("？"))
            if cut > 0:
                s = head[: cut + 1].strip()
            else:
                s = head.strip()

    # The rewrite must be single-paragraph per the system prompt; collapse
    # any stray line breaks that survived.
    s = s.replace("\r", "").replace("\n", "")
    return s.strip()


_SPEECH_RE = re.compile(r"<S>.*?<E>", re.DOTALL)


def count_speech_tags(text: str) -> int:
    return len(_SPEECH_RE.findall(text))


async def rewrite_single(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_url: str,
    model_name: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retry: int,
    index: int,
    top_p: float = 1.0,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
) -> tuple[int, str]:
    """Rewrite a single prompt with retry logic."""
    async with semaphore:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            # vLLM OpenAI-compat extension: top_k / repetition_penalty live
            # under `extra_body` for the official client, but the raw HTTP
            # endpoint accepts them at the top level too.
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

        for attempt in range(retry):
            try:
                async with session.post(
                    api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(f"[{index}] HTTP {resp.status}: {error_text[:200]}")
                        if attempt < retry - 1:
                            await asyncio.sleep(1)
                            continue
                        return index, f"[ERROR] HTTP {resp.status}"

                    result = await resp.json()
                    content = result["choices"][0]["message"]["content"]
                    cleaned = extract_rewrite(content)
                    logger.debug(f"[{index}] Done ({len(content)} -> {len(cleaned)} chars)")
                    return index, cleaned

            except asyncio.TimeoutError:
                logger.warning(f"[{index}] Timeout (attempt {attempt + 1}/{retry})")
                if attempt < retry - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"[{index}] Error: {e} (attempt {attempt + 1}/{retry})")
                if attempt < retry - 1:
                    await asyncio.sleep(1)

        return index, f"[ERROR] Failed after {retry} retries"


async def batch_rewrite(
    prompts: list[str],
    config: dict,
    system_prompt: str,
) -> list[str]:
    """Rewrite all prompts concurrently."""
    rewrite_cfg = config["rewrite"]
    api_url = rewrite_cfg["api_base"]
    model_name = rewrite_cfg["model_name"]
    concurrency = rewrite_cfg["concurrency"]
    temperature = rewrite_cfg["temperature"]
    max_tokens = rewrite_cfg["max_tokens"]
    timeout = rewrite_cfg["timeout"]
    retry = rewrite_cfg["retry"]
    top_p = rewrite_cfg.get("top_p", 1.0)
    top_k = rewrite_cfg.get("top_k", -1)
    repetition_penalty = rewrite_cfg.get("repetition_penalty", 1.0)

    semaphore = asyncio.Semaphore(concurrency)
    results = [""] * len(prompts)

    logger.info(f"Starting batch rewrite: {len(prompts)} prompts, concurrency={concurrency}")
    logger.info(f"API: {api_url}, model: {model_name}")

    async with aiohttp.ClientSession() as session:
        tasks = [
            rewrite_single(
                session, semaphore, api_url, model_name,
                system_prompt, text, temperature, max_tokens,
                timeout, retry, i,
                top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            for i, text in enumerate(prompts)
        ]

        # Progress tracking
        done_count = 0
        total = len(tasks)
        start_time = time.time()

        for coro in asyncio.as_completed(tasks):
            index, content = await coro
            results[index] = content
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                elapsed = time.time() - start_time
                speed = done_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {done_count}/{total} ({speed:.1f} prompts/s)")

    # Report errors
    errors = [i for i, r in enumerate(results) if r.startswith("[ERROR]")]
    if errors:
        logger.warning(f"{len(errors)} prompts failed: indices {errors}")

    # Sanity-check: <S><E> pair counts should match between input and output
    # (the rewriter is required to preserve speech spans verbatim).
    mismatches = []
    for i, (src, out) in enumerate(zip(prompts, results)):
        if out.startswith("[ERROR]"):
            continue
        in_n = count_speech_tags(src)
        out_n = count_speech_tags(out)
        if in_n > 0 and in_n != out_n:
            mismatches.append((i, in_n, out_n))
    if mismatches:
        logger.warning(
            f"{len(mismatches)} prompts have speech-tag count mismatch "
            f"(input vs rewrite): {mismatches[:10]}{'...' if len(mismatches) > 10 else ''}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch rewrite prompts using Qwen3-4B")
    parser.add_argument("--input", "-i", required=True, help="Input file (txt or jsonl)")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--format", "-f", default="txt", choices=["txt", "jsonl"],
                        help="Output format (default: txt)")
    parser.add_argument("--config", "-c", default=None, help="Config yaml path")
    parser.add_argument("--concurrency", type=int, default=None, help="Override concurrency")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load config
    config = load_config(args.config)

    # Override from CLI
    if args.concurrency is not None:
        config["rewrite"]["concurrency"] = args.concurrency
    if args.temperature is not None:
        config["rewrite"]["temperature"] = args.temperature

    # Load system prompt
    system_prompt = load_system_prompt(config)
    logger.info(f"System prompt: {system_prompt[:80]}...")

    # Read input
    prompts = read_prompts(args.input)
    if not prompts:
        logger.error("No prompts loaded. Check input file.")
        return

    # Run rewrite
    start = time.time()
    results = asyncio.run(batch_rewrite(prompts, config, system_prompt))
    elapsed = time.time() - start

    logger.info(f"Completed in {elapsed:.1f}s ({len(prompts) / elapsed:.1f} prompts/s)")

    # Write output
    write_results(args.output, results, format=args.format)
    logger.info(f"Done! Output: {args.output}")


if __name__ == "__main__":
    main()
