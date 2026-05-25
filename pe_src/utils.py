"""Utility functions for the rewriter."""

import json
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load config from yaml file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompt(config: dict) -> str:
    """Load system prompt from template file."""
    prompt_file = config["rewrite"]["system_prompt_file"]
    # Resolve relative to pe_src directory
    if not os.path.isabs(prompt_file):
        prompt_file = Path(__file__).parent / prompt_file
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_prompts(input_path: str) -> list[str]:
    """Read prompts from txt (one per line) or jsonl file."""
    prompts = []
    ext = Path(input_path).suffix.lower()

    with open(input_path, "r", encoding="utf-8") as f:
        if ext == ".jsonl" or ext == ".json":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    prompts.append(obj["text"])
        else:
            # txt: one prompt per line, literal \n represents newlines
            for line in f:
                line = line.strip()
                if line:
                    # Restore literal \\n to actual newlines
                    prompts.append(line.replace("\\n", "\n"))

    logger.info(f"Loaded {len(prompts)} prompts from {input_path}")
    return prompts


def write_results(output_path: str, results: list[str], format: str = "txt"):
    """Write rewritten results to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if format == "jsonl":
            for r in results:
                f.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")
        else:
            for r in results:
                # Flatten to single line
                f.write(r.replace("\n", "\\n") + "\n")

    logger.info(f"Wrote {len(results)} results to {output_path}")


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
