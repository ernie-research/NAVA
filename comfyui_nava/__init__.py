"""
NAVA ComfyUI custom node package.

Installation
------------
Two options:

1. Symlink (recommended — stays in sync with the NAVA repo):
       cd <ComfyUI root>/custom_nodes
       ln -s <path-to-NAVA>/comfyui_nava .

2. Copy:
       cp -r <path-to-NAVA>/comfyui_nava <ComfyUI root>/custom_nodes/

Then restart ComfyUI.  The six NAVA nodes appear under the "NAVA" category
in the node browser:
    - NAVA Model Loader
    - NAVA Image Captioner (Qwen3-VL)   (image → caption text)
    - NAVA Prompt Compose               (caption + speech → prompt with <S><E>)
    - NAVA Prompt Rewriter (optional, can be bypassed via `enabled` toggle)
    - NAVA Sampler
    - NAVA Save Video

Requirements
------------
All NAVA dependencies must be installed in the same Python environment as
ComfyUI (torch, torchaudio, torchvision, safetensors, yaml, Pillow, scipy,
flash-attn).  Run:
    pip install -r <path-to-NAVA>/requirements.txt
inside the ComfyUI venv before launching.

Notes
-----
- This package must reside inside the NAVA repository root so that
  nava_src/, configs/, and checkpoint files are reachable via relative paths.
- Single-GPU inference only (no torchrun / sequence parallel).
  For multi-GPU SP inference use inference_nava.py / gradio_demo/ directly.
- Peak VRAM: ~80 GB (no offload), ~48 GB (t5_offload), ~42 GB (t5+group offload).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
