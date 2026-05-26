# Vendored: ltx_core

This directory contains a minimal subset of [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video), vendored into NAVA so users do not need to clone the upstream 170 GB repo just to load NAVA's audio VAE.

- **Upstream**: https://github.com/Lightricks/LTX-Video
- **Commit**: `ae855f8538843825f9015a419cf4ba5edaf5eec2`
- **Vendored on**: 2026-05-26
- **Subset**: `ltx_core/{types.py, utils.py, model/{audio_vae,common,model_protocol.py}, loader/, components/{__init__.py, patchifiers.py}}` — the transitive closure required by `nava_src/vae/local_audio_vae.py` to instantiate the audio encoder/decoder/vocoder.
- **License**: see `LICENSE` in this directory (LTX-2 Community License Agreement). The full upstream `LICENSE` is preserved verbatim.
- **Modifications**: internal absolute imports rewritten from `ltx_core.X` to `nava_src.vendor.ltx_core.X` so the package is importable in-place without `sys.path` manipulation. No functional changes.
