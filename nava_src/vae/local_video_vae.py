import os
import threading
import subprocess
import tempfile
import torch
import numpy as np
from PIL import Image
from types import SimpleNamespace

_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}


class SampleClass:
    def __init__(self, sample):
        self.content = sample

    def sample(self):
        return self.content


_RATIO2HWS = {
    "1/1":  [(1,1),(2,2),(3,3),(4,4),(5,5),(7,7),(9,9),(12,12),(16,16),(20,20),(25,25),(32,32),(40,40),(51,51),(60,60)],
    "4/5":  [(1,1),(1,2),(2,3),(3,4),(4,6),(6,8),(8,10),(10,13),(13,17),(17,22),(22,28),(28,35),(36,44),(44,56),(54,66)],
    "3/4":  [(1,1),(1,2),(2,3),(3,4),(4,6),(5,8),(7,10),(9,13),(12,17),(16,22),(21,28),(27,36),(34,45),(43,57),(52,68)],
    "2/3":  [(1,1),(1,2),(2,4),(3,5),(4,7),(5,9),(7,12),(9,15),(12,19),(16,24),(20,31),(25,39),(31,49),(41,62),(46,74)],
    "9/16": [(1,1),(1,2),(1,4),(2,5),(3,7),(4,9),(6,12),(8,16),(12,21),(13,25),(17,32),(24,42),(30,52),(36,64),(44,80)],
    "1/2":  [(1,1),(1,2),(1,4),(2,6),(3,8),(4,10),(6,13),(8,17),(10,22),(13,28),(17,36),(22,45),(28,56),(36,72),(42,84)],
    "2/5":  [(1,1),(1,2),(1,4),(2,7),(3,9),(4,12),(5,16),(7,20),(9,25),(12,32),(16,40),(20,51),(25,64),(32,80),(38,96)],
}
_RES2SCALE = {256: 9, 512: 12, 640: 13, 960: 15}


def _compute_tgt_hw_vid(ori_height, ori_width, resolution=960):
    """Aspect-ratio bucket resize matching demo_fastapi.py compute_tgt_hw_vid
    (used when resize_crop_by_nearest_ratio=True)."""
    ratio = min(ori_height / ori_width, ori_width / ori_height)
    if ratio >= (1/1 + 4/5) / 2:
        key = "1/1"
    elif ratio >= (4/5 + 3/4) / 2:
        key = "4/5"
    elif ratio >= (3/4 + 2/3) / 2:
        key = "3/4"
    elif ratio >= (2/3 + 9/16) / 2:
        key = "2/3"
    elif ratio >= (9/16 + 1/2) / 2:
        key = "9/16"
    elif ratio >= (1/2 + 2/5) / 2:
        key = "1/2"
    else:
        key = "2/5"
    scale_num = _RES2SCALE[resolution]
    short_side, long_side = _RATIO2HWS[key][scale_num - 1]
    short_side *= 16
    long_side *= 16
    if ori_height > ori_width:
        return long_side, short_side  # (height, width)
    else:
        return short_side, long_side  # (height, width)


def _find_ratio_from_dims(w, h, resolution=960):
    """Returns (height, width) using aspect-ratio bucket strategy."""
    return _compute_tgt_hw_vid(h, w, resolution=resolution)


def _find_ratio(image_path, resolution=960):
    image = Image.open(image_path)
    w, h = image.size
    return _compute_tgt_hw_vid(h, w, resolution=resolution)


def _load_video_frames(video_path: str, frame_length: int,
                       tgt_height: int, tgt_width: int,
                       src_height: int, src_width: int,
                       fps: float = 8.0, start: float = 0.0) -> torch.Tensor:
    """Use ffmpeg to transcode at target fps, scale (preserving aspect ratio) then center-crop
    to tgt_height x tgt_width — matching server clip_video + preprocess_image pipeline.
    Returns float32 tensor [T, C, H, W] in [-1, 1]."""
    import torchvision.io as tio

    duration = frame_length / fps + 0.5  # small buffer so cut_sample always has enough frames

    # Replicate server scale computation: scale so both dims >= target, then center-crop
    scale = max(tgt_height / src_height, tgt_width / src_width)
    fin_h = int(round(src_height * scale) / 2) * 2
    fin_w = int(round(src_width  * scale) / 2) * 2

    suffix = os.path.splitext(video_path)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-vf", f"scale={fin_w}:{fin_h}",
            "-r", str(fps),
            "-c:v", "h264", "-preset", "fast", "-tune", "zerolatency",
            "-crf", "18", "-an",
            tmp_path, "-y",
        ]
        subprocess.run(cmd, check=True)
        vframes, _, _ = tio.read_video(tmp_path, pts_unit="sec", output_format="TCHW")
    finally:
        os.unlink(tmp_path)

    total = len(vframes)
    if total == 0:
        raise ValueError(f"No frames decoded from: {video_path}")

    # cut_sample: take first frame_length frames, pad last frame if short
    if total < frame_length:
        pad = frame_length - total
        vframes = torch.cat([vframes, vframes[-1:].expand(pad, -1, -1, -1)], dim=0)
    else:
        vframes = vframes[:frame_length]

    # center crop to exact target dims (server uses Ftrans.center_crop)
    _, _, h, w = vframes.shape
    top  = (h - tgt_height) // 2
    left = (w - tgt_width)  // 2
    vframes = vframes[:, :, top:top + tgt_height, left:left + tgt_width]

    # normalize: x/255*2-1, matching server preprocess_image (min=-1, max=1)
    frames = vframes.float() / 255.0 * 2.0 - 1.0  # [T, C, H, W]
    return frames


def _resize_center_crop(image, target_h, target_w):
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resize_w = int(round(src_w * scale))
    resize_h = int(round(src_h * scale))
    image = image.resize((resize_w, resize_h), Image.LANCZOS)
    left = (resize_w - target_w) // 2
    top = (resize_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))
    return image


class LocalVideoVAEAdapter:
    """Wraps Wan2_2_VAE to provide VAEServerAdapter-compatible interface."""

    def __init__(self, wan_vae, resolution=960):
        self.wan_vae = wan_vae
        self.resolution = resolution
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
        self._lock = threading.Lock()

    @property
    def dtype(self):
        return torch.float32

    def encode(self, x, rank=-1, frame_length=5, fps=8, target_height=None, target_width=None):
        """
        Encode an image (i2v first frame) or video file, or a pre-loaded tensor.

        Args:
            x: local file path (str) or tensor [B, C, H, W]
               - video file (.mp4 etc.): encodes frame_length frames
               - image file (.jpg/.png etc.): encodes as single-frame (i2v)
            frame_length: number of frames to sample from video
            target_height/target_width: override auto aspect-ratio selection
        Returns:
            SimpleNamespace(latent_dist=SampleClass(sample=tensor))
            tensor shape [T_lat, H_lat, W_lat, z_dim] (t,h,w,c format)
        """
        if isinstance(x, str) and os.path.splitext(x)[1].lower() in _VIDEO_EXTS:
            # --- video path ---
            if target_height and target_width:
                height, width = target_height, target_width
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height",
                     "-of", "csv=p=0", x],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                src_w, src_h = [int(v) for v in result.stdout.strip().split(",")]
            else:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height",
                     "-of", "csv=p=0", x],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                src_w, src_h = [int(v) for v in result.stdout.strip().split(",")]
                height, width = _find_ratio_from_dims(src_w, src_h, resolution=self.resolution)

            frames = _load_video_frames(x, frame_length=frame_length,
                                        tgt_height=height, tgt_width=width,
                                        src_height=src_h, src_width=src_w,
                                        fps=fps)
            # frames: [T, C, H, W]
            device = self.wan_vae.device
            video_tensor = frames.to(device).permute(1, 0, 2, 3).unsqueeze(0)
            # → [1, C, T, H, W]
            with self._lock, torch.no_grad():
                latent = self.wan_vae.wrapped_encode(video_tensor)
            # [1, z_dim, T_lat, H_lat, W_lat] → [T_lat, H_lat, W_lat, z_dim]
            latent = latent.permute(0, 2, 3, 4, 1).squeeze(0)
            return SimpleNamespace(latent_dist=SampleClass(sample=latent))

        elif isinstance(x, str):
            # --- image path ---
            if target_height and target_width:
                height, width = target_height, target_width
            else:
                height, width = _find_ratio(x, resolution=self.resolution)
            image = Image.open(x).convert("RGB")
            image = _resize_center_crop(image, height, width)
            img_tensor = torch.from_numpy(np.array(image)).float()
            img_tensor = (img_tensor / 255.0) * 2.0 - 1.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            img_tensor = x
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)

        device = self.wan_vae.device
        img_tensor = img_tensor.to(device)
        # [B, 3, H, W] → [B, 3, 1, H, W]
        video_tensor = img_tensor.unsqueeze(2)

        with self._lock, torch.no_grad():
            latent = self.wan_vae.wrapped_encode(video_tensor)
        # latent: [B, z_dim, 1, h_latent, w_latent]
        # → permute to [B, 1, h_latent, w_latent, z_dim]
        latent = latent.permute(0, 2, 3, 4, 1)
        # squeeze batch if B=1 → [1, h_latent, w_latent, z_dim]
        latent = latent.squeeze(0)

        return SimpleNamespace(latent_dist=SampleClass(sample=latent))

    def decode(self, z, rank=-1, cpu_offload=False, tiled=False,
               tile_size=(22, 40), tile_stride=(14, 26)):
        """
        Decode video latents.

        Args:
            z: tensor [t, h, w, c] (latent space video)
            cpu_offload: accumulate decoded frames on CPU to reduce GPU peak memory
        Returns:
            SimpleNamespace(sample=tensor[T, C, H, W]) in [-1, 1]
        """
        device = self.wan_vae.device
        # [t, h, w, c] → [c, t, h, w]
        z = z.permute(3, 0, 1, 2).to(device)
        # → [1, c, t, h, w]
        z = z.unsqueeze(0)

        with torch.no_grad():
            decoded = self.wan_vae.wrapped_decode(z, cpu_offload=cpu_offload,
                                                  tiled=tiled, tile_size=tile_size,
                                                  tile_stride=tile_stride)
        # decoded: [1, 3, T_decoded, H, W] (may be on CPU if cpu_offload)
        # → squeeze batch → [3, T, H, W] → permute → [T, 3, H, W]
        decoded = decoded.squeeze(0).permute(1, 0, 2, 3)

        return SimpleNamespace(sample=decoded)
