import torch
import numpy as np
from PIL import Image
from types import SimpleNamespace


class SampleClass:
    def __init__(self, sample):
        self.content = sample

    def sample(self):
        return self.content


def _find_ratio(image_path, resolution=960):
    image = Image.open(image_path)
    w, h = image.size
    ratio = w / h
    if resolution == 640:
        if ratio == 1:
            height, width = 640, 640
        elif ratio > 1:
            height, width = 480, 832
        else:
            height, width = 832, 480
    elif resolution == 960:
        if ratio == 1:
            height, width = 960, 960
        elif ratio > 1:
            height, width = 704, 1280
        else:
            height, width = 1280, 704
    else:
        raise ValueError("resolution must be 640 or 960")
    return height, width


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

    @property
    def dtype(self):
        return torch.float32

    def encode(self, x, rank=-1, frame_length=5, fps=8, target_height=None, target_width=None):
        """
        Encode a single image (for i2v first frame).

        Args:
            x: local image path (str) or tensor [B, C, H, W]
            target_height: explicit target height (skip _find_ratio if provided)
            target_width: explicit target width (skip _find_ratio if provided)
        Returns:
            SimpleNamespace(latent_dist=SampleClass(sample=tensor))
            where tensor shape is [1, h_latent, w_latent, z_dim] (t,h,w,c format)
        """
        if isinstance(x, str):
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

        with torch.no_grad():
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
