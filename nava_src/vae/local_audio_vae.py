import os
import io

import torch
import torchaudio
import numpy as np
from types import SimpleNamespace

from nava_src.vendor.ltx_core.model.audio_vae.audio_vae import AudioEncoder, AudioDecoder
from nava_src.vendor.ltx_core.model.audio_vae.model_configurator import (
    AudioEncoderConfigurator,
    AudioDecoderConfigurator,
    VocoderConfigurator,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
)
from nava_src.vendor.ltx_core.model.audio_vae.ops import AudioProcessor
from nava_src.vendor.ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from nava_src.vendor.ltx_core.types import Audio, AudioLatentShape

TARGET_SAMPLE_RATE = 16000


class SampleClass:
    def __init__(self, sample):
        self.content = sample

    def sample(self):
        return self.content


def _adapt_channels(audio, target_ch):
    actual_ch = audio.shape[1]
    if actual_ch == target_ch:
        return audio
    if actual_ch > target_ch:
        if target_ch == 1:
            return audio.mean(dim=1, keepdim=True)
        else:
            return audio[:, :1].expand(-1, target_ch, -1)
    else:
        return audio[:, :1].expand(-1, target_ch, -1)


class LtxAudioVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, vocoder, target_sample_rate=16000):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocoder = vocoder
        self.target_sample_rate = target_sample_rate
        self._audio_processor = AudioProcessor(
            target_sample_rate=encoder.sample_rate,
            mel_bins=encoder.mel_bins,
            mel_hop_length=encoder.mel_hop_length,
            n_fft=encoder.n_fft,
        ).to(device=self.device)

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @torch.no_grad()
    def wrapped_encode(self, audio_data):
        if audio_data.dim() == 2:
            audio_data = audio_data.unsqueeze(0)
        audio_data = audio_data.to(device=self.device)
        audio_data = _adapt_channels(audio_data, self.encoder.in_channels)
        audio_obj = Audio(waveform=audio_data, sampling_rate=self.target_sample_rate)
        mel = self._audio_processor.waveform_to_mel(audio_obj)
        latent = self.encoder(mel.to(dtype=next(self.encoder.parameters()).dtype))
        return self.encoder.patchifier.patchify(latent).transpose(1, 2)

    @torch.no_grad()
    def wrapped_decode(self, latents):
        latents = latents.to(device=self.device,
                             dtype=next(self.decoder.parameters()).dtype)
        latents = latents.transpose(1, 2)
        latent_shape = AudioLatentShape(
            batch=latents.shape[0],
            channels=self.decoder.z_channels,
            frames=latents.shape[1],
            mel_bins=latents.shape[2] // self.decoder.z_channels,
        )
        latents = self.decoder.patchifier.unpatchify(latents, latent_shape)
        mel = self.decoder(latents)
        waveform = self.vocoder(mel).float()
        vocoder_sr = self.vocoder.output_sampling_rate
        if vocoder_sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=vocoder_sr, new_freq=self.target_sample_rate
            )
        return waveform


def init_ltx_vae(ckpt_dir, device="cuda"):
    ckpt_path = os.path.join(ckpt_dir, "LTX2/ltx-2.3-22b-dev_audio_vae.safetensors")
    assert os.path.exists(ckpt_path), f"LTX audio VAE checkpoint not found: {ckpt_path}"
    encoder = SingleGPUModelBuilder(
        model_path=ckpt_path,
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=torch.device(device), dtype=torch.float32)
    decoder = SingleGPUModelBuilder(
        model_path=ckpt_path,
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=torch.device(device), dtype=torch.float32)
    vocoder = SingleGPUModelBuilder(
        model_path=ckpt_path,
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=torch.device(device), dtype=torch.float32)
    encoder.requires_grad_(False).eval()
    decoder.requires_grad_(False).eval()
    vocoder.requires_grad_(False).eval()
    return LtxAudioVAE(encoder, decoder, vocoder, target_sample_rate=TARGET_SAMPLE_RATE)


def _audio_post_process(waveform):
    limit = 0.99
    waveform = waveform.clamp(-limit, limit)
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    return waveform


class LocalAudioVAEAdapter:
    """Wraps LtxAudioVAE to provide VAEServerAdapter-compatible interface for inference."""

    def __init__(self, ltx_vae, spk_model=None, sample_rate=16000):
        self.ltx_vae = ltx_vae
        self.spk_model = spk_model
        self.sample_rate = sample_rate
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    @property
    def dtype(self):
        return torch.float32

    def encode(self, x, rank=-1, **kwargs):
        """
        For spk_embs extraction from audio.

        Args:
            x: dict with "bos_url" and "use_spk_emb" keys
        Returns:
            SimpleNamespace(latent_dist=SampleClass(sample=dict))
        """
        use_spk_emb = x.get("use_spk_emb", False) if isinstance(x, dict) else False
        spk_embs = torch.zeros((1, 192), dtype=torch.float32)

        if use_spk_emb and self.spk_model is not None and isinstance(x, dict):
            bos_url = x.get("bos_url", "")
            if os.path.exists(bos_url):
                try:
                    audio_data, sr = torchaudio.load(bos_url)
                    if sr != 16000:
                        audio_data = torchaudio.functional.resample(audio_data, orig_freq=sr, new_freq=16000)
                    # Truncate to first 30s (align with online server)
                    spk_len = int(30.0 * 16000)
                    if audio_data.shape[-1] > spk_len:
                        audio_data = audio_data[..., :spk_len]
                    spk_audio = audio_data.mean(dim=0, keepdim=True).to(self.ltx_vae.device)
                    with torch.no_grad():
                        spk_embs = self.spk_model(spk_audio).cpu()
                    print(f"[SpkEmb] Encoded {bos_url}, shape={spk_embs.shape}, norm={spk_embs.norm().item():.4f}")
                except Exception as e:
                    print(f"[SpkEmb] ERROR encoding {bos_url}: {e}")
            else:
                print(f"[SpkEmb] WARNING: file not found: {bos_url}")

        result = {
            "audio_latents": torch.zeros((1, 128), dtype=torch.float32),
            "spk_embs": spk_embs,
        }
        return SimpleNamespace(latent_dist=SampleClass(sample=result))

    def decode(self, z, rank=-1):
        """
        Decode audio latents to waveform.

        Args:
            z: tensor [1, 128, l] or dict {"data": list}
        Returns:
            SimpleNamespace(sample={"waveform": tensor, "sample_rate": int})
        """
        if isinstance(z, dict):
            audio_data = z["data"]
            z_tensor = torch.tensor(audio_data, dtype=torch.float32)
        else:
            z_tensor = z.float()

        if z_tensor.dim() == 2:
            z_tensor = z_tensor.unsqueeze(0)

        with torch.no_grad():
            waveform = self.ltx_vae.wrapped_decode(z_tensor)

        waveform = _audio_post_process(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        result = {
            "waveform": waveform.cpu(),
            "sample_rate": self.sample_rate,
        }
        return SimpleNamespace(sample=result)
