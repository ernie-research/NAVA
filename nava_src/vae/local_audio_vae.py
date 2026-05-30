import os
import io
import threading

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


_SQRT2_INV = 1.0 / (2 ** 0.5)  # ITU-R BS.775 downmix coefficient


def _downmix_to_stereo(audio: torch.Tensor) -> torch.Tensor:
    """audio: (B, C, T), C > 2 — ITU-R BS.775 standard downmix to stereo."""
    C = audio.shape[1]
    if C == 5:  # 5.0: L R C Ls Rs
        L, R, Cch, Ls, Rs = [audio[:, i] for i in range(5)]
        left  = L + _SQRT2_INV * Cch + _SQRT2_INV * Ls
        right = R + _SQRT2_INV * Cch + _SQRT2_INV * Rs
    elif C == 6:  # 5.1: L R C LFE Ls Rs
        L, R, Cch, _LFE, Ls, Rs = [audio[:, i] for i in range(6)]
        left  = L + _SQRT2_INV * Cch + _SQRT2_INV * Ls
        right = R + _SQRT2_INV * Cch + _SQRT2_INV * Rs
    elif C == 8:  # 7.1: L R C LFE Ls Rs SL SR
        L, R, Cch, _LFE, Ls, Rs, SL, SR = [audio[:, i] for i in range(8)]
        left  = L + _SQRT2_INV * Cch + _SQRT2_INV * Ls + _SQRT2_INV * SL
        right = R + _SQRT2_INV * Cch + _SQRT2_INV * Rs + _SQRT2_INV * SR
    else:  # unknown layout: mean → dual mono
        mono = audio.mean(dim=1, keepdim=True)
        return mono.expand(-1, 2, -1)
    return torch.stack([left, right], dim=1)


def _adapt_channels(audio: torch.Tensor, target_ch: int) -> torch.Tensor:
    """audio: (B, C, T) → (B, target_ch, T)"""
    actual_ch = audio.shape[1]
    if actual_ch == target_ch:
        return audio
    if actual_ch > target_ch:
        if target_ch == 1:
            return audio.mean(dim=1, keepdim=True)
        else:
            return _downmix_to_stereo(audio)
    else:
        # mono → stereo: zero-copy expand
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
        self._lock = threading.Lock()

    @property
    def dtype(self):
        return torch.float32

    # 1 latent timestep = sample_rate / audio_tokens_per_sec = 16000 / 25 = 640 samples
    _SAMPLES_PER_LATENT_STEP = TARGET_SAMPLE_RATE / 25.0

    def encode(self, x, rank=-1, **kwargs):
        """
        Encode audio to latents and optionally extract speaker embedding.

        Args:
            x: dict with keys:
               - "data_path": local audio file path
               - "use_spk_emb": bool
               - "start" (optional): clip start time in seconds
               - "duration" (optional): clip duration in seconds
               - "target_length" (optional): target latent timesteps; waveform is
                 zero-padded to int(target_length * 640) samples before encoding,
                 matching demo_fastapi_spk.py behaviour
        Returns:
            SimpleNamespace(latent_dist=SampleClass(sample=dict))
            sample = {"audio_latents": [tensor[C, T]], "spk_embs": tensor[1, D]}
        """
        use_spk_emb    = x.get("use_spk_emb", False)    if isinstance(x, dict) else False
        data_path      = x.get("data_path", "")          if isinstance(x, dict) else str(x)
        start          = x.get("start", None)             if isinstance(x, dict) else None
        duration       = x.get("duration", None)          if isinstance(x, dict) else None
        target_length  = x.get("target_length", None)     if isinstance(x, dict) else None

        spk_embs = torch.zeros((1, 192), dtype=torch.float32)
        audio_data = None

        if os.path.exists(data_path):
            try:
                wav, sr = torchaudio.load(data_path)
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
                if start is not None and duration is not None:
                    s = int(start * self.sample_rate)
                    e = int((start + duration) * self.sample_rate)
                    wav = wav[:, s:e]
                audio_data = wav  # [channels, samples]
            except Exception as e:
                print(f"[AudioVAE] load error {data_path}: {e}")
        else:
            print(f"[AudioVAE] file not found: {data_path}")

        # --- waveform zero-padding to target_length (before encode, matching server) ---
        if audio_data is not None and target_length is not None and target_length > 0:
            target_samples = int(target_length * self._SAMPLES_PER_LATENT_STEP)
            current_samples = audio_data.shape[-1]
            if current_samples < target_samples:
                pad_len = target_samples - current_samples
                audio_data = torch.nn.functional.pad(
                    audio_data, (0, pad_len), mode='constant', value=0.0
                )

        # --- encode latents ---
        audio_latents_ct = None
        if audio_data is not None:
            try:
                with self._lock:
                    latents = self.ltx_vae.wrapped_encode(audio_data)  # [1, C, T]
                audio_latents_ct = latents[0]                          # [C, T]
            except Exception as e:
                print(f"[AudioVAE] encode error {data_path}: {e}")

        if audio_latents_ct is None:
            audio_latents_ct = torch.zeros((128, 1), dtype=torch.float32)

        # --- speaker embedding ---
        if use_spk_emb and self.spk_model is not None and audio_data is not None:
            try:
                spk_wav = audio_data.mean(dim=0, keepdim=True)
                spk_len = int(30.0 * self.sample_rate)
                if spk_wav.shape[-1] > spk_len:
                    spk_wav = spk_wav[..., :spk_len]
                with torch.no_grad():
                    spk_embs = self.spk_model(spk_wav.to(self.ltx_vae.device)).cpu()
            except Exception as e:
                print(f"[SpkEmb] ERROR {data_path}: {e}")

        result = {
            "audio_latents": [audio_latents_ct],  # list of [C, T]
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
