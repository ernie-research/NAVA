"文本到音频数据集（纯音频推理，兼容带/不带 timbre 参考）"
import os
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial


def collate_fn(batch):
    out = {}
    processed_keys = {"idx", "captions", "audio_latents", "save_path", "spk_embs"}
    for k in processed_keys:
        vals = [b.get(k, None) for b in batch]
        if all(x is None for x in vals):
            vals = None
        out[k] = vals

    out["audio_seq_len"] = [
        b["audio_latents"].shape[0] if b["audio_latents"] is not None else 0
        for b in batch
    ]
    return out


class T2ADataset(Dataset):
    """
    纯音频推理数据集，兼容带/不带 timbre 参考。

    JSON 格式（每行一个 JSON，与 T2AVDataset 一致）：
      {"prompt": "文本描述"}
      {"prompt": "文本 <S>台词<E>", "spk_wavs": ["/abs/path/to/spk.wav"]}
      {"prompt": "...", "spk_wavs": ["/path/spk1.wav", "/path/spk2.wav"]}
    """

    def __init__(
        self,
        data_file: str,
        format: str = "json",
        duration: float = 10.0,
        audio_tokens_per_sec: float = 31.25,
        audio_latent_ch: int = 20,
        audio_vae=None,
        use_speech_special_token: bool = False,
    ):
        super().__init__()

        self.format = format
        self.duration = float(duration)
        self.audio_tokens_per_sec = audio_tokens_per_sec
        self.audio_latent_ch = audio_latent_ch
        self.audio_vae = audio_vae
        self.use_speech_special_token = use_speech_special_token

        assert audio_vae is not None, "audio_vae must be provided"

        self.data_list = []
        self.save_path_list = []

        if format == "json":
            with open(data_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.data_list.append(data)
                    prompt = data.get("prompt", data.get("text", ""))
                    prompt_slug = prompt[:20].replace(" ", "_").replace("/", "_")
                    self.save_path_list.append(f"idx{idx}_{prompt_slug}")
        elif format == "txt":
            with open(data_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    self.data_list.append(line)
                    self.save_path_list.append(f"idx{idx}_{line[:20]}")
        else:
            raise NotImplementedError(f"Unsupported format: {format}")

        print(f"[T2ADataset] Loaded {len(self.data_list)} samples from {data_file}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sample_spk_embs = None

        if isinstance(data, dict):
            text = data.get("prompt", data.get("text", ""))
            text = text.replace("<S>", "<S><extra_id_2>")
            if self.use_speech_special_token:
                text = text.replace("<S>", "<extra_id_0>").replace("<E>", "<extra_id_1>")

            spk_wavs = data.get("spk_wavs", None)
            if spk_wavs is not None and len(spk_wavs) > 0:
                sample_spk_embs = []
                for spk_wav in spk_wavs:
                    spk_embs = torch.zeros((1, 192), dtype=torch.float32)
                    if spk_wav and spk_wav != "None" and os.path.exists(spk_wav):
                        query = {"bos_url": spk_wav, "use_spk_emb": True}
                        result = self.audio_vae.encode(query).latent_dist.sample()
                        spk_embs = result["spk_embs"]
                    sample_spk_embs.append(spk_embs)
        else:
            text = data

        audio_len = math.ceil(self.duration * self.audio_tokens_per_sec)
        audio_latents = torch.zeros((audio_len, self.audio_latent_ch))

        return {
            "idx": idx,
            "audio_latents": audio_latents,
            "save_path": self.save_path_list[idx],
            "captions": text,
            "spk_embs": sample_spk_embs,
        }
