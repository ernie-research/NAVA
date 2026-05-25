"文本到图像数据集"
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import yaml
from functools import partial

def collate_fn(batch, is_packing=False):
    """
    - 使用 tokenizer 的 pad_token_id（如果提供），否则 0
    - 动态对齐到批内最大长度
    - 支持 images 全 None 或部分 None
    - data_state 若存在则堆叠为 [B, 1+num_sources]
    """
    out = {}
    processed_keys = {
        "idx",
        "captions",
        "audio_latents",
        "image_latents",
        "video_latents",
        "save_path",
        "spk_embs",
    }
    for k in processed_keys:
        vals = [b.get(k, None) for b in batch]
        if all(x is None for x in vals):
            vals = None
        out[k] = vals
    
    out["audio_seq_len"] = [
        b["audio_latents"].shape[-1] if b["audio_latents"] is not None else 0 for b in batch
    ]
    return out


class T2ADataset(Dataset):
    def __init__(
        self,
        data_file: str,
        format='txt',
        duration="10.0",
        audio_vae_server=None,
        use_speech_special_token=False,
    ):
        """
        文本到图像数据集的初始化构造函数

        Args:
            data_file (str): 数据文件路径，包含训练数据
            format (str, optional): 数据文件格式，支持'jsonl'或'txt'. 默认为'txt'
            resolution (int, optional): 图像分辨率. 默认为256
            patch_size (int, optional): 图像补丁大小. 默认为16
        """

        super().__init__()

        self.format = format
        self.duration = float(duration)
        self.use_speech_special_token = use_speech_special_token
        self.base_bos_url = "bos://bj-copy-secret/wangguan/temp/spk_wav"
        self.audio_vae_server = audio_vae_server
        assert audio_vae_server is not None, "audio_vae_server must be provided"

        self.data_list = []
        self.save_path_list = []

        if format == 'json':
            with open(data_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.data_list.append(data)
                    spk_wavs = data.get("spk_wavs", [])
                    save_path = f"idx{idx}_{'-'.join(spk_wavs)}"
                    self.save_path_list.append(save_path)
        elif format == 'txt':
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    id, text, tag = line.split('\t', 2)
                    self.data_list.append(text)
                    save_path = f"{tag}/{id}"
                    self.save_path_list.append(save_path)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sample_spk_embs = None

        text = data
        if isinstance(data, dict):
            text = data.get("text", "").replace("<S>", "<S><extra_id_2>")
            if self.use_speech_special_token:
                text = text.replace("<S>", "<extra_id_0>").replace("<E>", "<extra_id_1>")
            if "spk_wavs" in data:
                spk_wavs = data.get("spk_wavs", [])
                if len(spk_wavs) > 0:
                    sample_spk_embs = []
                for spk_wav in spk_wavs:
                    spk_embs = torch.zeros((1, 192), dtype=torch.float32)
                    if spk_wav != "None":
                        if ".wav" not in spk_wav:
                            spk_wav += ".wav"
                        spk_url = os.path.join(self.base_bos_url, spk_wav)
                        query = {
                            "bos_url": spk_url,
                            "use_spk_emb": True,
                        }
                        result = self.audio_vae_server.encode(query).latent_dist.sample()
                        spk_embs = result["spk_embs"]
                    sample_spk_embs.append(spk_embs)
        save_path = self.save_path_list[idx]
        num_frames = math.ceil(self.duration * 31.25)
        audio_latents = torch.zeros((num_frames, 20))

        return {
            "idx": idx,
            "audio_latents": audio_latents,  # [t, c]
            "save_path": save_path,
            "captions": text,
            "spk_embs": sample_spk_embs,
        }

if __name__ == "__main__":
    from nava_src.vae.vae_server import VAEServerAdapter
    cfg = yaml.safe_load(open("configs/audio_spk_eval.yaml", "r"))

    audio_vae_server = VAEServerAdapter(
        modality="audio", 
        scaling_factor=cfg["data"].get("audio_scaling_factor", 1.0),
        shift_factor=cfg["data"].get("shift_factor", 0.0),
        server_list="nava_src/data/server/audio_server.list",
        server_port=4431, 
    )
    ds = T2ADataset(
        data_file="benchmark/audio/multi_speaker.json",
        format="json",
        duration=cfg["data"]["audio_duration"],
        audio_vae_server=audio_vae_server,
        use_speech_special_token=cfg["data"]["use_speech_special_token"],
    )
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn), 
        drop_last=False,
        pin_memory=True
    )

    for step, batch in enumerate(dl):
        import pdb; pdb.set_trace()
        print("-" * 40)
        print("len(spk_embs): ", len(batch["spk_embs"][0]))
        print("captions: ", batch["captions"])