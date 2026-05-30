"文本到图像数据集"
import os
import json
import torch
import random
from typing_extensions import Self
from sympy import elliptic_f
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
import math
import subprocess

def find_ratio(image_path, resolution=640):
    image = Image.open(image_path)
    w, h = image.size
    ratio = w/h
    if resolution == 640:
        if ratio  == 1:
            height = 640
            width = 640
        elif ratio > 1:
            width = 832
            height = 480
        elif ratio < 1:
            width = 480
            height = 832
        return height, width
    elif resolution == 960:
        if ratio  == 1:
            height = 960
            width = 960
        elif ratio > 1:
            width = 1280
            height = 704
        elif ratio < 1:
            width = 704
            height = 1280
        return height, width
    else:
        raise ValueError("resolution must be 640 or 960")

def collate_fn(batch, is_packing=False):
    """
    - 使用 tokenizer 的 pad_token_id（如果提供），否则 0
    - 动态对齐到批内最大长度
    - 支持 images 全 None 或部分 None
    - data_state 若存在则堆叠为 [B, 1+num_sources]
    """
    out = {}

    all_keys = set().union(*(d.keys() for d in batch))

    # 3. 图像处理
    if "image_latents" in all_keys:
        raw_imgs = [b.get("image_latents", None) for b in batch]
        flat_imgs = []
        t_h_w_list = []
        valid_template = None  # 用来捕获 C 维度和 dtype
        # 第一次遍历：Flatten、收集形状、寻找这一批次里的“标准图”
        for idx, img in enumerate(raw_imgs):
            assert img is not None
            if img is None:
                flat_imgs.append(None)
                t_h_w_list.append((0, 0, 0))
            else:
                if img.dim() == 4 and not is_packing:   # non-packing mode, B = batch size
                    B, H, W, C = img.shape 
                    t_h_w_list.append((1, H, W))
                elif img.dim() == 3: # packing mode, B=1
                    B, H_W, C = img.shape
                    if is_packing:
                        # packing mode, 不改变pack函数内语义
                        t_h_w_list += batch[idx]["h_w_list"]
                    else:
                        t_h_w_list.append((H_W))
                flat_img = img.view(-1, C)
                flat_imgs.append(flat_img)
                
                # 记录第一个有效的图像作为模板 (获取 C 和 dtype)
                if valid_template is None:
                    valid_template = flat_img

        out["t_h_w_list"] = torch.tensor(t_h_w_list)


        # Case A: 整个 Batch 都是纯文 (valid_template 依然是 None)
        if valid_template is None and "image_latents" not in batch[0].keys():
            out["image_latents"] = None
        # Case B: 混合数据 或 纯图数据
        else:
            # 计算最大的token数 + 1（给eoi留位置）
            max_len = max(
                (img.shape[0] for img in flat_imgs if img is not None), default=0
            )
            C = valid_template.shape[-1]
            dtype = valid_template.dtype

            padded_imgs = torch.full(
                (len(batch), max_len, C), fill_value=0.0, dtype=dtype
            )
            for i, img in enumerate(flat_imgs):
                if img is not None:
                    cur_len = img.shape[0]
                    # 填入真实图像数据
                    padded_imgs[i, :cur_len, :] = img
            out["image_latents"] = padded_imgs


    # 4. 视频处理
    if "video_latents" in all_keys:
        raw_vids = [b.get("video_latents", None) for b in batch]
        flat_vids = []
        t_h_w_list = []
        valid_template = None  # 用来捕获 C 维度和 dtype
        # 第一次遍历：Flatten、收集形状、寻找这一批次里的“标准图”
        for idx, vid in enumerate(raw_vids):
            assert vid is not None
            if vid is None:
                flat_vids.append(None)
                t_h_w_list.append((0, 0, 0))
            else:
                if vid.dim() == 4 and not is_packing:   # non-packing mode, B = batch size
                    T, H, W, C = vid.shape 
                    t_h_w_list.append((T, H, W))
                elif vid.dim() == 3: # packing mode, B=1
                    _, T_H_W, C = vid.shape
                    if is_packing:
                        # packing mode, 不改变pack函数内语义
                        t_h_w_list += batch[idx]["h_w_list"]
                    else:
                        t_h_w_list.append((T_H_W))
                flat_vid = vid.view(-1, C)
                flat_vids.append(flat_vid)

                # 记录第一个有效的图像作为模板 (获取 C 和 dtype)
                if valid_template is None:
                    valid_template = flat_vid

        out["t_h_w_list"] = torch.tensor(t_h_w_list)


        # Case A: 整个 Batch 都是纯文 (valid_template 依然是 None)
        if valid_template is None and "video_latents" not in batch[0].keys():
            out["video_latents"] = None
        # Case B: 混合数据 或 纯图数据
        else:
            # 计算最大的token数 + 1（给eoi留位置）
            max_len = max(
                (vid.shape[0] for vid in flat_vids if vid is not None), default=0
            )
            C = valid_template.shape[-1]
            dtype = valid_template.dtype

            # 预分配内存：默认全填 pad_token_id
            padded_vids = torch.full(
                (len(batch), max_len, C), fill_value=0.0, dtype=dtype
            )
            for i, vid in enumerate(flat_vids):
                if vid is not None:
                    cur_len = vid.shape[0]
                    # 填入真实图像数据
                    padded_vids[i, :cur_len, :] = vid
            out["video_latents"] = padded_vids

    # =========================================================
    # 其他杂项处理 (data_state 等)
    # =========================================================
    processed_keys = {
        "image_latents",
        "videos_latents",
        "t_h_w_list"
    }
    remaining_keys = [k for k in all_keys if k not in processed_keys]
    for k in remaining_keys:
        # 注意：混合数据中，某些 key 可能只在部分样本中存在
        # 这里假设如果存在 data_state，所有样本都有，或者我们只收集有的。
        # 为了安全，取出来的 None 需要过滤或者补全，这里保持原逻辑：
        vals = [b.get(k, None) for b in batch]
        if all(x is None for x in vals):
            vals = None
        out[k] = vals
        if vals is not None:
            if any(v is None for v in vals):
                out[k] = vals
            elif vals and isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
    
    return out


class T2VDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        format='txt',
        width=256,
        height=256,
        frames=5,
        patch_size=16,
        video_vae=None
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
        self.resolution = video_vae.resolution if video_vae else 480
        self.width = width
        self.height = height
        self.frames = frames
        self.patch_size = patch_size

        self.text_list = []
        self.save_path_list = []
        self.first_frames = []
        self.video_vae = video_vae
        # self.img_list = []

        if format == 'json':
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.text_list.append(data['eb4_turbo8k_ch_1118'])
                    save_path = os.path.join(data["dimension"][0], f"{data['prompt_en']}.mp4")
                    self.save_path_list.append(save_path)
                    # todo: add i2v mode
        elif format == 'txt':
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split(" image_path:")) > 1:
                        img_path = line.split(" image_path:")[1]
                        line = line.split(" image_path:")[0]
                        self.first_frames.append(img_path)
                        print(line, img_path)
                    if not line:
                        continue
                    self.text_list.append(line)
                    self.save_path_list.append(line[:20]+line[-10:])
        elif format == 'i2v_json':
            for root_dir, _, files in os.walk(data_file):
                for json_file in sorted(files):
                    if not json_file.endswith('.json'):
                        continue
                    with open(os.path.join(root_dir, json_file), 'r', encoding='utf-8') as f:
                        data = json.loads(f.read())
                    self.first_frames.append(data["image_path"])
                    assert os.path.exists(data["image_path"])
                    self.text_list.append(data["prompt"])
                    rel_path = os.path.relpath(os.path.join(root_dir, json_file), data_file)
                    self.save_path_list.append(rel_path[:-5] + ".mp4")
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        save_path = self.save_path_list[idx]

        h, w = self.height // self.patch_size, self.width // self.patch_size
        video_latents = torch.randn((self.frames, h, w, 48))
        img_latents = None

        if len(self.first_frames) > 0:
            img_latents = self.video_vae.encode(
                self.first_frames[idx],
                rank=-1,
                frame_length=self.frames,
                fps=24
            ).latent_dist.sample()
            video_latents = torch.randn((self.frames, img_latents.shape[1], img_latents.shape[2], 48))

        return {
            "idx": idx,
            "video_latents": video_latents,  # [C,H,W]
            "first_frames": img_latents,
            "save_path": save_path,
            "captions": text,
        }

class T2AVDataset(Dataset):
    """
    Unified audio-video dataset.

    Input format: JSONL file, one JSON object per line.
    Each entry supports:
      - "prompt" (required): text caption. Also accepts "text" for backward compat.
      - "image_path" (optional): absolute path to first-frame image → enables i2v mode for this sample.
      - "spk_wavs" (optional): list of absolute paths to speaker reference WAVs (max 2).

    Examples:
      {"prompt": "一位男子在海边奔跑..."}
      {"prompt": "...", "image_path": "/path/to/img.png"}
      {"prompt": "...<S>Hello<E>...", "spk_wavs": ["/path/to/spk1.wav", "/path/to/spk2.wav"]}
      {"prompt": "...", "image_path": "/path/to/img.png", "spk_wavs": ["/path/to/spk.wav"]}
    """

    def __init__(
        self,
        data_file: str,
        format='json',
        width=256,
        height=256,
        frames=5,
        patch_size=16,
        fps=16,
        audio_tokens_per_sec=31.25,
        audio_vae=None,
        use_speech_special_token=False,
        video_vae=None
    ):
        super().__init__()

        self.format = format
        self.resolution = video_vae.resolution
        self.width = width
        self.height = height
        self.frames = frames
        self.patch_size = patch_size
        self.fps = fps
        self.audio_tokens_per_sec = audio_tokens_per_sec
        self.audio_vae = audio_vae
        self.use_speech_special_token = use_speech_special_token
        self.video_vae = video_vae

        self.data_list = []
        self.save_path_list = []
        self.first_frames = []

        if format == 'json':
            with open(data_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.data_list.append(data)

                    # Per-sample i2v: store image_path if present and exists
                    image_path = data.get("image_path", None)
                    if image_path and os.path.exists(image_path):
                        self.first_frames.append(image_path)
                    else:
                        self.first_frames.append(None)

                    # Save path from prompt slug
                    prompt = data.get("prompt", data.get("text", ""))
                    prompt_slug = prompt[:20].replace(" ", "_").replace("/", "_")
                    self.save_path_list.append(f"idx{idx}_{prompt_slug}")
        elif format == 'txt':
            with open(data_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    self.data_list.append(line)
                    self.first_frames.append(None)
                    self.save_path_list.append(f"{idx}_{line[:20]}_{line[-10:]}")
        else:
            raise NotImplementedError(f"Unsupported format: {format}")

        print(f"[T2AVDataset] Loaded {len(self.data_list)} samples from {data_file}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sample_spk_embs = None

        # Extract text caption
        if isinstance(data, dict):
            text = data.get("prompt", data.get("text", ""))
            text = text.replace("<S>", "<S><extra_id_2>")
            if self.use_speech_special_token:
                text = text.replace("<S>", "<extra_id_0>").replace("<E>", "<extra_id_1>")

            # Speaker embeddings from local WAV paths
            spk_wavs = data.get("spk_wavs", None)
            if spk_wavs is not None and len(spk_wavs) > 0:
                sample_spk_embs = []
                for spk_wav in spk_wavs:
                    spk_embs = torch.zeros((1, 192), dtype=torch.float32)
                    if spk_wav and spk_wav != "None" and os.path.exists(spk_wav):
                        query = {
                            "data_path": spk_wav,
                            "use_spk_emb": True,
                        }
                        result = self.audio_vae.encode(query).latent_dist.sample()
                        spk_embs = result["spk_embs"]
                    sample_spk_embs.append(spk_embs)
        else:
            text = data

        save_path = self.save_path_list[idx]

        h, w = self.height // self.patch_size, self.width // self.patch_size
        video_latents = torch.randn((self.frames, h, w, 48))

        video_duration = ((self.frames - 1) * 4 + 1) / self.fps
        audio_len = math.ceil(video_duration * self.audio_tokens_per_sec)
        audio_latents = torch.randn((audio_len, 48))

        # Per-sample i2v: encode first frame if image_path is available
        img_latents = None
        if self.first_frames[idx] is not None:
            img_latents = self.video_vae.encode(
                self.first_frames[idx],
                rank=-1,
                frame_length=self.frames,
                fps=24
            ).latent_dist.sample()
            video_latents = torch.randn((self.frames, img_latents.shape[1], img_latents.shape[2], 48))

        return {
            "idx": idx,
            "video_latents": video_latents,
            "first_frames": img_latents,
            "audio_latents": audio_latents,
            "save_path": save_path,
            "captions": text,
            "spk_embs": sample_spk_embs,
            "is_i2v": img_latents is not None,
        }
