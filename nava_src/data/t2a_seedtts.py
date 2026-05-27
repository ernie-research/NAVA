"""
SeedTTS Benchmark 数据集
用于零样本语音合成评估，使用参考音频提取音色，生成目标文本对应的音频

SeedTTS 数据格式 (meta.lst):
    utt_id|prompt_text|prompt_wav|infer_text

说明:
    - utt_id: 最终预测音频的文件名 (如 10002526-00000012 -> 10002526-00000012.wav)
    - prompt_text: 参考音频对应的文本
    - prompt_wav: 参考音频文件路径 (用于提取音色 spk_embs)
    - infer_text: 要生成的目标文本

关键逻辑:
    - spk_embs: 从 prompt_wav 提取的 speaker embedding
    - audio_latents: 根据估算的目标音频长度用 zeros 初始化，仅用于提供音频长度信息
    - 目标音频长度 = (目标文本长度 / 参考文本长度) * 参考音频时长 * audio_tokens_per_sec
"""
import os
from typing import Optional
from torch.utils.data import Dataset
import torch


def collate_fn(batch):
    """
    SeedTTS benchmark 的 collate 函数

    Args:
        batch: 包含多个样本的列表

    Returns:
        dict: 整理后的 batch 数据，兼容 pipeline 的 sample 方法
    """
    out = {}
    processed_keys = {
        "idx",
        "utt_id",
        "prompt_text",
        "prompt_audio_path",
        "captions",  # pipeline 使用的字段名，对应 target_text
        "spk_embs",
        "audio_latents",
        "save_path",
    }

    for k in processed_keys:
        vals = [b.get(k, None) for b in batch]
        if all(x is None for x in vals):
            vals = None
        out[k] = vals

    # 记录音频序列长度
    if "audio_latents" in out and out["audio_latents"]:
        out["audio_seq_len"] = [
            b["audio_latents"].shape[0] if b["audio_latents"] is not None else 0 for b in batch
        ]
    else:
        out["audio_seq_len"] = [0] * len(batch)

    # 兼容 pipeline 的 t_h_w_list（音频模式为 None）
    out["t_h_w_list"] = None

    return out


class SeedTTSDatasetWithVAE(Dataset):
    """
    SeedTTS Benchmark 数据集 (带 VAE 编码)

    直接在数据集中调用 vae_server 进行参考音频编码，提取 speaker embedding
    目标音频的 latents 用 zeros 初始化，仅用于提供长度信息
    """

    def __init__(
        self,
        meta_file: str,
        language: str = "zh",
        audio_vae=None,
        audio_tokens_per_sec: float = 31.25,
        audio_latent_ch: int = 20,
        use_speech_special_token: bool = False,
        use_avgen_format: bool = False,
        audio_base_url: Optional[str] = None,
        video_caption: str = "一位金色卷发、身穿红色西装外套的人在虚化处理的演播室环境中静止发言，神情严肃专注，正对镜头。画面中央突出人物主体，背景透过窗户隐约可见城市景观，整体色调柔和，以紫蓝为主。光线均匀，照亮人物面部与上半身，摄像机固定，采用中近景拍摄，确保面部表情与上半身清晰可见。",
        audio_caption_prefix: str = "音频中只有一位说话人，以平稳的语调陈述了一则新闻内容，语速正常，发音清晰。",
        audio_caption_suffix: str = "除了人声之外，背景中没有其他明显的声音。",
    ):
        """
        Args:
            meta_file (str): meta 文件路径
            language (str): 语言，"zh" 或 "en"
            audio_vae: 音频 VAE server 实例
            audio_tokens_per_sec (float): 音频 token 每秒数量
            audio_latent_ch (int): 音频 latent 通道数
            use_avgen_format (bool): 是否使用 avgen 格式，在 <S>...<E> 前后加中文音视频描述
            audio_base_url (str): prompt_wav 基础路径。绝对路径 / `bos://` URL 不受影响；
                相对路径会拼到 ``{audio_base_url}/{language}/{prompt_wav}``。
                默认取 meta 文件父目录的父目录（即假定 ``<root>/<lang>/meta.lst`` 布局）。
            video_caption (str): 视频描述，拼在最前面
            audio_caption_prefix (str): <S> 前的音频描述
            audio_caption_suffix (str): <E> 后的背景音描述
        """
        super().__init__()

        self.language = language
        self.audio_tokens_per_sec = audio_tokens_per_sec
        self.audio_latent_ch = audio_latent_ch
        self.audio_vae = audio_vae
        self.use_speech_special_token = use_speech_special_token
        self.use_avgen_format = use_avgen_format
        self.video_caption = video_caption.strip()
        self.audio_caption_prefix = audio_caption_prefix.strip()
        self.audio_caption_suffix = audio_caption_suffix.strip()

        # prompt_wav 解析根目录：默认从 meta_file 推断（假设 <root>/<lang>/meta.lst 布局）
        if audio_base_url is None:
            meta_dir = os.path.dirname(os.path.abspath(meta_file))
            audio_base_url = os.path.dirname(meta_dir)
        self.audio_base_url = audio_base_url

        self.samples = []

        # 读取 meta 文件
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Meta file not found: {meta_file}")

        with open(meta_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 解析每行数据: utt_id|prompt_text|prompt_wav|infer_text
                parts = line.split('|')
                if len(parts) != 4:
                    print(f"Warning: Invalid format at line {line_num}: {line}")
                    continue

                utt_id, prompt_text, prompt_wav, infer_text = parts

                self.samples.append({
                    "utt_id": utt_id,
                    "prompt_text": prompt_text,
                    "prompt_wav": prompt_wav,
                    "infer_text": infer_text,
                })

        print(f"[SeedTTSDatasetWithVAE] Loaded {len(self.samples)} samples from {meta_file}")

    def _estimate_audio_length(
        self,
        target_text: str,
        prompt_text: str,
        ref_audio_tokens: int,
    ) -> int:
        """
        根据目标文本长度、参考文本长度和参考音频 token 数，估算目标音频的 token 数

        公式: target_tokens = (target_text_len / prompt_text_len) * ref_audio_tokens

        Args:
            target_text: 目标文本
            prompt_text: 参考文本
            ref_audio_tokens: 参考音频的 token 数

        Returns:
            int: 估算的目标音频 token 数
        """
        target_len = len(target_text)
        prompt_len = len(prompt_text)

        if prompt_len == 0:
            prompt_len = 1

        # 按比例估算，不做 duration 限制
        estimated_tokens = int((target_len / prompt_len) * ref_audio_tokens)

        return estimated_tokens

    def _extract_spk_embedding(self, audio_path: str):
        """
        从参考音频提取 speaker embedding

        Args:
            audio_path: 参考音频路径

        Returns:
            tensor: speaker embedding [1, 192] 或 None
        """
        if self.audio_vae is None:
            return None

        try:
            resolved_path = audio_path

            # 绝对路径 / bos:// URL 不动；相对路径按 {audio_base_url}/{language}/{audio_path} 拼接
            if not (resolved_path.startswith("bos://") or os.path.isabs(resolved_path)):
                resolved_path = os.path.join(self.audio_base_url, self.language, audio_path)

            query = {
                "bos_url": resolved_path,
                "use_spk_emb": True,  # 提取 speaker embedding
            }

            rank = 0

            result = self.audio_vae.encode(
                query, rank=rank
            ).latent_dist.sample()

            # 提取 speaker embedding
            spk_embs = result["spk_embs"] if "spk_embs" in result else None
            # 获取参考音频 token 数，用于估算目标音频长度
            ref_audio_latents = result["audio_latents"] if "audio_latents" in result else None
            ref_audio_tokens = ref_audio_latents.shape[-1] if ref_audio_latents is not None else 313  # 默认约10秒
            if spk_embs is not None:
                spk_embs = [spk_embs]

            return spk_embs, ref_audio_tokens

        except Exception as e:
            print(f"[SeedTTSDatasetWithVAE] Error encoding audio {audio_path}: {e}")
            return None, 313  # 返回默认值

    def __getitem__(self, idx):
        """
        获取单个样本，包含 speaker embedding 和估算的目标音频长度

        Returns:
            dict: 包含编码特征的样本
        """
        sample = self.samples[idx]

        # 提取 speaker embedding 和参考音频 token 数
        spk_embs, ref_audio_tokens = self._extract_spk_embedding(sample["prompt_wav"])

        # 估算目标音频 token 数
        target_audio_tokens = self._estimate_audio_length(
            sample["infer_text"],
            sample["prompt_text"],
            ref_audio_tokens
        )

        if self.use_avgen_format:
            parts = []
            if self.video_caption:
                parts.append(self.video_caption)
            if self.audio_caption_prefix:
                parts.append(self.audio_caption_prefix)
            parts.append("<S><extra_id_2>" + sample["infer_text"] + "<E>")
            if self.audio_caption_suffix:
                parts.append(self.audio_caption_suffix)
            caption = " ".join(parts)
        else:
            caption = "<S><extra_id_2>" + sample["infer_text"] + "<E>"

        if self.use_speech_special_token:
            caption = caption.replace("<S>", "<extra_id_0>").replace("<E>", "<extra_id_1>")

        print(caption, "!!!!!!", self.audio_tokens_per_sec)

        # 用 zeros 初始化目标 audio_latents，仅用于提供长度信息
        audio_latents = torch.zeros((target_audio_tokens, self.audio_latent_ch), dtype=torch.float32)
    
        # 保存路径
        save_path = os.path.join(
            self.language,
            "wavs",
            f"{sample['utt_id']}.wav"
        )

        return {
            "idx": idx,
            "utt_id": sample["utt_id"],
            "prompt_text": sample["prompt_text"],
            "prompt_audio_path": sample["prompt_wav"],
            "captions": caption,  # pipeline 使用 captions 字段作为目标文本
            "target_text": sample["infer_text"],  # 保留原始字段名方便调试
            "spk_embs": spk_embs,  # [1, 192] 从参考音频提取的 speaker embedding
            "audio_latents": audio_latents,  # [T, 20] zeros 初始化，仅提供长度
            "save_path": save_path,
        }

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # 简单 sanity check：仅解析 meta，不连接 VAE server。
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a SeedTTS meta.lst file")
    parser.add_argument("meta_file", help="Path to meta.lst (utt_id|prompt_text|prompt_wav|infer_text)")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    args = parser.parse_args()

    dataset = SeedTTSDatasetWithVAE(
        meta_file=args.meta_file,
        language=args.language,
        audio_vae=None,  # 不做编码，仅看 meta 解析
    )
    print(f"Dataset size: {len(dataset)}")
    for i, sample in enumerate(dataset.samples[:3]):
        print(f"  [{i}] utt_id={sample['utt_id']}, prompt_wav={sample['prompt_wav']}")
