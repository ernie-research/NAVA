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
from .dataset_mixture import get_rope_index, create_sample_masks


class T2iDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        tokenizer,
        format='txt',
        resolution=256,
        patch_size=16,
        image_token_id=None,
        image_start_token_id=None,
        image_end_token_id=None,
        timestep_token_id=None,
        use_timestep=True,
        image_path=None,
        only_train_img=False
    ):
        """
        文本到图像数据集的初始化构造函数

        Args:
            data_file (str): 数据文件路径，包含训练数据
            tokenizer: 文本分词器，用于处理文本输入
            format (str, optional): 数据文件格式，支持'jsonl'或'txt'. 默认为'txt'
            resolution (int, optional): 图像分辨率. 默认为256
            patch_size (int, optional): 图像补丁大小. 默认为16
            image_token_id (int, optional): 图像标记ID. 默认为None
            image_start_token_id (int, optional): 图像开始标记ID. 默认为None
            image_end_token_id (int, optional): 图像结束标记ID. 默认为None
            timestep_token_id (int, optional): 时间步标记ID. 默认为None
            use_timestep (bool, optional): 是否使用时间步. 默认为True
            image_path (str, optional): 图像文件路径，当format为'jsonl'时使用. 默认为None
            only_train_img (bool, optional): 是否仅训练图像. 默认为False
        """

        super().__init__()

        self.tokenizer = tokenizer
        self.format = format
        self.resolution = resolution
        self.patch_size = patch_size
        self.image_token_id = image_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.timestep_token_id = timestep_token_id
        self.use_timestep = use_timestep
        self.only_train_img = only_train_img

        self.to_tensor = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self.text_list = []
        self.img_list = []

        if format == 'jsonl':
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.text_list.append(data['text'])
                    self.img_list.append(os.path.join(image_path, data['image']))
        elif format == 'txt':
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.text_list.append(line)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.text_list)

    def _encode_text(self, text: str):
        # 进入分词器
        enc = self.tokenizer(
            text,
            truncation=False,
            padding="do_not_pad",
            return_tensors="pt"
        )
        # 取出1D tensor
        ids = enc.input_ids[0]
        return ids, ids

    def __getitem__(self, idx):
        text = self.text_list[idx]
        input_ids, labels = self._encode_text(text)

        seq_len = input_ids.shape[0]

        position_ids = torch.arange(seq_len)

        if self.img_list:
            # load 本地图片
            raw_img = Image.open(self.img_list[idx]).convert("RGB")
            raw_img = self.to_tensor(raw_img)
        else:
            raw_img = None
        # 如果没有图像，random指定分辨率的img
        h, w = self.resolution // self.patch_size, self.resolution // self.patch_size
        img = torch.randn((1, h, w, 3))
        img_input_ids = torch.full((h*w,), fill_value=self.image_token_id)

        if self.use_timestep:
            input_ids = torch.concat(
                [
                    torch.tensor([self.tokenizer.pad_token_id]), input_ids, 
                    torch.tensor([self.timestep_token_id, self.image_start_token_id]), 
                    img_input_ids, torch.tensor([self.image_end_token_id])
                ]
            , dim=0)
        else:
            input_ids = torch.concat(
                [
                    torch.tensor([self.tokenizer.pad_token_id]), input_ids, 
                    torch.tensor([self.image_start_token_id]), img_input_ids, torch.tensor([self.image_end_token_id])
                ]
            , dim=0)

        position_ids, mrope_position_deltas = get_rope_index(
            input_ids.unsqueeze(0),
            torch.tensor([[1, h, w]]),
            image_token_id=self.image_token_id,
            video_token_id=-1,
            vision_start_token_id=self.image_start_token_id,
        )

        attention_mask, no_cond_attention_mask = create_sample_masks(
            input_ids,
            self.image_start_token_id,
            self.image_end_token_id,
            use_timestep=self.use_timestep
        )

        if self.only_train_img:
            from ..models.longcat_image.utils.model_utils import encode_prompt
            # 暂时写死
            prompt_template_encode_prefix = '<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n'
            prompt_template_encode_suffix = '<|im_end|>\n<|im_start|>assistant\n'
            input_ids, attention_mask = encode_prompt(
                prompt=text,
                tokenizer=self.tokenizer,
                text_tokenizer_max_length=512,
                prompt_template_encode_prefix=prompt_template_encode_prefix,
                prompt_template_encode_suffix=prompt_template_encode_suffix
            )
            no_cond_attention_mask = attention_mask.clone()

        return {
            "idx": idx,
            "input_ids": input_ids,
            "position_ids": position_ids.squeeze(1),
            "attention_mask": attention_mask,
            "no_cond_attention_mask": no_cond_attention_mask,
            "labels":    labels,
            "images":    img,  # [C,H,W]
            "raw_img": raw_img,
        }
