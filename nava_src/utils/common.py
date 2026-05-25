import os, random, torch
import numpy as np
import math

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_of(model):
    return next(model.parameters()).device

def autocast_dtype(use_bf16=True):
    return torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float16


def compute_tgt_ratio(ori_hight, ori_width):
    """
    计算的得到一个最接近的目标height, width
    """

    ratio = min(ori_hight / ori_width, ori_width / ori_hight)

    if ratio >= math.sqrt(1 / 1 * 4 / 5):
        key = "1/1"
    elif ratio >= math.sqrt(4 / 5 * 3 / 4):
        key = "4/5"
    elif ratio >= math.sqrt(3 / 4 * 2 / 3):
        key = "3/4"
    elif ratio >= math.sqrt(2 / 3 * 9 / 16):
        key = "2/3"
    elif ratio >= math.sqrt(9 / 16 * 1 / 2):
        key = "9/16"
    elif ratio >= math.sqrt(1 / 2 * 2 / 5):
        key = "1/2"
    else:
        key = "2/5"

    if ori_hight > ori_width:
        return key
    else:
        return "/".join(key.split("/")[::-1])