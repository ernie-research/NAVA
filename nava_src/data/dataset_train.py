import json, os, random, torch
import threading
import queue
import time
import copy
from re import T
from PIL import Image, ImageFile
from collections import defaultdict
import numpy as np
import io
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import traceback

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, List, Dict
from functools import partial
import re
import math
from collections import deque

from nava_src.utils.common import compute_tgt_ratio

import torch.nn.functional as F
import torch.distributed as dist

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PerfLogger:
    """Thread-safe performance logger for diagnosing multi-node training slowdowns."""

    def __init__(self, worker_id, print_interval=60, tag="DATA_PERF"):
        self.worker_id = worker_id
        self.print_interval = print_interval
        self.tag = tag
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {"count": 0, "total": 0.0, "max": 0.0, "min": float("inf")})
        self._last_print = time.monotonic()

    def record(self, key, elapsed):
        with self._lock:
            s = self._stats[key]
            s["count"] += 1
            s["total"] += elapsed
            if elapsed > s["max"]:
                s["max"] = elapsed
            if elapsed < s["min"]:
                s["min"] = elapsed

    def maybe_print(self, force=False):
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_print) < self.print_interval:
                return
            if not self._stats:
                return
            lines = [f"[{self.tag} {self.worker_id}] === Perf Summary ==="]
            for key in sorted(self._stats.keys()):
                s = self._stats[key]
                cnt = s["count"]
                avg = s["total"] / cnt if cnt > 0 else 0
                lines.append(
                    f"  {key}: avg={avg:.4f}s max={s['max']:.4f}s min={s['min']:.4f}s total={s['total']:.2f}s cnt={cnt}"
                )
            print("\n".join(lines), flush=True)
            # reset
            self._stats.clear()
            self._last_print = now


def filter_video_descriptions(text):
    '''
    remove caption templete for videogen
    '''
    filtered_text = text.replace('\n', '').replace('概述：', '').replace('细节：', '').replace('背景：', '').strip()
    filtered_text = re.sub(f'这段视频[^, 。]*(了|是|\.|,)', "", filtered_text)
    filtered_text = re.sub(f'这段画面[^, 。]*(了|是|\.|,)', "", filtered_text)
    filtered_text = re.sub(f'视频[^, 。]*(了|是|\.|,)', "", filtered_text)
    filtered_text = re.sub(f'画面[^, 。]*(了|是|\.|,)', "", filtered_text)
    filtered_text = re.sub(r'视频中，', '', filtered_text)
    filtered_text = re.sub(r'视频中', '', filtered_text)
    filtered_text = re.sub(r'画面中，', '', filtered_text)
    filtered_text = re.sub(r'画面中', '', filtered_text)
    return filtered_text


@dataclass
class DistInfo:
    world_rank: int = 0  # 当前进程编号
    world_size: int = 1  # 总进程


class _LazyJsonlSource:
    """
    惰性按行读取一个JSONL文件。
    支持：从指定偏移跳过、is_cycle（读到末尾后循环）、skip_ratio（抽稀）、返回坏样本数。
    结果：(sample_dict, item_id, nb_bad)
    """

    def __init__(
        self,
        path_list,
        start_skip=0,
        is_cycle=True,
        skip_ratio=0.0,
        shard_id=0,
        num_shards=1,
        file_idx=0,
        data_file_divisor=1,
    ):
        self.path_list = path_list
        self.start_skip = max(0, start_skip)  # 至少跳过0行
        self.is_cycle = is_cycle
        self.skip_ratio = skip_ratio
        self._opened_once = False
        self._line_id = 0
        self.data_file_divisor = data_file_divisor
        self.current_part_id = shard_id % self.data_file_divisor
        assert num_shards % self.data_file_divisor == 0
        self.num_shard_per_part = num_shards // self.data_file_divisor
        self.shard_id_in_part = shard_id // self.data_file_divisor
    
        while file_idx % self.data_file_divisor != self.current_part_id:
            file_idx += 1

        # 当前文件下标以及当前句柄
        self._file_idx = file_idx % len(self.path_list)
        self._f = None

    def _open(self):
        # 先关闭旧文件
        if self._f is not None:
            self._f.close()
        path = self.path_list[self._file_idx]
        # 只读模式打开
        self._f = open(path, "r", encoding="utf-8")
        self._opened_once = True
        # 重置行
        self._line_id = 0

    def _line_iter(self):
        if not self._opened_once:
            self._open()
        while True:
            line = self._f.readline()
            if line:
                yield line
            else:
                self._f.close()
                self._f = None
                self._file_idx += self.data_file_divisor
                if self._file_idx >= len(self.path_list):
                    if not self.is_cycle:
                        return
                    self._file_idx = self.current_part_id
                self._open()

    def __iter__(self) -> Iterator[Tuple[dict, int, int]]:
        # 记录跳过的有效样本
        skipped = 0
        sample_idx = 0  # 统计“所有合法样本”的全局计数，不区分 shard

        # 跳过start_skip行，记录坏行
        # TODO: @songyi05: 暂时不支持断点续训，等待后续修复
        line_iter = self._line_iter()

        while skipped < self.start_skip:
            try:
                line = next(line_iter)
            except StopIteration:
                return
            self._line_id += 1
            skipped += 1
            sample_idx += 1
        # 主循环
        for line in line_iter:
            if not line:
                if not self.is_cycle:
                    return
                continue
            self._line_id += 1
            if (sample_idx % self.num_shard_per_part) != self.shard_id_in_part:
                sample_idx += 1
                continue
            # 如果抽稀
            if self.skip_ratio > 0 and random.random() < self.skip_ratio:
                sample_idx += 1
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj, self._file_idx, self._line_id
            sample_idx += 1


# TODO: 如何加速不同 modal 读取
# 在文生图任务时，设计 prompt template 以对齐推理
# TODO: @hujiahao03 后续解耦多模态数据集读取
# TODO: @hujiahao03 未来测试使用多模态 adaptive layernorm 处理不同模态 token; 实现困难
class AudioVideoDataset(IterableDataset):
    """
    每个样本：可能只有文本，也可能有 (caption, image)，按 config.caption_first_ratio 决定顺序。
    jsonl 格式：{"text": "..."} 或 {"text": "...", "image": "path/to.jpg"}
    优化版：惰性读取 + 严格按源加权 + 逐源历史
    """

    def __init__(
        self,
        jsonl_or_src_list,
        batch_size=4,  # NOTE: 仅在宽高比分桶时有效
        io_workers=16,  # <--- IO线程数
        queue_size=32,   # 队列深度，决定了能抗多久的抖动 (50个batch)
        # 多数据源参数
        src_id2ratios=None,
        modal_prob={"text_to_audio": 1.0, "text_to_video": 0.0, "text_to_image": 0.0, "text_to_av": 0.0},
        transform_config=None,
        # 流式数据集参数
        workers2history_dict=None,
        num_shards=1,
        dist_info=None,  # 计算全局worker id
        random_src=True,  # True随机选，False轮询
        upper_count=float("inf"),
        is_cycle=True,
        skip_ratio=0,
        is_json=True,
        audio_vae_server=None,
        image_vae_server=None,
        video_vae_server=None,
        use_aspect_ratio_buckets=False, # 宽高比分桶
        use_length_buckets=False,  # 按长度分桶（audio/video/av）
        num_length_buckets=10,  # 长度分桶数量
        is_packing=False, # seq packing
        min_audio_duration=0.0,
        max_audio_duration=15.0,
        tgt_audio_duration=-1,
        video_min_frames=1,
        video_max_frames=5,
        video_tgt_frames=-1,
        video_fps=8,
        add_spk_emb=False,
        spk_emb_prob=0.9,
        use_speech_special_token=False,
        data_file_divisor=1,
        split_wav_mode=False,
        audio_tokens_per_sec=31.25,
        enable_perf_log=False,
        enable_ddp_bucket_sync=False,  # DDP分桶同步：所有rank每步从同一个bucket取数据
    ):
        """
        jsonl_or_src_list: json file or list of json file regex prob(0.2) and modal
        src_id2ratios: 配比 {"regex": [prob(0.2), modal]}
        """
        super().__init__()
        self.batch_size = batch_size
        # 处理多数据源或单数据源输入
        if isinstance(jsonl_or_src_list, list):
            # 多数据源模式
            self.src_id2pathes = defaultdict(list)
            # TODO：@songyi05: 待优化，目前使用naive for循环 原始字符串匹配
            for tp in jsonl_or_src_list:
                if len(tp) == 1:
                    jsonl = tp[0]
                    for key in src_id2ratios.keys():
                        if key in jsonl:
                            self.src_id2pathes[key].append(jsonl)
                            break
                elif len(tp) == 2:
                    name, jsonl = tp
                    self.src_id2pathes[name].append(jsonl)
                else:
                    assert False
        else:
            # 单数据源兼容模式
            self.src_id2pathes = {"default": [jsonl_or_src_list]}

        self.src_id2ratios = src_id2ratios or {k: 1.0 for k in self.src_id2pathes}

        self.src_ids = sorted(
            self.src_id2pathes.keys()
        )  # "81000200000381,81000200000382,....."
        # 各个数据源权重

        self.ratios = [float(self.src_id2ratios[s][0]) for s in self.src_ids]

        self.io_workers = io_workers
        self.queue_size = queue_size
        self.io_pool = ThreadPoolExecutor(max_workers=self.io_workers)
        self._producer_running = False
        self.fetch_lock = threading.Lock()
        self._thread_local = threading.local()

        self._local_iter_count = 0

        # 目标模态 batch 比例
        # 示例：支持三种模态
        self.modality_configs = {
            "text_to_audio": {
                "weight": modal_prob.get("text_to_audio", 1.0),
            },
            "text_to_video": {
                "weight": modal_prob.get("text_to_video", 0.0),
            },
            "text_to_image": {
                "weight": modal_prob.get("text_to_image", 0.0),
            },
            "text_to_av": {
                "weight": modal_prob.get("text_to_av", 0.0)
            },
        }

        # self.io_workers = io_workers
        # self.queue_size = queue_size

        # === 为每个模态创建一个独立的 Queue ===
        self.modality_queues = {}
        self.producer_threads = [] # 存放线程句柄

        for m in self.modality_configs.keys():
            # 过滤掉权重为0的模态
            if self.modality_configs[m]['weight'] <= 0:
                continue
            # 文本队列容量可以大一点
            if m == "text_only":
                limit = self.queue_size * self.batch_size * 5
            else:
                limit = self.queue_size * self.batch_size
            
            self.modality_queues[m] = queue.Queue(maxsize=limit)

        self.modality_mappers = {
            "text_to_audio": 0,
            "text_to_video": 1,
            "text_to_image": 2,
            "text_to_av": 3,
        }

        # t2i buffer
        self.t2a_buffer = deque()
        self.t2v_buffer = deque()
        self.t2i_buffer = deque()
        self.t2av_buffer = deque()
        self.modality_buffers_mapper = {
            "text_to_audio": self.t2a_buffer,
            "text_to_video": self.t2v_buffer,
            "text_to_image": self.t2i_buffer,
            "text_to_av": self.t2av_buffer,
        }

        # 宽高比分桶
        self.use_aspect_ratio_buckets = use_aspect_ratio_buckets
        if self.use_aspect_ratio_buckets:
            self.aspect_ratio_buckets_aud_gen = defaultdict(list)
            self.aspect_ratio_buckets_vid_gen = defaultdict(list)
            self.aspect_ratio_buckets_img_gen = defaultdict(list)
            self.aspect_ratio_buckets_av_gen = defaultdict(list)
            self.aspect_ratio_buckets_mapper = {
                "text_to_audio": self.aspect_ratio_buckets_aud_gen,
                "text_to_video": self.aspect_ratio_buckets_vid_gen,
                "text_to_image": self.aspect_ratio_buckets_img_gen,
                "text_to_av": self.aspect_ratio_buckets_av_gen,
            }

        self.is_packing = is_packing

        # 长度分桶（audio/video/av 按 latent 长度等距分为 N 个桶）
        self.use_length_buckets = use_length_buckets
        self.num_length_buckets = num_length_buckets

        # DDP 分桶同步：需要 use_length_buckets 一起开启
        self.enable_ddp_bucket_sync = enable_ddp_bucket_sync and self.use_length_buckets
        if self.enable_ddp_bucket_sync:
            self._bucket_sync_buf = torch.tensor([-1], dtype=torch.long)

        # 每个桶的容量上限：桶满时停止从 Queue 拉数据，让 Queue 背压阻塞 producer
        self._per_bucket_cap = self.queue_size * self.batch_size if self.use_length_buckets else 0

        # 新增多worker参数
        self.num_shards = num_shards
        self.dist_info = dist_info or DistInfo(0, 1)
        self.random_src = random_src  # bool
        self.upper_count = upper_count
        self.is_cycle = is_cycle  # bool
        self.skip_ratio = skip_ratio  # float
        self.is_json = is_json
        self.worker_id = None
        self.data_file_divisor = data_file_divisor

        self.audio_vae_server = audio_vae_server
        self.image_vae_server = image_vae_server
        self.video_vae_server = video_vae_server

        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration
        self.tgt_audio_duration = tgt_audio_duration
        self.add_spk_emb = add_spk_emb
        self.spk_emb_prob = spk_emb_prob
        self.use_speech_special_token = use_speech_special_token
        self.video_min_frames = video_min_frames
        self.video_max_frames = video_max_frames
        self.video_tgt_frames = video_tgt_frames
        self.video_fps = video_fps
        self.split_wav_mode = split_wav_mode
        self.audio_tokens_per_sec = audio_tokens_per_sec
        print(self.audio_tokens_per_sec)
        self.enable_perf_log = enable_perf_log

        if self.use_length_buckets:
            self.length_buckets_aud_gen = defaultdict(deque)
            self.length_buckets_vid_gen = defaultdict(deque)
            self.length_buckets_av_gen = defaultdict(deque)
            self.length_buckets_mapper = {
                "text_to_audio": self.length_buckets_aud_gen,
                "text_to_video": self.length_buckets_vid_gen,
                "text_to_av": self.length_buckets_av_gen,
            }
            # 预计算各模态的桶边界（等距划分）
            # Audio: latent 长度范围
            aud_min_len = math.ceil(self.min_audio_duration * self.audio_tokens_per_sec)
            aud_max_len = math.ceil(self.max_audio_duration * self.audio_tokens_per_sec)
            self._aud_bucket_boundaries = np.linspace(aud_min_len, aud_max_len, self.num_length_buckets + 1)
            # Video / AV: latent 帧数范围，原始帧数->latent帧数: (f-1)//4+1
            vid_min_latent = (self.video_min_frames - 1) // 4 + 1
            vid_max_latent = (self.video_tgt_frames - 1) // 4 + 1
            self._vid_bucket_boundaries = np.linspace(vid_min_latent, vid_max_latent, self.num_length_buckets + 1)

        # 历史记录：[num_workers, num_sources]
        if workers2history_dict is None:
            self.workers2history_dict = {
                wid: {src_id: {"file_idx": 0, "line_id": 0} for src_id in self.src_ids}
                for wid in range(self.num_shards)
            }
        elif isinstance(workers2history_dict, list):
            self.workers2history_dict = {
                wid: {src_id: {"file_idx": 0, "line_id": 0} for src_id in self.src_ids}
                for wid in range(self.num_shards)
            }
            for wid in range(self.num_shards):
                row = workers2history_dict[
                    wid
                ]  # 一个 worker 的所有源数据，例如 [f1,l1,f2,l2,..]
                for si, src_id in enumerate(self.src_ids):
                    f = row[2 * si]
                    l = row[2 * si + 1]
                    self.workers2history_dict[wid][src_id]["file_idx"] = f
                    self.workers2history_dict[wid][src_id]["line_id"] = l

        elif isinstance(workers2history_dict, int):
            per = workers2history_dict // max(1, self.num_shards)
            self.workers2history_dict = {
                wid: {
                    src_id: {"file_idx": 0, "line_id": per} for src_id in self.src_ids
                }
                for wid in range(self.num_shards)
            }
        else:
            self.workers2history_dict = workers2history_dict

            # 对齐当前 src_ids，给“新增的数据源”补上默认历史
            for wid in range(self.num_shards):
                if wid not in self.workers2history_dict:
                    self.workers2history_dict[wid] = {}
                for src_id in self.src_ids:
                    if src_id not in self.workers2history_dict[wid]:
                        self.workers2history_dict[wid][src_id] = {
                            "file_idx": 0,
                            "line_id": 0,
                        }

        self._sources = []
        self._inited = False
        self._perf = None

        # print("[DEBUG __init__] src_id2pathes =", self.src_id2pathes)
        # print("[DEBUG __init__] src_ids =", getattr(self, "src_ids", None))

    def __len__(self):
        # 惰性读取模式下，返回upper_count作为长度估计
        return int(self.upper_count) * len(self.src_ids)

    def set_distributed_info(self, dist_info: DistInfo):
        """设置分布式训练信息"""
        self.dist_info = dist_info
        return self
    
    def _get_tokenizer(self):
        """
        获取当前线程专属的 Tokenizer 副本。
        如果没有，就从 self.tok 深拷贝一份。
        """
        
        # 检查当前线程是否已经有了私有副本
        if not hasattr(self._thread_local, 'tokenizer'):
            self._thread_local.tokenizer = copy.deepcopy(self.tok)
        return self._thread_local.tokenizer

    # 获取worker的id
    def get_current_worker(self):
        """获取当前worker ID，支持分布式训练"""
        if self.dist_info is None:
            process_index = 0
            num_processes = 1
        else:
            process_index, num_processes = (
                self.dist_info.world_rank,
                self.dist_info.world_size,
            )
        # 获取当前的worker信息
        worker_info = torch.utils.data.get_worker_info()
        num_processes = max(1, num_processes)

        if worker_info is not None:
            # 进程数*每个进程的子进程=总进程数
            assert (
                num_processes * worker_info.num_workers == self.num_shards
            ), f"{num_processes} {worker_info.num_workers} {self.num_shards}"
            return process_index * worker_info.num_workers + worker_info.id
        # 单进程
        else:
            assert num_processes == self.num_shards, f"process: {num_processes} shards: {self.num_shards}"
            return process_index

    # 只进行一次
    def _init_sources_once(self):
        if self._inited:
            return
        self.worker_id = self.get_current_worker()
        self._perf = PerfLogger(worker_id=f"R{self.dist_info.world_rank}-W{self.worker_id}") if self.enable_perf_log else None

        # 设置随机种子
        base_seed = 20251111 + self.worker_id * 97
        random.seed(base_seed)
        torch.manual_seed(base_seed)
        np.random.seed(base_seed % (2**32 - 1))

        # 为每个数据源创建一个LazyJsonlSource对象，并跳过历史位置
        for src_id in self.src_ids:
            modal = self.src_id2ratios[src_id][1]
            file_idx = self.workers2history_dict[self.worker_id][src_id]["file_idx"]
            start_skip = self.workers2history_dict[self.worker_id][src_id]["line_id"]
            src = _LazyJsonlSource(
                path_list=self.src_id2pathes[src_id],
                start_skip=start_skip,
                is_cycle=self.is_cycle,
                skip_ratio=self.skip_ratio,  # 按比例随机跳过某系行
                shard_id=self.worker_id,
                num_shards=self.num_shards,
                file_idx=file_idx,
                data_file_divisor=self.data_file_divisor,
            )
            self._sources.append((src_id, iter(src), modal))

        # self._sources 是 [(src_id, iter, modal), ...]

        # 构建模态 -> 数据源列表 的映射
        self._sources_by_modal = defaultdict(list)
        for item in self._sources:
            src_id, src_iter, modal = item
            self._sources_by_modal[modal].append(item)

        # 同时记录每个模态对应的采样权重（如果 self.ratios 存在）
        if hasattr(self, "ratios") and self.ratios:
            # 假设 self.ratios 与 self._sources 顺序一一对应
            self._modal_ratios = defaultdict(list)
            self._modal_sources_list = defaultdict(list)  # 保持顺序，用于对齐 ratios

            for item, weight in zip(self._sources, self.ratios):
                modal = item[2]
                self._modal_sources_list[modal].append(item)
                self._modal_ratios[modal].append(weight)
        else:
            # 无权重，等概率
            self._modal_ratios = None

        self._inited = True

    def _fetch_raw_jsons(self, needed_modality, count):
        items = []
        # === 加锁：防止多个 Producer 线程同时操作 generator ===
        with self.fetch_lock:
            for _ in range(count):
                try:
                    # 1. 检查所需模态是否有可用源,如果没有直接break
                    # [WARN]: 如果数据模式没有采用cyclic，可能会陷入死锁
                    if needed_modality not in self._sources_by_modal or not self._sources_by_modal[needed_modality]:
                        break
                    
                    # 2. 采样：选择一个 Source (src_id, iterator, modal)
                    if self.random_src:
                        # 优先使用带权重的采样 (如果在 __init__ 里初始化了 ratios)
                        if hasattr(self, '_modal_ratios') and self._modal_ratios and needed_modality in self._modal_ratios:
                            candidates = self._modal_sources_list[needed_modality]
                            cand_weights = self._modal_ratios[needed_modality]
                            # random.choices 返回的是 list，取 [0]
                            # 注意：这里 candidates 和 weights 的长度必须对齐，由 __init__ 保证
                            src_id, src_iter, modal = random.choices(candidates, weights=cand_weights, k=1)[0]
                        else:
                            # 无权重，纯随机
                            modal_sources = self._sources_by_modal[needed_modality]
                            src_id, src_iter, modal = random.choice(modal_sources)
                    else:
                        # 轮询模式
                        modal_sources = self._sources_by_modal[needed_modality]
                        src_id, src_iter, modal = random.choice(modal_sources)
                    
                    # 3. 执行 next()，获取数据
                    # 如果 generator 耗尽，会在这里抛出 StopIteration
                    _t0_fetch = time.monotonic()
                    sample, file_idx, line_id = next(src_iter)
                    if self._perf:
                        self._perf.record("fetch_jsonl_next", time.monotonic() - _t0_fetch)
                    sample["src_name"] = src_id
                
                    # 4. 更新 Worker 的历史记录 (断点续训用)
                    self.workers2history_dict[self.worker_id][src_id]["file_idx"] = file_idx
                    self.workers2history_dict[self.worker_id][src_id]["line_id"] = line_id

                    # 5. 构造 data_state (扁平化的状态向量)
                    flat_hist = []
                    for sid in self.src_ids:
                        # 必须按 src_ids 的固定顺序拼接，否则 resume 会错位
                        pos = self.workers2history_dict[self.worker_id].get(sid, {"file_idx": 0, "line_id": 0})
                        flat_hist.append(pos["file_idx"])
                        flat_hist.append(pos["line_id"])
                    
                    # data_state 格式: [worker_id, file1, line1, file2, line2, ...]
                    data_state = [self.worker_id] + flat_hist

                    items.append((sample, needed_modality, data_state))

                except StopIteration:
                    # === 源耗尽清理逻辑 ===

                    # 1. 从主列表移除
                    self._sources = [(ii, it, mm) for (ii, it, mm) in self._sources if not (ii == src_id and it is src_iter)]
                    
                    # 2. 从模态映射表移除
                    if hasattr(self, '_sources_by_modal'):
                        self._sources_by_modal[modal] = [x for x in self._sources_by_modal[modal] if not (x[0] == src_id and x[1] is src_iter)]
                    
                    # 这里我们假设模态永远不会耗尽，所以不做多余处理
                    pass

                    # 当前这次 count 没取到数据，continue 继续尝试取下一个源
                    continue
                
                except Exception as e:
                    print(f"[Fetch Error in Worker {self.worker_id}] {e}")
                    traceback.print_exc()
                    continue
                
        return items

    def _process_and_dispatch(self, args, modality, target_q):
        """
        [新增函数] 在子线程中运行：生成数据 -> 校验 -> 直接 Put 进队列
        这样主线程就不需要 acquire GIL 来做搬运工了。
        """
        sample_obj, _, data_state = args
        max_size = target_q.maxsize
        try:
            # 1. 执行繁重的 IO/CPU 生成工作
            # 注意：这里我们强制把 generator 转为 list，确保在线程内执行完毕
            results = list(self._build_out(sample_obj, modality))
            
            # 2. 直接在线程内分发数据 (Distributed Putting)
            for s in results:
                # --- 基础校验逻辑 (从原代码迁移过来) ---
                if modality == "text_to_audio" and s.get("audio_latents") is None: continue
                if modality == "text_to_video" and (s["video_latents"] is None or s["video_latents"].shape[-1] == 1):
                    print(f"[Warn] {modality} return bad data, skip.")
                    continue
                if modality == "text_to_image" and (s["image_latents"] is None or s["image_latents"].shape[-1] == 1):
                    print(f"[Warn] {modality} return bad data, skip.")
                    continue
                if modality == "text_to_av" and (s["audio_latents"] is None or s["video_latents"] is None or s["video_latents"].shape[-1] == 1):
                    print(f"[Warn] {modality} return bad data, skip.")
                    continue
                
                # 附加状态信息
                s["data_state"] = torch.tensor(data_state, dtype=torch.long)
                
                target_q.put(s)
                
                    
        except Exception as e:
            pass

    def _single_modality_producer_loop(self, modality):
        """
        单模态生产者线程：只负责生产指定 modality 的数据。
        """
        print(f"[DEBUG] Worker {self.worker_id}: Producer for [{modality}] STARTED")
        
        try:
            # 1. 错峰启动
            time.sleep(random.uniform(0.1, 2.0))
            self._init_sources_once()

            target_queue = self.modality_queues[modality]
            BASE_WORKERS = self.io_workers
            MAX_WINDOW = int(BASE_WORKERS * 2.0) 

            flying_futures = []

            while True:
                # ==========================
                # A. 优先处理已完成的任务，释放出队列压力
                # ==========================
                done, not_done = wait(flying_futures, timeout=0.01, return_when=FIRST_COMPLETED)
                for f in done:
                    results, data_state, err = f.result()
                    if err is not None:
                        print(f"Error in future: {err}")
                        continue
                        
                    for s in results:
                        # 校验逻辑
                        if modality == "text_to_audio" and s.get("audio_latents") is None: continue
                        if modality == "text_to_video" and (s.get("video_latents") is None or s["video_latents"].shape[-1] == 1): continue
                        if modality == "text_to_image" and (s.get("image_latents") is None or s["image_latents"].shape[-1] == 1): continue
                        if modality == "text_to_av" and (s.get("audio_latents") is None or s.get("video_latents") is None or s["video_latents"].shape[-1] == 1): continue
                        
                        s["data_state"] = torch.tensor(data_state, dtype=torch.long)

                        # 【核心修复】：阻塞转移到模态的主线程，彻底释放 io_pool 资源
                        _t0_qput = time.monotonic()
                        target_queue.put(s)
                        _elapsed_qput = time.monotonic() - _t0_qput
                        if self._perf and _elapsed_qput > 0.01:
                            self._perf.record(f"producer_q_put_wait_{modality}", _elapsed_qput)
                        
                flying_futures = list(not_done)

                # ==========================
                # B. 检查队列是否满 (背压)
                # ==========================
                curr_size = target_queue.qsize()
                max_size = target_queue.maxsize
                pending_count = len(flying_futures)
                
                if curr_size + pending_count >= max_size:
                    time.sleep(0.02)
                    continue
                
                # ==========================
                # C. 填补窗口
                # ==========================
                raw_items = self._fetch_raw_jsons(modality, count=1)
                if not raw_items: 
                    # 【核心修复2】: 耗尽时，必须等待剩余任务完成并入队，否则丢失尾部数据
                    for f in as_completed(flying_futures):
                        results, data_state, err = f.result()
                        if err is not None: continue
                        for s in results:
                            if s.get("captions") is None: continue
                            s["data_state"] = torch.tensor(data_state, dtype=torch.long)
                            target_queue.put(s)
                    break # 跳出 while，当前模态读取完毕
                
                item = raw_items[0]
                # 注意：这里改用 _process_item_concurrently
                fut = self.io_pool.submit(self._process_item_concurrently, item)
                flying_futures.append(fut)

                if self._perf:
                    self._perf.maybe_print()

        except Exception as e:
            print(f"!!! FATAL ERROR IN PRODUCER [{modality}] (Worker {self.worker_id}): {e}")
            traceback.print_exc()
        finally:
            # 【核心修复3】：正常或异常退出时，必须发 None 信号，否则消费者 get() 永久死锁
            target_queue.put(None)
    
    def _single_modality_producer_loop_old(self, modality):
        """
        单模态生产者线程：只负责生产指定 modality 的数据。
        """
        print(f"[DEBUG] Worker {self.worker_id}: Producer for [{modality}] STARTED")
        
        try:
            # 1. 错峰启动
            time.sleep(random.uniform(0.1, 2.0))
            self._init_sources_once()

            # 2. 是否同步
            synchronous = False
            target_queue = self.modality_queues[modality]

            # 3. 动态窗口参数 (仅针对 heavy IO)
            BASE_WORKERS = self.io_workers
            # 如果有多个 heavy producer (例如同时有图和视频)，需要共享 worker 配额
            MAX_WINDOW = int(BASE_WORKERS * 2.0) 

            flying_futures = []

            while True:
                # ==========================
                # A. 检查队列是否满 (背压)
                # ==========================
                curr_size = target_queue.qsize()
                max_size = target_queue.maxsize
                if flying_futures:
                    flying_futures = [f for f in flying_futures if not f.done()]
                pending_count = len(flying_futures)
                
                # 核心逻辑：只有当 (库存 + 在途) < 总容量 时，才允许新生产。
                # 这样保证了每一个发出的任务，回来时都有“停车位”，绝对不会卡住。
                if curr_size + pending_count >= max_size:
                    # 此时已经饱和，必须休息。
                    # 由于 Worker 不会阻塞，它们会很快完成并释放资源。
                    time.sleep(0.02)
                    continue
                
                # 2. 填补窗口
                raw_items = self._fetch_raw_jsons(modality, count=1)
                if not raw_items: break
                
                item = raw_items[0]
                # 提交任务
                fut = self.io_pool.submit(self._process_and_dispatch, item, modality, target_queue)
                flying_futures.append(fut)

                if len(flying_futures) > MAX_WINDOW:
                    done, _ = wait(flying_futures, return_when=FIRST_COMPLETED)
                    flying_futures = [f for f in flying_futures if not f.done()]
                
                # 3. 收割结果 (Sliding Window)
                if not flying_futures:
                    time.sleep(0.01)
                    continue

        except Exception as e:
            print(f"!!! FATAL ERROR IN PRODUCER [{modality}] (Worker {self.worker_id}): {e}")
            traceback.print_exc()
            target_queue.put(None)        
        
    def _process_item_concurrently(self, args):
        """
        这个函数会在子线程运行。
        必须在这里把 generator 消耗完 (list化)，否则 IO 操作可能延迟到主线程迭代时才发生。
        """
        sample_obj, modal, data_state = args
        try:
            # 调用 _build_out (包含 BOS 下载/VAE 请求)
            # 关键：使用 list() 强制立即执行生成器里的代码
            _t0_build = time.monotonic()
            results = list(self._build_out(sample_obj, modal))
            if self._perf:
                self._perf.record(f"build_out_{modal}", time.monotonic() - _t0_build)
                for r in results:
                    if isinstance(r, dict):
                        for k in ("audio_latents", "video_latents", "image_latents"):
                            if k in r and r[k] is not None and hasattr(r[k], "element_size"):
                                self._perf.record(f"tensor_bytes_{k}", r[k].element_size() * r[k].numel())
            return results, data_state, None  # None 代表无异常
        except Exception as e:
            # 捕获异常，避免单个样本搞挂整个线程池
            return [], data_state, e
        
    # --- 辅助：根据 latent 长度计算所在桶编号 ---
    def _get_length_bucket_id(self, length, boundaries):
        """根据 length 和预计算的 boundaries 返回桶编号 (0 ~ num_length_buckets-1)"""
        idx = int(np.searchsorted(boundaries, length, side='right')) - 1
        return max(0, min(idx, self.num_length_buckets - 1))

    # --- 辅助：分发单个样本到桶 ---
    def _distribute_single_item(self, out):
        """分发单个样本到桶。返回 True 已放入，False 表示目标桶已满（调用方应 put back Queue）。"""
        # 音/图/视频
        data_mode = None
        if "audio_latents" in out and out["audio_latents"] is not None:
            if "video_latents" in out and out["video_latents"] is not None:
                data_mode = "text_to_av"
            else:
                data_mode = "text_to_audio"
        elif "video_latents" in out and out["video_latents"] is not None:
            data_mode = "text_to_video"
        elif "image_latents" in out and out["image_latents"] is not None:
            data_mode = "text_to_image"

        if data_mode and data_mode in self.modality_mappers:
            if self.use_length_buckets and data_mode in self.length_buckets_mapper:
                # 按 latent 长度分桶
                if data_mode == "text_to_audio":
                    seq_len = out["audio_latents"].shape[0]
                    bucket_id = self._get_length_bucket_id(seq_len, self._aud_bucket_boundaries)
                elif data_mode == "text_to_video":
                    seq_len = out["video_latents"].shape[0]
                    bucket_id = self._get_length_bucket_id(seq_len, self._vid_bucket_boundaries)
                else:  # text_to_av — 以视频 latent 长度为主
                    seq_len = out["video_latents"].shape[0]
                    bucket_id = self._get_length_bucket_id(seq_len, self._vid_bucket_boundaries)
                bucket = self.length_buckets_mapper[data_mode][bucket_id]
                if self._per_bucket_cap > 0 and len(bucket) >= self._per_bucket_cap:
                    return False  # 桶满，拒收
                bucket.append(out)
            elif self.use_aspect_ratio_buckets:
                aspect_ratio = str(out.get('aspect_ratio', '1.0'))
                if '.' in aspect_ratio and len(aspect_ratio) < 10: pass
                self.aspect_ratio_buckets_mapper[data_mode][aspect_ratio].append(out)
            else:
                self.modality_buffers_mapper[data_mode].append(out)
        return True

    def _buckets_need_refill(self, mod):
        """检查是否还需要从 Queue 拉数据。
        put-back 机制保证单桶不溢出，所以这里只在所有已有桶都满时才停止拉取。
        此时 Queue 里全是满桶 item，继续拉也只会 put back，浪费 CPU。
        """
        if not self.use_length_buckets or mod not in self.length_buckets_mapper or self._per_bucket_cap <= 0:
            return True
        mode_buckets = self.length_buckets_mapper[mod]
        if not mode_buckets:
            return True
        has_ready = any(len(b) >= self.batch_size for b in mode_buckets.values())
        if not has_ready:
            return True  # 没有桶能出 batch，必须继续拉
        # 所有已有桶（有数据的）都满了 → 停止
        non_empty = [b for b in mode_buckets.values() if len(b) > 0]
        if non_empty and all(len(b) >= self._per_bucket_cap for b in non_empty):
            return False
        return True

    def _check_buckets_and_pop_packed(self, target_modality):
        bs = self.batch_size

        # 长度分桶模式：按桶内样本数量做加权随机采样
        if self.use_length_buckets and target_modality in self.length_buckets_mapper:
            mode_buckets = self.length_buckets_mapper[target_modality]

            if self.enable_ddp_bucket_sync and dist.is_initialized():
                # --- DDP 同步路径 ---
                # 协议：每次调用固定 1 次 broadcast + 1 次 all_reduce，保证不 hang。

                buf = self._bucket_sync_buf
                if not buf.is_cuda:
                    buf = buf.cuda()
                    self._bucket_sync_buf = buf

                # Step 1: rank 0 按数量比例加权随机选桶，broadcast
                if dist.get_rank() == 0:
                    ready_keys = [k for k, v in mode_buckets.items() if len(v) >= bs]
                    if ready_keys:
                        weights = [len(mode_buckets[k]) for k in ready_keys]
                        chosen_key = random.choices(ready_keys, weights=weights, k=1)[0]
                        buf[0] = chosen_key
                    else:
                        buf[0] = -1
                dist.broadcast(buf, src=0)
                chosen = buf[0].item()

                # Step 2: 各 rank 判断自己能否出 batch
                local_result = None
                if chosen >= 0 and chosen in mode_buckets and len(mode_buckets[chosen]) >= bs:
                    local_result = chosen
                else:
                    # fallback: 本地按比例随机选
                    ready_keys = [k for k, v in mode_buckets.items() if len(v) >= bs]
                    if ready_keys:
                        weights = [len(mode_buckets[k]) for k in ready_keys]
                        local_result = random.choices(ready_keys, weights=weights, k=1)[0]

                # Step 3: all_reduce MIN 同步"所有 rank 都能出 batch"
                can_serve = torch.tensor([1 if local_result is not None else 0],
                                         dtype=torch.long, device=buf.device)
                dist.all_reduce(can_serve, op=dist.ReduceOp.MIN)

                if can_serve.item() == 0:
                    return None

                return [mode_buckets[local_result].popleft() for _ in range(bs)]
            else:
                # --- 非同步路径：本地按数量比例加权随机选桶 ---
                ready_keys = [k for k, v in mode_buckets.items() if len(v) >= bs]
                if ready_keys:
                    weights = [len(mode_buckets[k]) for k in ready_keys]
                    chosen = random.choices(ready_keys, weights=weights, k=1)[0]
                    return [mode_buckets[chosen].popleft() for _ in range(bs)]
                return None

        bucket = self.modality_buffers_mapper[target_modality]  # deque
        if len(bucket) < bs:
            return None
        # O(bs)，与 bucket 总长度无关
        return [bucket.popleft() for _ in range(bs)]

    def _monitor_loop(self):
        """
        监控线程：打印所有模态队列的水位，以及消费者端的分桶积压情况。
        """
        import time
        time.sleep(5) # 启动后先等一会
        
        while True:
            time.sleep(30) # 每 30 秒打印一次
            
            wid = getattr(self, 'worker_id', '?')

            # === 1. 监控各模态 Queue (生产者进度) ===
            # 这是 Worker 线程（Producer）产出后暂存的地方
            queue_stats = []
            if hasattr(self, "modality_queues"):
                for m_name, q in self.modality_queues.items():
                    queue_stats.append(f"{m_name}={q.qsize()}")
            else:
                queue_stats.append("No_Queues")
            
            queue_str = " | ".join(queue_stats)
            
            # === 2. 监控 Buckets (消费者组装进度) ===
            # 这是 Consumer 取出来后，分类暂存等待凑 Batch 的地方
            # 我们需要详细看到每个模态的积压情况
            
            buf_stats = []
            
            # 音频/图像/视频桶 (区分是否开启长度分桶 或 AR 分桶)
            if self.use_length_buckets:
                # 长度分桶模式：打印各模态各桶的积压
                for m_name, mode_buckets in self.length_buckets_mapper.items():
                    total = sum(len(b) for b in mode_buckets.values())
                    detail = ",".join(f"b{k}:{len(v)}" for k, v in sorted(mode_buckets.items()) if len(v) > 0)
                    buf_stats.append(f"{m_name}={total}({detail})")
                # image 走普通 buffer
                if "text_to_image" in self.modality_buffers_mapper:
                    buf_stats.append(f"text_to_image={len(self.modality_buffers_mapper['text_to_image'])}")
            elif self.use_aspect_ratio_buckets:
                # 遍历 mapper 下的所有模态 (text_to_image, image_und...)
                for m_name, mode_buckets in self.aspect_ratio_buckets_mapper.items():
                    # 计算该模态下所有 AR 子桶的总和
                    total = sum(len(b) for b in mode_buckets.values())
                    buf_stats.append(f"{m_name}={total}")
            else:
                # 普通模式：直接遍历 mapper
                for m_name, b in self.modality_buffers_mapper.items():
                    buf_stats.append(f"{m_name}={len(b)}")

            buf_str = " | ".join(buf_stats)

            # === 3. 打印 ===
            # Q: Queue 水位 (Producer)
            # Buf: Bucket 积压 (Consumer)
            print(f"[MONITOR W-{wid}] Q:[{queue_str}] | Buf:[{buf_str}]")
            if self._perf:
                self._perf.maybe_print(force=True)

    def _refill_buffer_nonblocking(self, mod, cap=256):
        """非阻塞补货：有多少拿多少，绝不等 Queue。桶满的 item 放回 Queue。"""
        if not self._buckets_need_refill(mod):
            return 0
        q = self.modality_queues[mod]
        moved = 0
        putback_streak = 0
        for _ in range(cap):
            if not self._buckets_need_refill(mod):
                break
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                return -1  # 终止信号
            if not self._distribute_single_item(item):
                q.put(item)  # 桶满，放回 Queue
                putback_streak += 1
                if putback_streak >= 3:
                    break  # Queue 里大多是满桶 item，停止白转
            else:
                putback_streak = 0
                moved += 1
        return moved

    def _refill_buffer_to_target(self, mod, target, cap):
        """把 mod 的 buffer 补到 target（最多搬 cap 个）。桶满的 item 放回 Queue。"""
        q = self.modality_queues[mod]

        if not self._buckets_need_refill(mod):
            return 0

        # 根据分桶模式选择正确的水位统计
        if self.use_length_buckets and mod in self.length_buckets_mapper:
            mode_buckets = self.length_buckets_mapper[mod]
            cur = sum(len(b) for b in mode_buckets.values())
            has_ready_bucket = any(len(b) >= self.batch_size for b in mode_buckets.values())
            if cur >= target and has_ready_bucket:
                return 0
        else:
            cur = len(self.modality_buffers_mapper[mod])
            if cur >= target:
                return 0

        need = max(1, min(target - cur, cap))
        moved = 0

        # 1) 先阻塞拿 1 个，保证推进
        _t0_bget = time.monotonic()
        item = q.get()
        if self._perf:
            self._perf.record(f"consumer_q_get_blocking_{mod}", time.monotonic() - _t0_bget)
        if item is None:
            return -1  # 终止信号
        if not self._distribute_single_item(item):
            q.put(item)  # 桶满，放回
        else:
            moved += 1
        need -= 1

        # 2) 再非阻塞补齐，水位够了就停
        putback_streak = 0
        for _ in range(need):
            if not self._buckets_need_refill(mod):
                break
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                return -1
            if not self._distribute_single_item(item):
                q.put(item)
                putback_streak += 1
                if putback_streak >= 3:
                    break
            else:
                putback_streak = 0
                moved += 1

        return moved


    def __iter__(self):
        """实现流式迭代"""
        # 1. 确保后台线程只启动一次
        if not self._producer_running:
            self._producer_running = True

            # 遍历权重不为0的所有模态，为每个模态启动一个线程
            for modality in self.modality_queues.keys():
                t = threading.Thread(
                    target=self._single_modality_producer_loop, 
                    args=(modality,), # 传参
                    daemon=True
                )
                t.start()
                self.producer_threads.append(t)

            # 监控线程，debug时使用
            t_mon = threading.Thread(target=self._monitor_loop, daemon=True)
            t_mon.start()

        produced = 0
        batch_size = self.batch_size
        target = batch_size * 16
        cap = 256

        modalities = sorted(list(self.modality_queues.keys()))  
        weights = [self.modality_configs[m]["weight"] for m in modalities]
        
        # 2. 获取当前 Local Worker ID
        worker_info = torch.utils.data.get_worker_info()
        local_worker_id = worker_info.id if worker_info is not None else 0
        
        # 3. 创建专属的随机数生成器。
        # 这样不同 Rank 的 Worker 0 会生成相同的模态序列，Worker 1 也会生成相同的序列
        # modality_rng = random.Random(2026 + local_worker_id)
        current_seed = 2026 + local_worker_id * 1000 + self._local_iter_count
        modality_rng = random.Random(current_seed)

        self._local_iter_count += 1

        # # 提前提取模态名和权重
        # # modalities = list(self.modality_configs.keys())
        # # weights = [self.modality_configs[m]["weight"] for m in modalities]
        # modalities = list(self.modality_queues.keys())  # ✅ 只抽有队列的模态
        # weights = [self.modality_configs[m]["weight"] for m in modalities]
  
        # 主循环：只管从队列拿，不管怎么生产
        _iter_batch_count = 0
        _iter_t0 = time.monotonic()
        while produced < self.upper_count:
            _batch_assemble_t0 = time.monotonic()
            target_modality = modality_rng.choices(modalities, weights=weights, k=1)[0]

            batch_data = None
            target_q = self.modality_queues[target_modality]

            while batch_data is None:
                # 1) 先查桶 —— buffer 有货就直接出，不阻塞等 Queue
                start_time = time.time()
                _t0_buf_hit = time.monotonic()
                batch_data = self._check_buckets_and_pop_packed(target_modality)
                if batch_data:
                    if self._perf:
                        self._perf.record("batch_from_buffer_hit", time.monotonic() - _t0_buf_hit)
                    # ★ 非阻塞补货：顺手把 Queue 里已有的搬到 buffer，绝不等
                    _t0_refill = time.monotonic()
                    moved = self._refill_buffer_nonblocking(target_modality, cap=cap)
                    if self._perf:
                        self._perf.record(f"refill_buffer_{target_modality}", time.monotonic() - _t0_refill)
                    break

                # 2) buffer 空了，必须从 Queue 补货（阻塞）
                start_time_1 = time.time()
                buf_len = len(self.modality_buffers_mapper[target_modality])
                _t0_refill2 = time.monotonic()
                moved = self._refill_buffer_to_target(target_modality, target=target, cap=cap)
                if self._perf:
                    self._perf.record(f"refill_buffer_{target_modality}", time.monotonic() - _t0_refill2)
                if moved == -1:
                    return

                # 3) 补完再 pop（补货的目标就是让这里大概率成功）
                start_time_2 = time.time()
                batch_data = self._check_buckets_and_pop_packed(target_modality)
                if batch_data:
                    end_time_2 = time.time()
                    produced += self.batch_size
                    continue

                # 4) 兜底：buffer 还不够（queue 太空/生产跟不上），阻塞再拿 1 个再试
                #    如果桶水位已经够了（DDP 同步失败导致的重试），不要继续塞数据
                if not self._buckets_need_refill(target_modality):
                    time.sleep(0.01)  # 等其他 rank 追上
                    continue
                _t0_fallback = time.monotonic()
                item = self.modality_queues[target_modality].get()
                if self._perf:
                    self._perf.record(f"consumer_fallback_get_{target_modality}", time.monotonic() - _t0_fallback)
                if item is None:
                    return
                if not self._distribute_single_item(item):
                    self.modality_queues[target_modality].put(item)  # 桶满，放回

            # batch_assemble_total timing
            if self._perf:
                self._perf.record("batch_assemble_total", time.monotonic() - _batch_assemble_t0)

            # 5) yield
            produced += self.batch_size
            _iter_batch_count += 1

            # 每 100 个 batch 打印迭代吞吐率
            if self._perf and _iter_batch_count % 100 == 0:
                _iter_elapsed = time.monotonic() - _iter_t0
                print(f"[DATA_PERF {self._perf.worker_id}] iter throughput: "
                      f"{_iter_batch_count} batches in {_iter_elapsed:.1f}s "
                      f"({_iter_batch_count / _iter_elapsed:.2f} batch/s)", flush=True)
                self._perf.maybe_print()

            yield batch_data

    def _build_out_aud(self, obj: dict):
        samples = []
        src_name = obj["src_name"]
        data_type = src_name.split("_")[-1]
        try:
            if "audio_splits_info" in obj:
                audio_splits_info = obj["audio_splits_info"]
                text_list = obj["text_list"]
            elif "video_info" in obj:
                audio_splits_info = obj["video_info"]
                text_list = obj["text_list"][:1]
            else:
                raise KeyError("Neither audio_splits_info nor video_info found in obj.")
            # text_list = obj["text_list"]
            assert len(text_list) == len(audio_splits_info), (
                f"Length of text_list ({len(text_list)}) must equal "
                f"length of audio_splits_info ({len(audio_splits_info)})."
            )
            for sample_audio_info, sample_text_info in zip(audio_splits_info, text_list):
                # construct audio caption / speech content
                text = sample_text_info["text"]
                is_audio_split = "audio_info_idx" in sample_text_info
                
                start = sample_text_info.get("media_start", None)
                end = sample_text_info.get("media_end", None)
                speech_starts = sample_text_info.get("speech_start", None)
                speech_ends = sample_text_info.get("speech_end", None)
                is_valid = sample_text_info.get("is_valid", True)
                
                # encode audio
                sample_spk_embs = []
                data_path = sample_audio_info["data_path"]
                duration = sample_audio_info.get("duration", None)
                is_valid = sample_audio_info.get("is_valid", is_valid)
                if not is_valid or duration > self.max_audio_duration or duration < self.min_audio_duration:
                    continue
                
                if data_type == "tts":
                    if "<S>" not in text:
                        text = "<S>" + text + "<E>"
                elif speech_starts and speech_ends:
                    num_speech_starts = text.count("<S>")
                    num_speech_ends = text.count("<E>")
                    assert num_speech_starts == num_speech_ends and num_speech_ends == len(speech_starts), \
                        f"Error: starts {num_speech_starts} not match with ends {num_speech_ends}, text: {text}"
                if self.add_spk_emb:
                    if "<S>" in text and "<E>" in text:
                        text = text.replace("<S>", "<S><extra_id_2>")
                if self.use_speech_special_token:
                    text = text.replace("<S>", "<extra_id_0>")
                    text = text.replace("<E>", "<extra_id_1>")
                assert self.audio_vae_server, "audio server is not ready."
                query = {
                    "data_path": data_path,
                    "use_spk_emb": (self.add_spk_emb and random.random() < self.spk_emb_prob) and data_type == "tts",
                }
                _t0_aud_enc = time.monotonic()
                result = self.audio_vae_server.encode(
                    query, rank=self.dist_info.world_rank,
                ).latent_dist.sample()
                if self._perf:
                    self._perf.record("vae_encode_audio", time.monotonic() - _t0_aud_enc)
                audio_latents = result["audio_latents"][0].permute(1, 0) # [L, 20]
                if self.add_spk_emb and data_type == "tts":
                    sample_spk_embs.append(result["spk_embs"]) # [b, d]
                if self.add_spk_emb and speech_starts and speech_ends:
                    for speech_start, speech_end in zip(speech_starts, speech_ends):
                        query.update({
                            "start": speech_start,
                            "duration": speech_end - speech_start,
                            "use_spk_emb": self.add_spk_emb and random.random() < self.spk_emb_prob
                        })
                        # print(query, self.spk_emb_prob, 111)
                        _t0_spk_enc = time.monotonic()
                        result = self.audio_vae_server.encode(
                            query, rank=self.dist_info.world_rank,
                        ).latent_dist.sample()
                        if self._perf:
                            self._perf.record("vae_encode_audio_spk", time.monotonic() - _t0_spk_enc)
                        sample_spk_embs.append(result["spk_embs"])

                sample = {
                    "captions": text,
                    "audio_latents": audio_latents,
                    "spk_embs": sample_spk_embs,
                }
                samples.append(sample)
            return samples
        except Exception as e:
            print(f"Error in building audio sample: {e}")
            # print(f"obj: {obj}")
            return []

    def _build_out_img(self, obj: dict):
        samples = []
        aspect_ratio = "0/0"
        try:
            image_info = obj["image_info"]
            text_list = obj["text_list"]
            assert len(image_info) == len(text_list), (
                f"Length of image_info ({len(image_info)}) must equal "
                f"length of text_list ({len(text_list)})."
            )
            for sample_image_info, sample_text_info in zip(image_info, text_list):
                text = sample_text_info["text"]
                is_valid = sample_text_info.get("is_valid", True)

                data_path = sample_image_info["data_path"]
                image_height = sample_image_info["image_height"]
                image_width = sample_image_info["image_width"]
                is_valid = sample_image_info.get("is_valid", is_valid)

                if not is_valid: # invalid sample
                    continue
                assert self.image_vae_server, "image server is not ready."
                _t0_img_enc = time.monotonic()
                image_latents = self.image_vae_server.encode(
                    data_path, rank=self.dist_info.world_rank
                ).latent_dist.sample()
                if self._perf:
                    self._perf.record("vae_encode_image", time.monotonic() - _t0_img_enc)
                if self.use_aspect_ratio_buckets:
                    aspect_ratio = compute_tgt_ratio(image_height, image_width)
                    _, h, w, _ = image_latents.shape
                    w_ratio, h_ratio = [int(i) for i in aspect_ratio.split("/")]
                    if (h < w and h_ratio >= w_ratio) or (h > w and h_ratio <= w_ratio):
                        aspect_ratio = f"{h_ratio}/{w_ratio}"
                sample = {
                    "captions": text,
                    "image_latents": image_latents,
                    "aspect_ratio": aspect_ratio,
                }
                samples.append(sample)
            return samples

        except Exception as e:
            print(f"Error in building image sample: {e}")
            print(f"obj: {obj}")
            return []

    def _build_out_vid(self, obj: dict):
        samples = []
        aspect_ratio = "0, 0/0"
        try:
            video_info = obj["video_info"]
            text_list = obj["text_list"]
            assert len(video_info) == len(text_list), (
                f"Length of video_info ({len(video_info)}) must equal "
                f"length of text_list ({len(text_list)})."
            )
            for sample_video_info, sample_text_info in zip(video_info, text_list):
                text = sample_text_info["text"]
                # text = filter_video_descriptions(text)
                is_valid = sample_text_info.get("is_valid", True)
                
                data_path = sample_video_info["data_path"]
                video_height = sample_video_info["image_height"]
                video_width = sample_video_info["image_width"]
                video_duration = sample_video_info["duration"]
                video_frames = int(video_duration * self.video_fps)
                is_valid = sample_video_info.get("is_valid", is_valid)
                if not is_valid:
                    print(f"invalid video sample: {obj}")
                    continue

                if video_frames < self.video_min_frames or video_frames > self.video_max_frames:
                    print(f"skip video due to frame number {video_frames}, \
                          not in [{self.video_min_frames}, {self.video_max_frames}]")
                    continue

                if self.use_aspect_ratio_buckets:
                    aspect_ratio = compute_tgt_ratio(video_height, video_width)

                assert self.video_vae_server, "video server is not ready."
                _t0_vid_enc = time.monotonic()
                video_latents = self.video_vae_server.encode(
                    data_path,
                    rank=self.dist_info.world_rank,
                    frame_length=self.video_tgt_frames,
                    fps=self.video_fps
                ).latent_dist.sample()
                if self._perf:
                    self._perf.record("vae_encode_video", time.monotonic() - _t0_vid_enc)
                sample = {
                    "captions": text,
                    "video_latents": video_latents,
                    "aspect_ratio": aspect_ratio,
                }
                samples.append(sample)
            return samples
        except Exception as e:
            print(f"Error in building video sample: {e}")
            traceback.print_exc()
            return []

    def _build_out_av(self, obj: dict):
        # notes: fake av data with video data
        samples = []
        aspect_ratio = "0, 0/0"
        try:
            video_info = obj["video_info"]
            text_list = obj["text_list"]
            assert len(video_info) == len(text_list), (
                f"Length of video_info ({len(video_info)}) must equal "
                f"length of text_list ({len(text_list)})."
            )
            for s_idx, (sample_video_info, sample_text_info) in enumerate(zip(video_info, text_list)):
                text = sample_text_info["text"]
                # text = filter_video_descriptions(text)
                is_valid = sample_text_info.get("is_valid", True)

                sample_spk_embs = []
                speech_starts = sample_text_info.get("speech_start", None)
                speech_ends = sample_text_info.get("speech_end", None)
                if speech_starts and speech_ends:
                    num_speech_starts = text.count("<S>")
                    num_speech_ends = text.count("<E>")
                    assert num_speech_starts == num_speech_ends and num_speech_ends == len(speech_starts), \
                        f"Error: starts {num_speech_starts} not match with ends {num_speech_ends}, text: {text}"
                if self.add_spk_emb:
                    if "<S>" in text and "<E>" in text:
                        text = text.replace("<S>", "<S><extra_id_2>")
                if self.use_speech_special_token:
                    text = text.replace("<S>", "<extra_id_0>")
                    text = text.replace("<E>", "<extra_id_1>")

                data_path = sample_video_info["data_path"]
                if self.split_wav_mode:
                    if "audio_info" not in obj:
                        print("audio_info not found in obj for split_wav_mode")
                        continue
                    audio_info = obj["audio_info"]
                    audio_data_path = audio_info[s_idx]["data_path"]
                else:
                    audio_data_path = data_path
                video_height = sample_video_info["image_height"]
                video_width = sample_video_info["image_width"]
                video_duration = sample_video_info["duration"]
                video_frames = int(video_duration * self.video_fps)
                is_valid = sample_video_info.get("is_valid", is_valid)
                if not is_valid:
                    print(f"invalid video sample: {obj}")
                    continue

                if video_frames < self.video_min_frames or video_frames > self.video_max_frames:
                    print(f"skip video due to frame number {video_frames}, not in [{self.video_min_frames}, {self.video_max_frames}]")
                    continue

                if self.use_aspect_ratio_buckets:
                    aspect_ratio = compute_tgt_ratio(video_height, video_width)
                
                assert self.audio_vae_server and self.video_vae_server, "video server or audio server is not ready."

                _t0_av_vid_enc = time.monotonic()
                video_latents = self.video_vae_server.encode(
                    data_path,
                    rank=self.dist_info.world_rank,
                    frame_length=self.video_tgt_frames,
                    fps=self.video_fps
                ).latent_dist.sample()
                if self._perf:
                    self._perf.record("vae_encode_video_av", time.monotonic() - _t0_av_vid_enc)

                # 对齐视频音频 这里假设视频从一开始取
                video_duration = ((video_latents.shape[0] - 1) * 4 + 1) / self.video_fps
                audio_length = math.ceil(video_duration * self.audio_tokens_per_sec)

                query = {
                    "data_path": audio_data_path,
                    "add_spk_emb": False,
                    "target_length": audio_length,
                }
                _t0_av_aud_enc = time.monotonic()
                audio_result = self.audio_vae_server.encode(
                    query, rank=self.dist_info.world_rank,
                ).latent_dist.sample()
                if self._perf:
                    self._perf.record("vae_encode_audio_av", time.monotonic() - _t0_av_aud_enc)
                audio_latents = audio_result["audio_latents"][0].permute(1, 0)
                if self.add_spk_emb and speech_starts and speech_ends:
                    for speech_start, speech_end in zip(speech_starts, speech_ends):
                        query.pop("target_length", None)
                        query.update({
                            "start": speech_start,
                            "duration": speech_end - speech_start,
                            "use_spk_emb": self.add_spk_emb and random.random() < self.spk_emb_prob
                        })
                        _t0_av_spk_enc = time.monotonic()
                        result = self.audio_vae_server.encode(
                            query, rank=self.dist_info.world_rank,
                        ).latent_dist.sample()
                        if self._perf:
                            self._perf.record("vae_encode_audio_spk", time.monotonic() - _t0_av_spk_enc)
                        sample_spk_embs.append(result["spk_embs"])

                # _t0_av_vid_enc = time.monotonic()
                # video_latents = self.video_vae_server.encode(
                #     data_path,
                #     rank=self.dist_info.world_rank,
                #     frame_length=self.video_tgt_frames,
                #     fps=self.video_fps
                # ).latent_dist.sample()
                # if self._perf:
                #     self._perf.record("vae_encode_video_av", time.monotonic() - _t0_av_vid_enc)

                # # 对齐视频音频 这里假设视频从一开始取
                # video_duration = ((video_latents.shape[0] - 1) * 4 + 1) / self.video_fps
                # audio_length = math.ceil(video_duration * self.audio_tokens_per_sec)
                if audio_length > audio_latents.shape[0]:
                    print(f"{audio_length}, {audio_latents.shape}, not equal !!!!, {data_path}")
                    audio_latents = torch.cat([audio_latents, torch.zeros(size=(audio_length-audio_latents.shape[0], audio_latents.shape[1]), device=audio_latents.device)], dim=0)
                else:
                    # print(f"{audio_length}, {audio_latents.shape}, is equal !!!!, {bos_url}")
                    audio_latents = audio_latents[:audio_length]

                samples = {
                    "captions": text,
                    "audio_latents": audio_latents,
                    "spk_embs": sample_spk_embs,
                    "video_latents": video_latents,
                    "aspect_ratio": aspect_ratio,
                }
            return samples
        except Exception as e:
            print(f"Error in building video sample: {e}")
            traceback.print_exc()
            return []


    def _build_out(self, obj: dict, chosen_modality: str):
        # 使用滑动窗口方法，规划 bsz
        """构建输出"""
        if chosen_modality == "text_to_audio":
            aud_sample = self._build_out_aud(obj)
            if isinstance(aud_sample, list):
                yield from aud_sample
            else:
                yield aud_sample
        elif chosen_modality == "text_to_image":
            img_sample = self._build_out_img(obj)
            if isinstance(img_sample, list):
                yield from img_sample
            else:
                yield img_sample
        elif chosen_modality == "text_to_video":
            vid_sample = self._build_out_vid(obj)
            if isinstance(vid_sample, list):
                yield from vid_sample
            else:
                yield vid_sample
        elif chosen_modality == "text_to_av":
            av_sample = self._build_out_av(obj)
            if isinstance(av_sample, list):
                yield from av_sample
            else:
                yield av_sample
        else:
            print(f"not support {chosen_modality}")
            raise NotImplementedError

def collate_fn(batch):
    out = {}
    process_keys = {
        "captions",
        "audio_latents",
        "image_latents",
        "video_latents",
        "spk_embs",
        "data_state",
        "audio_seq_len",
        "t_h_w_list",
    }
    for k in process_keys:
        vals = [b.get(k, None) for b in batch]
        if all(x is None for x in vals):
            vals = None
        out[k] = vals
    
    if out["audio_latents"]:
        out["audio_seq_len"] = [
            b["audio_latents"].shape[-1] if b["audio_latents"] is not None else 0 for b in batch
        ]
    if out["image_latents"]:
        out["t_h_w_list"] = [
            tuple(b["image_latents"].shape[:3]) if b["image_latents"] is not None else (0, 0, 0) for b in batch
        ]
    if out["video_latents"]:
        out["t_h_w_list"] = [
            tuple(b["video_latents"].shape[:3]) if b["video_latents"] is not None else (0, 0, 0) for b in batch
        ]
    # if out["spk_embs"]: # keep list
    #     out["spk_embs"] = torch.cat([b["spk_embs"] for b in batch], dim=0)

    out["data_state"] = torch.stack(out["data_state"], dim=0)

    return out


def collate_fn_batch(batchs):
    """
    批量处理函数，将多个批次数据分发到单个collate_fn函数处理
    """
    return [collate_fn(batch) for batch in batchs]


def main():
    import yaml
    import traceback

    cfg = yaml.safe_load(open("configs/nava.yaml", "r"))
    device = "cuda"

    from nava_src.models.nava.utils.model_loading_utils import init_wan_vae_2_2
    from nava_src.vae.local_video_vae import LocalVideoVAEAdapter
    import torch
    wan_vae = init_wan_vae_2_2(cfg["model"]["ckpt_dir"], rank=device)
    wan_vae.model.requires_grad_(False).eval()
    wan_vae.model = wan_vae.model.to(torch.bfloat16)
    video_vae_server = LocalVideoVAEAdapter(wan_vae, resolution=cfg["image_size"])

    from nava_src.vae.local_audio_vae import LocalAudioVAEAdapter, init_ltx_vae
    ltx_vae = init_ltx_vae(cfg["model"]["audio_vae_ckpt_dir"], device=device)
    audio_vae_server = LocalAudioVAEAdapter(ltx_vae, spk_model=None, sample_rate=16000)

    image_vae_server = None

    data = []
    with open(cfg["data"]["data_filelist"], "r") as f:
        for item in f.read().split('\n'):
            if not item: continue
            if len(item.split('\t')) == 3:
                idx, name, path = item.split('\t')
                data.append([name, path])
            elif len(item.split('\t')) == 2:
                idx, path = item.split('\t')
                data.append(([path]))
            else:
                assert False
    src_id2ratios = {}
    with open(cfg["data"]["data_weights"], "r") as f:
        for item in f.read().split("\n"):
            if not item:
                continue
            key, value, modal = item.split("\t")
            src_id2ratios[key] = [float(value), modal]

    ds = AudioVideoDataset(
        jsonl_or_src_list=data,
        dist_info=DistInfo(world_rank=0, world_size=1),
        batch_size=cfg['batch_size'],
        queue_size=cfg["data"].get("queue_size", 5),
        io_workers=cfg["data"].get("io_workers", 16),
        src_id2ratios=src_id2ratios,
        modal_prob=cfg["data"]["modal_prob"],
        num_shards=1,
        use_aspect_ratio_buckets=cfg["data"].get("use_aspect_ratio_buckets", False),
        use_length_buckets=cfg["data"].get("use_length_buckets", False),
        num_length_buckets=cfg["data"].get("num_length_buckets", 10),
        enable_ddp_bucket_sync=cfg["data"].get("enable_ddp_bucket_sync", False),
        audio_vae_server=audio_vae_server,
        image_vae_server=image_vae_server,
        video_vae_server=video_vae_server,
        min_audio_duration=cfg["data"].get("min_audio_duration", 0.0),
        max_audio_duration=cfg["data"].get("max_audio_duration", 15.0),
        video_min_frames=cfg["data"].get("video_min_frames", 1),
        video_max_frames=cfg["data"].get("video_max_frames", 5),
        video_tgt_frames=cfg["data"].get("video_tgt_frames", -1),
        video_fps=cfg["data"].get("video_fps", 8),
        add_spk_emb=cfg["data"].get("add_spk_emb", False),
        spk_emb_prob=cfg["data"].get("spk_emb_prob", 0.9),
        use_speech_special_token=cfg["data"].get("use_speech_special_token", False),
        data_file_divisor=cfg["data"].get("data_file_divisor", 1),
    )

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 先不启多进程 worker，逻辑确认 OK 再开
        collate_fn=partial(collate_fn_batch),
    )

    for step, batch in enumerate(dl):
        batch = batch[0]
        print(f"step {step}")
        print("  data_state:", batch["data_state"])
        print("-" * 40)


if __name__ == "__main__":
    main()