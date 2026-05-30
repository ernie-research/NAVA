# NAVA 训练说明

## 1. 数据格式

### 1.1 数据文件（JSONL）

每个数据集是一个 JSONL 文件，每行一条样本，字段如下：

```json
{
  "data_id": "唯一标识",
  "video_info": [
    {
      "data_path": "/abs/path/to/video.mp4",
      "fps": 25.0,
      "duration": 3.0,
      "image_width": 1920,
      "image_height": 1080
    }
  ],
  "text_list": [
    {
      "text": "视频描述文本，台词用 <S>...</E> 包裹，例如：他说：<S>你好<E>",
      "text_type": "caption",
      "speech_start": [0.0],
      "speech_end":   [2.76]
    }
  ],
  "audio_splits_info_tagging": [
    {
      "audio_duration": 3.0,
      "audio_info": {
        "caption_data": { ... }
      }
    }
  ]
}
```

关键字段说明：

- `video_info[0].data_path`：视频文件绝对路径，训练时同时作为视频源和音频源使用
- `text_list[0].text`：prompt，台词片段用 `<S>...</E>` 标记，模型据此学习音视频对齐
- `text_list[0].speech_start/speech_end`：台词时间戳（秒），用于提取 speaker embedding
- `audio_splits_info_tagging`：音频质量/内容标注，训练时用于过滤无效样本

`text_to_audio`（纯音频）样本与上述格式相同，区别是训练时不使用 `video_info`，只对音频做编码。

---

## 2. 数据列表与权重文件

### 2.1 数据列表（`.list`）

格式：`<idx>\t<set_name>\t<jsonl_path>` 或 `<idx>\t<jsonl_path>`

```
0	av_set1	data/av_set1/train_av_demo.json
1	av_set2	data/av_set2/train_av_demo.json
2	audio_set1	data/audio_set1/train_av_demo.json
```

- 第一列：序号（任意整数）
- 第二列：数据集名称（`set_name`，需与权重文件中的 key 对应）
- 第三列：JSONL 文件路径（相对于项目根目录）

三列格式时 `set_name` 作为索引键；两列格式时路径本身作为键。

### 2.2 权重文件（`.weight`）

格式：`<set_name>\t<weight>\t<modal>`

**纯 AV 训练（`av_data_demo.weight`）：**
```
av_set1	1	text_to_av
av_set2	2	text_to_av
```

**混训（`av_data_demo_mix.weight`）：**
```
av_set1	1	text_to_av
av_set2	2	text_to_av
audio_set1	1	text_to_audio
```

字段说明：

| 字段 | 含义 |
|------|------|
| `set_name` | 与 `.list` 中第二列对应 |
| `weight` | 该数据集的采样权重（相对值，数值越大采样越频繁） |
| `modal` | 训练模态，决定该数据集走哪条训练分支 |

支持的 `modal` 值：

| modal | 训练内容 |
|-------|---------|
| `text_to_av` | 同时生成视频 + 音频（主任务） |
| `text_to_audio` | 仅生成音频（纯音频任务） |
| `text_to_video` | 仅生成视频 |
| `text_to_image` | 仅生成图像 |

**采样逻辑**：训练时按各数据集 weight 做加权随机采样，同一模态内多个数据集之间按 weight 比例混合。例如上例中 `av_set2` 的采样概率是 `av_set1` 的 2 倍。

---

## 3. 配置文件

### 3.1 纯 AV 训练（`configs/nava.yaml`）

```yaml
data:
  data_filelist: data/av_data_demo.list
  data_weights: data/av_data_demo.weight   # 只含 text_to_av 的 weight 文件

  modal_prob:
    text_to_audio: 0.0   # 不启用纯音频任务
    text_to_video: 0.0
    text_to_image: 0.0
    text_to_av: 1        # 全部走 AV 联合生成
```

`modal_prob` 是各模态**任务级别**的开关：值 > 0 才会从对应模态的数据源里采样。配合 weight 文件使用——weight 文件决定数据集内部的采样比例，`modal_prob` 决定不同任务之间的总体比例。

### 3.2 混训（`configs/nava_mixtrain.yaml`）

```yaml
data:
  data_filelist: data/av_data_demo.list
  data_weights: data/av_data_demo_mix.weight  # 含 text_to_av + text_to_audio 的 weight 文件

  modal_prob:
    text_to_audio: 1     # 启用纯音频任务
    text_to_video: 0.0
    text_to_image: 0.0
    text_to_av: 1        # 同时启用 AV 联合任务

grad_accum_steps: 4      # 混训时建议开大，保证每个 AV 样本都有梯度
```

混训时需要同时满足两个条件：
1. weight 文件中有 `text_to_audio` 行（数据源）
2. config 中 `modal_prob.text_to_audio > 0`（任务开关打开）

两者缺一不可，否则纯音频数据不会被采样。

### 3.3 其他关键配置项

```yaml
data:
  video_fps: 24              # 视频采样帧率
  video_tgt_frames: 121      # 目标帧数（121 = 5秒@24fps，需满足 4N+1）
  max_audio_duration: 10.0   # 最长音频时长（秒）
  add_spk_emb: true          # 是否提取 speaker embedding
  spk_emb_prob: 0.9          # speaker embedding 使用概率

  use_length_buckets: true   # 启用长度分桶，相同时长的样本聚在一起（默认 false，需显式开启）
  num_length_buckets: 5      # 分桶数量
  enable_ddp_bucket_sync: true  # 多卡间同步桶分配

audio_loss_coff: 0.2         # 音频 loss 权重
vision_loss_coff: 1          # 视频 loss 权重
```

---

## 4. 启动训练

实际训练通过 `accelerate launch` 拉起，配合自动生成的 FSDP 配置文件，单机 8 卡 FSDP（`FULL_SHARD`，bf16），从项目根目录执行。

| 脚本 | 用途 |
|------|------|
| `train/train_nava_scarch_mix.sh` | 从零开始混训（scratch）：无 `--resume`，使用 `configs/nava_mixtrain.yaml` |
| `train/train_nava_sft.sh` | SFT / fine-tune：从已有 checkpoint 加载权重，**不恢复**步数和数据游标，从 step 0 重新训练 |

### 从零开始混训

```bash
bash train/train_nava_scarch_mix.sh
```

等价于：

```bash
accelerate launch --config_file fsdp_config_auto.yaml \
    train/train_nava.py \
    --config configs/nava_mixtrain.yaml
```

使用 `nava_mixtrain.yaml`（`modal_prob.text_to_audio: 1`，`grad_accum_steps: 4`），同时训练 AV 联合生成和纯音频生成。

### SFT / Fine-tune

```bash
bash train/train_nava_sft.sh
```

等价于：

```bash
accelerate launch --config_file fsdp_config_auto.yaml \
    train/train_nava.py \
    --config configs/nava.yaml \
    --resume NAVA.ckpt \
    --load_ckpt_only        # 只取权重，步数和数据位置重置
```

在预训练好的 checkpoint 基础上换新数据集从头训练。使用前将脚本中的 `NAVA.ckpt` 替换为实际 checkpoint 路径。

---

## 5. 断点续训（Resume）

checkpoint 每隔 `save_every`（默认 2500）步保存一次，路径为：

```
{out_dir}/step{N}.ckpt
```

checkpoint 内容：

| 字段 | 说明 |
|------|------|
| `state_dict` | 模型权重 |
| `ema_state` | EMA 权重（若启用） |
| `global_step` | 已训练步数 |
| `data_state` | 数据读取游标，记录每个 worker 读到了哪里 |

### 完整 resume（权重 + 步数 + 数据位置）

```bash
accelerate launch --config_file fsdp_config_auto.yaml \
    train_nava.py \
    --config configs/nava.yaml \
    --resume outputs/your_run/step5000.ckpt
```

恢复后训练从 `global_step=5000` 继续，数据从上次读取位置接续，不会重复消费已训练过的样本。

**多卡数量改变时**：`data_state` 的 worker 数量可能不匹配。代码会自动适配——取旧状态中各数据源的最大游标，广播给新的所有 worker，会有少量数据重复但不会遗漏。

### 仅加载权重（`--load_ckpt_only`）

用于迁移学习或从预训练模型 fine-tune，**不恢复** `global_step` 和数据游标，训练从 step 0 重新开始：

```bash
accelerate launch --config_file fsdp_config_auto.yaml \
    train_nava.py \
    --config configs/nava.yaml \
    --resume path/to/pretrained.ckpt \
    --load_ckpt_only
```

---

## 6. 超参传入

所有超参均通过 `--config` 指定的 yaml 文件控制，**不支持命令行逐参覆盖**。需要修改超参时直接编辑 yaml 或复制一份新 yaml。

常用超参位置速查：

| 超参 | yaml 路径 | 说明 |
|------|-----------|------|
| 学习率 | `lr` | 默认 `1e-4` |
| batch size | `batch_size` | 单卡 batch |
| 梯度累积 | `grad_accum_steps` | 等效 batch = batch_size × 卡数 × 该值 |
| 最大步数 | `max_steps` | |
| 保存间隔 | `save_every` | 单位：step |
| 输出目录 | `out_dir` | checkpoint 和 tensorboard 写入此处 |
| 音频 loss 权重 | `audio_loss_coff` | 默认 0.2 |
| 视频 loss 权重 | `vision_loss_coff` | 默认 1.0 |
| 目标帧数 | `data.video_tgt_frames` | 需满足 4N+1，如 121/241 |
| 最小帧数 | `data.video_min_frames` | 短于此帧数的视频丢弃 |
| 最大帧数 | `data.video_max_frames` | 长于此帧数的视频截断 |
| 视频帧率 | `data.video_fps` | |
| 最长音频 | `data.max_audio_duration` | 单位：秒 |
| 长度分桶 | `data.use_length_buckets` | 默认 `false`，开启后相同时长样本聚批，训练更稳定 |
| 分桶数量 | `data.num_length_buckets` | `use_length_buckets: true` 时生效，默认 5 |
| 混合精度 | `amp_dtype` | `bf16` / `fp16` / `null` |

---

## 7. 异步数据加载

### 架构概览

```
JSONL 文件
    │
    ▼
_fetch_raw_jsons()          ← 按 weight 加权随机从各数据源顺序读取原始 JSON
    │  （producer 线程，逐条）
    ▼
io_pool.submit(             ← ThreadPoolExecutor，并发 VAE encode（视频/音频）
    _process_item_concurrently
)   × io_workers 个并发 future
    │
    ▼
modality_queues[modal]      ← 每个启用模态对应一个独立 Queue（先进先出）
    │  maxsize = queue_size × batch_size
    ▼
__iter__() 消费者           ← 主训练循环从 Queue.get() 取 batch
```

每个启用的模态（`text_to_av` / `text_to_audio` 等）各有：
- **1 个 producer 线程**：负责读 JSON、提交 encode 任务、把结果放入 Queue
- **共享 io_pool**：`io_workers` 个线程并发做 VAE encode（最耗时的操作）

Queue 满时 producer 自动阻塞（背压），不会无限堆积内存。

### `io_workers` 设置

控制并发 VAE encode 的线程数。VAE encode 是 GPU 操作，内部通过锁串行上 GPU（保证线程安全），所以 `io_workers` 实际控制的是**同时在排队等待 GPU 的样本数**，相当于 encode 流水线的窗口大小。

| 场景 | 建议值 |
|------|--------|
| 调试 / 单卡小显存 | `2–4` |
| 正常训练 | `8–16` |
| encode 速度跟不上训练 | 适当调大，但超过 GPU encode 吞吐后无意义 |

对应 config 字段：`data.io_workers`

### `queue_size` 设置

Queue 容量 = `queue_size × batch_size`。队列越大，对 encode 速度抖动的缓冲能力越强，但占用更多 CPU 内存（每个样本含完整 latent tensor）。

| 场景 | 建议值 |
|------|--------|
| 内存紧张 | `4–8` |
| 正常训练 | `16–32` |
| 训练侧频繁等数据 | 先调大 `io_workers`，再考虑调大 `queue_size` |

对应 config 字段：`data.queue_size`

### `num_workers` 设置

PyTorch DataLoader 的进程数。**开启 `enable_ddp_bucket_sync: true` 时强制为 0**（DDP 分桶同步需要在主进程运行 `dist.broadcast`），此时所有 IO 完全由内部 `io_workers` 线程接管。

正常情况下保持 `num_workers: 0`，依赖 `io_workers` 做异步预取即可。

> `num_workers` 同时影响断点续训时的 `data_state` shard 数量：`num_shards = num_workers × GPU数`（`num_workers=0` 时按 1 算）。改变该值 resume 时会触发游标自动适配，可能有少量数据重复。
