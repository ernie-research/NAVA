# NAVA Training Guide

## 1. Data Format

### 1.1 Data File (JSONL)

Each dataset is a JSONL file with one sample per line:

```json
{
  "data_id": "unique_id",
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
      "text": "Video caption; speech spans wrapped in <S>...<E>, e.g.: he said: <S>Hello<E>",
      "text_type": "caption",
      "speech_start": [0.0],
      "speech_end":   [2.76]
    }
  ],
  "audio_splits_info_tagging": [
    {
      "audio_duration": 3.0,
      "audio_info": {
        "caption_data": {}
      }
    }
  ]
}
```

Key fields:

- `video_info[0].data_path` — absolute path to the video file, used as both the video and audio source during training.
- `text_list[0].text` — prompt; speech spans are marked with `<S>...<E>` for the model to learn audio-visual alignment.
- `text_list[0].speech_start/speech_end` — speech timestamps in seconds, used to extract speaker embeddings.
- `audio_splits_info_tagging` — audio quality / content annotations; used to filter invalid samples during training.

`text_to_audio` (audio-only) samples use the same format. The difference is that `video_info` is ignored — only the audio track is encoded.

---

## 2. Data List and Weight Files

### 2.1 Data List (`.list`)

Format: `<idx>\t<set_name>\t<jsonl_path>` or `<idx>\t<jsonl_path>`

```
0	av_set1	data/av_set1/train_av_demo.json
1	av_set2	data/av_set2/train_av_demo.json
2	audio_set1	data/audio_set1/train_av_demo.json
```

- Column 1: index (any integer).
- Column 2: dataset name (`set_name`), must match the key in the weight file.
- Column 3: path to the JSONL file (relative to the project root).

In the 3-column format `set_name` is used as the lookup key; in the 2-column format the path itself is the key.

### 2.2 Weight File (`.weight`)

Format: `<set_name>\t<weight>\t<modal>`

**AV-only training (`av_data_demo.weight`):**
```
av_set1	1	text_to_av
av_set2	2	text_to_av
```

**Mixed training (`av_data_demo_mix.weight`):**
```
av_set1	1	text_to_av
av_set2	2	text_to_av
audio_set1	1	text_to_audio
```

| Field | Description |
|-------|-------------|
| `set_name` | Matches column 2 in the `.list` file |
| `weight` | Relative sampling weight — higher means sampled more often |
| `modal` | Training modality, determines which training branch this dataset feeds |

Supported `modal` values:

| modal | Task |
|-------|------|
| `text_to_av` | Joint audio + video generation (main task) |
| `text_to_audio` | Audio-only generation |
| `text_to_video` | Video-only generation |
| `text_to_image` | Image-only generation |

**Sampling logic**: at each step, a dataset is drawn with probability proportional to its weight. Multiple datasets within the same modality are mixed at their weight ratio. In the example above, `av_set2` is sampled twice as often as `av_set1`.

---

## 3. Config Files

### 3.1 AV-only Training (`configs/nava.yaml`)

```yaml
data:
  data_filelist: data/av_data_demo.list
  data_weights: data/av_data_demo.weight   # weight file with text_to_av only

  modal_prob:
    text_to_audio: 0.0   # audio-only task disabled
    text_to_video: 0.0
    text_to_image: 0.0
    text_to_av: 1        # all samples go through joint AV generation
```

`modal_prob` is a per-task switch: a modality is only sampled when its value is > 0. It works together with the weight file — the weight file controls the sampling ratio within each modality; `modal_prob` controls the overall ratio across tasks.

### 3.2 Mixed Training (`configs/nava_mixtrain.yaml`)

```yaml
data:
  data_filelist: data/av_data_demo.list
  data_weights: data/av_data_demo_mix.weight  # weight file with text_to_av + text_to_audio

  modal_prob:
    text_to_audio: 1     # audio-only task enabled
    text_to_video: 0.0
    text_to_image: 0.0
    text_to_av: 1        # joint AV task also enabled

grad_accum_steps: 4      # recommended for mixed training to ensure every AV sample gets gradients
```

Mixed training requires both conditions to be met simultaneously:
1. The weight file contains `text_to_audio` rows (data source).
2. `modal_prob.text_to_audio > 0` in the config (task switch is on).

If either is missing, audio-only samples will not be drawn.

### 3.3 Other Key Config Options

```yaml
data:
  video_fps: 24              # video sampling frame rate
  video_tgt_frames: 121      # target frame count (121 = 5 s @ 24 fps; must satisfy 4N+1)
  max_audio_duration: 10.0   # maximum audio duration in seconds
  add_spk_emb: true          # whether to extract speaker embeddings
  spk_emb_prob: 0.9          # speaker embedding injection probability

  use_length_buckets: true   # group samples of similar length into the same batch (default false)
  num_length_buckets: 5      # number of length buckets
  enable_ddp_bucket_sync: true  # synchronize bucket assignment across GPUs

audio_loss_coff: 0.2         # audio loss weight
vision_loss_coff: 1          # video loss weight
```

---

## 4. Launching Training

Training is launched via `accelerate launch` with an auto-generated FSDP config. All scripts run single-node 8-GPU FSDP (`FULL_SHARD`, bf16) from the project root.

| Script | Purpose |
|--------|---------|
| `train/train_nava_scarch_mix.sh` | Mixed AV + audio-only training, warm-started from Wan2.2-5B weights (`configs/nava_mixtrain.yaml`) |
| `train/train_nava_sft.sh` | SFT / fine-tune: load weights from an existing checkpoint, reset step counter and data cursor |

### Mixed Training from Wan2.2-5B Warm Start

```bash
bash train/train_nava_scarch_mix.sh
```

Equivalent to:

```bash
accelerate launch --config_file fsdp_config_auto.yaml \
    train/train_nava.py \
    --config configs/nava_mixtrain.yaml \
    --resume Wan_5B.ckpt --load_ckpt_only
```

Uses `nava_mixtrain.yaml` (`modal_prob.text_to_audio: 1`, `grad_accum_steps: 4`). Warm-starts from `Wan_5B.ckpt` (weights only, step counter reset). Download `Wan_5B.ckpt` from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) and place it in the project root before running.

### SFT / Fine-tune

```bash
bash train/train_nava_sft.sh
```

Equivalent to:

```bash
accelerate launch --config_file fsdp_config_auto.yaml \
    train/train_nava.py \
    --config configs/nava.yaml \
    --resume NAVA.safetensors \
    --load_ckpt_only        # load weights only; step and data position are reset
```

Starts training from step 0 on a new dataset, using an existing pretrained checkpoint as initialization. Replace `NAVA.safetensors` in the script with the actual checkpoint path before running.

---

## 5. Resuming from a Checkpoint

Checkpoints are saved every `save_every` steps (default 2500) at:

```
{out_dir}/step{N}.ckpt
```

Checkpoint contents:

| Field | Description |
|-------|-------------|
| `state_dict` | Model weights |
| `ema_state` | EMA weights (if enabled) |
| `global_step` | Number of steps trained so far |
| `data_state` | Per-worker data cursor — records how far each worker has read |

### Full Resume (weights + step + data position)

```bash
accelerate launch --config_file fsdp_config_auto.yaml \
    train_nava.py \
    --config configs/nava.yaml \
    --resume outputs/your_run/step5000.ckpt
```

Training resumes from `global_step=5000`; data continues from the last cursor position with no repeated samples.

**When the number of GPUs changes**: the `data_state` worker count may not match. The code auto-adapts by broadcasting the maximum cursor across all data sources to every new worker — a small amount of data may be repeated but nothing is skipped.

### Weights Only (`--load_ckpt_only`)

For transfer learning or fine-tuning from a pretrained model. `global_step` and data cursors are **not** restored; training starts from step 0:

```bash
accelerate launch --config_file fsdp_config_auto.yaml \
    train_nava.py \
    --config configs/nava.yaml \
    --resume path/to/pretrained.safetensors \
    --load_ckpt_only
```

---

## 6. Hyperparameters

All hyperparameters are controlled via the YAML config file specified by `--config`. **There is no per-parameter CLI override** — edit the YAML directly or copy it to a new file.

Quick reference:

| Hyperparameter | YAML key | Notes |
|----------------|----------|-------|
| Learning rate | `lr` | Default `1e-4` |
| Batch size | `batch_size` | Per-GPU batch size |
| Gradient accumulation | `grad_accum_steps` | Effective batch = batch_size × GPUs × this value |
| Max steps | `max_steps` | |
| Save interval | `save_every` | In steps |
| Output directory | `out_dir` | Checkpoints and TensorBoard logs are written here |
| Audio loss weight | `audio_loss_coff` | Default `0.2` |
| Video loss weight | `vision_loss_coff` | Default `1.0` |
| Target frames | `data.video_tgt_frames` | Must satisfy 4N+1, e.g. 121 / 241 |
| Min frames | `data.video_min_frames` | Videos shorter than this are discarded |
| Max frames | `data.video_max_frames` | Videos longer than this are truncated |
| Video FPS | `data.video_fps` | |
| Max audio duration | `data.max_audio_duration` | In seconds |
| Length bucketing | `data.use_length_buckets` | Default `false`; when enabled, samples of similar length are batched together for more stable training |
| Number of buckets | `data.num_length_buckets` | Effective when `use_length_buckets: true`; default 5 |
| Mixed precision | `amp_dtype` | `bf16` / `fp16` / `null` |

---

## 7. Async Data Loading

### Architecture Overview

```
JSONL files
    │
    ▼
_fetch_raw_jsons()          ← weighted random draw from data sources; reads raw JSON sequentially
    │  (producer thread, one item at a time)
    ▼
io_pool.submit(             ← ThreadPoolExecutor; concurrent VAE encode (video / audio)
    _process_item_concurrently
)   × io_workers concurrent futures
    │
    ▼
modality_queues[modal]      ← one independent Queue per enabled modality (FIFO)
    │  maxsize = queue_size × batch_size
    ▼
__iter__() consumer         ← main training loop calls Queue.get() to pull batches
```

Each enabled modality (`text_to_av`, `text_to_audio`, etc.) gets:
- **1 producer thread** — reads JSON, submits encode tasks, pushes results into the Queue.
- **Shared `io_pool`** — `io_workers` threads that run VAE encode concurrently (the most expensive operation).

When the Queue is full the producer blocks automatically (back-pressure), preventing unbounded memory accumulation.

### `io_workers`

Controls the number of concurrent VAE encode threads. VAE encode is a GPU operation; an internal lock serializes GPU access for thread safety, so `io_workers` effectively sets the **number of samples queued for GPU encode at any one time** — i.e., the pipeline window size.

| Scenario | Recommended value |
|----------|-------------------|
| Debugging / small GPU | `2–4` |
| Normal training | `8–16` |
| Encode can't keep up with training | Increase; but going beyond GPU encode throughput has no effect |

Config key: `data.io_workers`

### `queue_size`

Queue capacity = `queue_size × batch_size`. A larger queue absorbs encode-speed variance better, but uses more CPU memory (each slot holds a full latent tensor).

| Scenario | Recommended value |
|----------|-------------------|
| Memory-constrained | `4–8` |
| Normal training | `16–32` |
| Training frequently stalls waiting for data | Increase `io_workers` first, then consider increasing `queue_size` |

Config key: `data.queue_size`

### `num_workers`

PyTorch DataLoader worker count. **Forced to 0 when `enable_ddp_bucket_sync: true`** (DDP bucket sync requires `dist.broadcast` in the main process). In that mode all I/O is handled by the internal `io_workers` threads.

Under normal circumstances keep `num_workers: 0` and rely on `io_workers` for async prefetching.

> `num_workers` also affects the `data_state` shard count on resume: `num_shards = num_workers × num_GPUs` (`num_workers=0` counts as 1). Changing this value triggers automatic cursor adaptation on resume, which may cause a small number of samples to be repeated.
