# Gradio 推理脚本 Plan

## 目标
创建一个 Gradio Web UI，用户输入短 prompt → 自动 rewrite → 调用 NAVA 模型生成音视频（SP=8）。

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│  Gradio UI (rank 0 only)                                │
│  输入: prompt, [可选图片], [可选speaker wav]             │
│  输出: 生成的视频（含音频）                              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Prompt Rewrite (Qwen3-8B, rank 0 only)                 │
│  短 prompt → 电影化长 prompt                            │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  NAVA Pipeline (SP=8, all 8 ranks 协同)                 │
│  rank 0 broadcast batch → all ranks 并行推理            │
│  → rank 0 收集结果 → 保存视频                          │
└─────────────────────────────────────────────────────────┘
```

## 关键设计决策

1. **torchrun 启动 8 卡**：Gradio server 只在 rank 0 运行，其他 rank 进入等待循环
2. **rank 间通信**：rank 0 收到请求后，broadcast prompt/参数给其他 rank，所有 rank 一起跑 sample()
3. **Rewrite 模型**：只在 rank 0 加载 Qwen3-8B（用 rewrite_single.py 的逻辑），不占推理 GPU
   - 或者：rewrite 模型放 CPU / 单独一张卡（如果 8 卡都被 NAVA 占满）
4. **单条推理**：batch_size=1，无需 DataLoader

## 文件结构

```
/root/paddlejob/workspace/env_run/NAVA/pe_src/
├── gradio_server.py        # 主入口：Gradio UI + rank 0 逻辑 + worker 循环
├── rewrite_single.py       # [已有] prompt rewrite 模块
├── nava_engine.py          # 封装 NAVA pipeline 初始化 + single sample
├── start_gradio.sh         # torchrun 启动脚本 (8卡 SP)
└── config.yaml             # [已有] 可复用或新建推理 config
```

## 各文件职责

### 1. `nava_engine.py` — NAVA 推理引擎封装

- `NAVAEngine.__init__(config, ckpt, device, rank, world_size, use_sp)`
  - 加载 pipeline（AudioVideoPipeline）
  - 加载 checkpoint
  - 如果 use_sp，patch backbone 为 SP 版本
- `NAVAEngine.generate(prompt, image_path=None, spk_wav_paths=None, ...)`
  - 构造 batch dict（captions, video_latents noise, audio_latents noise, t_h_w_list, first_frames, spk_embs）
  - 调用 pipe.sample()
  - 后处理：视频 tensor → mp4 文件路径
  - 返回输出视频路径

### 2. `gradio_server.py` — 主入口

- **rank 0 流程**：
  1. 加载 Qwen3 rewrite 模型（4bit，省显存，或放 CPU）
  2. 初始化 NAVAEngine
  3. 启动 Gradio server
  4. 收到请求 → rewrite prompt → broadcast → generate → 返回视频

- **rank 1-7 流程**：
  1. 初始化 NAVAEngine
  2. 进入无限循环：等 rank 0 broadcast 信号 → 执行 generate → 结果由 rank 0 收集

- **rank 间通信协议**：
  ```python
  # rank 0 broadcast:
  # 1. 先 broadcast 一个 control tensor: [cmd, prompt_len, ...]
  #    cmd=0: 退出, cmd=1: 推理
  # 2. broadcast prompt string (encode 为 tensor)
  # 3. 所有 rank 一起跑 pipe.sample()
  ```

### 3. `start_gradio.sh` — 启动脚本

```bash
torchrun --nproc_per_node=8 gradio_server.py \
    --config /path/to/config.yaml \
    --ckpt /path/to/checkpoint.pt \
    --port 7860
```

## 注意事项

- Rewrite 模型显存：如果 8 卡全被 NAVA 占，rewrite 用 CPU 推理（8B CPU 大约 30-60s，可接受因为是单次）或 4bit GPU（在 rank 0 的卡上额外占 ~5GB）
- SP 模式下所有 rank 必须用相同的随机种子
- Gradio 的 queue() 确保同一时间只处理一个请求（避免多请求冲突）
- 视频输出临时文件存放在 /tmp/nava_outputs/
