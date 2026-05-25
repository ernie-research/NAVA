## Input Data Format Examples

### Text-only (T2AV)

```jsonl
{"prompt": "一位男子在海边奔跑，镜头跟随其移动。写实电影感, 中景跟随, 自然光。背景是海浪声和风声。"}
{"prompt": "一只巨龙在城市上空喷出烈焰，建筑物被火焰照亮。整体氛围宏大震撼。"}
```

### Image-to-AudioVideo (I2AV)

```jsonl
{"prompt": "这段写实风格的视频中，男子正在演讲...<S>And then notice something.<E>", "image_path": "/data/frames/001.jpg"}
{"prompt": "一只猫从沙发上跳下，轻盈落地。", "image_path": "/data/frames/cat.png"}
```

### Multi-Speaker with Timbre Control

```jsonl
{"prompt": "警探压低声音说<S>Drop the weapon. Now.<E> 嫌疑人冷笑回应<S>You really think this ends here?<E>", "spk_wavs": ["/data/spk/cop.wav", "/data/spk/suspect.wav"]}
{"prompt": "女主播说道<S>欢迎来到今天的节目<E>", "spk_wavs": ["/data/spk/anchor.wav"]}
```

### Mixed (all modes in one file)

```jsonl
{"prompt": "纯文本场景，无人说话，海浪声和海鸥叫声。"}
{"prompt": "男子面对镜头说<S>Hello world<E>", "image_path": "/data/img.png", "spk_wavs": ["/data/spk.wav"]}
```