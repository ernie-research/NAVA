# SeedTTS Benchmark Layout

NAVA evaluates zero-shot speech synthesis on the [SeedTTS test set](https://github.com/BytedanceSpeech/seed-tts-eval). The benchmark itself is **not** redistributed in this repo — drop the official files into this directory using the layout below, then run `bash scripts/inference_seedtts.sh` from the repo root.

## Expected layout

```
infer_cases/seedtts/
├── zh/
│   ├── meta.lst           # one line per utterance
│   └── wavs/              # reference prompt WAVs
│       ├── ...wav
│       └── ...
└── en/
    ├── meta.lst
    └── wavs/
        └── ...
```

## `meta.lst` format

One utterance per line, four `|`-separated fields:

```
<utt_id>|<prompt_text>|<prompt_wav>|<infer_text>
```

| Field | Meaning |
|-------|---------|
| `utt_id` | Output filename stem; the generated speech is saved as `{utt_id}.wav`. |
| `prompt_text` | Transcript of the reference clip (used to estimate target length). |
| `prompt_wav` | Path to the reference clip. Relative paths resolve against `infer_cases/seedtts/{lang}/`; absolute paths and `bos://` URLs are passed through unchanged. |
| `infer_text` | The text to synthesize in the reference speaker's voice. |

## Run

```bash
# Chinese split (default)
bash scripts/inference_seedtts.sh

# English split
LANG=en bash scripts/inference_seedtts.sh

# Custom paths
DATA_FILE=/path/to/meta.lst OUT_DIR=eval_results/seedtts/custom \
    bash scripts/inference_seedtts.sh
```

Outputs land in `eval_results/seedtts/{lang}/{utt_id}.wav` by default.
