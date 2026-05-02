# StreamCorrect — Qwen3-ASR

Streaming ASR using Qwen3-ASR-1.7B with block-wise async beam search, with optional SpeechLM error corrector.

---

## Pretrained SpeechLM corrector checkpoints

Pretrained adapter checkpoints live in the Hugging Face dataset repo [`playwithmino/StreamCorrect_internal`](https://huggingface.co/datasets/playwithmino/StreamCorrect_internal), as **`ver1.zip`** (unpacks to **`ver1_ckpt/`**) and **`ver2.zip`** (**`ver2_ckpt/`**). On internal waihu test set **ver1** reaches **10.99%** CER and **ver2** reaches **12.88%** CER; however, **ver2** fine-tuned on **more training data** than **ver1**.

Download and extract from the terminal (no manual browser download):

```bash
pip install -U "huggingface_hub[cli]"   # if `hf` is not available

mkdir -p streamcorrect_ckpts && cd streamcorrect_ckpts
hf download playwithmino/StreamCorrect_internal ver1.zip ver2.zip \
  --repo-type dataset --local-dir .
unzip -q ver1.zip && unzip -q ver2.zip
cd ..
```

Use the extracted directory as `ERROR_CORRECTOR_CKPT`. Replace `/path/to/checkpoint` in the examples below with e.g. **`$(pwd)/streamcorrect_ckpts/ver1_ckpt`** or **`$(pwd)/streamcorrect_ckpts/ver2_ckpt`** (absolute paths are fine too).

```bash
AUDIO_DIR=/path/to/wav_dir \
OUTPUT_DIR=./batch_output \
WORKERS=4 \
GPUS=0,1,2,3 \
ERROR_CORRECTOR_CKPT="./streamcorrect_ckpts/ver1_ckpt" \
bash run_single_eval_qwen3asr_vllm.sh
```

---

## Quick start

All scripts are in `runs/`. Run them from the `runs/` directory.

### Single file

```bash
cd runs
bash run_single_eval_qwen3asr_vllm.sh
```

Override any setting with an environment variable:

```bash
AUDIO_PATH=/path/to/audio.wav \
OUTPUT_DIR=./my_output \
bash run_single_eval_qwen3asr_vllm.sh
```

With error corrector:

```bash
ERROR_CORRECTOR_CKPT=/path/to/checkpoint \
AUDIO_PATH=/path/to/audio.wav \
bash run_single_eval_qwen3asr_vllm.sh
```

**Output** (in `OUTPUT_DIR/`):
- `final_transcription.txt` — full transcript
- `segments_with_timing.json` — word-level timing
- `evaluation_result.json` — CER/MER (when `REFERENCE_FILE` is set)
- `*_beam_history.json` — beam search log for data synthesis

---

### Batch (directory of audio files)

```bash
cd runs
bash run_batch_eval_qwen3asr_vllm.sh
```

Override settings:

```bash
AUDIO_DIR=/path/to/wav_dir \
OUTPUT_DIR=./batch_output \
WORKERS=4 \
GPUS=0,1,2,3 \
bash run_batch_eval_qwen3asr_vllm.sh
```

With error corrector:

```bash
ERROR_CORRECTOR_CKPT=/path/to/checkpoint \
AUDIO_DIR=/path/to/wav_dir \
bash run_batch_eval_qwen3asr_vllm.sh
```

**Output** (in `OUTPUT_DIR/`):
- `batch_transcriptions.json` — all transcripts + latency summary
- `evaluation_results.json` — per-file and average CER/MER (when `REFERENCE_FILE` is set)
- `*_beam_history.json` — one per audio file

---

## Key variables

| Variable | Default | Description |
|---|---|---|
| `AUDIO_PATH` / `AUDIO_DIR` | (see script) | Input audio file or directory |
| `MODEL` | `Qwen/Qwen3-ASR-1.7B` | Model path or HF repo id |
| `LANGUAGE` | `zh` | Language code (`zh`, `yue`, `en`, …) |
| `BEAMS` | `4` | Beam width |
| `CHUNK_SIZE` | `0.5` | VAD chunk size in seconds |
| `INITIAL_BUFFER` | `1.0` | Seconds to buffer before first inference |
| `WORKERS` | `5` | Parallel workers (batch only) |
| `GPUS` | `0,1,2,3,4` | GPU IDs for workers (batch only) |
| `ERROR_CORRECTOR_CKPT` | *(unset)* | Corrector checkpoint path — enables corrector when set |
| `ERROR_CORRECTOR_TYPE` | `speechlm` | `speechlm` (audio+text) or `lm` (text-only) |
| `REFERENCE_FILE` | (see script) | Transcript JSON for CER evaluation |
| `OUTPUT_DIR` | `./output/…` | Directory for results |

---

## Training the error corrector

See `SpeechLMCorrector/train_qwen2audio.sh` (Qwen2-Audio backend) or `LMCorrector/train.sh` (text-only).
