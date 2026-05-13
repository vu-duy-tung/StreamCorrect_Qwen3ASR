# StreamCorrect — Qwen3-ASR

Streaming ASR using Qwen3-ASR-1.7B with block-wise async beam search, optional **SpeechLM** (Ultravox) as error correctors, and optional **locally fine-tuned Qwen3-ASR** weights for the streaming recognizer. For training the **Qwen3-ASR-0.6B** corrector, see [DOCUMENTS/QWEN-CORRECTOR.md](DOCUMENTS/QWEN-CORRECTOR.md).

---

## Pretrained checkpoints (Hugging Face)

Artifacts live in [`playwithmino/StreamCorrect_internal`](https://huggingface.co/datasets/playwithmino/StreamCorrect_internal).

### SpeechLM error corrector (LoRA on Ultravox)

- **`ver1.zip`** → **`ver1_ckpt/`**, **`ver2.zip`** → **`ver2_ckpt/`** — historical SpeechLM adapters. On our internal waihu test set **ver1** reaches **10.99%** CER and **ver2** **12.88%** CER; **ver2** used **more training data** than **ver1**.
- **`ver3.zip`** → **`ver3_ckpt/`** — current best SpeechLM corrector (same stack: base **`fixie-ai/ultravox-v0_5-llama-3_2-1b`** + LoRA). Use with **`ERROR_CORRECTOR_TYPE=speechlm`**.

### Finetuned Qwen3-ASR (streaming recognizer)

- **`finetuned_qwen3asr.zip`** (~12 GB, store-compressed) → **`finetuned_qwen3asr/`** — full Trainer checkpoint for the **streaming ASR** (weights + tokenizer files + optimizer state). Point **`RECOGNIZER_CKPT`** at this directory in **`runs/run_batch_eval_qwen3asr_vllm.sh`**. The bundle may omit **`preprocessor_config.json`**; the script merges missing processor files from **`RECOGNIZER_BASE_PROCESSOR`** (default **`MODEL`**, e.g. **`Qwen/Qwen3-ASR-1.7B`**).

### Download and unzip

```bash
pip install -U "huggingface_hub[cli]"   # if `hf` is not available

mkdir -p streamcorrect_ckpts && cd streamcorrect_ckpts
hf download playwithmino/StreamCorrect_internal \
  ver1.zip ver2.zip ver3.zip finetuned_qwen3asr.zip \
  --repo-type dataset --local-dir .
unzip -q ver1.zip && unzip -q ver2.zip && unzip -q ver3.zip
unzip -q finetuned_qwen3asr.zip   # large; creates finetuned_qwen3asr/
cd ..
```

Download only the archives you need (omit **`finetuned_qwen3asr.zip`** if you keep using the Hub **`MODEL`** id only).

### Using `ver3_ckpt` + finetuned streaming ASR

From the **`runs/`** directory, pass **absolute or repo-root-relative** paths. **`ERROR_CORRECTOR_BASE_MODEL`** must match the LoRA base in **`ver3_ckpt/adapter_config.json`** (Ultravox).

```bash
cd runs
REPO="$(cd .. && pwd)"
MODEL=Qwen/Qwen3-ASR-1.7B \
RECOGNIZER_CKPT="$REPO/streamcorrect_ckpts/finetuned_qwen3asr" \
RECOGNIZER_BASE_PROCESSOR=Qwen/Qwen3-ASR-1.7B \
ERROR_CORRECTOR_CKPT="$REPO/streamcorrect_ckpts/ver3_ckpt" \
ERROR_CORRECTOR_TYPE=speechlm \
ERROR_CORRECTOR_BASE_MODEL=fixie-ai/ultravox-v0_5-llama-3_2-1b \
bash run_batch_eval_qwen3asr_vllm.sh
```

Use **`ver1_ckpt`** / **`ver2_ckpt`** the same way: set **`ERROR_CORRECTOR_CKPT`** only (same **`ERROR_CORRECTOR_TYPE`** / base model).

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
| `MODEL` | `Qwen/Qwen3-ASR-1.7B` | Base Qwen3-ASR id or path when using `RECOGNIZER_CKPT` bootstrap |
| `RECOGNIZER_CKPT` | *(see `run_batch_eval_qwen3asr_vllm.sh`)* | Local fine-tuned ASR checkpoint dir; unset uses Hub **`MODEL`** only |
| `RECOGNIZER_BASE_PROCESSOR` | `$MODEL` | Source for missing processor/tokenizer files next to `RECOGNIZER_CKPT` |
| `LANGUAGE` | `zh` | Language code (`zh`, `yue`, `en`, …) |
| `BEAMS` | `4` | Beam width |
| `CHUNK_SIZE` | `0.5` | VAD chunk size in seconds |
| `INITIAL_BUFFER` | `1.0` | Seconds to buffer before first inference |
| `WORKERS` | `5` | Parallel workers (batch only) |
| `GPUS` | `0,1,2,3,4` | GPU IDs for workers (batch only) |
| `ERROR_CORRECTOR_CKPT` | *(unset)* | Corrector checkpoint path — enables corrector when set |
| `ERROR_CORRECTOR_BASE_MODEL` | *(see script)* | Base weights for LoRA correctors (Ultravox for `speechlm`, Qwen3-ASR-0.6B for `qwen3asr`) |
| `ERROR_CORRECTOR_TYPE` | `speechlm` | `speechlm`, `qwen3asr` (Qwen3-ASR-0.6B LoRA), or `lm` (text-only) |
| `REFERENCE_FILE` | (see script) | Transcript JSON for CER evaluation |
| `OUTPUT_DIR` | `./output/…` | Directory for results |

---

## Training the error corrector

### Training JSONL data (download and unzip)

The dataset repo [`playwithmino/StreamCorrect_internal`](https://huggingface.co/datasets/playwithmino/StreamCorrect_internal) includes **`synthesized_data_v3_jsonl.zip`**, which unpacks to a **`data/`** folder with three JSONL files (`waihu_…`, `aishell_…`, `kespeech_…`). Place them next to your other synthesized corpora so paths in the training configs resolve:

```bash
# From the repository root
mkdir -p SpeechLMCorrector/data/synthesized_data_v3
cd SpeechLMCorrector/data/synthesized_data_v3
hf download playwithmino/StreamCorrect_internal synthesized_data_v3_jsonl.zip \
  --repo-type dataset --local-dir .
unzip -o synthesized_data_v3_jsonl.zip
mv data/*.jsonl .
rmdir data 2>/dev/null || true
```

If **`Qwen3ASRCorrector/training_config.yaml`** (or SpeechLM YAML) lists JSONL paths you do not have locally, edit **`train_data_paths`** to match the files you downloaded, or add the missing files.

### Fine-tune Qwen3-ASR-0.6B as corrector (`Qwen3ASRCorrector/train.sh`)

[`Qwen3ASRCorrector/training_config.yaml`](Qwen3ASRCorrector/training_config.yaml) defines the base model (`Qwen/Qwen3-ASR-0.6B`), LoRA targets, data sources, and Trainer hyperparameters. Run from the **repository root**:

```bash
bash Qwen3ASRCorrector/train.sh
```

Multi-GPU (recommended over naive multi-device `python`):

```bash
NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash Qwen3ASRCorrector/train.sh
```

Optional flags: `--config …`, `--adapter /path/to/lora`, `--resume /path/to/checkpoint-XXXX`, `--output /path/to/out`, `--port PORT`. Checkpoints and the final adapter land under **`output_dir/final/`** (see YAML); point **`ERROR_CORRECTOR_CKPT`** there at inference and set **`ERROR_CORRECTOR_TYPE=qwen3asr`** (e.g. `runs/run_batch_eval_double_qwen3asr.sh`).

### Other corrector backends

- **SpeechLM / Qwen2-Audio:** `SpeechLMCorrector/train_qwen2audio.sh`
- **Text-only LM:** `LMCorrector/train.sh`
