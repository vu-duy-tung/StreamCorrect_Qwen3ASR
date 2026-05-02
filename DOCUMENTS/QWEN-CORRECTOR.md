# Qwen3-ASR error corrector (`Qwen3ASRCorrector/`)

This note describes how we **fine-tune Qwen/Qwen3-ASR-0.6B** as an **audio-conditioned ASR error corrector** (LoRA on the language backbone, frozen audio encoder), and how to **run inference** with the **double-Qwen3** batch script that pairs the large streaming ASR (**Qwen3-ASR-1.7B**, vLLM + beam search) with this corrector.

---

## Role in the stack

| Component | Model | Role |
|-----------|--------|------|
| Streaming ASR | `Qwen/Qwen3-ASR-1.7B` | Primary transcription (vLLM, async beam search). |
| Error corrector | `Qwen/Qwen3-ASR-0.6B` + LoRA | Consumes **the same audio chunk** plus **top‑k beam hypotheses** (and optional previous text), and generates a **single corrected transcript** for that segment. |

Training and inference prompts are aligned: chat-style template with `<|audio_start|><|audio_pad|><|audio_end|>`, optional `Previous: …`, and a `<candidates>…</candidates>` block (see `Qwen3ASRCorrector/data_utils.py` and `_build_qwen3asr_corrector_prompt` in `qwen3asr_streaming_vllm_beam_async.py`).

---

## Layout

| Path | Purpose |
|------|---------|
| `Qwen3ASRCorrector/training_config.yaml` | Hyperparameters, data sources, LoRA, Trainer settings. |
| `Qwen3ASRCorrector/training_qwen3asr.py` | Loads model/processor, builds datasets, applies freeze + LoRA, runs Hugging Face `Trainer`. |
| `Qwen3ASRCorrector/data_utils.py` | `CorrectorDataset`, weighted multi-source loader, `CorrectorDataCollator`. |
| `Qwen3ASRCorrector/train.sh` | Launcher for single-GPU Python or multi-GPU `torchrun`. |
| `runs/run_batch_eval_double_qwen3asr.sh` | Batch eval: **1.7B ASR** + **0.6B corrector** (`ERROR_CORRECTOR_TYPE=qwen3asr`). |

---

## Dependencies

Training and inference assume a CUDA environment with **`transformers`**, **`peft`**, **`PyYAML`**, **`librosa`**, and the **`qwen_asr`** package so `AutoModel` can instantiate `Qwen3ASRForConditionalGeneration`. The trainer imports `qwen_asr` before `from_pretrained` (see `training_qwen3asr.py`). Inference loads the corrector in `streaming/vac_processor.py`, which also requires `qwen_asr`.

---

## Training data format

Examples are **JSONL** lines compatible with `SpeechLMCorrector/data_synthesize.py` outputs (paths under `SpeechLMCorrector/data/synthesized_data_v3/` in the default config).

Each record should include:

- **`audio_path`** — path to WAV/FLAC (16 kHz assumed at load time).
- **`k_best_candidates`** — list of strings (beam hypotheses for that chunk).
- **`continuation_transcript`** or **`correct_text`** — supervision target for the chunk.

Optional fields used by the dataset:

- **`previous_transcript`** — prior decoded context for streaming continuity.
- **`timestamp`**, **`chunk_size`** — used with `librosa.load(..., offset=..., duration=...)` so the waveform matches the labeled segment.

The collator masks loss on the prompt and supervises only the assistant reply (correct text + `<|im_end|>`).

---

## Model graph (what is actually trained)

`AutoModel.from_pretrained` returns **`Qwen3ASRForConditionalGeneration`**, which wraps a **`thinker`** submodule (`Qwen3ASRThinkerForConditionalGeneration`). Training operates on **`outer.thinker`** only:

- **`audio_tower`** — Whisper-style encoder; **frozen** (`freeze_encoder: true` and post-LoRA re-freeze of encoder-prefixed params).
- **Text decoder (`model`)** — **LoRA** injected via PEFT (`TaskType.CAUSAL_LM`) on attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` by default).

The outer wrapper is not trained directly; only the thinker participates in `Trainer`.

---

## Configuration (`training_config.yaml`)

Key blocks:

- **`model_name`** — typically `Qwen/Qwen3-ASR-0.6B`.
- **`train_data_paths`** — list of `{ path, weight }`. **`weight`** is an integer **repeat factor per epoch on the training split only**; validation always uses each example once (see `build_multi_source_datasets` in `data_utils.py`).
- **`val_split_ratio`** — fraction of **audio files** held out for validation (file-level split so chunks from one recording do not leak between train and eval).
- **`output_dir`** — where checkpoints and `final/` adapter are written (override as needed).
- **Trainer** — batch sizes, gradient accumulation, cosine LR, `eval_strategy` / `save_steps`, `max_seq_length`, etc.
- **LoRA** — `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`.

Paths in the shipped YAML are relative to the **`Qwen3ASRCorrector/`** directory because `train.sh` runs from there.

---

## Running fine-tuning

From the repo root (or any cwd; the script `cd`s into `Qwen3ASRCorrector/`):

**Single GPU**

```bash
bash Qwen3ASRCorrector/train.sh
```

**Multi-GPU (DDP via `torchrun`; preferred over naive DataParallel with bf16)**

```bash
NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash Qwen3ASRCorrector/train.sh
```

**Useful `train.sh` flags**

| Flag | Meaning |
|------|---------|
| `--config FILE` | Alternate YAML config. |
| `--adapter PATH` | Continue tuning from an existing LoRA adapter (`--load_adapter`). |
| `--resume CKPT_DIR` | Resume Trainer state from `checkpoint-*` (`trainer_state.json`, optimizer, etc.). |
| `--output DIR` | Override `output_dir` in YAML. |
| `--port PORT` | `MASTER_PORT` for multi-GPU (default `29500`). |

**Inspect module names before a long run**

```bash
cd Qwen3ASRCorrector
python training_qwen3asr.py --config training_config.yaml --probe
```

**Artifacts**

On rank 0, after training, the script saves **`output_dir/final/`** with the LoRA adapter and a copy of the **processor** — this directory is what you pass as **`ERROR_CORRECTOR_CKPT`** at inference.

---

## Inference (double Qwen3 ASR)

`runs/run_batch_eval_double_qwen3asr.sh` runs **`qwen3asr_streaming_vllm_beam_async.py`** on an audio directory with:

- **`MODEL`** — streaming ASR (default `Qwen/Qwen3-ASR-1.7B`).
- **`ERROR_CORRECTOR_CKPT`** — path to the trained **`final/`** (or a Trainer checkpoint that contains the adapter files LoRA expects).
- **`ERROR_CORRECTOR_BASE_MODEL`** — base weights for merging LoRA (default `Qwen/Qwen3-ASR-0.6B`).
- **`ERROR_CORRECTOR_TYPE`** — must be **`qwen3asr`** for this stack.

Loading is implemented in **`streaming/vac_processor.py`**: load base **`AutoModel`**, take **`thinker`**, attach **`PeftModel.from_pretrained(thinker, checkpoint_path)`**, move to CUDA, eval mode.

Example overrides:

```bash
cd runs
AUDIO_DIR=/path/to/wavs \
REFERENCE_FILE=/path/to/transcript.json \
ERROR_CORRECTOR_CKPT=/path/to/qwen3asr_corrector/final \
ERROR_CORRECTOR_BASE_MODEL=Qwen/Qwen3-ASR-0.6B \
ERROR_CORRECTOR_TYPE=qwen3asr \
bash run_batch_eval_double_qwen3asr.sh
```

Other variables (`WORKERS`, `GPUS`, `BEAMS`, `CHUNK_SIZE`, `OUTPUT_DIR`, …) behave like the standard batch vLLM scripts.

---

## Practical notes

- **Resume semantics**: If `trainer_state.json` shows `epoch >= num_train_epochs`, training exits immediately unless you extend epochs or resume from an earlier step (see warning logic in `training_qwen3asr.py`).
- **dtype**: Training uses bf16 when supported; the collator casts **`input_features`** to the model dtype so encoder conv layers do not see float32/bf16 mismatches.
- **Data synthesis**: To build new JSONL corpora, use the **`SpeechLMCorrector`** pipeline (`data_synthesize.py` and related docs/scripts); point `train_data_paths` in YAML at your new files.

For streaming ASR internals (beam async, vLLM), see `DOCUMENTS/QWEN3ASR_BEAM_ASYNC.md` and `DOCUMENTS/QWEN3ASR_VLLM_VARIANTS.md`.
