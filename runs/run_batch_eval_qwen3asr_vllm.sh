#!/bin/bash
# Batch transcription with Qwen3-ASR-1.7B (vLLM async beam search)
# Supports parallel processing across multiple GPUs.
# Optional error corrector enabled when ERROR_CORRECTOR_CKPT is set.

set -e

CONDA_BASE="$(conda info --base 2>/dev/null || echo /data/mino/anaconda3)"
PYTHON="${CONDA_BASE}/envs/StreamCorrect/bin/python"
[ ! -x "$PYTHON" ] && echo "ERROR: Python not found at $PYTHON" && exit 1

TMP_RECOGNIZER_DIR=""
cleanup() {
    if [ -n "$TMP_RECOGNIZER_DIR" ] && [ -d "$TMP_RECOGNIZER_DIR" ]; then
        rm -rf "$TMP_RECOGNIZER_DIR"
    fi
}
trap cleanup EXIT

AUDIO_DIR="${AUDIO_DIR:-/data/mino/StreamCorrect/qwen3-asr-ft-dataset/m0_waihu_20250731/wav/test}"
REFERENCE_FILE="${REFERENCE_FILE:-/data/mino/StreamCorrect/qwen3-asr-ft-dataset/m0_waihu_20250731/transcript.json}"
MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
RECOGNIZER_CKPT="${RECOGNIZER_CKPT:-/data/mino/StreamCorrect/qwen3-asr-ft/checkpoint-204}"
RECOGNIZER_BASE_PROCESSOR="${RECOGNIZER_BASE_PROCESSOR:-$MODEL}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/batch_qwen3asr_offline}"
LANGUAGE="${LANGUAGE:-zh}"
CHUNK_SIZE="${CHUNK_SIZE:-0.5}"
INITIAL_BUFFER="${INITIAL_BUFFER:-1.0}"
BEAMS="${BEAMS:-4}"
MAX_FILES="${MAX_FILES:-1000}"
WORKERS="${WORKERS:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT-/data/mino/model_ckpts/waihu_offline_v3/checkpoint-3708}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-fixie-ai/ultravox-v0_5-llama-3_2-1b}"
ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-speechlm}"

ASR_MODEL="$MODEL"
if [ -n "$RECOGNIZER_CKPT" ]; then
    # vLLM Qwen3-ASR loading requires a directory with preprocessor_config.json.
    # Training checkpoints like checkpoint-XXX may not contain processor files.
    RECOGNIZER_CANDIDATES=(
        "$RECOGNIZER_CKPT"
        "$RECOGNIZER_CKPT/final"
        "$(dirname "$RECOGNIZER_CKPT")/final"
        "$(dirname "$RECOGNIZER_CKPT")"
    )

    RESOLVED_RECOGNIZER=""
    for candidate in "${RECOGNIZER_CANDIDATES[@]}"; do
        if [ -f "$candidate/preprocessor_config.json" ]; then
            RESOLVED_RECOGNIZER="$candidate"
            break
        fi
    done

    if [ -n "$RESOLVED_RECOGNIZER" ]; then
        ASR_MODEL="$RESOLVED_RECOGNIZER"
    else
        if [ ! -d "$RECOGNIZER_CKPT" ]; then
            echo "ERROR: RECOGNIZER_CKPT does not exist or is not a directory: $RECOGNIZER_CKPT" >&2
            exit 1
        fi

        echo "Recognizer checkpoint has no preprocessor files; bootstrapping from base processor..." >&2
        TMP_RECOGNIZER_DIR="$(mktemp -d /tmp/qwen3asr_recognizer_XXXXXX)"
        cp -a "$RECOGNIZER_CKPT/." "$TMP_RECOGNIZER_DIR/"

        # AutoProcessor.save_pretrained() for Qwen3-ASR may not emit
        # preprocessor_config.json. Copy required processor/tokenizer files
        # explicitly from the base model source (local dir or HF repo id).
        "$PYTHON" - <<PY || {
import os
import shutil

src = r"""$RECOGNIZER_BASE_PROCESSOR"""
dst = r"""$TMP_RECOGNIZER_DIR"""

required_or_useful = [
    "chat_template.json",
    "preprocessor_config.json",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
]

if os.path.isdir(src):
    for name in required_or_useful:
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, name))
else:
    from huggingface_hub import hf_hub_download
    for name in required_or_useful:
        try:
            cached = hf_hub_download(repo_id=src, filename=name)
        except Exception:
            continue
        shutil.copy2(cached, os.path.join(dst, name))

if not os.path.isfile(os.path.join(dst, "preprocessor_config.json")):
    raise SystemExit(2)
PY
            echo "ERROR: Failed to bootstrap processor files from RECOGNIZER_BASE_PROCESSOR=$RECOGNIZER_BASE_PROCESSOR" >&2
            echo "Checked for preprocessor_config.json in:" >&2
            for candidate in "${RECOGNIZER_CANDIDATES[@]}"; do
                echo "  - $candidate" >&2
            done
            rm -rf "$TMP_RECOGNIZER_DIR"
            exit 1
        }

        if [ ! -f "$TMP_RECOGNIZER_DIR/preprocessor_config.json" ]; then
            echo "ERROR: Bootstrap succeeded but preprocessor_config.json is still missing in $TMP_RECOGNIZER_DIR" >&2
            rm -rf "$TMP_RECOGNIZER_DIR"
            exit 1
        fi

        ASR_MODEL="$TMP_RECOGNIZER_DIR"
    fi
fi

echo "Audio dir:      $AUDIO_DIR"
echo "Model:          $ASR_MODEL"
echo "Recognizer ckpt:${RECOGNIZER_CKPT:- (disabled)}"
echo "Recog base proc:${RECOGNIZER_BASE_PROCESSOR}"
echo "Language:       $LANGUAGE"
echo "Chunk size:     $CHUNK_SIZE s"
echo "Initial buffer: $INITIAL_BUFFER s"
echo "Beams:          $BEAMS"
echo "Workers:        $WORKERS"
echo "GPUs:           $GPUS"
echo "Error corrector:${ERROR_CORRECTOR_CKPT:- (disabled)}"
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="$PYTHON $REPO_ROOT/qwen3asr_streaming_vllm_beam_async.py \"$AUDIO_DIR\" \
    --model \"$ASR_MODEL\" \
    --language $LANGUAGE \
    --chunk-size $CHUNK_SIZE \
    --initial-buffer $INITIAL_BUFFER \
    --beams $BEAMS \
    --workers $WORKERS \
    --gpus $GPUS \
    --output-dir \"$OUTPUT_DIR\" \
    --reference-file \"$REFERENCE_FILE\" \
    --log-level INFO"

[ -n "$MAX_FILES" ] && CMD="$CMD --max-files $MAX_FILES"

if [ -n "$ERROR_CORRECTOR_CKPT" ]; then
    CMD="$CMD --error-corrector-ckpt \"$ERROR_CORRECTOR_CKPT\""
    CMD="$CMD --error-corrector-base-model \"$ERROR_CORRECTOR_BASE_MODEL\""
    CMD="$CMD --error-corrector-type $ERROR_CORRECTOR_TYPE"
fi

eval $CMD

echo ""
echo "Results saved to: $OUTPUT_DIR"
