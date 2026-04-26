#!/bin/bash
# Batch transcription with Qwen3-ASR-1.7B (vLLM async beam search)
# Supports parallel processing across multiple GPUs.
# Optional error corrector enabled when ERROR_CORRECTOR_CKPT is set.

set -e

CONDA_BASE="$(conda info --base 2>/dev/null || echo /data/mino/anaconda3)"
PYTHON="${CONDA_BASE}/envs/StreamCorrect/bin/python"
[ ! -x "$PYTHON" ] && echo "ERROR: Python not found at $PYTHON" && exit 1

AUDIO_DIR="${AUDIO_DIR:-/data/mino/StreamCorrect/qwen3-asr-ft-dataset/m0_waihu_20250731/wav/test}"
REFERENCE_FILE="${REFERENCE_FILE:-/data/mino/StreamCorrect/qwen3-asr-ft-dataset/m0_waihu_20250731/transcript.json}"
MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/batch_qwen3asr}"
LANGUAGE="${LANGUAGE:-zh}"
CHUNK_SIZE="${CHUNK_SIZE:-0.5}"
INITIAL_BUFFER="${INITIAL_BUFFER:-1.0}"
BEAMS="${BEAMS:-4}"
MAX_FILES="${MAX_FILES:-}"
WORKERS="${WORKERS:-5}"
GPUS="${GPUS:-0,1,2,3,4}"

ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-fixie-ai/ultravox-v0_5-llama-3_2-1b}"
ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-speechlm}"

echo "Audio dir:      $AUDIO_DIR"
echo "Model:          $MODEL"
echo "Language:       $LANGUAGE"
echo "Chunk size:     $CHUNK_SIZE s"
echo "Initial buffer: $INITIAL_BUFFER s"
echo "Beams:          $BEAMS"
echo "Workers:        $WORKERS"
echo "GPUs:           $GPUS"
echo "Error corrector:${ERROR_CORRECTOR_CKPT:- (disabled)}"
echo ""

CMD="$PYTHON ../qwen3asr_streaming_vllm_beam_async.py \"$AUDIO_DIR\" \
    --model \"$MODEL\" \
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
