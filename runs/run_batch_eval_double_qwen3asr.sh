#!/bin/bash
# Batch transcription with Qwen3-ASR-1.7B (vLLM async beam search)
# using Qwen3-ASR-0.6B LoRA as the error corrector ("double Qwen3ASR").
# Supports parallel processing across multiple GPUs.

set -e

CONDA_BASE="$(conda info --base 2>/dev/null || echo /data/mino/anaconda3)"
PYTHON="${CONDA_BASE}/envs/StreamCorrect/bin/python"
[ ! -x "$PYTHON" ] && echo "ERROR: Python not found at $PYTHON" && exit 1

AUDIO_DIR="${AUDIO_DIR:-/data/mino/StreamCorrect/qwen3-asr-ft-dataset/m0_waihu_20250731/wav/test}"
REFERENCE_FILE="${REFERENCE_FILE:-/data/mino/StreamCorrect/qwen3-asr-ft-dataset/m0_waihu_20250731/transcript.json}"
MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/batch_qwen3asr_double}"
LANGUAGE="${LANGUAGE:-zh}"
CHUNK_SIZE="${CHUNK_SIZE:-0.5}"
INITIAL_BUFFER="${INITIAL_BUFFER:-1.0}"
BEAMS="${BEAMS:-4}"
MAX_FILES="${MAX_FILES:-1000}"
WORKERS="${WORKERS:-6}"
GPUS="${GPUS:-0,1,2,3,4,5}"

ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-/data/mino/model_ckpts/qwen3asr_corrector_v2/final}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-Qwen/Qwen3-ASR-0.6B}"
ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-qwen3asr}"

echo "Audio dir:       $AUDIO_DIR"
echo "Model:           $MODEL"
echo "Language:        $LANGUAGE"
echo "Chunk size:      $CHUNK_SIZE s"
echo "Initial buffer:  $INITIAL_BUFFER s"
echo "Beams:           $BEAMS"
echo "Workers:         $WORKERS"
echo "GPUs:            $GPUS"
echo "Error corrector: $ERROR_CORRECTOR_CKPT"
echo "Corrector base:  $ERROR_CORRECTOR_BASE_MODEL"
echo "Corrector type:  $ERROR_CORRECTOR_TYPE"
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="$PYTHON $REPO_ROOT/qwen3asr_streaming_vllm_beam_async.py \"$AUDIO_DIR\" \
    --model \"$MODEL\" \
    --language $LANGUAGE \
    --chunk-size $CHUNK_SIZE \
    --initial-buffer $INITIAL_BUFFER \
    --beams $BEAMS \
    --workers $WORKERS \
    --gpus $GPUS \
    --output-dir \"$OUTPUT_DIR\" \
    --reference-file \"$REFERENCE_FILE\" \
    --error-corrector-ckpt \"$ERROR_CORRECTOR_CKPT\" \
    --error-corrector-base-model \"$ERROR_CORRECTOR_BASE_MODEL\" \
    --error-corrector-type $ERROR_CORRECTOR_TYPE \
    --log-level INFO"

[ -n "$MAX_FILES" ] && CMD="$CMD --max-files $MAX_FILES"

eval $CMD

echo ""
echo "Results saved to: $OUTPUT_DIR"
