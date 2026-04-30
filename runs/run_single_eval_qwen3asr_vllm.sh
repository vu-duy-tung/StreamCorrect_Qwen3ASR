#!/bin/bash
# Single-file transcription with Qwen3-ASR-1.7B (vLLM async beam search)
# Optional error corrector enabled when ERROR_CORRECTOR_CKPT is set.

set -e

CONDA_BASE="$(conda info --base 2>/dev/null || echo /data/mino/anaconda3)"
PYTHON="${CONDA_BASE}/envs/StreamCorrect/bin/python"
[ ! -x "$PYTHON" ] && echo "ERROR: Python not found at $PYTHON" && exit 1

AUDIO_PATH="${AUDIO_PATH:-/data/mino/StreamCorrect/qwen3-asr-ft-dataset/m0_waihu_20250731/wav/test/0bf542938945463c92fa88a57b76418d.wav}"
REFERENCE_FILE="${REFERENCE_FILE:-/data/mino/StreamCorrect/qwen3-asr-ft-dataset/m0_waihu_20250731/transcript.json}"
MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/$(basename "${AUDIO_PATH%.*}")_qwen3asr}"
LANGUAGE="${LANGUAGE:-zh}"
CHUNK_SIZE="${CHUNK_SIZE:-0.5}"
INITIAL_BUFFER="${INITIAL_BUFFER:-1.0}"
BEAMS="${BEAMS:-4}"
GPU="${GPU:-1}"

ERROR_CORRECTOR_CKPT="${ERROR_CORRECTOR_CKPT:-/data/mino/model_ckpts/waihu_3/checkpoint-2646}"
ERROR_CORRECTOR_BASE_MODEL="${ERROR_CORRECTOR_BASE_MODEL:-fixie-ai/ultravox-v0_5-llama-3_2-1b}"
ERROR_CORRECTOR_TYPE="${ERROR_CORRECTOR_TYPE:-speechlm}"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "Audio:          $AUDIO_PATH"
echo "Model:          $MODEL"
echo "Language:       $LANGUAGE"
echo "Chunk size:     $CHUNK_SIZE s"
echo "Initial buffer: $INITIAL_BUFFER s"
echo "Beams:          $BEAMS"
echo "GPU:            $GPU"
echo "Error corrector:${ERROR_CORRECTOR_CKPT:- (disabled)}"
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="$PYTHON $REPO_ROOT/qwen3asr_streaming_vllm_beam_async.py \"$AUDIO_PATH\" \
    --model \"$MODEL\" \
    --language $LANGUAGE \
    --chunk-size $CHUNK_SIZE \
    --initial-buffer $INITIAL_BUFFER \
    --beams $BEAMS \
    --output-dir \"$OUTPUT_DIR\" \
    --reference-file \"$REFERENCE_FILE\" \
    --log-level INFO"

if [ -n "$ERROR_CORRECTOR_CKPT" ]; then
    CMD="$CMD --error-corrector-ckpt \"$ERROR_CORRECTOR_CKPT\""
    CMD="$CMD --error-corrector-base-model \"$ERROR_CORRECTOR_BASE_MODEL\""
    CMD="$CMD --error-corrector-type $ERROR_CORRECTOR_TYPE"
fi

eval $CMD
