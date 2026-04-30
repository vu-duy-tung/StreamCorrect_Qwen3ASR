#!/usr/bin/env bash
# Launch Qwen3-ASR-0.6B error corrector training.
# Run from the repo root or the Qwen3ASRCorrector/ directory.
#
# Single-GPU:
#   bash Qwen3ASRCorrector/train.sh
#
# Multi-GPU (DDP via torchrun — always prefer this over plain python with
# multiple CUDA_VISIBLE_DEVICES, because DataParallel breaks bf16 autocast):
#   NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash Qwen3ASRCorrector/train.sh
# Continue fine-tuning from a saved adapter:
#   bash Qwen3ASRCorrector/train.sh --adapter /path/to/adapter
# Resume optimizer/scheduler state from trainer checkpoint:
#   bash Qwen3ASRCorrector/train.sh --resume /path/to/checkpoint-1200
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

CONFIG="training_config.yaml"
ADAPTER=""
RESUME_CHECKPOINT=""
OUTPUT_DIR=""
PORT="${MASTER_PORT:-29500}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2"; shift 2 ;;
        --adapter)  ADAPTER="$2"; shift 2 ;;
        --resume)   RESUME_CHECKPOINT="$2"; shift 2 ;;
        --output)   OUTPUT_DIR="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--config FILE] [--adapter PATH] [--resume CKPT_DIR] [--output DIR] [--port PORT]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ARGS="--config $CONFIG"
[ -n "$ADAPTER" ] && ARGS="$ARGS --load_adapter $ADAPTER"
[ -n "$RESUME_CHECKPOINT" ] && ARGS="$ARGS --resume_from_checkpoint $RESUME_CHECKPOINT"
[ -n "$OUTPUT_DIR" ] && ARGS="$ARGS --output_dir $OUTPUT_DIR"

# ── Optional: probe model layer names before a full run ──────────────────────
# python training_qwen3asr.py --config "$CONFIG" --probe; exit 0

# ── Determine GPU count ───────────────────────────────────────────────────────
NPROC="${NPROC:-1}"
if [ "$NPROC" -gt 1 ]; then
    echo "Multi-GPU: launching $NPROC workers via torchrun (DDP)."
    torchrun \
        --nproc_per_node="$NPROC" \
        --master_port="$PORT" \
        training_qwen3asr.py $ARGS
else
    echo "Single-GPU: launching python."
    python training_qwen3asr.py $ARGS
fi
