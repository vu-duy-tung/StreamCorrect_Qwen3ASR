#!/usr/bin/env python3
"""Fine-tune Qwen3-ASR as an ASR error corrector (LoRA on the text decoder).

Architecture notes
------------------
AutoModel.from_pretrained returns Qwen3ASRForConditionalGeneration, which is
an INFERENCE wrapper containing a `thinker` attribute of type
Qwen3ASRThinkerForConditionalGeneration. Only the thinker implements forward()
and is suitable for training.

Training graph
  outer model (Qwen3ASRForConditionalGeneration)
      └── thinker  ← the model we actually train
              ├── audio_tower  (Whisper-style encoder — frozen)
              └── model        (Qwen3 text decoder — LoRA applied here)

LoRA is applied to the thinker; the audio_tower is frozen throughout.

Usage:
    python training_qwen3asr.py --config training_config.yaml
"""

import argparse
import json
import logging
import os
import random as _random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Subset
from transformers import (
    AutoModel,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data_utils import CorrectorDataCollator, CorrectorDataset, build_multi_source_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_qwen3asr")


# ── Config helper ─────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(cfg: dict):
    model_name: str = cfg["model_name"]

    # qwen_asr registers Qwen3ASRForConditionalGeneration with AutoModel.
    # Without this import, from_pretrained raises ValueError for qwen3_asr.
    try:
        import qwen_asr as _qwen_asr  # noqa: F401
        logger.info("qwen_asr imported — qwen3_asr architecture registered.")
    except ImportError:
        logger.warning(
            "qwen_asr not found. Loading may fail if transformers does not "
            "know the qwen3_asr architecture natively."
        )

    logger.info("Loading processor for %s …", model_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info("Loading outer model %s …", model_name)
    # Do NOT use trust_remote_code=True here: qwen_asr already registered the
    # class with AutoModel. trust_remote_code would cause transformers to
    # download the Hub stub class instead, which has no forward() implementation.
    outer = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Extract the thinker: it is Qwen3ASRThinkerForConditionalGeneration and
    # is the only submodule that implements forward() and computes the LM loss.
    model = outer.thinker
    logger.info("Training model class: %s", type(model).__name__)

    return model, processor


# ── Encoder freezing ──────────────────────────────────────────────────────────

def freeze_encoder(model) -> None:
    """Freeze audio encoder (audio_tower) so it never receives gradients."""
    frozen = 0
    for name, param in model.named_parameters():
        if _is_encoder_param(name):
            param.requires_grad = False
            frozen += 1
    logger.info("Froze %d encoder parameter tensors.", frozen)


def _is_encoder_param(name: str) -> bool:
    # Inside Qwen3ASRThinkerForConditionalGeneration the Whisper encoder lives
    # under the audio_tower prefix. The broader list catches any sub-modules.
    encoder_prefixes = ("audio_tower", "audio_encoder", "encoder", "feature_extractor", "conv")
    return any(name.startswith(p) or f".{p}." in name for p in encoder_prefixes)


# ── LoRA ──────────────────────────────────────────────────────────────────────

def apply_lora(model, cfg: dict):
    target_modules: list[str] = cfg.get(
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # peft may have wrapped encoder attention projections that share names like
    # q_proj. Re-freeze those adapters so they never get gradients.
    re_frozen = 0
    for name, param in model.named_parameters():
        if _is_encoder_param(name):
            param.requires_grad = False
            re_frozen += 1
    logger.info("Re-froze %d encoder LoRA tensors after peft wrapping.", re_frozen)

    model.print_trainable_parameters()
    return model


def load_existing_adapter(model, adapter_path: str):
    """Load an existing LoRA adapter and keep it trainable for continued tuning."""
    logger.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    # Keep encoder frozen even when adapter weights are loaded.
    re_frozen = 0
    for name, param in model.named_parameters():
        if _is_encoder_param(name):
            param.requires_grad = False
            re_frozen += 1
    logger.info("Re-froze %d encoder tensors after adapter load.", re_frozen)

    model.print_trainable_parameters()
    return model


# ── File-level split ──────────────────────────────────────────────────────────

def file_level_split(dataset: CorrectorDataset, val_ratio: float = 0.05, seed: int = 42):
    """Split by audio file so no file's chunks appear in both train and eval.

    Sample-level random_split lets adjacent chunks from the same recording end
    up in both splits, giving the model a memorisation shortcut via
    previous_transcript and speaker identity. Splitting at file level prevents
    this and produces a cleaner generalisation signal.

    Returns (train_subset, eval_subset) as torch.utils.data.Subset objects.
    """
    file_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(dataset.examples):
        file_to_indices[ex["audio_path"]].append(i)

    files = list(file_to_indices.keys())
    rng = _random.Random(seed)
    rng.shuffle(files)

    n_val_files = max(1, int(len(files) * val_ratio))
    val_files = set(files[:n_val_files])

    train_indices, val_indices = [], []
    for f, idxs in file_to_indices.items():
        (val_indices if f in val_files else train_indices).extend(idxs)

    logger.info(
        "File-level split: %d train files / %d val files → %d train samples / %d val samples",
        len(files) - n_val_files, n_val_files, len(train_indices), len(val_indices),
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ── Training ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training_config.yaml")
    parser.add_argument(
        "--load_adapter",
        default=None,
        help="Path to an existing LoRA adapter to continue fine-tuning from.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a Trainer checkpoint directory (checkpoint-XXXX) for stateful resume.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output_dir in config.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Load model, print layer names, then exit.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    set_seed(cfg.get("seed", 42))

    model, processor = load_model_and_processor(cfg)

    if args.probe:
        print("\n=== Named modules (thinker) ===")
        for name, _ in model.named_modules():
            print(" ", name)
        sys.exit(0)

    # ── Data ─────────────────────────────────────────────────────────────────
    val_ratio = cfg.get("val_split_ratio", 0.05)
    seed = cfg.get("seed", 42)
    val_path: Optional[str] = cfg.get("val_data_path")

    if "train_data_paths" in cfg:
        # Multiple weighted sources — file-level split with weights applied only
        # to the training portion (val always weight=1 for an unbiased metric).
        train_dataset, eval_dataset = build_multi_source_datasets(
            sources=cfg["train_data_paths"],
            val_ratio=val_ratio,
            seed=seed,
            sample_rate=16_000,
        )
    else:
        # Backward-compatible single-path mode.
        full_dataset = CorrectorDataset(cfg["data_path"], sample_rate=16_000)
        if val_path and Path(val_path).exists():
            train_dataset = full_dataset
            eval_dataset = CorrectorDataset(val_path, sample_rate=16_000)
        else:
            train_dataset, eval_dataset = file_level_split(
                full_dataset, val_ratio=val_ratio, seed=seed,
            )

    logger.info("Train: %d  Val: %d", len(train_dataset), len(eval_dataset))

    # ── Collator ─────────────────────────────────────────────────────────────
    # Determine the model's floating-point dtype so the collator can cast
    # input_features (always float32 from the processor) to match — required
    # for bfloat16 conv layers in the audio encoder.
    model_dtype = next(
        (p.dtype for p in model.parameters() if p.is_floating_point()), None
    )
    logger.info("Model dtype: %s", model_dtype)
    collator = CorrectorDataCollator(
        processor=processor,
        max_seq_length=cfg.get("max_seq_length", 512),
        model_dtype=model_dtype,
    )
    logger.info("Audio tensor key: '%s'", collator.audio_key)

    # ── Freeze encoder + apply LoRA ───────────────────────────────────────────
    if cfg.get("freeze_encoder", True):
        freeze_encoder(model)

    if args.load_adapter:
        model = load_existing_adapter(model, args.load_adapter)
    else:
        model = apply_lora(model, cfg)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    output_dir: str = cfg.get("output_dir", "./checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 200),
        save_strategy=cfg.get("eval_strategy", "steps"),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        logging_steps=cfg.get("logging_steps", 10),
        report_to=cfg.get("report_to", "none"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    if args.resume_from_checkpoint:
        state_file = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        if os.path.isfile(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as fh:
                    state = json.load(fh)
                resumed_epoch = float(state.get("epoch", 0.0))
                target_epochs = float(training_args.num_train_epochs)
                if resumed_epoch >= target_epochs:
                    logger.warning(
                        "resume checkpoint epoch=%.3f already reached target num_train_epochs=%.3f; "
                        "training will exit immediately unless you increase num_train_epochs or start from an earlier checkpoint.",
                        resumed_epoch,
                        target_epochs,
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not read %s (%s). Continuing.", state_file, exc)

    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save LoRA adapter + processor only on global rank 0.
    if trainer.is_world_process_zero():
        # Avoid peft warning when adapter config misses base model path.
        if hasattr(model, "peft_config"):
            for peft_cfg in model.peft_config.values():
                if not getattr(peft_cfg, "base_model_name_or_path", None):
                    peft_cfg.base_model_name_or_path = cfg["model_name"]

        final_dir = os.path.join(output_dir, "final")
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        logger.info("Saved LoRA adapter to %s", final_dir)
        logger.info(
            "To load at inference: outer=AutoModel.from_pretrained('%s', ...); "
            "from peft import PeftModel; outer.thinker = PeftModel.from_pretrained(outer.thinker, '%s')",
            cfg["model_name"], final_dir,
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
