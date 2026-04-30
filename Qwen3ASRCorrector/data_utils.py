"""Dataset and DataCollator for Qwen3-ASR error corrector training.

Qwen3ASRForConditionalGeneration is a CAUSAL LM, not seq2seq.
Audio features (mel spectrogram) are injected into the token-embedding stream at
<|audio_pad|> placeholder positions inside input_ids.

Training uses a single flat sequence:
  [system] [user: <audio_tokens> + candidates + context] [assistant: correct_text <|im_end|>]
  labels = [-100 for every prompt token] + [correct_text_ids ..., eos_id]

Loss is computed only on the response (correct_text + <|im_end|>).
"""

import json
import logging
import random as _random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an ASR error corrector. "
    "Listen to the audio and, given the k-best hypotheses below, "
    "output the single best corrected transcript for this audio segment."
)

_CAND_TMPL = "<candidates>\n{body}</candidates>"


def _build_candidates_text(candidates: list[str]) -> str:
    body = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates)) + "\n"
    return _CAND_TMPL.format(body=body)


# ── Dataset ──────────────────────────────────────────────────────────────────

class CorrectorDataset(Dataset):
    """Reads the synthesized JSONL produced by SpeechLMCorrector/data_synthesize.py.

    Required fields per line:
      audio_path             : str   – path to WAV/FLAC
      k_best_candidates      : list  – ASR top-k strings
      continuation_transcript: str   – ground-truth for this chunk
    Optional fields:
      previous_transcript    : str   – what was transcribed before this chunk
      timestamp              : float – end time of the chunk in seconds
      chunk_size             : float – duration of the chunk in seconds
    """

    def __init__(self, jsonl_path: str, sample_rate: int = 16_000):
        self.sample_rate = sample_rate
        self.examples: list[dict] = []
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        # Slice audio to the relevant chunk so the model hears only what it corrects.
        end_time = float(ex.get("timestamp", 0.0))
        chunk_size = float(ex.get("chunk_size", end_time))
        start_time = max(0.0, end_time - chunk_size)

        audio, _ = librosa.load(
            ex["audio_path"],
            sr=self.sample_rate,
            mono=True,
            offset=start_time,
            duration=chunk_size if chunk_size > 0 else None,
        )

        candidates: list[str] = ex.get("k_best_candidates", [ex.get("asr_text", "")])
        correct_text: str = ex.get("correct_text") or ex.get("continuation_transcript", "")
        previous: str = ex.get("previous_transcript", "")

        return {
            "audio": audio,
            "candidates": candidates,
            "correct_text": correct_text,
            "previous_transcript": previous,
        }


# ── Multi-source loader ───────────────────────────────────────────────────────

def build_multi_source_datasets(
    sources: list[dict],
    val_ratio: float = 0.05,
    seed: int = 42,
    sample_rate: int = 16_000,
) -> tuple["CorrectorDataset", "CorrectorDataset"]:
    """Load multiple weighted JSONL sources and return a clean file-level train/val split.

    Each entry in *sources* must have:
      - ``path``   : str  – path to the synthesized JSONL file
      - ``weight`` : int  – integer repeat multiplier applied to training examples
                            (weight=2 means each training example appears twice per epoch)

    Weights are applied **only to the training portion** so the validation metric
    is never inflated by repeated examples.  The val set always uses weight=1.

    The split is performed at the audio-file level (same as ``file_level_split``):
    all chunks from a given recording end up entirely in train or entirely in val,
    preventing the model from using ``previous_transcript`` or speaker identity as
    a shortcut.
    """
    # Step 1 – load each source as a plain list of example dicts.
    per_source: list[tuple[list[dict], int]] = []
    for src in sources:
        path = src["path"]
        weight = int(src.get("weight", 1))
        examples: list[dict] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        per_source.append((examples, weight))
        logger.info("Loaded %d examples from %s (weight=%d)", len(examples), path, weight)

    # Step 2 – collect every example keyed by audio file path (across all sources).
    # Value is a list of (example_dict, weight) tuples so we can apply the correct
    # weight when building the train split later.
    file_to_items: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for examples, weight in per_source:
        for ex in examples:
            file_to_items[ex["audio_path"]].append((ex, weight))

    # Step 3 – shuffle files and select val files.
    files = list(file_to_items.keys())
    rng = _random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_files = set(files[:n_val])
    train_files = set(files[n_val:])

    # Step 4 – build example lists.
    #   Val:   no repetition (weight=1) for a clean, unbiased eval metric.
    #   Train: repeat each example according to its source weight.
    train_examples: list[dict] = []
    val_examples: list[dict] = []
    for fpath, items in file_to_items.items():
        if fpath in val_files:
            val_examples.extend(ex for ex, _ in items)
        else:
            for ex, w in items:
                train_examples.extend([ex] * w)

    rng.shuffle(train_examples)

    logger.info(
        "Multi-source split: %d train files / %d val files  →  %d train samples / %d val samples",
        len(train_files), len(val_files), len(train_examples), len(val_examples),
    )

    # Step 5 – wrap in CorrectorDataset instances (bypass __init__ since examples
    # are already loaded).
    def _make_dataset(examples: list[dict]) -> CorrectorDataset:
        ds = CorrectorDataset.__new__(CorrectorDataset)
        ds.sample_rate = sample_rate
        ds.examples = examples
        return ds

    return _make_dataset(train_examples), _make_dataset(val_examples)


# ── DataCollator ──────────────────────────────────────────────────────────────

@dataclass
class CorrectorDataCollator:
    """Collates batches for causal LM training with Qwen3ASRForConditionalGeneration.

    The processor inserts the correct number of <|audio_pad|> tokens based on audio
    length, then tokenises the full prompt string. We append the response tokens and
    mask the prompt positions in labels with -100.

    Batch tensor keys passed to model.forward():
      input_ids             – (B, L_prompt + L_resp) – prompt + response
      attention_mask        – (B, L_prompt + L_resp) – 1 for real tokens
      labels                – (B, L_prompt + L_resp) – -100 for prompt, ids for response
      input_features        – (B, n_mels, T)         – audio mel spectrogram
      feature_attention_mask– (B, T)                 – audio padding mask
    """

    processor: Any
    max_seq_length: int = 512
    # If set, input_features is cast to this dtype before being returned.
    # Set to the model's floating-point dtype (e.g. torch.bfloat16) so that
    # DataParallel replicas and bf16 autocast both see the right tensor type.
    model_dtype: Optional[torch.dtype] = None

    def __post_init__(self) -> None:
        self.tok = self.processor.tokenizer
        # <|im_end|> closes each chat turn and is the response EOS.
        self.eos_id: int = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id: int = (
            self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_id
        )

    @property
    def audio_key(self) -> str:
        return "input_features"

    def _build_prompt_str(self, candidates: list[str], previous: str) -> str:
        """Build the full text prompt including audio placeholder and candidates.

        The processor's replace_multimodal_special_tokens() will expand the single
        <|audio_pad|> into N copies matching the actual audio length.
        """
        cands_txt = _build_candidates_text(candidates)
        user_body = "<|audio_start|><|audio_pad|><|audio_end|>\n"
        if previous:
            user_body += f"Previous: {previous}\n"
        user_body += cands_txt
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_body}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        audios = [f["audio"] for f in features]
        prompts = [
            self._build_prompt_str(f["candidates"], f["previous_transcript"])
            for f in features
        ]

        # processor expands <|audio_pad|> tokens, tokenises, and left-pads to the
        # longest prompt in the batch.
        proc_out = self.processor(
            text=prompts,
            audio=audios,
            sampling_rate=16_000,
            padding=True,
            return_tensors="pt",
        )

        prompt_ids: torch.Tensor = proc_out["input_ids"]           # (B, L_p)
        prompt_mask: torch.Tensor = proc_out["attention_mask"]     # (B, L_p)
        input_features: torch.Tensor = proc_out["input_features"]  # (B, n_mels, T)
        feat_mask: torch.Tensor | None = proc_out.get("feature_attention_mask")

        # The processor always returns float32. Cast to match the model dtype so
        # bfloat16 conv layers in the audio encoder don't see a type mismatch.
        if self.model_dtype is not None:
            input_features = input_features.to(self.model_dtype)

        B, L_p = prompt_ids.shape

        # Tokenise each response: correct_text + <|im_end|>
        resp_ids_list: list[list[int]] = []
        for f in features:
            ids = self.tok.encode(f["correct_text"], add_special_tokens=False)
            resp_ids_list.append(ids + [self.eos_id])

        max_resp = max(len(r) for r in resp_ids_list)
        max_total = min(L_p + max_resp, self.max_seq_length)

        full_ids = torch.full((B, max_total), self.pad_id, dtype=torch.long)
        full_labels = torch.full((B, max_total), -100, dtype=torch.long)
        full_mask = torch.zeros(B, max_total, dtype=torch.long)

        for i, resp_ids in enumerate(resp_ids_list):
            # Prompt block (processor left-pads, so attention_mask marks real tokens)
            p_end = min(L_p, max_total)
            full_ids[i, :p_end] = prompt_ids[i, :p_end]
            full_mask[i, :p_end] = prompt_mask[i, :p_end]
            # labels for prompt stay -100 (masked)

            # Response block appended right after the prompt block.
            # Clamp r_start to (max_total - 1) so at least 1 response token
            # always fits. Without this guard, when the prompt fills the entire
            # budget (L_p >= max_total), r_end - r_start can be ≤ 0, leaving
            # all labels as -100 and producing nan loss via CE(x, all_ignored).
            r_start = min(L_p, max_total - 1)
            r_end = min(r_start + len(resp_ids), max_total)
            r_slice = resp_ids[: r_end - r_start]
            if r_slice:
                t = torch.tensor(r_slice, dtype=torch.long)
                full_ids[i, r_start:r_end] = t
                full_labels[i, r_start:r_end] = t
                full_mask[i, r_start:r_end] = 1

        result: dict[str, torch.Tensor] = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "labels": full_labels,
            "input_features": input_features,
        }
        if feat_mask is not None:
            result["feature_attention_mask"] = feat_mask
        return result
