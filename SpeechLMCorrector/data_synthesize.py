"""Synthesize error-correction training data from Qwen3-ASR beam histories.

Pipeline
--------
1. Collect .wav files from a training folder.
2. Run batch streaming inference (no error corrector) to produce per-file beam histories.
3. Align each beam candidate against the ground-truth reference and emit
   (previous_transcript, k_best_candidates, continuation_transcript) triples
   for use in SpeechLM / LM corrector training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import unicodedata
from typing import Any

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def run_batch_inference(
    audio_paths: list[str],
    reference_file: str,
    batch_script: str,
    batch_output_dir: str,
    chunk_size: int,
    num_candidates: int,
    gpus: str,
    num_workers: int,
    initial_buffer: float,
) -> None:
    """Run batch streaming inference, writing per-file beam histories to batch_output_dir.

    A temporary symlink directory is created so the batch script sees only the
    desired audio files (it otherwise consumes every .wav under AUDIO_DIR).
    """
    link_dir = os.path.join(batch_output_dir, "_audio_links")
    if os.path.exists(link_dir):
        shutil.rmtree(link_dir)
    os.makedirs(link_dir, exist_ok=True)

    for ap in audio_paths:
        abs_target = os.path.abspath(ap)
        link_name = os.path.join(link_dir, os.path.basename(ap))
        if os.path.exists(link_name):
            h = hex(abs(hash(abs_target)))[2:10]
            base, ext = os.path.splitext(os.path.basename(ap))
            link_name = os.path.join(link_dir, f"{base}__{h}{ext}")
        os.symlink(abs_target, link_name)

    env = {
        **os.environ,
        "AUDIO_DIR": link_dir,
        "REFERENCE_FILE": reference_file,
        "OUTPUT_DIR": batch_output_dir,
        "NUM_FILES": str(len(audio_paths)),
        "VAC_CHUNK_SIZE": str(chunk_size / 1000.0),
        "BEAM_SIZE": str(num_candidates),
        "NUM_WORKERS": str(num_workers),
        "GPUS": gpus,
        "USE_ERROR_CORRECTOR": "false",
        "HF_HUB_OFFLINE": "1",
        "INITIAL_BUFFER": str(initial_buffer),
    }
    print(f"  AUDIO_DIR:   {link_dir}")
    print(f"  OUTPUT_DIR:  {batch_output_dir}")
    print(f"  GPUS: {gpus}  NUM_WORKERS: {num_workers}  files: {len(audio_paths)}")

    result = subprocess.run(["bash", batch_script], env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Batch script {batch_script} failed (exit {result.returncode})")


def load_per_file_outputs(
    batch_output_dir: str,
) -> dict[str, tuple[str | None, list[dict[str, Any]]]]:
    """Return {audio_basename: (reference_or_None, beam_history)} for every output file."""
    ref_by_basename: dict[str, str] = {}
    eval_path = os.path.join(batch_output_dir, "evaluation_results.json")
    if os.path.exists(eval_path):
        try:
            with open(eval_path, encoding="utf-8") as f:
                eval_data = json.load(f)
            for item in eval_data.get("per_file_results", []):
                if item.get("file") and item.get("reference"):
                    ref_by_basename[item["file"]] = item["reference"]
        except Exception as e:
            print(f"Warning: could not load evaluation_results.json: {e}")

    results: dict[str, tuple[str | None, list[dict[str, Any]]]] = {}
    for fname in os.listdir(batch_output_dir):
        if not fname.endswith("_beam_history.json"):
            continue
        try:
            with open(os.path.join(batch_output_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {fname}: {e}")
            continue
        audio_path = data.get("audio_path", "")
        if not audio_path:
            continue
        basename = os.path.basename(audio_path)
        results[basename] = (ref_by_basename.get(basename), data.get("history", []))
    return results


# ---------------------------------------------------------------------------
# Text normalization + alignment
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def _normalize_text(s: str) -> str:
    """Canonical form used for both alignment and training labels.

    Steps: strip ASR special tokens → drop U+FFFD → NFKC → drop whitespace
    and Unicode punctuation → lowercase ASCII letters.
    """
    if not s:
        return ""
    s = _SPECIAL_TOKEN_RE.sub("", s)
    s = s.replace("\ufffd", "")
    s = unicodedata.normalize("NFKC", s)
    out: list[str] = []
    for ch in s:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        out.append(ch.lower() if ch.isascii() and ch.isalpha() else ch)
    return "".join(out)


def _align_prev_end(norm_prev: str, norm_ref: str) -> int:
    """Return j* in [0, len(norm_ref)] that minimises edit_dist(norm_prev, norm_ref[:j*]).

    Semi-global Needleman-Wunsch: norm_prev is fully consumed, norm_ref end is free.
    Ties broken toward j closest to len(norm_prev).
    """
    n, m = len(norm_prev), len(norm_ref)
    if not n or not m:
        return 0
    prev_row = list(range(m + 1))
    for i, pc in enumerate(norm_prev, start=1):
        curr = [i] + [0] * m
        for j, rc in enumerate(norm_ref, start=1):
            curr[j] = min(
                prev_row[j - 1] + (0 if pc == rc else 1),
                prev_row[j] + 1,
                curr[j - 1] + 1,
            )
        prev_row = curr
    best_j, best_val = 0, prev_row[0]
    for j in range(1, m + 1):
        v = prev_row[j]
        if v < best_val or (v == best_val and abs(j - n) < abs(best_j - n)):
            best_val, best_j = v, j
    return best_j


def synthesize_samples(
    audio_path: str,
    ref: str,
    beam_history: list[dict[str, Any]],
    num_candidates: int,
    chunk_size: int,
    initial_buffer: float,
) -> list[dict[str, Any]]:
    """Convert one file's beam history into error-correction training samples.

    For each history entry:
      1. Normalize prev and top-k candidates.
      2. Compute chunk duration (end_time of current entry minus end_time of previous).
         For the first entry the duration equals end_time directly.
      3a. Short chunk (duration < initial_buffer for first entry, < chunk_size_s for
          later entries): audio has run out after this chunk, so use norm_ref[end_pos:]
          as the continuation (everything from the alignment point to the end).
      3b. Normal chunk: align norm_prev against norm_ref, slice pred_len chars.
    """
    if not ref:
        return []
    norm_ref = _normalize_text(ref)
    if not norm_ref:
        return []

    chunk_size_s = chunk_size / 1000.0
    samples: list[dict[str, Any]] = []
    for i, entry in enumerate(beam_history):
        norm_prev = _normalize_text(entry.get("previous_transcript") or "")
        norm_topk = [
            _normalize_text(t)
            for t in (entry.get("topk") or [])
            if isinstance(t, str)
        ]
        norm_topk = [t for t in norm_topk[:num_candidates] if t]
        if not norm_topk:
            continue

        end_time = float(entry.get("end_time", 0.0))
        prev_end_time = float(beam_history[i - 1].get("end_time", 0.0)) if i > 0 else 0.0
        chunk_duration = end_time - prev_end_time
        threshold = initial_buffer if i == 0 else chunk_size_s
        is_short_chunk = chunk_duration < threshold

        end_pos = _align_prev_end(norm_prev, norm_ref)
        if end_pos >= len(norm_ref):
            continue

        if is_short_chunk:
            continuation = norm_ref[end_pos:]
        else:
            pred_len = len(norm_topk[0]) - len(norm_prev)
            if pred_len <= 0:
                continue
            continuation = norm_ref[end_pos : end_pos + pred_len]
            if not continuation:
                continue

        samples.append({
            "k_best_candidates": norm_topk,
            "num_candidates": num_candidates,
            "chunk_size": chunk_size,
            "previous_transcript": norm_prev,
            "continuation_transcript": continuation,
            "audio_path": audio_path,
            "timestamp": float(entry.get("end_time", 0.0)),
        })
    return samples


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _append_samples(samples: list[dict[str, Any]], output_path: str) -> None:
    if not samples:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_existing_audio_paths(jsonl_path: str) -> list[str]:
    """Return audio paths already present in an existing output .jsonl file."""
    audio_paths: set[str] = set()
    if not os.path.exists(jsonl_path):
        return []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                s = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            audio_paths.add(s["audio_path"])
    return list(audio_paths)


def collect_audio_paths(folder: str) -> list[str]:
    """Collect .wav files from folder and its immediate subdirectories."""
    paths: list[str] = []
    for entry in os.listdir(folder):
        full = os.path.join(folder, entry)
        if os.path.isdir(full):
            paths.extend(
                os.path.join(full, f) for f in os.listdir(full) if f.endswith(".wav")
            )
        elif entry.endswith(".wav"):
            paths.append(full)
    return paths


def _log_failed(audio_path: str, log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(audio_path + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synthesize SpeechLM corrector training data from Qwen3-ASR beam histories."
    )
    p.add_argument("--audio-dir", required=True,
                   help="Folder of .wav training files (searched one level deep)")
    p.add_argument("--reference-file", required=True,
                   help="transcript.json with ground-truth references")
    p.add_argument("--output-dir", default="SpeechLMCorrector/data/sample_custom_data",
                   help="Directory for output .jsonl and temporary batch outputs")
    p.add_argument("--output-file", default="waihu_qwen3asr.jsonl",
                   help="Output filename inside --output-dir")
    p.add_argument("--batch-script", default="runs/run_batch_eval_qwen3asr_vllm.sh",
                   help="Path to the batch evaluation shell script")
    p.add_argument("--gpus", default="0,1,2,3,4,5",
                   help="Comma-separated GPU indices passed to the batch script")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Parallel workers (default: number of GPUs)")
    p.add_argument("--chunk-size", type=int, default=500,
                   help="VAC chunk size in ms")
    p.add_argument("--num-candidates", type=int, default=4,
                   help="Beam width / number of top-k candidates to store")
    p.add_argument("--max-files", type=int, default=10000,
                   help="Randomly sample at most this many files (0 = no limit)")
    p.add_argument("--initial-buffer", type=float, default=1.0,
                   help="Initial audio buffer size in seconds before first inference")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--keep-batch-output", action="store_true",
                   help="Keep the temporary batch output directory after synthesis")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    random.seed(args.seed)

    num_workers = args.num_workers or len(args.gpus.split(","))
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    failed_log = os.path.join(args.output_dir, "failed_audio.txt")

    existing_paths = load_existing_audio_paths(output_path)
    print(f"Existing output: {len(existing_paths)} audio paths ({output_path})")

    all_audio = collect_audio_paths(args.audio_dir)
    print(f"Found {len(all_audio)} audio files in {args.audio_dir}")
    done_basenames = {os.path.basename(p) for p in existing_paths}
    audio_paths = [ap for ap in all_audio if os.path.basename(ap) not in done_basenames]
    print(f"After filtering already-processed: {len(audio_paths)} files to process")
    if args.max_files and len(audio_paths) > args.max_files:
        audio_paths = random.sample(audio_paths, args.max_files)
        print(f"Sampled {len(audio_paths)} files (--max-files {args.max_files})")

    if not audio_paths:
        print("No new audio files to process. Exiting.")
        return

    batch_output_dir = os.path.abspath(os.path.join(args.output_dir, "_synth_batch_output"))
    if os.path.exists(batch_output_dir):
        shutil.rmtree(batch_output_dir)
    os.makedirs(batch_output_dir, exist_ok=True)

    print(f"\nRunning batch inference on {len(audio_paths)} files ...")
    run_batch_inference(
        audio_paths=audio_paths,
        reference_file=args.reference_file,
        batch_script=args.batch_script,
        batch_output_dir=batch_output_dir,
        chunk_size=args.chunk_size,
        num_candidates=args.num_candidates,
        gpus=args.gpus,
        num_workers=num_workers,
        initial_buffer=args.initial_buffer,
    )

    print("\nSynthesizing training samples from beam histories ...")
    refs_and_histories = load_per_file_outputs(batch_output_dir)
    path_by_basename = {os.path.basename(p): p for p in audio_paths}

    total = 0
    with tqdm(total=len(refs_and_histories), desc="Aligning", unit="file") as pbar:
        for basename, (ref, history) in refs_and_histories.items():
            real_path = path_by_basename.get(basename, basename)
            if ref is None:
                _log_failed(real_path, failed_log)
                pbar.update(1)
                continue

            samples = synthesize_samples(
                audio_path=real_path,
                ref=ref,
                beam_history=history,
                num_candidates=args.num_candidates,
                chunk_size=args.chunk_size,
                initial_buffer=args.initial_buffer,
            )

            _append_samples(samples, output_path)
            total += len(samples)
            pbar.update(1)
            pbar.set_postfix(samples=total)

    missing = [p for p in audio_paths if os.path.basename(p) not in refs_and_histories]
    for p in missing:
        _log_failed(p, failed_log)
    if missing:
        print(f"\nWARNING: {len(missing)} files had no beam_history.json (see {failed_log})")

    if not args.keep_batch_output:
        shutil.rmtree(batch_output_dir, ignore_errors=True)

    print(f"\n{total} samples written to {output_path}")


if __name__ == "__main__":
    main()
