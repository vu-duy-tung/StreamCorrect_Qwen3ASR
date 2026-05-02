"""Synthesize error-correction training data from Qwen3-ASR beam histories.

Pipeline
--------
1. Collect .wav files from a training folder.
2. Run batch streaming inference (no error corrector) to produce per-file beam histories.
3. For files missing `<wav>.align.json`, run Qwen3-ForcedAligner (same as precompute_alignments.py)
   and write caches next to each WAV (--no-inline-forced-align skips this step).
4. Align each beam candidate against the ground-truth reference and emit
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
import sys
import unicodedata
from pathlib import Path
from typing import Any

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_root_on_path() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_gpu_ids(gpus_csv: str) -> list[int]:
    return [int(g.strip()) for g in gpus_csv.split(",") if g.strip()]


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
    model: str,
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
        "MAX_FILES": str(len(audio_paths)),
        "CHUNK_SIZE": str(chunk_size / 1000.0),
        "BEAMS": str(num_candidates),
        "WORKERS": str(num_workers),
        "GPUS": gpus,
        "MODEL": model,
        "USE_ERROR_CORRECTOR": "false",
        "ERROR_CORRECTOR_CKPT": "",   # disable: batch script checks [ -n "$ERROR_CORRECTOR_CKPT" ]
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
    """Strip ASR special tokens and U+FFFD only; leave all other content intact."""
    if not s:
        return ""
    s = _SPECIAL_TOKEN_RE.sub("", s)
    s = s.replace("\ufffd", "")
    return s
# ---------------------------------------------------------------------------
# Alignment helpers — time-based primary, Smith-Waterman text fallback.
# ---------------------------------------------------------------------------

def _cache_path(audio_path: str) -> str:
    return audio_path + ".align.json"


def _load_align_cache(audio_path: str):
    import json as _json, os as _os
    p = _cache_path(audio_path)
    if not _os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return _json.load(fh)
    except Exception:
        return None


def _align_cache_has_timestamps(cache: list | None) -> bool:
    return bool(cache) and any(float(e.get("end") or 0.0) > 0.0 for e in cache)


def _is_punct_space(ch: str) -> bool:
    import unicodedata as _ud
    return ch.isspace() or _ud.category(ch).startswith("P")


def _align_by_time(end_time_sec: float, align_cache: list) -> int:
    """Linear scan: return gt char index after the last char whose end <= end_time_sec."""
    result = 0
    for entry in align_cache:
        if entry["end"] <= end_time_sec:
            result = entry["pos"] + 1
    return result


def _align_span_by_time(start_time_sec: float, end_time_sec: float, align_cache: list) -> tuple[int, int]:
    """Return [start_pos, end_pos) char span for a chunk time window.

    start_pos is the first character whose end is strictly after start_time_sec.
    end_pos   is the first character after all chars whose end <= end_time_sec.
    """
    if end_time_sec < start_time_sec:
        end_time_sec = start_time_sec
    start_pos = _align_by_time(start_time_sec, align_cache)
    end_pos = _align_by_time(end_time_sec, align_cache)
    if end_pos < start_pos:
        end_pos = start_pos
    return start_pos, end_pos


def _align_by_text_fallback(prev_text: str, gt_text: str) -> int:
    """Fitting alignment: find where prev_text ends in gt_text.

    Smith-Waterman DP with punctuation-aware gap costs:
      +2 exact match, +1 NFKC-normalised match, -1 gap in query,
      0 gap for punct/space in gt_text, -1 gap for other gt chars.
    Strict > update so ties go to the earliest j (no over-extension).
    """
    if not prev_text or not gt_text:
        return 0
    import unicodedata as _ud
    GAP = -1.0
    n, m = len(prev_text), len(gt_text)

    def _sc(a, b):
        if a == b: return 2.0
        a2 = _ud.normalize("NFKC", a).lower()
        b2 = _ud.normalize("NFKC", b).lower()
        return 1.0 if (a2 and b2 and a2 == b2) else -1.0

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            gt_ch = gt_text[j-1]
            match = dp[i-1][j-1] + _sc(prev_text[i-1], gt_ch)
            gap_q = dp[i-1][j]   + GAP
            gap_t = dp[i][j-1]   + (0.0 if _is_punct_space(gt_ch) else GAP)
            dp[i][j] = max(match, gap_q, gap_t)

    best_j, best_score = 0, dp[n][0]
    for j in range(1, m + 1):
        if dp[n][j] > best_score:
            best_score = dp[n][j]
            best_j = j
    return best_j


def _align_prev_end(prev_text: str, norm_ref: str,
                    end_time: float | None = None,
                    align_cache: list | None = None) -> int:
    """Find where prev_text ends in norm_ref.

    Primary : time-based lookup via Qwen3-ForcedAligner cache (.align.json).
    Fallback: Smith-Waterman fitting alignment with punct-aware gap costs.
    """
    if end_time is not None and align_cache:
        if any(e["end"] > 0.0 for e in align_cache):
            return _align_by_time(end_time, align_cache)
    return _align_by_text_fallback(prev_text, norm_ref)


def synthesize_samples(
    audio_path: str,
    ref: str,
    beam_history: list[dict[str, Any]],
    num_candidates: int,
    chunk_size: int,
    initial_buffer: float,
    align_cache: list | None = None,
) -> list[dict[str, Any]]:
    """Convert one file's beam history into error-correction training samples.

    For each history entry:
      1. Normalize prev and top-k candidates.
      2. Compute chunk duration (end_time of current entry minus end_time of previous).
      3. Use forced-aligner timestamps to map the chunk window
         [prev_end_time, end_time] to a character span in reference text.
    """
    if not ref:
        return []
    norm_ref = _normalize_text(ref)
    if not norm_ref:
        return []

    chunk_size_s = chunk_size / 1000.0
    samples: list[dict[str, Any]] = []
    _seen_keys: set = set()
    _align_cache = align_cache if align_cache is not None else _load_align_cache(audio_path)
    has_time_cache = _align_cache_has_timestamps(_align_cache)
    if not has_time_cache:
        # For unfixed_token_num=0 synthesis, continuation targets are defined by
        # chunk time windows. Skip files without forced-align caches.
        return []

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
        start_pos, end_pos = _align_span_by_time(prev_end_time, end_time, _align_cache)
        if start_pos >= len(norm_ref):
            continue
        continuation = norm_ref[start_pos:end_pos]
        if not continuation:
            continue

        _key = (norm_prev, float(entry.get("end_time", 0.0)))
        if _key in _seen_keys:
            continue
        _seen_keys.add(_key)
        samples.append({
            "k_best_candidates": norm_topk,
            "num_candidates": num_candidates,
            "chunk_size": chunk_size,
            "initial_buffer": initial_buffer,
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


def load_existing_audio_paths(jsonl_path: str) -> set[str]:
    """Return distinct audio paths already present in an existing output .jsonl file."""
    audio_paths: set[str] = set()
    if not os.path.exists(jsonl_path):
        return audio_paths
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                s = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            ap = s.get("audio_path")
            if isinstance(ap, str) and ap:
                audio_paths.add(ap)
    return audio_paths


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
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B",
                   help="ASR model id/path passed to the batch script via MODEL env")
    p.add_argument("--gpus", default="0,1,2,3,4,5",
                   help="Comma-separated GPU indices passed to the batch script")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Parallel workers (default: number of GPUs)")
    p.add_argument("--chunk-size", type=int, default=500,
                   help="VAC chunk size in ms")
    p.add_argument("--num-candidates", type=int, default=4,
                   help="Beam width / number of top-k candidates to store")
    p.add_argument("--max-files", type=int, default=10000,
                   help="Target total number of distinct audio files in the output "
                        "(--max-files 0 = no cap). When resuming, only enough new files "
                        "are sampled to reach this total.")
    p.add_argument("--initial-buffer", type=float, default=1.0,
                   help="Initial audio buffer size in seconds before first inference")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--overwrite-output", action="store_true",
                   help="Delete existing output JSONL before running (fresh run). "
                        "Default is to append: skip audios already present and fill up to "
                        "--max-files.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--keep-batch-output", action="store_true",
                   help="Keep the temporary batch output directory after synthesis")
    p.add_argument(
        "--no-inline-forced-align",
        action="store_true",
        help="Do not run Qwen3-ForcedAligner when .align.json is missing (those files are skipped).",
    )
    p.add_argument(
        "--align-batch-size",
        type=int,
        default=8,
        help="Audios per align() batch per GPU for inline forced alignment (default: 8).",
    )
    p.add_argument(
        "--align-language",
        default="Chinese",
        help="Language argument passed to Qwen3-ForcedAligner for inline alignment.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    random.seed(args.seed)

    num_workers = args.num_workers or len(args.gpus.split(","))
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    failed_log = os.path.join(args.output_dir, "failed_audio.txt")

    if args.overwrite_output and os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing output (--overwrite-output): {output_path}")

    existing_paths = load_existing_audio_paths(output_path)
    n_existing = len(existing_paths)
    print(f"Existing output: {n_existing} distinct audio file(s) in {output_path}")

    all_audio = collect_audio_paths(args.audio_dir)
    print(f"Found {len(all_audio)} audio files in {args.audio_dir}")
    done_basenames = {os.path.basename(p) for p in existing_paths}
    pool = [ap for ap in all_audio if os.path.basename(ap) not in done_basenames]
    print(f"Not yet in output JSONL: {len(pool)} file(s) in --audio-dir")

    if args.max_files > 0:
        target_total = args.max_files
        need = max(0, target_total - n_existing)
        take = min(need, len(pool))
        if need == 0:
            print(
                f"Already at or above --max-files ({target_total}) distinct audios "
                f"({n_existing} in file). Nothing to do."
            )
            return
        if take < need:
            print(
                f"WARNING: only {take} file(s) available in pool but need {need} "
                f"to reach --max-files {target_total}."
            )
        audio_paths = random.sample(pool, take) if take < len(pool) else list(pool)
        print(
            f"Sampling {len(audio_paths)} new file(s) "
            f"(target total distinct audios: {target_total}, already have {n_existing})."
        )
    else:
        audio_paths = pool
        print(f"Processing all {len(audio_paths)} file(s) not yet in output (no --max-files cap).")

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
        model=args.model,
    )

    print("\nSynthesizing training samples from beam histories ...")
    refs_and_histories = load_per_file_outputs(batch_output_dir)
    path_by_basename = {os.path.basename(p): p for p in audio_paths}

    if not args.no_inline_forced_align:
        align_todo: list[dict[str, str]] = []
        for basename, (ref, history) in refs_and_histories.items():
            if ref is None:
                continue
            real_path = path_by_basename.get(basename, basename)
            if _align_cache_has_timestamps(_load_align_cache(real_path)):
                continue
            align_todo.append({"audio_path": real_path, "text": ref})
        if align_todo:
            gpu_ids = _parse_gpu_ids(args.gpus)
            if not gpu_ids:
                print(
                    "WARNING: empty --gpus; cannot run inline forced alignment. "
                    "Use --no-inline-forced-align to silence."
                )
            else:
                print(
                    f"\nInferring {len(align_todo)} missing `<wav>.align.json` cache(s) "
                    f"(Qwen3-ForcedAligner on GPU(s) {','.join(map(str, gpu_ids))}) …"
                )
                _ensure_repo_root_on_path()
                from precompute_alignments import run_forced_alignment_jobs

                run_forced_alignment_jobs(
                    align_todo,
                    gpu_ids,
                    args.align_language,
                    args.align_batch_size,
                )

    total = 0
    skipped_no_align = 0
    with tqdm(total=len(refs_and_histories), desc="Aligning", unit="file") as pbar:
        for basename, (ref, history) in refs_and_histories.items():
            real_path = path_by_basename.get(basename, basename)
            if ref is None:
                _log_failed(real_path, failed_log)
                pbar.update(1)
                continue

            loaded_align = _load_align_cache(real_path)
            if not _align_cache_has_timestamps(loaded_align):
                skipped_no_align += 1
                pbar.update(1)
                continue

            samples = synthesize_samples(
                audio_path=real_path,
                ref=ref,
                beam_history=history,
                num_candidates=args.num_candidates,
                chunk_size=args.chunk_size,
                initial_buffer=args.initial_buffer,
                align_cache=loaded_align,
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

    if skipped_no_align:
        hint = (
            "run precompute_alignments.py or remove --no-inline-forced-align."
            if args.no_inline_forced_align
            else "alignment failed for those paths (see precompute_alignments logs above)."
        )
        print(
            f"\nWARNING: {skipped_no_align} file(s) had ASR beam history but were skipped "
            f"because `<audio>.align.json` is missing or has no timestamps ({hint})"
        )

    if not args.keep_batch_output:
        # Remove per-file beam histories and audio symlinks; keep evaluation_results.json
        link_dir = os.path.join(batch_output_dir, "_audio_links")
        if os.path.exists(link_dir):
            shutil.rmtree(link_dir, ignore_errors=True)
        for fname in os.listdir(batch_output_dir):
            if fname.endswith("_beam_history.json"):
                os.remove(os.path.join(batch_output_dir, fname))

    print(f"\n{total} samples written to {output_path}")


if __name__ == "__main__":
    main()
