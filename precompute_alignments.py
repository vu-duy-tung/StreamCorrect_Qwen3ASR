#!/usr/bin/env python3
"""Pre-compute Qwen3-ForcedAligner character-level timestamp caches.

Supports two input modes:
  --reference-file  Raw reference JSON: list of {audio_path, text_zh}.
                    Each unique audio file is aligned against its full transcript.
                    This is the preferred mode before running data_synthesize.py.
  --data            Legacy mode: synthesized JSONL with {audio_path, correct_text}.

Writes a .align.json cache file alongside every audio file.  The cache is read
by data_synthesize.py for time-based chunk boundary alignment.

Cache format (list of per-character dicts):
    [{"char": "你", "pos": 0, "start": 0.00, "end": 0.15}, ...]

"start"/"end" are in seconds.  Punctuation characters that the aligner does not
assign a segment get the end time of the preceding character so that the
monotone linear scan in _align_by_time stays correct.

Usage (reference-file mode, multi-GPU):
    python precompute_alignments.py \\
        --reference-file /data/mino/StreamCorrect/KeSpeech/reference.json \\
        --audio-dir /data/mino/StreamCorrect/KeSpeech/wav \\
        --gpus 0,1,2,3,4,5 \\
        --batch-size 8

Usage (legacy JSONL mode):
    python precompute_alignments.py \\
        --data synthesized_train.jsonl \\
        --gpus 0 \\
        --batch-size 8
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute_alignments")


# ── Audio directory scanner ───────────────────────────────────────────────────

def _build_audio_map(audio_dir: str) -> dict[str, str]:
    """Recursively scan audio_dir and return {basename: fullpath} for every WAV/FLAC."""
    audio_map: dict[str, str] = {}
    for p in Path(audio_dir).rglob("*"):
        if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg") and p.is_file():
            audio_map[p.name] = str(p)
    logger.info("Scanned %s: %d audio files indexed.", audio_dir, len(audio_map))
    return audio_map


def _resolve_path(audio_path: str, audio_dir: str, audio_map: dict[str, str]) -> str:
    """Return the full path for audio_path.

    Resolution order:
      1. Already absolute — return as-is.
      2. Look up basename in audio_map (handles nested subdirectories).
      3. Fall back to os.path.join(audio_dir, audio_path).
    """
    if os.path.isabs(audio_path):
        return audio_path
    name = os.path.basename(audio_path)
    if name in audio_map:
        return audio_map[name]
    if audio_dir:
        return os.path.join(audio_dir, audio_path)
    return audio_path


# ── Cache path convention (must match data_synthesize.py) ────────────────────

def cache_path(audio_path: str) -> str:
    return audio_path + ".align.json"


# ── Aligner output → serialisable cache ──────────────────────────────────────

def _build_cache(segments, gt_text: str) -> list[dict]:
    """Map aligner segments to a per-character list aligned to gt_text positions.

    When the aligner assigns one segment per character the mapping is 1-to-1.
    Otherwise we match by character content and interpolate timestamps for any
    unmatched characters (typically punctuation) by inheriting the preceding
    character's end time so the timeline stays monotone.
    """
    if len(segments) == len(gt_text):
        return [
            {
                "char":  s.text,
                "pos":   i,
                "start": float(s.start_time),
                "end":   float(s.end_time),
            }
            for i, s in enumerate(segments)
        ]

    cache: list[dict] = []
    seg_idx = 0
    for pos, ch in enumerate(gt_text):
        if seg_idx < len(segments) and segments[seg_idx].text == ch:
            cache.append({
                "char":  ch,
                "pos":   pos,
                "start": float(segments[seg_idx].start_time),
                "end":   float(segments[seg_idx].end_time),
            })
            seg_idx += 1
        else:
            prev_end = cache[-1]["end"] if cache else 0.0
            cache.append({"char": ch, "pos": pos, "start": prev_end, "end": prev_end})

    return cache


# ── Per-worker processing ─────────────────────────────────────────────────────

def _worker(gpu_id: int, entries: list[dict], language: str, batch_size: int) -> None:
    """Subprocess: load aligner on gpu_id, process all entries in batches."""
    import torch
    from qwen_asr import Qwen3ForcedAligner

    device = f"cuda:{gpu_id}"
    logger.info("Loading Qwen3-ForcedAligner-0.6B on %s (%d files) …", device, len(entries))
    try:
        aligner = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            dtype=torch.bfloat16,
            device_map=device,
        )
    except Exception as exc:
        logger.error("Failed to load aligner on %s: %s", device, exc)
        return

    done = 0
    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        _process_batch(aligner, batch, language)
        done += len(batch)
        logger.info("[GPU %d] %d / %d", gpu_id, done, len(entries))

    logger.info("[GPU %d] finished.", gpu_id)


def _process_batch(aligner, batch: list[dict], language: str) -> None:
    audios: list = []
    for entry in batch:
        try:
            audio_np, _ = librosa.load(entry["audio_path"], sr=16_000, mono=True)
            audios.append((audio_np, 16_000))
        except Exception as exc:
            logger.warning("Audio load failed %s: %s", entry["audio_path"], exc)
            audios.append(None)

    valid_idx = [i for i, a in enumerate(audios) if a is not None]
    if not valid_idx:
        return

    valid_audios = [audios[i]          for i in valid_idx]
    valid_texts  = [batch[i]["text"]   for i in valid_idx]

    try:
        results = aligner.align(audio=valid_audios, text=valid_texts, language=language)
    except Exception as exc:
        logger.warning("Aligner failed on batch starting at '%s': %s",
                       batch[valid_idx[0]]["audio_path"], exc)
        return

    for result_idx, entry_idx in enumerate(valid_idx):
        entry = batch[entry_idx]
        try:
            segments = results[result_idx]
            cache    = _build_cache(segments, entry["text"])
            out      = cache_path(entry["audio_path"])
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Cache write failed %s: %s", entry["audio_path"], exc)


def run_forced_alignment_jobs(
    todo: list[dict],
    gpus: list[int],
    language: str,
    batch_size: int,
) -> None:
    """Run Qwen3-ForcedAligner on ``todo`` entries.

    Each entry must be ``{\"audio_path\": str, \"text\": str}``.
    Writes ``cache_path(audio_path)`` for each successful alignment.
    """
    if not todo:
        return
    n = len(gpus)
    if n == 0:
        logger.error("run_forced_alignment_jobs: empty GPU list.")
        return
    if n == 1:
        _worker(gpus[0], todo, language, batch_size)
        return

    chunks = [todo[i::n] for i in range(n)]
    logger.info(
        "Splitting %d files across %d GPUs: %s",
        len(todo),
        n,
        [len(c) for c in chunks],
    )

    # Spawn workers re-import this module; ensure repo root is importable when the
    # parent script was launched from another directory (e.g. SpeechLMCorrector/).
    repo_root = str(Path(__file__).resolve().parent)
    _prev_pp = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        repo_root
        if _prev_pp is None
        else f"{repo_root}{os.pathsep}{_prev_pp}"
    )

    try:
        mp.set_start_method("spawn", force=True)
        procs = []
        for gpu_id, chunk in zip(gpus, chunks):
            if not chunk:
                continue
            p = mp.Process(
                target=_worker,
                args=(gpu_id, chunk, language, batch_size),
                name=f"aligner-gpu{gpu_id}",
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                logger.error("Worker %s exited with code %d", p.name, p.exitcode)
    finally:
        if _prev_pp is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = _prev_pp


# ── Entry-list builders ───────────────────────────────────────────────────────

def _load_from_reference(ref_path: str, audio_dir: str = "",
                         audio_map: dict | None = None) -> list[dict]:
    """Read a reference JSON and return deduplicated {audio_path, text} dicts."""
    with open(ref_path, encoding="utf-8") as f:
        refs = json.load(f)

    audio_map = audio_map or {}
    seen: set = set()
    entries: list[dict] = []
    for r in refs:
        ap = r.get("audio_path", "")
        if not ap or ap in seen:
            continue
        seen.add(ap)
        ap = _resolve_path(ap, audio_dir, audio_map)
        text = r.get("text_zh") or r.get("text") or r.get("correct_text") or ""
        if text:
            entries.append({"audio_path": ap, "text": text})
    return entries


def _load_from_jsonl(jsonl_path: str, audio_dir: str = "",
                     audio_map: dict | None = None) -> list[dict]:
    """Read a synthesised JSONL and return deduplicated {audio_path, text} dicts.

    Supports both 'correct_text' and 'continuation_transcript' field names.
    Note: the JSONL stores per-chunk continuations, not the full transcript.
    Prefer --reference-file when the full ground-truth transcripts are available.
    """
    audio_map = audio_map or {}
    seen: set = set()
    entries: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ap = r.get("audio_path", "")
            if not ap or ap in seen:
                continue
            seen.add(ap)
            ap = _resolve_path(ap, audio_dir, audio_map)
            text = r.get("correct_text") or r.get("continuation_transcript") or ""
            if text:
                entries.append({"audio_path": ap, "text": text})
    return entries


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute forced-aligner caches (.align.json) for data synthesis."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--reference-file",
        help="Reference JSON: list of {audio_path, text_zh}. "
             "Each unique audio file is aligned to its full transcript. "
             "Preferred mode before running data_synthesize.py.",
    )
    group.add_argument(
        "--data",
        help="Legacy mode: synthesised JSONL with {audio_path, correct_text}.",
    )
    parser.add_argument(
        "--audio-dir",
        default="",
        help="Directory containing audio files. Prepended to relative audio_path "
             "values from the reference JSON. Not needed if audio_path is absolute.",
    )
    parser.add_argument("--language",   default="Chinese")
    parser.add_argument("--gpus",       default="0",
                        help="Comma-separated GPU IDs, e.g. '0,1,2,3' (default: 0)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Audio files per aligner call per GPU (default: 8)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-compute caches even if .align.json already exists")
    args = parser.parse_args()

    # ── Build audio path map (recursive scan) ─────────────────────────────────
    audio_map: dict[str, str] = {}
    if args.audio_dir:
        audio_map = _build_audio_map(args.audio_dir)

    # ── Load + deduplicate entries ────────────────────────────────────────────
    if args.reference_file:
        all_entries = _load_from_reference(args.reference_file, args.audio_dir, audio_map)
        logger.info("Reference file: %d unique audio files.", len(all_entries))
    else:
        all_entries = _load_from_jsonl(args.data, args.audio_dir, audio_map)
        logger.info("JSONL: %d unique audio files (note: text is per-chunk only).",
                    len(all_entries))

    todo = [
        e for e in all_entries
        if args.overwrite or not os.path.exists(cache_path(e["audio_path"]))
    ]
    already_done = len(all_entries) - len(todo)
    logger.info("%d already cached, %d to process.", already_done, len(todo))

    if not todo:
        logger.info("Nothing to do.")
        return

    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    run_forced_alignment_jobs(todo, gpus, args.language, args.batch_size)

    logger.info("All done.")


if __name__ == "__main__":
    main()
