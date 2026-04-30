# Qwen3-ASR Async Block-Wise Beam Search

**Implementation:** `qwen3asr_streaming_vllm_beam_async.py`

---

## Why this variant exists

The streaming pipeline evolved through four backends (see `QWEN3ASR_VLLM_VARIANTS.md`). The key bottleneck in earlier vLLM implementations was not compute — it was the number of engine IPC round-trips. `vLLM.beam_search()` internally loops `generate(max_tokens=1)` once per token, paying ~22 ms of serialisation overhead each time. At 16 output tokens and beam width 4, that alone floors latency at ~360 ms.

This variant breaks the coupling: it asks vLLM for `BLOCK_SIZE` tokens per call and performs beam expansion in Python using the per-position logprobs vLLM already returns. This cuts round-trips from `max_tokens` down to `max_tokens / BLOCK_SIZE` while preserving true beam-search candidate quality.

| Metric | HF `generate` | vLLM native beam | **Block-async (B=4)** |
|---|---|---|---|
| Per-chunk latency (bw=4, 2 s audio) | 213 ms | ~307–540 ms | **~100–130 ms** |
| Engine IPC trips / 16 tokens | in-process | 16 | **4** |
| Candidate quality | true top-k | true top-k | matches native exactly at B=4 |

---

## Architecture

```
Qwen3ASRBackendASR          (ASRBase subclass)
  ├── _beam_search()         block-wise vLLM beam expansion
  └── infer_chunk()          audio accumulation + beam_search call

Qwen3ASROnline              (OnlineProcessorInterface subclass)
  ├── init()                 reset per-VAD-segment state
  ├── insert_audio_chunk()   append audio to pending/all buffers
  ├── process_iter()         streaming chunk inference
  ├── finish()               final flush + committed-text update
  └── _beam_history[]        cross-segment beam log (data synthesis)
```

Both classes share a `streaming_state` dict that persists within a VAD segment:

```python
{
    'chunk_id':         int,    # increments each infer_chunk call
    'unfixed_chunk_num': 2,     # chunks before prefix locking kicks in
    'unfixed_token_num': 3,     # last k tokens kept "soft" (re-decoded)
    'audio_accum':      ndarray,# cumulative audio since VAD onset
    '_raw_decoded':     str,    # best transcript from previous chunk
}
```

---

## Block-wise beam search (`_beam_search`)

### Prefix construction

Before generating, the algorithm builds a fixed text prefix from the previous chunk's best transcript. This prefix is fed to the model as a hard constraint so the new tokens grow from a stable context:

```
norm_raw_decoded = _normalize_text(state['_raw_decoded'])
cur_ids = tokenizer.encode(norm_raw_decoded)
k = unfixed_token_num   # default 3
end_idx = max(0, len(cur_ids) - k)
prefix = tokenizer.decode(cur_ids[:end_idx])
```

The last `k` tokens are deliberately excluded from the prefix — they represent uncertain tail tokens that benefit from being re-decoded with the new audio context. If decoding `cur_ids[:end_idx]` produces a U+FFFD replacement character (a split multi-byte token boundary), `k` is incremented until a clean decode is found.

The final model prompt is: `<system_prompt + language_token> + prefix`.

### The loop

```
beams = [(cum_logprob=0.0, gen_tokens=[], eos=False)]
completed = []

while steps_left > 0 and beams:
    block = min(BLOCK_SIZE, steps_left)

    # 1. One batched generate() — all active beams, block tokens each.
    outputs = vllm.generate(
        prompts=[prompt_prefix + beam_tokens for each beam],
        SamplingParams(max_tokens=block, temperature=0, logprobs=2*num_beams)
    )

    # 2. For each beam, walk the greedy block token-by-token.
    #    At every position p, emit one alternative beam per entry
    #    in the logprobs dict (alt_token != greedy[p]).
    #      alt_score = cum_before_p + logprob(alt_token at p)
    #      alt_tokens = base_gen + greedy[:p] + [alt_token]
    #    Greedy-path EOS → move to completed. Alt EOS → move to completed.

    # 3. Prune: keep top-num_beams by length-normalised score.
    beams = top_k(new_beams, key=cum_logprob / seq_len)
    steps_left -= block

# Final merge + re-rank by length-normalised score (EOS token excluded from length).
all_final = sorted(completed + beams, key=length_norm_score)[:num_beams]
```

### Why the approximation is negligible

An alternative spawned at position `p` carries `(BLOCK_SIZE - p - 1)` tokens of greedy-context bias before the next model call re-scores it under the branched context. At `BLOCK_SIZE=4` this is at most 3 stale tokens in a ~10–20 token output. Empirically, top-1 and full top-4 candidates match native vLLM beam search exactly.

At `BLOCK_SIZE=8`, top-1 is still identical; 2nd–4th candidates may reorder slightly.

### Return values

```python
return norm_candidates, norm_best_raw, _normalize_text(prefix)
#       list[str]       str             str
```

`norm_candidates` are full transcripts in normalized form: `prefix + new_tokens`.  
`_normalize_text(prefix)` is the `beam_prefix` — the exact boundary the corrector must respect.

---

## Unfixed token window and the corrector boundary

The beam prefix is the single most important alignment invariant. The corrector is trained to receive:

- `previous_text` = the prefix used by beam search = `previous[:-k]` in normalized form
- `candidates` = full transcripts beginning at that same prefix

If `previous_text` were the full `previous` transcript (including the `k` tail tokens that were re-decoded), the corrector would see text that the beam search already "moved past", breaking the training/inference alignment.

`infer_chunk` returns `beam_prefix` as a third value, and both `process_iter` and `finish` use it directly:

```python
# process_iter / finish:
candidates, state, beam_prefix = asr.infer_chunk(...)
corrected_suffix = _run_error_corrector(..., previous_text=beam_prefix, ...)
corrected_top1   = beam_prefix + _normalize_text(corrected_suffix)

_beam_history.append({'previous_transcript': beam_prefix, ...})
```

If a chunk produces no candidates (e.g. very short audio), `last_beam_prefix` from the previous chunk is used as fallback.

---

## Streaming pipeline

### `process_iter`

Called every VAC chunk. Skips inference if:
- No pending audio
- Initial buffer not yet filled (`_speech_audio_samples < initial_buffer * sr`)

On the first call after the initial buffer fills, feeds the full accumulated audio (not just the latest chunk) so the model sees complete context from VAD onset.

Output: `{'first_token_latency': ...}`. Does not return a transcript directly — text emission happens in `finish`.

### `finish`

Called at VAD segment end. Flushes pending audio; if the initial buffer was never filled (short utterance), feeds the full accumulated audio.

Falls back to `last_candidates` / `last_beam_prefix` if this final call produces no output (can happen with very short trailing fragments).

Computes the text delta relative to `committed_text` using a longest-common-subsequence fallback when the new full text does not have `committed_text` as a strict prefix.

Calls `self.init()` to reset per-segment state, but `_beam_history` is preserved across segments (reset by `reset_beam_history()` at file boundary).

### Beam history

`_beam_history` is a list of dicts appended on every inference:

```python
{
    'end_time':             float,   # seconds from VAD onset
    'previous_transcript':  str,     # beam_prefix (normalized)
    'topk':                 list[str],
    'source':               'process_iter' | 'finish',
}
```

Used for offline data synthesis: the history captures what prefix the beam search used and what candidates it produced, forming training pairs for the error corrector.

---

## Text normalization (`_normalize_text`)

Applied to all text that crosses the beam-search / corrector boundary:

1. Strip ASR special tokens (`<|...|>`)
2. Remove U+FFFD replacement characters
3. NFKC unicode normalization
4. Remove all whitespace and punctuation characters
5. Lowercase ASCII letters

This matches the normalization applied at training time in `data_synthesize_qwen3asr.py`.

---

## Error corrector integration (`_run_error_corrector`)

Supports two corrector types:

| Type | Model | Input format |
|---|---|---|
| `lm` | Text-only LMCorrector | `format_instruction_for_correction(candidates, previous)` |
| `speechlm` (default) | Audio+text; Ultravox or Qwen2-Audio | `<\|audio\|>` prefix (Ultravox) or Qwen2-Audio chat template |

Model type is auto-detected from `model.config.model_type`. Both types generate at most 8 new tokens (the corrected suffix). The corrector is skipped if audio is shorter than 1600 samples (~0.1 s).

---

## Tunable parameters

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model ID or local path |
| `--beams` / `-b` | `4` | Beam width |
| `--beam-block-size` | `auto` | Tokens per vLLM call. `auto` uses `MAX_TOKENS` for audio <2 s, `8` for <5 s, `4` otherwise |
| `--initial-buffer` | `1.0` | Seconds of speech to buffer before first inference |
| `--use-error-corrector` | off | Enable the SpeechLM error corrector |
| `--error-corrector-type` | `speechlm` | `speechlm` or `lm` |
| `--error-corrector-ckpt` | — | Path to corrector checkpoint |

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `QWEN3_BEAM_BLOCK_SIZE` | CLI `--beam-block-size` or `auto` | Override block size at runtime. Env takes priority over CLI. |
| `QWEN3_BEAM_MAX_TOKENS` | `16` | Maximum new tokens generated per chunk |

### vLLM engine settings (in `Qwen3ASRBackendASR.__init__`)

| Setting | Value | Reason |
|---|---|---|
| `gpu_memory_utilization` | `0.4` | Leaves headroom for the error corrector model |
| `max_num_seqs` | `max(2*beams, 8)` | Fits all beams in one batched call |
| `max_model_len` | `4096` | |
| `enable_prefix_caching` | `True` | Reuses KV cache for the shared prompt prefix across beams |
