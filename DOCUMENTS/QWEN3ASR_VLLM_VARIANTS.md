# Qwen3-ASR streaming variants

Four sibling scripts drive the streaming ASR pipeline. They share the same outer protocol (chunked audio accumulation, 2 s windows, `force_language`, error-corrector hand-off) and differ in the backend and in how they produce the `k` candidates fed to the corrector.

## At a glance

| Script | Backend | Decoding | Produces | Per-chunk latency* | Candidate quality |
|---|---|---|---|---|---|
| `qwen3asr_streaming.py` | HuggingFace `transformers` | `model.generate(num_beams=k, num_return_sequences=k, do_sample=False)` | true top-k | ~213 ms | true beam search |
| `qwen3asr_streaming_vllm_sampling.py` | vLLM V1 | Sampling (`n=k`, temp>0) | k diverse samples | ~150 ms | stochastic |
| `qwen3asr_streaming_vllm_beam.py` | vLLM V1 | Native `LLM.beam_search()` | true top-k | **~307-540 ms** | true beam search |
| `qwen3asr_streaming_vllm_beam_async.py` | vLLM V1 | **Block-wise beam search** | true top-k (length-normalized) | **~100-130 ms** | matches native top-1 exactly, k-best equivalent |

\*Real 2 s Mandarin clips, beam width 4, GPU-warm, single model instance.

## Evolution of the decoding stack

### 1. HuggingFace `transformers` - `qwen3asr_streaming.py`

The original implementation. One call to `model.generate(num_beams=k, num_return_sequences=k, do_sample=False)` per audio chunk produces true top-k candidates in ~213 ms. Quality is ideal but the backend does not share KV-cache across concurrent streams and scales poorly.

### 2. vLLM (sampling) - `qwen3asr_streaming_vllm_sampling.py`

Switching to vLLM brings paged KV cache, chunked prefill, and batched scheduling. With `SamplingParams(n=k, temperature>0)` the k candidates come back in ~150 ms - about 30 % faster than HF. The catch: they are **stochastic samples**, not the true top-k, so the error corrector sees noisier hypotheses.

### 3. vLLM (true beam search) - `qwen3asr_streaming_vllm_beam.py`

vLLM exposes `LLM.beam_search()` which is supposed to restore true top-k semantics on the fast engine. In practice it ran at **~307-540 ms/chunk** - slower than HF. Profiling the 16-token output showed where the time was going:

```
16 decoded tokens  ->  16 separate generate(max_tokens=1) calls
each call           ~= 22 ms of engine-IPC overhead (wake engine,
                       push request, run one decode step, pull logits,
                       return to Python driver)
floor               ~= 16 * 22 ms = 360 ms, independent of beam width
```

vLLM V1's beam search is implemented as a Python-level loop over `generate(max_tokens=1)`. Each call crosses the engine-IPC boundary. Prefix caching and larger `max_num_seqs` did not help because **the bottleneck was not compute; it was round-trip count**. That was the signal to stop tuning vLLM knobs and move the beam expansion out of the engine loop entirely.

### 4. Block-wise async beam search - `qwen3asr_streaming_vllm_beam_async.py`

Idea in one line: **ask vLLM for several tokens at a time instead of one, and do the beam branching in Python using the per-position logprobs vLLM already returns**.

Same top-k quality as native, but 3x fewer (or 4x fewer, at `B=8`) IPC trips. Per-chunk latency drops to ~100 ms.

## How async beam search works (intuition + details)

### Why one `generate()` call per token is wasteful

Every `generate(max_tokens=1)` pays a fixed IPC tax (request serialisation, engine wake-up, result marshalling) that is larger than the actual decode. If you instead ask for `B` tokens at a time, you pay the tax once and amortise it over `B` decodes. vLLM happily returns the full log-prob table at every one of those `B` positions, for free.

So the question becomes: **can we still do a correct beam search if we only get to consult the engine every `B` tokens?** The answer is yes, because the logprobs at each intermediate position tell us exactly what alternatives would have been considered.

### The algorithm, walk-through style

Assume beam width `bw=4` and block size `B=4`.

**Step A - batched greedy block.** Take the current beams (initially one, with an empty continuation). Send **one** batched `generate()` request with `max_tokens=B`, `temperature=0`, `logprobs=2*bw`. vLLM returns, for each beam:

- a greedy B-token continuation (same as if we had called it B times in a row), and
- at each of those B positions, the top-`2*bw` alternative tokens with their logprobs.

Because we batch all beams in one call and reuse the cached prompt prefix, this is ~1 IPC trip for the whole block.

**Step B - expand alternatives at every position.** For each beam, walk the B greedy tokens one by one. At each position `p`, the top-`2*bw` logprobs tell us the most promising alternatives. Each alternative spawns a candidate beam:

```
candidate.tokens  = base.tokens + greedy[:p] + [alt_tok]     # truncated + branched
candidate.score   = base.score  + sum(greedy_lp[:p]) + alt_lp
```

This is exactly the set of candidates native beam search would have produced at position `p` - we just compute them in Python from the logprobs instead of making a fresh engine call. A greedy token landing on a stop id (`<|im_end|>`, `<|endoftext|>`, `<|im_start|>`) is moved to `completed` and its beam stops expanding.

**Step C - prune to `bw` beams.** All expansions (up to `bw * B * 2*bw` of them) are sorted by the length-normalised score `cum_logprob / seq_len`. The top `bw` become the beams for the next block. Length normalisation matches vLLM's default `length_penalty=1.0` and is crucial: without it, short alternative branches that just started would always outrank the longer greedy path that has accumulated more (negative) logprobs.

**Step D - iterate.** Repeat steps A-C until `QWEN3_BEAM_MAX_TOKENS` is exhausted or all beams have terminated. Merge `completed + leftover beams`, re-rank with the same length-normalised score, return top-k.

### Where the approximation sits

The only difference from native beam search is **when** an alternative branch gets re-scored by the model:

- Native: at position `p` we take alt_tok, then at position `p+1` we feed the model `[..., alt_tok]` and get fresh logprobs for the next decision.
- Block-async: at position `p` we take alt_tok, but the model does not see the branch until the **next** block boundary. Within the current block, the alternative's score uses the logprobs from the greedy context, not from the branched context.

This means an alternative carries at most `B-1` tokens of greedy-context bias before it is re-contextualised. For short ASR outputs (typically 5-25 tokens at 2 s) this is empirically negligible: at `B=4` the top-4 candidates match native cum-scores exactly (see correctness check below). At `B=8` the top-1 is still identical, the lower candidates reorder slightly.

### Correctness check (real 2 s Mandarin clip, bw=4, B=4)

Native and block-async returned **identical** sequences with **identical** cum-scores:

```
[0] cum=-0.212  '我最近比较忙，没有关注中。'
[1] cum=-2.494  '哦，最近比较忙，没有关注中。'
[2] cum=-2.484  '我最近比较忙，没有关注公。'
[3] cum=-4.521  '哦，最近比较忙，没有关注公。'
```

### Efficiency gains

| Metric | HF | vLLM native beam | Block-async (B=4) | vs HF | vs native |
|---|---|---|---|---|---|
| Per-chunk latency (bw=4, 2 s) | 213 ms | 307 ms | 103 ms | **2.1x** | **3.0x** |
| Engine-IPC round trips / 16 tokens | in-process | 16 | 4 | - | **4x** |
| Batch shape per IPC | - | 1 beam x 1 tok | bw beams x B tok | - | - |

At `QWEN3_BEAM_BLOCK_SIZE=8` the cost drops to ~50 ms/chunk (~4x vs HF, ~6x vs native vLLM) with top-1 still identical but slightly reordered 2nd-4th candidates.

### Tunable env vars

| Var | Default | Effect |
|---|---|---|
| `QWEN3_BEAM_BLOCK_SIZE` | 4 | Tokens decoded per engine call. Larger = fewer IPC trips, more greedy-context bias before a branch is re-scored. `4` matches native exactly; `8` is ~2x faster with minor k-best reordering. |
| `QWEN3_BEAM_MAX_TOKENS` | 16 | Maximum generated tokens per streaming chunk. |
