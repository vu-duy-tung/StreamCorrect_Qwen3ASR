# StreamCorrect: Qwen3-ASR Async Beam Search with Error Corrector on Waihu Internal Testset

## Results

### Setup

- **Model:** Qwen3-ASR-1.7B with block-wise async beam search (beam width 4)
- **Error corrector:** SpeechLM corrector fine-tuned on Waihu-style ASR error correction data
- **Dataset:** Waihu internal testset — 684 utterances ~ 23.9 minutes in total, ranges from audio of 0.58s to audio of 21.94s
- **Metric:** Character Error Rate (CER), First Token Latency (FTL), Last Token Latency (LTL)

---

### Accuracy

| System | CER | MER |
|---|---|---|
| Qwen3-ASR (no corrector) | 14.08% | 14.14% |
| Qwen3-ASR + Error Corrector | **13.40%** | **13.46%** |
| **Improvement** | **−0.68% (−4.8% relative)** | **−0.68%** |

The error corrector reduces recognition errors by 0.68% of CER (~4.8% relative reduction).

#### Qualitative examples

The corrector shows three distinct fix patterns in this testset.

**Phonetic confusion — domain terms.** The base model transcribes what it hears phonetically, landing on a wrong but plausible-sounding word. The corrector, conditioned on both the audio and the top-k hypotheses, recovers the intended term:

| Reference | Without EC | With EC |
|---|---|---|
| 十一点 (11 o'clock) | 百分之十一 (11 percent) | 十一点 |
| 持平 (remain flat) | 食品 (food) | 持平 |
| 一百九十九 (199) | 一九九 (1-9-9) | 一百九十 |
| 不知道为什么现在不是生物制剂 | 不知道为什么现在不制生物制剂 | 不知道为什么现在不是生物制剂 |

**Repetition removal.** Short acknowledgement phrases in telephone speech are frequently doubled by the base model due to audio overlap or trailing echo at segment boundaries. The corrector reliably collapses these:

| Reference | Without EC | With EC |
|---|---|---|
| 是的 | 是的是的 | 是的 |
| 好的 | 好的好的 | 好的 |
| 能够还款 | 能够还款能够还款 | 能够还款 |
| 三个月左右 | 三个月三个月左右 | 三个月左右 |

**Greeting prefix cleanup.** The base model occasionally prepends conversational filler ("喂", "你好") that appears in the audio but not in the reference transcript. The corrector removes these:

| Reference | Without EC | With EC |
|---|---|---|
| 你好 | 喂你好 | 你好 |

These examples reflect the data distribution the corrector was trained on — Waihu call-centre conversations where repetition artifacts, greeting noise, and domain-specific financial vocabulary are common error sources.

---

### Latency

**First Token Latency (FTL)** — time from audio start to first emitted word:

| System | Min | Median | Max |
|---|---|---|---|
| Without corrector | 23.8 ms | 70.8 ms | 56,791 ms |
| With corrector | 111.9 ms | 342.9 ms | 44,330 ms |

**Last Token Latency (LTL)** — time from audio end to final word emitted:

| System | Min | Median | Max |
|---|---|---|---|
| Without corrector | 0.0 ms | 41.7 ms | 56,686 ms |
| With corrector | 0.0 ms | 164.5 ms | 9,729 ms |

**Key observations:**

- The error corrector adds roughly **272 ms to median FTL** and **123 ms to median LTL** — the direct cost of running the corrector model on each finalized segment.
- Median FTL and LTL increase with the corrector (71ms→343ms and 42ms→165ms).
---

## How Qwen3-ASR Async Beam Search Works

### The latency problem with standard beam search

A standard beam search on a language model works by running one decoding step at a time, choosing the best candidate tokens, and repeating. In vLLM (the serving engine used here), each of those steps is a separate request to the GPU engine. For a typical 16-token output with beam width 4, that means 16 round-trips — and each round-trip carries a fixed overhead of roughly 22 ms just for request handling, regardless of how fast the model itself runs. This floors the minimum latency at around 350 ms even on warm hardware.

The block-wise async approach breaks this bottleneck by asking vLLM for several tokens at a time rather than one. Instead of 16 separate requests, generating in blocks of 4 tokens means only 4 round-trips — a 4× reduction in overhead.

### Generating in blocks without losing beam quality

The insight that makes this possible is that vLLM already returns the probability of every alternative token at every position it generates, not just the one it chose. So when vLLM generates a block of 4 tokens for a given beam, we also get a ranked list of all the plausible alternatives the model considered at each of those 4 positions.

From this information, the system reconstructs what a full beam search would have produced: at each position within the block, it branches off alternative hypotheses using those stored probabilities, scores them by their accumulated log-probability normalized by length, and keeps only the top-k hypotheses going into the next block. The result is that the top-1 candidate is identical to what native beam search would produce, and the full top-k ranking matches exactly when the block size is 4.

At larger block sizes (e.g. 8 tokens per request), throughput improves further and top-1 is still the same, but the lower-ranked candidates may reorder slightly — an acceptable trade-off for most use cases.

### Handling the boundary between consecutive chunks

In streaming ASR, each audio chunk is not processed in isolation. The model carries forward a running transcript from earlier chunks as a text prefix to condition the next generation on. Simply replaying the entire previous transcript as a prefix creates a subtle problem: the last few tokens of that transcript were themselves generated under uncertainty and may not represent the final committed text.

To handle this, the system designates the last 3 tokens of the previous transcript as "soft" — they are excluded from the fixed prefix and regenerated from scratch given the expanded audio context. This means the beam search always starts from a prefix that is firmly committed, and the re-decoded tail is free to correct itself as more audio arrives. The exact boundary where the prefix ends (called the beam prefix) is then passed forward to the error corrector so it operates on the same portion of text the beam search was built on.

### Initial buffer and VAD integration

The system integrates with voice activity detection (VAD). When a new speech segment begins, the system waits for at least 1 second of speech audio to accumulate before running the first inference. This avoids triggering beam search on fragments too short to produce a meaningful hypothesis, and ensures the first inference sees enough context to commit confidently. Once the buffer is filled, subsequent inferences run on each new incoming chunk while accumulating all audio seen so far in the segment.

The final inference at the end of a VAD segment (the "flush") is given the complete accumulated audio for the segment, ensuring the last words are transcribed with full context even if the final audio chunk was short.

### Error corrector integration

The error corrector receives three things: the audio for the current segment, the beam prefix (the firmly committed portion of the previous transcript), and the top-k candidate transcripts produced by beam search. Its task is to produce a corrected suffix — the portion of the transcript that follows the beam prefix — which is then concatenated with the prefix to form the final output.

This design keeps the corrector input format identical to what it was trained on: the candidates always start from the same prefix, and the corrector never has to "undo" text it was not asked to evaluate. The corrector supports both a text-only mode and an audio-conditioned mode (SpeechLM), where the audio signal provides additional evidence for resolving phonetically ambiguous errors.

A beam history log is maintained across the full file, recording the beam prefix, top-k candidates, and timestamp for every inference. This log is used to generate new training data for the corrector from production transcripts without requiring additional annotation.
