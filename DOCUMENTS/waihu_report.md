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

---

## Failure Analysis — Where the System Still Falls Short

Evaluated on 684 matched utterances from the Waihu testset (with error corrector active).
Overall CER is **13.3%**, with 470/684 utterances (68.7%) transcribed perfectly.
The remaining 214 errors break down as follows.

### Error categories

| Category | Files | CER contribution |
|---|---|---|
| Substitution / deletion | 203 | 0.1164 (87% of total error) |
| Hallucination / insertion | 7 | 0.0058 |
| Repetition | 2 | 0.0010 |
| Truncation | 2 | 0.0010 |

Almost all damage is substitutions and deletions — the ASR transcribes something
plausible but wrong, rather than generating garbage.

### Short audio is the dominant failure driver

Error rate climbs sharply as utterance length falls:

| Reference length | Error count | Avg CER |
|---|---|---|
| 1–2 chars | 21 | 1.05 |
| 3–4 chars | 41 | 0.61 |
| 5–8 chars | 60 | 0.37 |
| 9–15 chars | 48 | 0.26 |
| 16+ chars | 44 | 0.22 |

62 of the 214 errors are on utterances of 4 characters or fewer.
Single-character utterances (`对`, `有`, `行`, `一`) are almost always wrong —
the model cannot reliably distinguish them without sentence context.

### Specific failure patterns

**1. Leading truncation (initial-buffer timing)**

The initial 1-second buffer gate sometimes discards the very start of an utterance
before the buffer fills, causing the leading content to be lost:

| Reference | Generated | CER |
|---|---|---|
| 一千二百三十四你好 | 你好 | 0.78 |
| 三差五的就是收回二十多万 | 收回二十多万 | 0.50 |
| 口袋里的钱比较预计今年年底 | 你预计今年年 | 0.62 |

**2. English phrases in Chinese speech**

Mixed Chinese/English is handled reasonably (CER 0.14–0.25),
but a purely English phrase is semantically translated into Chinese rather than
transcribed phonetically:

| Reference | Generated | CER |
|---|---|---|
| the research | 做研究 | 1.00 |
| sara电话 | 骚扰电话 | 0.25 |
| ok反正总共... | okokok反正总共... | 0.22 |

**3. Numbers and amounts**

Numeric expressions are frequently truncated or mis-expanded:

| Reference | Generated | CER |
|---|---|---|
| 百分之十八 | 十八 | 0.60 |
| 三百四十九 | 三百四 | 0.40 |
| 一百一十 | 金额 | 1.00 |

**4. Domain-specific / low-frequency vocabulary**

Rare company names and banking terms are rewritten as acoustically similar common phrases:

| Reference | Generated | CER |
|---|---|---|
| 我行综合部 | 然后综合个 | 0.60 |
| 就是金银福善 | 就是经营不善 | 0.50 |
| 用于新的伤害后 | 你心有山海富利 | 1.00 |

**5. Speaker disfluencies in reference**

Some references contain genuine stuttering. The ASR normalises these, creating
apparent errors against the literal reference:

| Reference | Generated | CER |
|---|---|---|
| 我我我明明明 | 我我明天 | 0.50 |
| 你给我等会儿你给我说方案了 | 你给我干不等会等会等我给我什么方案了 | 0.62 |

### Summary

The primary bottleneck is **short utterances (≤ 4 chars)**, where insufficient acoustic
context makes reliable transcription impossible regardless of error corrector quality.
Secondary issues — leading truncation, pure-English phrases, and domain-specific vocabulary —
each affect a small number of files. The error corrector is largely not the bottleneck:
in most failure cases, all ASR top-k candidates are wrong, leaving the corrector
nothing correct to select from.

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
