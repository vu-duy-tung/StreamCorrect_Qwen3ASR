"""
Streaming ASR using Qwen3-ASR-1.7B as the backbone,
with optional error correction using the existing Qwen2-Audio/Ultravox corrector.

Replaces Whisper with Qwen3-ASR for producing top-k beam search candidates,
while keeping the error corrector unchanged.

Usage:
    python qwen3asr_streaming_vllm_beam_async.py audio.wav --language zh --beams 4 --error-corrector-ckpt /path/to/ckpt
"""

import os
import re
import sys
import time
import logging
import argparse
import unicodedata

import numpy as np
import torch

from streaming.base import OnlineProcessorInterface, ASRBase

logger = logging.getLogger(__name__)

LANG_CODE_TO_NAME = {
    'yue': 'Cantonese', 'zh': 'Chinese', 'en': 'English',
    'ja': 'Japanese', 'ko': 'Korean', 'de': 'German',
    'fr': 'French', 'es': 'Spanish', 'pt': 'Portuguese',
    'ru': 'Russian', 'ar': 'Arabic', 'hi': 'Hindi',
    'id': 'Indonesian', 'it': 'Italian', 'th': 'Thai',
    'vi': 'Vietnamese', 'tr': 'Turkish', 'ms': 'Malay',
    'auto': None,
}


# ---------------------------------------------------------------------------
# Minimal ASR-artifact stripper — removes model special tokens and the Unicode
# replacement character (U+FFFD) that can appear when tokenization fails.
# No whitespace removal, no punctuation stripping, no case folding.
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")
_SEMANTIC_PUNCTUATION = {"%", "％"}
_QWEN3ASR_CORRECTOR_SUFFIX_SYSTEM_PROMPT = (
    "You are an ASR error corrector. "
    "Listen to the audio and, given the previous confirmed transcript plus "
    "the k-best full hypotheses below, output only the corrected suffix that "
    "should be appended after the previous transcript. Do not repeat the "
    "previous transcript."
)
_QWEN3ASR_CORRECTOR_SUFFIX_BEAM_SWITCH_SYSTEM_PROMPT = (
    "You are an ASR error corrector. "
    "Listen to the audio and compare the k-best full hypotheses below. "
    "Use candidate 1 only when the audio supports it; otherwise use the "
    "candidate whose suffix best matches the audio, with minimal correction. "
    "Output only the corrected suffix that should be appended after the "
    "previous confirmed transcript. Do not repeat the previous transcript."
)
_QWEN3ASR_CORRECTOR_FULL_TRANSCRIPT_SYSTEM_PROMPT = (
    "You are an ASR error corrector. "
    "Listen to the audio and, given the k-best full hypotheses below, output "
    "only the corrected full transcript. Do not add explanations."
)
_QWEN3ASR_CORRECTOR_SYSTEM_PROMPTS = {
    "suffix": _QWEN3ASR_CORRECTOR_SUFFIX_SYSTEM_PROMPT,
    "suffix_beam_switch": _QWEN3ASR_CORRECTOR_SUFFIX_BEAM_SWITCH_SYSTEM_PROMPT,
    "full_transcript": _QWEN3ASR_CORRECTOR_FULL_TRANSCRIPT_SYSTEM_PROMPT,
}


def _normalize_text(s):
    """Strip ASR special tokens and U+FFFD only; leave all other content intact."""
    if not s:
        return ""
    s = _SPECIAL_TOKEN_RE.sub("", s)
    s = s.replace("\ufffd", "")
    return s


def _strip_qwen3asr_transcript_markup(text: str) -> str:
    """Remove Qwen3-ASR transcript wrappers from corrector generations.

    Dense Qwen3-ASR correctors sometimes answer with the ASR serialization
    prefix, e.g. ``language Chinese<asr_text>你好``.  For EC output that prefix is
    transport noise, while a later occurrence indicates a hallucinated format
    restart and should be discarded with the tail.
    """
    s = _normalize_text(text or "")
    if not s:
        return ""

    start = re.match(r"\s*language\s+\S+\s*<asr_text>\s*", s, flags=re.IGNORECASE)
    if start is not None:
        s = s[start.end():]

    s = re.sub(r"[ \t\n]*language\s+\S+\s*<asr_text>.*", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"\n`{3}.*", "", s, flags=re.DOTALL)
    return s.strip()


def _strip_intermediate_tail_artifacts(text: str) -> str:
    """Strip unstable trailing artifacts for intermediate chunk hypotheses.

    Applied only during process_iter() (not finish()):
    - trailing special tokens like <|...|>
    - trailing whitespace
    - trailing punctuation/control/separator chars
    """
    if not text:
        return ""

    s = text
    while True:
        changed = False

        # Remove trailing model special tokens.
        m = re.search(r"<\|[^|]*\|>\s*$", s)
        if m is not None:
            s = s[:m.start()]
            changed = True

        # Remove trailing whitespace/punctuation/control/separator chars.
        while s:
            ch = s[-1]
            cat = unicodedata.category(ch)
            if ch == "\ufffd" or ch.isspace() or (
                cat.startswith(("P", "C", "Z")) and ch not in _SEMANTIC_PUNCTUATION
            ):
                s = s[:-1]
                changed = True
            else:
                break

        if not changed:
            break

    return s


def _is_short_number_truncation(prev_text: str, final_text: str) -> bool:
    prev_c = _compact_for_acceptance(prev_text)
    final_c = _compact_for_acceptance(final_text)
    if not prev_c or not final_c:
        return False
    if len(prev_c) > 8 or len(prev_c) <= len(final_c):
        return False
    it = iter(prev_c)
    if not all(ch in it for ch in final_c):
        return False
    return _is_numberish(prev_text) and _is_numberish(final_text)


def _strip_matching_prefix(text: str, prefix: str) -> str:
    """Return the suffix of text after prefix, tolerating punctuation-only drift."""
    text = _normalize_text(text or "")
    prefix = _normalize_text(prefix or "")
    if not text or not prefix:
        return text

    if text.startswith(prefix):
        return text[len(prefix):]

    # The training/eval normalizer ignores punctuation, but runtime strings may
    # differ by commas/periods around the prefix boundary.  Keep only a small,
    # conservative fallback so the corrector cannot duplicate the committed
    # prefix when it emits a full transcript.
    def _compact(s: str) -> str:
        return "".join(
            ch
            for ch in s
            if not (
                unicodedata.category(ch).startswith(("P", "C", "Z"))
                and ch not in _SEMANTIC_PUNCTUATION
            )
        )

    compact_prefix = _compact(prefix)
    compact_text = _compact(text)
    if compact_prefix and compact_text.startswith(compact_prefix):
        consumed = 0
        kept = 0
        for idx, ch in enumerate(text):
            if not (
                unicodedata.category(ch).startswith(("P", "C", "Z"))
                and ch not in _SEMANTIC_PUNCTUATION
            ):
                kept += 1
            if kept >= len(compact_prefix):
                consumed = idx + 1
                break
        return text[consumed:]

    return text


def _normalize_corrector_suffix(response: str, previous_text: str, candidates: list[str]) -> str:
    """Coerce a corrector generation into the suffix expected by the streamer."""
    suffix = _strip_intermediate_tail_artifacts(_strip_qwen3asr_transcript_markup(response or "")).strip()
    prev = _strip_intermediate_tail_artifacts(_normalize_text(previous_text or "")).strip()
    if not suffix:
        return ""

    suffix = _strip_matching_prefix(suffix, prev).strip()
    if not suffix:
        return ""

    def _compact_with_ends(s: str) -> tuple[str, list[int]]:
        compact_chars = []
        ends = []
        for idx, ch in enumerate(s):
            if (
                unicodedata.category(ch).startswith(("P", "C", "Z"))
                and ch not in _SEMANTIC_PUNCTUATION
            ):
                continue
            compact_chars.append(ch)
            ends.append(idx + 1)
        return "".join(compact_chars), ends

    compact_prev, _ = _compact_with_ends(prev)
    compact_suffix, suffix_ends = _compact_with_ends(suffix)

    # A full-transcript generation that is already covered by the committed
    # prefix is not a suffix. Drop it instead of duplicating the prefix.
    if compact_prev and compact_suffix and len(compact_suffix) >= 2:
        if compact_prev.startswith(compact_suffix) or compact_suffix in compact_prev:
            return ""

    # Some dense Qwen3-ASR corrector checkpoints occasionally answer with a
    # near-full transcript instead of the requested continuation, e.g. repeating
    # the prefix with a minor filler/character drift.  The streamer cannot revise
    # already committed prefix text, so keep only any tail after the part that
    # aligns to the end of the prefix; if the answer is almost entirely covered by
    # the prefix, treat it as an empty suffix.
    if compact_prev and compact_suffix and len(compact_suffix) >= 4:
        import difflib
        matcher = difflib.SequenceMatcher(None, compact_prev, compact_suffix)
        blocks = matcher.get_matching_blocks()
        covered = sum(block.size for block in blocks)
        coverage = covered / max(1, len(compact_suffix))
        end_aligned = [
            block for block in blocks
            if block.size > 0 and block.a + block.size == len(compact_prev)
        ]
        if end_aligned:
            block = max(end_aligned, key=lambda b: (b.size, b.b))
            tail_compact_idx = block.b + block.size
            if tail_compact_idx >= len(compact_suffix):
                return ""
            if coverage >= 0.65:
                suffix = suffix[suffix_ends[tail_compact_idx - 1]:].strip()
                if not suffix:
                    return ""
        elif coverage >= 0.8:
            return ""

    # If the model emits a shortened or overlapping full transcript instead of
    # a pure suffix, append only the part not already covered by the prefix.
    max_overlap = min(len(compact_prev), len(compact_suffix))
    for n in range(max_overlap, 0, -1):
        if compact_prev.endswith(compact_suffix[:n]):
            suffix = suffix[suffix_ends[n - 1]:].strip()
            break
    if not suffix:
        return ""

    for cand in candidates or []:
        cand_suffix = _strip_matching_prefix(
            _strip_intermediate_tail_artifacts(_normalize_text(cand or "")).strip(),
            prev,
        ).strip()
        if cand_suffix and suffix.startswith(cand_suffix + cand_suffix):
            suffix = cand_suffix
            break

    return suffix


def _normalize_corrector_full_transcript(response: str) -> str:
    return _strip_qwen3asr_transcript_markup(response or "")


def _compact_for_acceptance(text: str) -> str:
    chars = []
    for ch in _normalize_text(text or ""):
        cat = unicodedata.category(ch)
        if ch.isspace() or (
            cat.startswith(("P", "C", "Z")) and ch not in _SEMANTIC_PUNCTUATION
        ):
            continue
        chars.append(unicodedata.normalize("NFKC", ch).lower())
    return "".join(chars)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nd = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            nd[j] = min(
                dp[j] + 1,
                nd[j - 1] + 1,
                dp[j - 1] + (0 if ca == cb else 1),
            )
        dp = nd
    return dp[-1]


def _is_numberish(text: str) -> bool:
    s = _compact_for_acceptance(text)
    if not s:
        return False
    numeric_chars = set("0123456789零一二三四五六七八九十百千万亿两幺〇点百分之元块号%％.")
    numeric = sum(ch in numeric_chars for ch in s)
    return numeric >= 2 and numeric / max(1, len(s)) >= 0.5


def _suffixes_after_prefix(candidates: list[str], previous_text: str) -> list[str]:
    prev = _strip_intermediate_tail_artifacts(_normalize_text(previous_text or "")).strip()
    suffixes = []
    for cand in candidates or []:
        cand_norm = _strip_intermediate_tail_artifacts(_normalize_text(cand or "")).strip()
        if not cand_norm:
            continue
        suffixes.append(_strip_matching_prefix(cand_norm, prev).strip())
    return suffixes


def _accept_corrector_suffix(
    corrected_suffix: str,
    previous_text: str,
    candidates: list[str],
    mode: str,
) -> tuple[bool, str]:
    """Runtime safety gate for suffix EC.

    This is deliberately conservative and off by default.  It does not choose
    transcripts manually; it only rejects high-risk EC generations and lets the
    normal ASR top-1 pass through.  The rule protects the strict streaming
    interface where an over-short suffix cannot be repaired later.
    """
    if mode in {"", "off", None}:
        return True, "off"
    if mode not in {"conservative", "conservative_v2"}:
        return True, f"unknown_mode:{mode}"

    suffixes = _suffixes_after_prefix(candidates, previous_text)
    top1_suffix = suffixes[0] if suffixes else ""
    corr = _strip_intermediate_tail_artifacts(_normalize_text(corrected_suffix or "")).strip()
    top1_c = _compact_for_acceptance(top1_suffix)
    corr_c = _compact_for_acceptance(corr)

    if corr_c == top1_c:
        return True, "same_as_top1"
    if not corr_c:
        return (not top1_c), "empty_suffix"
    if mode == "conservative_v2" and not top1_c:
        # If ASR top-1 has no remaining suffix after the committed prefix,
        # a non-empty EC suffix is usually a hallucinated tail append. This was
        # the dominant residual regression pattern for 0.6B suffix EC.
        return False, "empty_top1_suffix_nonempty_ec"

    top1_len = len(top1_c)
    corr_len = len(corr_c)

    # One-shot short utterances are high-risk: switching from a correct short
    # ASR top-1 to another short beam ("一" -> "嗯", "我" -> "嗯") creates large
    # CER spikes. Let training solve these later; conservative runtime should
    # preserve top-1 unless the generation is identical.
    if mode == "conservative_v2" and not _compact_for_acceptance(previous_text) and top1_len <= 2:
        return False, "short_initial_switch"

    # Number-like suffixes are high-cost mistakes; only allow punctuation-only
    # changes in conservative mode.
    if _is_numberish(top1_suffix) or _is_numberish(corr):
        return False, "numberish_changed"

    # Most observed 0.6B regressions are deletions/shortening of already-good
    # top-1 continuations, especially short repeated utterances.
    if corr_len < top1_len:
        if top1_len <= 8:
            return False, "short_suffix_shrink"
        deletion = top1_len - corr_len
        if deletion >= max(2, int(round(top1_len * 0.2))):
            return False, "large_suffix_shrink"

    # Avoid over-generating from tiny final candidates.
    if top1_len <= 3 and corr_len > top1_len + 2:
        return False, "tiny_top1_overgenerate"

    # Keep accepted generations close to at least one ASR beam suffix.  This
    # still permits small generative fixes such as homophones, but rejects freer
    # rewrites that current 0.6B suffix models often get wrong.
    cand_compacts = [_compact_for_acceptance(s) for s in suffixes if _compact_for_acceptance(s)]
    if cand_compacts:
        best_dist = min(_edit_distance(corr_c, c) for c in cand_compacts)
        best_ratio = best_dist / max(1, len(corr_c), min(len(c) for c in cand_compacts))
        if best_ratio > 0.35:
            return False, f"far_from_candidates:{best_ratio:.2f}"

    return True, "accepted"


def _recent_audio_window(audio_np, seconds: float = 0.5, min_samples: int = 1600):
    audio = np.asarray(audio_np, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0) if audio.shape[0] <= 2 else audio[0]
    if audio.shape[0] <= min_samples:
        return audio
    n = max(min_samples, int(round(seconds * 16000)))
    return audio[-min(n, audio.shape[0]):]


def _build_qwen3asr_candidates_text(candidates):
    body = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates)) + "\n"
    return f"<candidates>\n{body}</candidates>"


def _build_qwen3asr_corrector_prompt(candidates, previous_text, output_mode="suffix"):
    if output_mode not in _QWEN3ASR_CORRECTOR_SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported Qwen3-ASR corrector output_mode: {output_mode}")
    user_body = "<|audio_start|><|audio_pad|><|audio_end|>\n"
    if output_mode.startswith("suffix") and previous_text:
        user_body += f"Previous: {previous_text}\n"
    user_body += _build_qwen3asr_candidates_text(candidates)
    return (
        f"<|im_start|>system\n{_QWEN3ASR_CORRECTOR_SYSTEM_PROMPTS[output_mode]}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def qwen3asr_args(parser):
    group = parser.add_argument_group('Qwen3-ASR')
    group.add_argument(
        '--model', type=str, default='Qwen/Qwen3-ASR-1.7B',
        help='Qwen3-ASR model path or HuggingFace repo id.',
    )
    group.add_argument(
        '--beams', '-b', type=int, default=4,
        help='Beam width for beam search (default: 4).',
    )
    group.add_argument(
        '--beam-block-size', type=str, default='auto',
        help='Tokens decoded per vLLM call. "auto" adapts to audio length; '
             'larger = faster but coarser beam approximation. (default: auto)',
    )
    group.add_argument(
        '--initial-buffer', type=float, default=1.0,
        help='Seconds of speech to buffer after VAD onset before first inference. (default: 1.0)',
    )

    ec = parser.add_argument_group('Error corrector')
    ec.add_argument(
        '--error-corrector-ckpt', type=str, default=None,
        help='Path to error corrector checkpoint. Enables the corrector when set.',
    )
    ec.add_argument(
        '--error-corrector-base-model', type=str, default=None,
        help='Base model for the error corrector (required for LoRA checkpoints).',
    )
    ec.add_argument(
        '--error-corrector-type', type=str, choices=['speechlm', 'lm', 'qwen3asr'], default='speechlm',
        help='Corrector type: "speechlm" (audio+text), "lm" (text-only), or "qwen3asr" (Qwen3-ASR-0.6B LoRA).',
    )
    ec.add_argument(
        '--error-corrector-audio-window',
        type=str,
        choices=['runtime', 'cumulative'],
        default='runtime',
        help='Audio window passed to the error corrector. "runtime" preserves the current chunk/final window; '
             '"cumulative" passes all speech audio in the VAD segment, matching SpeechLM ver3 training.',
    )
    ec.add_argument(
        '--error-corrector-max-new-tokens',
        type=int,
        default=16,
        help='Maximum new tokens for Qwen3-ASR error-corrector suffix generation. (default: 16)',
    )
    ec.add_argument(
        '--error-corrector-output-mode',
        type=str,
        choices=['suffix', 'suffix_beam_switch', 'full_transcript'],
        default='suffix',
        help='Qwen3-ASR corrector generation target. full_transcript is applied only at finish().',
    )
    ec.add_argument(
        '--error-corrector-scope',
        type=str,
        choices=['all', 'finish'],
        default='all',
        help='When to run the error corrector. "all" runs on process_iter and finish; '
             '"finish" leaves streaming ASR state untouched and corrects only final flushes.',
    )
    ec.add_argument(
        '--error-corrector-acceptance-mode',
        type=str,
        choices=['off', 'conservative', 'conservative_v2'],
        default='off',
        help='Optional suffix EC safety gate. "off" preserves reproduction behavior; '
             '"conservative" rejects high-risk suffix rewrites and falls back to ASR top-1. '
             '"conservative_v2" additionally rejects empty-top1 tail appends and short initial switches.',
    )



def qwen3_asr_factory(args):
    logger.setLevel(args.log_level)
    asr = Qwen3ASRBackendASR(
        language=args.language,
        model_path=args.model,
        beams=args.beams,
        logdir=getattr(args, 'output_dir', None),
        initial_buffer=args.initial_buffer,
        beam_block_size=args.beam_block_size,
        seed=getattr(args, 'seed', 42),
        error_corrector_audio_window=getattr(args, 'error_corrector_audio_window', 'runtime'),
        error_corrector_max_new_tokens=getattr(args, 'error_corrector_max_new_tokens', 16),
        error_corrector_output_mode=getattr(args, 'error_corrector_output_mode', 'suffix'),
        error_corrector_scope=getattr(args, 'error_corrector_scope', 'all'),
        error_corrector_acceptance_mode=getattr(args, 'error_corrector_acceptance_mode', 'off'),
    )
    return asr, Qwen3ASROnline(asr)


class Qwen3ASRBackendASR(ASRBase):

    sep = ''

    def __init__(
        self,
        language,
        model_path,
        beams,
        logdir,
        initial_buffer=1.0,
        beam_block_size="auto",
        seed=42,
        error_corrector_audio_window="runtime",
        error_corrector_max_new_tokens=16,
        error_corrector_output_mode="suffix",
        error_corrector_scope="all",
        error_corrector_acceptance_mode="off",
    ):
        self.language = language
        self.beams = beams
        self.logdir = logdir
        self.initial_buffer = initial_buffer
        self.beam_block_size = beam_block_size
        self.seed = seed
        self.error_corrector_audio_window = error_corrector_audio_window
        self.error_corrector_max_new_tokens = error_corrector_max_new_tokens
        self.error_corrector_output_mode = error_corrector_output_mode
        self.error_corrector_scope = error_corrector_scope
        self.error_corrector_acceptance_mode = error_corrector_acceptance_mode
        self.force_language = LANG_CODE_TO_NAME.get(language)

        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
        from qwen_asr.core.vllm_backend.qwen3_asr import Qwen3ASRForConditionalGeneration
        from vllm import ModelRegistry
        
        try:
            ModelRegistry.register_model(
                "Qwen3ASRForConditionalGeneration",
                Qwen3ASRForConditionalGeneration
            )
        except Exception:
            pass # already registered

        logger.info(f'Loading Qwen3-ASR (vLLM backend) model from {model_path}')
        # Beam search uses max_num_seqs parallel sequences per step.
        # Use 2*beams because beam_search expands each beam into 2*beam_width candidates.
        self.qwen3 = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=0.4,
            max_num_seqs=max(2 * beams, 8),
            max_model_len=4096,
            enable_prefix_caching=True,
            disable_log_stats=True,
            seed=seed,
        )
        logger.info(f'Language: {language} -> {self.force_language}')

    def init_state(self):
        """Initializes a custom streaming state for pseudo-streaming via HuggingFace."""
        import numpy as np
        unfixed_chunk_num = int(os.environ.get('QWEN3_UNFIXED_CHUNK_NUM', '0'))
        unfixed_token_num = int(os.environ.get('QWEN3_UNFIXED_TOKEN_NUM', '0'))
        return {
            'chunk_id': 0,
            'unfixed_chunk_num': unfixed_chunk_num,
            'unfixed_token_num': unfixed_token_num,
            'audio_accum': np.zeros(0, dtype=np.float32),
            '_raw_decoded': ''
        }

    def infer_chunk(self, audio_chunk, state, is_last=False):
        """Feed audio into state and run beam search.

        The caller (Qwen3ASROnline) handles initial-buffer gating: this method
        is only called once enough speech audio has accumulated, or on the final
        flush.  Each call appends the new audio to the cumulative audio_accum and
        runs beam search on the full accumulated audio so far.
        """
        candidates = []
        best_raw = state['_raw_decoded']
        beam_prefix = ''

        if audio_chunk is not None and audio_chunk.shape[0] > 0:
            if state['audio_accum'].shape[0] == 0:
                state['audio_accum'] = audio_chunk
            else:
                state['audio_accum'] = np.concatenate([state['audio_accum'], audio_chunk], axis=0)

        if state['audio_accum'].shape[0] == 0:
            return candidates, state, beam_prefix

        try:
            candidates, best_raw, beam_prefix = self._beam_search(state['audio_accum'], state, self.beams)
            if candidates and len(candidates) > 0:
                next_raw = _normalize_text(best_raw)
                # For intermediate streaming chunks, keep the carry-over prefix
                # conservative by removing unstable tail artifacts.
                if not is_last:
                    next_raw = _strip_intermediate_tail_artifacts(next_raw)
                state['_raw_decoded'] = next_raw
                state['chunk_id'] += 1
        except Exception as e:
            import traceback
            logger.warning(f'Beam search failed ({e})\n{traceback.format_exc()}\nfalling back to greedy')
            results = self.qwen3.transcribe(
                audio=(state['audio_accum'], 16000),
                language=self.force_language,
            )
            text = results[0].text.strip()
            candidates = [_normalize_text(text)]
            state['_raw_decoded'] = text
            state['chunk_id'] += 1

        return candidates, state, beam_prefix

    def _beam_search(self, audio_np, state, num_beams):
        """Block-wise TRUE beam search - batches K token steps per vLLM round to
        amortize the ~22ms-per-step engine IPC overhead.

        Strategy:
          Instead of vLLM's native beam_search which does 16 sequential
          generate(max_tokens=1) calls (363ms at bw=4), we issue a batched
          generate(max_tokens=BLOCK_SIZE, logprobs=2*num_beams) once per block.
          At each block boundary we perform standard beam expansion using
          the per-position logprobs returned by vLLM, keep top-`num_beams`
          beams, and continue. This preserves true beam search semantics
          while cutting the number of round-trips from max_tokens -> max_tokens/BLOCK_SIZE.

        Controlled by env QWEN3_BEAM_BLOCK_SIZE (default 4).
        """
        import os as _os
        from vllm import SamplingParams
        from qwen_asr.inference.utils import parse_asr_output

        processor = self.qwen3.processor
        tokenizer = processor.tokenizer

        MAX_TOKENS = int(_os.environ.get('QWEN3_BEAM_MAX_TOKENS', '16'))

        # Resolve block size: CLI arg or env override, with auto-sizing.
        block_arg = self.beam_block_size
        env_block = _os.environ.get('QWEN3_BEAM_BLOCK_SIZE')
        if env_block is not None:
            block_arg = env_block  # env takes priority

        if str(block_arg) == 'auto':
            audio_dur = audio_np.shape[0] / 16000
            if audio_dur < 2.0:
                BLOCK_SIZE = MAX_TOKENS  # single-shot for short audio
            elif audio_dur < 5.0:
                BLOCK_SIZE = 8
            else:
                BLOCK_SIZE = 4
            logger.info(f'[Qwen3-ASR vLLM block-beam] auto block_size={BLOCK_SIZE} (audio={audio_dur:.2f}s)')
        else:
            BLOCK_SIZE = int(block_arg)

        raw_decoded = _normalize_text(state['_raw_decoded'])
        prefix = ""
        if state['chunk_id'] >= state['unfixed_chunk_num']:
            cur_ids = tokenizer.encode(raw_decoded, add_special_tokens=False)
            k = int(state['unfixed_token_num'])
            while True:
                end_idx = max(0, len(cur_ids) - k)
                prefix = tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                if '\ufffd' not in prefix:
                    break
                if end_idx == 0:
                    prefix = ""
                    break
                k += 1

        prompt_prefix = self.qwen3._build_text_prompt(
            context='', force_language=self.force_language,
        ) + prefix

        # Pre-tokenize the shared prompt prefix ONCE.
        prompt_token_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)

        eos_id = tokenizer.eos_token_id
        # Treat <|im_end|>, <|endoftext|>, <|im_start|> as stop tokens to avoid continuation hallucinations.
        # <think> and </think> are Qwen3 thinking-mode tokens; stop immediately if the model
        # spontaneously emits them so they never appear in ASR candidates.
        stop_ids = set()
        if eos_id is not None: stop_ids.add(eos_id)
        for _tkn in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<think>", "</think>"):
            _id = tokenizer.convert_tokens_to_ids(_tkn)
            if _id is not None and _id >= 0: stop_ids.add(_id)

        # Beam state: each beam = (cum_logprob, token_ids_after_prompt, eos_flag)
        # token_ids_after_prompt contains only the NEW tokens generated beyond the prompt.
        # We keep a single initial beam seeded with empty continuation.
        beams = [(0.0, [], False)]
        completed = []

        steps_left = MAX_TOKENS
        while steps_left > 0 and beams:
            block = min(BLOCK_SIZE, steps_left)

            # Build batched prompt list: one TokensPrompt per active beam.
            batched_prompts = []
            for _cum, gen_toks, _done in beams:
                tokens_prompt = {
                    "prompt_token_ids": list(prompt_token_ids) + list(gen_toks),
                    "multi_modal_data": {"audio": [audio_np]},
                }
                batched_prompts.append(tokens_prompt)

            # One batched generate() call: all beams x `block` tokens.
            # logprobs=2*num_beams gives top-2*bw alternatives at EVERY position -
            # sufficient for true beam expansion at the end of this block.
            sp = SamplingParams(
                n=1,
                temperature=0.0,
                max_tokens=block,
                logprobs=2 * num_beams,
                skip_special_tokens=False,
                stop_token_ids=list(stop_ids) if stop_ids else None,
            )
            try:
                outputs = self.qwen3.model.generate(
                    prompts=batched_prompts,
                    sampling_params=sp,
                    use_tqdm=False,
                )
            except Exception:
                raise

            # Expand each beam into alternatives at EVERY position within its
            # block (not just the last position). Starting from cumulative
            # logprob `base_cum`, we walk along the greedy path token-by-token
            # while simultaneously emitting alternatives at each step.
            #
            # At position P, an alternative token T != greedy[P] produces a
            # candidate beam = base_gen + greedy[:P] + [T] with score
            #   base_cum + sum(greedy_logprobs[:P]) + logprob(T at pos P)
            # These alternative beams are re-scored in the next block by running
            # the model on their new prefix. This preserves true beam-search
            # semantics with `block_size` granularity of re-scoring.
            new_beams = []
            for beam_idx, out in enumerate(outputs):
                base_cum, base_gen, _ = beams[beam_idx]
                seq = out.outputs[0]
                produced_ids = list(seq.token_ids)
                if not produced_ids:
                    new_beams.append((base_cum, base_gen, True))
                    continue

                per_pos_logprobs = seq.logprobs  # list[dict[int, Logprob]] or None

                # Walk the greedy path token-by-token, accumulating logprob.
                # At each position emit alternatives from that position's logprob dict.
                cum_at_pos = base_cum  # cum logprob BEFORE taking greedy[pos]
                for pos, tid in enumerate(produced_ids):
                    d = per_pos_logprobs[pos] if per_pos_logprobs is not None and pos < len(per_pos_logprobs) else {}
                    greedy_lp_obj = d.get(tid) if d else None
                    greedy_lp = greedy_lp_obj.logprob if greedy_lp_obj is not None else 0.0

                    # Emit alternatives at this position (competing with the greedy token).
                    for alt_tid, alt_lp_obj in d.items():
                        if alt_tid == tid:
                            continue
                        alt_cum = cum_at_pos + alt_lp_obj.logprob
                        alt_gen = base_gen + produced_ids[:pos] + [alt_tid]
                        alt_eos = (alt_tid in stop_ids)
                        if alt_eos:
                            completed.append((alt_cum, alt_gen, True))
                        else:
                            new_beams.append((alt_cum, alt_gen, False))

                    cum_at_pos += greedy_lp

                    # If greedy path hits EOS here, end the greedy-path walk.
                    if tid in stop_ids:
                        completed.append((cum_at_pos, base_gen + produced_ids[:pos + 1], True))
                        break
                else:
                    # The for-else branch runs only if we did not break due to EOS:
                    # emit the full greedy block as a continuing beam.
                    new_beams.append((cum_at_pos, base_gen + produced_ids, False))

            # Keep only top-`num_beams` for next block round.
            # Use length-normalized score (matches vLLM native beam_search with
            # length_penalty=1.0): score = cum_logprob / seq_len.
            def _norm_score(b):
                cum, gen, _d = b
                seq_len = max(1, len(gen))
                return cum / seq_len
            new_beams.sort(key=_norm_score, reverse=True)
            beams = new_beams[:num_beams]
            steps_left -= block

        # Merge completed + leftover beams; sort by length-normalized score.
        def _norm_score_final(b):
            cum, gen, _d = b
            # Mirror vLLM: if last token is EOS, count length excluding EOS.
            L = len(gen)
            if L > 0 and (gen[-1] in stop_ids):
                L -= 1
            L = max(1, L)
            return cum / L
        all_final = completed + beams
        all_final.sort(key=_norm_score_final, reverse=True)
        all_final = all_final[:num_beams]

        if not all_final:
            raise RuntimeError('Block beam search produced 0 sequences')

        candidates = []
        best_raw_decoded = None
        for i, (cum_lp, gen_toks, _done) in enumerate(all_final):
            # Strip EOS and following.
            clean_toks = []
            for t in gen_toks:
                if t in stop_ids:
                    break
                clean_toks.append(t)
            new_text = tokenizer.decode(clean_toks, skip_special_tokens=True)
            raw_complete = prefix + new_text
            if i == 0:
                best_raw_decoded = raw_complete
            _, text = parse_asr_output(raw_complete, user_language=self.force_language)
            # Off-distribution alternative beam paths hallucinate a format restart after
            # the real transcript: "...句。[ \t\n]+language Chinese<asr_text>重复" or
            # "...句。\n```python...". Strip everything from the first whitespace-preceded
            # "language X<asr_text>" and from any "\n```" code-block start.
            # Strip any "language X<asr_text>" restart anywhere — with or without
            # preceding whitespace (alternative beams append it directly after 。).
            text = re.sub(r'[ \t\n]*language\s+\S+<asr_text>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'\n`{3}.*', '', text, flags=re.DOTALL)
            text = text.strip()
            candidates.append(text)

        logger.info(f'[Qwen3-ASR vLLM block-beam] {len(candidates)} candidates (block_size={BLOCK_SIZE}):')
        for i, cc in enumerate(candidates):
            logger.info(f'  [{i}] {cc}')

        return candidates, best_raw_decoded, prefix

    def warmup(self, audio, init_prompt=''):
        logger.info('Warming up Qwen3-ASR...')
        results = self.qwen3.transcribe(
            audio=(audio, 16000), language=self.force_language,
        )
        logger.info(f'Warmup result: {results[0].text}')

    def transcribe(self, audio, init_prompt=''):
        raise NotImplementedError('Use Qwen3ASROnline.process_iter()')

    def use_vad(self):
        pass

    def set_translate_task(self):
        pass


class Qwen3ASROnline(OnlineProcessorInterface):

    def __init__(self, asr):
        self.asr = asr
        self.online = self
        # Beam history for data synthesis. Persists across VAD segments within
        # a file; cleared externally via reset_beam_history() at file boundary.
        self._beam_history = []
        self.init()

    def reset_beam_history(self):
        self._beam_history = []

    def get_beam_history(self):
        return list(self._beam_history)

    def init(self, offset=None):
        self.pending_audio = []
        self.all_audio = []
        self.offset = offset if offset is not None else 0.0
        self.is_last = False
        self.committed_text = ''
        self.first_token_latency = None
        self._first_token_generated = False
        self.frame_delay = False
        self.end = self.offset
        self.streaming_state = self.asr.init_state()
        self.last_candidates = []
        self.last_beam_prefix = ''  # prefix used by most recent infer_chunk
        # Top-1 text produced by the most recent infer_chunk within the
        # CURRENT VAD segment.  Used as the "already-seen" prefix when
        # recording beam history for the next chunk.  Resets per segment.
        self._last_top1_in_segment = ''
        self._speech_audio_samples = 0  # total speech audio fed since VAD onset
        self._initial_buffer_samples = int(round(self.asr.initial_buffer * self.SAMPLING_RATE))
        self._initial_buffer_done = False

    def insert_audio_chunk(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        self.pending_audio.append(audio)
        self.all_audio.append(audio)

    def process_iter(
        self, start_time=None, *,
        corrector_model=None, corrector_processor=None, corrector_type='speechlm',
    ):
        if not self.pending_audio:
            return {'first_token_latency': self.first_token_latency}

        new_audio = np.concatenate(self.pending_audio, axis=0)
        self.pending_audio = []

        # Signal to the main loop that we have pending speech content that
        # will need to be flushed via finish() -- even if the initial-buffer
        # gate prevents inference from running this iteration.
        self.frame_delay = True

        # Track speech audio for initial-buffer gating.
        # We count from VAD onset (when init() was called by VAC with a start offset).
        self._speech_audio_samples += new_audio.shape[0]

        all_audio_arr = np.concatenate(self.all_audio, axis=0) if self.all_audio else np.zeros((0,))

        # If initial buffer not yet filled, accumulate without inferring.
        if not self._initial_buffer_done:
            if self._speech_audio_samples < self._initial_buffer_samples:
                return {'first_token_latency': self.first_token_latency}
            # Initial buffer just filled: pass ALL accumulated audio to the
            # first inference so the model sees the full context from VAD onset.
            self._initial_buffer_done = True
            feed_audio = all_audio_arr
        else:
            feed_audio = new_audio

        if all_audio_arr.shape[0] < 1600:
            return {'first_token_latency': self.first_token_latency}

        self.end = self.offset + all_audio_arr.shape[0] / self.SAMPLING_RATE

        candidates, self.streaming_state, beam_prefix = self.asr.infer_chunk(feed_audio, self.streaming_state, is_last=False)

        if candidates:
            self.last_candidates = candidates  # keep full for finish() fallback
            beam_prefix = _strip_intermediate_tail_artifacts(beam_prefix)
            self.last_beam_prefix = beam_prefix

            # Strip the unfixed tail tokens from each candidate before beam history and
            # error corrector. Mid-stream hypotheses always end with sentence-final
            # punctuation (pseudo-streaming artefact); stripping unfixed_token_num tokens
            # removes it and aligns with the training data format.
            # self.last_candidates retains the full text so finish() is unaffected.
            unfixed_n = int(self.streaming_state.get("unfixed_token_num", 0))
            _tok = self.asr.qwen3.processor.tokenizer
            hist_candidates = _strip_tail_tokens(candidates, unfixed_n, _tok)
            hist_candidates = [_strip_intermediate_tail_artifacts(c) for c in hist_candidates]
            hist_candidates = [c for c in hist_candidates if c]
            if not hist_candidates:
                hist_candidates = [_strip_intermediate_tail_artifacts(candidates[0])]
                hist_candidates = [c for c in hist_candidates if c] or [candidates[0]]

            corrected_top1 = hist_candidates[0] if hist_candidates else candidates[0]
            ec_debug = None
            if (
                corrector_model is not None
                and all_audio_arr.shape[0] >= 1600
                and self.asr.error_corrector_output_mode.startswith("suffix")
                and self.asr.error_corrector_scope == "all"
            ):
                corrector_audio = (
                    all_audio_arr
                    if self.asr.error_corrector_audio_window == "cumulative"
                    else feed_audio
                )
                corrected_suffix = _run_error_corrector(
                    audio_np=corrector_audio,
                    candidates=hist_candidates,
                    previous_text=beam_prefix,
                    corrector_model=corrector_model,
                    corrector_processor=corrector_processor,
                    corrector_type=corrector_type,
                    max_new_tokens=self.asr.error_corrector_max_new_tokens,
                    output_mode="suffix",
                    return_debug=True,
                )
                if isinstance(corrected_suffix, dict):
                    ec_debug = corrected_suffix
                    corrected_suffix = ec_debug.get("response")
                if corrected_suffix is not None:
                    accepted, reason = _accept_corrector_suffix(
                        corrected_suffix,
                        previous_text=beam_prefix,
                        candidates=hist_candidates,
                        mode=self.asr.error_corrector_acceptance_mode,
                    )
                    if accepted:
                        corrected_top1 = beam_prefix + corrected_suffix
                    else:
                        print(f"[EC gate] rejected process_iter suffix: {reason}")
                    if ec_debug is not None:
                        ec_debug["accepted"] = accepted
                        ec_debug["accept_reason"] = reason
                        ec_debug["applied_text"] = corrected_top1

            history_entry = {
                "segment_offset": float(self.offset),
                "end_time": float(self.end),
                "previous_transcript": str(beam_prefix),
                "topk": list(hist_candidates),
                "source": "process_iter",
            }
            if ec_debug is not None:
                history_entry["error_corrector"] = ec_debug
            self._beam_history.append(history_entry)
            self._last_top1_in_segment = corrected_top1 if corrected_top1 else self._last_top1_in_segment

        self.frame_delay = True
        return {'first_token_latency': self.first_token_latency}

    def finish(
        self, start_time=None, *,
        corrector_model=None, corrector_processor=None, corrector_type='speechlm',
    ):
        new_audio = np.concatenate(self.pending_audio, axis=0) if self.pending_audio else None
        self.pending_audio = []

        # Compute the true end time from all accumulated audio up front so that
        # the beam_history entry records the correct end_time (previously self.end
        # was only updated in process_iter, causing finish() to emit a duplicate
        # or 0.0 timestamp).
        all_audio_arr = np.concatenate(self.all_audio, axis=0) if self.all_audio else np.zeros((0,), dtype=np.float32)
        self.end = self.offset + all_audio_arr.shape[0] / self.SAMPLING_RATE

        # On final flush: if initial-buffer was never filled via process_iter
        # (audio shorter than initial_buffer), feed the full accumulated speech
        # audio instead of just the trailing fragment so infer_chunk sees the
        # whole utterance.
        if not self._initial_buffer_done:
            self._initial_buffer_done = True
            if all_audio_arr.shape[0] > 0:
                new_audio = all_audio_arr

        candidates, self.streaming_state, beam_prefix = self.asr.infer_chunk(new_audio, self.streaming_state, is_last=True)
        if not candidates:
            candidates = self.last_candidates
            beam_prefix = self.last_beam_prefix

        finish_history_entry = None
        if candidates:
            finish_history_entry = {
                'segment_offset': float(self.offset),
                'end_time': float(self.end),
                'previous_transcript': str(beam_prefix),
                'topk': list(candidates),
                'source': 'finish',
            }
            self._beam_history.append(finish_history_entry)
            # Note: _last_top1_in_segment will be updated below after
            # the corrector runs (if enabled), so we don't set it from
            # the raw candidate here.

        norm_committed = self.committed_text

        full_text = ''
        if candidates and not all(c.strip() == '' for c in candidates):
            top1_text = candidates[0].strip()
            raw_top1_text = top1_text
            fallback_reason = None
            if (
                self.last_candidates
                and self.last_candidates[0].strip()
                and _is_short_number_truncation(self.last_candidates[0], top1_text)
            ):
                top1_text = self.last_candidates[0].strip()
                fallback_reason = "short_number_final_truncation"

            if corrector_model is not None and all_audio_arr.shape[0] >= 1600:
                if self.asr.error_corrector_audio_window == "cumulative":
                    corrector_audio = all_audio_arr
                else:
                    corrector_audio = (
                        new_audio
                        if new_audio is not None and new_audio.shape[0] >= 1600
                        else _recent_audio_window(all_audio_arr)
                    )
                corrected_text = _run_error_corrector(
                    audio_np=corrector_audio,
                    candidates=candidates,
                    previous_text=beam_prefix,
                    corrector_model=corrector_model,
                    corrector_processor=corrector_processor,
                    corrector_type=corrector_type,
                    max_new_tokens=self.asr.error_corrector_max_new_tokens,
                    output_mode=self.asr.error_corrector_output_mode,
                    return_debug=True,
                )
                ec_debug = corrected_text if isinstance(corrected_text, dict) else None
                if ec_debug is not None:
                    corrected_text = ec_debug.get("response")
                if corrected_text is not None:
                    if self.asr.error_corrector_output_mode == "full_transcript":
                        full_text = corrected_text
                        accepted, reason = True, "full_transcript"
                    else:
                        top1_suffix = _strip_intermediate_tail_artifacts(
                            _strip_matching_prefix(top1_text, beam_prefix)
                        ).strip()
                        accepted, reason = _accept_corrector_suffix(
                            corrected_text,
                            previous_text=beam_prefix,
                            candidates=candidates,
                            mode=self.asr.error_corrector_acceptance_mode,
                        )
                        # Empty suffix is a valid stop signal only when the
                        # beam prefix already covers top-1.  For one-shot short
                        # utterances beam_prefix is often empty; accepting an
                        # empty EC output there deletes the ASR hypothesis.
                        if not accepted:
                            print(f"[EC gate] rejected finish suffix: {reason}")
                            full_text = top1_text
                        elif not corrected_text and top1_suffix:
                            full_text = top1_text
                        else:
                            full_text = beam_prefix + corrected_text
                    if ec_debug is not None:
                        ec_debug["accepted"] = accepted
                        ec_debug["accept_reason"] = reason
                        ec_debug["applied_text"] = full_text
                        if finish_history_entry is not None:
                            finish_history_entry["error_corrector"] = ec_debug
                else:
                    full_text = top1_text
            else:
                full_text = top1_text
            if fallback_reason and finish_history_entry is not None:
                finish_history_entry["finish_fallback"] = {
                    "reason": fallback_reason,
                    "raw_top1": raw_top1_text,
                    "applied_top1": top1_text,
                }

        if full_text:
            self._last_top1_in_segment = full_text

        delta = full_text[len(norm_committed):] if full_text.startswith(norm_committed) else ''
        if not delta and full_text and not full_text.startswith(norm_committed):
            import difflib
            sm = difflib.SequenceMatcher(None, norm_committed, full_text)
            match = sm.find_longest_match(0, len(norm_committed), 0, len(full_text))
            if match.size > 0:
                delta = full_text[match.b + match.size:]
            else:
                delta = full_text

        if not self._first_token_generated and start_time is not None and delta:
            self.first_token_latency = time.time() - start_time
            self._first_token_generated = True

        saved_offset = self.offset
        saved_end = self.end
        saved_ftl = self.first_token_latency
        
        # committed_text is always stored in normalized form.
        self.committed_text = norm_committed + delta
        self.init()

        if not delta:
            return {'first_token_latency': saved_ftl}

        return {
            'start': saved_offset,
            'end': saved_end,
            'text': delta,
            'tokens': [],
            'words': [{
                'start': saved_offset, 'end': saved_end,
                'text': delta, 'tokens': [],
            }],
            'first_token_latency': saved_ftl,
        }


# ---------------------------------------------------------------------------
# Error corrector
# ---------------------------------------------------------------------------

def _strip_tail_tokens(candidates: list, n_tokens: int, tokenizer) -> list:
    """Strip the last n_tokens from each decoded candidate string.

    Applied in process_iter (is_last=False) to remove the unfixed tail before
    recording beam history and before passing to the error corrector.
    The full candidates are kept separately as self.last_candidates so that
    finish() can still use them as a fallback without truncation.
    """
    result = []
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        end_idx = max(0, len(ids) - n_tokens)
        if end_idx == 0:
            result.append("")
            continue
        decoded = tokenizer.decode(ids[:end_idx])
        while "�" in decoded and end_idx > 0:
            end_idx -= 1
            decoded = tokenizer.decode(ids[:end_idx]) if end_idx > 0 else ""
        result.append(decoded)
    return result


def _token_confidence(gen_out, new_token_ids):
    """Mean log-prob of generated tokens (requires return_dict_in_generate=True)."""
    if not hasattr(gen_out, 'scores') or not gen_out.scores:
        return float('-inf')
    log_probs = []
    for score, tid in zip(gen_out.scores, new_token_ids):
        log_probs.append(torch.log_softmax(score[0], dim=-1)[tid].item())
    return sum(log_probs) / len(log_probs) if log_probs else float('-inf')


def _run_error_corrector(
    audio_np, candidates, previous_text,
    corrector_model, corrector_processor, corrector_type,
    return_confidence=False,
    max_new_tokens=16,
    output_mode="suffix",
    return_debug=False,
):
    """Run the SpeechLM, LM, or Qwen3-ASR corrector on top-k candidates."""
    prev_display = previous_text
    while prev_display.endswith('\ufffd'):
        prev_display = prev_display[:-1]

    cleaned = []
    for text in candidates:
        while text.endswith('\ufffd'):
            text = text[:-1]
        if text.strip():
            cleaned.append(text)
    if not cleaned:
        if return_debug:
            return {
                "type": corrector_type,
                "output_mode": output_mode,
                "raw_response": None,
                "response": None,
                "confidence": None,
            }
        return (None, None) if return_confidence else None

    # ---- LM (text-only) corrector ----
    if corrector_type == 'lm':
        from LMCorrector.training import format_instruction_for_correction
        instruction = format_instruction_for_correction(
            k_best_candidates=cleaned,
            previous_transcript=prev_display,
        )
        bos_token = corrector_processor.bos_token or ''
        full_text = f'{bos_token}{instruction}\n'
        inputs = corrector_processor(
            full_text, return_tensors='pt', truncation=True, max_length=512,
        )
        model_device = next(corrector_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_out = corrector_model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                output_scores=return_confidence,
                return_dict_in_generate=return_confidence,
                pad_token_id=corrector_processor.pad_token_id,
                eos_token_id=corrector_processor.eos_token_id,
            )
        gen = gen_out.sequences if return_confidence else gen_out
        input_length = inputs['input_ids'].shape[1]
        response = corrector_processor.decode(
            gen[0, input_length:], skip_special_tokens=True,
        ).strip()
        confidence = _token_confidence(gen_out, gen[0, input_length:]) if return_confidence else None

        print('============ LM CORRECTOR =============')
        print(f'Previous: {prev_display}')
        print(f'Candidates: {cleaned}')
        print(f'Corrected suffix: {response}')
        print('=======================================')
        if return_debug:
            return {
                "type": "lm",
                "output_mode": output_mode,
                "raw_response": response,
                "response": response,
                "confidence": confidence,
            }
        return (response, confidence) if return_confidence else response

    # ---- Qwen3-ASR corrector (audio + text, aligned with Qwen3ASRCorrector training) ----
    if corrector_type == 'qwen3asr':
        prompt = _build_qwen3asr_corrector_prompt(cleaned, prev_display, output_mode)
        audio_array = np.asarray(audio_np, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0) if audio_array.shape[0] <= 2 else audio_array[0]

        inputs = corrector_processor(
            text=[prompt],
            audio=[audio_array],
            sampling_rate=16000,
            padding=True,
            return_tensors='pt',
        )

        model_device = next(corrector_model.parameters()).device
        inputs = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        tokenizer = corrector_processor.tokenizer
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or eos_id < 0:
            eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.pad_token_id

        with torch.no_grad():
            gen_out = corrector_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=return_confidence,
                return_dict_in_generate=return_confidence,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
            )

        gen = gen_out.sequences if return_confidence else gen_out
        input_length = inputs['input_ids'].shape[1]
        new_tokens = gen[0, input_length:]

        stop_at = len(new_tokens)
        for i, tid in enumerate(new_tokens.tolist()):
            if tid == eos_id:
                stop_at = i
                break
        raw_response = tokenizer.decode(new_tokens[:stop_at], skip_special_tokens=True).strip()
        if output_mode == "full_transcript":
            response = _normalize_corrector_full_transcript(raw_response)
        else:
            response = _normalize_corrector_suffix(raw_response, prev_display, cleaned)
        confidence = _token_confidence(gen_out, gen[0, input_length:]) if return_confidence else None

        print('========= QWEN3ASR CORRECTOR ==========')
        print(f'Previous: {prev_display}')
        print(f'Candidates: {cleaned}')
        print(f'Raw response: {raw_response}')
        print(f'Corrected {output_mode}: {response}')
        print('=======================================')
        if return_debug:
            return {
                "type": "qwen3asr",
                "output_mode": output_mode,
                "raw_response": raw_response,
                "response": response,
                "confidence": confidence,
            }
        return (response, confidence) if return_confidence else response

    # ---- SpeechLM corrector (audio + text) ----
    from SpeechLMCorrector.training_qwen2audio import format_instruction_for_correction
    instruction = format_instruction_for_correction(
        k_best_candidates=cleaned,
        previous_transcript=prev_display,
    )

    audio_array = np.asarray(audio_np, dtype=np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=0) if audio_array.shape[0] <= 2 else audio_array[0]

    model_config = getattr(corrector_model, 'config', None)
    if model_config is None and hasattr(corrector_model, 'base_model'):
        model_config = getattr(corrector_model.base_model, 'config', None)
    model_type = getattr(model_config, 'model_type', 'ultravox')

    if model_type == 'qwen2_audio':
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant specialized in ASR error correction.'},
            {'role': 'user', 'content': [
                {'type': 'audio', 'audio_url': 'placeholder'},
                {'type': 'text', 'text': instruction},
            ]},
        ]
        full_text = corrector_processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        inputs = corrector_processor(
            text=full_text, audios=[audio_array],
            return_tensors='pt', sampling_rate=16000, padding=True,
        )
    else:
        # Ultravox
        bos_token = corrector_processor.tokenizer.bos_token or ''
        full_text = f'{bos_token}<|audio|>\n{instruction}\n'
        inputs = corrector_processor(
            audio=audio_array, text=full_text,
            return_tensors='pt', sampling_rate=16000,
        )

    model_device = next(corrector_model.parameters()).device
    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        gen_out = corrector_model.generate(
            **inputs, max_new_tokens=8, do_sample=False,
            output_scores=return_confidence,
            return_dict_in_generate=return_confidence,
            pad_token_id=corrector_processor.tokenizer.pad_token_id,
            eos_token_id=corrector_processor.tokenizer.eos_token_id,
        )
    gen = gen_out.sequences if return_confidence else gen_out

    input_length = inputs['input_ids'].shape[1]
    new_tokens = gen[:, input_length:]
    response = corrector_processor.tokenizer.decode(
        new_tokens[0], skip_special_tokens=True,
    ).strip()
    confidence = _token_confidence(gen_out, gen[0, input_length:]) if return_confidence else None

    print('======== SPEECHLM CORRECTOR ============')
    print(f'Previous: {prev_display}')
    print(f'Candidates: {cleaned}')
    print(f'Corrected suffix: {response}')
    print('========================================')
    if return_debug:
        return {
            "type": "speechlm",
            "output_mode": output_mode,
            "raw_response": response,
            "response": response,
            "confidence": confidence,
        }
    return (response, confidence) if return_confidence else response





if __name__ == '__main__':
    from streaming.asr_runner import main_simulation_from_file
    main_simulation_from_file(qwen3_asr_factory, add_args=qwen3asr_args)
