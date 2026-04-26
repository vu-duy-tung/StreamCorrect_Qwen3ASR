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
# Text normalization -- matches the canonical form used in
# data_synthesize_qwen3asr.py so that inference-time prompts align with
# the training data format.
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def _normalize_text(s):
    """Canonical form: strip ASR specials, U+FFFD, whitespace, punctuation;
    NFKC fold; lowercase ASCII."""
    if not s:
        return ""
    s = _SPECIAL_TOKEN_RE.sub("", s)
    s = s.replace("\ufffd", "")
    s = unicodedata.normalize("NFKC", s)
    out = []
    for ch in s:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P"):
            continue
        if ch.isascii() and ch.isalpha():
            ch = ch.lower()
        out.append(ch)
    return "".join(out)


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
        '--error-corrector-type', type=str, choices=['speechlm', 'lm'], default='speechlm',
        help='Corrector type: "speechlm" (audio+text, default) or "lm" (text-only).',
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
    )
    return asr, Qwen3ASROnline(asr)


class Qwen3ASRBackendASR(ASRBase):

    sep = ''

    def __init__(self, language, model_path, beams, logdir, initial_buffer=1.0, beam_block_size="auto"):
        self.language = language
        self.beams = beams
        self.logdir = logdir
        self.initial_buffer = initial_buffer
        self.beam_block_size = beam_block_size
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
            disable_log_stats=True
        )
        logger.info(f'Language: {language} -> {self.force_language}')

    def init_state(self):
        """Initializes a custom streaming state for pseudo-streaming via HuggingFace."""
        import numpy as np
        return {
            'chunk_id': 0,
            'unfixed_chunk_num': 2,
            'unfixed_token_num': 3,
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
                # Store normalized form so prefix construction on the next
                # chunk sees the same canonical format as training data.
                state['_raw_decoded'] = _normalize_text(best_raw)
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
            state['_raw_decoded'] = _normalize_text(text)
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

        # Normalize the raw decoded text so the prefix matches training format
        # (no punctuation, whitespace, or special tokens).
        norm_raw_decoded = _normalize_text(state['_raw_decoded'])
        prefix = ""
        if state['chunk_id'] >= state['unfixed_chunk_num']:
            cur_ids = tokenizer.encode(norm_raw_decoded, add_special_tokens=False)
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
        stop_ids = set()
        if eos_id is not None: stop_ids.add(eos_id)
        for _tkn in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
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
            new_text = tokenizer.decode(clean_toks, skip_special_tokens=False)
            new_text = new_text.replace("<|im_end|>", "")
            raw_complete = prefix + new_text
            if i == 0:
                best_raw_decoded = raw_complete
            _, text = parse_asr_output(raw_complete, user_language=self.force_language)
            candidates.append(text.strip())

        # Normalize candidates so downstream (delta extraction, beam history,
        # error corrector) all see canonical form matching training data.
        norm_candidates = [_normalize_text(c) for c in candidates]
        norm_best_raw = _normalize_text(best_raw_decoded)

        logger.info(f'[Qwen3-ASR vLLM block-beam] {len(norm_candidates)} candidates (block_size={BLOCK_SIZE}):')
        for i, cc in enumerate(norm_candidates):
            logger.info(f'  [{i}] {cc}')

        return norm_candidates, norm_best_raw, _normalize_text(prefix)

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
            self.last_candidates = candidates
            self.last_beam_prefix = beam_prefix

            # Run error corrector chunk-by-chunk (matching training format):
            # previous_text = previous[:-k] (the fixed prefix used by beam search)
            # candidates = full transcripts starting from that prefix
            # The corrector returns a suffix to append after previous_text.
            corrected_top1 = candidates[0]  # default: raw top-1
            if corrector_model is not None and all_audio_arr.shape[0] >= 1600:
                corrected_suffix = _run_error_corrector(
                    audio_np=all_audio_arr,
                    candidates=candidates,
                    previous_text=beam_prefix,
                    corrector_model=corrector_model,
                    corrector_processor=corrector_processor,
                    corrector_type=corrector_type,
                )
                if corrected_suffix is not None:
                    corrected_top1 = beam_prefix + _normalize_text(corrected_suffix)

            self._beam_history.append({
                'end_time': float(self.end),
                'previous_transcript': str(beam_prefix),
                'topk': list(candidates),
                'source': 'process_iter',
            })
            # Update per-segment top-1 memory for the NEXT chunk.
            # Use the corrected top-1 so subsequent chunks build on the
            # corrected transcript rather than the raw ASR output.
            self._last_top1_in_segment = _normalize_text(corrected_top1) if corrected_top1 else self._last_top1_in_segment

        self.frame_delay = True
        return {'first_token_latency': self.first_token_latency}

    def finish(
        self, start_time=None, *,
        corrector_model=None, corrector_processor=None, corrector_type='speechlm',
    ):
        new_audio = np.concatenate(self.pending_audio, axis=0) if self.pending_audio else None
        self.pending_audio = []

        # On final flush: if initial-buffer was never filled via process_iter
        # (audio shorter than initial_buffer), feed the full accumulated speech
        # audio instead of just the trailing fragment so infer_chunk sees the
        # whole utterance.
        if not self._initial_buffer_done:
            self._initial_buffer_done = True
            all_audio_arr = np.concatenate(self.all_audio, axis=0) if self.all_audio else np.zeros((0,), dtype=np.float32)
            if all_audio_arr.shape[0] > 0:
                new_audio = all_audio_arr

        candidates, self.streaming_state, beam_prefix = self.asr.infer_chunk(new_audio, self.streaming_state, is_last=True)
        if not candidates:
            candidates = self.last_candidates
            beam_prefix = self.last_beam_prefix

        if candidates:
            self._beam_history.append({
                'end_time': float(self.end),
                'previous_transcript': str(beam_prefix),
                'topk': list(candidates),
                'source': 'finish',
            })
            # Note: _last_top1_in_segment will be updated below after
            # the corrector runs (if enabled), so we don't set it from
            # the raw candidate here.

        all_audio_arr = np.concatenate(self.all_audio, axis=0) if self.all_audio else np.zeros((0,))

        # committed_text is always in normalized form; candidates are already
        # normalized by _beam_search.
        norm_committed = _normalize_text(self.committed_text)

        full_text = ''
        if candidates and not all(c.strip() == '' for c in candidates):
            top1_text = candidates[0].strip()

            if corrector_model is not None and all_audio_arr.shape[0] >= 1600:
                corrected_suffix = _run_error_corrector(
                    audio_np=all_audio_arr,
                    candidates=candidates,
                    previous_text=beam_prefix,
                    corrector_model=corrector_model,
                    corrector_processor=corrector_processor,
                    corrector_type=corrector_type,
                )
                if corrected_suffix is not None:
                    full_text = beam_prefix + _normalize_text(corrected_suffix)
                else:
                    full_text = top1_text
            else:
                full_text = top1_text

        # Update _last_top1_in_segment to reflect the corrected full_text
        # (not the raw candidate), so the beam_history and subsequent
        # processing see the corrected version.
        if full_text:
            self._last_top1_in_segment = _normalize_text(full_text)

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

def _run_error_corrector(
    audio_np, candidates, previous_text,
    corrector_model, corrector_processor, corrector_type,
):
    """Run the SpeechLM or LM error corrector on top-k beam search candidates."""
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
        return None

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
            gen = corrector_model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=corrector_processor.pad_token_id,
                eos_token_id=corrector_processor.eos_token_id,
            )
        response = corrector_processor.decode(
            gen[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
        ).strip()

        print('============ LM CORRECTOR =============')
        print(f'Previous: {prev_display}')
        print(f'Candidates: {cleaned}')
        print(f'Corrected suffix: {response}')
        print('=======================================')
        return response

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
        gen = corrector_model.generate(
            **inputs, max_new_tokens=8, do_sample=False,
            pad_token_id=corrector_processor.tokenizer.pad_token_id,
            eos_token_id=corrector_processor.tokenizer.eos_token_id,
        )

    input_length = inputs['input_ids'].shape[1]
    new_tokens = gen[:, input_length:]
    response = corrector_processor.tokenizer.decode(
        new_tokens[0], skip_special_tokens=True,
    ).strip()

    print('======== SPEECHLM CORRECTOR ============')
    print(f'Previous: {prev_display}')
    print(f'Candidates: {cleaned}')
    print(f'Corrected suffix: {response}')
    print('========================================')
    return response


if __name__ == '__main__':
    from streaming.asr_runner import main_simulation_from_file
    main_simulation_from_file(qwen3_asr_factory, add_args=qwen3asr_args)
