import os
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoConfig
from peft import PeftModel
from streaming.base import OnlineProcessorInterface
from streaming.vad_iterator import FixedVADIterator

import torch
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)


class VACProcessor(OnlineProcessorInterface):
    """Wraps an OnlineProcessorInterface with Silero VAD.

    Receives small audio chunks, runs VAD, and forwards speech segments
    to the inner ASR processor. Triggers finish() on end-of-speech.
    """

    def __init__(
        self,
        online_chunk_size,
        online,
        min_buffered_length=1,
        min_speech_duration_ms=0,
        use_error_corrector=False,
        error_corrector_ckpt=None,
        error_corrector_base_model=None,
        error_corrector_type="speechlm",  # "speechlm", "lm", or "qwen3asr"
    ):
        self.online_chunk_size = online_chunk_size
        self.online = online
        self.min_buffered_frames = int(min_buffered_length * self.SAMPLING_RATE)

        # VAC:
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        self.vac = FixedVADIterator(model)  # Default options: 500ms silence, 100ms padding, etc.
        self.min_speech_frames = int(min_speech_duration_ms * self.SAMPLING_RATE / 1000)

        self.init()

        # Error corrector initialization
        self._corrector_model = None
        self._corrector_processor = None
        self._corrector_type = error_corrector_type  # "speechlm" or "lm"

        if use_error_corrector:
            if error_corrector_type == "lm":
                # LM corrector (text-only, Llama with LoRA)
                from transformers import AutoModelForCausalLM
                
                base_model_id = error_corrector_base_model or "meta-llama/Llama-3.2-1B"
                if error_corrector_ckpt:
                    checkpoint_path = error_corrector_ckpt
                else:
                    checkpoint_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "LMCorrector/runs/llama_lora_finetuned"
                    )
                logger.info(f"Loading LM corrector from {checkpoint_path}")
                
                # Load tokenizer
                self._corrector_processor = AutoTokenizer.from_pretrained(
                    base_model_id,
                    trust_remote_code=True
                )
                if self._corrector_processor.pad_token is None:
                    self._corrector_processor.pad_token = self._corrector_processor.eos_token
                    self._corrector_processor.pad_token_id = self._corrector_processor.eos_token_id
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    device_map="cuda",
                )
                
                # Load LoRA adapter
                self._corrector_model = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_path,
                    is_trainable=False,
                )
                self._corrector_model.eval()
                logger.info("LM corrector loaded successfully")
            elif error_corrector_type == "qwen3asr":
                # Qwen3-ASR corrector (audio + text; training-aligned prompt in runtime script)
                base_model_id = error_corrector_base_model or "Qwen/Qwen3-ASR-0.6B"
                checkpoint_path = error_corrector_ckpt
                if not checkpoint_path:
                    raise ValueError(
                        "--error-corrector-ckpt is required when --error-corrector-type qwen3asr"
                    )
                logger.info(f"Loading Qwen3-ASR corrector from {checkpoint_path}")

                # Ensure qwen3_asr architecture is registered in AutoModel.
                try:
                    import qwen_asr as _qwen_asr  # noqa: F401
                except ImportError as e:
                    raise ImportError(
                        "qwen_asr is required for --error-corrector-type qwen3asr"
                    ) from e

                self._corrector_processor = AutoProcessor.from_pretrained(
                    base_model_id,
                    trust_remote_code=True,
                )
                base_outer = AutoModel.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=False,
                )
                thinker = base_outer.thinker.to("cuda")
                self._corrector_model = PeftModel.from_pretrained(
                    thinker,
                    checkpoint_path,
                    is_trainable=False,
                ).to("cuda")
                self._corrector_model.eval()
                logger.info("Qwen3-ASR corrector loaded successfully")
            else:
                # SpeechLM corrector (audio + text, with LoRA)
                base_model_id = error_corrector_base_model or "fixie-ai/ultravox-v0_5-llama-3_2-1b"
                if error_corrector_ckpt:
                    checkpoint_path = error_corrector_ckpt
                else:
                    checkpoint_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "SpeechLMCorrector/ultravox_lora_continued_more_erroneous_5/checkpoint-2895"
                    )
                logger.info(f"Loading SpeechLM corrector from {checkpoint_path}")

                # Load processor
                self._corrector_processor = AutoProcessor.from_pretrained(
                    base_model_id,
                    trust_remote_code=True
                )

                # Detect model type and load with the appropriate class
                model_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
                model_type = getattr(model_config, 'model_type', '')
                
                if model_type == 'qwen2_audio':
                    from transformers import Qwen2AudioForConditionalGeneration
                    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                        base_model_id,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        device_map="cuda",
                    )
                else:
                    base_model = AutoModel.from_pretrained(
                        base_model_id,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        device_map="cuda",
                    )

                # Load LoRA adapter
                self._corrector_model = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_path,
                    is_trainable=False,
                )
                self._corrector_model.eval()
                logger.info("SpeechLM corrector loaded successfully")
        else:
            logger.info("Error corrector disabled")

    @property
    def first_token_latency(self):
        """Delegate first_token_latency to the wrapped online processor."""
        return getattr(self.online, 'first_token_latency', None)

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames
        self._speech_frames_sent = 0

    def clear_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        logger.info(f"VAD result: {res}")
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0] - self.buffer_offset
            frame = max(0, frame)

            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self._speech_frames_sent += len(send_audio)
                self.buffer_offset += len(self.audio_buffer)
                self.clear_buffer()

            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                if frame > 0:
                    send_audio = self.audio_buffer[:frame]
                    self.online.insert_audio_chunk(send_audio)
                    self.current_online_chunk_buffer_size += len(send_audio)
                    self._speech_frames_sent += len(send_audio)
                self.is_currently_final = True
                keep_frames = min(len(self.audio_buffer) - frame, self.min_buffered_frames)
                self.buffer_offset += len(self.audio_buffer) - keep_frames
                self.audio_buffer = self.audio_buffer[-keep_frames:]

            else:
                beg = max(0, res["start"] - self.buffer_offset)
                end = max(0, res["end"] - self.buffer_offset)
                self.status = 'nonvoice'
                if beg < end:
                    send_audio = self.audio_buffer[beg:end]
                    self.online.init(offset=((beg + self.buffer_offset) / self.SAMPLING_RATE))
                    self.online.insert_audio_chunk(send_audio)
                    self.current_online_chunk_buffer_size += len(send_audio)
                    self._speech_frames_sent += len(send_audio)
                self.is_currently_final = True
                keep_frames = min(len(self.audio_buffer) - end, self.min_buffered_frames)
                self.buffer_offset += len(self.audio_buffer) - keep_frames
                self.audio_buffer = self.audio_buffer[-keep_frames:]

        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self._speech_frames_sent += len(self.audio_buffer)
                self.buffer_offset += len(self.audio_buffer)
                self.clear_buffer()
            else:
                self.buffer_offset += max(0, len(self.audio_buffer) - self.min_buffered_frames)
                self.audio_buffer = self.audio_buffer[-self.min_buffered_frames:]

        logger.info(f"Current online chunk buffer size: {self.current_online_chunk_buffer_size}")

    def process_iter(self, start_time=None):
        if self.is_currently_final:
            return self.finish(start_time=start_time)
        elif self.current_online_chunk_buffer_size >= self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter(
                start_time=start_time,
                corrector_model=self._corrector_model,
                corrector_processor=self._corrector_processor,
                corrector_type=self._corrector_type,
            )
            return ret
        else:
            logger.info(f"No online update, only VAD. {self.status}")
            return {"first_token_latency": self.first_token_latency}

    def finish(self, start_time=None):
        if self.min_speech_frames > 0 and self._speech_frames_sent < self.min_speech_frames:
            logger.info(
                f"Speech too short ({self._speech_frames_sent} frames < min {self.min_speech_frames}); skipping."
            )
            self.init()
            return {'text': '', 'first_token_latency': None}

        # Fallback: if VAD never detected any speech (e.g. for very short or
        # quiet clips that fall below Silero VAD's minimum-speech-duration
        # threshold), flush whatever is still in the VAC audio_buffer into the
        # online processor so the ASR can still transcribe it.
        if self._speech_frames_sent == 0 and len(self.audio_buffer) > 0:
            logger.info(
                f"VAD detected no speech; falling back to raw audio_buffer ({len(self.audio_buffer)} frames)."
            )
            self.online.init(offset=self.buffer_offset / self.SAMPLING_RATE)
            self.online.insert_audio_chunk(self.audio_buffer)
            self._speech_frames_sent += len(self.audio_buffer)
            self.clear_buffer()

        ret = self.online.finish(
            start_time=start_time,
            corrector_model=self._corrector_model,
            corrector_processor=self._corrector_processor,
            corrector_type=self._corrector_type,
        )
        self.init()
        return ret