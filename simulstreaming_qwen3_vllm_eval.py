import os
import json
import time
import argparse
import logging
import urllib.request
import multiprocessing as mp
from typing import Tuple, Dict, Any, List
import types

import numpy as np
import soundfile as sf
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

try:
    from evaluate import calculate_cer, calculate_mer
    def compute_error_rate(ref, hyp, lang="zh"):
        return calculate_cer(ref, hyp, lang) * 100.0  # Percentage
    def compute_mer(ref, hyp, lang="zh"):
        return calculate_mer(ref, hyp, lang) * 100.0
except ImportError:
    def compute_error_rate(ref, hyp, lang="zh"):
        return 0.0
    def compute_mer(ref, hyp, lang="zh"):
        return 0.0

logger = logging.getLogger(__name__)

def _streaming_transcribe_beam(self, pcm16k: np.ndarray, state: Any, beams: int = 1) -> Any:
    """
    Custom streaming transcribe with beam search.
    Overrides Qwen3ASRModel.streaming_transcribe to retain top-k candidates.
    """
    if self.backend != "vllm":
        raise ValueError("streaming_transcribe_beam() is supported only for vLLM backend.")
    if state is None or pcm16k is None:
        raise ValueError("state and pcm16k must not be None.")

    x = np.asarray(pcm16k)
    if x.ndim != 1:
        x = x.reshape(-1)

    if x.dtype == np.int16:
        x = (x.astype(np.float32) / 32768.0)
    else:
        x = x.astype(np.float32, copy=False)

    if x.shape[0] > 0:
        state.buffer = np.concatenate([state.buffer, x], axis=0)

    while state.buffer.shape[0] >= state.chunk_size_samples:
        chunk = state.buffer[: state.chunk_size_samples]
        state.buffer = state.buffer[state.chunk_size_samples :]

        if hasattr(state, "audio_accum") and state.audio_accum.shape[0] == 0:
            state.audio_accum = chunk
        else:
            state.audio_accum = np.concatenate([getattr(state, "audio_accum", np.array([], dtype=np.float32)), chunk], axis=0)

        prefix = ""
        if getattr(state, "chunk_id", 0) < getattr(state, "unfixed_chunk_num", 0):
            prefix = ""
        else:
            cur_ids = self.processor.tokenizer.encode(getattr(state, "_raw_decoded", ""))
            k = int(getattr(state, "unfixed_token_num", 0))
            while True:
                end_idx = max(0, len(cur_ids) - k)
                prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                if '\ufffd' not in prefix:
                    break
                else:
                    if end_idx == 0:
                        prefix = ""
                        break
                    k += 1

        prompt = getattr(state, "prompt_raw", "") + prefix
        inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

        outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)

        candidates = []
        for out in outputs[0].outputs:
            raw_decoded = (prefix + out.text) if prefix else out.text
            lang, txt = parse_asr_output(raw_decoded, user_language=getattr(state, "force_language", None))
            candidates.append(txt)

        state._raw_decoded = (prefix + outputs[0].outputs[0].text) if prefix else outputs[0].outputs[0].text
        state.language, state.text = parse_asr_output(state._raw_decoded, user_language=getattr(state, "force_language", None))
        state.candidates = candidates
        state.chunk_id = getattr(state, "chunk_id", 0) + 1

    return state


def evaluate_streaming_audio(
    asr_model: Qwen3ASRModel,
    audio_path: str,
    step_ms: int = 500,
    beams: int = 1
) -> Dict[str, Any]:
    
    with sf.SoundFile(audio_path) as f:
        wav = f.read(dtype="float32", always_2d=False)
        sr = f.samplerate

    # Resample to 16k if needed
    if sr != 16000:
        dur = wav.shape[0] / float(sr)
        n16 = int(round(dur * 16000))
        if n16 > 0:
            x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
            x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
            wav16k = np.interp(x_new, x_old, wav).astype(np.float32)
        else:
            wav16k = np.zeros((0,), dtype=np.float32)
    else:
        wav16k = wav.astype(np.float32)

    step = int(round(step_ms / 1000.0 * 16000))
    state = asr_model.init_streaming_state(
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=2.0,
    )

    pos = 0
    first_token_latency = None
    last_token_latency = None
    previous_text = ""
    
    final_chunk_start_time = None
    
    # Process audio chunks
    while pos < wav16k.shape[0]:
        chunk_start_time = time.time()
        seg = wav16k[pos : pos + step]
        
        is_last_chunk = (pos + step) >= wav16k.shape[0]
        pos += seg.shape[0]
        
        if beams > 1:
            _streaming_transcribe_beam(asr_model, seg, state, beams=beams)
            current_candidates = getattr(state, "candidates", [state.text])
        else:
            asr_model.streaming_transcribe(seg, state)
            current_candidates = [state.text]

        chunk_end_time = time.time()
        
        current_text = state.text
        
        # Calculate First Token Latency (FTL)
        if first_token_latency is None and current_text.strip() != "":
            first_token_latency = (chunk_end_time - chunk_start_time) * 1000.0  # in ms
            
        if is_last_chunk:
            final_chunk_start_time = chunk_start_time

    if final_chunk_start_time is None:
        final_chunk_start_time = time.time()
        
    # Note: LTL is from when the full audio is received (the final chunk start) to the last token
    asr_model.finish_streaming_transcribe(state)
    finish_time = time.time()
    final_text = state.text
    
    if first_token_latency is None and final_text.strip() != "":
        first_token_latency = (finish_time - final_chunk_start_time) * 1000.0

    last_token_latency = (finish_time - final_chunk_start_time) * 1000.0

    if first_token_latency is None: # In case of very short empty audio
        first_token_latency = 0.0

    return {
        "text": final_text,
        "candidates": getattr(state, "candidates", [final_text]),
        "first_token_latency_ms": first_token_latency,
        "last_token_latency_ms": last_token_latency
    }


def _process_chunk(gpu_id: str, files_chunk: List[str], args, references: Dict[str, str], tmp_out: str):
    # Set CUDA device for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Must import here to avoid initializing CUDA context before setting the device
    from qwen_asr import Qwen3ASRModel
    
    logger.info(f"[GPU {gpu_id}] Initializing Qwen3-ASR LLM model...")
    asr = Qwen3ASRModel.LLM(
        model=args.model_path,
        gpu_memory_utilization=0.6,
        max_new_tokens=32,
    )
    
    if getattr(args, "beams", 1) > 1:
        from vllm import SamplingParams
        asr.sampling_params = SamplingParams(
            temperature=0.2, # slight temperature to get diverse candidates since beam search params are missing
            max_tokens=32,
            n=args.beams,
        )

    logger.info(f"[GPU {gpu_id}] Warming up the model...")
    warmup_state = asr.init_streaming_state(
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=2.0,
    )
    warmup_audio = np.zeros(16000, dtype=np.float32)
    asr.streaming_transcribe(warmup_audio[:8000], warmup_state)
    asr.streaming_transcribe(warmup_audio[8000:], warmup_state)
    asr.finish_streaming_transcribe(warmup_state)
    
    results = []
    
    for count, wav_file in enumerate(files_chunk, start=1):
        audio_path = os.path.join(args.audio_dir, wav_file)
        ref_key = wav_file.split(".")[0]
        ground_truth = references.get(ref_key, "")
        
        if not ground_truth:
            continue
            
        try:
            res = evaluate_streaming_audio(asr, audio_path, step_ms=args.step_ms, beams=getattr(args, 'beams', 1))
            hyp_text = res["text"]
            hyp_candidates = res.get("candidates", [hyp_text])
            ftl = res["first_token_latency_ms"]
            ltl = res["last_token_latency_ms"]
            
            cer_val = compute_error_rate(ground_truth, hyp_text, lang=args.lan)
            mer_val = compute_mer(ground_truth, hyp_text, lang=args.lan)
            
            results.append({
                "file": wav_file,
                "hypothesis": hyp_text,
                "candidates": hyp_candidates,
                "reference": ground_truth,
                "cer": cer_val,
                "mer": mer_val,
                "first_token_latency_ms": ftl,
                "last_token_latency_ms": ltl
            })
            
            if count % 10 == 0:
                logger.info(f"[GPU {gpu_id}] Processed {count}/{len(files_chunk)} files.")
                
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Error processing {wav_file}: {e}")
            
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-ASR streaming")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("--reference-file", required=True, help="JSON file with ground truth transcriptions")
    parser.add_argument("--model-path", default="Qwen/Qwen3-ASR-1.7B", help="Path or repo ID for Qwen3-ASR model")
    parser.add_argument("--output-dir", default="save_dir/qwen3_streaming_eval", help="Output directory")
    parser.add_argument("--step-ms", type=int, default=500, help="Streaming step size in ms")
    parser.add_argument("--beams", type=int, default=1, help="Number of beams for beam search. If >1, returns top-k candidates.")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU indices to use (e.g. '0,1,2,3')")
    parser.add_argument("--lan", type=str, default="zh", help="Language for CER/MER evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    with open(args.reference_file, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
        
    if isinstance(ref_data, list):
        references = {}
        for item in ref_data:
            path = item.get("audio_path", "")
            key = path.split("/")[-1].split(".")[0]
            text = item.get("text_zh", item.get("text", ""))
            references[key] = text
    else:
        references = ref_data

    wav_files = [f for f in os.listdir(args.audio_dir) if f.endswith(".wav")]
    logger.info(f"Found {len(wav_files)} WAV files to process.")
    
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_list:
        gpu_list = ["0"]
        
    num_gpus = len(gpu_list)
    chunk_size = len(wav_files) // num_gpus + (1 if len(wav_files) % num_gpus != 0 else 0)
    
    if chunk_size == 0:
        logger.warning("No files to process or chunk size is 0.")
        return

    file_chunks = [wav_files[i:i + chunk_size] for i in range(0, len(wav_files), chunk_size)]
    
    ctx = mp.get_context("spawn")
    processes = []
    tmp_files = []
    
    logger.info(f"Starting tracking across {num_gpus} GPUs. Splitting workload into chunks of up to {chunk_size} files.")
    
    for i, chunk in enumerate(file_chunks):
        if i >= num_gpus:
            break
        gpu_id = gpu_list[i]
        tmp_out = os.path.join(args.output_dir, f"tmp_results_gpu{gpu_id}_{i}.json")
        tmp_files.append(tmp_out)
        
        p = ctx.Process(target=_process_chunk, args=(gpu_id, chunk, args, references, tmp_out))
        processes.append(p)
        p.start()
        
    failed_workers = []
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed_workers.append((p.pid, p.exitcode))

    if failed_workers:
        raise RuntimeError(f"One or more workers failed: {failed_workers}")
        
    logger.info("All processes complete! Consolidating findings...")
        
    all_results = []
    for tmp_out in tmp_files:
        if os.path.exists(tmp_out):
            with open(tmp_out, "r", encoding="utf-8") as f:
                all_results.extend(json.load(f))
            os.remove(tmp_out)
            
    count = len(all_results)
    
    if count > 0:
        avg_cer = sum(r["cer"] for r in all_results) / count
        avg_mer = sum(r["mer"] for r in all_results) / count
        avg_ftl = sum(r["first_token_latency_ms"] for r in all_results) / count
        avg_ltl = sum(r["last_token_latency_ms"] for r in all_results) / count
        
        evaluation_results = {
            "average_cer": avg_cer,
            "average_mer": avg_mer,
            "average_first_token_latency_ms": avg_ftl,
            "average_last_token_latency_ms": avg_ltl,
            "per_file_results": all_results
        }
        
        out_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Evaluation complete! Results saved to {out_path}")
        logger.info(f"Average CER: {avg_cer:.2f}%")
        logger.info(f"Average MER: {avg_mer:.2f}%")
        logger.info(f"Average FTL: {avg_ftl:.2f}ms")
        logger.info(f"Average LTL: {avg_ltl:.2f}ms")
    else:
        logger.warning("No files were successfully processed.")

    # Cleanup multi-gpu hang
    try:
        from vllm.distributed import destroy_model_parallel
        import ray
        destroy_model_parallel()
        ray.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        pass

if __name__ == "__main__":
    main()
