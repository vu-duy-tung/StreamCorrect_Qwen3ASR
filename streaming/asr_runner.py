#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import random
from functools import lru_cache
from multiprocessing import get_context

import torch
import numpy as np
import librosa

logger = logging.getLogger(__name__)


@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


def processor_args(parser):
    group = parser.add_argument_group("Streaming")
    group.add_argument(
        '--chunk-size', type=float, default=0.5,
        help='VAD chunk size in seconds — how often the model is called during streaming. (default: 0.5)',
    )
    group.add_argument(
        '--language', '--lang', type=str, default='zh', dest='language',
        help='Source language code, e.g. zh, yue, en. (default: zh)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save transcriptions and evaluation results.',
    )
    parser.add_argument(
        '-l', '--log-level', dest='log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Log verbosity level. (default: INFO)',
    )


def asr_factory(args, factory):
    asr, online = factory(args)
    from streaming.vac_processor import VACProcessor
    online = VACProcessor(
        args.chunk_size,
        online,
        use_error_corrector=bool(getattr(args, 'error_corrector_ckpt', None)),
        error_corrector_ckpt=getattr(args, 'error_corrector_ckpt', None),
        error_corrector_base_model=getattr(args, 'error_corrector_base_model', None),
        error_corrector_type=getattr(args, 'error_corrector_type', 'speechlm'),
    )
    return asr, online


def set_logging(args, logger):
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)


def simulation_args(parser):
    parser.add_argument(
        'audio_path', type=str,
        help='Path to a single .wav file, or a directory of audio files for batch mode.',
    )
    parser.add_argument(
        '--max-files', type=int, default=None,
        help='Maximum number of audio files to process in batch mode.',
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of parallel workers for batch processing. Each worker uses its own GPU. (default: 1)',
    )
    parser.add_argument(
        '--gpus', type=str, default='0,1,2,3,4,5,6,7',
        help='Comma-separated GPU IDs for parallel workers, e.g. "0,1,2,3". (default: 0,1,2,3,4,5,6,7)',
    )
    parser.add_argument(
        '--reference-file', type=str, default=None,
        help='Path to transcript JSON (list of {audio_path, text_zh}) for automatic CER evaluation.',
    )


def get_audio_files(path):
    extensions = ('wav', 'mp3', 'flac', 'm4a')
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = []
        for root, _, names in os.walk(path):
            for name in sorted(names):
                if name.lower().rsplit('.', 1)[-1] in extensions:
                    files.append(os.path.join(root, name))
        return files
    else:
        raise ValueError(f"Path does not exist: {path}")


def _worker_process_files(worker_id, gpu_id, audio_files, args_dict, factory_module, factory_name):
    import argparse, importlib
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    load_audio.cache_clear()
    args = argparse.Namespace(**args_dict)
    factory = getattr(importlib.import_module(factory_module), factory_name)
    logging.basicConfig(format=f'[Worker {worker_id} GPU {gpu_id}] %(levelname)s\t%(message)s')
    logger_w = logging.getLogger(f"worker_{worker_id}")
    logger_w.setLevel(args.log_level)
    logger_w.info(f"Starting with {len(audio_files)} files")
    asr, online = asr_factory(args, factory)
    if audio_files:
        asr.warmup(load_audio_chunk(audio_files[0], 0, 1))
    results = []
    for idx, f in enumerate(audio_files, 1):
        logger_w.info(f"[{idx}/{len(audio_files)}] {os.path.basename(f)}")
        try:
            results.append(process_single_audio_file(f, args, asr, online, args.chunk_size, factory))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({'file': f, 'duration': 0, 'segments': [],
                            'final_text': '', 'first_token_latency': None,
                            'last_token_latency': None, 'error': str(e)})
    try:
        from vllm.distributed import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    # Delete model objects so vLLM's __del__ releases GPU/semaphore resources
    # before the process is forcibly killed in _process_wrapper.
    del online, asr
    return results


def _process_wrapper(worker_id, gpu_id, files, arg_dict, fac_mod, fac_name, queue):
    try:
        queue.put({worker_id: _worker_process_files(worker_id, gpu_id, files, arg_dict, fac_mod, fac_name)})
    except Exception as e:
        queue.put({worker_id: e})
        import traceback; traceback.print_exc()
    finally:
        import logging as _log
        # Silence vLLM's "engine core died" ERROR that fires when we
        # terminate the engine's child process below.
        _log.getLogger('vllm').setLevel(_log.CRITICAL)
        import multiprocessing
        for p in multiprocessing.active_children():
            p.terminate(); p.join(timeout=1)
            if p.is_alive(): p.kill()
        os._exit(0)


def run_parallel_batch(audio_files, args, factory_module, factory_name, num_workers, gpu_list):
    worker_files = [[] for _ in range(num_workers)]
    for i, f in enumerate(audio_files):
        worker_files[i % num_workers].append(f)
    worker_gpus = [gpu_list[i % len(gpu_list)] for i in range(num_workers)]
    args_dict = vars(args).copy()
    logger.info(f"Parallel batch: {num_workers} workers on GPUs {worker_gpus}")
    ctx = get_context('spawn')
    queue = ctx.SimpleQueue()
    processes = []
    submitted = 0
    import time as _time
    STAGGER_SECS = 8  # prevent simultaneous vLLM engine init (GPU resource contention)
    for wid in range(num_workers):
        if worker_files[wid]:
            p = ctx.Process(target=_process_wrapper,
                            args=(wid, worker_gpus[wid], worker_files[wid],
                                  args_dict, factory_module, factory_name, queue),
                            daemon=False)
            p.start(); processes.append(p); submitted += 1
            if wid < num_workers - 1:
                _time.sleep(STAGGER_SECS)
    all_results = []
    for _ in range(submitted):
        try:
            res_dict = queue.get()
            for wid, res in res_dict.items():
                if isinstance(res, list):
                    all_results.extend(res)
                else:
                    logger.error(f"Worker {wid} failed: {res}")
        except Exception as e:
            logger.error(f"Queue get failed: {e}")
    return all_results


def process_single_audio_file(audio_path, args, asr, online, min_chunk, factory):
    if hasattr(online, 'is_currently_final'):
        online.is_currently_final = False
    inner = getattr(online, 'online', online)
    if hasattr(inner, 'reset_beam_history'):
        inner.reset_beam_history()

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    logger.info(f"Processing: {os.path.basename(audio_path)} ({duration:.2f}s)")

    beg = 0.0
    end = beg + min_chunk
    start_time = None
    all_transcriptions = []
    first_token_latency = None
    last_token_latency = None
    last_speech_end = None

    def output_transcript(o, now=None):
        if 'start' in o:
            ts = o['start']; te = o['end']; text = o['text']
            t = now if now is not None else (time.time() - _proc_start)
            logger.debug(f"{t*1000:.1f} {ts*1000:.0f} {te*1000:.0f} {text}")
            print(f"{t*1000:.4f} {ts*1000:.0f} {te*1000:.0f} {text}", flush=True)
            all_transcriptions.append({'emission_time': t, 'start': ts, 'end': te, 'text': text.strip()})

    _proc_start = time.time()

    # Computationally-unaware simulation: feed audio in fixed chunks without real-time pacing.
    while True:
        a = load_audio_chunk(audio_path, beg, end)
        online.insert_audio_chunk(a)
        if start_time is None:
            start_time = time.time()
        last_speech_end = time.time()
        try:
            o = online.process_iter(start_time=start_time)
            if first_token_latency is None and o.get('first_token_latency') is not None:
                first_token_latency = o['first_token_latency']
        except AssertionError as e:
            logger.error(f"assertion error: {e}")
            o = {}
        output_transcript(o, now=end)
        if 'text' in o:
            last_token_latency = time.time() - last_speech_end
        if end >= duration:
            break
        beg = end
        end = min(end + min_chunk, duration)

    # Flush remaining audio.
    print(online.online.frame_delay)
    get_remained = online.online.frame_delay
    o = online.finish(start_time=start_time)
    if first_token_latency is None and o.get('first_token_latency') is not None:
        first_token_latency = o['first_token_latency']
    if hasattr(online, 'is_currently_final'):
        online.is_currently_final = False
    if not get_remained and o and o.get('text'):
        get_remained = True
        last_speech_end = time.time()
    if get_remained:
        output_transcript(o)
        if 'text' in o:
            last_token_latency = time.time() - last_speech_end

    if first_token_latency is not None:
        print(f"\nFirst Token Latency: {first_token_latency*1000:.2f} ms")
    if last_token_latency is not None:
        print(f"Last Token Latency:  {last_token_latency*1000:.2f} ms")

    final_text = ' '.join(s['text'] for s in all_transcriptions)
    if final_text:
        print("\n" + "="*80 + "\nFINAL TRANSCRIPTION:\n" + final_text + "\n" + "="*80)

    output_dir = getattr(args, 'output_dir', None)
    if output_dir and hasattr(inner, 'get_beam_history'):
        try:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(audio_path))[0]
            with open(os.path.join(output_dir, f"{base}_beam_history.json"), 'w', encoding='utf-8') as f:
                json.dump({'audio_path': audio_path, 'duration': duration, 'final_text': final_text,
                           'first_token_latency_ms': first_token_latency*1000 if first_token_latency else None,
                           'last_token_latency_ms': last_token_latency*1000 if last_token_latency else None,
                           'history': inner.get_beam_history()}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write beam history: {e}")

    return {'file': audio_path, 'duration': duration, 'segments': all_transcriptions,
            'final_text': final_text, 'first_token_latency': first_token_latency,
            'last_token_latency': last_token_latency}


def _cleanup_vllm():
    try:
        from vllm.distributed import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    import multiprocessing as _mp
    logging.getLogger('vllm.v1.engine.core_client').setLevel(logging.CRITICAL)
    logging.getLogger('vllm.engine.multiprocessing.core_client').setLevel(logging.CRITICAL)
    for p in _mp.active_children():
        try:
            p.terminate(); p.join(timeout=1)
            if p.is_alive(): p.kill()
        except Exception:
            pass


def _save_batch_results(batch_results, audio_files, output_dir, reference_file, language):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    ftl = [r['first_token_latency'] for r in batch_results if r.get('first_token_latency')]
    ltl = [r['last_token_latency'] for r in batch_results if r.get('last_token_latency')]
    summary = {
        'total_files': len(audio_files),
        'processed_files': len(batch_results),
        'average_first_token_latency_ms': sum(ftl)/len(ftl)*1000 if ftl else None,
        'average_last_token_latency_ms': sum(ltl)/len(ltl)*1000 if ltl else None,
        'results': [{'file': os.path.basename(r['file']), 'duration': r['duration'],
                     'transcription': r['final_text'],
                     'first_token_latency_ms': r['first_token_latency']*1000 if r.get('first_token_latency') else None,
                     'last_token_latency_ms': r['last_token_latency']*1000 if r.get('last_token_latency') else None}
                    for r in batch_results]
    }
    with open(os.path.join(output_dir, 'batch_transcriptions.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not reference_file:
        return
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from evaluate import load_references, evaluate_transcriptions
        refs = load_references(reference_file, language=language)
        generated = {os.path.basename(r['file']): r['final_text'] for r in batch_results}
        ftl_map = {os.path.basename(r['file']): r['first_token_latency'] for r in batch_results if r.get('first_token_latency')}
        ltl_map = {os.path.basename(r['file']): r['last_token_latency'] for r in batch_results if r.get('last_token_latency')}
        eval_results = evaluate_transcriptions(refs, generated, language)
        eval_results['average_first_token_latency_ms'] = sum(ftl)/len(ftl)*1000 if ftl else None
        eval_results['average_last_token_latency_ms'] = sum(ltl)/len(ltl)*1000 if ltl else None
        for row in eval_results['per_file_results']:
            fn = row['file']
            if fn in ftl_map: row['first_token_latency_ms'] = ftl_map[fn]*1000
            if fn in ltl_map: row['last_token_latency_ms'] = ltl_map[fn]*1000
        print("\n" + "="*80 + "\nEVALUATION RESULTS\n" + "="*80)
        print(f"Matched: {eval_results['matched_files']} / {eval_results['total_files']}")
        print(f"CER: {eval_results['average_cer']*100:.2f}%   MER: {eval_results['average_mer']*100:.2f}%")
        print("="*80)
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {output_dir}/evaluation_results.json")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback; traceback.print_exc()


def main_simulation_from_file(factory, add_args=None):
    import argparse
    parser = argparse.ArgumentParser()
    processor_args(parser)
    if add_args is not None:
        add_args(parser)
    simulation_args(parser)
    args = parser.parse_args()

    set_logging(args, logger)
    random_seed(21)

    audio_path = args.audio_path
    is_directory = os.path.isdir(audio_path)

    if is_directory:
        audio_files = get_audio_files(audio_path)
        if not audio_files:
            logger.error(f"No audio files found in: {audio_path}")
            sys.exit(1)
        if args.max_files:
            audio_files = audio_files[:args.max_files]
        logger.info(f"Batch mode: {len(audio_files)} files")

        gpu_list = [g.strip() for g in args.gpus.split(',')]
        num_workers = min(args.workers, len(audio_files))

        if num_workers > 1:
            batch_results = run_parallel_batch(
                audio_files, args, factory.__module__, factory.__name__, num_workers, gpu_list
            )
        else:
            asr, online = asr_factory(args, factory)
            asr.warmup(load_audio_chunk(audio_files[0], 0, 1))
            batch_results = []
            for idx, f in enumerate(audio_files, 1):
                logger.info(f"[{idx}/{len(audio_files)}] {os.path.basename(f)}")
                try:
                    batch_results.append(process_single_audio_file(f, args, asr, online, args.chunk_size, factory))
                except Exception as e:
                    logger.error(f"Error: {e}")
                    import traceback; traceback.print_exc()
                    raise

        print(f"\nBatch complete: {len(batch_results)}/{len(audio_files)} files processed")
        _save_batch_results(batch_results, audio_files, args.output_dir, args.reference_file, args.language)

    else:
        # Single file
        duration = len(load_audio(audio_path)) / 16000
        logger.info(f"Duration: {duration:.2f}s")
        asr, online = asr_factory(args, factory)
        asr.warmup(load_audio_chunk(audio_path, 0, 1))
        print("ASR warmup complete.\n")

        result = process_single_audio_file(audio_path, args, asr, online, args.chunk_size, factory)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, 'final_transcription.txt'), 'w', encoding='utf-8') as f:
                f.write(result['final_text'])
            with open(os.path.join(args.output_dir, 'segments_with_timing.json'), 'w', encoding='utf-8') as f:
                json.dump({'audio_file': audio_path, 'duration': result['duration'],
                           'segments': result['segments']}, f, indent=2, ensure_ascii=False)

        if args.reference_file and result['final_text']:
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from evaluate import load_references, calculate_cer, calculate_mer
                refs = load_references(args.reference_file, language=args.language)
                filename = os.path.basename(audio_path)
                ref_text = refs.get(filename)
                if ref_text:
                    cer = calculate_cer(ref_text, result['final_text'], args.language)
                    mer = calculate_mer(ref_text, result['final_text'], args.language)
                    print(f"\nCER: {cer*100:.2f}%   MER: {mer*100:.2f}%")
                    if args.output_dir:
                        with open(os.path.join(args.output_dir, 'evaluation_result.json'), 'w', encoding='utf-8') as f:
                            json.dump({'file': filename, 'reference': ref_text,
                                       'generated': result['final_text'], 'cer': cer, 'mer': mer,
                                       'first_token_latency_ms': result['first_token_latency']*1000 if result.get('first_token_latency') else None,
                                       'last_token_latency_ms': result['last_token_latency']*1000 if result.get('last_token_latency') else None},
                                      f, indent=2, ensure_ascii=False)
                else:
                    logger.warning(f"No reference found for {filename}")
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                import traceback; traceback.print_exc()

    _cleanup_vllm()
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)
