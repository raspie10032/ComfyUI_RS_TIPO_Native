"""Compare Gemma inference with saved pre-change runtime and GGUF operations.

Uses real local ComfyUI models and LoRA, with no image/API calls. Run each
model in a fresh process with CUDA_VISIBLE_DEVICES set to the intended GPU.
"""
import argparse
from contextlib import contextmanager
import gc
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import time
from unittest.mock import patch


PEOPLE = [
    ('silver hair', 'blue eyes', 'red dress, white sandals', 'red book'),
    ('brown hair', 'green eyes', 'blue blouse, black skirt', 'blue mug'),
    ('blonde hair', 'purple eyes', 'green jacket, beige trousers', 'green umbrella'),
    ('black hair', 'brown eyes', 'yellow sundress, straw hat', 'yellow fan'),
    ('pink hair', 'grey eyes', 'white kimono, red obi', 'white parasol'),
    ('blue hair', 'amber eyes', 'orange hoodie, denim shorts', 'orange bag'),
    ('purple hair', 'red eyes', 'pink cardigan, pleated skirt', 'pink bouquet'),
    ('green hair', 'golden eyes', 'purple evening gown, pearl necklace', 'purple folder'),
    ('red hair', 'hazel eyes', 'black suit, striped tie', 'black camera'),
    ('white hair', 'teal eyes', 'brown coat, leather boots', 'brown basket'),
]


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--baseline-runtime', type=Path, required=True)
    parser.add_argument('--baseline-gguf-ops', type=Path)
    parser.add_argument('--baseline-dequant', type=Path,
        help='Saved dequant.py used by baseline GGUF operations only.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', default='scale-20pct-001')
    parser.add_argument('--counts', type=int, nargs='+', default=[2, 6, 10])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--profile-tokens', type=int, default=0)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--allow-token-differences', action='store_true',
        help='Record numerical/generation differences without failing exact equality; slot audits still fail.')
    parser.add_argument('--numerical-probe', action='store_true',
        help='Compare native LoRA layer outputs on prefill/decode against weight merging.')
    parser.add_argument('--lifecycle-checks', action='store_true',
        help='Check exclusions, cancellation and identical output after ComfyUI unload/reload.')
    parser.add_argument('--compare-ple-placement', action='store_true',
        help='Compare GPU-resident PLE against CPU row offload; reload/warm up outside timing.')
    args = parser.parse_args()
    if args.repeats < 1 or not args.counts or any(n < 1 or n > 10 for n in args.counts):
        parser.error('Use positive repeats and 1–10 participants.')
    if args.profile_only and args.profile_tokens < 1:
        parser.error('--profile-only requires positive --profile-tokens.')
    if args.baseline_dequant and not args.baseline_gguf_ops:
        parser.error('--baseline-dequant requires --baseline-gguf-ops.')
    sys.argv = [sys.argv[0]]  # ComfyUI must not consume benchmark arguments.
    sys.path.insert(0, str(args.comfy_root.resolve()))
    root = Path(__file__).resolve().parents[1]
    load_module('rs_tipo_native', root / '__init__.py', package=True)
    from rs_tipo_native import gemma_runtime as runtime
    from rs_tipo_native.gemma_prompt import instruction
    before = load_module('rs_tipo_native.gemma_runtime_before', args.baseline_runtime)
    from rs_tipo_native import tipo_gguf_ops as gguf_ops
    original_unpack, original_project = gguf_ops.unpack, gguf_ops.GGUFOps.Embedding.project
    original_linear = gguf_ops.GGUFOps.Linear.forward
    old_ops = None
    if args.baseline_gguf_ops:
        old_ops = load_module('rs_tipo_native.tipo_gguf_ops_before', args.baseline_gguf_ops)
        old_ops.PackedWeight = gguf_ops.PackedWeight
        if args.baseline_dequant:
            old_dequant = load_module('rs_tipo_native.gguf_dequant_before', args.baseline_dequant)
            old_ops.dequantize = old_dequant.dequantize
    import psutil
    import torch
    import comfy.model_management as mm

    torch.set_num_threads(4)
    device = mm.get_torch_device()
    process = psutil.Process()
    rows = []
    result = {'model': args.model, 'adapter': args.adapter, 'temperature': args.temperature,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device),
              'allow_token_differences': args.allow_token_differences,
              'compare_ple_placement': args.compare_ple_placement,
              'per_count_warmup_tokens': 4,
              'baseline_runtime': str(args.baseline_runtime.resolve()), 'rows': rows}
    if args.baseline_gguf_ops:
        result['baseline_gguf_ops'] = str(args.baseline_gguf_ops.resolve())
    if args.baseline_dequant:
        result['baseline_dequant'] = str(args.baseline_dequant.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')

    def memory():
        return {'allocated_mib': torch.cuda.memory_allocated(device) / 2**20,
                'reserved_mib': torch.cuda.memory_reserved(device) / 2**20,
                'rss_mib': process.memory_info().rss / 2**20}

    print('LOAD', args.model, flush=True)
    start = time.perf_counter()
    clip, info = runtime.load_gemma(args.model, args.adapter)
    result.update(load_seconds=time.perf_counter() - start, model_info=info)

    def placement():
        return {'patcher_model_mib': clip.patcher.model_size() / 2**20,
                'patcher_loaded_mib': clip.patcher.loaded_size() / 2**20,
                'tables': [{'name': name, 'device': str(m.weight.device),
                            'packed_mib': m.weight.nbytes / 2**20,
                            'cpu_offload': getattr(m, 'cpu_offload', False)}
                    for name, m in clip.patcher.model.named_modules()
                    if name.endswith('.embed_tokens_per_layer')]}
    def short_generate(request, limit):
        # Warmup/profiling intentionally stop mid-response. Bypass the node's
        # complete-output check, while using its exact native sampler settings.
        tokens = clip.tokenize(request, min_length=1, thinking=False, skip_template=False)
        transformer = clip.cond_stage_model.gemma4.transformer
        with runtime.token_constraints(transformer, clip.decode, (),
                mm.throw_exception_if_processing_interrupted, lambda n: None):
            with torch.inference_mode():
                return clip.generate(tokens, do_sample=False, max_length=limit,
                    temperature=.01, top_k=64, top_p=.95, min_p=.05,
                    repetition_penalty=1.05, seed=42)

    tokens = clip.tokenize('Write one English sentence about a garden.', min_length=1,
                           thinking=False, skip_template=False)
    with torch.inference_mode():
        clip.generate(tokens, do_sample=False, max_length=24, seed=42)
    torch.cuda.synchronize(device)
    result['after_warmup'] = memory()
    save()

    warmup_request = 'Describe two adult women in a garden in one sentence.'
    warmup_ids = list(short_generate(warmup_request, 4))
    if args.numerical_probe:
        if old_ops is None or not hasattr(gguf_ops, 'linear_lora_patches'):
            raise ValueError('--numerical-probe requires baseline GGUF ops and the LoRA bypass implementation')
        probes = []

        def probe(layer, x):
            output = original_linear(layer, x)
            if gguf_ops.linear_lora_patches(layer.weight) is not None and not layer.weight_function:
                reference = old_ops.GGUFOps.Linear.forward(layer, x)
                difference = output.float() - reference.float()
                ref_rms = reference.float().square().mean().sqrt().clamp_min(1e-12)
                probes.append({'shape': list(x.shape), 'weight_shape': list(layer.weight.shape),
                    'finite': bool(torch.isfinite(output).all()),
                    'max_abs': float(difference.abs().max()),
                    'relative_rms': float(difference.square().mean().sqrt() / ref_rms)})
            return output

        with patch.object(gguf_ops.GGUFOps.Linear, 'forward', probe):
            short_generate(warmup_request, 4)
        result['numerical_probe'] = {
            'calls': len(probes), 'all_finite': bool(probes) and all(row['finite'] for row in probes),
            'max_abs': max((row['max_abs'] for row in probes), default=None),
            'max_relative_rms': max((row['relative_rms'] for row in probes), default=None),
            'layers': probes}
        save()

    for count in (() if args.profile_only else args.counts):
        people = PEOPLE[:count]
        common = (f'성인 여성 {count}명이 넓은 정원에서 단체 기념사진을 찍고 있다. '
                  '각자 자기 소품을 들고 카메라를 바라본다. 전신 단체 사진, 낮, 맑은 날씨.')
        characters = tuple(f'외형: ({hair}:1.15), {eyes}. 의상: {outfit}. '
                           f'holding a {prop}, looking at camera.'
                           for hair, eyes, outfit, prop in people)
        request = instruction(common, characters, output_format='natural_language',
            length='long', max_added_tags=0, expand_common=True,
            metadata={'rating': 'safe'})
        short_generate(request, 4)
        torch.cuda.synchronize(device)
        for repeat in range(args.repeats):
            # Alternate order to reduce warmup/clock/allocator bias.
            modes = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
            for mode in modes:
                print('START', count, repeat, mode, flush=True)
                if args.compare_ple_placement:
                    mm.unload_all_models()
                    mm.soft_empty_cache()
                    gguf_ops.set_ple_offload(clip.patcher, enabled=mode == 'after')
                    short_generate(request, 4)
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                row = {'count': count, 'repeat': repeat, 'mode': mode,
                       'memory_before': memory(), 'calls': []}
                if args.compare_ple_placement:
                    row['placement_before'] = placement()
                stop = threading.Event()
                peak_rss = [process.memory_info().rss]

                def monitor():
                    while not stop.wait(.02):
                        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

                thread = threading.Thread(target=monitor, daemon=True)
                original_clip_generate = clip.generate
                constraint = before.token_constraints if mode == 'before' else runtime.token_constraints
                call = {}

                def observe_generate(*a, **kw):
                    call.clear()
                    call['start'] = time.perf_counter()
                    ids = original_clip_generate(*a, **kw)
                    torch.cuda.synchronize(device)
                    row['calls'].append({'token_ids': list(ids),
                        'generation_seconds': time.perf_counter() - call['start'],
                        'ttft_seconds': call.get('ttft')})
                    return ids

                @contextmanager
                def observe_constraints(transformer, decode, excluded, interrupt, progress):
                    def observe_progress(n):
                        if n == 1:
                            torch.cuda.synchronize(device)
                            call['ttft'] = time.perf_counter() - call['start']
                        progress(n)
                    with constraint(transformer, decode, excluded, interrupt, observe_progress) as report:
                        yield report

                thread.start()
                start = time.perf_counter()
                try:
                    with patch.object(runtime, 'token_constraints', observe_constraints), \
                         patch.object(gguf_ops, 'unpack', old_ops.unpack if mode == 'before' and old_ops else original_unpack), \
                         patch.object(gguf_ops.GGUFOps.Embedding, 'project', old_ops.GGUFOps.Embedding.project if mode == 'before' and old_ops else original_project), \
                         patch.object(gguf_ops.GGUFOps.Linear, 'forward', old_ops.GGUFOps.Linear.forward if mode == 'before' and old_ops else original_linear), \
                         patch.object(clip, 'generate', observe_generate):
                        scene, slots, report = runtime.run_scene(common, characters,
                            model=args.model, adapter=args.adapter, output_format='natural_language',
                            length='long', max_added_tags=0, expand_common=True,
                            metadata={'rating': 'safe'}, max_tokens=3072,
                            temperature=args.temperature, seed=42)
                    missing, foreign = [], []
                    for i, (slot, person) in enumerate(zip(slots, people), 1):
                        hair, eyes, outfit, prop = person
                        facts = [f'({hair}:1.15)', eyes, *outfit.split(', ')]
                        facts += [w for w in prop.split() if w not in ('bouquet', 'mug')]
                        missing += [(i, fact) for fact in facts if fact.casefold() not in slot.casefold()]
                        foreign += [(i, other[0]) for other in people if other[0] != hair
                            and re.search(r'(?<!\w)' + re.escape(other[0]) + r'(?!\w)', slot, re.I)]
                    row.update(status='success', common=scene, characters=slots, report=report,
                               audit={'expected_slots': count, 'actual_slots': len(slots),
                                      'missing_literals': missing, 'foreign_hair': foreign})
                except Exception as exc:
                    row.update(status='error', error=str(exc), failed_calls=getattr(exc, 'calls', []))
                finally:
                    torch.cuda.synchronize(device)
                    row['wall_seconds'] = time.perf_counter() - start
                    stop.set()
                    thread.join()
                row.update(memory_after=memory(), peak_rss_mib=peak_rss[0] / 2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
                reference = next((r for r in rows if r['count'] == count and r['mode'] == 'before'), None)
                if reference:
                    row['matches_first_baseline_tokens'] = ([c['token_ids'] for c in row['calls']] ==
                        [c['token_ids'] for c in reference['calls']])
                    row['matches_first_baseline_output'] = (row.get('common'), row.get('characters')) == (
                        reference.get('common'), reference.get('characters'))
                rows.append(row)
                save()
                print('RESULT', count, repeat, mode, row['status'], round(row['wall_seconds'], 3),
                      row.get('matches_first_baseline_tokens'), flush=True)

    if args.profile_tokens:
        import comfy.lora
        functions = [(gguf_ops, 'dequantize', 'GGUF dequantize'),
                     (gguf_ops, 'unpack', 'GGUF unpack'),
                     (comfy.lora, 'calculate_weight', 'LoRA calculate_weight')]
        if hasattr(gguf_ops, 'add_lora_output'):
            functions.append((gguf_ops, 'add_lora_output', 'LoRA output bypass'))
        from contextlib import ExitStack
        def annotate(fn, name):
            def observed(*a, **kw):
                with torch.profiler.record_function(name):
                    return fn(*a, **kw)
            return observed
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], profile_memory=True) as prof:
            with ExitStack() as stack:
                for module, name, label in functions:
                    stack.enter_context(patch.object(module, name, annotate(getattr(module, name), label)))
                short_generate('Describe two adult women in a garden in one sentence.',
                               args.profile_tokens)
            torch.cuda.synchronize(device)
        events = prof.key_averages()
        result['profile'] = [{'name': e.key, 'device_type': str(e.device_type), 'count': e.count,
            'cpu_total_us': e.cpu_time_total, 'cpu_self_us': e.self_cpu_time_total,
            'device_total_us': e.device_time_total, 'device_self_us': e.self_device_time_total,
            'device_self_memory_bytes': e.self_device_memory_usage}
            for e in events if e.key in {label for _, _, label in functions} or
            e.key in ('aten::index', 'aten::arange', 'aten::mm', 'aten::_to_copy', 'aten::copy_')]
        args.output.with_suffix('.profile.txt').write_text(events.table(
            sort_by='self_device_time_total', row_limit=35))
        save()

    result['before_unload'] = memory()
    result['placement_before_unload'] = placement()
    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()
    torch.cuda.synchronize(device)
    result['after_comfy_unload'] = memory()
    result['placement_after_unload'] = placement()
    result['comfy_loaded_models_after_unload'] = len(mm.current_loaded_models)
    if args.lifecycle_checks:
        reloaded_ids = list(short_generate(warmup_request, 4))
        result['placement_after_reload'] = placement()
        transformer = clip.cond_stage_model.gemma4.transformer
        original_local_sampler = transformer.__dict__.get('sample_token')
        had_local_sampler = 'sample_token' in transformer.__dict__
        cancelled = False

        def cancel_after_two(count):
            if count == 2:
                mm.interrupt_current_processing(True)

        try:
            tokens = clip.tokenize('Write a short description of a girl in a garden.',
                                   min_length=1, thinking=False, skip_template=False)
            with runtime.token_constraints(transformer, clip.decode, (),
                    mm.throw_exception_if_processing_interrupted, cancel_after_two):
                with torch.inference_mode():
                    clip.generate(tokens, do_sample=False, max_length=8, seed=42)
        except mm.InterruptProcessingException:
            cancelled = True
        finally:
            mm.interrupt_current_processing(False)
        sampler_restored = (('sample_token' in transformer.__dict__) == had_local_sampler
            and transformer.__dict__.get('sample_token') is original_local_sampler)
        recovered_ids = list(short_generate(warmup_request, 4))
        raw, excluded_info = runtime.generate(clip,
            'Reply with exactly this sentence: The girl is wearing a red dress.',
            max_tokens=128, temperature=0., seed=42, excluded=('red',))
        exclusion_ok = (excluded_info['exclusion']['blocked_candidates'] > 0
                        and re.search(r'(?<!\w)red(?!\w)', raw, re.IGNORECASE) is None)
        result['lifecycle'] = {'reload_tokens_match_warmup': reloaded_ids == warmup_ids,
            'cancellation_raised': cancelled, 'sampler_restored': sampler_restored,
            'recovery_tokens_match_warmup': recovered_ids == warmup_ids,
            'exclusion_ok': exclusion_ok, 'exclusion_info': excluded_info, 'exclusion_raw': raw}
        mm.unload_all_models(); mm.soft_empty_cache(); gc.collect(); torch.cuda.synchronize(device)
        result['after_lifecycle_unload'] = memory()
        result['comfy_loaded_models_after_lifecycle_unload'] = len(mm.current_loaded_models)
    save()
    print('UNLOAD', result['after_comfy_unload'], flush=True)
    if any(r['status'] != 'success'
           or (not args.allow_token_differences and (r.get('matches_first_baseline_tokens') is False
               or r.get('matches_first_baseline_output') is False))
           or r.get('audit', {}).get('missing_literals')
           or r.get('audit', {}).get('foreign_hair')
           or r.get('audit', {}).get('actual_slots') != r['count'] for r in rows):
        raise SystemExit('Benchmark failed; inspect JSON for output/slot differences.')
    if (result.get('numerical_probe', {}).get('all_finite') is False
            or (args.lifecycle_checks and not all(result['lifecycle'][key] for key in
                ('reload_tokens_match_warmup', 'cancellation_raised', 'sampler_restored',
                 'recovery_tokens_match_warmup', 'exclusion_ok')))):
        raise SystemExit('Numerical/lifecycle check failed; inspect JSON.')


if __name__ == '__main__':
    main()
