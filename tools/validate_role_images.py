"""Small paired Anima image comparison on an idle, explicitly selected server.

Uses original / previous guarded / focused prompts from validate_role_additions.
Equivalent prompts intentionally reuse the same seed and may hit Comfy's cache.
The output records image pixels, complete API graphs, timings and prompt IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompts', type=Path)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[891024073])
    parser.add_argument('--modes', nargs='+', choices=('original', 'previous', 'focused'),
                        default=('original', 'previous', 'focused'))
    args = parser.parse_args()
    from PIL import Image
    import io

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(args.url + path, data,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    queue = request('/queue')
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('Selected server is busy; do not submit another batch.')
    info = request('/object_info')
    stats = request('/system_stats')
    if not any('5060' in device['name'] for device in stats['devices']):
        raise RuntimeError('This local study is reserved for the spare 5060.')

    def defaults(name):
        result = {}
        for key, schema in info[name]['input'].get('required', {}).items():
            options = schema[1] if len(schema) > 1 else {}
            if 'default' in options:
                result[key] = options['default']
            elif isinstance(schema[0], list):
                result[key] = schema[0][0]
        return result

    args.output.mkdir(parents=True, exist_ok=True)
    record_path = args.output / 'results.json'
    records = json.loads(record_path.read_text()) if record_path.exists() else []
    done = {(r['case'], r['mode'], r.get('seed', 891024073)) for r in records}
    prompts = json.loads(args.prompts.read_text())
    jobs = [(case, seed) for case in prompts['cases'] for seed in args.seeds]
    for case, seed in jobs:
        for mode in args.modes:
            if (case['name'], mode, seed) in done:
                continue
            people = defaults('RSMultiCharacterPrompts')
            count = len(case[mode])
            people['character_count'] = count
            for i, text in enumerate(case[mode], 1):
                people.update({f'character{i}': text, f'character{i}_enable': True,
                               f'character{i}_x': .5 if count == 1 else .28 + .44 * (i-1)/(count-1),
                               f'character{i}_y': .5})
            sampler = defaults('RSMultiCharacterSampler')
            sampler.update(model=['5', 0], clip=['5', 1], vae=['3', 0], character_pipe=['4', 0],
                prompt='masterpiece, best quality, adult women, ' + case['common'],
                negative_prompt='low quality, worst quality, bad anatomy, text, watermark, signature',
                width=1152, height=896, seed=seed, steps=30, cfg=4.,
                sampler_name='euler_ancestral', scheduler='sgm_uniform', autosave=False)
            graph = {
                '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'anima-base-v1.0.safetensors', 'weight_dtype': 'default'}},
                '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen_3_06b_base.safetensors', 'type': 'stable_diffusion'}},
                '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_vae.safetensors'}},
                '4': {'class_type': 'RSMultiCharacterPrompts', 'inputs': people},
                '5': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['2', 0],
                    'lora_name': 'anima/RSCV5-intermediate/RSCV5_Anima-step00015000-comfy.safetensors',
                    'strength_model': .9, 'strength_clip': 1.}},
                '6': {'class_type': 'RSMultiCharacterSampler', 'inputs': sampler},
                '7': {'class_type': 'SaveImage', 'inputs': {'images': ['6', 0],
                    'filename_prefix': f"role-study/{case['name']}-{seed}-{mode}"}},
            }
            graph_path = args.output / f"{case['name']}-{seed}-{mode}-api.json"
            graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2))
            started = time.monotonic()
            accepted = request('/prompt', {'prompt': graph, 'client_id': 'rs-tipo-role-images'})
            pid = accepted['prompt_id']
            # Save the live handle immediately. A stopped observer must inspect
            # this ID before any resubmission.
            (args.output / 'live.json').write_text(json.dumps({'prompt_id': pid,
                'url': args.url, 'case': case['name'], 'mode': mode, 'seed': seed}))
            print(f"submitted {case['name']}/{mode}: {pid}", flush=True)
            while True:
                history = request('/history/' + pid).get(pid)
                if history is not None:
                    break
                time.sleep(1)
            if history['status']['status_str'] != 'success':
                (args.output / 'failure.json').write_text(json.dumps(history, indent=2))
                raise RuntimeError(f'Generation failed: {pid}; inspect failure.json')
            saved = history['outputs']['7']['images'][0]
            from urllib.parse import urlencode
            with urllib.request.urlopen(args.url + '/view?' + urlencode(saved), timeout=30) as response:
                raw = response.read()
            target = args.output / f"{case['name']}-{seed}-{mode}.png"
            target.write_bytes(raw)
            with Image.open(io.BytesIO(raw)) as image:
                pixel_hash = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
            row = {'case': case['name'], 'mode': mode, 'prompt_id': pid,
                   'seed': seed,
                   'seconds_queue_to_file': round(time.monotonic() - started, 3),
                   'file': str(target), 'pixel_sha256': pixel_hash, 'history': history}
            records.append(row)
            record_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
            (args.output / 'live.json').unlink()
            print(f"done {case['name']}/{mode}: {row['seconds_queue_to_file']}s", flush=True)


if __name__ == '__main__':
    main()
