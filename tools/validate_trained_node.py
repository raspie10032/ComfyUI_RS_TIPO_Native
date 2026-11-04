"""Exercise the installed public node through ComfyUI's real execution queue.

No image generation, model stubs, or alternate inference implementation.
Keep raw reports for review: these checks prove execution, not semantic quality.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error


def request(base, path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base + path, data=body,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'ComfyUI HTTP {exc.code}: {exc.read().decode()}') from None


CASES = [
    dict(name='ko-handover', common_prompt='오후 카페에서 두 여성이 꽃다발을 주고받는다.',
         characters=['왼쪽의 금발 파란 눈 여성은 흰 셔츠를 입고 꽃다발을 건넨다.',
                     '오른쪽의 검은 머리 갈색 눈 여성은 파란 카디건을 입고 꽃다발을 받는다.']),
    dict(name='en-weighted', common_prompt='Two women tie an apron in a sunny kitchen.',
         characters=['(blonde hair:1.2), blue eyes, white shirt, on left, standing still while her apron is tied',
                     'black hair, brown eyes, blue cardigan, behind the other woman, tying her apron strings']),
    dict(name='ko-three-roles', common_prompt='세 여성이 작업실에서 연을 함께 만든다.',
         characters=['금발 여성은 초록 셔츠를 입고 연의 틀을 붙잡는다.',
                     '갈색 머리 여성은 파란 셔츠를 입고 연의 틀에 종이를 붙인다.',
                     '검은 머리 여성은 빨간 셔츠를 입고 옆에서 가위를 건넨다.']),
    dict(name='en-short', common_prompt='Two women share an umbrella in the rain.',
         characters=['silver hair, holding umbrella', 'red hair, standing under the umbrella']),
    dict(name='ko-common-only', common_prompt='이른 아침 안개가 낀 조용한 숲길, 수채화.', characters=[]),
    dict(name='en-count-shrink', common_prompt='One woman reads in a quiet library.',
         characters=['brown hair, green eyes, yellow sweater, reading an open book']),
]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--base', default='http://127.0.0.1:8196')
    cli.add_argument('--adapter', default='local:scale-20pct-001')
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--case', choices=[case['name'] for case in CASES])
    cli.add_argument('--format', choices=['natural_language', 'tags'])
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schema = request(args.base, '/object_info/RSTIPOPrompt')['RSTIPOPrompt']
    defaults = {}
    for section in ('required', 'optional'):
        for name, spec in schema['input'].get(section, {}).items():
            if len(spec) > 1 and 'default' in spec[1]:
                defaults[name] = spec[1]['default']
            elif isinstance(spec[0], list):
                defaults[name] = spec[0][0]
    results = []
    for case in CASES:
        if args.case and case['name'] != args.case:
            continue
        for fmt in ([args.format] if args.format else ('natural_language', 'tags')):
            inputs = dict(defaults, character_count=len(case['characters']),
                          common_prompt=case['common_prompt'], model_family='gemma_tipo',
                          language_model='heretic_ara_q4', tipo_lora=args.adapter,
                          expand_common=True, output_format=fmt, seed=42, temperature=0,
                          generation_max_tokens=2048, excluded_tags='watermark, signature')
            inputs.update({f'character{i}': text for i, text in enumerate(case['characters'], 1)})
            if case['name'] == 'en-count-shrink':
                inputs.update(character2='DO NOT OUTPUT STALE SECOND CHARACTER',
                              character3='DO NOT OUTPUT STALE THIRD CHARACTER')
            graph = {'1': {'class_type': 'RSTIPOPrompt', 'inputs': inputs}}
            # Execute actual downstream STRING consumers and BuildPipe finalizer.
            for output in range(1, 4):
                graph[str(output + 1)] = {'class_type': 'PreviewAny',
                                         'inputs': {'source': ['1', output]}}
            if case['characters']:
                graph['5'] = {'class_type': 'RSMultiCharacterFinalize',
                              'inputs': {'building_pipe': ['1', 0]}}
                graph['6'] = {'class_type': 'PreviewAny', 'inputs': {'source': ['5', 0]}}
            started = time.monotonic()
            row = dict(case=case['name'], format=fmt, inputs=inputs)
            try:
                handle = request(args.base, '/prompt', {'prompt': graph})
                row['prompt_id'] = handle['prompt_id']
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    history = request(args.base, '/history/' + handle['prompt_id'])
                    record = history.get(handle['prompt_id'])
                    if record:
                        break
                    time.sleep(.5)
                else:
                    raise TimeoutError('Prompt still pending/running after 300 seconds; not resubmitted.')
                if record['status']['status_str'] != 'success':
                    raise RuntimeError(json.dumps(record['status'], ensure_ascii=False))
                report = json.loads(record['outputs']['1']['tipo_report'][0])
                row['report'] = report
                assert report['adapter'] == args.adapter and report['lora_targets'] == 190
                assert len(report['slots']) == len(case['characters']) + 1
                assert report['building_pipe_available'] == bool(case['characters'])
                for slot in report['slots']:
                    assert 'STALE' not in slot['output']
                    assert 'watermark' not in slot['output'].lower()
                    assert 'signature' not in slot['output'].lower()
                if case['name'] == 'en-weighted':
                    assert '(blonde hair:1.2)' in report['slots'][1]['output']
                # Fixed output indexes must send empty strings above active count.
                if len(case['characters']) < 2:
                    assert record['outputs']['4']['text'] == ['']
                row.update(error=None, downstream=record['outputs'])
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            row['seconds'] = round(time.monotonic() - started, 3)
            results.append(row)
            args.output.write_text(json.dumps({'scope': __doc__, 'rows': results},
                                             ensure_ascii=False, indent=2))
            print(json.dumps({k: row[k] for k in ('case', 'format', 'error', 'seconds')},
                             ensure_ascii=False), flush=True)
    if any(row['error'] for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
