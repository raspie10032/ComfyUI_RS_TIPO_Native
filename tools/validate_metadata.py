"""Check shipped metadata on an explicitly selected idle ComfyUI test server.

Usage: python tools/validate_metadata.py PORT
Records prompt IDs before observing them; rerunning resumes an outstanding job.
Does not generate images or call NovelAI.
"""
import json
from pathlib import Path
import sys
import time
import urllib.request

BASE = 'http://127.0.0.1:' + str(int(sys.argv[1]))
DEST = Path(__file__).resolve().parents[1] / 'validation' / f'metadata-{sys.argv[1]}.json'
PENDING = DEST.with_suffix('.pending.json')


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=10))


def main():
    info = get('/object_info/RSTIPOPrompt')['RSTIPOPrompt']
    assert all(key in info['input']['optional'] for key in ('rating', 'quality', 'quality_era'))
    previous = json.loads(DEST.read_text()) if DEST.exists() else []
    pending = json.loads(PENDING.read_text()) if PENDING.exists() else None
    queue = get('/queue')
    jobs = [row[1] for rows in queue.values() for row in rows]
    assert all(pending and job == pending['prompt_id'] for job in jobs), 'Unrelated jobs; leave them running.'
    defaults = {}
    for name, schema in info['input']['required'].items():
        if len(schema) > 1 and 'default' in schema[1]:
            defaults[name] = schema[1]['default']
        elif isinstance(schema[0], list):
            defaults[name] = schema[0][0]
    common = '2girls, cafe, sitting, indoors, wooden table'
    characters = ('(blonde hair:1.2), blue eyes, pink cardigan, on left, offering cup, extending arms',
                  'black hair, brown eyes, yellow sweater, on right, receiving cup, reaching out')
    cases = [
        ('omitted', {}, {}),
        ('explicit_unspecified', {'rating': 'unspecified', 'quality': 'unspecified', 'quality_era': 'unspecified'}, {}),
        ('rating', {'rating': 'safe'}, {'rating': 'safe'}),
        ('quality', {'quality': 'good quality'}, {'quality': 'good quality'}),
        ('combined', {'rating': 'safe', 'quality': 'masterpiece', 'quality_era': 'newest'},
         {'quality': 'masterpiece, newest', 'rating': 'safe'}),
    ]
    for label, settings, expected in cases:
        if any(row['case'] == label for row in previous):
            continue
        if pending:
            assert pending['case'] == label
            pid = pending['prompt_id']
        else:
            inputs = {**defaults, 'character_count': 2, 'common_prompt': common,
                      'character1': characters[0], 'character2': characters[1],
                      'character10': 'STALE INPUT MUST NOT LEAK', 'expand_common': True,
                      'max_new_tokens': 96, 'seed': 8910, 'temperature': .5,
                      'excluded_tags': 'watermark, signature, text', **settings}
            request = urllib.request.Request(BASE + '/prompt', json.dumps({
                'prompt': {'1': {'class_type': 'RSTIPOPrompt', 'inputs': inputs}},
                'client_id': 'rs-tipo-metadata-validation'}).encode(), headers={'Content-Type': 'application/json'})
            pid = json.load(urllib.request.urlopen(request, timeout=15))['prompt_id']
            PENDING.parent.mkdir(parents=True, exist_ok=True)
            PENDING.write_text(json.dumps({'case': label, 'prompt_id': pid}))
        print(label, 'observing', pid, flush=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            history = get('/history/' + pid)
            if pid in history:
                record = history[pid]
                assert record['status']['status_str'] == 'success', record['status']
                report = json.loads(record['outputs']['1']['tipo_report'][0])
                assert report['metadata'] == expected
                assert len(report['slots']) == 3
                assert 'STALE INPUT' not in json.dumps(report)
                for row, source in zip(report['slots'], (common, *characters)):
                    assert row['metadata'] == expected
                    assert row['output'].startswith(source)
                    assert '\n'.join(f'{k}: {v}' for k, v in expected.items()) in row['input']
                    if not expected:
                        assert row['input'].startswith('target:')
                    assert 'rating:' not in row['output'] and 'quality:' not in row['output']
                    assert row['exclusion']['tags'] == ['signature', 'text', 'watermark']
                previous.append({'case': label, 'prompt_id': pid, 'report': report})
                DEST.write_text(json.dumps(previous, ensure_ascii=False, indent=2))
                PENDING.unlink()
                pending = None
                print(label, 'PASS', report['seconds'], flush=True)
                break
            time.sleep(.5)
        else:
            raise TimeoutError(f'Continue observing {pid}; pending handle is saved. Do not resubmit.')
    baseline = previous[0]['report']['slots']
    explicit = previous[1]['report']['slots']
    for before, after in zip(baseline, explicit):
        assert before['input'] == after['input']
        assert before['token_ids'] == after['token_ids']
        assert before['output'] == after['output']
    print('Omitted vs explicit unspecified: identical inputs, token IDs and outputs PASS', flush=True)


if __name__ == '__main__':
    main()
