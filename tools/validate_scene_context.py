"""Exercise the shipped node with only common/character inputs on a live server."""
import json
from pathlib import Path
import sys
import time
import urllib.request

BASE = 'http://127.0.0.1:' + (sys.argv[1] if len(sys.argv) > 1 else '8188')
DEST = Path(__file__).resolve().parents[1] / 'validation' / ('automatic-context-' + BASE.rsplit(':', 1)[1] + '.json')

def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=10))

info = get('/object_info/RSTIPOPrompt')['RSTIPOPrompt']
assert 'interaction_prompt' not in info['input']['optional']
assert not any(get('/queue').values()), 'Other jobs are running; do not interrupt.'
records = []
cases = [
    ('give_receive', '2girls, cafe, sitting, indoors',
     ('(blonde hair:1.2), blue eyes, pink cardigan, on left, offering cup, extending arms',
      'black hair, brown eyes, yellow sweater, on right, receiving cup, reaching out')),
    ('hug', '2girls, garden, autumn leaves',
     ('blonde hair, pink cardigan, on left, hugging, arms around other',
      'black hair, yellow sweater, on right, leaning on other, head on shoulder')),
    ('dance', '2girls, ballroom, evening',
     ('blonde hair, pink dress, on left, leading dance, hand on other waist',
      'black hair, yellow dress, on right, following dance, hand on other shoulder')),
]
if BASE.endswith(':8188'):
    cases += [('give_receive_new_appearance', cases[0][1], (cases[0][2][0], cases[0][2][1].replace('black hair', 'red hair').replace('yellow sweater', 'green jacket'))),
              ('scene_only', 'garden, sunlight', ()),
              ('three_people', '3girls, garden, playing together', ('blonde hair, holding hands', 'black hair, holding hands', 'red hair, waving'))]
else:
    cases = cases[:1]

for name, common, people in cases:
    inputs = {}
    for key, schema in info['input']['required'].items():
        if len(schema) > 1 and 'default' in schema[1]:
            inputs[key] = schema[1]['default']
        elif isinstance(schema[0], list):
            inputs[key] = schema[0][0]
    inputs.update(character_count=len(people), common_prompt=common, expand_common=True,
                  max_new_tokens=64, temperature=0., seed=8910, character10='STALE PRIVATE INPUT')
    inputs.update({f'character{i}': text for i, text in enumerate(people, 1)})
    graph = {'1': {'class_type': 'RSTIPOPrompt', 'inputs': inputs}}
    req = urllib.request.Request(BASE+'/prompt', json.dumps({'prompt': graph}).encode(), headers={'Content-Type': 'application/json'})
    pid = json.load(urllib.request.urlopen(req, timeout=10))['prompt_id']
    DEST.with_suffix('.pending.json').write_text(json.dumps({'prompt_id': pid, 'case': name}))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        history = get('/history/' + pid)
        if pid in history:
            record = history[pid]
            assert record['status']['status_str'] == 'success', record['status']
            report = json.loads(record['outputs']['1']['tipo_report'][0])
            assert report['context_source'] == 'common_and_character_inputs'
            assert len(report['participant_roles']) == len(people)
            assert 'STALE PRIVATE INPUT' not in json.dumps(report)
            for row in report['slots']:
                if isinstance(row['slot'], int):
                    assert row['original'] == people[row['slot']-1]
                    assert row['output'].startswith(row['original'])
                    assert 'Person' in row['interaction']
            if name == 'give_receive_new_appearance':
                previous = next(r['report'] for r in records if r['case'] == 'give_receive')
                assert report['shared_context'] == previous['shared_context']
                before = next(r for r in previous['slots'] if r['slot'] == 1)
                after = next(r for r in report['slots'] if r['slot'] == 1)
                assert before['input'] == after['input']
                assert before['token_ids'] == after['token_ids']
                assert before['output'] == after['output']
            records.append({'case': name, 'prompt_id': pid, 'report': report})
            DEST.write_text(json.dumps(records, ensure_ascii=False, indent=2))
            DEST.with_suffix('.pending.json').unlink(missing_ok=True)
            print(name, 'PASS', report['seconds'], flush=True)
            break
        time.sleep(.25)
    else:
        raise TimeoutError(f'Observe existing prompt {pid}, do not resubmit.')
