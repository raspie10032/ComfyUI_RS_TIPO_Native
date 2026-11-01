"""Paired, deterministic source-protection checks on an idle resident server.

This measures a declared lexical fault set and unchanged prompt/pipe transport;
it is not an image-quality or general semantic-understanding benchmark.
"""
import copy
import json
from pathlib import Path
import re
import time
import urllib.request

BASE = 'http://127.0.0.1:8188'
DEST = Path(__file__).resolve().parents[1] / 'validation' / 'grounding-live.json'
CASES = [
 ('cup', '2girls, university cafe, afternoon, wooden table',
  'Two women sit across a cafe table. The woman on the left offers a cup with both hands. The woman on the right reaches out to accept that same cup. They look at each other.',
  '(blonde hair:1.2), blue eyes, pink cardigan, on left, offering cup, extending arms',
  'black hair, brown eyes, yellow sweater, on right, receiving cup, reaching out'),
 ('hug', '2girls, garden, autumn leaves',
  'Two women hug each other. The woman on the left wraps her arms around her friend. The woman on the right leans into her embrace with her head resting on her shoulder.',
  'blonde hair, blue eyes, pink cardigan, on left, hugging, arms around other',
  'black hair, brown eyes, yellow sweater, on right, leaning on other, head on shoulder'),
 ('dance', '2girls, ballroom, evening, warm lighting',
  'Two women dance together as partners. The left woman leads, holding her partner by the waist. The right woman follows and rests her hand on the leading woman’s shoulder.',
  'blonde hair, blue eyes, pink dress, on left, leading dance, hand on other waist',
  'black hair, brown eyes, yellow dress, on right, following dance, hand on other shoulder'),
]


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=10))


def main():
    if any(get('/queue').values()):
        raise RuntimeError('Resident queue is busy; observe it before submitting.')
    info = get('/object_info')
    assert 'protect_source' in info['RSTIPOPrompt']['input']['optional']
    def node(kind, **values):
        inputs = {}
        for name, row in info[kind]['input'].get('required', {}).items():
            if len(row) > 1 and 'default' in row[1]:
                inputs[name] = row[1]['default']
            elif isinstance(row[0], list):
                inputs[name] = row[0][0]
        return {'class_type':kind, 'inputs':{**inputs, **values}}
    records = []
    for label, common, interaction, first, second in CASES:
        for protected in (False, True):
            graph = {
                '1':node('RSTIPOPrompt', common_prompt=common, character_count=2,
                    character1=first, character2=second, character3='STALE PRIVATE INPUT',
                    protect_source=protected,
                    expand_common=True, max_new_tokens=96, temperature=0., seed=8910),
                '2':node('RSMultiCharacterFinalize', building_pipe=['1',0]),
                '3':node('PreviewAny', source=['2',0]),
                '4':node('PreviewAny', source=['1',1]),
                '5':node('PreviewAny', source=['1',2]),
                '6':node('PreviewAny', source=['1',3]),
                '7':node('PreviewAny', source=['1',4]),
            }
            request = urllib.request.Request(BASE+'/prompt',
                json.dumps({'prompt':graph, 'client_id':'rs-tipo-grounding'}).encode(),
                headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=10) as response:
                pid = json.load(response)['prompt_id']
            # Persist the live handle before observing it. A timeout never
            # authorizes resubmitting the same job.
            DEST.with_suffix('.pending.json').write_text(json.dumps({'prompt_id':pid}))
            deadline = time.monotonic()+180
            while time.monotonic() < deadline:
                history = get('/history/'+pid)
                if pid in history:
                    record = history[pid]
                    assert record['status']['status_str']=='success', record['status']
                    out = record['outputs']
                    report = json.loads(out['1']['tipo_report'][0])
                    assert out['5']['text'][0].startswith(first)
                    assert out['6']['text'][0].startswith(second)
                    assert out['7']['text'][0]==''
                    assert 'STALE PRIVATE INPUT' not in out['3']['text'][0]
                    assert interaction in out['4']['text'][0]
                    for number, source in ((1,first), (2,second)):
                        row = next(r for r in report['slots'] if r['slot']==number)
                        assert row['output']==out[str(number+4)]['text'][0]
                        assert row['output'] in out['3']['text'][0]
                        assert source in row['output']
                    if protected:
                        prior = records[-1]['report']
                        # Same input and inference; protection is observable as
                        # a post-decode policy, not a claim of smarter logits.
                        assert [r['text'] for r in prior['slots']]==[r['text'] for r in report['slots']]
                        for row in report['slots']:
                            assert not re.search(r'\b(?:hair|eyes|3d|realistic|solo|upper body|short sleeves|bare shoulders|asian)\b', row['added_tags'])
                            if label=='dance':
                                assert 'hand on own hip' not in row['added_tags']
                            if label=='cup':
                                assert 'looking at viewer' not in row['added_tags']
                    records.append({'case':label, 'protected':protected, 'prompt_id':pid, 'report':report})
                    DEST.write_text(json.dumps(records,ensure_ascii=False,indent=2))
                    DEST.with_suffix('.pending.json').unlink()
                    print(label, protected, 'PASS',
                          [(r['slot'],r['added_tags'],len(r['rejected_tags'])) for r in report['slots']], flush=True)
                    break
                time.sleep(.5)
            else:
                raise TimeoutError(f'Continue observing {pid}; do not resubmit.')
    summary = {}
    for protected in (False,True):
        rows = [r for case in records if case['protected']==protected for r in case['report']['slots']]
        summary[str(protected)] = dict(requests=3, slots=len(rows),
            added_tags=sum(len([t for t in row['added_tags'].split(',') if t.strip()]) for row in rows),
            rejected_tags=sum(len(row['rejected_tags']) for row in rows),
            seconds=sum(row['seconds'] for row in rows))
    DEST.with_name('grounding-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
