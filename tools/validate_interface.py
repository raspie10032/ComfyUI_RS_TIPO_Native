"""Run STRING interface/interaction cases on the idle resident ComfyUI."""
import json
from pathlib import Path
import time
import urllib.request

BASE = 'http://127.0.0.1:8188'
DEST = Path(__file__).resolve().parents[1] / 'validation' / 'interface-live.json'
def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=10))
def main():
    if any(get('/queue').values()):
        raise RuntimeError('Server has other jobs; do not interrupt them.')
    info = get('/object_info')
    def node(kind, **values):
        inputs = {}
        for name, row in info[kind]['input'].get('required', {}).items():
            if len(row) > 1 and 'default' in row[1]:
                inputs[name] = row[1]['default']
            elif isinstance(row[0], list):
                inputs[name] = row[0][0]
        return {'class_type': kind, 'inputs': {**inputs, **values}}
    records = []
    scenarios = [
        ('give_receive', '2girls, university cafe, afternoon, wooden table',
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
    for label, common, interaction, first, second in scenarios:
        for coordinated in (False, True):
            graph = {
                '1': node('RSTIPOPrompt', character_count=2, common_prompt=common,
                    character1=first, character2=second, character3='STALE INPUT MUST NOT LEAK',
                    interaction_prompt=interaction if coordinated else '',
                    expand_common=False, max_new_tokens=96, seed=8910, temperature=0.),
                '2': node('RSMultiCharacterFinalize', building_pipe=['1', 0]),
                '3': node('PreviewAny', source=['2', 0]),
                '4': node('PreviewAny', source=['1', 1]),
                '5': node('PreviewAny', source=['1', 2]),
                '6': node('PreviewAny', source=['1', 3]),
                '7': node('PreviewAny', source=['1', 4]),
            }
            req = urllib.request.Request(BASE+'/prompt', json.dumps({'prompt':graph, 'client_id':'rs-tipo-interface-validation'}).encode(), headers={'Content-Type':'application/json'})
            pid = json.load(urllib.request.urlopen(req, timeout=10))['prompt_id']
            deadline = time.monotonic()+120
            while time.monotonic() < deadline:
                history = get('/history/'+pid)
                if pid in history:
                    row = history[pid]
                    assert row['status']['status_str']=='success', row['status']
                    out = row['outputs']
                    assert out['5']['text'][0].startswith(first)
                    assert out['6']['text'][0].startswith(second)
                    assert out['7']['text'][0]=='', out['7']
                    assert 'STALE INPUT MUST NOT LEAK' not in out['3']['text'][0]
                    if coordinated:
                        assert interaction in out['4']['text'][0]
                        assert interaction in out['3']['text'][0]
                    records.append({'case':label, 'coordinated':coordinated, 'record':row})
                    DEST.write_text(json.dumps(records, ensure_ascii=False, indent=2))
                    print(label, 'shared' if coordinated else 'tags_only', 'PASS', flush=True)
                    break
                time.sleep(.25)
            else:
                raise TimeoutError(f'Observe existing prompt {pid}; do not resubmit.')
if __name__=='__main__':
    main()
