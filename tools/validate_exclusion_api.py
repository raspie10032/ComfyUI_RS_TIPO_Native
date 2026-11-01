"""Verify optional exclusion on current or old running ComfyUI; never submit twice."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8188)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    def get(path):
        return json.load(urllib.request.urlopen(base+path, timeout=10))
    if any(get('/queue').values()):
        raise RuntimeError('Server busy; leave its jobs running.')
    info = get('/object_info')
    kinds = ('RSTIPOPrompt', 'RSTIPONativeExpand', 'RSTIPONativeCompletion',
             'RSTIPONativeControlledExpand', 'RSTIPONativePipeExpand')
    assert all('excluded_tags' in info[k]['input']['optional'] for k in kinds)
    records = []
    for excluded in ('3d, long sleeves, white shirt', '3d, long_sleeves, white_shirt'):
        prompt = {'1': {'class_type': 'RSTIPOPrompt', 'inputs': {
            'character_count': 2, 'expand_common': True, 'length': 'short',
            'max_new_tokens': 48, 'seed': 1203, 'temperature': .5,
            'common_weight': .5, 'character_weight': 1.5,
            'common_prompt': '2girls, cafe, sitting at table',
            'character1': '(blonde hair:1.2), pink cardigan, offering cup',
            'character2': 'black hair, yellow sweater, receiving cup',
            'protect_source': True, 'advanced': False, 'excluded_tags': excluded}}}
        req = urllib.request.Request(base+'/prompt', json.dumps({'prompt': prompt,
            'client_id': 'rs-tipo-011-exclusion'}).encode(), headers={'Content-Type': 'application/json'})
        pid = json.load(urllib.request.urlopen(req, timeout=15))['prompt_id']
        print(args.port, 'submitted', pid, flush=True)
        deadline = time.monotonic()+180
        while time.monotonic()<deadline:
            history = get('/history/'+pid)
            if pid in history:
                row = history[pid]
                assert row['status']['status_str']=='success', row['status']
                report = json.loads(row['outputs']['1']['tipo_report'][0])
                assert len(report['slots'])==3
                for slot in report['slots']:
                    assert slot['exclusion']['tags']==['3d', 'long sleeves', 'white shirt']
                    tags = {t.strip().replace('_',' ').casefold() for t in slot['added_tags'].split(',')}
                    assert not tags.intersection({'3d','long sleeves','white shirt'})
                records.append({'prompt_id':pid, 'report':report})
                break
            time.sleep(.25)
        else:
            raise TimeoutError(f'Observe prompt {pid}; do not resubmit.')
    for a,b in zip(records[0]['report']['slots'],records[1]['report']['slots']):
        assert a['token_ids']==b['token_ids']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'port':args.port,'records':records,'underscore_same_ids':True},ensure_ascii=False,indent=2))
    print(args.port,'PASS: common + two characters; source protection ON; underscore same tokens',flush=True)


if __name__ == '__main__':
    main()
