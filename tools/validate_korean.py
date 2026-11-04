"""Text-only actual ComfyUI Korean conversion audit. Explicit isolated port required."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 8190 <= args.port <= 8198:
        raise ValueError('Use an isolated test server on 8190–8198.')
    base = f'http://127.0.0.1:{args.port}'

    def get(path):
        return json.load(urllib.request.urlopen(base + path, timeout=10))

    def post(path, obj):
        request = urllib.request.Request(base+path,json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(request,timeout=10))

    q = get('/queue')
    if q['queue_running'] or q['queue_pending']:
        raise RuntimeError('Test queue is busy; observe existing work, do not resubmit.')
    infos = get('/object_info')
    custom = [i['python_module'] for i in infos.values() if 'custom_nodes' in i.get('python_module','')]
    if not custom or any('ComfyUI_RS_TIPO_Native' not in n for n in custom):
        raise RuntimeError('Only the TIPO extension may be enabled on this test server.')
    cases = []
    for case in json.loads(args.cases.read_text()):
        for variant in ('english','korean_tags','korean_sentence'):
            cases.append((case['name'],variant,case[variant],{}))
    cases.extend([
        ('weighted','mixed',{'common':'2girls, cafe, 컵 주고받기','characters':[
            '(금발:1.2), blue_eyes, 흰 셔츠, 컵 건네기',
            'brown hair, green eyes, [파란 원피스], 컵 받기']},{}),
        ('tea_three','korean_sentence',{'common':'성인 여성 세 명이 정원에서 차를 준비한다.','characters':[
            '금발 여성이 빈 찻잔을 양손으로 받친다.',
            '갈색 머리 여성이 주전자를 기울여 첫 번째 여성의 찻잔에 차를 따른다.',
            '검은 머리 여성이 접시에 쿠키를 담아 탁자 위에 놓는다.']},{}),
        ('scene_only','korean_sentence',{'common':'비 오는 저녁의 빈 카페, 나무 탁자와 창밖 가로등','characters':[]},{}),
        ('english_explicit','english',json.loads(args.cases.read_text())[0]['english'],{'input_language':'english'}),
        ('exclude_conflict','mixed',{'common':'blue_eyes','characters':[]},{'excluded_tags':'파란 눈'}),
    ])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rows = []
    for name,variant,source,options in cases:
        inputs = {'character_count':len(source['characters']), 'common_prompt':source['common'],
            'expand_common':True,'length':'long','max_new_tokens':96,'seed':42,'temperature':.5,
            'common_weight':.5,'character_weight':1.5,'advanced':False,'protect_source':True,
            'excluded_tags':'watermark, 서명, 글자','rating':'safe','quality':'masterpiece','quality_era':'newest',
            'input_language':'auto','translation_max_tokens':4096,
            **{f'character{i}':t for i,t in enumerate(source['characters'],1)}, **options}
        # The English compatibility pair must have exactly equal non-language inputs.
        if variant=='english':
            inputs['excluded_tags']='watermark, signature, text'
        tick=time.monotonic()
        pid=post('/prompt',{'prompt':{'1':{'class_type':'RSTIPOPrompt','inputs':inputs}},'client_id':'rs-tipo-korean-audit'})['prompt_id']
        # Persist the handle before waiting; an observation timeout is not permission to resubmit.
        args.output.with_suffix('.pending.json').write_text(json.dumps({'prompt_id':pid,'case':name,'variant':variant}))
        deadline=time.monotonic()+300
        while True:
            history=get('/history/'+pid)
            if pid in history:
                record=history[pid];break
            if time.monotonic()>deadline:
                raise TimeoutError(f'Observe existing prompt {pid}; do not resubmit.')
            time.sleep(.5)
        report=json.loads(record['outputs']['1']['tipo_report'][0]) if record['status']['status_str']=='success' else None
        row={'case':name,'variant':variant,'prompt_id':pid,'inputs':inputs,'status':record['status'],
             'report':report,'seconds':round(time.monotonic()-tick,3)}
        rows.append(row)
        args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'case':name,'variant':variant,'status':record['status']['status_str'],
                          'seconds':row['seconds']},ensure_ascii=False),flush=True)
        expected='error' if name=='exclude_conflict' else 'success'
        if record['status']['status_str']!=expected:
            raise RuntimeError(f'Unexpected {row["status"]}: inspect saved report.')
    baseline=next(r['report'] for r in rows if r['case']=='cup' and r['variant']=='english')
    explicit=next(r['report'] for r in rows if r['case']=='english_explicit')
    for a,b in zip(baseline['slots'],explicit['slots']):
        for key in ('input','output','text','token_ids','added_tags','proposed_tags','rejected_tags'):
            if key in a: assert a[key]==b[key],(key,a,b)
    assert not baseline['language']['applied'] and not explicit['language']['applied']
    for row in rows:
        if row['report']:
            source=row['report']['language'].get('converted')
            if source:
                for slot in row['report']['slots']:
                    original=source['common'] if slot['slot']=='common' else source['characters'][slot['slot']-1]
                    assert slot['output'].startswith(original),(row['case'],slot)
    args.output.with_suffix('.pending.json').unlink()
    print('Korean ComfyUI text audit PASS',flush=True)


if __name__=='__main__':main()
