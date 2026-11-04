"""Additional direct Gemma TIPO checks; no image generation."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
cli=argparse.ArgumentParser();cli.add_argument('--comfy-root',type=Path,required=True);args=cli.parse_args()
sys.path.insert(0,str(args.comfy_root));root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rs_tipo_native',root/'__init__.py',submodule_search_locations=[str(root)])
pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
from rs_tipo_native.gemma_runtime import run_scene
cases=[
 ('three-korean','성인 여성 세 명이 정원에서 차를 준비한다.',('왼쪽 금발 파란 눈 여성이 흰 셔츠를 입고 찻잔을 받친다.','가운데 검은 머리 갈색 눈 여성이 파란 카디건을 입고 찻주전자로 첫 번째 여성의 잔에 차를 따른다.','오른쪽 갈색 머리 초록 눈 여성이 노란 스웨터를 입고 접시에 쿠키를 놓는다.'),{}),
 ('weights','cafe',('(blonde hair:1.2), blue eyes, white shirt, offering cup',),{}),
 ('exclusion','a woman in a garden',('brown hair, green eyes, yellow sweater, reading a book',),{'excluded':('watermark','text','black hair')}),
 ('pilot001','a woman in a garden',('brown hair, green eyes, yellow sweater, reading a book',),{'adapter':'pilot-001'}),
 ('image30k','a woman in a garden',('brown hair, green eyes, yellow sweater, reading a book',),{'adapter':'image-30k'}),
 ('nativeq4','a woman in a garden',('brown hair, green eyes, yellow sweater, reading a book',),{'model':'heretic_ara_q4'}),
]
rows=[];out=root/'validation/gemma-tipo/edges.json'
for name,common,chars,kw in cases:
 start=time.monotonic()
 try:
  scene,slots,report=run_scene(common,chars,output_format='natural_language',expand_common=True,seed=42,temperature=.3,**kw);error=None
 except Exception as e:scene=slots=report=None;error=f'{type(e).__name__}: {e}'
 row=dict(case=name,source=dict(common=common,characters=chars),common=scene,characters=slots,report=report,error=error,seconds=round(time.monotonic()-start,3));rows.append(row);out.write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps(row,ensure_ascii=False),flush=True)
