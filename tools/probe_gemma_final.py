"""Full public-node I/O with Gemma + TIPO LoRA, no image generation."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
cli=argparse.ArgumentParser();cli.add_argument('--comfy-root',type=Path,required=True);args=cli.parse_args()
sys.path.insert(0,str(args.comfy_root))
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rs_tipo_native',root/'__init__.py',submodule_search_locations=[str(root)])
pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
from rs_tipo_native.nodes import RSTIPOPrompt
from rs_tipo_native.gemma_runtime import run_scene
from unittest.mock import patch
cases=json.loads((root/'tests/fixtures/korean_cases.json').read_text())
out=root/'validation/gemma-tipo';out.mkdir(exist_ok=True)
rows=[]
for case in cases:
 for lang in ('english','korean_sentence'):
  source=case[lang]
  for fmt in ('natural_language','tags'):
   start=time.monotonic()
   try:
    with patch('rs_tipo_native.interface.pipe_adapter',return_value=None):
     result=RSTIPOPrompt().expand(character_count=len(source['characters']),common_prompt=source['common'],
       **{f'character{i}':t for i,t in enumerate(source['characters'],1)},
       expand_common=True,output_format=fmt,generation_max_tokens=2048,temperature=.3,seed=42)
    info=json.loads(result['ui']['tipo_report'][0]); outputs=result['result'][1:2+len(source['characters'])];error=None
   except Exception as exc: info=None;outputs=None;error=f'{type(exc).__name__}: {exc}'
   row=dict(case=case['name'],language=lang,format=fmt,source=source,outputs=outputs,error=error,report=info,seconds=round(time.monotonic()-start,3))
   rows.append(row);(out/'public-node.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps(row,ensure_ascii=False),flush=True)
