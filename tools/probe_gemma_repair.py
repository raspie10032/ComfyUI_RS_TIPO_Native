import sys,json,importlib.util,time,argparse
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--comfy-root',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.comfy_root));root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rs_tipo_native',root/'__init__.py',submodule_search_locations=[str(root)]);pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
import rs_tipo_native.gemma_runtime as r
original=r.generate;rows=[]
for name,common,chars,kw in [('weights','cafe',('(blonde hair:1.2), blue eyes, white shirt, offering cup',),{}),('q4','a woman in a garden',('brown hair, green eyes, yellow sweater, reading a book',),{'model':'heretic_ara_q4'})]:
 calls=[]
 def observe(*a,**k):
  result=original(*a,**k);calls.append(result);return result
 start=time.monotonic()
 with patch.object(r,'generate',side_effect=observe):
  try:scene,slots,info=r.run_scene(common,chars,output_format='natural_language',expand_common=True,temperature=.3,seed=42,**kw);err=None
  except Exception as e:scene=slots=info=None;err=str(e)
 rows.append(dict(case=name,scene=scene,slots=slots,info=info,error=err,calls=calls,seconds=time.monotonic()-start));(root/'validation/gemma-tipo/repair.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(name,err,flush=True)
