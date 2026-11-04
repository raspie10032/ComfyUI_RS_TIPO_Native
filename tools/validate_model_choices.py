"""Real public-node I/O for both native model families; no image generation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--comfy-root', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(args.comfy_root))
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('rs_tipo_native', root/'__init__.py',
        submodule_search_locations=[str(root)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    spec.loader.exec_module(pkg)
    from rs_tipo_native.nodes import RSTIPOPrompt
    import torch
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        dict(name='english-pair', common_prompt='Two women in a cafe, warm afternoon light.',
             character1='blonde hair, blue eyes, pink cardigan, on left, offering a cup',
             character2='black hair, brown eyes, blue jacket, on right, receiving the cup'),
        dict(name='korean-pair', common_prompt='두 여성이 햇빛이 드는 카페에 있다.',
             character1='금발, 파란 눈, 분홍 카디건, 왼쪽에서 컵을 건네는 여성',
             character2='검은 머리, 갈색 눈, 파란 재킷, 오른쪽에서 그 컵을 받는 여성'),
    ]
    rows = []
    for family in ('tipo_2_1', 'gemma_tipo'):
        for case in cases:
            for fmt in ('tags', 'natural_language'):
                started = time.monotonic()
                inputs = dict(character_count=2, model_family=family, output_format=fmt,
                    expand_common=True, temperature=0, seed=42, generation_max_tokens=1024,
                    max_added_tags=6, excluded_tags='watermark, signature',
                    **{k:v for k,v in case.items() if k != 'name'})
                row = dict(family=family, case=case['name'], format=fmt, inputs=inputs)
                try:
                    result = RSTIPOPrompt().expand(**inputs)
                    row.update(outputs=list(result['result'][1:4]),
                        report=json.loads(result['ui']['tipo_report'][0]), error=None)
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                row['seconds'] = round(time.monotonic()-started, 3)
                rows.append(row)
                args.output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
                    source=str(root), rows=rows), ensure_ascii=False, indent=2))
                print(json.dumps({k: row[k] for k in ('family','case','format','error','seconds')},
                    ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
