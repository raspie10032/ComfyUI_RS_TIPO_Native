"""Compare actual TIPO completions with the previous guard and focused guard.

Run using ComfyUI's Python. CUDA_VISIBLE_DEVICES can isolate a spare GPU.
No image generation or external API is performed by this script.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rs_tipo_native', ROOT / '__init__.py',
                                            submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)

CASES = (
    ('cup', '2girls, cafe, afternoon, wooden table, passing a cup, looking at each other', (
        'blonde hair, blue eyes, white shirt, on left, offering cup, extending arms',
        'brown hair, green eyes, blue dress, on right, receiving cup, reaching out')),
    ('dance', '2girls, ballroom, evening, dancing together', (
        'pink hair, red eyes, black dress, leading dance, hand on other waist',
        'brown hair, blue eyes, white dress, following dance, hand on other shoulder')),
    ('backs', '2girls, garden, back to back, standing', (
        'black hair, blue eyes, green jacket, arms crossed',
        'blonde hair, brown eyes, red jacket, hands behind back')),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='JSON list with name, common, original fields')
    args = parser.parse_args()
    import torch
    from rs_tipo_native.runtime import load_weights, MODEL_REVISION
    from rs_tipo_native.expansion import expand_owned
    from rs_tipo_native.scene_context import scene_context
    torch.set_num_threads(4)
    model, tokenizer = load_weights(args.model)
    model.to(args.device)
    rows = []
    output = {'revision': MODEL_REVISION, 'device': args.device,
              'sampling': dict(seed=8910, temperature=0., max_new_tokens=96), 'cases': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.cases is None else [
        (row['name'], row['common'], row['original'])
        for row in json.loads(args.cases.read_text())]
    for name, common, characters in cases:
        case = {'name': name, 'common': common, 'original': characters,
                'previous': [], 'focused': [], 'reports': []}
        rows.append(case)
        for i, character in enumerate(characters):
            context, roles = scene_context(common, characters, i)
            settings = dict(interaction=context, interaction_weight=.5,
                peer_actions=tuple(action for j, actions in enumerate(roles)
                                   if j != i for action in actions), **output['sampling'])
            before, old = expand_owned(model, tokenizer, common, character, **settings)
            after, new = expand_owned(model, tokenizer, common, character,
                max_added_tags=6, grounded_additions=True, **settings)
            # Same raw sample is essential: this iteration changes selection,
            # not the model weights, source attention or generated candidates.
            assert old['token_ids'] == new['token_ids']
            assert before.startswith(character) and after.startswith(character)
            case['previous'].append(before)
            case['focused'].append(after)
            case['reports'].append({'slot': i + 1, 'previous': old, 'focused': new})
            args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
            print(f"{name}/{i+1}: {old['added_tags']} -> {new['added_tags'] or '(unchanged)'}", flush=True)


if __name__ == '__main__':
    main()
