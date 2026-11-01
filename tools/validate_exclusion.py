"""Real-weight paired test of decode-time exclusion; no removal guard involved."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rs_tipo_native', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from rs_tipo_native.expansion import expand_owned
from rs_tipo_native.exclusion import TagExclusion, normalize, parse_excluded
from rs_tipo_native.runtime import load_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, tokenizer = load_weights(args.model)
    model.to(args.device)
    scenes = [
        ('cup', '2girls, cafe, afternoon, sitting at table',
         ['blonde hair, blue eyes, pink cardigan, on left, offering cup',
          'black hair, brown eyes, yellow sweater, on right, receiving cup']),
        ('dance', '2girls, ballroom, dancing together, evening',
         ['blonde hair, blue eyes, pink dress, leading dance, hand on other waist',
          'black hair, brown eyes, yellow dress, following dance, hand on other shoulder']),
        ('hug', '2girls, garden, autumn, hugging each other',
         ['blonde hair, blue eyes, pink cardigan, arms around other',
          'black hair, brown eyes, yellow sweater, leaning on other, head on shoulder'])]
    report = {'device': args.device, 'protect_source': False, 'cases': []}
    for label, common, people in scenes:
        source_keys = {normalize(p) for text in [common, *people] for p in text.split(',')}
        baseline = []
        for i, person in enumerate(people):
            baseline.append(expand_owned(model, tokenizer, common, person, protect_source=False,
                seed=913 + i, temperature=.5, max_new_tokens=96)[1])
        bans = list(dict.fromkeys(normalize(tag) for row in baseline
            for tag in row['added_tags'].split(',') if normalize(tag) and normalize(tag) not in source_keys))[:8]
        assert bans, 'Need actual baseline tags to exercise the intervention'
        case = {'scene': label, 'excluded': bans, 'slots': []}
        for i, person in enumerate(people):
            rows = []
            for excluded in (', '.join(bans), ', '.join(t.replace(' ', '_') for t in bans)):
                output, data = expand_owned(model, tokenizer, common, person, protect_source=False,
                    seed=913+i, temperature=.5, max_new_tokens=96, excluded_tags=excluded)
                assert output.startswith(person)
                assert not data['rejected_tags'], 'No post-removal guard is allowed in this study'
                checker = TagExclusion(tokenizer, parse_excluded(excluded))
                assert checker.allows(data['token_ids'], tokenizer.token_to_id('<|eos|>')), data['raw_completion']
                rows.append(data)
            assert rows[0]['token_ids'] == rows[1]['token_ids'], 'underscore/space behavior differs'
            case['slots'].append({'baseline': baseline[i], 'constrained': rows[0], 'underscore_same_ids': True})
            print(label, i+1, 'blocked', rows[0]['exclusion']['blocked_candidates'],
                  'baseline:', baseline[i]['added_tags'], '=>', rows[0]['added_tags'], flush=True)
        report['cases'].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Real-model paired tests passed; outputs require semantic review.', flush=True)


if __name__ == '__main__':
    main()
