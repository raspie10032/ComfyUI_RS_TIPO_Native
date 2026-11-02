"""Replay addition selection on saved, real model completions; no inference."""
import argparse
import json
from pathlib import Path

from validate_role_additions import package  # Register the local package.
from rs_tipo_native.additions import select_additions
from rs_tipo_native.expansion import append_completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study = json.loads(args.record.read_text())
    study['selection_evidence'] = 'replayed selection over unchanged real model completions'
    study['selection_source'] = str(args.record)
    for case in study['cases']:
        case['focused'] = []
        for i, reports in enumerate(case['reports']):
            old = reports['previous']
            tags, rejected = select_additions(old['original'], old['added_tags'].split(','),
                                             common=case['common'], limit=6)
            out, additions = append_completion(old['original'], ', '.join(tags))
            reports['focused'] = {**old, 'output': out, 'added_tags': additions,
                'rejected_tags': old['rejected_tags'] + rejected,
                'max_added_tags': 6, 'grounded_additions': True}
            case['focused'].append(out)
            print(case['name'], i + 1, additions or '(unchanged)')
    args.output.write_text(json.dumps(study, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
