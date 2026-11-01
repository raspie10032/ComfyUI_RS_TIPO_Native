"""Real-checkpoint attention intervention and paired text-output study."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rs_tipo_native", ROOT / "__init__.py",
                                            submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from rs_tipo_native.runtime import load_weights
from rs_tipo_native.control import controlled_prompt, OwnedAttention
from rs_tipo_native.expansion import expand_owned
from rs_tipo_native.generation import complete


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanism-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, tokenizer = load_weights(args.model)
    model = model.to("cuda:0")
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    scenes = [
        ("campus", "2girls, university campus, autumn, afternoon, walking together",
         ["(blonde hair:1.2), blue eyes, pink cardigan", "black hair, brown eyes, yellow sweater"],
         [{"black hair", "brown eyes", "yellow sweater", "black cardigan"},
          {"blonde hair", "blue eyes", "pink cardigan", "pink sweater"}]),
        ("cafe", "2girls, cafe, sitting at table, tea, warm lighting",
         ["blonde hair, blue eyes, white dress", "blonde hair, green eyes, black jacket"],
         [{"green eyes", "black jacket", "black dress"},
          {"blue eyes", "white dress", "white jacket"}]),
        ("rain", "2girls, city street, rain, sharing umbrella, evening",
         ["red hair, green eyes, black coat", "silver hair, purple eyes, white coat"],
         [{"silver hair", "purple eyes", "white coat"},
          {"red hair", "green eyes", "black coat"}]),
    ]
    common, character = scenes[0][1], scenes[0][2][0]
    prompt, neutral = controlled_prompt(tokenizer, common, character, common_weight=1, character_weight=1)
    ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    plain = model(ids).float()
    explicit = model(ids, attention_mask=neutral.mask(0, ids.shape[1], device, dtype)).float()
    neutral_delta = float((plain-explicit).abs().max())
    assert torch.equal(plain.argmax(-1), explicit.argmax(-1))
    _, weighted = controlled_prompt(tokenizer, common, character, common_weight=.25, character_weight=2)
    changed = model(ids, attention_mask=weighted.mask(0, ids.shape[1], device, dtype)).float()
    assert not torch.equal(explicit, changed), "Attention intervention had no effect on real-model logits"
    # Change *all* common tokens without changing their positions/length. At zero
    # common weight the entire decoder must be independent of their values.
    _, zero = controlled_prompt(tokenizer, common, character, common_weight=0, character_weight=1)
    altered = ids.clone()
    for index, owner in enumerate(zero.owners):
        if owner == 0:
            altered[0, index] = tokenizer.token_to_id("<|unk|>") or 0
    mask = zero.mask(0, ids.shape[1], device, dtype)
    groups = zero.mlp_groups(0, ids.shape[1], device)
    zero_delta = float((model(ids, attention_mask=mask, token_owners=groups).float() -
                        model(altered, attention_mask=mask, token_owners=groups).float()).abs().max())
    assert zero_delta == 0, f"Common information leaked: {zero_delta}"
    zero_altered = altered.clone()
    # A second owner is explicitly unreadable in both prefill and final output.
    split = len(neutral.owners)//2
    owners = tuple(-1 if i == 0 else 2 if 1 <= i < split else 1 for i in range(len(neutral.owners)))
    blocked = OwnedAttention(owners, (1.,)*len(owners), 1)
    altered = ids.clone()
    altered[0, 1:split] = 0
    mask = blocked.mask(0, ids.shape[1], device, dtype)
    groups = blocked.mlp_groups(0, ids.shape[1], device)
    blocked_delta = float((model(ids, attention_mask=mask, token_owners=groups).float() -
                           model(altered, attention_mask=mask, token_owners=groups).float()).abs().max())
    assert blocked_delta == 0, f"Other character leaked: {blocked_delta}"

    class PerturbedTokenizer:
        def __init__(self, ids):
            self.ids = ids
        def encode(self, *args, **kwargs):
            return SimpleNamespace(ids=self.ids)
        def __getattr__(self, name):
            return getattr(tokenizer, name)

    sampled_isolation = {}
    for name, policy, modified in (("zero_common", zero, zero_altered),
                                   ("blocked_owner", blocked, altered)):
        settings = dict(max_new_tokens=16, seed=72157, temperature=.5,
                        attention_control=policy)
        normal = complete(model, tokenizer, prompt, **settings)
        perturbed = complete(model, PerturbedTokenizer(modified[0].tolist()), prompt, **settings)
        assert normal.token_ids == perturbed.token_ids, f"{name}: score processing leaked input"
        sampled_isolation[name] = {"matched_tokens": len(normal.token_ids), "ids": normal.token_ids}
    records = []
    for scene, common, characters, forbidden in ([] if args.mechanism_only else scenes):
        flat = complete(model, tokenizer, "target: <|long|>\ntag: " +
                        ", ".join([common, *characters]) + ", ",
                        max_new_tokens=96, seed=72157, temperature=.5)
        records.append({"scene": scene, "mode": "flattened", "text": flat.text, "seconds": flat.seconds})
        for slot, character in enumerate(characters):
            for cw, ow in ((0., 1.), (1., 1.), (.5, 1.5), (.25, 2.)):
                output, row = expand_owned(model, tokenizer, common, character,
                    common_weight=cw, character_weight=ow, max_new_tokens=96,
                    seed=72157, temperature=.5)
                assert output.startswith(character)
                added = {t.strip().casefold() for t in row["added_tags"].split(",")}
                row.update(scene=scene, slot=slot+1, mode="owned",
                           exact_conflicts=sorted(added & forbidden[slot]))
                records.append(row)
                print(f"{scene} slot {slot+1} {cw}/{ow}: {row['seconds']:.3f}s; "
                      f"conflicts={row['exact_conflicts']}; {row['added_tags']}", flush=True)
                args.output.write_text(json.dumps({"partial": True, "records": records}, ensure_ascii=False, indent=2))
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "neutral_logit_max_delta": neutral_delta,
              "weighted_logit_max_delta": float((explicit-changed).abs().max()),
              "zero_common_perturbation_max_delta": zero_delta,
              "blocked_owner_perturbation_max_delta": blocked_delta,
              "sampled_isolation": sampled_isolation,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "seed": 72157, "records": records,
              "metric_limit": "Exact conflicting-tag matches in six hand-specified prompts; not a semantic quality benchmark."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"CONTROL STUDY COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
