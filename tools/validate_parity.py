"""Compare against locally inspected official HF export; not used by the nodes."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rs_tipo_native", ROOT / "__init__.py",
                                            submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from rs_tipo_native.runtime import load_weights
from rs_tipo_native.model import KVCache
from rs_tipo_native.generation import choose_token, complete


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--reference-source", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    upstream = types.ModuleType("official_kohaku")
    upstream.__path__ = [str(args.reference_source.resolve())]
    sys.modules[upstream.__name__] = upstream
    reference_code = importlib.import_module("official_kohaku.modeling_kohaku")
    cfg = reference_code.KohakuConfig(**json.loads((args.model / "config.json").read_text()))
    with torch.device("meta"):
        reference = reference_code.KohakuForCausalLM(cfg)
    reference.load_state_dict(load_file(str(args.model / "model.safetensors")), strict=True, assign=True)
    # Materialize the nonpersistent RoPE buffer, then retain its official float32 frequencies.
    reference.model.rotary_emb.inv_freq = 1.0 / (cfg.rope_theta ** (
        torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
    reference = reference.to(device=device, dtype=torch.float16).eval()
    reference.model.rotary_emb.inv_freq = 1.0 / (cfg.rope_theta ** (
        torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim))
    native, tokenizer = load_weights(args.model)
    native = native.to(device)
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(args.model), local_files_only=True)
    records = []
    prompts = {
        "common": "target: <|long|>\ntag: 2girls, university campus, autumn, afternoon, ",
        "character_a": "target: <|long|>\ntag: 1girl, blonde hair, blue eyes, pink cardigan, ",
        "character_b": "target: <|long|>\ntag: 1girl, black hair, brown eyes, yellow sweater, ",
    }
    for case, prompt in prompts.items():
        ids = tokenizer.encode(prompt).ids
        assert ids == hf_tokenizer.encode(prompt), "Tokenizer/BOS mismatch"
        tensor = torch.tensor([ids], device=device)
        expected = reference(tensor, use_cache=False).logits.float()
        actual = native(tensor, last_only=False).float()
        difference = (actual - expected).abs()
        match = int((actual.argmax(-1) == expected.argmax(-1)).sum())
        assert match == len(ids), f"{case}: prefill greedy predictions disagree"
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
        cache = KVCache(native.config, len(ids) + 32, device, torch.float16)
        native_logits = native(tensor, cache=cache)[0, -1]
        ref_out = reference(tensor, use_cache=True)
        ref_logits, ref_cache = ref_out.logits[0, -1], ref_out.past_key_values
        history = list(ids)
        generated = []
        max_decode_delta = 0.0
        decode_deltas = []
        rng = torch.Generator(device="cpu").manual_seed(72157)
        start = time.perf_counter()
        for step in range(32):
            a = choose_token(native_logits, history, rng, temperature=0)
            b = choose_token(ref_logits, history, rng, temperature=0)
            assert a == b, f"{case}: greedy mismatch at generated token {step}: {a} != {b}"
            delta = float((native_logits - ref_logits).abs().max())
            decode_deltas.append(delta)
            max_decode_delta = max(max_decode_delta, delta)
            generated.append(a)
            history.append(a)
            if a == tokenizer.token_to_id("<|eos|>") or step == 31:
                break
            token = torch.tensor([[a]], device=device)
            native_logits = native(token, cache=cache)[0, -1]
            ref_out = reference(token, past_key_values=ref_cache, use_cache=True)
            ref_logits, ref_cache = ref_out.logits[0, -1], ref_out.past_key_values
        row = dict(case=case, prompt=prompt, input_tokens=len(ids),
                   prefill_max_abs_delta=float(difference.max()),
                   prefill_mean_abs_delta=float(difference.mean()),
                   prefill_argmax_matches=match,
                   decode_max_abs_delta=max_decode_delta,
                   per_token_logit_max_abs_delta=decode_deltas,
                   greedy_match_tokens=len(generated), completion=tokenizer.decode(generated),
                   paired_decode_seconds=time.perf_counter() - start)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        records.append(row)
    # Fixed seed sampled outputs, isolated requests, and real EOS/length behavior.
    repeats = [complete(native, tokenizer, prompts["character_a"], max_new_tokens=128,
                        seed=72157) for _ in range(2)]
    assert repeats[0].token_ids == repeats[1].token_ids, "Same-seed repeat failed"
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "parameters": sum(p.numel() for p in native.parameters()),
              "cases": records,
              "sampled_repeat": {"seed": 72157, "equal": True,
                                 "tokens": len(repeats[0].token_ids),
                                 "finish_reason": repeats[0].finish_reason,
                                 "completion": repeats[0].text,
                                 "seconds": [r.seconds for r in repeats]},
              "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PARITY PASS: {args.output}", flush=True)


if __name__ == "__main__":
    main()
