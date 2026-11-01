# Native baseline validation — 2026-11-01

This validates the independent TIPO execution baseline. Character ownership and
input-span attention control are still research work; these results do not prove
that character attributes remain separate or that prompt quality improves.

## Environment and provenance

- GPU: RTX 4070 Ti SUPER; PyTorch 2.14.0+cu130.
- Model: KBlueLeaf/TIPO-v2.1-1B-A200M, revision
  `f5a318524a4ab30cdbbf51816cf406170f454e65`.
- Native parameter count: 990,784,256; FP16 weights about 1.85 GiB.
- Reference: official PyTorch Kohaku implementation, loaded only for parity tests.
- Inference dependencies: PyTorch, safetensors, tokenizers, ComfyUI ModelPatcher.
  llama.cpp and Transformers are not imported by the native inference package.

## Numerical comparison

Three official-format prompts were tested: a common campus scene, a blonde
character in a pink cardigan, and a black-haired character in a yellow sweater.
Full prefill logits matched the official implementation exactly in all three
cases. Greedy decoding matched 32 tokens per case, 96 tokens total.

The first last-token-only decode projection differed by up to 0.125 in FP16 logits
(0.0625 for two cases); following cached decode comparisons were identical.
This is not a claim of bitwise equality for every execution path. The observed
difference did not change the tested greedy sequences. More sampling cases may
diverge. A sampled request repeated with seed 72157 produced the same 117 tokens
and terminated at EOS both times.

Reproduce against the downloaded official reference with `tools/validate_parity.py`;
see its command-line help. Machine-readable local evidence is in ignored
`validation/parity.json`. Reference code and weights are excluded from Git.

## ComfyUI execution and lifecycle

`tools/validate_comfy.py` ran against isolated server port 8196, with only this
extension enabled. Both node classes registered and executed through `/prompt`.

| Case | Observed result |
| --- | --- |
| Greedy completion, 64 tokens | Success; 1.1648 seconds in lifecycle run, 1.4239 seconds in initial smoke |
| Weighted source string expansion | Success; 115 output tokens, EOS, 2.2264 seconds |
| Source string preservation | Original `(blonde hair:1.2), blue eyes, pink cardigan` retained as output prefix |
| Context overflow | Explicit execution error; no silent truncation |
| Model unload | PyTorch reserved VRAM changed from 2,013,265,920 to 33,554,432 bytes |
| Reload | Subsequent generation succeeded; reserved memory returned to the prior level |
| Cancellation | ComfyUI returned `execution_interrupted`; next generation succeeded |

Times come from generation metadata and exclude model acquisition/loading.
Preserving prompt weighting syntax does not mean TIPO interprets it as an
attention multiplier. Cancellation was observed at the node boundary; the unit
suite also checks interruption propagation during the generation call.

The native package passed 9 CPU tests: cached/full equivalence, cached multi-token
causality, request isolation, seed/global RNG isolation, context and sampling
errors, cancellation propagation, strict checkpoint/type checking, and dependency
isolation. Synthetic tests supplement the actual checkpoint/API tests.

Local raw API evidence is in ignored `validation/comfy-api.json` and
`validation/comfy-full.json`. Run lifecycle tests only against an isolated server:
they unload models and interrupt execution.

## Installed local runtime

The project was linked into `/home/raspie/AI/ComfyUI/custom_nodes/` and the
`comfyui.service` user service restarted after verifying its queue was empty.
The isolated TIPO server was stopped. The existing port 8188 runtime registered
both nodes and executed a real completion successfully: 32 output tokens,
0.9309 seconds of generation, 1.27 seconds reported for the complete prompt.
Local evidence: ignored `validation/production-smoke.json` and service journal.
This was an API execution test, not a frontend interaction or image-quality test.

## Remaining scope

Character/common ownership masks, weighted attention spans, multi-character pipe
integration, Windows and non-CUDA devices, partial offloading, quantization, and
public packaging have not been validated. No image-generation improvement or
llama.cpp speed comparison is claimed.
