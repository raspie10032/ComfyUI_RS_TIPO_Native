# Gemma TIPO correction validation — 2026-11-02

> Historical validation of the Gemma integration on 2026-11-02. Version 0.2.0
> also exposes the independent TIPO 2.1 backend through `model_family`.
> See [current usage](usage.ko.md) for the released interface.

## Scope and actual execution

The active `RSTIPOPrompt` now runs **one Gemma 4 E2B base with its TIPO LoRA**.
The base is `p-e-w/gemma-4-E2B-it-heretic-ara`; the default adapter is
`pilot-002`. Native BF16 and the native-PyTorch Q4 path were executed.
Each loaded adapter matched all **190 target projections**. There is no
Gemma-translation → standalone-Kohaku handoff in this node.
Deprecated standalone TIPO node classes remain hidden for old saved graphs.
Their old results are historical, not evidence about Gemma TIPO.

The same adapted CLIP/model instance handles Korean/English and both formats.
It reads the shared scene and all active character inputs in one request and
returns numbered prose fields. Natural-language output uses these sentences.
Tag output renders each caption using the **same adapted model instance**.
Typical calls: 1–2 for prose; 1–2 plus the nonempty output fields for tags.
One validation retry is bounded; wrong slots, lost weighted clauses or incomplete
outputs produce an error rather than silently using another model.

## Actual node I/O matrix

`tools/probe_gemma_final.py`: four scenes (cup transfer, dance, hair braiding,
garden reading) × English/Korean × tags/natural language = **16 successful
public-node calls**, with temperature 0.3, seed 42, adapter pilot-002 and native
BF16. No images were generated. Raw prompts, raw model outputs, accepted results
and rejected added tags are stored locally in `validation/gemma-tipo/public-node.json`.
This fixture disconnects the external RS registry bridge; it tests public-node
STRING results. Actual installed MC/NAI data classes are tested separately by
`tools/validate_pipe_protocols.py`.

Manual reading of these eight natural-language cases found the specified hair,
eyes, clothing, side and the offering/receiving, leading/following and braiding/
having-hair-braided roles retained. This is a small fixed-seed sample, not a
measured general accuracy score. Some English phrasing remains awkward.
For example, Korean braiding became “receiving a braid on her head from the
other character”; the owner was retained but the wording needs review.

## Edge cases with the real model

`tools/probe_gemma_edges.py`: **6 successful calls**:

- Korean three-person tea preparation: cup holding / pouring into that cup /
  placing cookies remain in separate numbered fields.
- English `(blonde hair:1.2)` survives verbatim inside natural-language output.
- Excluded `glasses` is absent. This sample did not prove an attempted forbidden
  token was intercepted; the sampler interception is additionally unit-tested.
- pilot-001 and image-30k load and execute with all 190 patches. A simple one-person
  sample is a loading smoke test, not a comparison of adapter quality.
- `heretic_ara_q4` + pilot-002 executes natively on PyTorch and returns prose.
  Its latency and memory behavior differ from BF16; this is not a speed claim.

## Tag rendering: improvements and remaining limits

The joint prose is the source of ownership. A lexical caption guard rejects tag
additions unsupported by that caption; explicit English source tags stay at the
front, weighted clauses stay intact, and appearance/position anchors are kept.
The panel exposes raw model text and rejected additions so this guard cannot be
mistaken for improved model understanding. Excluded phrases are blocked while
sampling; a final assertion also prevents a program-derived tag anchor from
reintroducing an excluded phrase. This is literal phrase matching, not semantic
or cross-language exclusion.

The tag output is still experimental. In the Korean cup example the final first
character retained appearance, left and cup but lost the complete giving action.
Some shared location/count details were omitted, and near-duplicate tags remain.
Natural-language output in that same matrix retained the transfer roles. Do not
claim that every semantic tag-conversion defect is solved.

## UI, protocol and scope

An isolated ComfyUI on port 8196, using the 5060, exercises the Korean two-person
natural-language node in the actual browser UI. It checks saved/reloaded controls,
structuredClone of the serialized graph, distinct socket Y coordinates, successful
execution, adapter identity and all 190 patches. Evidence is in local
`validation/gemma-ui.json` and `validation/gemma-ui.png`.

Python suite: **97 passed**; JavaScript interface/socket/download suites: **3 passed**.
The installed MC/NAI protocol check preserves masks, LoRAs, positions, text slots,
settings and snapshots, and shrunk counts exclude old slots. Native Gemma APIs
are required; older ComfyUI without those APIs gets an explicit upgrade error
before model downloads. This is not a claim of old-runtime inference support.

The regular ComfyUI source installation is a symlink to this checkout. Its resident
process started before these edits and has an active job, so this validation does
not claim that process has reloaded the new backend. The isolated test server is
stopped after validation. The code is saved on a correction branch; no release or
GitHub main deployment is implied by this report.

## Reproduce

```bash
python -m unittest discover -s tests -v
node tests/test_interface.mjs
node tests/test_sockets.mjs
node tests/test_download_status.mjs
CUDA_VISIBLE_DEVICES=1 /path/to/comfy/.venv/bin/python tools/probe_gemma_final.py --comfy-root /path/to/comfy
CUDA_VISIBLE_DEVICES=1 /path/to/comfy/.venv/bin/python tools/probe_gemma_edges.py --comfy-root /path/to/comfy
python tools/validate_pipe_protocols.py /path/to/comfy --output /tmp/gemma-pipes.json
```

The example graph is `examples/gemma-tipo-korean.json`; it uses regular STRING
primitives, one RS TIPO node, pilot-002 and natural-language output. Import it as
JSON to inspect the actual I/O test. It generates no image.
