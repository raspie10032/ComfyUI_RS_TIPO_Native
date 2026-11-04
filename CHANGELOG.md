# Changelog

## 0.2.4 — 2026-11-04

- Compare the individual tags inside explicit weighted groups when appending
  model output. Plain echoes such as `open mouth` no longer duplicate the
  source `(open mouth, determined:1.3)` or consume the addition budget.
- Compare case, whitespace, underscores and literal-parenthesis escapes
  without changing the original prompt, weight values or character names.
  Preserve accepted weighted groups as whole output units.
- Add five regression tests covering the reported trailing-comma group,
  nested weights, NAI numeric weights, literal names, generated-group echoes
  and the public node's common/character outputs and addition budget.
- Keep inference, model placement, upstream credits and licenses unchanged.

## 0.2.3 — 2026-11-04

- Remove ComfyUI literal-parenthesis escapes from intermediate Gemma prose
  input, where copied escapes could produce invalid JSON. Keep strict JSON
  parsing, report the actual decoder error, and retry once with explicit
  escaping instructions rather than guessing or altering malformed output.
- Plan English tag inputs using their unweighted meaning. Preserve the
  original character identity, escapes and weighted groups in the final node
  output. For direct natural-language output, move an inserted sentence
  article outside an otherwise exact weighted group; report the precise field
  and missing clause when repair is still required.
- Reject sentence echoes in the caption-to-tag stage and retry with a compact
  tag-conversion request using the same adapted model.
- Keep commas inside parentheses/brackets together during tag deduplication
  and addition-budget enforcement, so a group such as `(smile, blush:1.5)`
  cannot be truncated halfway through.
- Add eight regression tests and verify a two-character escaped-name/weighted
  prompt on the live ComfyUI server with both BF16 and Q4 plus the
  `scale-20pct-001` adapter. This checks text execution and source preservation;
  downstream image generation and general semantic fidelity are not established
  by this patch. See [validation notes](docs/gemma-json-weight-fix.ko.md).

## 0.2.2 — 2026-11-04

- Keep the Gemma GGUF per-layer embedding (PLE) table in CPU RAM and gather
  packed token rows before transferring them to the inference device. Exclude
  this table from ComfyUI GPU-load and partial-offload accounting. BF16 model
  placement is unchanged; PLE weight patches remain unsupported.
- Apply supported additive Linear LoRAs with ComfyUI's native `LoRAAdapter.h()`
  instead of rebuilding a full delta weight for every token. Preserve the
  original weight-merging path for special patches and older adapter APIs.
  No persistent device-factor cache or additional dependency is introduced.
- Avoid redundant vocabulary scans, token reads and cumulative decoding when
  excluded tags are empty. Keep progress, cancellation and sampler restoration.
- Decode sequential output-weight chunks through packed-storage slices and
  reuse Q4_K/Q5_K floating buffers for minimum subtraction. Retain FP32
  dequantization and the original packed weights.
- Preserve persistent packed-weight version counters across inference-mode
  loading, cloning and `to(copy=True)` so ComfyUI unload/reload works correctly.
- Document real Q4/BF16 text comparisons, sampled-token checks and model
  lifecycle checks. In the tested Q4 setup, PLE placement saved about 1.795 GiB
  and buffer reuse saved another 33 MiB of peak allocated VRAM. The separate
  Linear LoRA comparison reduced elapsed time by about 19–20%. These measured
  effects have different baselines; performance and token equality are not
  guaranteed for all inputs, models or devices. Image-model alternation was
  not part of these measurements.
- Retain TIPO/Gemma upstream attribution and City96's Apache-2.0 notices;
  describe local GGUF decoder changes in the vendored notice.

## 0.2.1 — 2026-11-04

- Add Comfy Registry metadata for `rs-tipo-native` by `raspie10032`, with a
  ComfyUI 0.41.0 support baseline and upstream attribution. Registry packaging
  retains runtime files, examples, documentation and licenses, and excludes
  development/training tools. Publishing runs only after validation succeeds.
- Repair malformed Gemma prose using the previous response and the failed field
  names. Preserve accepted common/character fields instead of replacing them
  with shorter rewrites. Retry once using the same adapted model.
- Disambiguate explicit `외형: … . 의상: … .` / `Appearance: … . Outfit: … .`
  inputs as appearance, worn clothing/accessories and action. Explicitly held or
  carried items remain props; ordinary unlabelled inputs keep their old path.
- Check literal English outfit items in wearing clauses. Failed repair raises
  an error with per-call raw responses and validation errors; it does not send
  invalid prose downstream or silently fall back to another model.
- Add regression coverage and a real-model 2/4/6/8/10-character text evaluation,
  including reversed 10-character slots. This verifies selected text attributes,
  not image fidelity or general semantic accuracy. See the evaluation document
  for the observed parasol/umbrella substitution.

## 0.2.0 — 2026-11-04

- Integrate official TIPO 2.1 and Gemma with a TIPO LoRA into the same
  `RS TIPO Native (Experimental)` node. Choose `tipo_2_1` or `gemma_tipo`
  using `model_family`; retain the `RSTIPOPrompt` identifier and legacy graphs.
- Add native Gemma BF16 and Q4 execution with ComfyUI model management.
  Gemma requires ComfyUI's native Gemma 4 text-generation APIs. Neither backend
  requires a separate inference server or llama.cpp.
- Offer `pilot-001`, `pilot-002`, `image-30k`, and the newly published
  `scale-20pct-001` adapters. Download pinned files on first execution and verify
  their size and SHA-256 before reusing them locally.
- Add a `scale-20pct-001` workflow that automatically downloads its adapter.
  The adapter is an experimental **CC BY-NC 4.0 non-commercial** release;
  `pilot-002` remains the default. Check each model's own license.
- Add Korean Gemma input, common/character prompt handling, a result panel
  with source and raw-output inspection, and a bounded tag-format retry.
  Execution does not guarantee semantic accuracy; omissions remain documented.
- Publish the Korean usage guide and preserve upstream author attribution,
  adapted-source notices, and third-party licenses.

TIPO research and official checkpoints belong to the TIPO authors,
KohakuBlueleaf and their contributors. The Gemma LoRAs are an independent
adaptation; no upstream endorsement is implied. See [NOTICE](NOTICE) and
[source references](docs/references.md).
