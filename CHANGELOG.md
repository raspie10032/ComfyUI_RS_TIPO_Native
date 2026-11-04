# Changelog

## 0.2.8 — 2026-11-05

- Recover malformed common-string quotes, line breaks and separators only
  when a complete, independently valid character object identifies the field
  boundary. Preserve later common text, literal inner quotes and every
  character description. Never fill in missing participant fields.
- For other JSON syntax failures, retry once with a syntax-only request
  containing the previous response and required shape. Do not repeat the
  original expansion task while repairing JSON. Keep raw diagnostics and
  existing slot, source-weight and explicit-exclusion checks.
- Add regression coverage for early quotes, inner quotes, missing commas,
  line breaks, escapes, 0/1/2/10 participant routing, bounded repair and rejected
  incomplete/ambiguous structures. No new dependencies or model changes.
  Preserve upstream credits and licenses.

## 0.2.7 — 2026-11-05

- Remove the remaining sentence-word and punctuation rejection in Gemma's
  caption-to-tag stage. A generated phrase such as `lighting is soft and even`
  now reaches downstream unchanged instead of exhausting a retry and failing.
- Join wrapped output lines before tag deduplication so later details are
  retained. Keep the tag-conversion instruction, source weights, addition
  budget, explicit exclusions and bounded repair for empty/JSON tag responses.
  Sentence-like wording is accepted; this does not guarantee compact tags.
- Add regression coverage for the reported lighting clause, sentence wording,
  punctuation, wrapped lines and duplicate removal. No model, dependency or
  native resource-management changes. Preserve upstream credits and licenses.

## 0.2.6 — 2026-11-04

- Remove Gemma's lexical caption-to-tag content guard. Preserve generated
  tags without judging their vocabulary, roles, clothes, props or positions.
  Remove only duplicate tags, including duplicates of individual tags inside
  weighted source groups; keep original weights and escaped character names.
- Remove the common-expansion novelty check and literal outfit-wording
  validation. Valid output is no longer rejected or regenerated because its
  content fails those hand-written word comparisons.
- Remove unrelated concrete scene, hair, clothing and prop examples from
  joint planning and tag-format repair. The example output shape uses empty
  strings and the actual participant keys, with source data kept separate.
- Retain structural JSON, output-format, participant-count, source-weight and
  explicitly requested exclusion validation. Keep a bounded format-repair
  retry using the same adapted model, with raw responses in diagnostics.
- Keep the configured addition budget and final string-based added-tag counts.
  Neither deduplication nor the result panel attempts to judge semantic quality.
  No new runtime dependencies or model-weight changes. Preserve upstream
  attribution and licenses. See [validation and limits](docs/gemma-expansion-retry-fix.ko.md).

## 0.2.5 — 2026-11-04

- Restore visual expansion in Gemma's prose planning stage. Ask for new
  compatible common lighting/atmosphere and character gaze/hands/posture
  before the same model renders captions as tags. Keep the positive detail
  budget during malformed-response repair instead of asking for zero details.
- Detect a common-field restatement using a conservative English lexical
  check. Retry that field alone as prose when all characters are accepted,
  preserving their descriptions. Fail with recorded diagnostics if the common
  restatement remains after the existing one-retry limit. This check is not a
  semantic judge and deliberately does not assess Korean translation novelty.
- Remove empty comma items from the unweighted English planning input,
  including a trailing comma exposed by a weighted group. Preserve the
  original prompt, escaped names and weight groups in the final tag output.
- For Korean prose planning, ask for all supplied appearance and worn outfits
  before optional action details. The checked Q4 case retained both outfits;
  Korean common expansion and time-of-day omissions remain model limitations.
- Show field-level final added-tag counts, `unchanged`, `disabled` or
  `translated` status in the result panel. Count string additions after
  deduplication and the addition budget, without claiming semantic novelty
  or counting translation as new detail.
- Add regression coverage for common-only repair, exhausted expansion repair,
  zero detail budgets and final report counts. Keep model weights, runtime
  dependencies, native resource management and upstream attribution unchanged.
  See [text validation and limits](docs/gemma-expansion-fix.ko.md).

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
