# Changelog

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
