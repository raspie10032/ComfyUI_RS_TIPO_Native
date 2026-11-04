# Changelog

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
