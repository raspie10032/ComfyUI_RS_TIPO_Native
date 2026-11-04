> Historical standalone TIPO implementation/test record. The current active node uses Gemma + TIPO LoRA. See [current usage](usage.ko.md) and [Gemma validation](gemma-tipo-validation.md).

# Automatic scene/role context and socket repair — 2026-11-01

The main `RSTIPOPrompt` UI accepts `common_prompt`, active `character1`–`character10`
STRINGs, and an optional RS building pipe. There is **no separate interaction
input**. Character output indexes and the building-pipe output remain stable.

## What changed

`scene_context.py` gathers recognized English action/position clauses from the
original active character inputs, adds the original common scene, and identifies
the current target's role. Each target gets that caption plus its own complete
source. Other characters' appearance/LoRA/style clauses are omitted from the
caption. Unknown or mixed appearance/action clauses stay private. Scene context
uses the common-attention weight; a zero common weight also blocks this shared
caption. The target's original tag source keeps its character-attention weight.
This does not introduce another model call. Complete natural-language reasoning,
role planning and arbitrary multilingual action extraction are not implemented.

Only active slots contribute. Empty/zero-character scenes do not fabricate
participants. Input-slot numbers do not imply left/right placement. The derived
caption appears in the results panel and is not appended to the final image
prompt. Original strings, role clauses, prompt weights and pipe settings are
retained. When protection is enabled, an exact action belonging only to a peer
is rejected as a new target addition; this limited guard is not semantic proof.

The former prototype caption socket is removed on restoring older workflows.
Its upstream source node is retained, and any backing string is retained as
`rs_tipo_legacy_interaction` workflow metadata. It is not used for generation.
Move any wanted shared description into the common STRING.

## UI defect and fix

Zero-height converted STRING widgets placed multiple linked inputs at the same
connection coordinate. Their invisible labels made the overlap harder to trace.
Each visible STRING now has its own 20px row, and modern frontends receive
explicit socket labels. Linked advanced inputs remain visible even when the
advanced toggle is off. The report uses an inner flex container so legacy core
cannot reset its layout when managing the outer DOM widget.

## Reproduction

```sh
python -m unittest discover -s tests -v
node tests/test_interface.mjs
node tests/test_sockets.mjs
python tools/validate_scene_context.py 8188
# Install Playwright separately to use the optional isolated UI check.
node tools/check_socket_layout.cjs 8188 current
```

The live probe refuses to start while another request is queued. It checks cup
handoff, hug, dance, peer appearance replacement, a scene without characters and
three participants. It verifies source preservation, omission of inactive slot
10 and identical target input/tokens/output after changing only a peer's
appearance. Raw records stay local under `validation/automatic-context-*.json`.
A successful probe proves execution and tested invariants, not image quality.

Verified on Linux with the installed TIPO checkpoint and modern PyTorch:

| Runtime | Text generation | Socket layout / reload |
| --- | --- | --- |
| Current 0.41.0 / frontend 1.55.17 | Six actual inference requests | Unique input coordinates, counts 0/1/2/3/10, linked seed 42, `structuredClone`, original STRING restore |
| ComfyUI 0.3.26 / frontend 1.11.8 | Actual two-character inference | Same UI checks |
| ComfyUI 0.3.64 / frontend 1.27.10 | Actual two-character inference | Same UI checks; known core `undefined.id` startup error, independently reproduced without TIPO |

The source suite contains 49 passing Python checks plus two JavaScript suites.
No Windows/older PyTorch test or downstream image-generation comparison is
claimed for this revision. The generated tags remain experimental: outputs
can add unwanted expressions or context, omit useful actions and require review.
