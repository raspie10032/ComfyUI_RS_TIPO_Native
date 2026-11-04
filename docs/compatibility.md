> Historical standalone TIPO implementation/test record. The current active node uses Gemma + TIPO LoRA. See [current usage](usage.ko.md) and [Gemma validation](gemma-tipo-validation.md).

# Older ComfyUI and follow-up protection — 2026-11-01

## Scope and test environment

This is a text-expansion experiment. Actual TIPO checkpoint inference and browser
behavior were tested. No image generation or NovelAI API requests were made.

| ComfyUI backend | Frontend | Execution and browser result |
| --- | --- | --- |
| v0.3.26 official source archive | 1.11.8, pinned by that release | PASS |
| v0.3.64 official source archive | 1.27.10, pinned by that release | PASS, with independent frontend startup error noted below |
| Resident 0.41.0, commit `5cc4b870094c89090db54bd8d12649dcaf89f2ba` | 1.55.17 | PASS |

All servers used the existing Linux Python 3.12.14, PyTorch 2.14.0+cu130 and
installed tokenizer/safetensors stack. Thus this proves old ComfyUI and frontend
API compatibility with that runtime. It does **not** prove old PyTorch wheels,
old Python versions, Windows, or every intervening ComfyUI version. The native
model still requires `torch>=2.5` and the other declared requirements.

Old source archives came from official
[v0.3.26](https://github.com/comfyanonymous/ComfyUI/tree/v0.3.26) and
[v0.3.64](https://github.com/comfyanonymous/ComfyUI/tree/v0.3.64).
They ran on isolated ports 8194 and 8195, shared only the model data and TIPO
extension, and used separate user/output directories. The v0.3.64 pinned
workflow-template package was supplied through an isolated Python import path.
The resident ComfyUI was not downgraded. Old servers were terminated after testing.

## Reproduced frontend faults and fixes

1. In frontend 1.11, core force-input conversion runs after extension node
   creation. Creating/removing character sockets before it finishes produced
   duplicate STRING inputs and execution failures. Initial creation now leaves
   socket conversion to core and defers synchronization.
2. Synchronizing while graph configuration is incomplete reset Primitive STRING
   values. Configuration now schedules refresh after graph restoration finishes.
3. Some frontend revisions omit the hidden STRING backing widget that
   PrimitiveNode expects. Missing backing widgets are created with zero UI size;
   the user still sees STRING connection sockets.
4. Widget arrays vary between frontend revisions, particularly due to converted
   widgets and seed controls. TIPO saves values by widget name and restores them
   once during configuration. Normal subsequent refreshes do not overwrite edits.
5. Recreated sockets retain string widget configuration where the frontend
   exposes the core configuration API. Linked inactive sockets stay present,
   marked inactive, and preserve their indexes/links.

The example now uses four core Primitive STRING nodes and the TIPO output panel.
It no longer depends on newer StringMultiline/PreviewAny backend classes.

## Actual browser procedure

An isolated Playwright Chromium context opened each actual ComfyUI frontend.
The test loaded `examples/shared-interaction.json`, verified the input values,
changed the active count 2→3→1, and connected an extra STRING source and output.
It then serialized and structured-cloned the graph, reloaded it, and checked:

- Count and source strings survived; no duplicate prompt sockets appeared.
- Linked inactive character inputs/outputs remained connected and labeled.
- Inactive connections retained their original socket indexes; the JavaScript
  fixture additionally checks shrinking and re-enabling slots.
- A fresh load of the example queued a real 48-token-per-character TIPO request
  through the frontend's normal queue path.
- Character results appeared in the original/added/excluded/expanded UI panel.

Saved local evidence is under ignored `validation/compatibility/ui-*.json`,
`ui-*.png`, browser test scripts and server logs. The current browser test had
no uncaught page errors. The old v0.3.26 test also had none.

Frontend 1.27.10 logs `Cannot read properties of undefined (reading 'id')` while
registering its own canvas navigation settings. The same error was reproduced
with TIPO's browser scripts replaced by empty scripts in that isolated test;
its stack is in core `onChange`/settings registration. The graph, queue and
results test still passed. This is recorded rather than claimed as an error-free
frontend session. The core-only stack is saved in
`validation/compatibility/0.3.64-core-startup.json`.

## Follow-up output protection

Protection still operates after native inference; this update does not claim to
train or improve the model's semantic understanding. New checks reject:

- Conflicting recognized tags among already accepted additions.
- Additions conflicting with standalone shared posture, setting or gaze tags.
- Unrequested comic panels and speech/thought bubbles.
- Recognized open-mouth emoticons when a closed-mouth expression is supplied or
  already accepted.

Six real requests repeated the previous cup/hug/dance protection OFF/ON cases.
The decoded model outputs were unchanged from the recorded baseline. That is a
regression result, not evidence of new quality gains. The original source strings
and weights were preserved, and active/inactive outputs finalized correctly
through the installed RS pack.

An additional real request used shared `sitting, indoors, looking away` and two
different offer/receive role strings. Comparing the **same decoded tags** with
the previous committed filter and this filter showed these newly rejected tags:

| Slot | Previously accepted, now rejected |
| --- | --- |
| 1 | `3koma`, `comic`, `looking at another`, `speech bubble` |
| 2 | `4koma`, `:d`, `comic`, `looking at another`, `speech bubble` |

The `:d` in slot 2 conflicted with an already accepted `closed mouth`. Inputs,
raw decoded tags, final strings and rejection reasons are saved in ignored
`validation/shared-constraints.json`. This narrow example shows source/scene
protection working; it is not a general interaction accuracy score.

## Regression checks and limits

42 Python unit checks and both JavaScript regression files pass. The JavaScript
fixtures cover deferred core conversion, missing STRING backing widgets, named
restoration without overriding interactive edits, inactive links and stable
output indexes. The Python suite covers shared constraints and conflicts between
accepted additions in addition to the earlier inference/control/pipe checks.

```sh
python -m unittest discover -s tests -v
node tests/test_interface.mjs
node tests/test_sockets.mjs
```

Unknown synonyms, negation, ambiguous participant roles and new kinds of
incompatibility remain possible. This update does not correct contradictory
user input, invent a role plan, or prove better generated image interactions.
Full-checkpoint parity, unload/reload and cancellation results remain in
[the native validation report](validation.md).
