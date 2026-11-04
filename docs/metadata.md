> Historical standalone TIPO implementation/test record. The current active node uses Gemma + TIPO LoRA. See [current usage](usage.ko.md) and [Gemma validation](gemma-tipo-validation.md).

# Optional TIPO rating and quality conditioning

Implemented and checked 2026-11-02. This is the first follow-up to the local
2026-11-02 multi-model role study: pass the metadata that TIPO already understands
before changing its training.
The implementation is in `metadata.py`, `control.py`, `expansion.py`,
`nodes.py` and `web/interface.js`.

## What is passed

The [official TIPO v2.1 model card](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M)
defines `rating` and `quality` as optional fields before `target:`. For example:

```text
quality: masterpiece, newest
rating: safe
target: <|long|> <|long_to_tag|>
long: <shared action context>
tag: <common prompt>, <this character's prompt>,
```

`rating` offers `safe`, `sensitive`, `nsfw` and `explicit`; the official explicit
mapping is passed as `nsfw, explicit`. `quality` offers `masterpiece`,
`best quality`, `great quality`, `good quality`, `normal quality`, `low quality`
and `worst quality`. Advanced `quality_era` appends `newest`, `recent`, `mid`,
`early` or `old` to the quality field. An era requires a selected quality.

Every option defaults to `unspecified`. With those defaults no metadata lines
are inserted. There is no automatic rating selection. Each active character
receives the selected conditions in its own request; common expansion receives
them when enabled. Other characters' private appearance prompts stay separate.

The header remains readable at normal attention weight even when common prompt
weight is zero. It does not acquire a character owner. Token offsets for common,
character and shared-action attention are computed after inserting the header.
The header is not appended to the final image prompt. Reports expose the actual
metadata in both the summary and each target's raw model input.

`excluded_tags` still controls candidate tokens during decoding. Rating and
quality are model guidance, not exact tag exclusion or a guarantee of output
quality. This change does not implement the subsequent global-addition or
action-contradiction improvements proposed in the study.

## Verification

Python tests cover the official vocabulary, invalid settings before model
allocation, unchanged default prompts and ownership, header attention with a
zero common weight, every target's propagation, original text/weight
preservation, exclusions and appended widget order for older saved workflows.

Actual TIPO execution was checked on the RTX 5060 Ti using separate local test
servers: ComfyUI 0.41.0 with frontend 1.55.17 and ComfyUI 0.3.26 with frontend
1.11.8. Both used the installed ComfyUI Python environment and official pinned
TIPO weights. This is Linux runtime evidence; it is not a Windows execution test.

Each server processed five cases: omitted fields, explicit unspecified fields,
rating only, quality only and rating + quality + era. Each case expanded one
common prompt and two characters: 30 target completions across both versions.
The same seed and sampling controls were used. Omitted and explicit unspecified
cases had identical model inputs, generated token IDs and final outputs. All
selected cases reported the expected metadata on all three targets, preserved
the source prefixes and kept the inactive tenth character out of the requests.
These were text expansions, not new diffusion images or an image-quality study.

The browser check loads an existing example, selects metadata, serializes and
reloads it, checks the resulting API inputs, queues the actual frontend prompt
and receives the metadata in the result panel. The socket-layout check also
exercises different active character counts, connected seed visibility and
STRING preservation. Local evidence lives under the ignored `validation/`
directory; model weights and machine-specific histories are not distributed.

The current frontend's queued request was confirmed successful in server
history after the test observer initially matched `/prompt` but missed the
newer `/api/prompt` route. Its completed report was replayed into the panel;
the job was not restarted. The old frontend check received its report over the
live websocket. The reproduction tool now accepts either route and optionally
an existing prompt ID as a second argument to inspect a completed run.

Reproduce on explicitly selected, idle test servers:

```sh
python -m unittest discover -s tests -v
node tests/test_interface.mjs
node tests/test_sockets.mjs
node tests/test_download_status.mjs
python tools/validate_metadata.py TEST_PORT
node tools/check_metadata_ui.cjs TEST_PORT
node tools/check_socket_layout.cjs TEST_PORT metadata
```

The browser tools require Playwright and a locally available Chromium executable;
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` can select it. They open an isolated headless
browser and do not control the user's tabs. The live validators queue text
inference only on the supplied server, and reject an unrelated busy queue.
