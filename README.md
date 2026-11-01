# ComfyUI RS TIPO Native

Independent experimental project for running **TIPO v2.1** using native PyTorch
and ComfyUI model management, with no llama.cpp, GGUF or Transformers in the
inference path, with experimental input-span attention control.

GitHub: https://github.com/raspie10032/ComfyUI_RS_TIPO_Native

## Documentation

- **[한국어 설치·사용법 — 입력 연결, 역할 분담, 결과 확인](docs/usage.ko.md)**
- [References — original sources, adopted features and license boundaries](docs/references.md)
- [Workflow PNG — download original and drop into ComfyUI](examples/shared-scene-workflow.png)
- [Example JSON](examples/shared-interaction.json)
- [Tested legacy/current ComfyUI versions and limits](docs/compatibility.md)
- [Automatic role context and socket repair verification](docs/automatic-context.md)

## STRING interface

Search for **RS TIPO Prompt (Experimental)**. Connect common and character
prompts as separate STRING sockets. Choose 0–10 characters. The outputs are an
RS building pipe, the common STRING, and individual character STRINGs. The pipe
and STRINGs contain the same expanded texts; they do not generate separately.
An optional incoming RS pipe retains LoRA/mask/position/reference/camera settings.
STRING outputs work without installing the MultiCharacter pack. Pipe output
requires `ComfyUI_RS_MultiCharacter_Sampler` and at least one character.

The character count is authoritative: reducing 5 to 3 excludes slots 4–5 from
execution. Connected inactive sockets remain marked inactive to preserve links;
inactive STRING outputs are empty. Expand the advanced toggle for inference
settings. A results panel shows original, added, excluded (with reasons) and
expanded text per slot.

**The scene and participant roles come from the existing inputs.** Put the
shared situation/interaction in `common_prompt` and each participant's appearance
and action in its own character STRING. There is no separate interaction input.
The node automatically builds caption context from the common scene and
recognized action/position clauses across active characters, using TIPO's learned
`long_to_tag` format. Each target also receives its own complete prompt with
separate attention weights. Other participants' private appearance stays out of
that target's inference input. The results panel shows the derived context.

This action projection is conservative English tag matching, not general
relationship understanding. Unrecognized clauses stay in their own character
input. The node does not invent missing roles or screen positions, and does not
append the gathered role list to the final scene. Scene expansion defaults OFF;
turn it on to experiment with adding scene tags. All additions need review.
Source protection defaults ON: it removes recognized new appearance/style and
composition changes, participant-specific scene additions and known action or
contact-target conflicts. Explicit source weights stay in the output, while
TIPO reads their unweighted tag contents. This is conservative lexical filtering,
not reliable semantic planning; it may reject useful details or miss conflicts.
[Actual paired tests and limitations](docs/source-protection.md).
[Interface validation and observed limitations](docs/interface.md).
Load [the shared interaction example](examples/shared-interaction.json) in ComfyUI.
It uses core Primitive STRING nodes and the TIPO results panel, and performs text
inference only. It does not require newer Text/Preview backend nodes.
Legacy and current frontend checks, and their limits, are recorded in
[compatibility and follow-up validation](docs/compatibility.md).

## Current scope

- Kohaku decoder, grouped-query attention, QK normalization, RoPE and sparse MoE.
- Official safetensors checkpoint names, strictly loaded with no ignored tensors.
- ComfyUI `ModelPatcher` for loading/offloading; one model cached in host memory.
- Request-local KV cache and sampling generator, including deterministic seed 0.
- Completion, baseline expansion, controlled expansion and shared pipe expansion.
- Token-owner masking during prefill and decode; common/character attention weights.
- Independent cache per character; other participants share recognized actions, not private appearance.
- Prompt-only edits of shared draft/final pipes and modern NAI contexts.
- Cancellation and progress during decoding; context overflow is an explicit error.

**Still experimental:** isolation keeps private character appearance separate, but it
does not stop TIPO inventing contradictory attributes from its learned priors.
Complete semantic conflict prevention and model quantization remain unfinished. This is an experimental release.
Separate character prompts must not be flattened into the baseline node.

The model currently requires full residency during a request (about 1.85 GiB of
FP16 parameters, plus runtime/cache memory). ComfyUI can offload the whole model
between requests; partial per-layer offloading is not supported yet. This project
does not claim to be faster than llama.cpp.

## Setup

Use the Python environment that runs ComfyUI. Install `requirements.txt` if needed.
Model download is explicit and happens separately from import or node execution:

```sh
python -m pip install huggingface_hub
python tools/download_model.py /path/to/ComfyUI/models/tipo/TIPO-v2.1-1B-A200M
```

The downloader pins revision `f5a318524a4ab30cdbbf51816cf406170f454e65` of
[the official model](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M).
Only model/tokenizer data is downloaded; remote Python code is not executed.
The existing GGUF cannot be used as a safetensors checkpoint.

Place or link this project under `ComfyUI/custom_nodes/`, restart ComfyUI, and
search for **RS TIPO Native**. All nodes load the model internally.
`RS TIPO Native Expand` accepts a tag string and preserves the original string
while appending the generated first tag field. Check `info` for the complete raw
completion and finish reason. `length` and `max_new_tokens` are different:
the former is a learned target, the latter is a hard output limit.

## Controlled expansion and character pipes

`RS TIPO Native Controlled Expand` takes a common scene and one character. The
common and character source tokens receive separate attention multipliers. A
multiplier adds `log(weight)` to attention scores before softmax in every layer;
0 blocks that source. 1 is neutral. Generated history remains at weight 1.
These controls change TIPO's internal calculation, independently of image prompt
syntax such as `(blonde hair:1.2)`, which is preserved verbatim in the result.

`RS TIPO Native Pipe Expand` reads an RS building pipe, final character pipe or
modern NAI context. It expands active character slots independently; negatives,
literal text slots, masks, coordinates, LoRAs, references and camera settings are
preserved. Connect it **after Character Prompts**; LoRA/Position can be before or
after it. A draft must have a known active count. Finalize it before sampling as
usual. This editor does not run image generation or call NovelAI.

```text
Character Prompts -> [LoRA / Position] -> TIPO Pipe Expand -> Finalize -> Sampler
Common scene STRING -------------------> | common_prompt STRING -------> Sampler
```

An empty common input inherits the pipe's scene. A draft with no source scene
cannot store the common prompt: use this node's `common_prompt` STRING output on
the sampler too. Finished pipes/contexts store the edited scene. `expand_common`
is off by default: TIPO can invent hair/eye/clothing tags in a common caption.
Enable it to test full-scene expansion and inspect the generated tags.

Each character sees the **original** common caption, not another character's
completion or an expanded common caption. Other-character text is excluded before
tokenization, with an independent KV cache. Owner masks also block indirect
cross-owner attention during prefill. At strict zero/blocked settings MoE tokens
dispatch by owner too, avoiding FP16 rounding changes caused by blocked tokens
changing another owner's expert batching shapes.

The 0.5 common / 1.5 character defaults are experimental starting settings, not
an established quality optimum. `info` records raw output, additions, source,
weights, token IDs, generation time and finish reason for every slot. Compare
1/1 with other settings using a fixed seed; inspect the actual content.

For numerical comparison use `RS TIPO Native Completion`, the official TIPO
completion format, and temperature 0. This is a completion model, not a chat model:

```text
target: <|long|>
tag: 1girl, garden,
```

## Verification

```sh
python -m unittest discover -s tests -v
```

Full-checkpoint parity and ComfyUI API smoke results belong in
[`docs/validation.md`](docs/validation.md). Unit tests using a tiny synthetic model
do not establish real-model quality or runtime compatibility.
The attention/pipe experiment is documented in
[`docs/owned-attention.md`](docs/owned-attention.md).

## Upstream and licenses

The architecture is adapted from the official
[KohakUwULLM HF export](https://github.com/KohakuBlueleaf/KohakUwULLM/tree/d4a52f77c683e6a0622d76a4f42752d97ced97ba/src/kohakuwullm/export/hf),
whose code matches the pinned model repository's architecture source byte for byte.
See `KohakUwULLM-APACHE-LICENSE` and `NOTICE` for upstream attribution.
The pinned model card identifies model weights as Kohaku License 1.0, separately
from the Apache-2.0 architecture code. See [references and adaptation scope](docs/references.md).
New project code is licensed under **GPL-3.0-only**; see [LICENSE](LICENSE) and
[COPYING](COPYING). Adapted upstream files retain the Apache-2.0 attribution.
No model weights are stored in this Git repository.

The trial does not yet offer a user-specified unwanted-tag/blocklist input.
Automatic source protection only covers the documented recognized conflicts.
