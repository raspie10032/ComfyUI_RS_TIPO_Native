# ComfyUI RS TIPO Native

**0.1.1 experimental:** generation-time excluded tags and automatic first-run model download.

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
- [Role-focused additions: paired image study and limitations](docs/role-additions.md)
- [Complex interactions: three participants, hair braiding and helping up](docs/role-complex.md)

## STRING interface

Use **RS TIPO Prompt (Experimental)** for new workflows. It is the only
non-deprecated node in this pack. The four early research nodes are hidden from
the default node picker and retained for loading saved workflows;
[legacy behavior and migration](docs/legacy-nodes.md).

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
- One user-facing node for common-only and separate character STRING expansion,
  with an optional RS building pipe. Four legacy research nodes remain loadable.
- Token-owner masking during prefill and decode; common/character attention weights.
- Independent cache per character; other participants share recognized actions, not private appearance.
- Prompt-only edits of shared draft/final pipes and modern NAI contexts.
- Cancellation and progress during decoding; context overflow is an explicit error.
- Optional `excluded_tags`: reject candidate tokens during generation, not a final
  remove pass. Case, underscores and whitespace are normalized. The results panel
  distinguishes generation blocking from existing source protection after decoding.
- First execution downloads missing model files with progress and SHA-256 verification.

## Excluded tags (0.1.1)

Enter comma/newline-separated tags, e.g. `3d, white_shirt, long sleeves`, in
`excluded_tags`. `long_sleeves` and `LONG  SLEEVES` are equivalent. This applies
to all active generated outputs; empty keeps the previous decoding path.
The sampler rejects a next-token candidate when its decoded tag matches the
normalized list, then samples another candidate. Rejected tokens are never
appended to the output or KV cache. It does not add the unwanted tags to TIPO's
positive prompt, fine-tune the model or remove them after generation.

Source strings and weights are authoritative. An excluded tag already present
in an active source is an error before loading the model. Fix one side instead
of silently rewriting the source. This is lexical control: synonyms and free-form
descriptions may still appear. Blocking a short prefix may constrain longer tags,
and alternatives can be awkward; it is not full semantic negative prompting.
[Implementation, real-model comparisons and limits](docs/exclusion-0.1.1.md).

**Still experimental:** isolation keeps private character appearance separate, but it
does not stop TIPO inventing contradictory attributes from its learned priors.
Complete semantic conflict prevention and model quantization remain unfinished. This is an experimental release.
Keep separate character prompts on their numbered STRING inputs.

The model currently requires full residency during a request (about 1.85 GiB of
FP16 parameters, plus runtime/cache memory). ComfyUI can offload the whole model
between requests; partial per-layer offloading is not supported yet. This project
does not claim to be faster than llama.cpp.

## Setup

Use the Python environment that runs ComfyUI. Install `requirements.txt` if needed.
No project-specific prebuilt wheel or C++/CUDA compilation is required. Inference
uses ComfyUI's PyTorch installation (`torch>=2.5`), plus `safetensors>=0.4` and
`tokenizers>=0.21`. These dependencies may themselves install standard binary
wheels; they are not a separate RS TIPO runtime.
Missing model files download automatically on the first node execution, into
`ComfyUI/models/tipo/TIPO-v2.1-1B-A200M` (about **1.85 GiB**). Existing complete
installs, including configured extra `tipo` paths, are reused offline. Progress
is shown during transfer. The node panel also shows the current filename,
percentage, received/total MiB, and SHA-256 verification/loading status.
Interrupt/retry retains the resumable download cache.
There is no download at extension import. Install Python dependencies using
ComfyUI's own Python:

```sh
python -m pip install -r requirements.txt
```

Optional manual prefetch for offline machines uses the same downloader:

```sh
python tools/download_model.py /path/to/ComfyUI/models/tipo/TIPO-v2.1-1B-A200M
```

The downloader pins revision `f5a318524a4ab30cdbbf51816cf406170f454e65` of
[the official model](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M).
Only model/tokenizer data is downloaded; remote Python code is not executed.
The existing GGUF cannot be used as a safetensors checkpoint.

Place or link this project under `ComfyUI/custom_nodes/`, restart ComfyUI, and
search for **RS TIPO Prompt**. The node loads the model internally.

## One prompt node

- **Common prompt only:** choose `character_count=0`, turn `expand_common` ON,
  connect a common STRING and use the common STRING output. OFF passes it through.
- **One or more characters:** choose 1–10, connect the common scene and each
  character STRING. Each character keeps its own role, appearance and cache.
- **RS pipeline:** use the same node's `building_pipe` output. An optional incoming
  pipe keeps LoRA, mask, position, reference and camera settings; the connected
  STRING table supplies the prompts. Add literal text slots downstream.

```text
Common + character STRINGs -> RS TIPO Prompt -> LoRA / Position -> Finalize -> Sampler
                                  | common STRING -----------------------> Sampler
```

Advanced settings expose the token budget, seed, temperature and common/character
source-attention weights. `length` is a learned target; `max_new_tokens` is a hard
output limit. The 0.5 common / 1.5 character defaults are experimental settings,
not an established quality optimum. Original prompt weights remain verbatim;
these attention controls affect TIPO's calculation rather than image sampling.

The node's results panel shows original, added, rejected and expanded tags.
The four low-level completion/expansion nodes now carry ComfyUI's `DEPRECATED`
flag. Their class IDs, inputs, outputs and inference behavior are preserved for
old graphs, including the legacy pipe-only editor. If deprecated nodes are
explicitly shown in ComfyUI settings, they appear under `RS/TIPO Native/Legacy`.
See [legacy nodes and migration](docs/legacy-nodes.md) for differences that prevent
silently substituting one node for another.

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

Generation-time exclusions and automatic source protection are separate controls;
neither provides complete semantic conflict prevention.
