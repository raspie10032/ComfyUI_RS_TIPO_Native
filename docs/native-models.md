# Native model selection and first-use downloads

Use `model_family` on **RS TIPO Native (Experimental)**. Inputs, numbered STRING
outputs and the RS building pipe are shared. Model selection changes the
inference backend; it never chains a translator and a separate prompt model.

| Choice | Runtime | Input language | First-use download |
|---|---|---|---|
| `gemma_tipo` | ComfyUI Gemma 4 + one selected TIPO LoRA | Korean/English → English output, experimental meaning accuracy | Q4 base + tokenizer + adapter, about 3.27 GiB |
| `tipo_2_1` | Native Kohaku model through ComfyUI ModelPatcher | English | Official 1B model/config/tokenizer, about 1.85 GiB |

Gemma's `language_model=auto` reuses `gemma4-e2b-native-bf16.safetensors` if
present. Otherwise it fetches the Q4 base. Explicit `heretic_ara_q4` selects the
downloadable base. Arbitrary Gemma families and already merged TIPO weights are not supported.
The default is `presampling-fresh-r8-v1` with its matching instruction and
text LoRA strength 0.25, temperature 0. Strength 0 runs the same instruction
without adapter patches or an adapter download. Local candidates do not silently
replace a saved selection. TIPO 2.1 hides/ignores Gemma-only base and adapter choices.
Select `tipo_lora=scale-20pct-001` for the published 2026-11-03 20% training
candidate (rank 8, alpha 16). Its two pinned files download on first execution,
about 45.3 MiB, and are verified before use. This experimental release is
**CC BY-NC 4.0 (non-commercial)**. An existing `local:scale-20pct-001` bundle
contains the same weights; saved local selections continue to work.

## Sources and locations

- [Gemma Q4 base](https://huggingface.co/mradermacher/gemma-4-E2B-it-heretic-ara-GGUF),
  revision `c02d14dccce74f69566f8b815e648cc9eba243f7`:
  `models/text_encoders/rs_gemma_q4/`.
- [Gemma tokenizer](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara),
  revision `c9a1d4c031981f14d86eeb0c7d87de7fafd34513`, in the same folder.
- [Legacy pilot-002 adapter](https://huggingface.co/raspie/gemma4-tipo-lora-pilot-002),
  revision `c31d210508c0dfb9f3e08981baa73f2e06eafd0a`:
  `models/loras/rs_gemma_tipo/pilot-002/`.
  Other published adapters are pinned in `tipo_native_lora.py`.
- [20% training candidate](https://huggingface.co/raspie/gemma4-tipo-lora-scale-20pct-001),
  revision `3668df754885175408ac247a9e85309fa6528486`:
  `models/loras/rs_gemma_tipo/scale-20pct-001/`.
- [TIPO 2.1 official checkpoint](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M),
  revision `f5a318524a4ab30cdbbf51816cf406170f454e65`:
  `models/tipo/TIPO-v2.1-1B-A200M/`.

Only files needed by the chosen backend are fetched. Text expansion does not
download the vision projector. No llama.cpp or prebuilt llama.cpp wheel is
required. Weight licenses remain those of the linked publishers; see
[references](references.md) for adapted source attribution.

## Progress, resume, integrity and offline operation

The node shows filename, percent, bytes and verification/loading/generation
states. Transfers use Hugging Face Hub's cache, file locks and partial downloads.
Restarting an interrupted download reuses its cached partial file. ComfyUI's
interrupt is checked during Python progress callbacks; native transfer code
may not stop immediately between callbacks.

Revision, size and SHA-256 are pinned. Already validated unchanged files reuse
an in-process signature cache; a new process rechecks files. Invalid existing
files produce an error naming the file instead of being overwritten. Move the
damaged file aside and retry. Missing-space checks run before download.

Set `RS_MODELS_OFFLINE=1` or `HF_HUB_OFFLINE=1` to require local files. An
incomplete offline installation lists its missing files. Private `local:`
adapters require their exported manifest and files on disk; they cannot be
downloaded from the developer's computer.

## What has and has not been validated

On 2026-11-03, the public `RSTIPOPrompt.expand` function ran English tags/prose
with both backends and Korean tags/prose with Gemma TIPO. A Korean TIPO 2.1
request was rejected before download/inference. Fixed-revision small-file
downloads were tested against the public repositories; complete weight sets
already on disk were used for inference. Transfer interruption, offline use,
cache reuse and corruption checks also have isolated regression tests.

Browser checks cover backend selection, save/reload, hidden Gemma-only settings,
distinct prompt sockets and absence of JavaScript errors. These are execution
checks, not a semantic quality score. TIPO 2.1 prose still invents appearance
and loses actions; use its tag mode as the starting point. Gemma TIPO still has
Korean meaning/omission limitations described in the existing evaluation.

Neither model file size nor successful native loading establishes the VRAM
requirement or useful output quality. See the published adapter's model card
for its training sources, non-commercial license and measured limitations.

The [new default adapter](https://huggingface.co/raspie/gemma4-tipo-lora-presampling-fresh-r8-v1)
is pinned in `tipo_native_lora.py`, at `models/loras/rs_gemma_tipo/presampling-fresh-r8-v1/`.
See [instruction, evaluation and source terms](gemma-presampling-defaults.ko.md).
