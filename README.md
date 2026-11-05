# RS TIPO Native — experimental

TIPO research is the work of **the TIPO paper's authors**. The official TIPO
model and Kohaku/KGen implementations are published by **KohakuBlueleaf
(Shih-Ying Yeh)** and their respective contributors.
Thank you for making these foundations available. This independent extension
adds ComfyUI-native execution and prompt/pipe integration, and also supports a
separate Gemma-based TIPO LoRA backend.
[TIPO paper](https://arxiv.org/abs/2411.08127) ·
[Official TIPO model](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M) ·
[KohakUwULLM](https://github.com/KohakuBlueleaf/KohakUwULLM) ·
[KGen](https://github.com/KohakuBlueleaf/KGen).
See [sources, adaptations and licenses](docs/references.md) for the contribution boundaries.

Choose **Gemma TIPO** (a Gemma base with a TIPO LoRA applied) or **TIPO 2.1**
as a standalone ComfyUI native PyTorch model. The
`RS TIPO Native (Experimental)` node reads a common prompt and up to ten character STRINGs,
then returns common/character STRINGs and an optional RS building pipe.

These are alternative backends selected by `model_family`, never a two-model
translation chain. Korean input requires Gemma TIPO; TIPO 2.1 accepts English.
[한국어 사용법](docs/usage.ko.md) · [Model sources and download behavior](docs/native-models.md)

**Version 0.2.14:** keep result and download-status panels inside the node when
display scaling, browser/canvas zoom or node width changes. Long summaries and
results scroll internally; selectors and Copy stay on one row. Refresh the
browser after updating. No new runtime dependencies or generation changes.

Version 0.2.13: detach explicit SD/NAI numeric weights before Gemma expansion
and restore them afterward. Weight punctuation changes no longer discard a
field's generated details. Restore missing weighted phrases and remove duplicate
tags while retaining new descriptions. Korean weighted phrases use one extra
batched translation call on the same Gemma instance; failed translation preserves
the original weighted phrase with a visible warning. No new runtime dependencies.
See [weight handling, validation and limits](docs/gemma-weight-handling.ko.md).

Version 0.2.12: default to the published `presampling-fresh-r8-v1` adapter
with its matching instruction, **text LoRA strength 0.25** and greedy temperature 0.
Pinned first-use download with size/SHA-256 checks. Strength 0 uses the same
instruction without adapter patches or an adapter download. No new runtime
dependencies. Experimental non-commercial adapter; source terms remain applicable.
See [defaults, evaluation and limits](docs/gemma-presampling-defaults.ko.md).

Version 0.2.11: audit the entire Gemma response path. Repair unambiguous JSON
framing, retry malformed/empty responses once, and keep only explicitly owned
character fields. After exhausted format repair, restore affected source fields;
failed tag rendering retains its own caption. Recovery is visible in the result
panel and model details. GPU/runtime errors and user cancellation still propagate.
No new runtime dependencies or model-weight changes.
See [audit findings, validation and limits](docs/gemma-runtime-audit.ko.md).

Version 0.2.10: blank common/character strings retain their own source and
continue without a contract retry. Valid neighboring descriptions stay intact;
restored fields are recorded in model details as `source_restored_fields`.
This prevents an empty field from stopping the workflow; it does not guarantee
new detail or translation for that field. Missing/wrong slot keys and non-string
values still require bounded repair. See [behavior and limits](docs/gemma-empty-fields-fix.ko.md).

Version 0.2.9: remove intermediate-planning grammar and language rejection.
Nonempty tag lists, labels and short fragments continue to tag rendering;
natural-language output keeps the returned wording. Contract retries only repair
the reported error without repeating expansion. JSON structure, character slots,
source weights and user-specified exclusions remain checked.
See [validation and limits](docs/gemma-planner-text-fix.ko.md).

Version 0.2.8: recover broken common-string quotes and separators when the
complete character object is independently valid. Keep the common description,
including text after an early quote, and preserve every character slot. Other
JSON syntax errors get one formatting-only retry without repeating the expansion
task. Incomplete or ambiguous structures still fail with raw diagnostics.
Generated content still passes through with only tag deduplication, source
preservation and the configured addition budget.
See [JSON repair and limits](docs/gemma-json-repair.ko.md) and
[deduplication behavior](docs/gemma-expansion-retry-fix.ko.md).

**Version 0.2.2:** reduce native Gemma GGUF memory and repeated LoRA work.
Keep the per-layer embedding (PLE) table in CPU RAM and transfer selected rows;
use ComfyUI's native LoRA output calculation for supported Linear adapters;
reuse Q4_K/Q5_K dequantization buffers. No new runtime dependencies or GPU cache.
The measured Q4 setup used about **1.78 GiB peak allocated VRAM for ten
characters**, compared with about 3.60 GiB before these optimizations. These are
process-local PyTorch measurements, not total GPU use; CPU RAM still holds PLE.
See the [measurements, limits and reproduction guide](docs/gemma-inference-optimization.ko.md).

Version 0.2.1 introduced prose repair and field-level diagnostics;
version 0.2.9 removes its remaining grammar rejection. See the [repair and 2–10 character evaluation](docs/gemma-output-repair.ko.md).

Version 0.2.0 unified native TIPO 2.1 / Gemma TIPO, verified automatic
downloads for `scale-20pct-001`, and an inspectable result panel.
See the [release notes](CHANGELOG.md).

## Install and use

Comfy Registry: **[RS TIPO Native](https://registry.comfy.org/publishers/raspie10032/nodes/rs-tipo-native)**
(`rs-tipo-native`, publisher `raspie10032`). Search this name in a Registry-enabled
ComfyUI Manager, or use `comfy node install rs-tipo-native`. Restart ComfyUI and
refresh the browser after installation. The Registry compatibility baseline is
**ComfyUI 0.41.0 or newer**; Gemma also requires its native Gemma 4 generation APIs.
Runtime checks report missing APIs before downloading model weights.

For a manual installation:

Install this repository in `ComfyUI/custom_nodes`, install `requirements.txt`
using ComfyUI's Python, and restart ComfyUI. Gemma TIPO requires a current
ComfyUI with native Gemma 4 text generation; TIPO 2.1 uses this extension's
vendored Kohaku architecture and ComfyUI ModelPatcher. No llama.cpp wheel or inference server is
used. An older ComfyUI without that API reports the missing feature before a
large download; it is not claimed to support native Gemma inference.

1. Add **RS TIPO Native (Experimental)** (the same `RSTIPOPrompt` ID as the
   earlier RS TIPO Prompt). Connect the common and numbered
   character STRING inputs. Set the active character count; stale higher
   numbered inputs do not participate.
2. Select **model_family**: `gemma_tipo` or `tipo_2_1`.
   For Gemma TIPO, the published `presampling-fresh-r8-v1` adapter is the
   default with its matching instruction, **TIPO LoRA strength 0.25**, and greedy
   temperature 0. It downloads on first execution. `pilot-002`, `pilot-001`,
   `image-30k` and `scale-20pct-001` remain selectable. Strength controls only
   the text adapter; 0 uses the same instruction without adapter patches/download.
   TIPO 2.1 hides and ignores Gemma-only choices and strength.
   Local bundles appear as `local:<name>`. The previous
   `local:presampling-fresh-r8-v1` selection keeps the same instruction.
   See [instruction + weak LoRA defaults](docs/gemma-presampling-defaults.ko.md).
   The [2026-11-03 20% training candidate](https://huggingface.co/raspie/gemma4-tipo-lora-scale-20pct-001)
   downloads on first execution when `scale-20pct-001` is selected. It is
   experimental and **CC BY-NC 4.0 (non-commercial)**. Its weights match the
   existing `local:scale-20pct-001` bundle; it does not replace the default.
3. Select **tags** or **natural_language**. Both outputs are English. Korean
   input is interpreted by the same adapted Gemma. TIPO 2.1 is English-only and
   its natural-language output may invent attributes or lose character roles;
   prefer Gemma TIPO for coordinated character prose.
4. Enable **expand_common** to also rewrite the common prompt. When disabled,
   it is returned exactly as supplied, including Korean.
5. Use the individual STRING outputs, or connect the building pipe to the RS
   MultiCharacter Sampler. An incoming pipe retains positions, masks, LoRAs,
   references and other non-prompt state.

Examples: [New Gemma default (automatic download)](examples/gemma-tipo-presampling.json) ·
[Gemma Korean (legacy adapter)](examples/gemma-tipo-korean.json) ·
[TIPO 2.1 English tags](examples/tipo-2.1-english.json) ·
[Gemma TIPO scale-20pct-001 (automatic download)](examples/gemma-tipo-scale-20pct.json) ·
[Previously installed local weights](examples/gemma-tipo-trained-local.json).

The result panel offers **Output**, **Source**, and **Model details / raw output**
views, with all prompts or one selected slot and a Copy button. Applied adapter
identity, rank, weight checksum and LoRA target count appear in model details.
This makes it easy to inspect useful expansion without treating successful
execution as a guarantee of meaning accuracy.
See the [trained-node execution and UI checks](docs/trained-node-validation.ko.md)
for actual outputs, the tag-format retry and remaining omissions.

The advanced **language_model** selector chooses the Gemma *base*, not a second
translation model. `auto` uses a recognized local native E2B base when present,
otherwise the pinned Q4 base downloads on first execution. `heretic_ara_q4`
forces that native Q4 route. Models are downloaded into ComfyUI model folders;
TIPO adapters go under `models/loras/rs_gemma_tipo`. Download progress appears
on the node. TIPO 2.1 downloads from the official pinned KBlueLeaf repository
into `models/tipo/TIPO-v2.1-1B-A200M` (~1.85 GiB).
Missing downloads resume through Hugging Face Hub, with SHA-256 validation and
cache reuse. Existing invalid files are reported, not silently replaced.
Use `RS_MODELS_OFFLINE=1` or `HF_HUB_OFFLINE=1` for offline use after installing
the pinned assets.

## Execution and limits

- Gemma+TIPO reads all active inputs together and returns numbered prose fields.
  This retains the relationship between participants instead of expanding each
  input with no knowledge of the scene.
- Natural-language mode uses those prose fields directly. JSON is only an
  internal routing format; the user receives STRINGs.
- Tag mode uses the **same patched instance** to render each resulting caption
  as tags. Explicit color/appearance/side phrases from that caption seed the tag
  prefix. Generated tags are deduplicated without a caption-vocabulary allowlist
  or semantic content filter. Original English tag inputs remain unchanged at the beginning; new
  additions are limited by `max_added_tags`. Korean is translated rather than
  copied to the final tag output.
- Excluded phrases are blocked during token sampling, with underscore and space
  equivalence, and checked again before routing output. A source that explicitly
  contains a banned phrase is an error. This is literal phrase blocking, not a
  semantic guarantee against every synonym or grammatical variant.
- Explicit SD `(phrase:1.3)` and NAI `1.3::phrase ::` numeric weights are held
  by the node, detached for Gemma input, then restored per field after expansion.
  Missing weighted phrases are reinserted without discarding other generated
  details. Korean weighted phrases require one extra batched translation call;
  failed translation retains the original phrase and records a warning.
- Blank string fields retain their own source, with recovery recorded in model
  details; this may leave the field unexpanded or untranslated. Invalid/incomplete
  JSON and wrong character keys use one correction with
  the same model. If that fails, independently owned valid fields survive and
  affected fields revert to their own source. Ambiguous documents revert the
  entire scene to its sources. Failed tag conversion retains its own caption.
  Duplicate JSON keys, arrays of participants and unknown slot keys are never
  guessed into slots. Explicit exclusion conflicts in the source fail before
  loading the model. No fallback switches to another model.
- The adapters are experimental. They can omit details, add unwanted tags, or
  misunderstand an interaction. The result panel exposes source, final output,
  raw model calls, active base/adapter and applied LoRA target count.
- Prose and tag rendering are prompt generation, not diffusion conditioning.
  Legacy Kohaku attention fields are hidden for Gemma; TIPO 2.1 uses the
  owner-separated attention and source protection for tag expansion. Neither
  backend guarantees preservation of all meaning or attributes.

Legacy completion/expansion classes remain registered solely for older saved
workflows and research reproduction. They are hidden from new-node selection;
new workflows use the shared RS TIPO Prompt interface.

[Gemma TIPO I/O validation](docs/gemma-tipo-validation.md) ·
[References](docs/references.md) · [License](LICENSE)
