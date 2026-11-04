# Native TIPO sources and adaptation

Reviewed 2026-11-04. The shared **RS TIPO Native (Experimental)** node selects
either the official standalone TIPO 2.1 checkpoint or a separate Gemma TIPO
LoRA backend. The original TIPO backend is currently supported.

## Upstream credit and current execution paths

**TIPO research is credited to the paper's authors. The official model and
Kohaku/KGen implementations are published by KohakuBlueleaf (Shih-Ying Yeh)
and their respective contributors.** The paper credits Shih-Ying Yeh,
Yi Li, Sang-Hyun Park, Giyeong Oh, Xuehai Wang, Min Song, Youngjae Yu and
Shang-Hong Lai. The Kohaku architecture and KGen implementations are credited
to their respective upstream authors. We thank them for their public work.

- [TIPO paper](https://arxiv.org/abs/2411.08127): the upstream prompt
  presampling research. This independent extension adds native ComfyUI
  integration, character routing, source guards and RS pipe/UI integration.
- [`model_family=tipo_2_1`](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M/tree/f5a318524a4ab30cdbbf51816cf406170f454e65):
  loads the official checkpoint without retraining its weights. Its architecture
  is adapted from [KohakUwULLM](https://github.com/KohakuBlueleaf/KohakUwULLM/tree/d4a52f77c683e6a0622d76a4f42752d97ced97ba/src/kohakuwullm/export/hf);
  [KGen](https://github.com/KohakuBlueleaf/KGen) was consulted for the learned
  prompt format. ComfyUI ModelPatcher manages native PyTorch inference.
- `model_family=gemma_tipo`: loads a supported Gemma base with a selected
  TIPO-format LoRA. This is a separately adapted model, not the official TIPO
  checkpoint. Google DeepMind's [Gemma](https://huggingface.co/google/gemma-4-E2B-it),
  [p-e-w's derived base](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara)
  and [mradermacher's quantized distribution](https://huggingface.co/mradermacher/gemma-4-E2B-it-heretic-ara-GGUF)
  are separate upstream contributions.

The shared node runs the selected backend; it does not chain Gemma translation
and official TIPO inference. Model weight terms remain separate from this
extension's code license. See the license section below and the linked publishers.

## Gemma backend sources

The Gemma path uses:

- [Gemma E2B base](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara),
  pinned by the vendored RS Seasonal catalog. The LoRA configuration names this
  base; arbitrary Gemma family checkpoints are not claimed compatible.
- [TIPO pilot-002 LoRA](https://huggingface.co/raspie/gemma4-tipo-lora-pilot-002),
  revision `c31d210508c0dfb9f3e08981baa73f2e06eafd0a`. Rank 8, alpha 16,
  applied at strength 1 to all 190 declared target projections. Other supported
  adapter revisions, file sizes and SHA-256 values are in tipo_native_lora.py.
- [Gemma TIPO scale-20pct-001](https://huggingface.co/raspie/gemma4-tipo-lora-scale-20pct-001),
  revision `3668df754885175408ac247a9e85309fa6528486`: the experimental rank-8
  adapter completed on 2026-11-03. This release is **CC BY-NC 4.0** because its
  training uses GBC synthetic annotations with non-commercial terms; it does
  not inherit the pilot release's Apache-2.0 label. See its model card, LICENSE
  and NOTICE for source credits, training provenance and evaluation limits.
- [RS Seasonal Prompt Generator](https://github.com/raspie10032/RS-Seasonal-Prompt-Generator):
  the native Gemma Q4 loader, tokenizer integration and native LoRA conversion
  were vendored from local source revision `354cc9b`. See NOTICE and the included
  Apache license. Joint scene routing, bounded response validation and transient
  sampling exclusions are the changes made in this repository.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI): host CLIP tokenization,
  native Gemma generation, ModelPatcher LoRA load/offload, progress and interrupt.
  The Q4 path dequantizes the pinned base for native PyTorch execution; GGUF file
  size is not a claim about VRAM consumption.

The adapters are TIPO-format LoRAs on Gemma. They are not the official
KBlueLeaf TIPO standalone checkpoint. Model licenses are separate from this
extension's code license and remain attached to their respective sources.

---

# Earlier implementation notes and source provenance

The material below records the earlier standalone Kohaku/TIPO experiments and
their attribution. The architecture, source and license references remain
relevant to today's selectable TIPO 2.1 backend. Dated experiment controls and
the former Korean interpreter describe earlier implementations; use the
[current guide](usage.ko.md) for the shared node's supported controls.

# References, provenance and adaptations

Reviewed 2026-11-02. This page distinguishes upstream model/architecture work
from the controls implemented in this independent experimental extension.

## Sources and how they are used

| Source | What this project uses or learns from it | Relevant local files |
| --- | --- | --- |
| [TIPO paper — Text to Image with Text Presampling for Prompt Optimization](https://arxiv.org/abs/2411.08127) | The upstream text-presampling approach: expand a prompt before image diffusion. It does not establish this extension's character ownership or interaction accuracy. | Project purpose; baseline expansion |
| [Official TIPO v2.1 model and pinned model card](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M/tree/f5a318524a4ab30cdbbf51816cf406170f454e65) | Official safetensors weights, tokenizer, config and learned tag/caption format. The extension loads this checkpoint; it does not retrain TIPO. | `runtime.py`, `tools/download_model.py`, `control.py` |
| [KohakUwULLM official Kohaku HF implementation, pinned source](https://github.com/KohakuBlueleaf/KohakUwULLM/tree/d4a52f77c683e6a0622d76a4f42752d97ced97ba/src/kohakuwullm/export/hf) | Architecture code adapted for native inference: RMSNorm, QK normalization, grouped-query attention, RoPE, dense blocks, sparse MoE routing and official checkpoint tensor layout. | `model.py` |
| [KGen official repository](https://github.com/KohakuBlueleaf/KGen) and [prompt formatter `apply_tipo_prompt`](https://github.com/KohakuBlueleaf/KGen/blob/fecfe05341f021f916f9b23e1c93f23b4c50d9bb/src/kgen/executor/tipo.py) | Learned `target`, `long`, `tag` ordering and `long_to_tag` caption-to-tag use were consulted. Shared interaction is passed as a caption rather than invented chat/role syntax. KGen is not an inference dependency. | `control.py`, `expansion.py` |
| [ComfyUI model management](https://github.com/comfyanonymous/ComfyUI/blob/5cc4b870094c89090db54bd8d12649dcaf89f2ba/comfy/model_management.py) and [ModelPatcher](https://github.com/comfyanonymous/ComfyUI/blob/5cc4b870094c89090db54bd8d12649dcaf89f2ba/comfy/model_patcher.py) | Existing host APIs for device selection, whole-model load/offload, progress and interruption. The project integrates those APIs rather than running a separate inference server. | `runtime.py`, `nodes.py` |
| [RS MultiCharacter Sampler](https://github.com/raspie10032/ComfyUI_RS_MultiCharacter_Sampler) / [RS NAI API Request](https://github.com/raspie10032/ComfyUI_RS_NAI_API_Request) | Existing building/final pipe and modern NAI-context data protocols. Prompt editing retains non-prompt settings using installed types/adapters. NAI network requests are not performed by TIPO. | `interface.py`, `expansion.py` |

The actual model revision used by the loader and downloader is
`f5a318524a4ab30cdbbf51816cf406170f454e65`.
The architecture source revision is
`d4a52f77c683e6a0622d76a4f42752d97ced97ba`.
The original architecture comparison found the official model export and the
pinned upstream architecture source identical; see [baseline validation](validation.md).

The formatter inspected during development is the installed `tipo-kgen 0.3.1`
`kgen/executor/tipo.py::apply_tipo_prompt`. The KGen source link above is pinned
to the repository revision inspected during this documentation pass,
`fecfe05341f021f916f9b23e1c93f23b4c50d9bb`; it is not asserted to be the wheel's
build revision. The installed and pinned `apply_tipo_prompt` function ASTs were
also compared and matched in this documentation pass. There is no KGen or
Transformers import in the native inference
package. Reference/parity tools can require the official Transformers implementation.

## Changes implemented in this project

The **earlier Korean interpreter** used host-native Gemma 4 `CLIP.tokenize/generate/decode`,
also used by [Seasonal's native implementation](https://github.com/raspie10032/RS-Seasonal-Prompt-Generator/blob/main/gemma_tipo_native_node.py).
No Seasonal source/runtime is imported. `language.py` adds fixed clause IDs,
program-preserved weights/English spans, joint scene context, validation and one
corrective attempt. `language_runtime.py` uses ComfyUI-managed native loading and
no TIPO adapter. Gemma weight terms/distribution remain separate; existing local
weights are reused. This paragraph is retained as an implementation record:
the current shared node uses its selected Gemma+LoRA or standalone TIPO backend,
not this earlier interpreter chain.

For 0.1.1, [Hugging Face generation utilities](https://huggingface.co/docs/transformers/internal/generation_utils)
provide the reference for decoding-time token constraints. This project implements
its own request-local decoded-tag candidate rejection; it does not import the
Transformers generation runtime or claim semantic negative-prompt training.
[Hub file-download documentation](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)
documents pinned revisions, `local_dir` and `tqdm_class`, used by the automatic
downloader with an additional pinned SHA-256 manifest.

The upstream decoder was adapted to remove its Transformers runtime dependency
and use a request-owned KV cache, direct safetensors loading and ComfyUI model
management. Baseline parity is checked against the official implementation.

The experimental controller adds these behaviors:

- Optional official `rating` and `quality` metadata before `target:`. The explicit
  rating maps to `nsfw, explicit`; quality can include an era. The model card
  supplies these vocabularies; this extension adds choices, validation, per-target
  propagation and an execution report. Defaults omit both fields. These are model
  guidance rather than a content filter or guaranteed image-quality improvement.

- Separate scene and character token ownership, with each character in an
  independent request/cache. Another character's private prompt is excluded.
- Source-span `log(weight)` biases before attention softmax during both prefill
  and decoding; causal/owner masks and readable-history repetition penalties.
- Owner-separated MoE dispatch where strict isolation needs it, to prevent
  blocked inputs changing readable-token FP16 rounding through expert batching.
- A shared caption derived from the common scene and recognized participant
  action/position clauses, in the upstream learned format. No separate user
  interaction input. Each participant retains its complete private source;
  appearance written in the common scene remains visible to every participant.
- Preservation of original prompt strings/weights, conservative filtering of
  recognized conflicting additions, and visible reasons for rejected tags.
- Separate STRING inputs/outputs, optional RS building-pipe creation/editing,
  and a results panel with legacy frontend connection/restoration handling.

These additions do not constitute upstream TIPO features or an upstream-endorsed
relationship planner. They do not establish improved image-generation quality.
Filtering recognizes a limited vocabulary and can reject useful additions or
miss contradictions. See [source protection](source-protection.md),
[ownership experiment](owned-attention.md) and [compatibility](compatibility.md).

## Licenses and redistribution boundaries

- The adapted architecture source is Apache-2.0; preserve
  [NOTICE](../NOTICE) and [the upstream Apache license](../KohakUwULLM-APACHE-LICENSE).
  The upstream source credits Shih-Ying Yeh (2024).
- The pinned **model card identifies the weights as Kohaku License 1.0**.
  Model-weight terms are separate from the architecture license; consult the
  [official pinned repository](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M/tree/f5a318524a4ab30cdbbf51816cf406170f454e65).
  An architecture license alone does not license the model weights.
- KGen's repository supplies its own
  [Apache-2.0 license](https://github.com/KohakuBlueleaf/KGen/blob/fecfe05341f021f916f9b23e1c93f23b4c50d9bb/LICENSE).
  It was consulted for prompt-format compatibility; this is not a distribution
  of the KGen inference stack.
- Weights, downloaded reference Python and raw local validation outputs are not
  committed in this project. The automatic downloader and optional CLI fetch the official
  pinned model data; it does not execute downloaded Python code.
- New project code is distributed under [GPL-3.0-only](../LICENSE); the complete
  text is in [COPYING](../COPYING). Adapted KohakUwULLM code retains its
  Apache-2.0 attribution. The model weight terms remain separate.
- Public source: https://github.com/raspie10032/ComfyUI_RS_TIPO_Native

## Reading order

1. [한국어 설치·사용법](usage.ko.md)
   - [Backend selection, automatic downloads and quality limits](native-models.md)
2. [Interface and shared interaction](interface.md)
3. [Source protection](source-protection.md)
4. [Native inference baseline](validation.md)
5. [Owned attention and pipe checks](owned-attention.md)
6. [Legacy/current compatibility checks](compatibility.md)
7. [Trained Gemma TIPO execution and measured limits](trained-node-validation.ko.md)
8. [Published scale-20pct-001 training summary, sources and license](https://huggingface.co/raspie/gemma4-tipo-lora-scale-20pct-001)
