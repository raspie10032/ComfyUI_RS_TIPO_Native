# References, provenance and adaptations

Reviewed 2026-11-01. This page distinguishes upstream model/architecture work
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

The upstream decoder was adapted to remove its Transformers runtime dependency
and use a request-owned KV cache, direct safetensors loading and ComfyUI model
management. Baseline parity is checked against the official implementation.

The experimental controller adds these behaviors:

- Separate scene and character token ownership, with each character in an
  independent request/cache. Another character's private prompt is excluded.
- Source-span `log(weight)` biases before attention softmax during both prefill
  and decoding; causal/owner masks and readable-history repetition penalties.
- Owner-separated MoE dispatch where strict isolation needs it, to prevent
  blocked inputs changing readable-token FP16 rounding through expert batching.
- A shared interaction caption in the upstream learned format, while preserving
  each participant's own appearance and role source. Shared prose still exposes
  whatever the user puts in it to all participants.
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
  committed in this project. The explicit download tool fetches the official
  pinned model data; it does not execute downloaded Python code.
- A license for this project's new code has not been selected. The local project
  has no public release or GitHub remote configured at this documentation pass.
  This page records provenance; it does not assign a new project-wide license.

## Reading order

1. [한국어 설치·사용법](usage.ko.md)
2. [Interface and shared interaction](interface.md)
3. [Source protection](source-protection.md)
4. [Native inference baseline](validation.md)
5. [Owned attention and pipe checks](owned-attention.md)
6. [Legacy/current compatibility checks](compatibility.md)
