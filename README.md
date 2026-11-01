# ComfyUI RS TIPO Native

Independent experimental project for running **TIPO v2.1** using native PyTorch
and ComfyUI model management, with no llama.cpp, GGUF or Transformers in the
inference path. This is a baseline for later input-span attention control research.

## Current scope

- Kohaku decoder, grouped-query attention, QK normalization, RoPE and sparse MoE.
- Official safetensors checkpoint names, strictly loaded with no ignored tensors.
- ComfyUI `ModelPatcher` for loading/offloading; one model cached in host memory.
- Request-local KV cache and sampling generator, including deterministic seed 0.
- Completion and tag expansion nodes; no changes to RS MultiCharacter Sampler.
- Cancellation and progress during decoding; context overflow is an explicit error.

**Not implemented yet:** character ownership isolation, weighted input spans,
regional/NAI pipe integration, conflict prevention, model quantization, and public
release. Separate character prompts must not be flattened into the baseline node.
The baseline can invent conflicting attributes just like ordinary TIPO.

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
search for **RS TIPO Native**. Both nodes load the model internally.
`RS TIPO Native Expand` accepts a tag string and preserves the original string
while appending the generated first tag field. Check `info` for the complete raw
completion and finish reason. `length` and `max_new_tokens` are different:
the former is a learned target, the latter is a hard output limit.

For numerical comparison use `RS TIPO Native Completion`, the official TIPO
completion format, and temperature 0. This is a completion model, not a chat model:

```text
target: <|long|>
tag: 1girl, garden,
```

## Verification

```sh
python tests/test_native.py
```

Full-checkpoint parity and ComfyUI API smoke results belong in
[`docs/validation.md`](docs/validation.md). Unit tests using a tiny synthetic model
do not establish real-model quality or runtime compatibility.

## Upstream and licenses

The architecture is adapted from the official
[KohakUwULLM HF export](https://github.com/KohakuBlueleaf/KohakUwULLM/tree/d4a52f77c683e6a0622d76a4f42752d97ced97ba/src/kohakuwullm/export/hf),
whose code matches the pinned model repository's architecture source byte for byte.
See `KohakUwULLM-APACHE-LICENSE` and `NOTICE` for upstream attribution.
Model weights have their own terms, separately from the architecture code.
No model weights are stored in this Git repository. This project has not been
published; the license for new project code will be selected before publication.
