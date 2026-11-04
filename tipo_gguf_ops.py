# Vendored from RS-Seasonal-Prompt-Generator 354cc9b (Apache-2.0).
"""Packed GGUF weights with ComfyUI patching and device-local PyTorch operations.

Tensor metadata and delayed LoRA handling adapted from City96/ComfyUI-GGUF
(Apache-2.0); see vendor/gguf_torch/NOTICE.md. No llama.cpp runtime.
"""
import torch
import gguf
import comfy.ops
import comfy.utils
import comfy.lora
import comfy.model_patcher
import comfy.model_management

try:
    from comfy.weight_adapter.lora import LoRAAdapter
except ImportError:
    # Older ComfyUI versions can still use their original weight-merging path.
    LoRAAdapter = None

try:
    from .vendor.gguf_torch.dequant import dequantize, dequantize_functions
except ImportError:
    from vendor.gguf_torch.dequant import dequantize, dequantize_functions


class PackedWeight(torch.Tensor):
    @staticmethod
    def __new__(cls, data, qtype, logical_shape):
        # Patcher loads can run inside inference_mode(). Keep a version counter
        # on persistent weight wrappers so later offload/reload can alias them.
        # This wraps the same packed storage; it does not clone model bytes.
        with torch.inference_mode(False):
            return torch.Tensor._make_subclass(cls, data, False)

    def __init__(self, data, qtype, logical_shape):
        self.qtype = qtype
        self.logical_shape = torch.Size(logical_shape)
        self.patches = []

    @property
    def shape(self):
        return getattr(self, "logical_shape", super().shape)

    def to(self, *args, **kwargs):
        # Packed bytes must never be converted to the model's floating dtype.
        copy = kwargs.pop('copy', False)
        device, _, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        data = self.as_subclass(torch.Tensor).to(device=device, non_blocking=non_blocking,
                                               copy=copy)
        result = PackedWeight(data, self.qtype, self.shape)
        result.patches = self.patches.copy()
        return result

    def detach(self):
        return self

    def clone(self, *args, **kwargs):
        result = PackedWeight(self.as_subclass(torch.Tensor).clone(*args, **kwargs), self.qtype, self.shape)
        result.patches = self.patches.copy()
        return result


def unpack(weight, device, dtype, rows=None, *, apply_patches=True):
    if not isinstance(weight, PackedWeight):
        if rows is not None:
            index = rows if isinstance(rows, slice) else rows.to(weight.device)
            return weight[index].to(device=device, dtype=dtype)
        return weight.to(device=device, dtype=dtype)
    if weight.qtype not in dequantize_functions:
        raise ValueError(f"No PyTorch GGUF decoder for {weight.qtype}; CPU fallback is disabled")
    data = weight.as_subclass(torch.Tensor)
    shape = weight.shape
    if rows is not None:
        # PLE has billions of values. Gather packed rows BEFORE dequantization.
        block, size = gguf.GGML_QUANT_SIZES[weight.qtype]
        if shape[-1] % block:
            raise ValueError("GGUF embedding rows are not block aligned")
        data = data.reshape(shape[0], shape[1] // block * size)
        # An offloaded PLE/embedding must not transfer its entire packed table
        # for a handful of tokens. Select on the storage device first.
        if isinstance(rows, slice):
            # Sequential output chunks are already contiguous. A view avoids
            # allocating indices and gathering/copying the same packed bytes.
            data = data[rows].to(device)
            shape = (data.shape[0], shape[1])
        else:
            data = data[rows.to(data.device)].to(device)
            shape = (*rows.shape, shape[1])
    else:
        data = data.to(device)
    result = dequantize(data, weight.qtype, shape, dtype=torch.float32).to(dtype)
    if apply_patches and weight.patches:
        if rows is not None:
            raise ValueError("Embedding LoRA is not supported by this GGUF path")
        for patches, key in weight.patches:
            result = comfy.lora.calculate_weight(patches, result, key, intermediate_dtype=dtype)
    return result


def linear_lora_patches(weight):
    """Return only native, additive 2D LoRAs; otherwise keep weight merging.

    Validate the entire patch chain before bypassing any of it. Offsets,
    transforms, DoRA, LoCon and model-strength changes require weight semantics.
    No device tensors are cached here or on the shared adapter objects.
    """
    if (LoRAAdapter is None or not callable(LoRAAdapter.__dict__.get('h'))
            or not isinstance(weight, PackedWeight) or not weight.patches):
        return None
    result = []
    for patches, _key in weight.patches:
        for strength, adapter, strength_model, offset, function in patches:
            if (type(adapter) is not LoRAAdapter or strength_model != 1.0
                    or offset is not None or function is not None
                    or getattr(adapter, 'is_conv', False)
                    or getattr(adapter, 'multiplier', 1.0) != 1.0):
                return None
            up, down, alpha, mid, dora, reshape = adapter.weights
            if (mid is not None or dora is not None or reshape is not None
                    or up.ndim != 2 or down.ndim != 2 or down.shape[0] == 0
                    or tuple(weight.shape) != (up.shape[0], down.shape[1])
                    or up.shape[1] != down.shape[0]):
                return None
            result.append((strength, adapter))
    return result or None


def add_lora_output(x, output, patches):
    """Use ComfyUI's LoRA h(x), moving only the small factors for this call."""
    for strength, adapter in patches:
        if strength == 0:
            continue
        up, down, alpha, *_ = adapter.weights
        local = LoRAAdapter(adapter.loaded_keys, (
            comfy.model_management.cast_to_device(up, x.device, x.dtype),
            comfy.model_management.cast_to_device(down, x.device, x.dtype),
            alpha, None, None, None))
        # Native h() includes alpha/rank and multiplier. Never mutate adapters
        # shared between patcher clones, or allocate another scaled delta.
        local.multiplier = strength
        output = output + local.h(x, output)
    return output


class PackedModule:
    comfy_cast_weights = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ("weight", "bias"):
            key = prefix + name
            if key in state_dict:
                value = state_dict[key]
                self._parameters[name] = torch.nn.Parameter(value, requires_grad=False)
            elif name == "weight" or self.bias is not None:
                missing_keys.append(key)

    def _weight(self, x):
        w = unpack(self.weight, x.device, x.dtype)
        for function in self.weight_function:
            w = function(w)
        return w


class GGUFOps(comfy.ops.manual_cast):
    class Linear(PackedModule, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features, self.out_features = in_features, out_features
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.weight_function, self.bias_function = [], []

        def forward(self, x):
            patches = linear_lora_patches(self.weight) if not self.weight_function else None
            if patches is not None:
                weight = unpack(self.weight, x.device, x.dtype, apply_patches=False)
                output = torch.nn.functional.linear(x, weight,
                    self.bias.to(x) if self.bias is not None else None)
                del weight
                return add_lora_output(x, output, patches)
            return torch.nn.functional.linear(x, self._weight(x),
                self.bias.to(x) if self.bias is not None else None)

    class Embedding(PackedModule, comfy.ops.manual_cast.Embedding):
        cpu_offload = False

        def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None, **kwargs):
            torch.nn.Module.__init__(self)
            self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
            self.padding_idx, self.max_norm, self.norm_type = None, None, 2.0
            self.scale_grad_by_freq, self.sparse = False, False
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.weight_function, self.bias_function = [], []

        def _apply(self, fn, recurse=True):
            if not self.cpu_offload:
                return super()._apply(fn, recurse=recurse)
            # ModelPatcher full loads also call model.to(device). Keep the
            # row-only table on CPU without briefly allocating it on the GPU.
            weight = self._parameters.pop('weight')
            try:
                return super()._apply(fn, recurse=recurse)
            finally:
                self._parameters['weight'] = weight

        def forward(self, input, out_dtype=None):
            return unpack(self.weight, input.device, out_dtype or torch.float32, rows=input)

        def project(self, x, chunk=8192):
            # Tied output projection: bounded temporary allocation, never a
            # complete 262144 x 1536 float weight matrix.
            output = []
            for start in range(0, self.num_embeddings, chunk):
                rows = slice(start, min(start + chunk, self.num_embeddings))
                w = unpack(self.weight, x.device, x.dtype, rows=rows)
                output.append(torch.nn.functional.linear(x, w))
            return torch.cat(output, dim=-1)


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    def model_size(self):
        if self.size <= 0:
            self.size = comfy.model_management.module_size(self.model) - sum(
                comfy.model_management.module_size(m) for m in self.model.modules()
                if getattr(m, 'cpu_offload', False))
        return self.size

    def _load_list(self, *args, **kwargs):
        # Exclude CPU row tables from both loading and partial-unload accounting.
        # They require neither a full-weight GPU buffer nor GPU patching.
        return [entry for entry in super()._load_list(*args, **kwargs)
                if not getattr(entry[-2], 'cpu_offload', False)]

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False,
                               return_weight=False, force_cast=False):
        weight = comfy.utils.get_attr(self.model, key)
        if not isinstance(weight, PackedWeight):
            return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)
        result = weight.to(device=device_to)
        result.patches = [(self.patches[key], key)] if key in self.patches else []
        if return_weight:
            return result
        comfy.utils.set_attr_param(self.model, key, result)
        return result

    def load(self, *args, force_patch_weights=False, **kwargs):
        return super().load(*args, force_patch_weights=True, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if isinstance(p, PackedWeight):
                    p.patches = []
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def clone(self, *args, **kwargs):
        result = super().clone(*args, **kwargs)
        result.__class__ = GGUFModelPatcher
        return result


def set_ple_offload(patcher, enabled=True):
    """Set row-only PLE placement before loading a GGUF model (shared by clones)."""
    if patcher.loaded_size():
        raise RuntimeError('Unload the GGUF model before changing PLE placement')
    for name, module in patcher.model.named_modules():
        if name.endswith('.embed_tokens_per_layer') and isinstance(module, GGUFOps.Embedding):
            if enabled and module.weight.device.type != 'cpu':
                raise RuntimeError('PLE weights must be on CPU before enabling row offload')
            if any(key.startswith(name + '.') for key in patcher.patches):
                raise ValueError('PLE weight patches are not supported by row offload')
            module.cpu_offload = bool(enabled)
    # Cached total must represent loadable GPU bytes, including after toggling.
    patcher.size = 0
