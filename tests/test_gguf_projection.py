"""Output-row views must retain exact weights without a packed gather copy."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gguf
import torch
import test_native


def load_ops():
    # The tested operations need torch/GGUF only. Keep CI independent of a
    # full ComfyUI checkout; stubs apply only while defining the module.
    comfy = ModuleType('comfy')
    comfy.__path__ = []
    modules = {'comfy': comfy}
    for name in ('ops', 'utils', 'lora', 'model_patcher', 'model_management'):
        module = ModuleType('comfy.' + name)
        setattr(comfy, name, module)
        modules['comfy.' + name] = module
    class ManualCast:
        Linear = torch.nn.Linear
        Embedding = torch.nn.Embedding
    comfy.ops.manual_cast = ManualCast
    comfy.model_patcher.ModelPatcher = object
    path = Path(__file__).resolve().parents[1] / 'tipo_gguf_ops.py'
    spec = importlib.util.spec_from_file_location('rs_tipo_native.projection_test_ops', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


ops = load_ops()


def packed_weight(rows=17, columns=256):
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_0]
    generator = torch.Generator().manual_seed(817)
    data = torch.randint(0, 256, (rows * columns // block, size), dtype=torch.uint8,
                         generator=generator)
    data[:, :2] = torch.tensor([.125], dtype=torch.float16).view(torch.uint8)
    return ops.PackedWeight(data, gguf.GGMLQuantizationType.Q4_0, (rows, columns))


class GGUFProjectionTests(unittest.TestCase):
    def test_weight_created_during_inference_keeps_persistent_version_counter(self):
        with torch.inference_mode():
            weight = packed_weight()
            copied = weight.to(device='cpu', copy=True)
            cloned = weight.clone()
        for value in (weight, copied, cloned):
            with self.subTest(value=id(value)):
                self.assertEqual(value._version, 0)
                value.as_subclass(torch.Tensor)
        self.assertEqual(weight.untyped_storage().data_ptr(),
                         weight.as_subclass(torch.Tensor).untyped_storage().data_ptr())

    def test_slice_equals_indexed_decode_and_keeps_packed_storage(self):
        weight = packed_weight()
        expected = ops.unpack(weight, 'cpu', torch.float32, rows=torch.arange(3, 9))
        seen = []
        original = ops.dequantize
        def observe(data, *args, **kwargs):
            seen.append(data.untyped_storage().data_ptr())
            return original(data, *args, **kwargs)
        with patch.object(ops, 'dequantize', observe):
            actual = ops.unpack(weight, 'cpu', torch.float32, rows=slice(3, 9))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(seen, [weight.untyped_storage().data_ptr()])

    def test_projection_matches_indexed_chunks_with_tail_and_leading_dimensions(self):
        weight = packed_weight()
        layer = ops.GGUFOps.Embedding(17, 256)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        x = torch.randn(2, 3, 256, generator=torch.Generator().manual_seed(821))
        expected = torch.cat([torch.nn.functional.linear(x,
            ops.unpack(weight, 'cpu', x.dtype, rows=torch.arange(start, min(start + 5, 17))))
            for start in range(0, 17, 5)], dim=-1)
        with patch('torch.arange', side_effect=AssertionError('unnecessary output-row indices')):
            actual = layer.project(x, chunk=5)
        self.assertTrue(torch.equal(actual, expected))

    def test_float_weight_slice_and_arbitrary_embedding_rows_are_unchanged(self):
        weight = packed_weight()
        full = ops.unpack(weight, 'cpu', torch.float32)
        self.assertTrue(torch.equal(ops.unpack(full, 'cpu', torch.float16, rows=slice(2, 8)),
                                    full[2:8].half()))
        rows = torch.tensor([[3, 1, 3], [9, 0, 16]])
        actual = ops.unpack(weight, 'cpu', torch.float32, rows=rows)
        self.assertTrue(torch.equal(actual, full[rows]))

    def test_offloaded_table_survives_parent_move_without_any_weight_transfer(self):
        table = ops.GGUFOps.Embedding(17, 256)
        table.weight = torch.nn.Parameter(packed_weight(), requires_grad=False)
        table.cpu_offload = True
        model = torch.nn.ModuleDict({'ple': table, 'other': torch.nn.Linear(2, 2)})
        original = table.weight
        model.to('meta')
        self.assertIs(table.weight, original)
        self.assertEqual(table.weight.device.type, 'cpu')
        self.assertEqual(model['other'].weight.device.type, 'meta')
        rows = torch.tensor([[3, 1, 3], [9, 0, 16]])
        expected = ops.unpack(original, 'cpu', torch.float32)[rows]
        self.assertTrue(torch.equal(table(rows), expected))

    def test_failed_apply_restores_offloaded_parameter(self):
        table = ops.GGUFOps.Embedding(17, 256)
        table.weight = torch.nn.Parameter(packed_weight(), requires_grad=False)
        table.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        table.cpu_offload = True
        original = table.weight
        with self.assertRaisesRegex(RuntimeError, 'transfer failed'):
            table._apply(lambda x: (_ for _ in ()).throw(RuntimeError('transfer failed')))
        self.assertIs(table.weight, original)

    def test_only_ple_is_offloaded_and_accounting_excludes_its_packed_bytes(self):
        model = torch.nn.Module()
        model.model = torch.nn.ModuleDict({
            'embed_tokens_per_layer': ops.GGUFOps.Embedding(17, 256),
            'embed_tokens': ops.GGUFOps.Embedding(17, 256)})
        for m in model.model.values():
            m.weight = torch.nn.Parameter(packed_weight(), requires_grad=False)
        patcher = SimpleNamespace(model=model, size=0, patches={}, loaded_size=lambda: 0)
        def size(module):
            return sum(p.nbytes for p in module.parameters())
        with patch.object(ops.comfy.model_management, 'module_size', side_effect=size, create=True):
            ops.set_ple_offload(patcher)
            self.assertTrue(model.model['embed_tokens_per_layer'].cpu_offload)
            self.assertFalse(model.model['embed_tokens'].cpu_offload)
            self.assertEqual(ops.GGUFModelPatcher.model_size(patcher), size(model) // 2)
            ops.set_ple_offload(patcher, False)
            self.assertEqual(ops.GGUFModelPatcher.model_size(patcher), size(model))
        patcher.loaded_size = lambda: 1
        with self.assertRaisesRegex(RuntimeError, 'Unload'):
            ops.set_ple_offload(patcher)
        patcher.loaded_size = lambda: 0
        patcher.patches = {'model.embed_tokens_per_layer.weight': []}
        with self.assertRaisesRegex(ValueError, 'PLE weight patches'):
            ops.set_ple_offload(patcher)


if __name__ == '__main__':
    unittest.main()
