"""Regression checks for native LoRA dispatch, fallback and shared state.

The tiny adapter implements the upstream h(x) contract so CI needs no ComfyUI.
Real LoRAAdapter numerical and generation checks use the GPU benchmark tool.
"""
import unittest
from unittest.mock import patch

import torch
from test_gguf_projection import ops, packed_weight


class NativeAdapter:
    multiplier = 1.0

    def __init__(self, loaded_keys, weights):
        self.loaded_keys, self.weights = loaded_keys, weights

    def h(self, x, base_out):
        up, down, alpha, *_ = self.weights
        scale = (alpha / down.shape[0] if alpha is not None else 1.0) * self.multiplier
        return torch.nn.functional.linear(torch.nn.functional.linear(x, down), up) * scale


class GGUFLoraBypassTests(unittest.TestCase):
    def setUp(self):
        self.native = patch.object(ops, 'LoRAAdapter', NativeAdapter)
        self.native.start()
        self.addCleanup(self.native.stop)
        self.cast = patch.object(ops.comfy.model_management, 'cast_to_device',
            side_effect=lambda tensor, device, dtype: tensor.to(device=device, dtype=dtype), create=True)
        self.cast.start()
        self.addCleanup(self.cast.stop)
        self.weight = packed_weight(rows=17, columns=256)
        generator = torch.Generator().manual_seed(412)
        self.up = torch.randn(17, 4, dtype=torch.float64, generator=generator) * .1
        self.down = torch.randn(4, 256, dtype=torch.float64, generator=generator) * .1
        self.adapter = NativeAdapter({'test'}, (self.up, self.down, 8., None, None, None))
        self.layer = ops.GGUFOps.Linear(256, 17)
        self.layer.weight = torch.nn.Parameter(self.weight, requires_grad=False)
        self.layer.bias = torch.nn.Parameter(torch.linspace(-1., 1., 17, dtype=torch.float64),
                                             requires_grad=False)
        self.x = torch.randn(2, 3, 256, dtype=torch.float64, generator=generator)

    def attach(self, strength=1., adapter=None, strength_model=1., offset=None, function=None):
        self.weight.patches = [([(strength, adapter or self.adapter, strength_model, offset, function)], 'weight')]

    def test_multiple_scaled_loras_match_merged_math_without_merging(self):
        self.attach(.7)
        second = NativeAdapter({'second'}, (self.up * .3, self.down, None, None, None, None))
        self.weight.patches[0][0].append((-.2, second, 1., None, None))
        base = ops.unpack(self.weight, 'cpu', torch.float64, apply_patches=False)
        merged = base + .7 * 2 * (self.up @ self.down) - .2 * ((self.up * .3) @ self.down)
        expected = torch.nn.functional.linear(self.x, merged, self.layer.bias)
        with patch.object(ops.comfy.lora, 'calculate_weight',
                          side_effect=AssertionError('full LoRA matrix allocated'), create=True):
            actual = self.layer(self.x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(self.adapter.multiplier, 1.)
        self.assertEqual(second.multiplier, 1.)
        self.assertIs(self.adapter.weights[0], self.up)
        self.assertIs(self.adapter.weights[1], self.down)
        self.assertEqual(set(vars(self.adapter)), {'loaded_keys', 'weights'})

    def test_zero_strength_avoids_even_small_factor_transfers(self):
        self.attach(0.)
        expected = torch.nn.functional.linear(self.x,
            ops.unpack(self.weight, 'cpu', self.x.dtype, apply_patches=False), self.layer.bias)
        actual = self.layer(self.x)
        self.assertTrue(torch.equal(actual, expected))
        ops.comfy.model_management.cast_to_device.assert_not_called()

    def test_unsupported_patch_chains_fall_back_in_full(self):
        variants = [
            dict(strength_model=.5), dict(offset=(0, 0, 4)), dict(function=lambda x: x),
            *[dict(adapter=NativeAdapter({'test'}, (self.up, self.down, 8., *extra)))
              for extra in ((torch.ones(1), None, None), (None, torch.ones(17), None),
                            (None, None, (17, 256)))],
            dict(adapter=NativeAdapter({'test'}, (self.up[:, :2], self.down, 8., None, None, None))),
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.attach(**variant)
                # A valid patch before the unsupported patch must not leak
                # through as a partial bypass or be applied twice.
                self.weight.patches.insert(0, ([(1., self.adapter, 1., None, None)], 'first'))
                sentinel = torch.zeros(17, 256, dtype=self.x.dtype)
                with patch.object(self.layer, '_weight', return_value=sentinel) as merge, \
                     patch.object(ops, 'add_lora_output', side_effect=AssertionError('partial bypass')):
                    actual = self.layer(self.x)
                merge.assert_called_once_with(self.x)
                self.assertTrue(torch.equal(actual, self.layer.bias.expand_as(actual)))

    def test_weight_functions_and_missing_native_api_keep_original_path(self):
        self.attach()
        for use_wrapper in (False, True):
            with self.subTest(use_wrapper=use_wrapper):
                self.layer.weight_function = [lambda x: x] if use_wrapper else []
                with patch.object(ops, 'LoRAAdapter', NativeAdapter if use_wrapper else None), \
                     patch.object(self.layer, '_weight', return_value=torch.zeros(17, 256, dtype=self.x.dtype)) as merge:
                    self.layer(self.x)
                merge.assert_called_once_with(self.x)

    def test_inherited_placeholder_h_does_not_enable_bypass(self):
        class AdapterWithInheritedH(NativeAdapter):
            pass

        self.attach(adapter=AdapterWithInheritedH({'old-api'}, self.adapter.weights))
        with patch.object(ops, 'LoRAAdapter', AdapterWithInheritedH):
            self.assertIsNone(ops.linear_lora_patches(self.weight))

    def test_unpatched_and_float_weights_keep_original_path(self):
        self.assertIsNone(ops.linear_lora_patches(self.weight))
        self.assertIsNone(ops.linear_lora_patches(torch.zeros(17, 256)))
        self.attach()
        clone = self.weight.to(device='cpu')
        self.assertIsNot(clone.patches, self.weight.patches)
        clone.patches = []
        self.assertIsNone(ops.linear_lora_patches(clone))
        self.assertIsNotNone(ops.linear_lora_patches(self.weight))


if __name__ == '__main__':
    unittest.main()
