"""Packed affine decoders must match GGUF and leave model bytes reusable."""
import unittest

import gguf
import numpy as np
import torch
import test_native

from rs_tipo_native.vendor.gguf_torch.dequant import dequantize


class GGUFDequantTests(unittest.TestCase):
    def test_affine_k_quants_match_independent_gguf_reference_without_mutation(self):
        generator = np.random.default_rng(941)
        for qtype in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K):
            block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
            blocks = generator.integers(0, 256, (37, type_size), dtype=np.uint8)
            # Finite, varied half-precision scale/min values, including zeros
            # and negative scales; all remaining packed bits are unconstrained.
            scales = generator.choice(np.array([0., -.125, .0625, 1.5, 2**-20],
                                                dtype=np.float16), size=(37, 2))
            blocks[:, :4] = scales.view(np.uint8)
            original = blocks.copy()
            expected = torch.from_numpy(gguf.quants.dequantize(blocks, qtype))
            tensor = torch.from_numpy(blocks)
            for target in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(qtype=qtype.name, target=target):
                    actual = dequantize(tensor, qtype, (37, block_size),
                                        dtype=torch.float32).to(target)
                    self.assertTrue(torch.equal(actual, expected.to(target)))
                    self.assertTrue(actual.is_contiguous())
                    self.assertTrue(np.array_equal(blocks, original))
            repeated = dequantize(tensor, qtype, (37, block_size), dtype=torch.float32)
            self.assertTrue(torch.equal(repeated, expected))


if __name__ == '__main__':
    unittest.main()
