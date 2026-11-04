import importlib.util
from pathlib import Path
import sys
import subprocess
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rs_tipo_native", ROOT / "__init__.py",
                                            submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from rs_tipo_native.model import KohakuConfig, KohakuForCausalLM, KVCache
from rs_tipo_native.generation import complete


def make_model():
    cfg = KohakuConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                       intermediate_size=24, moe_intermediate_size=8,
                       n_routed_experts=4, num_experts_per_tok=2,
                       max_position_embeddings=32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(871)
        model = KohakuForCausalLM(cfg).eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm" in name:
                    param.fill_(1)
                else:
                    param.normal_(0, 0.08)
    return model


def make_tokenizer():
    vocab = {"<|unk|>": 0, "<|bos|>": 1, "<|eos|>": 2, "girl": 3,
             **{f"x{i}": i for i in range(4, 32)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="<|bos|> $A",
                                                              special_tokens=[("<|bos|>", 1)])
    return tokenizer


class NativeTests(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_and_cached_decode_match_full_forward(self):
        model = make_model()
        ids = torch.tensor([[1, 3, 4, 5, 6, 7]])
        expected = model(ids, last_only=False)
        cache = KVCache(model.config, 12, "cpu", torch.float32)
        first = model(ids[:, :3], cache=cache, last_only=False)
        torch.testing.assert_close(first, expected[:, :3], atol=1e-6, rtol=1e-5)
        for i in range(3, ids.shape[1]):
            part = model(ids[:, i:i+1], cache=cache)
            torch.testing.assert_close(part, expected[:, i:i+1], atol=1e-6, rtol=1e-5)
        self.assertEqual(cache.position, 6)

    @torch.inference_mode()
    def test_cached_multi_token_chunk_uses_offset_causal_mask(self):
        model = make_model()
        ids = torch.tensor([[1, 3, 4, 5, 6, 7]])
        cache = KVCache(model.config, 10, "cpu", torch.float32)
        model(ids[:, :3], cache=cache)
        actual = model(ids[:, 3:], cache=cache, last_only=False)
        torch.testing.assert_close(actual, model(ids, last_only=False)[:, 3:], atol=1e-6, rtol=1e-5)

    def test_request_cache_does_not_leak_between_prompts(self):
        model, tokenizer = make_model(), make_tokenizer()
        a = complete(model, tokenizer, "girl x4", max_new_tokens=5, seed=0)
        complete(model, tokenizer, "x14 x19", max_new_tokens=5, seed=32)
        b = complete(model, tokenizer, "girl x4", max_new_tokens=5, seed=0)
        self.assertEqual(a.token_ids, b.token_ids)

    def test_seed_zero_does_not_change_global_rng(self):
        model, tokenizer = make_model(), make_tokenizer()
        state = torch.random.get_rng_state().clone()
        complete(model, tokenizer, "girl", max_new_tokens=4, seed=0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_context_overflow_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "context overflow"):
            complete(make_model(), make_tokenizer(), "girl " * 30, max_new_tokens=4)

    def test_empty_input_and_bad_sampling_are_reported(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            complete(make_model(), make_tokenizer(), "  ")
        with self.assertRaisesRegex(ValueError, "sampling"):
            complete(make_model(), make_tokenizer(), "girl", min_p=-1)

    def test_comfy_interrupt_propagates(self):
        def cancel():
            raise InterruptedError("user cancelled")
        with self.assertRaises(InterruptedError):
            complete(make_model(), make_tokenizer(), "girl", max_new_tokens=4, interrupt=cancel)

    def test_checkpoint_shape_and_type_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "Kohaku"):
            KohakuConfig.from_dict({"model_type": "llama"})
        with self.assertRaises(RuntimeError):
            make_model().load_state_dict({}, strict=True)

    def test_native_import_does_not_load_llama_or_transformers(self):
        # Training tests deliberately import transformers. The public native
        # import contract must be checked in a fresh process, not discovery order.
        script = '''
import importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("rs_tipo_native", root / "__init__.py",
                                            submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from rs_tipo_native.model import KohakuConfig, KohakuForCausalLM, KVCache
from rs_tipo_native.generation import complete
assert "llama_cpp" not in sys.modules
assert "transformers" not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script, str(ROOT)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
