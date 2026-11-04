from dataclasses import dataclass, replace
import unittest

import torch

from test_native import make_model, make_tokenizer
from rs_tipo_native.control import OwnedAttention, controlled_prompt
from rs_tipo_native.model import KVCache
from rs_tipo_native.generation import complete
from rs_tipo_native.expansion import append_completion, expand_pipe


class ControlTests(unittest.TestCase):
    @torch.inference_mode()
    def test_blocked_owner_changes_cannot_reach_target_at_any_layer(self):
        model = make_model()
        policy = OwnedAttention((-1, 0, 1, 1, 2, 2, 1), (1.,)*7, 1)
        a = torch.tensor([[1, 3, 4, 5, 6, 7, 8]])
        b = a.clone()
        b[:, 4:6] = torch.tensor([21, 22])
        mask = policy.mask(0, 7, "cpu", torch.float32)
        groups = policy.mlp_groups(0, 7, "cpu")
        out_a = model(a, attention_mask=mask, token_owners=groups)
        out_b = model(b, attention_mask=mask, token_owners=groups)
        torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
        # The same perturbation reaches the target under ordinary causal attention.
        self.assertFalse(torch.equal(model(a), model(b)))

    @torch.inference_mode()
    def test_zero_common_weight_also_blocks_indirect_prefill_leakage(self):
        model = make_model()
        policy = OwnedAttention((-1, 0, 0, 1, 1), (1., 0., 0., 1.5, 1.5), 1)
        a = torch.tensor([[1, 3, 4, 5, 6]])
        b = torch.tensor([[1, 21, 22, 5, 6]])
        mask = policy.mask(0, 5, "cpu", torch.float32)
        groups = policy.mlp_groups(0, 5, "cpu")
        torch.testing.assert_close(model(a, attention_mask=mask, token_owners=groups),
                                   model(b, attention_mask=mask, token_owners=groups),
                                   rtol=0, atol=0)

    @torch.inference_mode()
    def test_controlled_cached_decode_matches_full_with_offsets(self):
        model = make_model()
        ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8]])
        policy = OwnedAttention((-1, 0, 0, 1, 1), (1., .5, .5, 1.5, 1.5), 1)
        cache = KVCache(model.config, 12, "cpu", torch.float32)
        model(ids[:, :5], cache=cache, attention_mask=policy.mask(0, 5, "cpu", torch.float32))
        actual = model(ids[:, 5:], cache=cache, attention_mask=policy.mask(5, 2, "cpu", torch.float32))
        expected = model(ids, attention_mask=policy.mask(0, 7, "cpu", torch.float32))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_shared_interaction_uses_learned_caption_format_with_own_attention(self):
        tokenizer = make_tokenizer()
        text, policy = controlled_prompt(tokenizer, "garden", "blonde hair",
            interaction="Two women hug each other.")
        self.assertIn("<|long_to_tag|>", text)
        self.assertIn("long: Two women hug each other.\ntag: garden, blonde hair", text)
        self.assertIn(0, policy.owners)
        self.assertIn(1, policy.owners)
        self.assertIn(1., policy.weights)
        # No artificial role instruction/character-number protocol is invented.
        self.assertNotIn("character_1:", text)

    def test_weight_is_log_bias_and_causality_is_preserved(self):
        policy = OwnedAttention((-1, 0, 1), (1., .5, 2.), 1)
        mask = policy.mask(3, 1, "cpu", torch.float32)
        probability = mask.softmax(-1)[0]
        self.assertAlmostEqual(float(probability[2]/probability[1]), 4., places=6)
        self.assertTrue(torch.isneginf(policy.mask(0, 3, "cpu", torch.float32)[0, 1:]).all())
        self.assertEqual(float(mask[0, -1]), 0.)  # History weight is neutral.

    def test_blocked_tokens_are_excluded_from_repetition_penalty(self):
        policy = OwnedAttention((-1, 0, 1, 2, 1), (1., 0., 1.5, 1., 1.5), 1)
        self.assertEqual(policy.readable_history([10, 20, 30, 40, 50]), [10, 30, 50])

    def test_unit_weights_preserve_ordinary_token_sequence(self):
        model, tokenizer = make_model(), make_tokenizer()
        prompt = "girl x4 x5"
        ids = tokenizer.encode(prompt).ids
        policy = OwnedAttention(tuple([-1] + [1]*(len(ids)-1)), (1.,)*len(ids), 1)
        settings = dict(max_new_tokens=6, seed=17, temperature=0)
        self.assertEqual(complete(model, tokenizer, prompt, **settings).token_ids,
            complete(model, tokenizer, prompt, attention_control=policy, **settings).token_ids)

    def test_prompt_owners_match_whole_encoding(self):
        tokenizer = make_tokenizer()
        text, policy = controlled_prompt(tokenizer, "girl", "x4 x5", common_weight=.3)
        self.assertEqual(len(policy.owners), len(tokenizer.encode(text).ids))
        self.assertIn(0, policy.owners)
        self.assertIn(1, policy.owners)
        self.assertEqual(policy.owners[-1], 1)

    def test_added_tags_preserve_source_and_deduplicate(self):
        source = "(blonde hair:1.2), blue eyes"
        result, added = append_completion(source, "blue eyes, garden, garden\ncaption: changed")
        self.assertEqual(result, source + ", garden")
        self.assertEqual(added, "garden")

    def test_weighted_groups_and_literal_names_are_indivisible_tags(self):
        from rs_tipo_native.expansion import split_tags
        source = r'aria \(original, alternate\), (smile, blush:1.5)'
        self.assertEqual(split_tags(source), [r'aria \(original, alternate\)', '(smile, blush:1.5)'])
        combined, added = append_completion(source, '(smile, blush:1.5), (blue jacket, long sleeves:1.2)')
        self.assertEqual(combined, source + ', (blue jacket, long sleeves:1.2)')
        self.assertEqual(added, '(blue jacket, long sleeves:1.2)')

    def test_pipe_keeps_objects_and_never_passes_other_characters_to_run(self):
        @dataclass(frozen=True)
        class Character:
            prompt: str
            mask: object
            lora: str
            uc: str = "bad"
        @dataclass(frozen=True)
        class Pipe:
            characters: tuple
            prompt: str = "campus"
            negative_prompt: str = "low quality"
            text_regions: tuple = ("HELLO",)
        mask = object()
        source = Pipe((Character("blonde", mask, "a"), Character("black hair", mask, "b")))
        calls = []
        def run(common, character):
            calls.append((common, character))
            return (common if character is None else character) + ", extra", {}
        changed, scene, report = expand_pipe(source, run, expand_common=True)
        self.assertEqual(calls, [("campus", None), ("campus", "blonde"), ("campus", "black hair")])
        self.assertIs(changed.characters[0].mask, mask)
        self.assertEqual(changed.negative_prompt, source.negative_prompt)
        self.assertEqual(changed.text_regions, source.text_regions)
        self.assertEqual(changed.characters[1].lora, "b")
        self.assertEqual(source.characters[0].prompt, "blonde")

    def test_draft_skips_inactive_and_text_slots(self):
        @dataclass(frozen=True)
        class Slot:
            number: int
            prompt: str
            enabled: bool = True
            kind: str = "character"
        @dataclass(frozen=True)
        class Draft:
            slots: tuple
            prompt_count: int = 3
            source_pipe: object = None
            camera_settings: tuple = ("deferred",)
        source = Draft((Slot(1, "first"), Slot(2, "disabled", False),
                        Slot(3, "literal", kind="text"), Slot(4, "hidden")))
        calls = []
        def run(common, character):
            calls.append(character)
            return character + ", extra", {}
        changed, scene, _ = expand_pipe(source, run, common_prompt="campus")
        self.assertEqual(calls, ["first"])
        self.assertEqual(scene, "campus")
        self.assertEqual(changed.camera_settings, source.camera_settings)
        self.assertEqual(changed.slots[1:], source.slots[1:])

    def test_invalid_control_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            OwnedAttention((-1, 1), (1., float("nan")), 1)
        with self.assertRaisesRegex(ValueError, "match"):
            complete(make_model(), make_tokenizer(), "girl", max_new_tokens=1,
                attention_control=OwnedAttention((-1,), (1.,), 1))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
