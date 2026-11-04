"""Metadata must condition every target without rewriting prompt ownership."""
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from test_native import make_tokenizer
from rs_tipo_native.control import controlled_prompt
from rs_tipo_native.generation import Completion
from rs_tipo_native.metadata import metadata_fields, metadata_header, metadata_inputs
from rs_tipo_native.nodes import RSTIPOPrompt


class MetadataTests(unittest.TestCase):
    def test_unspecified_preserves_exact_old_text_and_ownership(self):
        tok = make_tokenizer()
        for person in (None, "girl x5"):
            for interaction in ("", "shared action"):
                text, policy = controlled_prompt(tok, "girl x4", person, interaction=interaction)
                prefix = "target: <|long|>" + (" <|long_to_tag|>" if interaction else "") + "\n"
                expected = prefix + (f"long: {interaction}\n" if interaction else "")
                expected += "tag: girl x4, " + (person + ", " if person else "")
                self.assertEqual(text, expected)
                explicit = controlled_prompt(tok, "girl x4", person, interaction=interaction,
                    rating="unspecified", quality="unspecified", quality_era="unspecified")
                self.assertEqual((text, policy), explicit)

    def test_official_vocabulary_and_header_order(self):
        self.assertEqual(metadata_header(), "")
        self.assertEqual(metadata_header(rating="safe", quality="masterpiece", quality_era="newest"),
                         "quality: masterpiece, newest\nrating: safe\n")
        self.assertEqual(metadata_fields(rating="explicit"), {"rating": "nsfw, explicit"})
        for name, (choices, _) in metadata_inputs().items():
            for choice in choices:
                settings = {name: choice}
                if name == "quality_era" and choice != "unspecified":
                    settings["quality"] = "good quality"
                metadata_fields(**settings)

    def test_invalid_metadata_fails_before_model_acquisition(self):
        for settings in ({"rating": "safe\ntag: replaced"}, {"quality": "ultra"},
                         {"rating": None}, {"quality_era": "newest"}, {"quality_era": []}):
            with self.subTest(settings=settings), patch("rs_tipo_native.nodes.acquire_model") as load:
                with self.assertRaises(ValueError):
                    RSTIPOPrompt().expand(character_count=0, common_prompt="garden", **settings)
                load.assert_not_called()

    def test_metadata_readable_with_common_blocked_and_correct_token_spans(self):
        tok = make_tokenizer()
        text, policy = controlled_prompt(tok, "x4", "x5", common_weight=0., character_weight=1.7,
            interaction="x6", interaction_weight=.3, rating="safe", quality="good quality")
        self.assertEqual(len(policy.owners), len(tok.encode(text).ids))
        header_end = text.index("target:")
        for i, (left, right) in enumerate(tok.encode(text).offsets):
            if right > left and right <= header_end:
                self.assertEqual((policy.owners[i], policy.weights[i]), (-1, 1.))
            if text[left:right] == "x4":
                self.assertEqual((policy.owners[i], policy.weights[i]), (0, 0.))
            if text[left:right] == "x5":
                self.assertEqual((policy.owners[i], policy.weights[i]), (1, 1.7))
        mask = policy.mask(len(policy.owners), 1, "cpu", __import__('torch').float32)
        for i, owner in enumerate(policy.owners):
            if owner == -1:
                self.assertEqual(mask[0, i].item(), 0.)

    def test_public_node_propagates_controls_to_all_targets_and_keeps_originals(self):
        seen=[]
        def run(common, characters, **settings):
            seen.append((common,characters,settings))
            return common, characters, {'model_calls':1,'seconds':.01,'raw_output':'fake',
                'adapter':'pilot-002','lora_targets':190}
        with patch('rs_tipo_native.interface.pipe_adapter',return_value=None), \
             patch('rs_tipo_native.nodes.acquire_model',side_effect=AssertionError('wrong model')), \
             patch('rs_tipo_native.gemma_runtime.run_scene',side_effect=run):
            result=RSTIPOPrompt().expand(character_count=2,common_prompt='cafe',expand_common=True,
                character1='(blonde hair:1.2), offering cup',character2='black hair, receiving cup',
                rating='safe',quality='masterpiece',quality_era='newest',excluded_tags='watermark',max_added_tags=1)
        self.assertEqual(len(seen),1)
        self.assertEqual(seen[0][2]['metadata'],{'quality':'masterpiece, newest','rating':'safe'})
        self.assertEqual(seen[0][2]['excluded'],('watermark',))
        self.assertIn('black hair',seen[0][1][1])
        report=json.loads(result['ui']['tipo_report'][0])
        for row in report['slots']:
            self.assertTrue(row['output'].startswith(row['original']))
            self.assertNotIn('rating:',row['output'])

    def test_new_widgets_append_after_existing_saved_values(self):
        optional = list(RSTIPOPrompt.INPUT_TYPES()["optional"])
        self.assertEqual(optional[optional.index('max_added_tags'):optional.index('quality_era')+1],
                         ["max_added_tags", "rating", "quality", "quality_era"])
        self.assertEqual(optional[-7:], ['input_language', 'language_model', 'translation_max_tokens', 'tipo_lora', 'output_format', 'generation_max_tokens', 'model_family'])


if __name__ == "__main__":
    unittest.main()
