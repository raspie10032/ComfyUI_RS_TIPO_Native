import unittest
from unittest.mock import patch

import test_native
from rs_tipo_native.additions import select_additions
from rs_tipo_native.expansion import expand_owned
from rs_tipo_native.generation import Completion
from rs_tipo_native.nodes import RSTIPOPrompt


class AdditionsTests(unittest.TestCase):
    def test_transfer_keeps_supported_object_without_inventing_a_scene(self):
        kept, rejected = select_additions('blue eyes, offering cup',
            ['holding cup', 'sitting', 'saucer', 'tea', 'heterochromia', 'holding pen'],
            common='cafe, passing a cup')
        self.assertEqual(kept, ['holding cup', 'sitting'])
        self.assertEqual(len(rejected), 4)

    def test_peer_roles_cannot_ground_an_action(self):
        self.assertEqual(select_additions('brown hair, waving',
            ['holding cup', 'receiving cup'], common='cafe')[0], [])

    def test_sufficient_roles_return_original_without_synonym_padding(self):
        self.assertEqual(select_additions('(offering_cup:1.2), extending arms',
            ['offering cup', 'outstretched arms'])[0], [])
        original = '(offering_cup:1.2), extending arms'
        with patch('rs_tipo_native.expansion.complete', return_value=Completion(
                'offering cup, outstretched arms', [], 1, 'eos', .1)):
            out, row = expand_owned(None, test_native.make_tokenizer(), 'cafe', original,
                grounded_additions=True, max_added_tags=6)
        self.assertEqual(out, original)
        self.assertEqual(row['added_tags'], '')

    def test_shared_action_is_available_without_copying_peer_appearance(self):
        self.assertEqual(select_additions('blue eyes', ['dancing', 'smile', 'holding hands'],
            common='ballroom, dancing together')[0], ['dancing', 'smile'])
        self.assertEqual(select_additions('blue eyes', ['smile'], common='happy')[0], ['smile'])

    def test_keep_helpful_transfer_staging_instead_of_only_literal_synonyms(self):
        self.assertEqual(select_additions('offering cup, extending arms',
            ['outstretched hand', 'looking at another', 'holding pen'],
            common='passing a cup')[0], ['outstretched hand', 'looking at another'])
        self.assertEqual(select_additions('standing, sad', ['sitting', 'smile'])[0], [])

    def test_do_not_invent_contacts_from_generic_body_part_overlap(self):
        self.assertEqual(select_additions('leading dance, hand on other waist',
            ['holding hands', 'hand on other shoulder', 'dancing'])[0], [])

    def test_budget_limits_both_protected_and_unprotected_outputs(self):
        raw = Completion('warm lighting, garden, tree', [], 1, 'eos', .1)
        for protected in (False, True):
            with patch('rs_tipo_native.expansion.complete', return_value=raw):
                out, row = expand_owned(None, test_native.make_tokenizer(), 'outdoors',
                    max_added_tags=1, grounded_additions=True, protect_source=protected)
            self.assertEqual(out, 'outdoors, warm lighting')
            self.assertEqual(len(row['rejected_tags']), 2)

    def test_invalid_budget_fails_before_model_loading(self):
        for bad in (True, 0, 33, 1.5):
            with self.subTest(bad=bad), patch('rs_tipo_native.nodes.acquire_model') as model:
                with self.assertRaisesRegex(ValueError, 'max_added_tags'):
                    RSTIPOPrompt().expand(character_count=0, common_prompt='garden', max_added_tags=bad)
                model.assert_not_called()

    def test_budget_does_not_count_original_or_duplicate_candidates(self):
        raw = Completion('garden, garden, tree, tree, grass', [], 1, 'eos', .1)
        with patch('rs_tipo_native.expansion.complete', return_value=raw):
            out, row = expand_owned(None, test_native.make_tokenizer(), 'garden',
                max_added_tags=2, protect_source=False)
        self.assertEqual(out, 'garden, tree, grass')
        self.assertEqual(row['rejected_tags'], [])

    def test_legacy_workflows_have_a_default_without_changing_output_indexes(self):
        optional = RSTIPOPrompt.INPUT_TYPES()['optional']
        self.assertEqual(optional['max_added_tags'][1]['default'], 6)
        self.assertGreater(list(optional).index('max_added_tags'), list(optional).index('excluded_tags'))
        self.assertEqual(RSTIPOPrompt.RETURN_NAMES[1:4], ('common_prompt', 'character1', 'character2'))
