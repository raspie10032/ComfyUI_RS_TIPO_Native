import unittest
from unittest.mock import patch

import test_native
from rs_tipo_native.grounding import model_text, guard_additions
from rs_tipo_native.control import controlled_prompt
from rs_tipo_native.expansion import expand_owned
from rs_tipo_native.generation import Completion


class GroundingTests(unittest.TestCase):
    def test_weighted_model_input_keeps_literal_names_and_artist_colons(self):
        source = r'(blonde hair:1.2), (artist:name:0.8), name \(series\), 1.2::blue eyes::'
        self.assertEqual(model_text(source), r'blonde hair, artist:name, name \(series\), blue eyes')
        prompt, _ = controlled_prompt(test_native.make_tokenizer(), 'garden', source)
        self.assertNotIn(':1.2)', prompt)
        self.assertIn(r'name \(series\)', prompt)

    def test_guard_keeps_partner_action_rejects_identity_and_own_gesture(self):
        kept, rejected = guard_additions('(blonde hair:1.2), blue eyes, pink dress, leading dance, hand on other waist',
            ['brown hair', 'green eyes', '3d', 'dress', 'dancing', 'hand on own hip',
             'looking at viewer', 'looking at another', 'warm lighting', 'following dance'],
            interaction='They look at each other and dance.')
        self.assertEqual(kept, ['dancing', 'looking at another', 'warm lighting'])
        self.assertEqual(len(rejected), 6) # already-supplied dress is redundant

    def test_scene_cannot_acquire_any_participant_action_or_appearance(self):
        kept, rejected = guard_additions('campus, autumn',
            ['black hair', 'holding cup', 'sitting', 'smile', 'blush', 'tree', 'warm lighting'],
            character=False)
        self.assertEqual(kept, ['tree', 'warm lighting'])
        self.assertEqual(len(rejected), 5)

    def test_known_role_conflict_and_unrequested_orientation(self):
        kept, rejected = guard_additions('on left, offering cup, standing',
            ['receiving cup', 'sitting', 'hug from behind', 'reaching out', 'holding cup'])
        self.assertEqual(kept, ['reaching out', 'holding cup'])
        self.assertEqual(len(rejected), 3)
        self.assertEqual(guard_additions('hug from behind', ['hug from behind'])[0], [])

    def test_target_contact_and_transfer_role_stay_authoritative(self):
        kept, rejected = guard_additions('following dance, hand on other shoulder',
            ["hand on another's back", "hand on another's shoulder", 'dancing'])
        self.assertEqual(kept, ["hand on another's shoulder", 'dancing'])
        self.assertEqual(rejected[0]['reason'], 'partner contact target conflicts with supplied role')
        self.assertEqual(guard_additions('receiving cup', ['drinking', 'holding cup'])[0], ['holding cup'])

    def test_expand_protects_weighted_original_and_reports_raw_rejections(self):
        result = Completion('brown hair, looking at another, outstret', [1], 10, 'length', .1)
        original = '(blonde hair:1.2), blue eyes, offering cup'
        with patch('rs_tipo_native.expansion.complete', return_value=result):
            expanded, row = expand_owned(None, test_native.make_tokenizer(), 'cafe', original)
        self.assertEqual(expanded, original + ', looking at another')
        self.assertEqual(row['rejected_tags'][0]['tag'], 'brown hair')
        self.assertEqual(row['text'], result.text)
        self.assertNotIn('outstret', row['proposed_tags'])

    def test_opt_out_preserves_raw_tags_but_never_truncated_tag(self):
        result = Completion('brown hair, reaching out, coff', [1], 10, 'length', .1)
        with patch('rs_tipo_native.expansion.complete', return_value=result):
            expanded, row = expand_owned(None, test_native.make_tokenizer(), 'cafe', 'blonde hair', protect_source=False)
        self.assertEqual(expanded, 'blonde hair, brown hair, reaching out')
        self.assertFalse(row['protect_source'])
        self.assertEqual(row['rejected_tags'], [])

    def test_one_unterminated_tag_is_not_added_but_eos_tag_is(self):
        tokenizer = test_native.make_tokenizer()
        for reason, expected in [('length','blonde hair'), ('eos','blonde hair, holding cup')]:
            with patch('rs_tipo_native.expansion.complete', return_value=Completion('holding cup', [], 1, reason, .1)):
                output, _ = expand_owned(None, tokenizer, 'cafe', 'blonde hair')
            self.assertEqual(output, expected)

    def test_expression_conflicts_and_duplicate_sources(self):
        kept, _ = guard_additions('(blue eyes:1.3), smile', ['blue eyes', 'open mouth', 'closed mouth', 'smile'])
        self.assertEqual(kept, ['open mouth'])

if __name__ == '__main__':
    unittest.main()
