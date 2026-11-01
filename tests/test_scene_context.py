import unittest

import test_native
from rs_tipo_native.scene_context import action_clauses, scene_context
from rs_tipo_native.nodes import RSTIPOPrompt


class SceneContextTests(unittest.TestCase):
    def test_no_separate_interaction_input(self):
        schema = RSTIPOPrompt.INPUT_TYPES()
        self.assertNotIn('interaction_prompt', schema['optional'])

    def test_appearance_does_not_cross_into_role_summary(self):
        text, roles = scene_context('cafe, 2girls', (
            '(blonde hair:1.2), blue eyes, pink cardigan, on left, offering cup, extending arms',
            'black hair, yellow sweater, on right, receiving cup, reaching out'), 0)
        self.assertIn('Person 1: on left, offering cup, extending arms.', text)
        self.assertIn('Person 2: on right, receiving cup, reaching out.', text)
        self.assertTrue(text.endswith('The person being described: on left, offering cup, extending arms.'))
        for forbidden in ('blonde', 'black hair', 'pink', 'yellow', 'eyes', '1.2'):
            self.assertNotIn(forbidden, text)
        self.assertEqual(roles[1], ('on right', 'receiving cup', 'reaching out'))

    def test_roles_change_context_but_private_appearance_does_not(self):
        first = 'offering cup, red hair'
        before, _ = scene_context('cafe', (first, 'receiving cup, blue eyes'), 0)
        after, _ = scene_context('cafe', (first, 'receiving cup, green eyes'), 0)
        self.assertEqual(before, after)
        changed, _ = scene_context('cafe', (first, 'waving, green eyes'), 0)
        self.assertNotEqual(before, changed)

    def test_no_fabricated_action_or_screen_position(self):
        text, roles = scene_context('garden', ('blonde hair', 'brown eyes'), 1)
        self.assertEqual(text, 'garden')
        self.assertEqual(roles, ((), ()))
        self.assertNotIn('left', text)

    def test_weighted_actions_and_mixed_clauses(self):
        self.assertEqual(action_clauses('(holding hands:1.3), hand on other waist, holding pink dress, looking at blue eyes'),
                         ('holding hands', 'hand on other waist'))

    def test_partner_only_action_does_not_become_targets_role(self):
        from rs_tipo_native.grounding import guard_additions
        kept, rejected = guard_additions('red hair, waving', ['smile', 'holding hands'],
            common='garden', peer_actions=('holding hands',))
        self.assertEqual(kept, ['smile'])
        self.assertEqual(len(rejected), 1)
        self.assertIn('another participant', rejected[0]['reason'])
        self.assertEqual(guard_additions('waving', ['holding hands'],
            common='holding hands', peer_actions=('holding hands',))[0], ['holding hands'])

    def test_scene_only_does_not_invent_participants(self):
        self.assertEqual(scene_context('night, garden', ()), ('night, garden', ()))
