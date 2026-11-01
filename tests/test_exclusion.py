import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

import test_native
from rs_tipo_native.exclusion import TagExclusion, normalize, parse_excluded, validate_sources
from rs_tipo_native.generation import choose_token, complete
from rs_tipo_native.nodes import RSTIPOPrompt


def tokenizer():
    vocab = {'[UNK]': 0, 'red': 1, 'hair': 2, 'dress': 3, ',': 4, 'red_hair': 5,
             'hairs': 6, 'blue': 7}
    t = Tokenizer(models.WordPiece(vocab, unk_token='[UNK]'))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.decoder = decoders.WordPiece(cleanup=False)
    return t


class ExclusionTests(unittest.TestCase):
    def test_spaces_underscores_case_and_weights(self):
        self.assertEqual(parse_excluded('LONG_HAIR, long hair\n(long  hair:1.2), 1.5::long_hair::'), ('long hair',))
        self.assertEqual(normalize('ＲＥＤ＿ＨＡＩＲ'), 'red hair')

    def test_multitoken_and_single_token_alternative_encodings(self):
        c = TagExclusion(tokenizer(), parse_excluded('red hair'))
        self.assertTrue(c.allows([], 1))
        self.assertFalse(c.allows([1], 2))
        self.assertFalse(c.allows([], 5))
        self.assertTrue(c.allows([1], 3))
        self.assertTrue(c.allows([7, 2, 4, 1], 3))
        self.assertEqual(c.report()['blocked_candidates'], 2)

    def test_no_substring_block_in_longer_complete_tag(self):
        c = TagExclusion(tokenizer(), ('hair',))
        self.assertTrue(c.allows([1], 2))  # red hair is not the tag hair
        self.assertTrue(c.allows([], 6))  # hairs is not hair
        self.assertFalse(c.allows([], 2))

    def test_original_conflict_before_model_load_and_inactive_ignored(self):
        with patch('rs_tipo_native.nodes.acquire_model') as load:
            with self.assertRaisesRegex(ValueError, 'conflicts with character 1'):
                RSTIPOPrompt().expand(character_count=1, character1='(RED_HAIR:1.2)', excluded_tags='red hair')
            load.assert_not_called()
        validate_sources(('red hair',), [('character', 'blue hair')])

    def test_sampling_rechecks_topk_and_greedy_after_block(self):
        for temp in (0., .5):
            hits = []
            def allowed(t):
                hits.append(t)
                return t == 2
            token = choose_token(torch.tensor([3., 2., 1.]), [], torch.Generator().manual_seed(0),
                                 temperature=temp, top_k=1, token_allowed=allowed)
            self.assertEqual(token, 2)
            self.assertEqual(hits, [0, 1, 2])

    def test_dead_end_and_cancellation_are_explicit(self):
        with self.assertRaisesRegex(RuntimeError, 'every next token'):
            choose_token(torch.ones(3), [], torch.Generator(), token_allowed=lambda _: False)
        def cancel():
            raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):
            choose_token(torch.ones(3), [], torch.Generator(), token_allowed=lambda _: False, interrupt=cancel)

    def test_unrelated_exclusion_preserves_token_ids_and_request_isolation(self):
        m, t = test_native.make_model(), test_native.make_tokenizer()
        args = dict(max_new_tokens=8, seed=12)
        before = complete(m, t, 'girl', **args)
        unrelated = complete(m, t, 'girl', excluded_tags='red_hair', **args)
        self.assertEqual(before.token_ids, unrelated.token_ids)
        self.assertEqual(unrelated.exclusion['blocked_candidates'], 0)
        complete(m, t, 'girl', excluded_tags='x4', **args)
        self.assertEqual(before.token_ids, complete(m, t, 'girl', **args).token_ids)

    def test_special_tokens_and_limits_not_accepted(self):
        for invalid in ('<|eos|>', ','.join('tag'+str(i) for i in range(257))):
            with self.assertRaises(ValueError):
                parse_excluded(invalid)
