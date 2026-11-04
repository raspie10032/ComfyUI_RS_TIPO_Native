import json
import unittest
from unittest.mock import patch

import test_native
from rs_tipo_native.gemma_prompt import parse_response
from rs_tipo_native.gemma_runtime import run_scene, GemmaOutputError


class GemmaJSONTests(unittest.TestCase):
    def parse(self, raw, count=1):
        return parse_response(raw, 'cafe', ('adult character',)*count,
            output_format='natural_language', expand_common=True)

    def test_early_common_quote_keeps_later_lighting_and_all_characters(self):
        for count in (0, 1, 2, 10):
            with self.subTest(count=count):
                characters = {str(i): f'The character is adult number {i} and holds a cup.'
                              for i in range(1, count+1)}
                common = 'The scene is a cafe with adult friends". The lighting is soft and even.'
                raw = '{"common": "'+common+'\n"characters": '+json.dumps(characters)+'}'
                scene, slots = self.parse(raw, count)
                self.assertEqual(scene, common)
                self.assertEqual(slots, tuple(characters.values()))

    def test_inner_common_quotes_are_preserved(self):
        common = 'The scene is a cafe with a "Welcome" sign. The lighting is warm.'
        raw = '{"common": "'+common+'", "characters": {"1": "The adult holds a cup."}}'
        self.assertEqual(self.parse(raw), (common, ('The adult holds a cup.',)))

    def test_common_missing_comma_does_not_trigger_generation_retry(self):
        raw = ('{"common": "The scene is a cafe. The lighting is soft." '
               '"characters": {"1": "The adult holds a cup."}}')
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw,
                 {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}})) as generate:
            scene, slots, report = run_scene('cafe', ('adult holding a cup',),
                output_format='natural_language', expand_common=True)
        self.assertEqual(scene, 'The scene is a cafe. The lighting is soft.')
        self.assertEqual(slots, ('The adult holds a cup.',))
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(report['calls'][0]['raw'], raw)

    def test_common_literal_linebreak_and_invalid_escape_survive(self):
        raw = ('{"common": "The scene is a cafe \\(indoor\\).\nThe lighting is soft.", '
               '"characters": {"1": "The adult holds a cup."}}')
        self.assertEqual(self.parse(raw)[0],
                         'The scene is a cafe \\(indoor\\).\nThe lighting is soft.')

    def test_valid_json_is_unchanged_including_escapes_and_key_order(self):
        common = 'The scene is a cafe with a "Welcome" sign.\nThe path is C:\\new\\art.'
        characters = {'1': 'The adult wears a blue coat.'}
        raw = json.dumps({'characters': characters, 'common': common})
        self.assertEqual(self.parse(raw), (common, tuple(characters.values())))

    def test_common_recovery_still_honors_source_weights(self):
        raw = ('{"common": "The scene is a (colorful theme:1.3) cafe". '
               'The lighting is soft.\n"characters": {"1": "The adult holds a cup."}}')
        scene, _ = parse_response(raw, '(colorful theme:1.3)', ('adult',),
            output_format='natural_language', expand_common=True)
        self.assertIn('(colorful theme:1.3)', scene)
        with self.assertRaisesRegex(ValueError, 'weighted source clause'):
            parse_response(raw.replace('(colorful theme:1.3)', 'colorful'),
                '(colorful theme:1.3)', ('adult',),
                output_format='natural_language', expand_common=True)

    def test_recovery_refuses_incomplete_or_ambiguous_structure(self):
        for raw in (
            '{"common": "The scene is a cafe". The lighting is soft.',
            '{"common": "The scene is a cafe". "characters": {"1": "The adult holds a cup."}',
            '{"common": "The scene is a cafe". "extra": "value", "characters": {}}',
            '{"common": "The scene is a cafe". "characters": {"1": "The adult holds a cup.\\"}}',
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, 'invalid/incomplete JSON'):
                self.parse(raw)

    def test_recovery_does_not_hide_wrong_character_slots(self):
        raw = ('{"common": "The scene is a cafe". The lighting is soft.\n'
               '"characters": {"2": "The adult holds a cup."}}')
        with self.assertRaisesRegex(ValueError, 'slot count/type mismatch'):
            self.parse(raw)

    def test_unrecoverable_json_retry_only_repairs_format(self):
        raw = '{"common": "The scene is a cafe.", "characters": {"1": "The adult holds a "blue" cup."}}'
        repaired = json.dumps({'common': 'The scene is a cafe.',
            'characters': {'1': 'The adult holds a "blue" cup.'}})
        responses = iter((raw, repaired))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                 (next(responses), {'seconds': .1, 'output_tokens': 5,
                    'exclusion': {'blocked_candidates': 0}})) as generate:
            scene, slots, report = run_scene('cafe', ('adult holding a blue cup',),
                output_format='natural_language', expand_common=True)
        self.assertEqual(slots, ('The adult holds a "blue" cup.',))
        retry = generate.call_args_list[1].args[1]
        self.assertTrue(retry.startswith('Repair JSON syntax only.'))
        self.assertNotIn('EXPANSION CHECK', retry)
        self.assertNotIn('add at least one', retry)
        self.assertIn(json.dumps({'previous_response': raw}), retry)
        self.assertEqual(report['model_calls'], 2)

    def test_failed_json_repair_remains_bounded_and_keeps_raw_responses(self):
        raw = '{"common": "The scene is a cafe.", "characters": {"1": "unfinished'
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw,
                 {'seconds': .1, 'output_tokens': 5})) as generate:
            with self.assertRaises(GemmaOutputError) as caught:
                run_scene('cafe', ('adult',), output_format='natural_language', expand_common=True)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual([call['raw'] for call in caught.exception.calls], [raw, raw])
