import json
import unittest
from unittest.mock import patch

import test_native
from rs_tipo_native.prompt_weights import detach_weights
from rs_tipo_native.gemma_runtime import run_scene
from rs_tipo_native.tipo_native_lora import PRESAMPLING_ADAPTER

INFO = {'seconds': .1, 'output_tokens': 12}


class WeightHandlingTests(unittest.TestCase):
    def test_nested_escaped_and_nai_groups_roundtrip_without_numeric_model_input(self):
        for source, plain in (
            ('((blue eyes:1.1), silver hair:1.30), smile', 'blue eyes, silver hair, smile'),
            (r'(alice \(game\):1.2), red coat', r'alice \(game\), red coat'),
            ('1.25::blue eyes, silver hair ::, smile', 'blue eyes, silver hair , smile'),
            ('(blue eyes:-.5), [smile], (red coat)', 'blue eyes, [smile], (red coat)'),
        ):
            with self.subTest(source=source):
                plan = detach_weights(source)
                self.assertEqual(plan.render(), source)
                self.assertEqual(plan.plain(), plain)

    def test_group_members_are_deduplicated_but_every_new_detail_survives(self):
        plan = detach_weights('(open mouth, determined, one eye closed, :1.3)')
        self.assertEqual(plan.restore('open mouth, determined, one_eye_closed, red coat, soft lighting', tags=True),
            '(open mouth, determined, one eye closed, :1.3), red coat, soft lighting')

    def test_changed_or_missing_phrase_restores_only_the_weighted_phrase(self):
        plan = detach_weights('(silver hair:1.30)')
        self.assertEqual(plan.restore('white hair, blue eyes, standing by a window', tags=True),
            '(silver hair:1.30), white hair, blue eyes, standing by a window')
        self.assertEqual(plan.restore('She stands by a window.', tags=False),
            'She stands by a window. (silver hair:1.30)')

    def test_prose_reinserts_original_group_without_losing_sentence(self):
        plan = detach_weights('(blue coat:1.2), smiling')
        self.assertEqual(plan.restore('She wears a blue coat beside a window.', tags=False),
            'She wears a (blue coat:1.2) beside a window.')

    def test_signed_fraction_and_escaped_name_duplicates_use_same_weight_parser(self):
        for source, generated, expected in (
            ('(blue eyes:-.5)', 'blue_eyes, smiling', '(blue eyes:-.5), smiling'),
            (r'(alice \(game\):1.2)', 'alice (game), smiling', r'(alice \(game\):1.2), smiling'),
            ('1.25::blue eyes ::', 'blue_eyes, smiling', '1.25::blue eyes ::, smiling'),
        ):
            with self.subTest(source=source):
                self.assertEqual(detach_weights(source).restore(generated, tags=True), expected)

    def test_korean_translation_is_paired_by_phrase_not_repeated_number(self):
        plan = detach_weights('(파란 눈:1.2), (빨간 코트:1.2)')
        translations = {'파란 눈': 'blue eyes', '빨간 코트': 'red coat'}
        self.assertEqual(plan.translation_terms(), ('파란 눈', '빨간 코트'))
        self.assertEqual(plan.plain(translations), 'blue eyes, red coat')
        self.assertEqual(plan.restore('red coat, blue eyes, smiling', tags=True, translations=translations),
            '(blue eyes:1.2), (red coat:1.2), smiling')

    def test_model_input_is_unweighted_and_good_expansion_needs_no_retry(self):
        raw = json.dumps({'common': 'garden, flower bed, stone path',
            'characters': {'1': 'silver hair, blue eyes, embroidered collar'}})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, INFO)) as generate:
            scene, slots, info = run_scene('(garden:1.3)', ('(silver hair:1.2), blue eyes',),
                adapter=PRESAMPLING_ADAPTER, expand_common=True)
        self.assertEqual(generate.call_count, 1)
        self.assertNotIn('(garden:1.3)', generate.call_args.args[1])
        self.assertNotIn('(silver hair:1.2)', generate.call_args.args[1])
        self.assertEqual(scene, '(garden:1.3), flower bed, stone path')
        self.assertEqual(slots, ('(silver hair:1.2), blue eyes, embroidered collar',))
        self.assertEqual(info['warnings'], [])

    def test_disabled_common_is_exact_even_if_korean_and_weighted(self):
        source = '(한국 정원:1.3), 아침'
        raw = json.dumps({'common': 'changed', 'characters': {'1': 'red coat, smiling'}})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, INFO)) as generate:
            scene, _, _ = run_scene(source, ('red coat',), adapter=PRESAMPLING_ADAPTER)
        self.assertEqual(scene, source)
        self.assertEqual(generate.call_count, 1)

    def test_translation_and_expansion_share_the_model_without_sending_weights(self):
        responses = iter((json.dumps({'1': 'blue eyes', '2': 'red coat'}),
            json.dumps({'common': '', 'characters': {'1': 'blue eyes, red coat, smiling, embroidery'}})))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})) as load, \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), INFO)) as generate:
            _, slots, info = run_scene('', ('(파란 눈:1.2), (빨간 코트:1.2)',), adapter=PRESAMPLING_ADAPTER)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(slots, ('(blue eyes:1.2), (red coat:1.2), smiling, embroidery',))
        self.assertNotIn(':1.2', '\n'.join(call.args[1] for call in generate.call_args_list))
        self.assertEqual(info['calls'][0]['stage'], 'weight_translation')

    def test_failed_translation_preserves_weights_and_generated_details(self):
        responses = iter(('bad JSON', json.dumps({'common': '', 'characters': {'1': 'blue eyes, smiling'}})))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), INFO)):
            _, slots, info = run_scene('', ('(파란 눈:1.2)',), adapter=PRESAMPLING_ADAPTER)
        self.assertEqual(slots, ('(파란 눈:1.2), blue eyes, smiling',))
        self.assertEqual(info['warnings'][0]['action'], 'original_weighted_phrase_preserved')

    def test_weights_cannot_cross_numbered_slots_including_ten_people(self):
        sources = tuple(f'(unique identity {i}:1.2)' for i in range(1, 11))
        raw = json.dumps({'common': '', 'characters': {
            str(i): f'unique identity {i}, extra detail {i}' for i in range(1, 11)}})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, INFO)):
            _, slots, _ = run_scene('', sources, adapter=PRESAMPLING_ADAPTER)
        self.assertEqual(slots, tuple(f'{source}, extra detail {i}' for i, source in enumerate(sources, 1)))


if __name__ == '__main__':
    unittest.main()
