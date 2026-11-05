"""Failures at the actual generation/parser boundary, not just fluent prose."""
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_native
from rs_tipo_native.gemma_json import decode_document
from rs_tipo_native.gemma_prompt import parse_response, tag_response
from rs_tipo_native.gemma_runtime import generate, run_scene

INFO = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}


class ResilienceTests(unittest.TestCase):
    def test_extra_braces_do_not_regenerate_or_lose_any_of_ten_participants(self):
        for count in range(11):
            with self.subTest(count=count):
                source = tuple(f'adult {i}, blue coat' for i in range(1, count+1))
                characters = {str(i): f'Adult {i} waves in a blue coat.' for i in range(1, count+1)}
                raw = json.dumps({'common': 'Cafe, soft lighting.', 'characters': characters}) + '\n}'
                with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
                     patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, INFO)) as call:
                    scene, slots, report = run_scene('cafe', source, output_format='natural_language', expand_common=True)
                self.assertEqual((scene, slots), ('Cafe, soft lighting.', tuple(characters.values())))
                self.assertEqual(call.call_count, 1)
                self.assertIn('extra_closing_delimiters', report['calls'][0]['json_repairs'])

    def test_json_repairs_preserve_strings_across_key_orders_and_character_counts(self):
        for count in range(11):
            characters = {str(i): f'Adult {i}: "hello", \\(name\\), }}], blue coat.\nWaving.'
                          for i in range(1, count+1)}
            expected = {'common': 'Cafe "North", \\(room\\), soft light.\nMorning.', 'characters': characters}
            for reverse in (False, True):
                raw = json.dumps(dict(reversed(list(expected.items()))) if reverse else expected)
                for variant in (raw, raw+'\n}\n}', '```json\n'+raw+'\n```',
                                raw.replace('\\\\(', '\\(').replace('\\\\)', '\\)'),
                                raw[:-1]+',}', raw.replace('\\n', '\n')):
                    with self.subTest(count=count, reverse=reverse, variant=variant[:12]):
                        self.assertEqual(decode_document(variant), expected)

    def test_duplicate_keys_and_multiple_documents_are_never_routed(self):
        for raw in ('{"common":"a","common":"b","characters":{}}',
                    '{"common":"a","characters":{"1":"a","1":"b"}}',
                    '{"common":"a","characters":{}} {"common":"b","characters":{}}',
                    '{"common":"a","characters":{}} arbitrary trailing prose'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_response(raw, 'cafe', ('adult',), output_format='natural_language', expand_common=True)

    def test_recoverable_missing_slots_keep_other_numbered_outputs(self):
        raw = '{"characters":{"2":"Second adult holds a book.","1":null}}'
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, INFO)) as call:
            scene, slots, report = run_scene('cafe', ('first adult', 'second adult', 'third adult'),
                output_format='natural_language', expand_common=True)
        self.assertEqual((scene, slots), ('cafe', ('first adult', 'Second adult holds a book.', 'third adult')))
        self.assertEqual(call.call_count, 2)
        self.assertEqual(report['calls'][-1]['source_restored_fields'], ['common', 'character1', 'character3'])

    def test_unknown_keys_lists_and_duplicate_keys_restore_sources_without_guessing(self):
        for raw in ('{"common":"Elsewhere","characters":{"2":"Wrong adult"}}',
                    '{"common":"Elsewhere","characters":["Unknown owner"]}',
                    '{"common":"Elsewhere","characters":{"1":"A","1":"B"}}', 'not JSON', ''):
            with self.subTest(raw=raw), \
                 patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
                 patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, INFO)) as call:
                scene, slots, report = run_scene('cafe', ('original adult',),
                    output_format='natural_language', expand_common=True)
                self.assertEqual((scene, slots), ('cafe', ('original adult',)))
                self.assertEqual(call.call_count, 2)
                self.assertEqual(report['warnings'][0]['action'], 'source_restored_after_retry')

    def test_one_lost_weight_does_not_discard_valid_neighboring_descriptions(self):
        raw = '{"common":"Cafe, morning light.","characters":{"1":"Blue coat.","2":"Adult reading."}}'
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, INFO)):
            scene, slots, report = run_scene('cafe', ('(blue coat:1.2)', 'adult'),
                output_format='natural_language', expand_common=True)
        self.assertEqual((scene, slots), ('Cafe, morning light.', ('(blue coat:1.2).', 'Adult reading.')))
        self.assertNotIn('source_restored_fields', report['calls'][-1])
        self.assertEqual(report['model_calls'], 1)

    def test_complete_json_at_token_limit_is_accepted(self):
        raw = '{"common":"cafe","characters":{"1":"Adult waving."}}'
        info = {**INFO, 'finish_reason': 'length'}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, info)) as call:
            _, slots, report = run_scene('cafe', ('adult',), output_format='natural_language')
        self.assertEqual(slots, ('Adult waving.',))
        self.assertEqual(call.call_count, 1)
        self.assertEqual(report['calls'][0]['finish_reason'], 'length')

    def test_tag_wrappers_and_multiline_text_preserve_content(self):
        expected = 'blue coat, lighting is soft and even, looking back'
        for raw in (expected, '```\n'+expected+'\n```', '```text\n'+expected+'\n```',
                    json.dumps(expected), json.dumps(expected.split(', ')),
                    json.dumps({'tags': expected}), '```json\n'+json.dumps({'tags': expected})+'\n```'):
            with self.subTest(raw=raw):
                self.assertEqual(tag_response(raw), expected)
        for raw in ('{"common":"cafe","characters":{}}', '[1,2]', '{"tags":null}', '{"tags":"a","prompt":"b"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                tag_response(raw)

    def test_attention_punctuation_is_prompt_text_not_a_content_filter(self):
        for text in ('[blue coat], looking back', '{blue coat}, looking back',
                     '{{blue coat}}, looking back', '(blue coat:1.2), determined',
                     '[soft light, warm light], looking at viewer', '"blue coat, looking back'):
            with self.subTest(text=text):
                self.assertEqual(tag_response(text), text)

    def test_empty_tag_outputs_preserve_other_slots_and_caption_only_for_failed_slot(self):
        raw = '{"common":"Cafe.","characters":{"1":"Adult waving.","2":"Adult reading."}}'
        responses = iter((raw, 'cafe, soft light', '', '', 'reading, book'))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **kw: (next(responses), INFO)):
            scene, slots, report = run_scene('cafe', ('first adult', 'second adult'), output_format='tags', expand_common=True)
        self.assertEqual((scene, slots), ('cafe, soft light', ('Adult waving.', 'reading, book')))
        self.assertEqual(report['warnings'][0]['fields'], [1])
        self.assertEqual(report['model_calls'], 5)

    def test_gpu_runtime_failures_and_user_cancellation_are_not_converted_to_success(self):
        for exc in (RuntimeError('CUDA out of memory'), InterruptedError('User cancelled')):
            with self.subTest(exc=exc), \
                 patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
                 patch('rs_tipo_native.gemma_runtime.generate', side_effect=exc) as call:
                with self.assertRaises(type(exc)):
                    run_scene('cafe', ('adult',))
                self.assertEqual(call.call_count, 1)

    def test_invalid_request_fails_before_loading_weights(self):
        with patch('rs_tipo_native.gemma_runtime.load_gemma') as load:
            with self.assertRaisesRegex(ValueError, '24000'):
                run_scene('cafe '*5000, ())
        load.assert_not_called()

    def test_source_exclusion_conflict_fails_before_loading_weights(self):
        with patch('rs_tipo_native.gemma_runtime.load_gemma') as load:
            with self.assertRaisesRegex(ValueError, 'source conflicts'):
                run_scene('cafe', ('Adult with blue eyes.',), excluded=('blue eyes',))
        load.assert_not_called()

    def test_invalid_tag_json_retries_once_and_recovers_own_caption(self):
        raw = '{"common":"","characters":{"1":"Adult wearing a blue coat."}}'
        responses = iter((raw, '{"common":"wrong slot"}', '{"tags":null}'))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **kw: (next(responses), INFO)):
            _, slots, report = run_scene('', ('adult',), output_format='tags')
        self.assertIn('Adult wearing a blue coat.', slots[0])
        self.assertTrue(report['calls'][-1]['caption_preserved'])
        self.assertEqual(report['model_calls'], 3)

    def test_generation_reports_empty_and_capped_output_and_always_cleans_prefetch(self):
        # Exercise real generate(), including the previously uncaught early
        # RuntimeErrors, without importing a GPU or a second inference engine.
        comfy = ModuleType('comfy'); comfy.__path__ = []
        mm = ModuleType('comfy.model_management')
        mm.throw_exception_if_processing_interrupted = Mock()
        utils = ModuleType('comfy.utils'); utils.ProgressBar = Mock()
        cleanup = Mock()
        prefetch = SimpleNamespace(cleanup_prefetch_queues=cleanup)
        for raw, ids, expected in (('', [], 'empty'), ('abc', [1,2,3], 'length'), ('abc', [1], 'stop')):
            with self.subTest(expected=expected):
                transformer = SimpleNamespace(sample_token=lambda *args: None)
                original = transformer.sample_token
                clip = SimpleNamespace(tokenize=lambda *a, **kw: {'gemma4': []},
                    cond_stage_model=SimpleNamespace(gemma4=SimpleNamespace(transformer=transformer)),
                    generate=lambda *a, **kw: ids, decode=lambda *a: raw)
                with patch.dict(sys.modules, {'comfy':comfy, 'comfy.model_management':mm,
                        'comfy.utils':utils, 'comfy.model_prefetch':prefetch}):
                    text, report = generate(clip, 'cafe', max_tokens=3)
                self.assertEqual(text, raw)
                self.assertEqual(report['finish_reason'], expected)
                self.assertIs(transformer.sample_token, original)
        self.assertEqual(cleanup.call_count, 3)

    def test_public_node_restored_korean_is_not_reported_as_translated(self):
        from rs_tipo_native.nodes import RSTIPOPrompt
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=('', INFO)), \
             patch('rs_tipo_native.interface.pipe_adapter', return_value=None):
            result = RSTIPOPrompt().expand(character_count=1, common_prompt='카페', character1='성인, 파란 재킷')
        report = json.loads(result['ui']['tipo_report'][0])
        self.assertEqual(report['slots'][1]['output'], '성인, 파란 재킷')
        self.assertEqual(report['slots'][1]['expansion_status'], 'unchanged')
        self.assertIn('source_restored_after_retry', report['slots'][1]['recovery_actions'])
