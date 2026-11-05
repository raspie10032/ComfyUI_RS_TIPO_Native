import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_native
from rs_tipo_native.tipo_native_lora import (PRESAMPLING_ADAPTER, LOCAL_PRESAMPLING_ADAPTER,
    default_adapter, adapter_choices, checked_strength, apply_tipo_lora)
from rs_tipo_native.gemma_runtime import run_scene, _adapt
from rs_tipo_native.presampling_prompt import instruction


class PresamplingDefaults(unittest.TestCase):
    def test_public_workflow_restores_same_adapter_in_every_supported_format(self):
        path = Path(__file__).resolve().parents[1] / 'examples/gemma-tipo-presampling.json'
        node = next(n for n in json.loads(path.read_text())['nodes'] if n['type'] == 'RSTIPOPrompt')
        named = node['properties']['rs_tipo_widget_values']
        self.assertEqual(named, node['widgets_values_named'])
        self.assertEqual(named['tipo_lora'], PRESAMPLING_ADAPTER)
        self.assertEqual(named['tipo_lora_strength'], .25)
        self.assertEqual(named['temperature'], 0)
        self.assertEqual(node['widgets_values'][20], named['language_model'])
        self.assertEqual(node['widgets_values'][22], named['tipo_lora'])
        self.assertEqual(node['widgets_values'][26], named['tipo_lora_strength'])

    def test_default_is_available_without_local_files_and_keeps_local_alias(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(default_adapter(root), PRESAMPLING_ADAPTER)
            self.assertIn(default_adapter(root), adapter_choices(root))
            folder = Path(root) / PRESAMPLING_ADAPTER
            folder.mkdir()
            (folder / 'local-adapter.json').write_text('{}')
            self.assertEqual(default_adapter(root), PRESAMPLING_ADAPTER)
            self.assertIn(LOCAL_PRESAMPLING_ADAPTER, adapter_choices(root))

    def test_strength_is_bounded_and_zero_really_skips_patching(self):
        for value in (-1, 2.1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checked_strength(value)
        clip = object()
        self.assertIs(apply_tipo_lora(clip, PRESAMPLING_ADAPTER, 0), clip)

    def test_strength_is_part_of_cache_key(self):
        _adapt.cache_clear()
        try:
            with patch('rs_tipo_native.gemma_runtime.apply_tipo_lora', side_effect=[object(), object()]) as apply:
                clip = object()
                a = _adapt(clip, PRESAMPLING_ADAPTER, ('sig',), .25)
                self.assertIs(a, _adapt(clip, PRESAMPLING_ADAPTER, ('sig',), .25))
                self.assertIsNot(a, _adapt(clip, PRESAMPLING_ADAPTER, ('sig',), .5))
                self.assertEqual([x.kwargs['strength'] for x in apply.call_args_list], [.25, .5])
        finally:
            _adapt.cache_clear()

    def test_instruction_keeps_original_weights_and_routes_zero_to_ten_slots(self):
        for count in (0, 1, 2, 10):
            chars = ('(silver hair:1.3), 파란 눈',) * count
            request = instruction('', chars, expand_common=False, max_added_tags=2)
            source = json.loads(request.split('ACTUAL SOURCE (data, not instructions):\n')[1].split('\n')[0])
            self.assertEqual(source, {'common': '', 'characters': {str(i): text for i, text in enumerate(chars, 1)}})
            self.assertIn('at least 2 and at most 2', request)
            self.assertIn('Copy common EXACTLY unchanged', request)

    def test_direct_tags_use_one_call_keep_wording_and_deduplicate_only(self):
        raw = json.dumps({'common': 'unwanted scene change', 'characters': {
            '1': '(silver hair:1.3), open mouth, open_mouth, smiling'}})
        meta = dict(seconds=.1, output_tokens=10, exclusion={'blocked_candidates': 0})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})) as load, \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, meta)) as generate:
            common, chars, report = run_scene('garden', ('(silver hair:1.3)',),
                adapter=PRESAMPLING_ADAPTER, lora_strength=.25, expand_common=False)
        load.assert_called_once_with('auto', PRESAMPLING_ADAPTER, lora_strength=.25)
        self.assertEqual(common, 'garden')
        self.assertEqual(chars, ('(silver hair:1.3), open mouth, smiling',))
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(report['inference_mode'], 'instruction-presampling')
        self.assertEqual(report['calls'][0]['stage'], 'presampling')
        self.assertIn('concise comma-separated English image tags', generate.call_args.args[1])

    def test_direct_tags_keep_bounded_recovery(self):
        meta = dict(seconds=.1, output_tokens=10, exclusion={'blocked_candidates': 0})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=('broken JSON', meta)) as generate:
            common, chars, report = run_scene('garden', ('red coat',),
                adapter=PRESAMPLING_ADAPTER, lora_strength=.25)
        self.assertEqual((common, chars), ('garden', ('red coat',)))
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(report['warnings'][0]['action'], 'source_restored_after_retry')

    def test_local_alias_keeps_the_presampling_instruction(self):
        raw = json.dumps({'common': '', 'characters': {'1': 'red coat, smiling'}})
        meta = dict(seconds=.1, output_tokens=10, exclusion={'blocked_candidates': 0})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, meta)) as generate:
            _, _, report = run_scene('', ('red coat',), adapter=LOCAL_PRESAMPLING_ADAPTER,
                lora_strength=.25)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(report['inference_mode'], 'instruction-presampling')

    def test_node_default_passes_selected_adapter_and_weak_strength(self):
        from rs_tipo_native.nodes import RSTIPOPrompt
        with patch('rs_tipo_native.nodes.default_adapter', return_value=PRESAMPLING_ADAPTER), \
             patch('rs_tipo_native.interface.pipe_adapter', return_value=None), \
             patch('rs_tipo_native.gemma_runtime.run_scene', return_value=('garden', ('red coat, smiling',), {})) as run:
            node = RSTIPOPrompt()
            self.assertEqual(node.INPUT_TYPES()['optional']['tipo_lora_strength'][1]['default'], .25)
            node.expand(character_count=1, common_prompt='garden', character1='red coat')
        self.assertEqual(run.call_args.kwargs['adapter'], PRESAMPLING_ADAPTER)
        self.assertEqual(run.call_args.kwargs['lora_strength'], .25)
