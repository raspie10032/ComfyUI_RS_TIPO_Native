"""Korean/English must enter the SAME Gemma base + TIPO LoRA runtime."""
import json
import unittest
from unittest.mock import patch
import test_native
from rs_tipo_native.nodes import RSTIPOPrompt
from rs_tipo_native.gemma_prompt import instruction, parse_response, validate_prose

class LanguageTests(unittest.TestCase):
    def test_one_adapted_model_for_both_languages_and_formats(self):
        for source in ('cafe', '오후 카페'):
            for form in ('tags', 'natural_language'):
                report={'adapter':'pilot-002','lora_targets':190,'model_calls':1,'seconds':.01,'raw_output':'raw'}
                with patch('rs_tipo_native.interface.pipe_adapter',return_value=None), \
                     patch('rs_tipo_native.nodes.acquire_model',side_effect=AssertionError('Separate Kohaku model forbidden')), \
                     patch('rs_tipo_native.gemma_runtime.run_scene',return_value=(source,('blonde hair',),report)) as run:
                    result=RSTIPOPrompt().expand(character_count=1,common_prompt=source,character1='blonde hair',output_format=form)
                self.assertEqual(run.call_count,1)
                self.assertEqual(run.call_args.kwargs['adapter'],'pilot-002')
                self.assertEqual(run.call_args.kwargs['output_format'],form)
                self.assertEqual(result['result'][1:3],(source,'blonde hair'))

    def test_joint_context_and_role_ownership_in_request(self):
        prompt=instruction('차 준비',('찻잔 받치기','차 따르기','쿠키 놓기'),output_format='natural_language')
        self.assertIn('찻잔 받치기',prompt);self.assertIn('차 따르기',prompt);self.assertIn('쿠키 놓기',prompt)
        self.assertIn('offering/receiving',prompt)
        self.assertIn('["1", "2", "3"]',prompt)

    def test_wrong_number_types_and_weight_changes_fail(self):
        for raw in ('{}','[]','{"common":"x","characters":{"2":"wrong"}}',
                    '{"common":"x","characters":["merged"]}',
                    '{"common":"x","characters":{"1":"blue hair"}}'):
            with self.assertRaises(ValueError):
                parse_response(raw,'x',('(red hair:1.2)',),output_format='tags',expand_common=True)
        self.assertEqual(parse_response('{"common":"changed","characters":{}}','keep',(),
            output_format='tags',expand_common=False),('keep',()))

    def test_natural_format_must_be_actual_english_prose(self):
        for text in ('blonde hair, blue eyes, on left','한국어 그대로',''):
            with self.assertRaises(ValueError): validate_prose((),(text,))
        validate_prose(('Two people are in a cafe.',),('The first character has blonde hair.',))

    def test_exclusion_space_and_underscore_equivalence(self):
        with self.assertRaisesRegex(ValueError,'excluded'):
            parse_response('{"common":"a cafe","characters":{"1":"a woman with blue_eyes"}}',
                'cafe',('a woman',),output_format='tags',expand_common=True,excluded=('blue eyes',))
