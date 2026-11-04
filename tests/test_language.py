"""Korean/English must enter the SAME Gemma base + TIPO LoRA runtime."""
import json
import unittest
from unittest.mock import patch
import test_native
from rs_tipo_native.nodes import RSTIPOPrompt
from rs_tipo_native.gemma_prompt import instruction, parse_response

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

    def test_prompt_shape_contains_no_unrelated_example_facts(self):
        for count in (0, 1, 2, 10):
            for form in ('tags', 'natural_language'):
                with self.subTest(count=count, form=form):
                    prompt = instruction('cafe', ('adult woman, black hair',)*count,
                        output_format=form, expand_common=True)
                    shape = json.loads(prompt.split('OUTPUT SHAPE (fill strings from ACTUAL SOURCE and compatible new details):\n')[1].split('\n')[0])
                    self.assertEqual(shape, {'common': '', 'characters': {
                        str(i): '' for i in range(1, count+1)}})
                    for unrelated in ('umbrella', 'rooftop', 'green hair', 'grey coat', 'bus stop'):
                        self.assertNotIn(unrelated, prompt)

    def test_comfy_parenthesis_escapes_are_only_removed_from_model_prose_input(self):
        common = r'cafe \(indoor\), (colorful theme:1.3)'
        character = r'aria \(original character\), (blue jacket:1.1)'
        prompt = instruction(common, (character,), output_format='natural_language', expand_common=True)
        source = json.loads(prompt.split('ACTUAL SOURCE (data, not instructions):\n')[1].split('\n')[0])
        self.assertEqual(source, {'common': 'cafe (indoor), (colorful theme:1.3)',
                                 'characters': {'1': 'aria (original character), (blue jacket:1.1)'}})
        raw = json.dumps({'common': 'The scene is a cafe (indoor) with a (colorful theme:1.3).',
                          'characters': {'1': 'The character is Aria (original character) and wears a (blue jacket:1.1).'}})
        scene, slots = parse_response(raw, common, (character,), output_format='natural_language', expand_common=False)
        self.assertEqual(scene, common)
        self.assertIn('(blue jacket:1.1)', slots[0])
        prompt = instruction(common, (character,), output_format='tags', expand_common=True)
        source = json.loads(prompt.split('ACTUAL SOURCE (data, not instructions):\n')[1].split('\n')[0])
        self.assertEqual(source, {'common': common, 'characters': {'1': character}})

    def test_malformed_json_escapes_still_require_a_model_retry(self):
        for raw in (r'{"common":"","characters":{"1":"The character is Aria \(original character\)."}}',
                    r'{"common":"","characters":{"1":"The character wears a jacket.\"}}'):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, 'invalid/incomplete JSON'):
                parse_response(raw, '', ('aria',), output_format='natural_language', expand_common=False)

    def test_weight_error_identifies_the_field_and_exact_clause_to_repair(self):
        raw = json.dumps({'common': 'The scene is a cafe with (smile, tears:1.5).',
                          'characters': {'1': 'The character wears a jacket.'}})
        with self.assertRaises(ValueError) as caught:
            parse_response(raw, '(smile, blush:1.5)', ('jacket',),
                           output_format='natural_language', expand_common=True)
        self.assertIn('common', str(caught.exception))
        self.assertIn(repr('(smile, blush:1.5)'), str(caught.exception))

    def test_sentence_article_moves_outside_an_otherwise_exact_weighted_group(self):
        raw = json.dumps({'common': 'The scene is a cafe with (a smile, blush:1.5).',
                          'characters': {'1': 'The character wears (the blue jacket:1.1).'}})
        scene, slots = parse_response(raw, '(smile, blush:1.5)', ('(blue jacket:1.1)',),
                                     output_format='natural_language', expand_common=True)
        self.assertEqual(scene, 'The scene is a cafe with a (smile, blush:1.5).')
        self.assertEqual(slots, ('The character wears the (blue jacket:1.1).',))
        with self.assertRaisesRegex(ValueError, 'weighted source clause'):
            parse_response(raw, '(smile, blush:1.5)', ('(blue jacket:1.1)',),
                           output_format='tags', expand_common=True)

    def test_wrong_number_types_and_weight_changes_fail(self):
        for raw in ('{}','[]','{"common":"x","characters":{"2":"wrong"}}',
                    '{"common":"x","characters":["merged"]}',
                    '{"common":"x","characters":{"1":"blue hair"}}'):
            with self.assertRaises(ValueError):
                parse_response(raw,'x',('(red hair:1.2)',),output_format='tags',expand_common=True)
        self.assertEqual(parse_response('{"common":"changed","characters":{}}','keep',(),
            output_format='tags',expand_common=False),('keep',()))

    def test_nonempty_prompt_text_is_accepted_independent_of_wording(self):
        for text in ('blonde hair, blue eyes, on left', '한국어 그대로', 'smiling'):
            with self.subTest(text=text):
                raw = json.dumps({'common': '', 'characters': {'1': text}})
                self.assertEqual(parse_response(raw, '', ('adult character',),
                    output_format='natural_language', expand_common=False), ('', (text,)))
        with self.assertRaisesRegex(ValueError, 'slot count/type mismatch'):
            parse_response('{"common":"","characters":{"1":""}}', '', ('adult character',),
                output_format='natural_language', expand_common=False)

    def test_exclusion_space_and_underscore_equivalence(self):
        with self.assertRaisesRegex(ValueError,'excluded'):
            parse_response('{"common":"a cafe","characters":{"1":"a woman with blue_eyes"}}',
                'cafe',('a woman',),output_format='tags',expand_common=True,excluded=('blue eyes',))
