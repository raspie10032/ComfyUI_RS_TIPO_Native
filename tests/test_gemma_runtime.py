import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import test_native
from rs_tipo_native.gemma_runtime import token_constraints, run_scene
from rs_tipo_native.gemma_prompt import (caption_anchors, deduplicate_tag_rendering, scene_count_anchors,
    prose_source, labelled_character_fields)

class GemmaRuntimeTests(unittest.TestCase):
    def test_blank_common_reaches_public_node_with_original_weights_and_new_tags(self):
        from rs_tipo_native.nodes import RSTIPOPrompt
        raw = json.dumps({'common': '', 'characters': {'1': 'The adult wears a blue jacket.'}})
        responses = iter((raw, 'colorful_theme, cafe, rim lighting', 'blue jacket, waving'))
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                 (next(responses), info)), \
             patch('rs_tipo_native.interface.pipe_adapter', return_value=None):
            result = RSTIPOPrompt().expand(character_count=1, expand_common=True,
                common_prompt='(colorful theme:1.3), cafe', character1='(blue jacket:1.2), adult',
                tipo_lora='pilot-002', tipo_lora_strength=1.0)
        self.assertEqual(result['result'][1], '(colorful theme:1.3), cafe, rim lighting')
        self.assertEqual(result['result'][2], '(blue jacket:1.2), adult, waving')
        report = json.loads(result['ui']['tipo_report'][0])
        self.assertEqual(report['calls'][0]['source_restored_fields'], ['common'])
        self.assertEqual(report['slots'][0]['added_tags'], ['rim lighting'])

    def test_blank_common_keeps_valid_characters_and_continues_to_tags_without_retry(self):
        raw = json.dumps({'common': '', 'characters': {
            '1': 'The adult wears a red coat and waves.',
            '2': 'The adult wears a blue jacket and holds a book.'}})
        responses = iter((raw, 'cafe, soft lighting, soft_lighting',
            'red coat, waving', 'blue jacket, holding book'))
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                 (next(responses), info)) as generate:
            scene, slots, report = run_scene('(colorful theme:1.3), cafe',
                ('adult, red coat', 'adult, blue jacket'), output_format='tags', expand_common=True)
        self.assertEqual(scene, '(colorful theme:1.3), cafe, soft lighting')
        self.assertEqual(slots, ('red coat, waving', 'blue jacket, holding book'))
        self.assertEqual(generate.call_count, 4)
        self.assertIn('CAPTION:\ncolorful theme, cafe\nTAGS:', generate.call_args_list[1].args[1])
        self.assertEqual(report['calls'][0]['source_restored_fields'], ['common'])
        self.assertNotIn('validation_error', report['calls'][0])

    def test_blank_character_restores_own_slot_without_retry_or_changing_other_outputs(self):
        raw = json.dumps({'common': 'The scene is a cafe.',
            'characters': {'2': 'The second adult waves.', '1': '   '}})
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, info)) as generate:
            scene, slots, report = run_scene('cafe', ('first adult', 'second adult'),
                output_format='natural_language', expand_common=True)
        self.assertEqual((scene, slots), ('The scene is a cafe.', ('first adult', 'The second adult waves.')))
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(report['calls'][0]['source_restored_fields'], ['character1'])


    def test_tag_list_planning_continues_to_tag_rendering_without_grammar_retry(self):
        for count in (0, 1, 2, 10):
            with self.subTest(count=count):
                captions = ('1girl, blue eyes, white shirt, looking at camera',
                    'Appearance: silver hair. Outfit: blue jacket.', '웃으며 손을 흔든다.', 'smiling')
                characters = tuple(captions[i % len(captions)] for i in range(count))
                raw = json.dumps({'common': 'cafe, warm lighting',
                    'characters': {str(i): text for i, text in enumerate(characters, 1)}})
                rendered = tuple(f'character {i}, waving, waving' for i in range(1, count+1))
                responses = iter((raw, 'cafe, warm lighting, warm_lighting', *rendered))
                info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
                with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
                     patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                         (next(responses), info)) as generate:
                    scene, slots, report = run_scene('cafe', ('adult character',)*count,
                        output_format='tags', expand_common=True)
                self.assertEqual(scene, 'cafe, warm lighting')
                prefixes = ('blue eyes, white shirt, ', 'silver hair, blue jacket, ', '', '')
                self.assertEqual(slots, tuple(prefixes[(i-1) % len(prefixes)]+f'character {i}, waving'
                    for i in range(1, count+1)))
                self.assertEqual(generate.call_count, count+2)
                self.assertEqual([call['caption'] for call in report['calls'][2:]], list(characters))
                self.assertNotIn('validation_error', report['calls'][0])

    def test_natural_output_wording_is_preserved_without_grammar_retry(self):
        for text in ('blue eyes, white shirt', 'Appearance: silver hair. Outfit: jacket.',
                     '웃으며 손을 흔든다.', 'smiling'):
            with self.subTest(text=text):
                raw = json.dumps({'common': 'warm lighting', 'characters': {'1': text}})
                info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
                with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
                     patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, info)) as generate:
                    scene, slots, report = run_scene('cafe', ('adult character',),
                        output_format='natural_language', expand_common=True)
                self.assertEqual((scene, slots), ('warm lighting', (text,)))
                self.assertEqual(generate.call_count, 1)
                self.assertNotIn('validation_error', report['calls'][0])

    def test_zero_detail_budget_allows_a_faithful_common_rewrite(self):
        raw = json.dumps({'common': 'The scene is a cafe.', 'characters': {}})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw,
                 {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}})) as generate:
            scene, slots, report = run_scene('cafe', (), output_format='natural_language',
                expand_common=True, max_added_tags=0)
        self.assertEqual(scene, 'The scene is a cafe.')
        self.assertEqual(slots, ())
        self.assertEqual(report['model_calls'], 1)
        self.assertEqual(generate.call_count, 1)

    def test_empty_exclusions_skip_decode_vocab_scan_and_token_read(self):
        from unittest.mock import Mock
        token = object()  # Must be returned unchanged, without indexing/item().
        sampler = Mock(return_value=token)
        model = SimpleNamespace(sample_token=sampler)
        logits, generator = object(), object()
        decode, interrupt, progress = Mock(), Mock(), Mock()
        with patch('torch.isfinite', side_effect=AssertionError('unnecessary vocab scan')):
            with token_constraints(model, decode, (), interrupt, progress) as report:
                self.assertIs(model.sample_token(logits, .3, generator=generator), token)
                self.assertIs(model.sample_token(logits, .3, generator=generator), token)
        self.assertIs(model.sample_token, sampler)
        self.assertEqual(sampler.call_count, 2)
        sampler.assert_called_with(logits, .3, generator=generator)
        decode.assert_not_called()
        self.assertEqual(interrupt.call_count, 2)
        self.assertEqual([call.args for call in progress.call_args_list], [(1,), (2,)])
        self.assertEqual(report['blocked_candidates'], 0)
        self.assertEqual(report['hits'], {})

    def test_empty_exclusions_preserve_sampled_tokens_and_generator_state(self):
        class Transformer:
            def sample_token(self, logits, generator):
                return torch.multinomial(logits.softmax(-1), 1, generator=generator)
        model = Transformer()
        logits = torch.tensor([[1., .5, -.5]])
        baseline = torch.Generator().manual_seed(91)
        actual = torch.Generator().manual_seed(91)
        expected = [model.sample_token(logits, baseline) for _ in range(12)]
        with token_constraints(model, str, (), lambda: None, lambda n: None):
            result = [model.sample_token(logits, actual) for _ in range(12)]
        self.assertTrue(torch.equal(torch.cat(result), torch.cat(expected)))
        self.assertTrue(torch.equal(actual.get_state(), baseline.get_state()))
        self.assertNotIn('sample_token', model.__dict__)

    def test_empty_exclusions_restore_sampler_after_progress_error(self):
        original = lambda *args: object()
        model = SimpleNamespace(sample_token=original)
        def fail(n):
            raise RuntimeError('progress failed')
        with self.assertRaisesRegex(RuntimeError, 'progress failed'):
            with token_constraints(model, str, (), lambda: None, fail):
                model.sample_token(object())
        self.assertIs(model.sample_token, original)

    def test_exclusion_rejects_next_token_and_restores_instance(self):
        class Transformer:
            def sample_token(self,logits,*args,**kwargs): return logits.argmax(-1)
        model=Transformer();pieces={0:'blue ',1:'eyes',2:'hair'}
        with token_constraints(model,lambda ids:''.join(pieces[i] for i in ids),('blue eyes',),lambda:None,lambda n:None) as report:
            self.assertEqual(int(model.sample_token(torch.tensor([[5.,0.,0.]]))),0)
            self.assertEqual(int(model.sample_token(torch.tensor([[0.,5.,1.]]))),2)
        self.assertNotIn('sample_token',model.__dict__)
        self.assertEqual(report['blocked_candidates'],1)
        self.assertEqual(int(model.sample_token(torch.tensor([[0.,5.,1.]]))),1)

    def test_cancellation_restores_sampler(self):
        original=lambda *args,**kwargs:None
        model=SimpleNamespace(sample_token=original)
        def cancel():raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):
            with token_constraints(model,str,(),cancel,lambda n:None):model.sample_token(torch.zeros(1,3))
        self.assertIs(model.sample_token,original)

    def test_prose_and_tag_rendering_reuse_exact_same_adapted_clip(self):
        clip=object();seen=[]
        responses=iter([json.dumps({'common':'Two people are in a cafe.','characters':{'1':'The first character has black hair and offers a cup.'}}),
                        'two people, cafe','black hair, offering cup'])
        def generate(actual,prompt,**kwargs):
            self.assertIs(actual,clip);seen.append(prompt)
            return next(responses),{'seconds':.1,'output_tokens':5,'exclusion':{'blocked_candidates':0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma',return_value=(clip,{'adapter':'pilot-002','lora_targets':190})), \
             patch('rs_tipo_native.gemma_runtime.generate',side_effect=generate):
            common,slots,info=run_scene('카페',('검은 머리, 컵 건네기',),expand_common=True)
        self.assertEqual(slots,('black hair, offering cup',));self.assertEqual(info['model_calls'],3)
        self.assertIn('검은 머리',seen[0]);self.assertIn('검은 머리',seen[0])

    def test_incorrect_slots_retry_same_model_then_restore_sources(self):
        info={'seconds':.1,'output_tokens':5,'exclusion':{'blocked_candidates':0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma',return_value=(object(),{})) as load, \
             patch('rs_tipo_native.gemma_runtime.generate',return_value=('{}',info)) as generate:
            scene, slots, report = run_scene('cafe',('black hair',),output_format='natural_language')
        self.assertEqual((scene, slots), ('cafe', ('black hair',)))
        self.assertEqual(report['warnings'][0]['action'], 'source_restored_after_retry')
        self.assertEqual(load.call_count,1);self.assertEqual(generate.call_count,2)

    def test_invalid_structure_retry_retains_raw_responses(self):
        raw = '{"common": "", "characters": {}}'
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw,
                 {'seconds': .1, 'output_tokens': 5})) as generate:
            _, slots, report = run_scene('', ('adult character',), output_format='natural_language')
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(slots, ('adult character',))
        self.assertEqual([call['raw'] for call in report['calls']], [raw, raw])
        self.assertIn('slot count/type mismatch', report['calls'][-1]['validation_error'])
        self.assertIn('Repair the reported contract error only', generate.call_args_list[1].args[1])
        self.assertNotIn('Rewrite the failed fields as English sentences', generate.call_args_list[1].args[1])

    def test_labelled_input_makes_outfit_meaning_explicit_without_changing_plain_input(self):
        source = '외형: (silver hair:1.2), blue eyes. 의상: satin ball gown, jeweled hairpin. 작은 상자를 연다.'
        normalized = prose_source(source)
        self.assertIn('(silver hair:1.2), blue eyes', normalized)
        self.assertIn('The character is wearing satin ball gown, jeweled hairpin.', normalized)
        self.assertTrue(normalized.endswith('작은 상자를 연다.'))
        self.assertEqual(prose_source('silver hair, wearing a red dress'), 'silver hair, wearing a red dress')
        self.assertEqual(labelled_character_fields('Appearance: silver hair. Outfit: dress.'),
                         ('silver hair', 'dress', ''))
        carried = '외형: silver hair. 의상: retro astronaut suit, helmet carried under arm. 카메라를 든다.'
        self.assertIn('wearing retro astronaut suit.', prose_source(carried))
        self.assertIn('also has helmet carried under arm.', prose_source(carried))

    def test_failed_repair_restores_source_weights_and_avoids_excluded_tags(self):
        for source, excluded, repaired in (
            ('(silver hair:1.2)', (), 'The character has silver hair.'),
            ('silver hair', ('glasses',), 'The character has silver hair and glasses.'),
        ):
            with self.subTest(source=source, excluded=excluded):
                responses = iter([
                    '{}',
                    json.dumps({'common': '', 'characters': {'1': repaired}})])
                with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
                     patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), {'seconds': .1, 'output_tokens': 5})):
                    _, slots, report = run_scene('', (source,), excluded=excluded, output_format='natural_language')
                if excluded:
                    self.assertEqual(slots, (source,))
                    self.assertEqual(report['warnings'][0]['action'], 'source_restored_after_retry')
                else:
                    self.assertEqual(slots, ('The character has (silver hair:1.2).',))
                    self.assertEqual(report['warnings'], [])

    def test_caption_tags_must_keep_explicit_attributes(self):
        self.assertEqual(caption_anchors('A woman with black hair and brown eyes wears a blue cardigan, standing behind.'),
                         ('black hair','brown eyes','blue cardigan','behind'))

    def test_single_tag_json_value_is_unwrapped_without_retry(self):
        clip = object()
        raw = json.dumps({'common': '', 'characters': {'1': 'The woman has black hair.'}})
        responses = iter([raw, '{"tags": "black hair"}', 'black hair'])
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        def generate(actual, prompt, **kwargs):
            self.assertIs(actual, clip)
            return next(responses), info
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(clip, {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=generate):
            _, slots, report = run_scene('', ('black hair',), output_format='tags')
        self.assertEqual(slots, ('black hair',))
        self.assertEqual(report['model_calls'], 2)
        self.assertNotIn('format_error', report['calls'][1])
        self.assertEqual(report['calls'][1]['raw'], '{"tags": "black hair"}')

    def test_tag_retry_is_bounded_and_preserves_caption_with_warning(self):
        raw = json.dumps({'common': '', 'characters': {'1': 'The woman has black hair.'}})
        responses = iter([raw, '', ''])
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), info)):
            _, slots, report = run_scene('', ('black hair',), output_format='tags')
        self.assertEqual(slots, ('black hair, The woman has black hair.',))
        self.assertEqual(report['model_calls'], 3)
        self.assertTrue(report['calls'][-1]['caption_preserved'])
        self.assertEqual(report['warnings'][0]['action'], 'caption_preserved_after_retry')

    def test_sentence_wording_in_tag_stage_is_preserved_without_retry(self):
        caption = 'The character has black hair.'
        raw = json.dumps({'common': '', 'characters': {'1': caption}})
        responses = iter([raw, caption])
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), info)):
            _, slots, report = run_scene('', ('black hair',), output_format='tags')
        self.assertEqual(slots, ('black hair, ' + caption,))
        self.assertNotIn('format_error', report['calls'][1])
        self.assertEqual(report['model_calls'], 2)

    def test_sentence_punctuation_does_not_reject_generated_content(self):
        caption = 'A colorful cafe. The warm glow casts soft shadows across the tables.'
        raw = json.dumps({'common': caption, 'characters': {}})
        responses = iter((raw, caption))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                 (next(responses), {'seconds': .1, 'output_tokens': 5,
                    'exclusion': {'blocked_candidates': 0}})):
            scene, _, report = run_scene('colorful cafe', (), output_format='tags', expand_common=True)
        self.assertEqual(scene, caption)
        self.assertNotIn('format_error', report['calls'][1])
        self.assertEqual(report['model_calls'], 2)

    def test_reported_common_lighting_clause_survives_with_only_duplicates_removed(self):
        caption = ('The scene is a colorful living room with a chair. '
            'The lighting is soft and even, highlighting the details of the room.')
        raw = json.dumps({'common': caption, 'characters': {}})
        rendered = ('colorful theme, open mouth, smile, one eye closed, half-closed eyes, '
            'blush, living roo m, on a chair, lighting is soft and even, '
            'highlighting the details of the room, colorful_theme, lighting is soft and even')
        responses = iter((raw, rendered))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                 (next(responses), {'seconds': .1, 'output_tokens': 5,
                    'exclusion': {'blocked_candidates': 0}})):
            scene, _, report = run_scene('(colorful theme:1.3), living roo m, on chair', (),
                output_format='tags', expand_common=True)
        self.assertEqual(scene, '(colorful theme:1.3), ' +
            rendered.rsplit(', colorful_theme', 1)[0].removeprefix('colorful theme, '))
        self.assertEqual(report['model_calls'], 2)
        self.assertNotIn('format_error', report['calls'][1])

    def test_wrapped_tag_lines_keep_later_details_and_deduplicate(self):
        raw = json.dumps({'common': 'The scene is a cafe.', 'characters': {}})
        responses = iter((raw, 'cafe,\nsoft lighting,\r\nbokeh,\nsoft_lighting'))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                 (next(responses), {'seconds': .1, 'output_tokens': 5,
                    'exclusion': {'blocked_candidates': 0}})):
            scene, _, report = run_scene('cafe', (), output_format='tags', expand_common=True)
        self.assertEqual(scene, 'cafe, soft lighting, bokeh')
        self.assertEqual(report['model_calls'], 2)

    def test_short_common_caption_continues_to_tags_without_rewriting(self):
        responses = iter((json.dumps({'common': 'cafe', 'characters': {}}),
            '2girls, cafe, soft lighting'))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                 (next(responses), {'seconds': .1, 'output_tokens': 5,
                    'exclusion': {'blocked_candidates': 0}})) as generate:
            scene, _, _ = run_scene('여성 두 명, 카페', (), output_format='tags', expand_common=True)
        self.assertEqual(scene, '2girls, cafe, soft lighting')
        self.assertEqual(generate.call_count, 2)
        self.assertIn('CAPTION:\ncafe\nTAGS:', generate.call_args_list[1].args[1])
        self.assertNotIn('only the NEW visual details', generate.call_args_list[1].args[1])

    def test_tag_planning_does_not_require_image_weight_syntax_in_english_prose(self):
        raw = json.dumps({'common': '', 'characters': {'1': 'The character has red hair and blue eyes.'}})
        responses = iter([raw, 'red hair, blue eyes'])
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), info)):
            _, slots, _ = run_scene('', ('(red hair, blue eyes:1.2)',), output_format='tags')
        self.assertEqual(slots, ('(red hair, blue eyes:1.2)',))

    def test_scene_count_comes_from_explicit_caption(self):
        self.assertEqual(scene_count_anchors('Two adult women are in a cafe.'),('2girls',))
        self.assertEqual(scene_count_anchors('The scene is a garden.'),())

    def test_derived_tag_prefix_cannot_reintroduce_exclusion(self):
        raw=json.dumps({'common':'Two adult women are in a cafe.','characters':{'1':'The character has black hair.'}})
        responses=iter([raw,'cafe','black hair'])
        info={'seconds':.1,'output_tokens':5,'exclusion':{'blocked_candidates':0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma',return_value=(object(),{})), \
             patch('rs_tipo_native.gemma_runtime.generate',side_effect=lambda *a,**kw:(next(responses),info)):
            scene, slots, report = run_scene('two women',('black hair',),expand_common=True,excluded=('2girls',))
        self.assertNotIn('2girls', scene)
        self.assertEqual(len(slots), 1)

    def test_common_restatement_does_not_trigger_content_retry(self):
        raw = json.dumps({'common': 'The scene is a cafe.',
            'characters': {'1': 'The character has red hair and looks down.'}})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw,
                 {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}})) as generate:
            scene, slots, report = run_scene('cafe', ('red hair',),
                output_format='natural_language', expand_common=True)
        self.assertEqual(scene, 'The scene is a cafe.')
        self.assertEqual(slots, ('The character has red hair and looks down.',))
        self.assertEqual(generate.call_count, 1)
        self.assertNotIn('validation_error', report['calls'][0])

    def test_valid_outfit_wording_is_not_rejected_by_source_vocabulary(self):
        caption = 'The character wears an elegant satin gown and a pearl choker.'
        raw = json.dumps({'common': '', 'characters': {'1': caption}})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw,
                 {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}})) as generate:
            _, slots, report = run_scene('',
                ('외형: silver hair. 의상: satin ball gown, pearl necklace.',),
                output_format='natural_language')
        self.assertEqual(slots, (caption,))
        self.assertEqual(generate.call_count, 1)
        self.assertNotIn('validation_error', report['calls'][0])

    def test_new_tag_vocabulary_survives_without_grounding_retry(self):
        responses = iter((json.dumps({'common': 'The lighting is soft with gentle shadows.',
            'characters': {}}), 'softly lit, bloom, rim lighting, hair ornament, bloom'))
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                 (next(responses), info)) as generate:
            scene, _, report = run_scene('colorful theme', (), output_format='tags', expand_common=True)
        self.assertEqual(scene, 'softly lit, bloom, rim lighting, hair ornament')
        self.assertEqual(generate.call_count, 2)
        self.assertNotIn('grounding_error', report['calls'][1])
        self.assertNotIn('rejected_tags', report['calls'][1])

    def test_only_tag_duplicates_are_removed_including_weighted_members(self):
        anchors = ('(colorful theme:1.3)', '(open mouth, determined, one eye closed, :1.3)')
        tags = deduplicate_tag_rendering(anchors,
            'colorful_theme, OPEN MOUTH, determined, one_eye_closed, softly lit, soft lighting, softly_lit, umbrella')
        self.assertEqual(tags, ', '.join((*anchors, 'softly lit', 'soft lighting', 'umbrella')))

    def test_new_roles_positions_clothes_and_props_are_not_filtered(self):
        caption = 'The woman on the left has blonde hair and is handing over a cup.'
        tags = deduplicate_tag_rendering(caption_anchors(caption),
            'offering cup, receiving cup, right, black dress, sofa, offering_cup')
        self.assertEqual(tags,
            'blonde hair, left, offering cup, receiving cup, right, black dress, sofa')

    def test_source_weight_survives_while_generated_position_is_kept(self):
        caption = 'The woman on the left has (blonde hair:1.2) and blue eyes, and is handing a cup to the woman on the right.'
        anchors = caption_anchors(caption)
        self.assertIn('(blonde hair:1.2)', anchors)
        self.assertNotIn('right', anchors)
        self.assertEqual(deduplicate_tag_rendering(anchors, 'blonde hair, right, handing cup'),
            '(blonde hair:1.2), blue eyes, left, right, handing cup')
