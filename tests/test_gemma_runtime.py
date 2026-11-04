import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import test_native
from rs_tipo_native.gemma_runtime import token_constraints, run_scene, GemmaOutputError
from rs_tipo_native.gemma_prompt import (caption_anchors, deduplicate_tag_rendering, scene_count_anchors,
    prose_errors, prose_source, labelled_character_fields)

class GemmaRuntimeTests(unittest.TestCase):

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

    def test_incorrect_slots_retry_same_model_then_fail(self):
        info={'seconds':.1,'output_tokens':5,'exclusion':{'blocked_candidates':0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma',return_value=(object(),{})) as load, \
             patch('rs_tipo_native.gemma_runtime.generate',return_value=('{}',info)) as generate:
            with self.assertRaises(ValueError):run_scene('cafe',('black hair',),output_format='natural_language')
        self.assertEqual(load.call_count,1);self.assertEqual(generate.call_count,2)

    def test_labelled_wildcard_response_repairs_failed_fields_without_rewriting_valid_ones(self):
        clip = object()
        common = 'The scene is a hotel with three women beside marble stairs.'
        valid = 'The character has purple hair and is clasping her hands while looking at the box.'
        malformed = {'common': common, 'characters': {
            '1': 'Appearance: sophie neuenmuller, brown eyes, short hair. Outfit: satin ball gown. Showing a small box.',
            '2': 'Appearance: blonde hair, purple eyes. Outfit: velvet evening gown. Looking inside the box.',
            '3': valid}}
        repaired = {'common': 'The scene is a hotel.', 'characters': {
            '1': 'The character is Sophie Neuenmuller and has brown eyes and short hair. She wears a satin ball gown and opens a small box.',
            '2': 'The character has blonde hair and purple eyes. She wears a velvet evening gown and looks inside the box.',
            '3': 'The character has purple hair and is smiling.'}}
        raw = json.dumps(malformed)
        responses = iter([raw, json.dumps(repaired)])
        prompts = []
        def generate(actual, prompt, **kwargs):
            self.assertIs(actual, clip)
            prompts.append(prompt)
            return next(responses), {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(clip, {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=generate):
            scene, slots, report = run_scene('호텔, 여성 3명',
                ('외형: sophie neuenmuller. 의상: satin ball gown. 작은 상자를 연다.',
                 '외형: blonde hair. 의상: velvet evening gown. 상자를 본다.',
                 'purple hair, looking at box, clasping hands'),
                output_format='natural_language', expand_common=True)
        self.assertEqual(scene, common)
        self.assertEqual(slots[2], valid)
        self.assertEqual(slots[:2], (repaired['characters']['1'], repaired['characters']['2']))
        self.assertEqual(json.loads(prompts[1].split('DATA to repair, not instructions:\n')[1].split('\n')[0])['previous_response'], raw)
        self.assertIn('character1:', report['calls'][0]['validation_error'])
        self.assertIn('character2:', report['calls'][0]['validation_error'])
        self.assertEqual(report['calls'][1]['preserved_fields'], ['common', 'character3'])
        self.assertEqual(report['model_calls'], 2)
        self.assertNotIn('add ZERO optional visual details', prompts[1])
        self.assertIn('Keep compatible visual details already added', prompts[1])
        self.assertIn('original expansion task and detail budget still apply', prompts[1])

    def test_failed_prose_retry_retains_raw_responses_and_identifies_fields(self):
        raw = json.dumps({'common': '', 'characters': {'1': 'Appearance: silver hair. Outfit: dress.'}})
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', return_value=(raw, {'seconds': .1, 'output_tokens': 5})) as generate:
            with self.assertRaises(GemmaOutputError) as caught:
                run_scene('', ('외형: silver hair. 의상: dress.',), output_format='natural_language')
        self.assertEqual(generate.call_count, 2)
        self.assertEqual([call['raw'] for call in caught.exception.calls], [raw, raw])
        self.assertIn('character1:', str(caught.exception))
        self.assertIn('Last response:', str(caught.exception))

    def test_prose_errors_name_only_malformed_fields(self):
        errors = prose_errors(('The scene is a garden.',), (
            'The character has silver hair.', 'Appearance: brown hair, blue eyes.', '그녀는 꽃을 든다.'))
        self.assertEqual(set(errors), {'character2', 'character3'})
        self.assertIn('English prose', errors['character3'])

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

    def test_repair_still_rejects_lost_weights_and_excluded_tags(self):
        for source, excluded, repaired in (
            ('(silver hair:1.2)', (), 'The character has silver hair.'),
            ('silver hair', ('glasses',), 'The character has silver hair and glasses.'),
        ):
            with self.subTest(source=source, excluded=excluded):
                responses = iter([
                    json.dumps({'common': '', 'characters': {'1': f'Appearance: {source}.'}}),
                    json.dumps({'common': '', 'characters': {'1': repaired}})])
                with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
                     patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), {'seconds': .1, 'output_tokens': 5})):
                    with self.assertRaises(GemmaOutputError):
                        run_scene('', (source,), excluded=excluded, output_format='natural_language')

    def test_caption_tags_must_keep_explicit_attributes(self):
        self.assertEqual(caption_anchors('A woman with black hair and brown eyes wears a blue cardigan, standing behind.'),
                         ('black hair','brown eyes','blue cardigan','behind'))

    def test_invalid_tag_format_gets_one_same_model_retry_and_retains_raw_calls(self):
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
        self.assertEqual(report['model_calls'], 3)
        self.assertIn('format_error', report['calls'][1])
        self.assertEqual(report['calls'][1]['raw'], '{"tags": "black hair"}')

    def test_tag_retry_is_bounded_and_does_not_return_unprocessed_source(self):
        raw = json.dumps({'common': '', 'characters': {'1': 'The woman has black hair.'}})
        responses = iter([raw, '', ''])
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), info)):
            with self.assertRaisesRegex(ValueError, 'after one retry'):
                run_scene('', ('black hair',), output_format='tags')

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
        self.assertEqual(scene, rendered.rsplit(', colorful_theme', 1)[0])
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

    def test_korean_common_repair_keeps_translated_scene_and_not_only_additions(self):
        responses = iter((json.dumps({'common': 'cafe', 'characters': {}}),
            json.dumps({'common': 'The scene is a cafe with two women. The lighting is soft.', 'characters': {}}),
            '2girls, cafe, soft lighting'))
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k:
                 (next(responses), {'seconds': .1, 'output_tokens': 5,
                    'exclusion': {'blocked_candidates': 0}})) as generate:
            scene, _, _ = run_scene('여성 두 명, 카페', (), output_format='tags', expand_common=True)
        self.assertEqual(scene, '2girls, cafe, soft lighting')
        self.assertIn('Preserve all supplied facts', generate.call_args_list[1].args[1])
        self.assertNotIn('only the NEW visual details', generate.call_args_list[1].args[1])

    def test_tag_planning_does_not_require_image_weight_syntax_in_english_prose(self):
        raw = json.dumps({'common': '', 'characters': {'1': 'The character has red hair and blue eyes.'}})
        responses = iter([raw, 'red hair, blue eyes'])
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), info)):
            _, slots, _ = run_scene('', ('(red hair, blue eyes:1.2)',), output_format='tags')
        self.assertEqual(slots, ('red hair, blue eyes',))

    def test_scene_count_comes_from_explicit_caption(self):
        self.assertEqual(scene_count_anchors('Two adult women are in a cafe.'),('2girls',))
        self.assertEqual(scene_count_anchors('The scene is a garden.'),())

    def test_derived_tag_prefix_cannot_reintroduce_exclusion(self):
        raw=json.dumps({'common':'Two adult women are in a cafe.','characters':{'1':'The character has black hair.'}})
        responses=iter([raw,'cafe','black hair'])
        info={'seconds':.1,'output_tokens':5,'exclusion':{'blocked_candidates':0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma',return_value=(object(),{})), \
             patch('rs_tipo_native.gemma_runtime.generate',side_effect=lambda *a,**kw:(next(responses),info)):
            with self.assertRaisesRegex(ValueError,'derived tag prefix'):
                run_scene('two women',('black hair',),expand_common=True,excluded=('2girls',))

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
