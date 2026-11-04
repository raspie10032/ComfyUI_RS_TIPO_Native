import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import test_native
from rs_tipo_native.gemma_runtime import token_constraints, run_scene, GemmaOutputError
from rs_tipo_native.gemma_prompt import (caption_anchors, guarded_tag_rendering, scene_count_anchors,
    prose_errors, prose_source, labelled_character_fields, labelled_outfit_errors)

class GemmaRuntimeTests(unittest.TestCase):
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
        self.assertEqual(labelled_outfit_errors((carried,), (
            'The character wears a retro astronaut suit and carries a helmet under her arm.',)), {})

    def test_labelled_outfits_cannot_be_lost_or_held_as_props(self):
        source = ('외형: silver hair. 의상: satin ball gown, pearl necklace. 상자를 연다.',)
        for bad in ('The character has silver hair and opens a box.',
                    'The character is holding a satin ball gown and pearl necklace.',
                    'The character wears a satin ball gown. She is holding a pearl necklace.'):
            with self.subTest(bad=bad):
                self.assertIn('character1', labelled_outfit_errors(source, (bad,)))
        self.assertEqual(labelled_outfit_errors(source, (
            'The character wears a satin ball gown and a pearl necklace. She opens a box.',)), {})
        self.assertEqual(labelled_outfit_errors(('holding a dress',), ('The character holds a dress.',)), {})

    def test_retry_retains_outfit_items_as_worn_clothing(self):
        responses = iter([
            json.dumps({'common': '', 'characters': {'1': 'The character is holding a satin ball gown.'}}),
            json.dumps({'common': '', 'characters': {'1': 'The character wears a satin ball gown and opens a box.'}})])
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (
                 next(responses), {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}})):
            _, slots, report = run_scene('', ('외형: silver hair. 의상: satin ball gown. 상자를 연다.',),
                                         output_format='natural_language')
        self.assertIn('wears a satin ball gown', slots[0])
        self.assertIn('Explicit outfit items', report['calls'][0]['validation_error'])

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
        responses = iter([raw, 'bad\nformat', 'bad\nformat'])
        info = {'seconds': .1, 'output_tokens': 5, 'exclusion': {'blocked_candidates': 0}}
        with patch('rs_tipo_native.gemma_runtime.load_gemma', return_value=(object(), {})), \
             patch('rs_tipo_native.gemma_runtime.generate', side_effect=lambda *a, **k: (next(responses), info)):
            with self.assertRaisesRegex(ValueError, 'after one retry'):
                run_scene('', ('black hair',), output_format='tags')

    def test_caption_tags_keep_weights_and_exclude_recipient_position(self):
        caption='The woman on the left has (blonde hair:1.2) and blue eyes, and is handing a cup to the woman on the right.'
        anchors=caption_anchors(caption)
        self.assertIn('(blonde hair:1.2)',anchors)
        self.assertNotIn('blonde hair',anchors)
        self.assertIn('left',anchors);self.assertNotIn('right',anchors)
        tags,rejected=guarded_tag_rendering(caption,anchors,'blonde hair, right, handing cup')
        self.assertIn('(blonde hair:1.2)',tags);self.assertNotIn(', right',tags)
        self.assertIn('right',{r['tag'] for r in rejected})

    def test_tag_rendering_rejects_peer_role_and_invented_clothing(self):
        caption='The woman on the left has blonde hair and is handing over a cup.'
        tags,rejected=guarded_tag_rendering(caption,caption_anchors(caption),
            'offering cup, receiving cup, right, black dress, sofa')
        self.assertIn('offering cup',tags)
        self.assertNotIn('receiving',tags);self.assertNotIn('right',tags)
        self.assertEqual({r['tag'] for r in rejected},{'receiving cup','right','black dress','sofa'})

    def test_tag_rendering_accepts_explicit_finite_verbs(self):
        tags,_=guarded_tag_rendering('The character offers a cup.',(), 'offering cup')
        self.assertEqual(tags,'offering cup')

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
