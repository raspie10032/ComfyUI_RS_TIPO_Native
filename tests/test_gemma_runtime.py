import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import test_native
from rs_tipo_native.gemma_runtime import token_constraints, run_scene
from rs_tipo_native.gemma_prompt import caption_anchors, guarded_tag_rendering, scene_count_anchors

class GemmaRuntimeTests(unittest.TestCase):
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
