import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_native
from rs_tipo_native.nodes import RSTIPOPrompt
from rs_tipo_native.tipo_native_lora import ensure_files


class ModelChoices(unittest.TestCase):
    def test_standalone_never_loads_gemma_and_ignores_inactive_slots(self):
        info = dict(seconds=.1, model_calls=2, raw_output='raw')
        with patch('rs_tipo_native.interface.pipe_adapter', return_value=None), \
             patch('rs_tipo_native.gemma_runtime.run_scene', side_effect=AssertionError('Gemma used')), \
             patch('rs_tipo_native.tipo_runtime.run_scene', return_value=('cafe', ('blonde hair',), info)) as run:
            result = RSTIPOPrompt().expand(model_family='tipo_2_1', character_count=1,
                common_prompt='cafe', character1='blonde hair', character2='STALE',
                protect_source=False, output_format='tags', quality='masterpiece')
        self.assertEqual(run.call_args.args, ('cafe', ('blonde hair',)))
        self.assertEqual(run.call_args.kwargs['metadata'], {'quality': 'masterpiece'})
        self.assertFalse(run.call_args.kwargs['protect_source'])
        self.assertEqual(result['result'][2:4], ('blonde hair', ''))
        self.assertEqual(json.loads(result['ui']['tipo_report'][0])['model_family'], 'TIPO 2.1')

    def test_standalone_invalid_inputs_fail_before_model_download(self):
        with patch('rs_tipo_native.runtime.acquire_model', side_effect=AssertionError('loaded')):
            for settings, message in [({'common_prompt': '카페'}, 'English'),
                ({'excluded_tags': '파란_눈'}, 'English'),
                ({'generation_max_tokens': 8192}, '4096'),
                ({'common_prompt': 'blue eyes', 'excluded_tags': 'blue_eyes'}, 'excluded')]:
                with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, message):
                    RSTIPOPrompt().expand(**{'character_count': 0, 'common_prompt': 'cafe',
                        'model_family': 'tipo_2_1', **settings})

    def test_unknown_backend_cannot_silently_choose_gemma(self):
        with self.assertRaisesRegex(ValueError, 'Select Gemma'):
            RSTIPOPrompt().expand(character_count=0, common_prompt='cafe', model_family='oops')


class ModelAssets(unittest.TestCase):
    def setUp(self):
        self.data = b'fixture verified weights'
        self.entry = dict(repo='example/fixture', revision='a'*40,
            files={'weights.bin': (len(self.data), hashlib.sha256(self.data).hexdigest())})

    def test_download_exact_revision_progress_cache_and_changed_file_rejection(self):
        def fetch(repo, name, **kw):
            self.assertEqual((repo, kw['revision']), ('example/fixture', 'a'*40))
            (Path(kw['local_dir']) / name).write_bytes(self.data)
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'HF_HUB_OFFLINE': '0', 'RS_MODELS_OFFLINE': '0'}), \
             patch('huggingface_hub.hf_hub_download', side_effect=fetch) as download:
            progress = []
            ensure_files(self.entry, folder, lambda n, t: progress.append((n,t)))
            self.assertEqual(progress[-1], (len(self.data), len(self.data)))
            ensure_files(self.entry, folder)
            self.assertEqual(download.call_count, 1)
            (Path(folder)/'weights.bin').write_bytes(b'X'*len(self.data))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                ensure_files(self.entry, folder)
            self.assertEqual(download.call_count, 1)

    def test_offline_cached_assets_work_and_missing_assets_fail_before_network(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'RS_MODELS_OFFLINE': '1'}), \
             patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('network')):
            with self.assertRaisesRegex(RuntimeError, 'Offline'):
                ensure_files(self.entry, folder)
            (Path(folder)/'weights.bin').write_bytes(self.data)
            ensure_files(self.entry, folder)

    def test_cancel_preserves_interrupt_and_hub_resume_data(self):
        class UserCancelled(Exception): pass
        calls = 0
        def progress(n, total):
            nonlocal calls
            calls += 1
            if calls > 1: raise UserCancelled()
        def fetch(repo, name, **kw):
            resumable = Path(kw['local_dir'])/'.cache'
            resumable.mkdir()
            (resumable/'partial.incomplete').write_bytes(b'partial')
            with kw['tqdm_class'](total=len(self.data), disable=True) as bar:
                bar.update(3)
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'HF_HUB_OFFLINE': '0', 'RS_MODELS_OFFLINE': '0'}), \
             patch('huggingface_hub.hf_hub_download', side_effect=fetch):
            with self.assertRaises(UserCancelled): ensure_files(self.entry, folder, progress)
            self.assertTrue((Path(folder)/'.cache/partial.incomplete').is_file())
            self.assertFalse((Path(folder)/'weights.bin').exists())

    def test_errors_do_not_expose_signed_urls_or_tokens(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'HF_HUB_OFFLINE': '0', 'RS_MODELS_OFFLINE': '0'}), \
             patch('huggingface_hub.hf_hub_download', side_effect=OSError('secret-signed-url')):
            with self.assertRaisesRegex(RuntimeError, 'Check network') as error:
                ensure_files(self.entry, folder)
            self.assertNotIn('secret-signed-url', str(error.exception))
