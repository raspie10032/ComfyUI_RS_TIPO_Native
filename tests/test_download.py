import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import test_native
from rs_tipo_native import download, runtime


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.data = {'config.json': b'config', 'tokenizer.json': b'tokenizer', 'model.safetensors': b'weights'}
        self.manifest = {n: (len(b), hashlib.sha256(b).hexdigest()) for n, b in self.data.items()}

    def test_first_download_progress_then_offline_reuse(self):
        with TemporaryDirectory() as tmp, patch.dict(download.FILES, self.manifest, clear=True):
            calls, progress, states = [], [], []
            def fetch(repo, filename, revision, local_dir, tqdm_class):
                calls.append((repo, filename, revision))
                (Path(local_dir) / filename).write_bytes(self.data[filename])
                with tqdm_class(total=len(self.data[filename]), disable=True) as bar:
                    bar.update(len(self.data[filename]))
            with patch('huggingface_hub.hf_hub_download', side_effect=fetch) as network:
                download.ensure_model(tmp, progress=lambda n, total: progress.append((n, total)),
                                      status=lambda *args: states.append(args))
                self.assertEqual(len(calls), 3)
                self.assertTrue(all(row[2] == download.MODEL_REVISION for row in calls))
                self.assertTrue(download.complete_folder(tmp))
                self.assertEqual(progress[-1][0], progress[-1][1])
                self.assertEqual(states[0][0], 'starting')
                self.assertEqual(states[-1][0], 'ready')
                for name in self.data:
                    self.assertTrue(any(s[0] == 'downloading' and s[1] == name for s in states))
                    self.assertTrue(any(s[0] == 'verifying' and s[1] == name for s in states))
                current = [s[2] for s in states if s[0] == 'downloading']
                self.assertEqual(current, sorted(current))
                network.reset_mock()
                download.ensure_model(tmp)
                network.assert_not_called()

    def test_resume_skips_verified_files_and_hash_failure_cannot_look_complete(self):
        with TemporaryDirectory() as tmp, patch.dict(download.FILES, self.manifest, clear=True):
            (Path(tmp) / 'config.json').write_bytes(self.data['config.json'])
            def bad(repo, filename, **kw):
                (Path(tmp) / filename).write_bytes(b'x' * len(self.data[filename]))
            with patch('huggingface_hub.hf_hub_download', side_effect=bad) as net:
                with self.assertRaisesRegex(RuntimeError, 'integrity'):
                    download.ensure_model(tmp)
                self.assertEqual(net.call_args.args[1], 'tokenizer.json')
            self.assertFalse(download.complete_folder(tmp))

    def test_status_targets_request_client_and_node_with_throttling(self):
        events = []
        server = SimpleNamespace(last_node_id='17', client_id='request-client',
                                 send_sync=lambda *args: events.append(args))
        with patch.dict(sys.modules, {'server': SimpleNamespace(PromptServer=SimpleNamespace(instance=server)),
                                      'comfy_execution.utils': SimpleNamespace(get_executing_context=lambda: None)}):
            send = runtime.download_status_sender()
            server.last_node_id, server.client_id = 'other-node', 'other-client'
            send('downloading', 'model.safetensors', 1, 100)
            send('downloading', 'model.safetensors', 2, 100)
            send('verifying', 'model.safetensors', 100, 100)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0][0], 'rs_tipo_download')
            self.assertEqual(events[0][1]['node'], '17')
            self.assertTrue(all(e[2] == 'request-client' for e in events))

    def test_interrupt_and_network_failure_leave_retryable_state(self):
        def cancel():
            raise InterruptedError('cancel')
        with TemporaryDirectory() as tmp, patch.dict(download.FILES, self.manifest, clear=True):
            with patch('huggingface_hub.hf_hub_download') as net:
                with self.assertRaises(InterruptedError):
                    download.ensure_model(tmp, interrupt=cancel)
                net.assert_not_called()
            with patch('huggingface_hub.hf_hub_download', side_effect=OSError('private signed URL')):
                with self.assertRaisesRegex(RuntimeError, 'Check network') as error:
                    download.ensure_model(tmp)
                self.assertNotIn('private signed URL', str(error.exception))

    def test_extra_model_path_complete_install_reused_without_comfy_import(self):
        with TemporaryDirectory() as tmp, patch.dict(download.FILES, self.manifest, clear=True):
            default, extra = Path(tmp)/'default', Path(tmp)/'extra'
            target = extra / download.MODEL_NAME
            target.mkdir(parents=True)
            for n, b in self.data.items():
                (target/n).write_bytes(b)
            folders = SimpleNamespace(models_dir=str(default), folder_names_and_paths={'tipo': True},
                                      get_folder_paths=lambda _: [str(extra)])
            with patch.dict(sys.modules, {'folder_paths': folders}), patch.object(runtime, 'ensure_model') as net:
                self.assertEqual(runtime.model_path(), target)
                net.assert_not_called()

    def test_one_shot_comfy_cancel_during_transfer_is_not_network_error(self):
        class ComfyCancel(Exception):
            pass
        calls = 0
        def interrupt():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ComfyCancel('stopped')
        def fetch(repo, filename, revision, local_dir, tqdm_class):
            with tqdm_class(total=2, disable=True) as bar:
                bar.update(1)
        with TemporaryDirectory() as tmp, patch.dict(download.FILES, self.manifest, clear=True):
            with patch('huggingface_hub.hf_hub_download', side_effect=fetch):
                with self.assertRaises(ComfyCancel):
                    download.ensure_model(tmp, interrupt=interrupt)
            self.assertFalse(download.complete_folder(tmp))
