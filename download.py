"""Pinned, resumable model acquisition shared by the nodes and CLI."""
import hashlib
import logging
from pathlib import Path
import threading

MODEL_NAME = 'TIPO-v2.1-1B-A200M'
MODEL_REPO = 'KBlueLeaf/' + MODEL_NAME
MODEL_REVISION = 'f5a318524a4ab30cdbbf51816cf406170f454e65'
FILES = {
    'config.json': (821, '0c10987f0eba8a0fcb58834de79714700be673a054130759ca97cc3467e8ae79'),
    'tokenizer.json': (2363483, 'e219a2a714e52f4cce5dfaa0de8dbd3f4aef5bbec6b6a7b057fde7cb0675e7e1'),
    'model.safetensors': (1981598832, 'e291ff39cd6ab43ce4537a68792141a7852b1432404db4c1aeb330398bf735ea'),
}
DOWNLOAD_LOCK = threading.RLock()


def complete_folder(path):
    path = Path(path)
    if (path / '.rs-tipo-download-incomplete').exists():
        return False
    return all((path / name).is_file() and (path / name).stat().st_size == size
               for name, (size, _) in FILES.items())


def verify_file(path, expected, interrupt):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            interrupt()
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f'TIPO model integrity check failed: {Path(path).name}. '
                           'Remove that damaged file and retry. / 손상된 파일을 지운 뒤 다시 실행해 주세요.')


def ensure_model(destination, *, interrupt=lambda: None, progress=lambda n, total: None,
                 status=lambda phase, filename, current, total: None):
    """Reuse complete installs offline; HF file locks cover concurrent processes.

    Incomplete transfers stay resumable in .cache. Final filenames are published
    by the Hub only after transfer. New files receive full SHA-256 verification.
    """
    destination = Path(destination)
    with DOWNLOAD_LOCK:
        if complete_folder(destination):
            return destination
        from huggingface_hub import hf_hub_download
        from tqdm.auto import tqdm
        total = sum(size for size, _ in FILES.values())
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / '.rs-tipo-download-incomplete'
        marker.touch()
        done = 0
        status('starting', '', done, total)
        for name, (size, checksum) in FILES.items():
            interrupt()
            path = destination / name
            if path.is_file() and path.stat().st_size == size:
                status('verifying', name, done, total)
                verify_file(path, checksum, interrupt)
            else:
                logging.info('RS TIPO: downloading %s (~1.85 GiB total) into %s', name, destination)
                offset = done
                cancelled = False

                class DownloadProgress(tqdm):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.received = self.n
                        self.report()

                    def report(self):
                        current = min(offset + self.received, offset + size)
                        progress(current, total)
                        status('downloading', name, current, total)

                    def update(self, n=1):
                        nonlocal cancelled
                        try:
                            interrupt()
                        except BaseException:
                            cancelled = True
                            raise
                        result = super().update(n)
                        self.received += n
                        self.report()
                        return result

                try:
                    hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION,
                                    local_dir=str(destination), tqdm_class=DownloadProgress)
                except Exception as exc:
                    if cancelled:
                        raise
                    interrupt()
                    raise RuntimeError(f'TIPO download failed for {name} ({type(exc).__name__}). '
                        'Check network, disk space and folder permissions; run again to resume. / '
                        '네트워크·여유 공간·폴더 권한을 확인하고 재실행해 주세요.') from None
                status('verifying', name, done + size, total)
                verify_file(path, checksum, interrupt)
            done += size
            progress(done, total)
        marker.unlink(missing_ok=True)
        status('ready', '', total, total)
        return destination
