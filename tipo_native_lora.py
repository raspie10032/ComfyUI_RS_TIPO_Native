# Vendored from RS-Seasonal-Prompt-Generator 354cc9b (Apache-2.0).
"""Built-in TIPO PEFT adapters, applied through ComfyUI's CLIP patcher.

No import-time I/O; only explicitly selected published files are fetched.
"""
import hashlib
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import threading


BASE_MODEL = "p-e-w/gemma-4-E2B-it-heretic-ara"
BASE_REVISION = "c9a1d4c031981f14d86eeb0c7d87de7fafd34513"
PRESAMPLING_ADAPTER = 'presampling-fresh-r8-v1'
LOCAL_PRESAMPLING_ADAPTER = 'local:' + PRESAMPLING_ADAPTER
CHOICES = ("none", "pilot-001", "pilot-002", "image-30k", "scale-20pct-001", PRESAMPLING_ADAPTER)
REQUIREMENTS = (
    "TIPO adapter for the unmerged p-e-w/gemma-4-E2B-it-heretic-ara base. "
    "Requires native Gemma 4 E2B (~9.57 GiB) or the built-in Q4 base (~3.19 GiB plus tokenizer). "
    "Each adapter downloads ~45.3 MiB once; all 190 projection targets must match. "
    "Native ComfyUI PyTorch execution; no llama.cpp. Download size is not VRAM usage. "
    "Text-only inference; no vision projector is required. "
    "Do not apply to an already merged TIPO model. "
    "Gemma 베이스에 적용할 TIPO LoRA입니다. 이미 병합된 모델에는 중복 적용하지 마세요."
)

# Public release identifiers and SHA-256 hashes, never personal dataset paths.
CATALOG = {
    PRESAMPLING_ADAPTER: {
        "repo": "raspie/gemma4-tipo-lora-presampling-fresh-r8-v1", "revision": "fc610b78d849315af0b9ae45a27c9c3aba09d992",
        "files": {
            "adapter_config.json": (11125, "ea6e8a22ffc1e65ac0bccecaaaa2832096acdea3e93bfccd4fddafd65438dcb1"),
            "adapter_model.safetensors": (47487272, "e0339ef0b200f823ffa2585415cd1b510ea90c182aaabd5b9e0b3e58e76bbf0e"),
        },
    },
    "scale-20pct-001": {
        "repo": "raspie/gemma4-tipo-lora-scale-20pct-001", "revision": "3668df754885175408ac247a9e85309fa6528486",
        "files": {
            "adapter_config.json": (11124, "b84e0a530ffbb6821052fd1cefb2fee19e46259b575b1c993174d522e820f132"),
            "adapter_model.safetensors": (47487272, "01dc2c9e5df07fd4437a146580158c2452e86a44b66543d5029de9d29889e28b"),
        },
    },
    "image-30k": {
        "repo": "raspie/gemma4-tipo-lora-image-30k", "revision": "5e919e81801a19066d6df8114d10730b1a9d1e5b",
        "files": {
            "adapter_config.json": (11125, "d3db6392a71e36e80486cc4eabe0bde581396399bf07b12f1a1c12157fababaa"),
            "adapter_model.safetensors": (47487272, "9cfae7497b9031923b238e7e94afd1a654cc9e86d732d39007fbf6fc093bf4dc"),
        },
    },
    "pilot-001": {
        "repo": "raspie/gemma4-tipo-lora-pilot-001", "revision": "432e2f074a200cf20137475e13020efa23b74aec",
        "files": {
            "adapter_config.json": (11125, "d3db6392a71e36e80486cc4eabe0bde581396399bf07b12f1a1c12157fababaa"),
            "adapter_model.safetensors": (47487272, "507ce9a126ef87506b7eb04858a711dec71b0c99908ff6dc28d8bf7721bb16ba"),
        },
    },
    "pilot-002": {
        "repo": "raspie/gemma4-tipo-lora-pilot-002", "revision": "c31d210508c0dfb9f3e08981baa73f2e06eafd0a",
        "files": {
            "adapter_config.json": (11125, "d3db6392a71e36e80486cc4eabe0bde581396399bf07b12f1a1c12157fababaa"),
            "adapter_model.safetensors": (47487272, "930d622ca59e5b9025fe8ce101bfb93de4ae8dff6ea0b3ef80abbbdecba90db7"),
        },
    },
}
_DOWNLOAD_LOCK = threading.Lock()
_TARGET = re.compile(r"model\.language_model\.layers\.(?:[0-9]|[12][0-9]|3[0-4])\.(?:self_attn\.(?:q|v|o)_proj|mlp\.(?:gate|up|down)_proj)$")
_LOCAL_NAME = re.compile(r'local:([A-Za-z0-9][A-Za-z0-9_-]{0,63})$')


def adapter_choices(root=None):
    """Published defaults plus explicitly installed local training bundles."""
    choices=[PRESAMPLING_ADAPTER,'pilot-002','pilot-001','image-30k','scale-20pct-001']
    if root is None:
        try:
            import folder_paths
            root=Path(folder_paths.models_dir)/'loras'/'rs_gemma_tipo'
        except (ImportError,AttributeError):
            return choices
    root=Path(root)
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            choice='local:'+directory.name
            if directory.is_dir() and not directory.is_symlink() and _LOCAL_NAME.fullmatch(choice) and (directory/'local-adapter.json').is_file():
                choices.append(choice)
    return choices


def adapter_details(directory, choice):
    """Describe verified weights without exposing private training paths."""
    directory = Path(directory)
    config = json.loads((directory / 'adapter_config.json').read_text())
    local = choice.startswith('local:')
    manifest = json.loads((directory / 'local-adapter.json').read_text()) if local else {}
    files = manifest.get('files', {}) if local else CATALOG[choice]['files']
    return {
        'kind': 'local-trained' if local else 'published',
        'rank': config.get('r'),
        'alpha': config.get('lora_alpha'),
        'weights_sha256': files.get('adapter_model.safetensors', (None, None))[1],
        'status': manifest.get('status', 'experimental published TIPO adapter'),
    }


def default_adapter(root=None):
    """The published weak-LoRA baseline is available on clean installations."""
    return PRESAMPLING_ADAPTER


def checked_strength(strength):
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError('TIPO LoRA strength must be finite and between 0 and 2.')
    return strength


def _local_bundle(choice,root,progress=None):
    match=_LOCAL_NAME.fullmatch(choice)
    if not match:raise ValueError('Invalid local TIPO adapter name')
    root=Path(root);directory=root/match[1]
    if directory.is_symlink() or directory.resolve().parent!=root.resolve():
        raise ValueError('Local TIPO adapter must be a direct model directory')
    manifest=json.loads((directory/'local-adapter.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!='rs-gemma-tipo-local/1' or set(manifest.get('files',{}))!={'adapter_config.json','adapter_model.safetensors'}:
        raise ValueError('Invalid local TIPO adapter manifest')
    for name,expected in manifest['files'].items():
        if not isinstance(expected,list) or len(expected)!=2 or type(expected[0]) is not int or expected[0]<=0 or not isinstance(expected[1],str) or not re.fullmatch('[a-f0-9]{64}',expected[1]):
            raise ValueError('Invalid local TIPO file hash/size')
        _check_file(directory/name,tuple(expected))
        if progress:progress(expected[0],expected[0])
    return directory


def _check_file(path, expected):
    size, digest = expected
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"TIPO model/adapter file has wrong size/type: {path}. Move it aside and retry, or install the published file offline.")
    signature = _file_stat(path)
    # If the filesystem cannot provide a change timestamp, do a full check
    # instead of trusting size/mtime alone (both can survive an in-place edit).
    verify = _check_digest if signature[-1] is not None else _check_digest.__wrapped__
    verify(str(path.resolve()), digest, signature)


def _windows_change_time(path):
    """Use Windows ChangeTime; Python st_ctime is creation time there.

    https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_basic_info
    Return None for unsupported filesystems/APIs so verification cannot be cached.
    """
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class FileBasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in
                    ("CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime")]
        _fields_.append(("FileAttributes", wintypes.DWORD))

    try:
        query = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
        query.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        query.restype = wintypes.BOOL
        info = FileBasicInfo()
        with Path(path).open("rb") as stream:
            handle = msvcrt.get_osfhandle(stream.fileno())
            if query(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
                return info.ChangeTime or None
    except (AttributeError, OSError):
        pass
    return None


def _file_stat(path):
    stat = Path(path).stat()
    changed = _windows_change_time(path) if os.name == "nt" else stat.st_ctime_ns
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, changed


@lru_cache(maxsize=32)
def _check_digest(filename, digest, signature):
    # Multi-GiB bases should be hashed once per unchanged local file, not once
    # per generation. Change time/inode invalidate edits even if mtime is restored.
    path = Path(filename)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != digest:
        raise ValueError(f"TIPO model/adapter checksum mismatch: {path}. Existing files are not overwritten; move it aside and retry.")
    if _file_stat(path) != signature:
        raise ValueError(f"TIPO model/adapter changed during verification: {path}")


def ensure_adapter(choice, root, progress=None):
    """Download two pinned files atomically, or validate an offline install."""
    if choice.startswith('local:'):
        return _local_bundle(choice,root,progress)
    if choice not in CATALOG:
        raise ValueError(f"Unknown TIPO LoRA: {choice}")
    entry = CATALOG[choice]
    directory = Path(root) / choice
    return ensure_files(entry, directory, progress)


def ensure_files(entry, directory, progress=None):
    """Fetch pinned assets with resumable Hub transfers and verify before use."""
    directory = Path(directory)
    with _DOWNLOAD_LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        missing = []
        for name, expected in entry["files"].items():
            if Path(name).name != name or name in (".", "..") or "\\" in name:
                raise ValueError("Model asset names must be plain filenames")
            destination = directory / name
            if destination.exists() or destination.is_symlink():
                _check_file(destination, expected)
            else:
                missing.append(name)
        if missing:
            if any(os.environ.get(key, "").lower() in ("1", "true", "yes", "on")
                   for key in ("RS_MODELS_OFFLINE", "HF_HUB_OFFLINE")):
                raise RuntimeError("Offline mode: missing model files: " + ", ".join(missing)
                    + f". Install the pinned files in {directory} or disable offline mode.")
            required = sum(entry["files"][name][0] for name in missing)
            if shutil.disk_usage(directory).free < required + 64 * 1024 * 1024:
                raise OSError(f"Not enough disk space for model download ({required} bytes plus 64 MiB reserve): {directory}")
        from .gemma_runtime import status
        from huggingface_hub import hf_hub_download
        from tqdm.auto import tqdm
        total = sum(v[0] for v in entry['files'].values())
        done = 0
        for name, expected in entry["files"].items():
            destination = directory / name
            if name not in missing:
                done += expected[0]
                if progress:
                    progress(done, total)
                continue
            if progress:
                progress(done, total)  # Check cancellation before network I/O.
            status('starting', name, done, total)
            offset, size = done, expected[0]
            cancelled = False

            class Transfer(tqdm):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.received = self.n
                    self.report()

                def report(self):
                    nonlocal cancelled
                    current = offset + min(self.received, size)
                    try:
                        if progress:
                            progress(current, total)
                    except BaseException:
                        cancelled = True
                        raise
                    status('downloading', name, current, total)

                def update(self, n=1):
                    result = super().update(n)
                    self.received += n
                    self.report()
                    return result

            try:
                hf_hub_download(entry['repo'], name, revision=entry['revision'],
                    local_dir=str(directory), tqdm_class=Transfer)
            except Exception as exc:
                if cancelled:
                    raise
                if progress:
                    progress(done, total)
                # Hub exceptions may contain signed URLs or request headers.
                raise RuntimeError(f'Model download failed: {name} ({type(exc).__name__}). '
                    'Check network, disk space and folder permissions; run again to resume. / '
                    '네트워크·저장 공간·폴더 권한을 확인하고 다시 실행해 주세요.') from None
            status('verifying', name, done + size, total)
            _check_file(destination, expected)
            done += size
            if progress:
                progress(done, total)
        status('ready', '', total, total)
    return directory


def adapter_mapping(config, tensors, model_state):
    """Validate the whole PEFT inventory before any model mutation."""
    if (config.get("peft_type") != "LORA" or config.get("bias") != "none"
            or config.get("base_model_name_or_path") != BASE_MODEL
            or config.get("revision") != BASE_REVISION
            or config.get("r") != 8 or config.get("lora_alpha") != 16):
        raise ValueError("Unsupported TIPO adapter config/base")
    if any(config.get(k) for k in ("use_dora", "use_rslora", "fan_in_fan_out", "lora_bias",
                                  "rank_pattern", "alpha_pattern", "modules_to_save", "target_parameters")):
        raise ValueError("Unsupported TIPO adapter variant")
    targets = config.get("target_modules", [])
    if (not isinstance(targets, list) or len(targets) != 190 or len(set(targets)) != 190
            or any(not isinstance(t, str) or not _TARGET.fullmatch(t) for t in targets)):
        raise ValueError("Expected 190 language-only TIPO targets")
    import torch
    mapping = {}
    expected_keys = set()
    for target in targets:
        prefix = "base_model.model." + target
        akey, bkey = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
        expected_keys.update((akey, bkey))
        key = "gemma4.transformer.model." + target.removeprefix("model.language_model.") + ".weight"
        if akey not in tensors or bkey not in tensors or key not in model_state:
            raise ValueError(f"Missing TIPO adapter/base tensor for {target}")
        a, b, w = tensors[akey], tensors[bkey], model_state[key]
        if (a.ndim != 2 or b.ndim != 2 or w.ndim != 2 or a.shape[0] != 8
                or b.shape[1] != 8 or (b.shape[0], a.shape[1]) != tuple(w.shape)
                or not a.is_floating_point() or not b.is_floating_point()
                or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise ValueError(f"TIPO tensor shape/rank/value mismatch: {target}")
        mapping[prefix] = key
    if set(tensors) != expected_keys:
        raise ValueError("Unexpected tensors in TIPO adapter; vision/PLE adapters are not supported")
    return mapping


def apply_tipo_lora(clip, choice="none", strength=1.0):
    strength = checked_strength(strength)
    if choice == "none" or strength == 0:
        return clip
    if choice not in CATALOG and not _LOCAL_NAME.fullmatch(choice):
        raise ValueError(f"Unknown TIPO LoRA: {choice}")
    patcher = getattr(clip, "patcher", None)
    if patcher is None or not callable(getattr(clip, "clone", None)):
        raise ValueError("TIPO LoRA requires a native Gemma 4 E2B CLIP. " + REQUIREMENTS)
    state = patcher.model_state_dict()
    embed = state.get("gemma4.transformer.model.embed_tokens.weight")
    if embed is None or tuple(embed.shape) != (262144, 1536):
        raise ValueError("TIPO LoRA requires the native Gemma 4 E2B base. " + REQUIREMENTS)
    import folder_paths
    import comfy.lora
    import comfy.utils
    import torch
    logging.warning("TIPO %s: %s", choice, REQUIREMENTS)
    progress = comfy.utils.ProgressBar(1)
    def update(current, total):
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()
        progress.update_absolute(current, total)
    directory = ensure_adapter(choice, Path(folder_paths.models_dir) / "loras" / "rs_gemma_tipo",
                               progress=update)
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = comfy.utils.load_torch_file(str(directory / "adapter_model.safetensors"), safe_load=True)
    mapping = adapter_mapping(config, tensors, state)
    if set(mapping.values()) & set(getattr(patcher, "patches", {})):
        raise ValueError("TIPO language weights already have patches. Use the unpatched base to avoid applying LoRA twice.")
    for source in mapping:
        tensors[source + ".alpha"] = torch.tensor(config["lora_alpha"], dtype=torch.float32)
    patches = comfy.lora.load_lora(tensors, mapping)
    if set(patches) != set(mapping.values()):
        raise RuntimeError("ComfyUI did not recognize all 190 TIPO adapter targets. Update ComfyUI.")
    result = clip.clone()
    applied = result.patcher.add_patches(patches, strength_patch=strength)
    if set(applied) != set(patches):
        raise RuntimeError("ComfyUI rejected TIPO adapter targets")
    return result
