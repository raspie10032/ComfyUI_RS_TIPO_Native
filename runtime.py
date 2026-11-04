"""ComfyUI-managed loading. All model execution is local native PyTorch."""
from functools import lru_cache
import json
from pathlib import Path
import threading
import time

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from .model import KohakuConfig, KohakuForCausalLM
from .download import MODEL_NAME, MODEL_REVISION, FILES, complete_folder, ensure_model

LOCK = threading.RLock()


def download_status_sender():
    """Bind transient status to the executing node and its requesting client."""
    from server import PromptServer
    server = PromptServer.instance
    if server is None:
        from .gemma_runtime import status
        return status
    node = server.last_node_id
    try:
        from comfy_execution.utils import get_executing_context
        context = get_executing_context()
        if context is not None:
            node = context.node_id
    except ImportError:  # Older supported ComfyUI.
        pass
    client = server.client_id
    previous, last = None, 0.0

    def send(phase, filename, current, total):
        nonlocal previous, last
        key, now = (phase, filename), time.monotonic()
        if key == previous and now - last < 0.2:
            return
        previous, last = key, now
        if node is not None and client is not None:
            server.send_sync('rs_tipo_download', dict(node=str(node), phase=phase,
                             filename=filename, current=current, total=total), client)
    return send


def model_path():
    import folder_paths
    roots = [Path(folder_paths.models_dir) / "tipo"]
    # Honor extra_model_paths.yaml entries named "tipo" as well as the default.
    if "tipo" in folder_paths.folder_names_and_paths:
        roots += [Path(p) for p in folder_paths.get_folder_paths("tipo")]
    for root in roots:
        candidate = root / MODEL_NAME
        if complete_folder(candidate):
            return candidate
    import comfy.model_management as mm
    from comfy.utils import ProgressBar
    bar = ProgressBar(sum(size for size, _ in FILES.values()))
    destination = next((root / MODEL_NAME for root in roots if (root / MODEL_NAME).is_dir()),
                       roots[0] / MODEL_NAME)
    status = download_status_sender()
    try:
        return ensure_model(destination, interrupt=mm.throw_exception_if_processing_interrupted,
                            progress=bar.update_absolute, status=status)
    except BaseException:
        status('stopped', '', 0, 0)
        raise


def load_weights(path, dtype=torch.float16):
    path = Path(path)
    required = ("config.json", "tokenizer.json", "model.safetensors")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete TIPO model folder: missing {', '.join(missing)}")
    cfg = KohakuConfig.from_dict(json.loads((path / "config.json").read_text()))
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    if tokenizer.get_vocab_size() != cfg.vocab_size:
        raise ValueError("TIPO tokenizer vocabulary does not match the checkpoint config.")
    with torch.device("meta"):
        model = KohakuForCausalLM(cfg)
    state = load_file(str(path / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    model = model.to(dtype=dtype).eval().requires_grad_(False)
    return model, tokenizer


@lru_cache(maxsize=1)
def managed_model(path):
    import comfy.model_management as mm
    from comfy.model_patcher import ModelPatcher
    device = mm.text_encoder_device()
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model, tokenizer = load_weights(path, dtype)
    patcher = ModelPatcher(model, load_device=device, offload_device=mm.text_encoder_offload_device())
    return patcher, tokenizer


def acquire_model():
    import comfy.model_management as mm
    patcher, tokenizer = managed_model(str(model_path().resolve()))
    # Stacked MoE expert parameters require full model residency; do not claim low-VRAM streaming.
    # ComfyUI can still offload this model entirely between node executions.
    mm.load_models_gpu([patcher], memory_required=256 * 1024**2, force_full_load=True)
    return patcher.model, tokenizer
