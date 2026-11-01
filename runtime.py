"""ComfyUI-managed loading. All model execution is local native PyTorch."""
from functools import lru_cache
import json
from pathlib import Path
import threading

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from .model import KohakuConfig, KohakuForCausalLM

MODEL_NAME = "TIPO-v2.1-1B-A200M"
MODEL_REVISION = "f5a318524a4ab30cdbbf51816cf406170f454e65"
LOCK = threading.RLock()


def model_path():
    import folder_paths
    roots = [Path(folder_paths.models_dir) / "tipo"]
    # Honor extra_model_paths.yaml entries named "tipo" as well as the default.
    if "tipo" in folder_paths.folder_names_and_paths:
        roots += [Path(p) for p in folder_paths.get_folder_paths("tipo")]
    for root in roots:
        candidate = root / MODEL_NAME
        if (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(
        f"TIPO native model is missing. Put config.json, tokenizer.json and model.safetensors "
        f"in models/tipo/{MODEL_NAME}/. The existing GGUF file cannot be used. "
        "Run this project's tools/download_model.py with that destination.")


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
