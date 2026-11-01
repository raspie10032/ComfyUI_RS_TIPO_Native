"""Request-local text sampling. No llama.cpp, Transformers, or global RNG changes."""
from dataclasses import dataclass
import math
import time

import torch

from .model import KVCache


@dataclass
class Completion:
    text: str
    token_ids: list[int]
    input_tokens: int
    finish_reason: str
    seconds: float


def choose_token(logits, history, rng, temperature=0.5, top_k=80, top_p=0.95,
                 min_p=0.05, repetition_penalty=1.17):
    scores = logits.detach().to(device="cpu", dtype=torch.float32).flatten().clone()
    if not torch.isfinite(scores).all():
        raise RuntimeError("TIPO produced invalid scores (NaN/Inf). Check the model and dtype.")
    if repetition_penalty != 1 and history:
        ids = torch.tensor(sorted(set(history)), dtype=torch.long)
        original = scores[ids]
        scores[ids] = torch.where(original < 0, original * repetition_penalty,
                                  original / repetition_penalty)
    if temperature == 0:
        return int(scores.argmax())
    scores /= temperature
    if top_k > 0:
        threshold = scores.topk(min(top_k, scores.numel())).values[-1]
        scores[scores < threshold] = -torch.inf
    if min_p > 0:
        scores[scores < scores.max() + math.log(min_p)] = -torch.inf
    if top_p < 1:
        values, indices = scores.sort(descending=True)
        remove = values.softmax(-1).cumsum(-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[indices[remove]] = -torch.inf
    return int(torch.multinomial(scores.softmax(-1), 1, generator=rng))


@torch.inference_mode()
def complete(model, tokenizer, prompt, *, max_new_tokens=128, seed=0,
             temperature=0.5, top_k=80, top_p=0.95, min_p=0.05,
             repetition_penalty=1.17, interrupt=lambda: None, progress=lambda n, total: None,
             attention_control=None):
    if not prompt.strip():
        raise ValueError("TIPO input is empty. Enter tags or a native completion prompt.")
    if not 1 <= max_new_tokens <= 4096:
        raise ValueError("Output token limit must be between 1 and 4096.")
    if not (0 <= temperature <= 5 and 0 < top_p <= 1 and 0 <= min_p <= 1
            and top_k >= 0 and repetition_penalty > 0):
        raise ValueError("Invalid TIPO sampling parameters.")
    input_ids = tokenizer.encode(prompt, add_special_tokens=True).ids
    if attention_control is not None and len(attention_control.owners) != len(input_ids):
        raise ValueError("TIPO attention spans do not match the encoded input.")
    total = len(input_ids) + max_new_tokens
    if total > model.config.max_position_embeddings:
        raise ValueError(f"TIPO context overflow: {len(input_ids)} input + {max_new_tokens} output "
                         f"> {model.config.max_position_embeddings}. Shorten the prompt or output.")
    if tokenizer.token_to_id("<|eos|>") is None:
        raise ValueError("TIPO tokenizer is missing its end-of-sequence token.")
    param = next(model.parameters())
    cache = KVCache(model.config, total, param.device, param.dtype)
    sequence = (list(input_ids) if attention_control is None
                else attention_control.readable_history(input_ids))
    generated = []
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    eos = tokenizer.token_to_id("<|eos|>")
    start = time.perf_counter()
    def forward(ids):
        mask = None if attention_control is None else attention_control.mask(
            cache.position, len(ids), param.device, param.dtype)
        groups = None if attention_control is None else attention_control.mlp_groups(
            cache.position, len(ids), param.device)
        return model(torch.tensor([ids], device=param.device), cache=cache,
                     attention_mask=mask, token_owners=groups)[0, -1]

    interrupt()
    scores = forward(input_ids)
    reason = "length"
    for step in range(max_new_tokens):
        interrupt()
        token = choose_token(scores, sequence, rng, temperature, top_k, top_p, min_p,
                             repetition_penalty)
        generated.append(token)
        sequence.append(token)
        progress(step + 1, max_new_tokens)
        if token == eos:
            reason = "eos"
            break
        if step + 1 < max_new_tokens:
            scores = forward([token])
    return Completion(tokenizer.decode(generated, skip_special_tokens=False), generated,
                      len(input_ids), reason, time.perf_counter() - start)
