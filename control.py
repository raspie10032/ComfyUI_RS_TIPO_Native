"""Causal, owner-aware attention bias for native TIPO input tokens."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class OwnedAttention:
    # -1 = format/header, 0 = common, positive numbers = character owners.
    owners: tuple[int, ...]
    weights: tuple[float, ...]
    output_owner: int

    def __post_init__(self):
        if not self.owners or len(self.owners) != len(self.weights):
            raise ValueError("TIPO attention needs one owner and weight per input token.")
        if self.output_owner < 0 or any(o < -1 for o in self.owners):
            raise ValueError("Invalid TIPO token owner.")
        if any(not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError("TIPO attention weights must be finite and nonnegative.")
        if self.owners[0] != -1 or self.weights[0] <= 0:
            raise ValueError("TIPO requires a readable format token before the owned spans.")

    def mask(self, position, count, device, dtype):
        end = position + count
        owners = torch.tensor(self.owners, device=device)
        weights = torch.tensor(self.weights, device=device, dtype=torch.float32)
        extra = end - len(self.owners)
        if extra > 0:
            owners = torch.cat((owners, owners.new_full((extra,), self.output_owner)))
            # Generated history has unit weight, independent of source-token emphasis.
            weights = torch.cat((weights, weights.new_ones(extra)))
        keys = owners[:end][None, :]
        queries = owners[position:end, None]
        allowed = (keys == -1) | (keys == queries) | ((queries > 0) & (keys == 0))
        causal = torch.arange(end, device=device)[None, :] <= torch.arange(
            position, end, device=device)[:, None]
        bias = weights[:end].log()[None, :].expand(count, -1).clone()
        return bias.masked_fill(~(allowed & causal), -torch.inf).to(dtype)

    def mlp_groups(self, position, count, device):
        # MoE dispatch batches tokens by expert. In FP16, changing *blocked*
        # tokens can change GEMM shapes and rounding for otherwise independent
        # tokens. Keep strict-isolation owners in separate dispatch groups too.
        if len({o for o in self.owners if o > 0}) < 2 and all(w > 0 for w in self.weights):
            return None
        labels = [self.owners[i] if i < len(self.owners) else self.output_owner
                  for i in range(position, position + count)]
        return torch.tensor(labels, device=device)

    def readable_history(self, token_ids):
        # Repetition penalties are another input path: a blocked prompt must
        # not still suppress its words in the sampler's score processor.
        return [token for token, owner, weight in zip(token_ids, self.owners, self.weights)
                if weight > 0 and (owner in (-1, self.output_owner) or
                                   (self.output_owner > 0 and owner == 0))]


def controlled_prompt(tokenizer, common, character=None, *, length="long",
                      common_weight=0.5, character_weight=1.5):
    """Keep the learned tag format; owner labels live outside the model text.

    Each target gets an independent sequence/cache. Other characters are never
    present. Offsets come from the *whole* encoding, including boundary merges.
    """
    if length not in ("short", "long", "very_long"):
        raise ValueError("Unknown TIPO target length.")
    for weight in (common_weight, character_weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("TIPO attention weights must be finite and nonnegative.")
    prefix = f"target: <|{length}|>\ntag: "
    common = common.strip().rstrip(",")
    character = None if character is None else character.strip().rstrip(",")
    if not common and not character:
        raise ValueError("Enter a common or character prompt before running TIPO.")
    text = prefix
    spans = []
    if common:
        start = len(text)
        text += common + ", "
        spans.append((start, len(text), 0, common_weight if character else 1.0))
    if character:
        start = len(text)
        text += character + ", "
        spans.append((start, len(text), 1, character_weight))
    encoding = tokenizer.encode(text, add_special_tokens=True)
    owners, weights = [], []
    for start, end in encoding.offsets:
        owner, weight = -1, 1.0
        if end > start:
            for left, right, candidate, strength in spans:
                if start < right and end > left:
                    owner, weight = candidate, strength
        owners.append(owner)
        weights.append(weight)
    return text, OwnedAttention(tuple(owners), tuple(weights), 1 if character else 0)
