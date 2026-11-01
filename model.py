# Kohaku architecture adapted from KohakUwULLM (Apache-2.0).
# Modified by Raspie: standalone inference, request-owned KV cache, no Transformers.
"""TIPO v2.1's native PyTorch decoder; checkpoint names match the official export."""

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class KohakuConfig:
    vocab_size: int = 65536
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 2048
    moe_intermediate_size: int = 384
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    scoring_func: str = "sigmoid"
    max_position_embeddings: int = 4096
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-6
    qk_norm: bool = True

    @classmethod
    def from_dict(cls, data):
        if data.get("model_type") != "kohaku":
            raise ValueError("Expected a TIPO v2.1 Kohaku config, not a Llama/GGUF model.")
        if data.get("tie_word_embeddings", False):
            raise ValueError("Tied embeddings are not supported by this TIPO checkpoint loader.")
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (self.weight.numel(),), self.weight, self.eps)


class KVCache:
    """Allocated once per completion, never shared across character calls."""

    def __init__(self, config, capacity, device, dtype):
        self.keys = [torch.empty((1, config.num_key_value_heads, capacity, config.head_dim),
                                 device=device, dtype=dtype)
                     for _ in range(config.num_hidden_layers)]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.position = 0
        self.capacity = capacity

    def append(self, layer, key, value):
        end = self.position + key.shape[2]
        if end > self.capacity:
            raise ValueError("TIPO context limit exceeded; shorten the input or output length.")
        self.keys[layer][:, :, self.position:end].copy_(key)
        self.values[layer][:, :, self.position:end].copy_(value)
        if self.position == 0:
            return key, value
        # Match the contiguous decode layout of the official DynamicCache. Strided
        # capacity views can select a different FP16 SDPA kernel and alter logits.
        return self.keys[layer][:, :, :end].contiguous(), self.values[layer][:, :, :end].contiguous()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.layer = layer
        self.heads, self.kv_heads, self.dim = (
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
        self.q_proj = nn.Linear(cfg.hidden_size, self.heads * self.dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.dim, cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.dim, cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()

    def forward(self, x, cos, sin, cache, attention_mask=None):
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.heads, self.dim))
        k = self.k_norm(self.k_proj(x).view(b, s, self.kv_heads, self.dim))
        v = self.v_proj(x).view(b, s, self.kv_heads, self.dim)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            k, v = cache.append(self.layer, k, v)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            is_causal=attention_mask is None and s > 1,
            enable_gqa=self.heads != self.kv_heads)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1))


class MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        e, h, d = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        self.gate = nn.Linear(d, e, bias=False)
        self.register_buffer("expert_bias", torch.zeros(e), persistent=True)
        self.gate_proj = nn.Parameter(torch.empty(e, h, d))
        self.up_proj = nn.Parameter(torch.empty(e, h, d))
        self.down_proj = nn.Parameter(torch.empty(e, d, h))
        self.shared_expert = MLP(d, h * cfg.n_shared_experts)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        logits = F.linear(flat.float(), self.gate.weight.float())
        if self.cfg.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.cfg.scoring_func == "softmax":
            scores = logits.softmax(-1)
        elif self.cfg.scoring_func == "sqrtsoftplus":
            scores = F.softplus(logits).sqrt()
        else:
            raise ValueError(f"Unsupported expert scoring: {self.cfg.scoring_func}")
        index = (scores + self.expert_bias).topk(self.cfg.num_experts_per_tok, dim=-1).indices
        weight = scores.gather(1, index)
        if self.cfg.norm_topk_prob and self.cfg.num_experts_per_tok > 1:
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-9)
        weight = (weight * self.cfg.routed_scaling_factor).to(x.dtype)
        out = torch.zeros_like(flat)
        # Preserve the official grouped dispatch and accumulation for baseline parity.
        for expert in index.unique():
            rows, slot = (index == expert).nonzero(as_tuple=True)
            taken = flat[rows]
            hidden = F.silu(taken @ self.gate_proj[expert].T) * (taken @ self.up_proj[expert].T)
            out.index_add_(0, rows, (hidden @ self.down_proj[expert].T) * weight[rows, slot, None])
        return (out + self.shared_expert(flat)).view(x.shape)


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.self_attn = Attention(cfg, layer)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size) if layer < cfg.first_k_dense else MoE(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cache, attention_mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, attention_mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class KohakuForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Decoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, cache=None, attention_mask=None, last_only=True):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("TIPO native currently accepts one nonempty sequence at a time.")
        seen = 0 if cache is None else cache.position
        seq = input_ids.shape[1]
        if seen + seq > self.config.max_position_embeddings:
            raise ValueError("TIPO supports at most 4096 tokens including the generated output.")
        x = self.model.embed_tokens(input_ids)
        # Float32 RoPE frequencies stay accurate when the model moves between dtypes/devices.
        inv = 1.0 / (self.config.rope_theta ** (
            torch.arange(0, self.config.head_dim, 2, device=x.device).float() / self.config.head_dim))
        pos = torch.arange(seen, seen + seq, device=x.device).float()
        angles = torch.cat((pos[:, None] * inv, pos[:, None] * inv), dim=-1)
        cos, sin = angles.cos().to(x.dtype)[None, :, None], angles.sin().to(x.dtype)[None, :, None]
        # SDPA's is_causal is top-left aligned; cached multi-token chunks need an offset mask.
        if attention_mask is None and seen and seq > 1:
            attention_mask = (torch.arange(seen + seq, device=x.device)[None, :] <=
                              torch.arange(seen, seen + seq, device=x.device)[:, None])
        for layer in self.model.layers:
            x = layer(x, cos, sin, cache, attention_mask)
        if cache is not None:
            cache.position += seq
        x = self.model.norm(x[:, -1:] if last_only else x)
        return self.lm_head(x)
