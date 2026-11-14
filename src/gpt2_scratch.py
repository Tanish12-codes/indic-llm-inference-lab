"""GPT-2 (124M) forward pass written from scratch in plain PyTorch.

Only the *weights* come from Hugging Face; every operation is ours.
Three gotchas when loading HF GPT-2 weights:
  1. HF stores linear layers as `Conv1D` with weight shape (in, out). nn.Linear wants
     (out, in), so those weights must be TRANSPOSED.
  2. GPT-2 uses the tanh approximation of GELU ("gelu_new").
  3. The output layer (lm_head) is TIED to the token embedding: logits = h @ wte.T
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class Attention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # Q, K, V in one matmul
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, layer_idx: int, cache=None, past_len: int = 0):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)               # now (B, H, past+T, D)
        S = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, H, T, S)
        # causal mask: query i sits at absolute position past_len+i and may see keys 0..past_len+i
        qpos = torch.arange(past_len, past_len + T, device=x.device)[:, None]
        kpos = torch.arange(S, device=x.device)[None, :]
        att = att.masked_fill(kpos > qpos, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))   # gotcha 2


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.eps)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.eps)
        self.mlp = MLP(cfg)

    def forward(self, x, layer_idx, cache=None, past_len=0):
        x = x + self.attn(self.ln_1(x), layer_idx, cache, past_len)    # pre-norm residual
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config = GPT2Config()):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)  # position embeddings
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.eps)

    def forward(self, idx: torch.Tensor, cache=None) -> torch.Tensor:
        """idx: (B, T) token ids. With a cache, only the NEW tokens are passed in."""
        B, T = idx.shape
        past_len = cache.seq_len if cache is not None else 0
        # THE classic KV-cache bug lives here: new tokens' positions start at the cache length.
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        if past_len + T > self.cfg.n_positions:
            raise ValueError(f"sequence {past_len + T} exceeds GPT-2's {self.cfg.n_positions} positions")
        x = self.wte(idx) + self.wpe(pos)[None]
        for i, block in enumerate(self.h):
            x = block(x, i, cache, past_len)
        x = self.ln_f(x)
        return x @ self.wte.weight.T          # gotcha 3: tied output layer -> (B, T, vocab)

    @classmethod
    def from_pretrained(cls, name: str = "gpt2") -> "GPT2":
        from transformers import GPT2LMHeadModel
        hf = GPT2LMHeadModel.from_pretrained(name)
        c = hf.config
        model = cls(GPT2Config(c.vocab_size, c.n_positions, c.n_embd, c.n_layer, c.n_head,
                               c.layer_norm_epsilon))
        sd = {k.removeprefix("transformer."): v for k, v in hf.state_dict().items()}
        conv1d = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        own = model.state_dict()
        with torch.no_grad():
            for k in own:
                w = sd[k]
                if k.endswith(conv1d):
                    w = w.t()                  # gotcha 1: Conv1D (in,out) -> Linear (out,in)
                assert own[k].shape == w.shape, (k, own[k].shape, w.shape)
                own[k].copy_(w)
        del hf
        return model.eval()


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate_uncached(model: GPT2, idx: torch.Tensor, n_new: int) -> torch.Tensor:
    """Greedy, no cache: every step re-runs the WHOLE sequence."""
    for _ in range(n_new):
        logits = model(idx)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model: GPT2, idx: torch.Tensor, n_new: int, cache) -> torch.Tensor:
    """Greedy with a KV cache: prefill the prompt once, then decode one token per step."""
    logits = model(idx, cache=cache)                    # prefill
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [idx, nxt]
    for _ in range(n_new - 1):                          # decode
        logits = model(nxt, cache=cache)                # only the newest token goes in
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat(out, dim=1)


def make_cache(kind: str, model: GPT2, batch: int = 1, max_len: int | None = None):
    """kind: 'concat' | 'prealloc' | 'int8'."""
    from src.kv_cache import ConcatKVCache, Int8KVCache, PreallocKVCache
    c = model.cfg
    p = next(model.parameters())
    max_len = max_len or c.n_positions
    if kind == "concat":
        return ConcatKVCache(c.n_layer)
    if kind == "prealloc":
        return PreallocKVCache(c.n_layer, batch, c.n_head, c.head_dim, max_len, p.dtype, p.device)
    if kind == "int8":
        return Int8KVCache(c.n_layer, batch, c.n_head, c.head_dim, max_len, p.dtype, p.device)
    raise ValueError(kind)
