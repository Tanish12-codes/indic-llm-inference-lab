"""MHA vs GQA vs MQA - one attention layer that supports any number of KV heads.

    n_kv_heads == n_heads -> MHA (every query head has its own K/V head)
    1 < n_kv_heads < n_heads -> GQA (groups of query heads share one K/V head)
    n_kv_heads == 1 -> MQA (all query heads share one K/V head)

Random weights: used ONLY to measure memory and speed. No quality claims.

Decode-only (one new token per call, T=1), with a preallocated cache. The grouped
attention is done WITHOUT copying K/V: the G query heads of a group become G rows of
one matrix that multiplies that group's single K. A naive version that copies K/V with
repeat_interleave is included to show why that throws the benefit away.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, naive_repeat: bool = False):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.h, self.kv, self.d = n_heads, n_kv_heads, d_model // n_heads
        self.g = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.d, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d, bias=False)   # smaller for GQA/MQA
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d, d_model, bias=False)
        self.naive_repeat = naive_repeat

    def forward(self, x, k_cache, v_cache, pos: int):
        """x: (B, 1, C). k_cache/v_cache: (B, KV, max_len, D) preallocated. pos: write index."""
        B, T, _ = x.shape
        assert T == 1, "decode-only layer"
        q = self.q_proj(x).view(B, self.h, self.d)                    # (B, H, D)
        k = self.k_proj(x).view(B, self.kv, 1, self.d)
        v = self.v_proj(x).view(B, self.kv, 1, self.d)
        k_cache[:, :, pos:pos + 1] = k                               # in-place append
        v_cache[:, :, pos:pos + 1] = v
        K, V = k_cache[:, :, :pos + 1], v_cache[:, :, :pos + 1]       # (B, KV, S, D) views
        if self.naive_repeat and self.g > 1:
            K = K.repeat_interleave(self.g, dim=1)                   # COPIES the cache G times
            V = V.repeat_interleave(self.g, dim=1)
            att = (q.view(B, self.h, 1, self.d) @ K.transpose(-2, -1)) / math.sqrt(self.d)
            y = F.softmax(att, -1) @ V                               # (B, H, 1, D)
        else:
            qg = q.view(B, self.kv, self.g, self.d)                  # group's query heads = rows
            att = (qg @ K.transpose(-2, -1)) / math.sqrt(self.d)    # (B, KV, G, S)
            y = F.softmax(att, -1) @ V                               # (B, KV, G, D)
        return self.o_proj(y.reshape(B, 1, self.h * self.d))


class AttnStack(nn.Module):
    """n_layer attention layers (residual), no MLP - isolates the attention/KV cost."""

    def __init__(self, n_layer, d_model, n_heads, n_kv_heads, naive_repeat=False):
        super().__init__()
        self.layers = nn.ModuleList(GroupedAttention(d_model, n_heads, n_kv_heads, naive_repeat)
                                    for _ in range(n_layer))
        self.n_kv, self.hd = n_kv_heads, d_model // n_heads

    def alloc_cache(self, batch, max_len, dtype, device):
        shape = (len(self.layers), batch, self.n_kv, max_len, self.hd)
        return (torch.randn(shape, dtype=dtype, device=device),
                torch.randn(shape, dtype=dtype, device=device))

    def forward(self, x, kc, vc, pos):
        for i, layer in enumerate(self.layers):
            x = x + layer(x, kc[i], vc[i], pos)
        return x
