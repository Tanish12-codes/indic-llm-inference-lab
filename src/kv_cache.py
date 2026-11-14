"""Three KV caches with the same interface.

    cache.seq_len                       -> how many tokens are already stored
    k_all, v_all = cache.update(layer, k_new, v_new)
                                        -> store new K/V (B, H, T_new, D) for this layer
                                           and return ALL K/V so far (B, H, S, D)
    cache.nbytes()                      -> bytes the cache's storage occupies right now

1. ConcatKVCache   : torch.cat a new, bigger tensor every step (simple, wasteful)
2. PreallocKVCache : one buffer sized for max_len, write new tokens in place
3. Int8KVCache     : preallocated int8 buffer + one FP scale per (token, head) vector;
                     dequantized back to float right before attention
"""
from __future__ import annotations

import torch


class ConcatKVCache:
    name = "concat"

    def __init__(self, n_layer: int):
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer

    @property
    def seq_len(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            # allocates a NEW tensor of length S+T and copies the old S entries into it
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)

    def used_nbytes(self) -> int:
        return self.nbytes()


class PreallocKVCache:
    name = "prealloc"

    def __init__(self, n_layer: int, batch: int, n_kv_heads: int, head_dim: int, max_len: int,
                 dtype=torch.float32, device="cpu"):
        shape = (n_layer, batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.lens = [0] * n_layer
        self.max_len = max_len

    @property
    def seq_len(self) -> int:
        return self.lens[0]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        s, t = self.lens[layer], k.shape[2]
        if s + t > self.max_len:
            raise RuntimeError(f"KV cache full: {s}+{t} > max_len={self.max_len}")
        self.k[layer, :, :, s:s + t] = k          # in-place write, no new allocation
        self.v[layer, :, :, s:s + t] = v
        self.lens[layer] = s + t
        return self.k[layer, :, :, :s + t], self.v[layer, :, :, :s + t]   # views, no copy

    def nbytes(self) -> int:
        """Whole buffer (sized for max_len) - what is actually reserved in VRAM."""
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    def used_nbytes(self) -> int:
        """Only the slots filled so far - comparable with the formula."""
        return self.nbytes() * self.seq_len // self.max_len


def quantize_int8(x: torch.Tensor):
    """Symmetric per-vector int8: one scale per last-dim vector (here: per token per head)."""
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-8) / 127.0
    q = torch.round(x.float() / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale


class Int8KVCache:
    name = "int8"

    def __init__(self, n_layer: int, batch: int, n_kv_heads: int, head_dim: int, max_len: int,
                 out_dtype=torch.float32, device="cpu", scale_dtype=torch.float16):
        shape = (n_layer, batch, n_kv_heads, max_len, head_dim)
        sshape = (n_layer, batch, n_kv_heads, max_len, 1)
        self.kq = torch.zeros(shape, dtype=torch.int8, device=device)
        self.vq = torch.zeros(shape, dtype=torch.int8, device=device)
        self.ks = torch.zeros(sshape, dtype=scale_dtype, device=device)
        self.vs = torch.zeros(sshape, dtype=scale_dtype, device=device)
        self.lens = [0] * n_layer
        self.max_len = max_len
        self.out_dtype = out_dtype

    @property
    def seq_len(self) -> int:
        return self.lens[0]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        s, t = self.lens[layer], k.shape[2]
        if s + t > self.max_len:
            raise RuntimeError(f"KV cache full: {s}+{t} > max_len={self.max_len}")
        kq, ks = quantize_int8(k)
        vq, vs = quantize_int8(v)
        self.kq[layer, :, :, s:s + t], self.ks[layer, :, :, s:s + t] = kq, ks.to(self.ks.dtype)
        self.vq[layer, :, :, s:s + t], self.vs[layer, :, :, s:s + t] = vq, vs.to(self.vs.dtype)
        self.lens[layer] = s + t
        e = s + t
        # Dequantize for attention. NOTE: this builds a temporary float copy of this layer's
        # K/V each step - a real int8 attention kernel would dequantize inside the kernel.
        k_all = self.kq[layer, :, :, :e].to(self.out_dtype) * self.ks[layer, :, :, :e].to(self.out_dtype)
        v_all = self.vq[layer, :, :, :e].to(self.out_dtype) * self.vs[layer, :, :, :e].to(self.out_dtype)
        return k_all, v_all

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.kq, self.vq, self.ks, self.vs))

    def used_nbytes(self) -> int:
        return self.nbytes() * self.seq_len // self.max_len


def kv_bytes_formula(n_layer: int, n_kv_heads: int, head_dim: int, tokens: int, batch: int,
                     bytes_per_elem: float) -> float:
    """2 (K and V) x layers x KV heads x head_dim x tokens x batch x bytes."""
    return 2 * n_layer * n_kv_heads * head_dim * tokens * batch * bytes_per_elem
