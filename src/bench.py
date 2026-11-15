"""Honest GPU timing and memory measurement.

Rules this file enforces:
  * warm-up runs first (the first CUDA call pays one-off costs: context, kernel loading, autotuning)
  * torch.cuda.synchronize() before starting and before stopping the clock - GPU work is
    asynchronous, so without it you time how fast Python *queues* work, not how fast it runs
  * several repeats; report median (robust to hiccups) plus min and max
  * memory: reset_peak_memory_stats() before, max_memory_allocated() after
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import torch


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn: Callable[[], object], device, warmup: int = 2, repeats: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(times), "min_s": min(times), "max_s": max(times),
            "repeats": repeats, "warmup": warmup}


def peak_memory(fn: Callable[[], object], device) -> dict:
    """Run fn once; return memory allocated before, peak during, and still-held after (bytes)."""
    if torch.device(device).type != "cuda":
        fn()
        return {"mem_before_B": float("nan"), "mem_peak_B": float("nan"), "mem_after_B": float("nan")}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    after = torch.cuda.memory_allocated()
    del out
    return {"mem_before_B": before, "mem_peak_B": peak, "mem_after_B": after,
            "peak_over_baseline_B": peak - before}


def bench_generation(model, ids, method: str, n_new: int, device, warmup=2, repeats=5) -> dict:
    """Time + memory for greedy generation of n_new tokens with one method.

    method: 'uncached' | 'concat' | 'prealloc' | 'int8'
    """
    from src.gpt2_scratch import generate_cached, generate_uncached, make_cache
    from src.kv_cache import kv_bytes_formula

    max_len = ids.shape[1] + n_new
    holder = {}

    def run():
        if method == "uncached":
            return generate_uncached(model, ids, n_new)
        cache = make_cache(method, model, batch=ids.shape[0], max_len=max_len)
        out = generate_cached(model, ids, n_new, cache)
        holder["cache"] = cache
        return out, cache

    t = time_fn(run, device, warmup, repeats)
    holder.clear()
    m = peak_memory(run, device)
    c = model.cfg
    elem = next(model.parameters()).element_size()
    row = {"method": method, "new_tokens": n_new, "prompt_tokens": ids.shape[1], **t, **m,
           "ms_per_token": 1000 * t["median_s"] / n_new, "tokens_per_s": n_new / t["median_s"]}
    if method != "uncached":
        cache = holder["cache"]
        # tokens stored = prompt + n_new - 1  (the very last generated token is never fed back in)
        stored = cache.seq_len
        row.update({
            "kv_tokens_stored": stored,
            "kv_used_bytes": cache.used_nbytes(),
            "kv_reserved_bytes": cache.nbytes(),
            "kv_formula_bytes": kv_bytes_formula(c.n_layer, c.n_head, c.head_dim, stored,
                                                 ids.shape[0], 1 if method == "int8" else elem),
            "kv_mem_delta_bytes": m.get("mem_after_B", float("nan")) - m.get("mem_before_B", float("nan")),
        })
    holder.clear()
    return row


def time_interleaved(fns: dict, device, warmup: int = 2, repeats: int = 7) -> dict:
    """Time several functions in ROUND-ROBIN order: A B C, A B C, ... (rotating who goes first).

    Why: a laptop GPU changes its clock speed with temperature and power state. If we time all of A
    and then all of B, a slow-down that happens during B looks like "B is slower". Interleaving
    spreads any drift evenly over every method, so the *comparison* stays fair.
    """
    names = list(fns)
    for _ in range(warmup):
        for n in names:
            fns[n]()
    times = {n: [] for n in names}
    for rep in range(repeats):
        order = names[rep % len(names):] + names[:rep % len(names)]   # rotate starting method
        for n in order:
            _sync(device)
            t0 = time.perf_counter()
            fns[n]()
            _sync(device)
            times[n].append(time.perf_counter() - t0)
    return {n: {"median_s": statistics.median(t), "min_s": min(t), "max_s": max(t),
                "repeats": repeats, "warmup": warmup, "interleaved": True} for n, t in times.items()}


def gpu_warmup(device, seconds: float = 3.0):
    """Keep the GPU busy for a few seconds so a laptop GPU ramps up to its working clock first."""
    if torch.device(device).type != "cuda":
        return
    x = torch.randn(2048, 2048, device=device)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        x = (x @ x).tanh_()
    torch.cuda.synchronize()
