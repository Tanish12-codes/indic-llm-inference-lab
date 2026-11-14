"""Greedy generation must give the SAME 200 tokens:
uncached (ours) == cached-concat (ours) == cached-prealloc (ours) == HF generate(do_sample=False).
"""
import pytest
import torch

from conftest import DEVICES
from src.gpt2_scratch import generate_cached, generate_uncached, make_cache

PROMPT = "The history of the Indian subcontinent begins with"
N_NEW = 200


def first_divergence(a, b):
    a, b = a.tolist()[0], b.tolist()[0]
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@pytest.fixture(scope="module", params=DEVICES)
def setup(request, our_gpt2, hf_gpt2, tok):
    dev = request.param
    ids = torch.tensor([tok.encode(PROMPT)], device=dev)
    return dev, our_gpt2.to(dev), hf_gpt2.to(dev), ids


def test_cached_equals_uncached_equals_hf(setup):
    dev, ours, hf, ids = setup
    unc = generate_uncached(ours, ids, N_NEW)
    cat = generate_cached(ours, ids, N_NEW, make_cache("concat", ours))
    pre = generate_cached(ours, ids, N_NEW, make_cache("prealloc", ours, max_len=ids.shape[1] + N_NEW))
    with torch.no_grad():
        ref = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=N_NEW,
                          do_sample=False, num_beams=1, pad_token_id=50256)
    for name, out in [("cached-concat", cat), ("cached-prealloc", pre), ("HF generate", ref)]:
        d = first_divergence(unc, out)
        assert d is None, f"[{dev}] {name} differs from uncached at token index {d}"
    assert unc.shape[1] == ids.shape[1] + N_NEW


def test_step_logits_match(setup):
    """Stronger than token equality: the cached logits at every decode step equal a full re-run."""
    dev, ours, _, ids = setup
    cache = make_cache("concat", ours)
    seq = ids
    with torch.no_grad():
        logits = ours(ids, cache=cache)
        for _ in range(20):
            full = ours(seq)[:, -1]
            assert (logits[:, -1] - full).abs().max().item() < 1e-3
            nxt = full.argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
            logits = ours(nxt, cache=cache)


def test_position_bug_is_detectable(setup):
    """If we 'forget' the cache length and give every new token position 0,
    the logits must differ - proving the tests above would catch the classic bug."""
    dev, ours, _, ids = setup
    with torch.no_grad():
        cache = make_cache("concat", ours)
        ours(ids, cache=cache)
        nxt = ours(ids)[:, -1].argmax(-1, keepdim=True)
        good = ours(nxt, cache=cache)[:, -1]
        # buggy version: position 0 instead of cache length
        x = ours.wte(nxt) + ours.wpe(torch.zeros(1, dtype=torch.long, device=dev))[None]
        cache2 = make_cache("concat", ours)
        ours(ids, cache=cache2)
        for i, b in enumerate(ours.h):
            x = b(x, i, cache2, ids.shape[1])
        bad = ours.ln_f(x)[:, -1] @ ours.wte.weight.T
    assert (good - bad).abs().max().item() > 1e-2
