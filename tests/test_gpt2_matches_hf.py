"""Our from-scratch GPT-2 must produce the same logits as Hugging Face's (FP32)."""
import pytest
import torch

from conftest import DEVICES

PROMPTS = [
    "The quick brown fox jumps over the lazy dog because",
    "In 1947, India became independent and",
    # GPT-2 does not *understand* Hindi - we only check the maths is identical on any bytes.
    "भारत एक विशाल देश है और यहाँ कई भाषाएँ बोली जाती हैं।",
]
ATOL = 1e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_hf(device, prompt, our_gpt2, hf_gpt2, tok):
    ours, hf = our_gpt2.to(device), hf_gpt2.to(device)
    ids = torch.tensor([tok.encode(prompt)], device=device)
    with torch.no_grad():
        a = ours(ids)
        b = hf(ids).logits
    diff = (a - b).abs().max().item()
    print(f"\n[{device}] {len(ids[0])} tokens, max |ours - HF| = {diff:.2e}  "
          f"(logit range {b.min().item():.1f}..{b.max().item():.1f})")
    assert a.shape == b.shape
    assert diff < ATOL, f"max abs diff {diff:.3e} >= {ATOL}"
    assert torch.equal(a.argmax(-1), b.argmax(-1))


def test_weights_are_tied(our_gpt2):
    # The output layer IS the embedding matrix: no separate lm_head parameter exists.
    names = [n for n, _ in our_gpt2.named_parameters()]
    assert not any("lm_head" in n for n in names)
    n_params = sum(p.numel() for p in our_gpt2.parameters())
    assert n_params == 124_439_808, n_params   # GPT-2 small, counting wte once
