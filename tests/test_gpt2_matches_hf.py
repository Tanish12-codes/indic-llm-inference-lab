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
# Tolerance = absolute 1e-4 + relative 1e-6 of the largest |logit|.
# Why relative: GPT-2 logits are ~ -100. Near 100, one float32 step ("ulp") is 7.6e-6, so 1e-4 is only
# ~13 ulps. Different (but equally correct) GPU kernels add numbers in a different order and can land
# ~15 ulps apart on long sequences. 1e-6 relative = ~13 more ulps at |logit|=100. A real bug (wrong
# GELU, missing transpose, wrong position) gives diffs of 1e-3 .. 10, i.e. 10x - 10,000,000x larger.
ATOL = 1e-4
RTOL = 1e-6


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_hf(device, prompt, our_gpt2, hf_gpt2, tok):
    ours, hf = our_gpt2.to(device), hf_gpt2.to(device)
    ids = torch.tensor([tok.encode(prompt)], device=device)
    with torch.no_grad():
        a = ours(ids)
        b = hf(ids).logits
    diff = (a - b).abs().max().item()
    tol = ATOL + RTOL * b.abs().max().item()
    pdiff = (a.softmax(-1) - b.softmax(-1)).abs().max().item()
    print(f"\n[{device}] {len(ids[0])} tokens, max |ours - HF| = {diff:.2e} (tol {tol:.2e}), "
          f"max prob diff = {pdiff:.1e}  (logit range {b.min().item():.1f}..{b.max().item():.1f})")
    assert a.shape == b.shape
    assert diff < tol, f"max abs diff {diff:.3e} >= tol {tol:.3e}"
    # Probabilities: 1e-5 = 0.001 percentage points. (An earlier 1e-6 limit was set without measuring
    # and was too strict: observed 1e-6..7e-6 - the same float rounding as above, after softmax.)
    assert pdiff < 1e-5, f"probability diff {pdiff:.2e}"
    assert torch.equal(a.argmax(-1), b.argmax(-1))


@pytest.mark.skipif(len(DEVICES) < 2, reason="needs CPU and GPU")
@pytest.mark.parametrize("prompt", PROMPTS)
def test_our_error_is_within_hf_noise_floor(prompt, our_gpt2, hf_gpt2, tok):
    """Strongest check: HF's OWN model disagrees with itself between CPU and GPU by some amount
    (the hardware's rounding 'noise floor'). Our model vs HF must not be worse than that."""
    ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        hf_cpu = hf_gpt2.to("cpu")(ids).logits
        hf_gpu = hf_gpt2.to("cuda")(ids.cuda()).logits.cpu()
        ours_gpu = our_gpt2.to("cuda")(ids.cuda()).cpu()
    floor = (hf_cpu - hf_gpu).abs().max().item()
    ours = (ours_gpu - hf_gpu).abs().max().item()
    print(f"\nnoise floor (HF cpu vs HF gpu) = {floor:.2e};  ours vs HF (gpu) = {ours:.2e}")
    assert ours <= floor


def test_weights_are_tied(our_gpt2):
    # The output layer IS the embedding matrix: no separate lm_head parameter exists.
    names = [n for n, _ in our_gpt2.named_parameters()]
    assert not any("lm_head" in n for n in names)
    n_params = sum(p.numel() for p in our_gpt2.parameters())
    assert n_params == 124_439_808, n_params   # GPT-2 small, counting wte once
