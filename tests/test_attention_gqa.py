"""The no-copy grouped attention must equal the naive repeat_interleave version,
and MHA through our module must equal plain attention."""
import torch

from src.attention import GroupedAttention


def _run(layer, x, kc, vc, pos):
    with torch.no_grad():
        return layer(x, kc.clone(), vc.clone(), pos)


def test_grouped_equals_naive_repeat():
    torch.manual_seed(0)
    for kvh in (12, 4, 1):
        fast = GroupedAttention(768, 12, kvh)
        slow = GroupedAttention(768, 12, kvh, naive_repeat=True)
        slow.load_state_dict(fast.state_dict())
        x = torch.randn(2, 1, 768)
        kc, vc = torch.randn(2, kvh, 40, 64), torch.randn(2, kvh, 40, 64)
        a, b = _run(fast, x, kc, vc, 30), _run(slow, x, kc, vc, 30)
        assert (a - b).abs().max().item() < 1e-5, kvh


def test_kv_projection_shrinks():
    assert GroupedAttention(768, 12, 4).k_proj.weight.shape == (256, 768)
    assert GroupedAttention(768, 12, 1).k_proj.weight.shape == (64, 768)
