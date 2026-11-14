"""Step 3 - KV cache correctness: greedy uncached == cached == HF generate for 200 tokens.
The pass/fail gate is tests/test_kv_cache_equivalence.py; this script records the numbers
and also shows what the classic position-id bug does to the output.
    .venv\\Scripts\\python.exe experiments\\03_kv_cache\\run.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.common import RESULTS, get_device, run_metadata  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
PROMPT = "The history of the Indian subcontinent begins with"
N_NEW = 200


@torch.no_grad()
def generate_buggy_positions(model, idx, n_new, cache):
    """Same as generate_cached, but every decoded token gets position id 0 instead of the
    cache length (the attention mask stays correct, so ONLY the position is wrong)."""
    logits = model(idx, cache=cache)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [idx, nxt]
    zero = torch.zeros(1, dtype=torch.long, device=idx.device)
    for _ in range(n_new - 1):
        past = cache.seq_len
        x = model.wte(nxt) + model.wpe(zero)[None]          # BUG: should be wpe(past)
        for i, block in enumerate(model.h):
            x = block(x, i, cache, past)
        nxt = (model.ln_f(x) @ model.wte.weight.T)[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat(out, dim=1)


def main():
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from src.gpt2_scratch import GPT2, generate_cached, generate_uncached, make_cache
    dev = get_device()
    tok = AutoTokenizer.from_pretrained("gpt2")
    ours = GPT2.from_pretrained("gpt2").to(dev)
    hf = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(dev).eval()
    ids = torch.tensor([tok.encode(PROMPT)], device=dev)
    P = ids.shape[1]
    outs = {
        "uncached (ours)": generate_uncached(ours, ids, N_NEW),
        "cached concat (ours)": generate_cached(ours, ids, N_NEW, make_cache("concat", ours)),
        "cached prealloc (ours)": generate_cached(ours, ids, N_NEW, make_cache("prealloc", ours, max_len=P + N_NEW)),
    }
    with torch.no_grad():
        outs["HF generate greedy"] = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=N_NEW,
                                                 do_sample=False, num_beams=1, pad_token_id=50256)
    outs["BUGGY: position 0 for new tokens"] = generate_buggy_positions(ours, ids, N_NEW,
                                                                         make_cache("concat", ours))
    ref = outs["uncached (ours)"][0, P:]
    rows = []
    for name, o in outs.items():
        gen = o[0, P:]
        same = (gen == ref)
        rows.append({"method": name, "new_tokens": N_NEW,
                     "identical_to_uncached_pct": 100 * same.float().mean().item(),
                     "first_divergence": int((~same).nonzero()[0]) if (~same).any() else None,
                     **run_metadata("gpt2", "fp32", 1, P + N_NEW, dev.type)})
        print(f"{name:34s} {rows[-1]['identical_to_uncached_pct']:6.1f}% identical  "
              f"first divergence: {rows[-1]['first_divergence']}")
    print("\nCorrect text :", tok.decode(ref[:40]))
    print("Buggy text   :", tok.decode(outs["BUGGY: position 0 for new tokens"][0, P:P + 40]))
    pd.DataFrame(rows).to_csv(RESULTS / "03_kv_equivalence.csv", index=False)
    print(f"saved {RESULTS / '03_kv_equivalence.csv'}")


if __name__ == "__main__":
    main()
