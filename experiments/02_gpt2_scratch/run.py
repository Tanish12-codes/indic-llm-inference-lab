"""Step 2 - From-scratch GPT-2 vs Hugging Face: logits difference (FP32), recorded to CSV.
The pass/fail gate is tests/test_gpt2_matches_hf.py; this script records the numbers.
    .venv\\Scripts\\python.exe experiments\\02_gpt2_scratch\\run.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.common import RESULTS, run_metadata  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def main():
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from src.gpt2_scratch import GPT2
    from test_gpt2_matches_hf import PROMPTS
    tok = AutoTokenizer.from_pretrained("gpt2")
    ours = GPT2.from_pretrained("gpt2")
    hf = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    rows = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        ours.to(dev), hf.to(dev)
        for p in PROMPTS:
            ids = torch.tensor([tok.encode(p)], device=dev)
            with torch.no_grad():
                a, b = ours(ids), hf(ids).logits
            rows.append({"prompt": p, "tokens": ids.shape[1],
                         "max_abs_diff": (a - b).abs().max().item(),
                         "mean_abs_diff": (a - b).abs().mean().item(),
                         "argmax_identical": bool(torch.equal(a.argmax(-1), b.argmax(-1))),
                         **run_metadata("gpt2 (ours vs HF eager)", "fp32", 1, ids.shape[1], dev)})
            print(f"[{dev}] {ids.shape[1]:3d} tokens  max|diff|={rows[-1]['max_abs_diff']:.2e}  "
                  f"argmax same={rows[-1]['argmax_identical']}  | {p[:40]}")
    ids = torch.tensor([tok.encode("The meaning of life is")], device=devices[-1])
    from src.gpt2_scratch import generate_uncached
    print("\nSample (greedy, ours):", tok.decode(generate_uncached(ours, ids, 25)[0]))
    pd.DataFrame(rows).to_csv(RESULTS / "02_gpt2_vs_hf.csv", index=False, encoding="utf-8")
    print(f"saved {RESULTS / '02_gpt2_vs_hf.csv'}")


if __name__ == "__main__":
    main()
