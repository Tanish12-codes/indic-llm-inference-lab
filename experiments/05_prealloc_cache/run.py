"""Step 5 - Optimization 1: preallocated KV cache (in-place writes) vs torch.cat.

Correctness first: tests/test_kv_cache_equivalence.py already checks prealloc == uncached == HF.
This script re-checks token equality, then measures.
    .venv\\Scripts\\python.exe experiments\\05_prealloc_cache\\run.py
Outputs: results/05_prealloc_vs_concat.csv, plots/05_prealloc_vs_concat.png
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.common import PLOTS, RESULTS, get_device, run_metadata, set_seed  # noqa: E402

spec = importlib.util.spec_from_file_location("exp04", ROOT / "experiments/04_speed_memory/run.py")
exp04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp04)

OUT = RESULTS / "05_prealloc_vs_concat.csv"
LENGTHS = [50, 100, 200, 500, 900]


def main():
    from transformers import AutoTokenizer
    from src.gpt2_scratch import GPT2, generate_cached, make_cache
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(0)
    dev = get_device()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2.from_pretrained("gpt2").to(dev)
    ids = torch.tensor([tok.encode(exp04.PROMPT)], device=dev)

    a = generate_cached(model, ids, 300, make_cache("concat", model))
    b = generate_cached(model, ids, 300, make_cache("prealloc", model, max_len=ids.shape[1] + 300))
    same = torch.equal(a, b)
    print(f"correctness: concat == prealloc for 300 tokens: {same}")
    if not same:
        sys.exit("Outputs differ - fix correctness before measuring speed.")

    from src.bench import gpu_warmup, peak_memory, time_interleaved
    from src.kv_cache import kv_bytes_formula
    gpu_warmup(dev)
    rows = []
    c = model.cfg
    for n in LENGTHS:
        caches = {}

        def make_run(method):
            def run():
                cache = make_cache(method, model, max_len=ids.shape[1] + n)
                out = generate_cached(model, ids, n, cache)
                caches[method] = cache
                return out, cache
            return run

        runs = {m: make_run(m) for m in ["concat", "prealloc"]}
        timing = time_interleaved(runs, dev, warmup=2, repeats=7)
        for method in ["concat", "prealloc"]:
            caches.clear()
            mem = peak_memory(runs[method], dev)
            cache = caches[method]
            t = timing[method]
            r = {"method": method, "new_tokens": n, "prompt_tokens": ids.shape[1], **t, **mem,
                 "ms_per_token": 1000 * t["median_s"] / n, "tokens_per_s": n / t["median_s"],
                 "kv_tokens_stored": cache.seq_len, "kv_used_bytes": cache.used_nbytes(),
                 "kv_reserved_bytes": cache.nbytes(),
                 "kv_formula_bytes": kv_bytes_formula(c.n_layer, c.n_head, c.head_dim, cache.seq_len, 1, 4),
                 "tokens_identical_to_concat": same,
                 **run_metadata("gpt2 (124M, from scratch)", "fp32", 1, ids.shape[1] + n, dev.type)}
            rows.append(r)
            print(f"{method:9s} n={n:4d}  median {r['median_s'] * 1000:8.1f} ms (min {r['min_s'] * 1000:.1f}, "
                  f"max {r['max_s'] * 1000:.1f})  {r['ms_per_token']:.3f} ms/token  "
                  f"peak+{r['peak_over_baseline_B'] / 2**20:.2f} MiB  kv reserved {r['kv_reserved_bytes'] / 2**20:.2f} MiB")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {OUT}")
    exp04.plot(OUT, PLOTS / "05_prealloc_vs_concat.png", title="Preallocated vs torch.cat KV cache")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        exp04.plot(OUT, PLOTS / "05_prealloc_vs_concat.png", title="Preallocated vs torch.cat KV cache")
    else:
        main()
