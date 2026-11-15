"""Step 4 - Speed and memory: cached vs uncached greedy generation (GPT-2 small, FP32).

Run AFTER the tests pass:
    .venv\\Scripts\\python.exe experiments\\04_speed_memory\\run.py
Outputs: results/04_speed_memory.csv, plots/04_speed_memory.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.bench import bench_generation  # noqa: E402
from src.common import PLOTS, RESULTS, get_device, run_metadata, set_seed  # noqa: E402

PROMPT = "The history of the Indian subcontinent begins with the first humans who arrived"
OUT = RESULTS / "04_speed_memory.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="50,100,200,500")
    ap.add_argument("--methods", default="uncached,concat")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from src.gpt2_scratch import GPT2
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(0)
    dev = get_device()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2.from_pretrained("gpt2").to(dev)
    ids = torch.tensor([tok.encode(PROMPT)], device=dev)
    weights_B = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"device={dev}  prompt tokens={ids.shape[1]}  weights={weights_B / 2**20:.1f} MiB")

    rows = []
    for n in [int(x) for x in args.lengths.split(",")]:
        for method in args.methods.split(","):
            r = bench_generation(model, ids, method, n, dev, warmup=2, repeats=args.repeats)
            r.update(run_metadata("gpt2 (124M, from scratch)", "fp32", 1, ids.shape[1] + n, dev.type))
            r["weights_bytes"] = weights_B
            rows.append(r)
            kv = (f" kv used={r['kv_used_bytes'] / 2**20:.2f} MiB formula={r['kv_formula_bytes'] / 2**20:.2f} MiB"
                  if "kv_used_bytes" in r else "")
            print(f"{method:9s} n={n:4d}  median {r['median_s'] * 1000:8.1f} ms "
                  f"(min {r['min_s'] * 1000:.1f}, max {r['max_s'] * 1000:.1f})  "
                  f"{r['ms_per_token']:.2f} ms/token  peak+{r.get('peak_over_baseline_B', float('nan')) / 2**20:.1f} MiB{kv}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")
    if args.out == str(OUT):
        plot()


def plot(csv=OUT, png=PLOTS / "04_speed_memory.png", title="Cached vs uncached"):
    import matplotlib.pyplot as plt
    from src import plotstyle
    plotstyle.apply()
    df = pd.read_csv(csv)
    order = ["uncached", "concat", "prealloc", "int8"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for m in [x for x in order if x in set(df.method)]:
        d = df[df.method == m].sort_values("new_tokens")
        kw = dict(color=plotstyle.color_for(m, order), marker="o", label=m)
        axes[0].plot(d.new_tokens, d.median_s, **kw)
        axes[0].fill_between(d.new_tokens, d.min_s, d.max_s, color=kw["color"], alpha=0.15, linewidth=0)
        axes[1].plot(d.new_tokens, d.ms_per_token, **kw)
        axes[2].plot(d.new_tokens, d.peak_over_baseline_B / 2**20, **kw)
    if "uncached" in set(df.method) and "concat" in set(df.method):
        u = df[df.method == "uncached"].set_index("new_tokens").median_s
        c = df[df.method == "concat"].set_index("new_tokens").median_s
        for n in u.index.intersection(c.index):
            axes[0].annotate(f"{u[n] / c[n]:.1f}x", (n, u[n]), xytext=(0, 6), textcoords="offset points",
                             ha="center", fontsize=8, color=plotstyle.TEXT2)
    # A run interrupted by the OS (sleep / frozen console) can have an absurd max. Keep the axis readable
    # and say so on the chart instead of hiding it.
    cap = 1.3 * df.median_s.max()
    axes[0].set_ylim(0, cap)
    off = df[df.max_s > cap]
    if len(off):
        axes[0].annotate(f"{len(off)} run(s) with max off-scale\n(OS interruption; median unaffected)",
                         (0.02, 0.80), xycoords="axes fraction", fontsize=8, color=plotstyle.TEXT2)
    axes[0].set(title="Total time (median; band = min-max)", xlabel="new tokens", ylabel="seconds")
    axes[1].set(title="Time per generated token", xlabel="new tokens", ylabel="ms / token")
    axes[2].set(title="Peak GPU memory above weights", xlabel="new tokens", ylabel="MiB above weights")
    for a in axes:
        t = a.get_title()
        a.set_title("")
        a.set_title(t, loc="left")
        a.legend()
    meta = df.iloc[0]
    fig.suptitle(f"{title} - GPT-2 small (small-model result), {meta.precision}, batch {meta.batch}, {meta.gpu}",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.TEXT2)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"saved {png}")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot()
    else:
        main()
