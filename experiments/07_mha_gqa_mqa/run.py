"""Step 7 - MHA vs GQA vs MQA (random weights): KV memory per token and decode speed only.

12 attention layers, d_model=768, 12 query heads of 64 dims (GPT-2-small shaped), FP16.
KV heads: 12 (MHA), 4 (GQA, groups of 3), 1 (MQA). Plus 'GQA-4 naive' that copies K/V with
repeat_interleave, to show why a real implementation must not do that.
    .venv\\Scripts\\python.exe experiments\\07_mha_gqa_mqa\\run.py
Outputs: results/07_mha_gqa_mqa.csv, plots/07_mha_gqa_mqa.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.attention import AttnStack  # noqa: E402
from src.common import PLOTS, RESULTS, get_device, run_metadata, set_seed  # noqa: E402
from src.kv_cache import kv_bytes_formula  # noqa: E402

OUT = RESULTS / "07_mha_gqa_mqa.csv"
VARIANTS = [("MHA (12 KV heads)", 12, False), ("GQA (4 KV heads)", 4, False),
            ("MQA (1 KV head)", 1, False), ("GQA-4 naive repeat", 4, True)]
L, D_MODEL, H = 12, 768, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--contexts", default="256,1024,4096,8192")
    ap.add_argument("--steps", type=int, default=20, help="decode steps per timed run")
    args = ap.parse_args()
    set_seed(0)
    dev = get_device()
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    elem = torch.finfo(dtype).bits // 8
    from src.bench import gpu_warmup, time_interleaved
    gpu_warmup(dev)
    rows = []
    for ctx in [int(c) for c in args.contexts.split(",")]:
        # Fast variants are timed together, interleaved. The naive variant makes big temporary
        # copies, so it is timed in its own group afterwards (to keep memory within ~5 GiB).
        for group in ([v for v in VARIANTS if not v[2]], [v for v in VARIANTS if v[2]]):
            built = {}
            for name, kvh, naive in group:
                need = kv_bytes_formula(L, kvh, D_MODEL // H, ctx + args.steps, args.batch, elem)
                if dev.type == "cuda":
                    free = torch.cuda.mem_get_info()[0]
                    extra = 2 * need / L * (H // kvh) if naive else 0
                    if need + extra + 4e8 > free:
                        print(f"skip {name} ctx={ctx}: needs {need / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB")
                        continue
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_allocated()
                model = AttnStack(L, D_MODEL, H, kvh, naive).to(dev, dtype).eval()
                kc, vc = model.alloc_cache(args.batch, ctx + args.steps, dtype, dev)
                measured = (kc.numel() + vc.numel()) * kc.element_size()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    measured_cuda = torch.cuda.memory_allocated() - before \
                        - sum(p.numel() * p.element_size() for p in model.parameters())
                else:
                    measured_cuda = measured
                built[name] = (model, kc, vc, kvh, naive, measured, measured_cuda)
            if not built:
                continue
            x = torch.randn(args.batch, 1, D_MODEL, device=dev, dtype=dtype)

            def make_run(model, kc, vc):
                def run():
                    with torch.no_grad():
                        y = x
                        for s_ in range(args.steps):
                            y = model(y, kc, vc, ctx + s_)
                    return y
                return run

            timing = time_interleaved({n: make_run(*b[:3]) for n, b in built.items()}, dev, warmup=2, repeats=7)
            for name, (model, kc, vc, kvh, naive, measured, measured_cuda) in built.items():
                t = timing[name]
                r = {"variant": name, "kv_heads": kvh, "q_heads": H, "naive_repeat": naive, "context": ctx,
                     "kv_bytes_per_token_formula": kv_bytes_formula(L, kvh, D_MODEL // H, 1, 1, elem),
                     "kv_bytes_per_token_measured": measured_cuda / ((ctx + args.steps) * args.batch),
                     "cache_bytes_measured": measured_cuda, "cache_bytes_tensors": measured,
                     "ms_per_decode_step": 1000 * t["median_s"] / args.steps,
                     "ms_per_step_min": 1000 * t["min_s"] / args.steps,
                     "ms_per_step_max": 1000 * t["max_s"] / args.steps, "interleaved": True,
                     **run_metadata("random-weight attention stack (12L, 768d, 12 q-heads)",
                                    str(dtype).replace("torch.", ""), args.batch, ctx, dev.type)}
                rows.append(r)
                print(f"ctx={ctx:5d} {name:20s} KV/token={r['kv_bytes_per_token_measured']:8.0f} B  "
                      f"cache={measured_cuda / 2**20:8.1f} MiB  {r['ms_per_decode_step']:.3f} ms/step "
                      f"(min {r['ms_per_step_min']:.2f}, max {r['ms_per_step_max']:.2f})")
            del built
            model = kc = vc = None          # drop loop variables too, so the cache memory is really freed
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT, index=False)
    plot()


def plot():
    import matplotlib.pyplot as plt
    from src import plotstyle
    plotstyle.apply()
    df = pd.read_csv(OUT)
    order = [v[0] for v in VARIANTS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    # per-token cost from the largest context (allocator rounding is negligible there)
    big = df[df.context == df.context.max()]
    per_tok = big.drop_duplicates("variant").set_index("variant").reindex(order).dropna(subset=["kv_heads"])
    per_tok = per_tok[~per_tok.naive_repeat.astype(bool)]
    bars = axes[0].bar(per_tok.index, per_tok.kv_bytes_per_token_measured / 1024,
                       color=[plotstyle.color_for(v, order) for v in per_tok.index], edgecolor=plotstyle.SURFACE)
    for b, v in zip(bars, per_tok.kv_bytes_per_token_measured / 1024):
        axes[0].annotate(f"{v:.1f} KiB", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                         textcoords="offset points", ha="center", fontsize=8, color=plotstyle.TEXT2)
    axes[0].set(title="KV cache per token (measured)", ylabel="KiB / token")
    axes[0].grid(axis="x", visible=False)
    for v in order:
        d = df[df.variant == v].sort_values("context")
        if len(d):
            axes[1].plot(d.context, d.ms_per_decode_step, marker="o", color=plotstyle.color_for(v, order),
                         ls="--" if d.naive_repeat.iloc[0] else "-", label=v)
    axes[1].set_xscale("log", base=2)
    axes[1].set(title=f"Decode step time, batch {int(df.batch.iloc[0])}", xlabel="context length (tokens)",
                ylabel="ms per step (median)")
    axes[1].legend()
    for ax in axes:
        t = ax.get_title()
        ax.set_title("")
        ax.set_title(t, loc="left")
    fig.suptitle(f"MHA vs GQA vs MQA - random weights, speed/memory only, {df.precision.iloc[0]}, {df.gpu.iloc[0]}",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.TEXT2)
    fig.tight_layout()
    fig.savefig(PLOTS / "07_mha_gqa_mqa.png", dpi=150)
    print("saved plots/07_mha_gqa_mqa.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot()
    else:
        main()
