"""Step 6 - Optimization 2: 8-bit KV cache (int8 + one scale per vector).

    .venv\\Scripts\\python.exe experiments\\06_int8_kv_cache\\run.py
Outputs:
    results/06_int8_agreement.csv   - % generated tokens identical to the FP32 cache, per prompt
    results/06_int8_memory_speed.csv - cache bytes and speed, int8 vs FP32 prealloc
    plots/06_int8_kv_cache.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.bench import bench_generation  # noqa: E402
from src.common import PLOTS, RESULTS, get_device, run_metadata, set_seed  # noqa: E402

PROMPTS = [
    "The history of the Indian subcontinent begins with the first humans who arrived",
    "In a small village near the river, an old teacher",
    "The recipe for a good cup of tea requires",
    "Scientists announced on Tuesday that the new telescope had",
    "मुंबई भारत का सबसे बड़ा शहर है और",            # Hindi bytes - GPT-2 can't understand, we only measure
]
N_NEW = 200
A_CSV = RESULTS / "06_int8_agreement.csv"
M_CSV = RESULTS / "06_int8_memory_speed.csv"


def main():
    from transformers import AutoTokenizer
    from src.gpt2_scratch import GPT2, generate_cached, make_cache
    from src.kv_cache import quantize_int8
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(0)
    dev = get_device()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2.from_pretrained("gpt2").to(dev)

    # 1) agreement with the FP32 cache
    rows = []
    for p in PROMPTS:
        ids = torch.tensor([tok.encode(p)], device=dev)
        P = ids.shape[1]
        ref = generate_cached(model, ids, N_NEW, make_cache("prealloc", model, max_len=P + N_NEW))[0, P:]
        cache8 = make_cache("int8", model, max_len=P + N_NEW)
        q8 = generate_cached(model, ids, N_NEW, cache8)[0, P:]
        same = (ref == q8)
        first = int((~same).nonzero()[0].item()) if (~same).any() else None
        # reconstruction error of the stored keys/values for layer 0
        cfg = model.cfg
        qkv = model.h[0].attn.c_attn(model.h[0].ln_1(model.wte(ids) + model.wpe(torch.arange(P, device=dev))[None]))
        kv = torch.cat(qkv.split(cfg.n_embd, dim=2)[1:], dim=2).view(1, P, 2 * cfg.n_head, cfg.head_dim)
        qv, sc = quantize_int8(kv)                     # one scale per (token, head) vector, as in the cache
        rel_err = ((qv.float() * sc - kv).norm() / kv.norm()).item()
        rows.append({"prompt": p, "prompt_tokens": P, "new_tokens": N_NEW,
                     "identical_positionwise_pct": 100 * same.float().mean().item(),
                     "identical_prefix_tokens": first if first is not None else N_NEW,
                     "first_divergence_index": first,
                     "layer0_kv_relative_quant_error": rel_err,
                     **run_metadata("gpt2 (124M, from scratch)", "fp32 compute, int8 KV", 1, P + N_NEW, dev.type)})
        print(f"{rows[-1]['identical_positionwise_pct']:6.1f}% identical, first divergence at "
              f"{first}  | {p[:50]}")
    pd.DataFrame(rows).to_csv(A_CSV, index=False, encoding="utf-8")

    # 2) memory + speed
    ids = torch.tensor([tok.encode(PROMPTS[0])], device=dev)
    mrows = []
    for n in [100, 200, 500, 900]:
        for method in ["prealloc", "int8"]:
            r = bench_generation(model, ids, method, n, dev)
            r.update(run_metadata("gpt2 (124M, from scratch)",
                                  "fp32" if method == "prealloc" else "fp32 compute, int8 KV + fp16 scales",
                                  1, ids.shape[1] + n, dev.type))
            mrows.append(r)
            print(f"{method:9s} n={n:4d}  kv used {r['kv_used_bytes'] / 2**20:7.2f} MiB  "
                  f"{r['ms_per_token']:.3f} ms/token  peak+{r['peak_over_baseline_B'] / 2**20:.1f} MiB")
    m = pd.DataFrame(mrows)
    fp = m[m.method == "prealloc"].set_index("new_tokens").kv_used_bytes
    m["kv_used_bytes_fp16_equivalent"] = m.new_tokens.map(fp) / 2
    m["saving_vs_fp32_pct"] = 100 * (1 - m.kv_used_bytes / m.new_tokens.map(fp))
    m["saving_vs_fp16_pct"] = 100 * (1 - m.kv_used_bytes / m.kv_used_bytes_fp16_equivalent)
    m.to_csv(M_CSV, index=False)
    print(m[m.method == "int8"][["new_tokens", "kv_used_bytes", "saving_vs_fp32_pct", "saving_vs_fp16_pct"]])
    plot()


def plot():
    import matplotlib.pyplot as plt
    from src import plotstyle
    plotstyle.apply()
    a = pd.read_csv(A_CSV)
    m = pd.read_csv(M_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    order = ["uncached", "concat", "prealloc", "int8"]
    for meth, lab in [("prealloc", "FP32 cache"), ("int8", "int8 cache")]:
        d = m[m.method == meth].sort_values("new_tokens")
        c = plotstyle.color_for(meth, order)
        axes[0].plot(d.new_tokens, d.kv_used_bytes / 2**20, marker="o", color=c, label=lab)
        axes[1].plot(d.new_tokens, d.ms_per_token, marker="o", color=c, label=lab)
    d = m[m.method == "prealloc"].sort_values("new_tokens")
    axes[0].plot(d.new_tokens, d.kv_used_bytes_fp16_equivalent / 2**20, ls="--", color=plotstyle.TEXT2,
                 lw=1.2, label="FP16 cache (computed: FP32/2)")
    axes[0].set(title="KV cache size", xlabel="new tokens", ylabel="MiB")
    axes[1].set(title="Decode speed", xlabel="new tokens", ylabel="ms / token")
    labels = [p[:22] + "..." for p in a.prompt]
    bars = axes[2].barh(labels, a.identical_positionwise_pct, color=plotstyle.color_for("int8", order),
                        edgecolor=plotstyle.SURFACE)
    for b, v, f in zip(bars, a.identical_positionwise_pct, a.first_divergence_index):
        txt = f"{v:.0f}%" + ("" if pd.isna(f) else f" (diverges at {int(f)})")
        axes[2].annotate(txt, (v, b.get_y() + b.get_height() / 2), xytext=(3, 0), textcoords="offset points",
                         va="center", fontsize=8, color=plotstyle.TEXT2)
    axes[2].set(title=f"Same tokens as FP32? ({int(a.new_tokens.iloc[0])} generated)",
                xlabel="% identical (position-wise)", xlim=(0, 125))
    axes[2].grid(axis="x", visible=True)
    axes[2].grid(axis="y", visible=False)
    for ax in axes[:2]:
        ax.legend()
    for ax in axes:
        t = ax.get_title()
        ax.set_title("")
        ax.set_title(t, loc="left")
    fig.suptitle(f"int8 KV cache - GPT-2 small (small-model result), {m.gpu.iloc[0]}",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.TEXT2)
    fig.tight_layout()
    fig.savefig(PLOTS / "06_int8_kv_cache.png", dpi=150)
    print("saved plots/06_int8_kv_cache.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot()
    else:
        with torch.no_grad():
            main()
