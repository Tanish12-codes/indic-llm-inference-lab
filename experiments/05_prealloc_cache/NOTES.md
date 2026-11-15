# Experiment 05 — Optimization 1: preallocated KV cache

## Prediction (written BEFORE running)
- Tokens identical to the concat version (it's the same maths, different storage).
- **Speed**: small gain. `torch.cat` copies the whole layer cache each step: at 500 tokens that's
  ~37 MiB per step in total ≈ 0.2–0.4 ms of copying vs a ~5 ms step → **≤ 5–10 %** faster, growing
  with length; near 0 % at 50 tokens (kernel-launch overhead dominates).
- **Memory**: prealloc reserves the full max length up front (here = prompt + n, so same final size);
  concat's peak is a bit higher because during `cat` the old and new tensors exist at the same time.
- Lesson: preallocation matters more for *memory predictability* (no fragmentation, known budget —
  the idea behind vLLM's paged KV cache) than raw speed at this tiny scale.

## Result
`results/05_prealloc_vs_concat.csv` — second run, **interleaved timing** (concat and prealloc take turns,
7 repeats each, 3 s GPU warm-up). GPT-2 small, FP32, batch 1, RTX 3050 Laptop. Tokens identical (300-token check): ✔

| new tokens | concat median (min) | prealloc median (min) | difference | peak extra mem concat / prealloc |
|---|---|---|---|---|
| 50  | 469 ms (437) | 463 ms (426) | −1 % | 13.7 / 15.8 MiB |
| 100 | 947 ms (914) | 960 ms (900) | +1 % | 16.6 / 20.0 MiB |
| 200 | 1.84 s (1.76) | 1.90 s (1.81) | +3 % | 23.7 / 26.9 MiB |
| 500 | 4.74 s (4.65) | 4.98 s (4.69) | +5 % | **53.6 / 48.0 MiB** |
| 900 | 10.1 s (8.15) | 11.8 s (8.33) | +17 % median, **+2 % min** | 74.2 / 75.5 MiB |

Both n=900 rows have very wide ranges; one prealloc repeat recorded an absurd max (1.29 × 10⁶ s) and in
the first run one concat repeat recorded 4,735 s — the laptop was suspended / the console frozen during a
timed run. The median and min ignore such outliers; the plot clips the axis and says so.

### Prediction vs result
- Same tokens: **yes.**
- Speed gain ≤ 5–10 %, ~0 % at 50 tokens: **no measurable gain at any length.** Differences are −1 % … +5 %
  (within run-to-run noise); at 900 the medians differ by 17 % but the best runs (min) differ by only 2 %.
- Concat's peak memory a bit higher: **only at 500 tokens (−10 % with prealloc)**; at short lengths prealloc is
  2–3 MiB *higher*, at 900 about equal.

## Why
- **The copy that preallocation saves is small next to the fixed per-step cost.** At 900 tokens, `torch.cat`
  copies about 2 × 12 × 915 × 768 × 4 B ≈ 67 MB per step. At the laptop GPU's memory bandwidth that's
  well under a millisecond, while each step costs ~10 ms of mostly fixed overhead (experiment 04). So
  the best possible saving is a few percent — below what this laptop can measure reliably.
- **Prealloc also adds its own small costs:** every step it slices views of a big buffer and writes in place;
  the attention matmul then reads a non-contiguous slice. Net effect on GPT-2 small: a wash.
- **Run-to-run drift is real and large on a laptop** (same work measured 10.4 → 13.9 ms/token across runs).
  That's why the second run interleaves the two methods: drift then hits both equally.
- **Where preallocation matters in practice:** memory *predictability*, not speed at this size. A server
  that knows each request's maximum size up front never runs out of memory halfway and never fragments
  memory with thousands of growing tensors. vLLM's *paged* KV cache takes this further: fixed-size blocks
  handed out on demand. Our measured allocator slack in experiment 04 (44 MiB for 36 MiB of tensors) is
  the problem it solves.
- Honest conclusion: **"preallocation made no measurable speed difference for GPT-2 small on this laptop."**
