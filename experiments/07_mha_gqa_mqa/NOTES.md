# Experiment 07 — MHA vs GQA vs MQA (random weights; speed and memory only)

## Setup
12 attention-only layers, d_model 768, 12 query heads × 64 dims, FP16, batch 8, preallocated cache.
KV heads 12 / 4 / 1, plus a "naive GQA" that copies K/V with repeat_interleave.
No quality claims — weights are random.

## Prediction (written BEFORE running)
- **KV per token** (FP16): MHA 2×12×12×64×2 = **36 KiB**, GQA-4 **12 KiB**, MQA **3 KiB** — measured
  should equal the formula exactly.
- **Decode speed**: at short context (256) all similar — launch overhead dominates. At 8192 × batch 8
  MHA must stream ~2.4 GB of cache per step (memory-bound, ~15 ms on a ~170 GB/s card); GQA ≈ 1/3 of
  that, MQA ≈ 1/12 → clear gaps.
- **Naive GQA**: copying K/V back to 12 heads every step destroys the benefit → about MHA speed or worse.

## Result
`results/07_mha_gqa_mqa.csv` — second run, **interleaved** (variants take turns within each context,
7 repeats, 3 s GPU warm-up), FP16, batch 8, 12 layers, 20 decode steps per timed run. Random weights.

**KV cache per token** (exact tensor bytes): MHA **36,864 B (36 KiB)** · GQA-4 **12,288 B (12 KiB)** · MQA **3,072 B (3 KiB)** —
identical to the formula. (CUDA allocator deltas are 0.1–4 % higher at short contexts because of block rounding.)

**Decode step time (ms, median):**

| context | MHA (12 KV) | GQA (4 KV) | MQA (1 KV) | GQA naive repeat | cache size MHA / GQA / MQA |
|---|---|---|---|---|---|
| 256  | 7.4 | 7.4 | 7.6 | 8.0 | 78 / 27 / 7 MiB |
| 1024 | 7.2 | 7.3 | 7.7 | 8.9 | 295 / 100 / 25 MiB |
| 4096 | 9.0 | 7.2 | 8.0 | **21.1** | 1,159 / 388 / 97 MiB |
| 8192 | **17.0** | **9.7** | **7.7** | **48.4** | 2,311 / 772 / 193 MiB |

The first, non-interleaved run had a non-monotonic curve (MHA 89 ms at 4096 but 56 ms at 8192, min–max
38–108 ms) — laptop clock changes during sequential timing. The interleaved run is smooth and its
min–max ranges are tight (e.g. MHA 8192: 17.04–17.14 ms).

### Prediction vs result
- KV/token 36 / 12 / 3 KiB, exact: **yes.**
- All similar at short context (launch overhead): **yes — ~7.5 ms for every variant up to 1024 tokens.**
- Clear gaps at 8192 (MHA ~15 ms, GQA ≈ 1/3, MQA ≈ 1/12): **MHA 17.0 ms ✔; GQA 9.7 and MQA 7.7** — the gaps are
  real but smaller than 3× and 12×, because both hit the ~7.5 ms fixed floor.
- Naive GQA ≈ MHA or worse: **much worse — 2.8× slower than MHA at 8192.**

## Why
- **Memory per token ∝ number of KV heads.** 12 → 4 → 1 heads gives exactly 3× and 12× less cache.
  This is the main reason modern models use GQA: Qwen2.5-0.5B (2 KV heads) stores 12 KiB/token
  vs GPT-2's 36 KiB despite being 4× bigger (experiment 08).
- **Speed follows bytes read, once the cache is big.** A decode step must read the whole cache: at
  8192 tokens × batch 8, MHA reads 2.3 GB per step, GQA 0.77 GB, MQA 0.19 GB. Above a fixed floor of
  ~7.5 ms (kernel launches for 12 layers × 20 steps' worth of small ops), time grows with those bytes —
  decode is **memory-bound**. MHA: +9.6 ms above the floor for 2.3 GB; GQA +2.2 ms for 0.77 GB.
  Below ~1,000 tokens the cache is too small to matter and everything sits on the floor.
- **Naive GQA destroys the benefit.** `repeat_interleave` copies each layer's K and V back to 12 heads
  every step — it *reads* the small cache and *writes and re-reads* an MHA-sized copy, so it moves more
  bytes than MHA itself. The memory saving at rest remains, but the speed is worst of all. Our fast path
  treats a group's query heads as rows of one matrix multiplying the group's single K — no copy
  (verified equal to the naive version in `tests/test_attention_gqa.py`).
- **No quality claims:** with random weights we measure cost only. Real GQA/MQA models lose a little
  quality vs MHA, which is why GQA (in between) is the popular compromise.
