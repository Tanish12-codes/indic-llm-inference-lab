# Experiment 06 — Optimization 2: int8 KV cache

## Setup
Store K and V as int8 with one FP16 scale per (token, head) vector of 64 numbers.
Dequantize to FP32 right before attention. Compare to the FP32 preallocated cache.

## Prediction (written BEFORE running)
- **Memory**: per 64 numbers, 64 B + 2 B scale = 66 B vs 256 B (FP32) → **≈ 74 % smaller**;
  vs an FP16 cache (128 B) → **≈ 48 % smaller**.
- **Agreement**: relative quantization error ≈ 0.5–1 %. Greedy decoding is a chain: one flipped
  near-tie changes everything after it. Expect some prompts 100 % identical and others to diverge
  after tens of tokens → average maybe **50–100 %** position-wise identical.
- **Speed**: *slower*, not faster, in plain PyTorch — every step we quantize the new token and
  dequantize the WHOLE cache into a temporary FP32 copy. Real systems fuse dequantization into the
  attention kernel, so they win on memory bandwidth. Our version only demonstrates the memory saving.

## Result
`results/06_int8_agreement.csv`, `results/06_int8_memory_speed.csv` — RTX 3050 Laptop, GPT-2 small (small-model result).

**Agreement** (200 greedy tokens, int8 cache vs FP32 cache): **100 % identical for all 5 prompts**
(4 English + 1 Hindi-bytes). Relative error of the stored K/V after int8 round-trip: **0.57–0.60 %**.

**Memory and speed:**

| new tokens | FP32 cache | int8 cache | saving vs FP32 | saving vs FP16 (computed) | ms/token FP32 → int8 | peak extra FP32 → int8 |
|---|---|---|---|---|---|---|
| 100 | 8.02 MiB | 2.07 MiB | 74.2 % | 48.4 % | 16.3 → 23.9 | 19.6 → 14.5 MiB |
| 200 | 15.05 MiB | 3.88 MiB | 74.2 % | 48.4 % | 14.8 → 28.6 | 26.9 → 15.3 MiB |
| 500 | 36.14 MiB | 9.32 MiB | 74.2 % | 48.4 % | 13.9 → 35.2 | 47.4 → 21.9 MiB |
| 900 | 64.27 MiB | 16.57 MiB | 74.2 % | 48.4 % | 19.6 → 39.5 | 75.5 → 30.8 MiB |

### Prediction vs result
- Memory −74 % vs FP32, −48 % vs FP16: **exactly as predicted** (66 bytes per 64 numbers instead of 256 or 128).
- Agreement 50–100 %: **better than predicted — 100 % on every prompt.**
- Slower, not faster: **yes, 1.5–2.5× slower.**

## Why
- **Memory:** each vector of 64 numbers becomes 64 int8 bytes + one 2-byte FP16 scale = 66 B,
  vs 256 B (FP32) or 128 B (FP16). 1 − 66/256 = 74.2 %, 1 − 66/128 = 48.4 %. Pure arithmetic — no surprise.
- **Why no token changed:** a ~0.6 % error in K and V nudges attention weights only slightly, and
  GPT-2's greedy choice usually wins by a clear margin over the runner-up. A flip needs a near-tie.
  With 5 prompts × 200 tokens we saw none — that's evidence, not a guarantee: more prompts, longer outputs,
  or sampling instead of greedy would eventually show differences. Real systems report quality metrics
  (perplexity, task scores), not just token agreement.
- **Why it's slower here:** every step we quantize the new token *and* dequantize the whole layer cache
  into a temporary FP32 copy, all as separate small kernels. Production int8/FP8 KV caches do the
  dequantization **inside the attention kernel**, so they read 4× fewer bytes from memory and get faster
  in the memory-bound decode phase. Our version demonstrates the *memory* half of the trade only.
- **Peak memory still drops** (75.5 → 30.8 MiB at 900 tokens) because the temporary FP32 copy exists
  for one layer at a time, while the stored cache shrinks for all 12 layers.
- Note: FP32 ms/token here (13.9–19.6) is higher than in experiment 04/05 (10–12) for the same work —
  laptop-GPU clock drift between runs. Comparisons *within* this table were measured back-to-back.
