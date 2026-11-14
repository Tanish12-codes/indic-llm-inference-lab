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
_(filled in after running)_

## Why
_(filled in after running)_
