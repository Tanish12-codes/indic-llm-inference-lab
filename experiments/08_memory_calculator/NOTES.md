# Experiment 08 — Memory calculator vs measurement

## Prediction (written BEFORE running)
- **GPT-2**: 124.4 M params (output tied) → 475 MiB in FP32; KV = 72 KiB/token FP32
  (36 KiB FP16). Calculator and measurement should agree exactly for KV, within a rounding error
  for weights.
- **Qwen2.5-0.5B**: ~494 M params (tied embeddings — 136 M of them are the embedding table!)
  → ~942 MiB in FP16. KV = 2 × 24 × **2 KV heads** × 64 × 2 B = **12 KiB/token** — *smaller than
  GPT-2's* despite 4× the parameters, thanks to GQA (14 query heads share 2 KV heads).
- **Param2-17B-A2.4B** (config only): MoE — weight memory follows the ~17 B *total* (≈ 32 GiB in
  BF16: does not fit a 6 GB card, even at int4 ≈ 8 GiB), compute follows ~2.4 B *active*.
  KV/token depends on its attention config — to be read from the published config.

## Result
_(filled in after running)_

## Why
_(filled in after running)_
