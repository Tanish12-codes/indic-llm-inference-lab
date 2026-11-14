# Experiment 04 — Speed and memory: cached vs uncached

## Setup
GPT-2 small (124M), FP32, batch 1, ~15-token prompt, greedy, 50/100/200/500 new tokens.
2 warm-up runs, `torch.cuda.synchronize()` around timers, 5 repeats → median (min–max).
**Small-model result**: real LLMs are 10–1000× bigger; trends carry over, absolute numbers don't.

## Prediction (written BEFORE running)
- **Cached**: roughly constant ms/token (~4–8 ms on an RTX 3050), because each step processes one
  token. Two floors: reading all 475 MiB of FP32 weights per token (≈3 ms at ~170 GB/s) and
  ~200 kernel launches per token (≈1–2 ms).
- **Uncached**: ms/token grows with length (step *k* re-processes all *k* tokens → total work ∝ n²).
- **Speed-up**: small at 50 tokens (≈1.1–1.5×) because at short lengths both are dominated by the
  same fixed costs (weights + kernel launches); larger at 500 (≈2–4×).
- **KV memory**: formula 2 × 12 × 12 × 64 × 4 B = **72 KiB per token** in FP32. Measured cache bytes
  should match the formula *exactly* for (prompt + n − 1) tokens (the last token is never fed back).
- **Peak memory**: uncached may have a *higher* peak at 500 tokens than cached, because it builds
  a full attention-score matrix (12 heads × T × T) every step — no cache, but big temporaries.

## Result
_(filled in after running — from `results/04_speed_memory.csv`)_

## Why
_(filled in after running)_
