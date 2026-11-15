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
`results/04_speed_memory.csv` — RTX 3050 6GB Laptop GPU, FP32, batch 1, 15-token prompt, median of 5 (min–max).
**Small-model result (GPT-2 124M).**

| new tokens | uncached | cached (concat) | speed-up | uncached ms/token | cached ms/token | peak extra memory: uncached / cached |
|---|---|---|---|---|---|---|
| 50  | 434 ms (412–489) | 453 ms (445–508) | **0.96×** | 8.7 | 9.1 | 33 / 14 MiB |
| 100 | 967 ms | 963 ms | **1.00×** | 9.7 | 9.6 | 53 / 17 MiB |
| 200 | 2.77 s | 1.85 s | **1.49×** | 13.9 | 9.3 | 92 / 24 MiB |
| 500 | 14.9 s (14.5–15.4) | 6.19 s (6.08–6.77) | **2.40×** | 29.8 | 12.4 | **207 / 54 MiB** |

KV cache, measured (sum of cache tensor bytes) vs formula 2 × 12 layers × 12 heads × 64 × (prompt + n − 1) × 4 B:
**4.50 = 4.50 MiB, 8.02 = 8.02, 15.05 = 15.05, 36.14 = 36.14 MiB — exact at every length** (72 KiB/token in FP32).
CUDA allocator delta: within 2 KiB of that up to 200 tokens; 44.1 MiB vs 36.1 MiB at 500 tokens.

### Prediction vs result
- Cached ≈ constant ms/token: **yes** (9.1 → 12.4), but slower than the predicted 4–8 ms.
- Uncached ms/token grows with length: **yes** (8.7 → 29.8).
- Speed-up small at 50 tokens (predicted 1.1–1.5×): **even smaller — 0.96×, i.e. no gain.** At 500 (predicted 2–4×): **2.4×.** ✔
- KV memory = formula exactly, for prompt + n − 1 tokens: **yes.**
- Uncached peak memory higher than cached at long lengths: **yes, 3.9× higher at 500 tokens.**

## Why
- **Why the cache doesn't help at 50 tokens — kernel-launch overhead.** Uncached at 50 tokens costs
  8.7 ms per token, cached 9.1 ms: almost the same, even though uncached does up to 65× more maths
  per step. So the time isn't the maths — it's the **fixed cost of every step**: Python plus a few
  hundred small GPU kernel launches (12 layers × layer-norms, matmuls, mask building, softmax, adds).
  The GPU finishes each tiny kernel and then waits for the CPU to send the next one. With ~9 ms of fixed
  overhead per step, saving a few microseconds of maths changes nothing. Only when the uncached
  sequence gets long (hundreds of tokens) does its real work exceed that floor, and the cache wins.
  (Min–max ranges overlap at 50 tokens, so 0.96× is "no difference", not "cache is slower".)
- **Why 9 ms instead of 4–8 ms:** Windows (WDDM driver) launches kernels slower than Linux, the laptop
  GPU runs at a lower power limit, and our code builds the causal mask with fresh `arange` calls every step
  (extra small kernels). A fused or compiled implementation (`torch.compile`, CUDA graphs) would remove
  much of this — that's exactly what inference engines do.
- **Why cached time per token still creeps up (9 → 12 ms):** each new token attends over a longer cache,
  and `torch.cat` copies the whole cache every step (see experiment 05).
- **Why uncached uses 4× more peak memory with no cache at all:** every uncached step recomputes
  everything for all 515 positions, including **logits for every position**: 515 × 50,257 × 4 B ≈ 104 MB
  for one tensor, of which we use only the last row. Plus 12 attention-score matrices (12 heads × 515 × 515).
  The cached step computes 1 position. Lesson: a cache trades a small, predictable memory cost (36 MiB)
  for avoiding large temporary costs.
- **Allocator rounding:** at 500 tokens each layer's K or V tensor is 1.58 MB, created anew by every
  `torch.cat`. PyTorch's CUDA caching allocator hands out memory in rounded blocks, so 24 such tensors
  occupy ~44 MiB instead of 36 MiB. The *tensors* match the formula; the *allocator* adds ~20 % slack —
  one more reason real engines preallocate.
