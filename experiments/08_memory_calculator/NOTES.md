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
`results/08_memory_calculator.csv` (second run, after fixing the weight-memory bug below).

| | GPT-2 | Qwen2.5-0.5B | Param2-17B-A2.4B (config only) |
|---|---|---|---|
| layers / hidden / q-heads / KV-heads / head_dim | 12 / 768 / 12 / 12 / 64 | 24 / 896 / 14 / 2 / 64 | 21 / 2048 / 32 / 8 / 64 |
| params: calculator | 124.4 M | 494.0 M | 17,151.1 M total, 2,554.9 M active/token |
| params: **measured** (loaded model) | **124.4 M** | **494.0 M** | — (never downloaded) |
| params: hub safetensors metadata | 137.0 M | 494.0 M | 17,151.1 M |
| weights | 474.7 MiB FP32 (measured; CUDA delta 475.5) | 942.3 MiB FP16 (measured; CUDA delta 950.2) | 31.95 GiB BF16 · 15.97 GiB int8 · 7.99 GiB int4 |
| KV per token (FP16) | 36 KiB | **12 KiB** | 42 KiB |
| KV per token measured vs calculator | 73,728 = 73,728 B (FP32) | 12,288 = 12,288 B (FP16) | — |

MoE details read from Param2's config: 64 experts, top-6 per token, 20 of 21 layers are MoE,
shared-expert width 4096. Our estimate is within 3,968 parameters (0.00002 %) of the hub's count.

### Prediction vs result
- GPT-2 124.4 M / 475 MiB FP32 / 72 KiB per token FP32: **exact.**
- Qwen2.5-0.5B ~494 M / ~942 MiB FP16 / 12 KiB per token — smaller than GPT-2's despite 4× params: **exact.**
- Param2 ≈ 32 GiB BF16, doesn't fit 6 GB even at int4 (≈ 8 GiB): **yes (31.95 / 7.99 GiB).** KV/token: 42 KiB.

### Bug found and fixed
The first run printed GPT-2 weights as 0.51 GiB because the calculator used the hub's count (137.0 M).
The extra **12.58 M ≈ 12 × 1024 × 1024** are GPT-2's stored causal-mask buffers (`attn.bias`, one per layer)
— saved in the checkpoint, but not learned weights. The calculator now always uses its own parameter
count and records the difference (`published_minus_estimate`).

## Why
- **Weights:** parameters × bytes. Embeddings are a big share of small models: Qwen2.5-0.5B's
  151,936 × 896 embedding table is 136 M of its 494 M parameters (tied with the output layer).
- **KV per token** depends only on layers × KV heads × head_dim — not on total size. GQA is why the
  0.5 B Qwen has a *third* of GPT-2's KV cost per token.
- **MoE:** all 64 experts must sit in memory (weight memory follows the 17.2 B total), but each token runs
  only 6 of them plus the shared expert (compute follows ~2.55 B active). So Param2 is cheap to *compute*
  for its size but expensive to *hold*. (The model name says A2.4B; our 2.55 B count includes the
  embedding/output table — conventions differ on whether to count it.)
- **CUDA delta vs tensor bytes:** for GPT-2 the KV delta (89,088 B/token) exceeds the tensors (73,728) — the
  caching allocator's block rounding again; for Qwen they're equal.
