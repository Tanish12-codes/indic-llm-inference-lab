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
_(filled in after running)_

## Why
_(filled in after running)_
