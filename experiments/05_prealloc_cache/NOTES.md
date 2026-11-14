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
_(filled in after running — from `results/05_prealloc_vs_concat.csv`)_

## Why
_(filled in after running)_
