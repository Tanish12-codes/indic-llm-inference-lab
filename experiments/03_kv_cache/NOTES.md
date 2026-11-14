# Experiment 03 — KV cache correctness

## Question
With greedy decoding, do uncached, cached (concat), cached (preallocated) and HF `generate` produce
exactly the same 200 tokens? And what does the classic position-id bug do?

## Prediction (written BEFORE running)
- All four: **100 % identical**. Caching changes *how much work* we do, not *what* is computed.
  (Tiny float differences from different matrix shapes could, in theory, flip a near-tie argmax;
  with GPT-2 greedy that's unlikely within 200 tokens.)
- Buggy version (every new token gets position 0): diverges **within the first 1–3 tokens** and
  turns repetitive/garbled, because the model thinks every new word is the first word of the text.

## Result
_(filled in after running — from `results/03_kv_equivalence.csv`)_

## Why
_(filled in after running)_
