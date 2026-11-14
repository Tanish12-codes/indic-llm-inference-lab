# Experiment 02 — GPT-2 forward pass from scratch

## Question
Does our ~150-line GPT-2 produce the same logits as Hugging Face's when given the same weights?

## Prediction (written BEFORE running)
- FP32 max |ours − HF| around **1e-5** on CPU and **1e-5 – 5e-5** on GPU (different kernel order of
  additions → last-digit rounding differences, not bugs). Below the 1e-4 bar.
- argmax (the predicted next token) identical at every position.
- If any of the three gotchas is missed: forgetting the transpose → shape assert fails or garbage;
  exact GELU instead of tanh-GELU → diffs ~1e-3; untied output → we'd need a weight that doesn't exist.

## Result
_(filled in after running — from `results/02_gpt2_vs_hf.csv`)_

## Why
_(filled in after running)_
