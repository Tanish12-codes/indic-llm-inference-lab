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
`results/02_gpt2_vs_hf.csv` — FP32, TF32 off, HF loaded with eager attention.

| Prompt | tokens | CPU: max \|ours − HF\| | GPU: max \|ours − HF\| | HF vs ITSELF (CPU vs GPU) | next token same? |
|---|---|---|---|---|---|
| English ("quick brown fox…") | 10 | 6.1e-5 | 6.9e-5 | 1.7e-4 | yes, all positions |
| English ("In 1947…") | 7 | 7.6e-5 | 4.6e-5 | 1.2e-4 | yes |
| Hindi (GPT-2 bytes) | 78 | 6.1e-5 | 1.1e-4 | 2.3e-4 | yes |

Max probability difference after softmax: 1.1e-6 … 7.2e-6. Logits range about −155 … −22.
Greedy sample from our model: "The meaning of life is not the same as the meaning of death."

### Prediction vs result
- Predicted ~1e-5 (CPU) and 1e-5–5e-5 (GPU). Actual **4.6e-5 – 1.1e-4**: a few times larger than
  predicted, but **always smaller than the gap between HF's own model on CPU and on GPU**
  (1.2e-4 – 2.3e-4). Next-token choice identical everywhere. ✔
- My first test limit (absolute 1e-4) failed once (GPU, 78 tokens: 1.14e-4). My second attempt
  added a probability limit of 1e-6 *without measuring first* — it failed 6 tests. Both were
  mistakes in the **test**, not the model. Final test: |diff| < 1e-4 + 1e-6·max|logit|, probability
  diff < 1e-5, identical argmax, **and ours-vs-HF ≤ HF-vs-HF noise floor**.

## Why
- **Float rounding, not a bug.** Near |logit| = 100 a float32 can only move in steps of 7.6e-6
  (1 "ulp"). 1.14e-4 = exactly **15 ulps** = 0.0001 % relative error. GPU matrix-multiply kernels add
  768 products in a different order from the CPU (and from HF's `Conv1D`/`addmm` path), and
  (a+b)+c ≠ a+(b+c) in the last bits. Longer sequences → more additions in attention → slightly more drift.
- **The noise-floor comparison is the convincing argument**: Hugging Face disagrees *with itself*
  across devices by up to 2.3e-4. Our implementation is closer to HF than HF is to itself.
- A real bug is 10×–10,000,000× bigger: exact GELU instead of tanh-GELU ≈ 1e-3; forgetting the Conv1D
  transpose breaks shapes or gives garbage; an untied head would need a weight that doesn't exist.
- Lesson: **choose test tolerances from a measured noise floor, not from a round number.**
