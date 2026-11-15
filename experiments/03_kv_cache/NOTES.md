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
`results/03_kv_equivalence.csv` — GPU, FP32, 200 greedy tokens after a 9-token prompt.

| Method | identical to uncached | first divergence |
|---|---|---|
| uncached (ours) | 100 % | — |
| cached, torch.cat (ours) | **100 %** | — |
| cached, preallocated (ours) | **100 %** | — |
| Hugging Face `generate(do_sample=False)` | **100 %** | — |
| BUGGY: new tokens get position 0 | **4 %** | token index **1** |

Correct text: "the arrival of the first Europeans in the region in the 16th century. The first Europeans…"
Buggy text: "the the,,,,,,,,,,,,,,,,,,,…"
Tests also pass on CPU and GPU: cached logits = full re-run logits at every one of 20 decode steps (< 1e-3).

### Prediction vs result
- All four correct versions 100 % identical: **correct.**
- Bug diverges within 1–3 tokens: **correct — at token index 1**, i.e. the *very first* decode step.
  (Token 0 comes from the prefill, which has correct positions; the first token produced through the
  cache is already wrong.)

## Why
- A cache doesn't change *what* is computed, only *how much* is recomputed: old tokens' K and V are
  the same numbers whether recomputed or saved. So the output is bit-for-bit the same here.
- The buggy model thinks every new word is "word number 0 of the text". Position embeddings are how
  GPT-2 knows word order, so it loses track of where it is and collapses into repeating commas.
  This is why "position id = current cache length" is the classic KV-cache bug — and why we test for it.
