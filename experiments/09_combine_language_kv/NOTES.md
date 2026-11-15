# Experiment 09 — KV memory of the same text, per language × tokenizer

## Prediction (written BEFORE running)
- View B (same KV/token for everyone, isolates the tokenizer): KV cost ∝ token count, so Hindi and
  Marathi under GPT-2's tokenizer cost **~4–6× English**; under Qwen **~2–3×**; under Param2 **~1–1.5×**.
- View A (each tokenizer with its own model): Qwen2.5-0.5B's GQA (12 KiB/token) partly cancels its
  tokenizer penalty vs GPT-2 (36 KiB/token FP16). Lesson: *two* levers — fewer tokens (better
  tokenizer) and fewer bytes per token (GQA/MQA, quantized KV).

## Result
`results/09_language_kv_cost.csv` — KV-cache memory (FP16) for 1,000 FLORES-like sentences of the same
meaning = mean tokens per sentence (step 1) × KV bytes per token (step 8). Arithmetic only.

| | GPT-2 (36 KiB/token) | Qwen2.5-0.5B (12 KiB/token) | Param2 (42 KiB/token) |
|---|---|---|---|
| **A) own model** — English | 952 MiB | 330 MiB | 1,383 MiB |
| Hindi | **6,900 MiB** (7.25×) | 1,406 MiB (4.27×) | 1,583 MiB (1.15×) |
| Marathi | 7,350 MiB (7.72×) | 1,476 MiB (4.48×) | 1,558 MiB (1.13×) |
| **B) same KV/token for all (Qwen's 12 KiB)** — English | 317 MiB | 330 MiB | 395 MiB |
| Hindi | 2,300 MiB | 1,406 MiB | **452 MiB** |
| Marathi | 2,450 MiB | 1,476 MiB | 445 MiB |

### Prediction vs result
- View B, Hindi/Marathi vs English: GPT-2 ~4–6× → **7.3–7.7×** (worse); Qwen ~2–3× → **4.3–4.5×** (worse);
  Param2 ~1–1.5× → **1.13–1.15×** ✔.
- View A, Qwen's GQA partly cancels its tokenizer penalty: **yes — Hindi under Qwen (1,406 MiB) needs less KV
  than under Param2 (1,583 MiB) despite 3.1× more tokens**, because Qwen stores 3.5× fewer bytes per token.

## Why
- **Two independent levers multiply:** KV memory = tokens × bytes-per-token.
  Lever 1, the tokenizer: GPT-2 → Param2 cuts Hindi tokens 5.1×. Lever 2, the attention layout: GQA with
  few KV heads (Qwen: 2) cuts bytes per token. Int8 KV (experiment 06) would cut another ~2× from FP16.
- **The same Hindi document costs 7× the memory of its English version under GPT-2's tokenizer.** On a 6 GB
  card that's the difference between serving a long Hindi document or not.
- **Caveat — A is not a fair model comparison:** Qwen2.5-0.5B and Param2-17B are very different models in
  size and ability; View A shows the cost of running *each* on its own tokens, not which is better.
  View B isolates the tokenizer effect cleanly.
