# Experiment 09 — KV memory of the same text, per language × tokenizer

## Prediction (written BEFORE running)
- View B (same KV/token for everyone, isolates the tokenizer): KV cost ∝ token count, so Hindi and
  Marathi under GPT-2's tokenizer cost **~4–6× English**; under Qwen **~2–3×**; under Param2 **~1–1.5×**.
- View A (each tokenizer with its own model): Qwen2.5-0.5B's GQA (12 KiB/token) partly cancels its
  tokenizer penalty vs GPT-2 (36 KiB/token FP16). Lesson: *two* levers — fewer tokens (better
  tokenizer) and fewer bytes per token (GQA/MQA, quantized KV).

## Result
_(filled in after running)_

## Why
_(filled in after running)_
