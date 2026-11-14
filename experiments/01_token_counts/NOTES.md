# Experiment 01 — Token counts across languages and tokenizers

## Question
For the *same meaning* (FLORES+ gives professionally translated parallel sentences),
how many tokens does English, Hindi and Marathi cost under GPT-2, Qwen2.5 and Param2?

## Prediction (written BEFORE running)
- **GPT-2** (byte-level BPE trained almost only on English web text): English ≈ 1.3 tokens/word.
  Devanagari characters are 3 UTF-8 bytes each and GPT-2 learned few merges for them, so
  Hindi/Marathi should land near **4–8 tokens/word** and **< 1 character per token** — i.e. a
  Hindi sentence costs roughly **4–6× the English tokens**.
- **Qwen2.5** (151k vocab, multilingual but Chinese/English-heavy): English ≈ 1.3; Hindi maybe
  **2–4 tokens/word** — better than GPT-2, still well above English.
- **Param2** (BharatGen, built for Indian languages): Hindi/Marathi should be closest to English,
  maybe **1.3–2 tokens/word**.
- **Marathi > Hindi** for every tokenizer: Marathi glues suffixes onto words (agglutinative),
  so it has fewer, longer words → more tokens per word.
- Round-trip `decode(encode(text)) == text` should be 100 % for the byte-level tokenizers
  (GPT-2, Qwen). Param2 might normalise some characters — we'll see.

## Result
_(filled in after running — numbers come from `results/01_token_counts_summary.csv`)_

## Why
_(filled in after running)_
