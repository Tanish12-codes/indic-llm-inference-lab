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
Run 2026-11-15, 200 aligned FLORES+ `dev` sentences, `results/01_token_counts_summary.csv`.
Tokenizer vocab sizes: GPT-2 50,257 · Qwen2.5 151,665 · Param2 128,000.

| Language | Tokenizer | tokens/word | chars/token | tokens/sentence | × English | round-trip OK |
|---|---|---|---|---|---|---|
| English | GPT-2   | 1.25 | 4.79 | 27.1 | 1.00 | 100 % |
| Hindi   | GPT-2   | 7.70 | 0.66 | 196.3 | **7.25** | 100 % |
| Marathi | GPT-2   | 11.23 | 0.63 | 209.1 | **7.72** | 100 % |
| English | Qwen2.5 | 1.30 | 4.61 | 28.1 | 1.00 | 100 % |
| Hindi   | Qwen2.5 | 4.71 | 1.08 | 120.0 | **4.27** | **86.5 %** |
| Marathi | Qwen2.5 | 6.77 | 1.04 | 126.0 | **4.48** | 100 % |
| English | Param2  | 1.55 | 3.85 | 33.7 | 1.00 | 100 % |
| Hindi   | Param2  | 1.51 | 3.36 | 38.6 | **1.15** | 100 % |
| Marathi | Param2  | 2.04 | 3.46 | 38.0 | **1.13** | 100 % |

Per-sentence spread (Hindi tokens ÷ English tokens for the same sentence, median [10th–90th pct]):
GPT-2 7.10 [5.76–9.14] · Qwen2.5 4.25 [3.33–5.32] · Param2 1.12 [0.96–1.41]. Marathi is similar.

One Hindi sentence (54 words, 299 characters, 785 UTF-8 bytes): **GPT-2 475 tokens, Qwen2.5 279,
Param2 71.** In GPT-2's split almost every piece is `\ufffd` — a token holding *part* of one
character's 3 bytes. Param2 keeps whole words such as "यूनिवर्सिटी" (university) as one token.

### Prediction vs result
- GPT-2 Hindi/Marathi ≈ 4–8 tokens/word → **7.7 / 11.2**: Hindi in range, Marathi worse. Predicted 4–6× English → **7.3–7.7×**: worse than predicted.
- Qwen2.5 Hindi ≈ 2–4 tokens/word → **4.7**: worse than predicted (but ~1.7× better than GPT-2).
- Param2 Hindi ≈ 1.3–2 → **1.5**: correct. Same meaning costs only ~13–15 % more tokens than English.
- Marathi > Hindi *per word*: correct for all three. But *per sentence* Marathi ≈ Hindi
  (Param2: 1.13× vs 1.15×). Marathi just has fewer, longer words — so fertility exaggerates the gap.
- Round-trip 100 % for byte-level tokenizers: **wrong for Qwen2.5 Hindi (27 of 200 fail).**

## Why
- **Bytes vs merges.** Every Devanagari character is 3 UTF-8 bytes. GPT-2 learned its merges from
  English web text, so it rarely merges Devanagari bytes: ~1.65 bytes per token → **less than one
  character per token** (0.66). The model has to read a Hindi word as a pile of byte fragments.
- **Vocabulary budget.** Qwen2.5 has 3× the vocabulary, but most new entries are Chinese/code/
  European; Devanagari gets a few hundred pieces (mostly single characters and matras), hence ~1 char/token.
  Param2 spends its budget on Indian languages: whole Hindi words become single tokens.
- **Surprise 1 — Param2 is *worse* at English** (33.7 vs 27.1 tokens/sentence, 1.55 tokens/word).
  Vocabulary is a fixed budget: slots given to Devanagari words are slots not given to English words.
  Every tokenizer trades one language against another.
- **Surprise 2 — Qwen2.5 Hindi round-trip fails 13.5 % (27/200). CONFIRMED cause: Unicode NFC.**
  `check_roundtrip.py` shows Qwen's tokenizer has an `NFC()` normaliser, and **27 of 27** failures
  equal `NFC(original)` exactly (0 need NFKC, 0 unexplained). The culprits are the *precomposed nukta
  letters* U+0958–U+095F (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़). Unicode marks them "composition-excluded", so NFC
  **splits** each one into base letter + nukta sign (e.g. U+095E फ़ → U+092B फ + U+093C ़). The text
  looks identical on screen but is 1 code point longer per such letter (id 31: 118 → 119 characters).
  No information is lost for a reader, but byte-exact equality — and string-matching code — breaks.
  GPT-2 and Param2 don't normalise, so they round-trip 100 %.
  Interview angle: "decode(encode(x)) == x" is a property of the *whole pipeline*, including normalisation —
  not guaranteed just because the BPE is byte-level.
- **Fairest metric = tokens for the same meaning** (the "× English" column), not tokens/word, because
  "word" means different amounts of meaning in different languages.

## What this means for inference cost (bridge to the next steps)
Every token costs one KV-cache slot and one decode step. Under GPT-2's tokenizer, a Hindi reply of the
same length costs **~7× the memory and ~7× the decode steps** of the English one; under Param2, ~1.15×.
Step 9 turns this into megabytes.
