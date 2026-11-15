# Indic LLM Inference Lab

**Question:** What does it cost — in memory and time — to run (infer) an LLM, and why does it cost
more for Indian languages?

**Short answer (to be confirmed by our numbers):** cost is paid *per token*. The same sentence in
Hindi or Marathi becomes several times more tokens under English-centric tokenizers, and every
token costs a slice of KV-cache memory and one decode step. Two levers fix it: better tokenizers
(fewer tokens) and leaner attention/caches (fewer bytes per token: GQA/MQA, int8 KV).

> Status: code complete, experiments being run step by step. Results sections are filled in
> from `results/*.csv` as each step passes. See `CONCEPTS.md` for every idea explained simply.

## Hardware
Windows 11 laptop · NVIDIA GeForce RTX 3050 6GB **Laptop** GPU (compute capability 8.6, 20 SMs,
~5 GiB free after the desktop) · Python 3.14.2 · torch 2.15.0+cu132 · transformers 5.23.0.
(`results/env_check.txt`)

## How to reproduce (Windows PowerShell, from this folder)
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1    # venv + torch(CUDA) + libs + env check
.\.venv\Scripts\hf auth login                                 # needed for gated FLORES+
.\run.ps1 1        # step 1: token counts (no GPU needed)
.\run.ps1 tests    # correctness gate: GPT-2 == HF, cached == uncached == HF generate
.\run.ps1 all      # tests again, then steps 2..9 (stops if anything fails)
```
Each run writes a UTF-8 log to `results/logs/`. Plots are always regenerated from CSVs
(`run.py --plot-only`).

## Layout
```
src/          gpt2_scratch.py  kv_cache.py  attention.py  memory_model.py  bench.py  common.py  plotstyle.py
experiments/  NN_name/run.py + NOTES.md (prediction written before running, then result + why)
tests/        test_gpt2_matches_hf.py  test_kv_cache_equivalence.py  test_attention_gqa.py
results/      *.csv (every row carries GPU, torch version, model, precision, batch, seq len, date)
plots/        *.png (made from the CSVs only)
```

## Results

### 1. Same meaning, different token cost
![tokens](plots/01_fertility_chars_per_token.png)

Tokens needed for the same 200 FLORES+ sentences, relative to English with the same tokenizer:

| Tokenizer | Hindi | Marathi |
|---|---|---|
| GPT-2 | 7.25× | 7.72× |
| Qwen2.5 | 4.27× | 4.48× |
| Param2 | 1.15× | 1.13× |

A tokenizer that never learned Devanagari spends **less than one character per token** (GPT-2: 0.66).
Param2 makes Hindi almost as cheap as English, but its English costs ~20 % more tokens than GPT-2's
(vocabulary is a fixed budget). Details: `experiments/01_token_counts/NOTES.md`.

## Limitations
_(written at the end, honestly)_
