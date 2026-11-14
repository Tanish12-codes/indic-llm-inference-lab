"""Step 9 - Combine: KV-cache memory for the SAME sentence, per language x tokenizer.

    KV bytes = (tokens the tokenizer produced, from step 1) x (KV bytes per token, from step 8)

Two views:
  A) each tokenizer with ITS OWN model (GPT-2 / Qwen2.5-0.5B / Param2) - real end-to-end cost,
     but mixes two effects: tokenizer AND architecture (e.g. Qwen's GQA makes KV/token small).
  B) every tokenizer priced at ONE fixed KV/token (Qwen2.5-0.5B, fp16) - isolates the tokenizer.
Arithmetic only; no model is run here. Inputs: results/01_token_counts_summary.csv,
results/08_memory_calculator.csv.  Outputs: results/09_language_kv_cost.csv, plots/09_language_kv_cost.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.common import PLOTS, RESULTS, run_metadata  # noqa: E402

TOK2MODEL = {"GPT-2": "gpt2", "Qwen2.5": "Qwen/Qwen2.5-0.5B",
             "Param2": "bharatgenai/Param2-17B-A2.4B-Thinking"}
FIXED = "Qwen/Qwen2.5-0.5B"
DOC_SENTENCES = 1000      # "a 1,000-sentence document" - roughly a long report


def main():
    tok = pd.read_csv(RESULTS / "01_token_counts_summary.csv")
    mem = pd.read_csv(RESULTS / "08_memory_calculator.csv").set_index("name")
    kv16 = mem["kv_per_token_fp16/bf16_bytes"]
    rows = []
    for _, r in tok.iterrows():
        repo = TOK2MODEL[r.tokenizer]
        own = kv16.get(repo, float("nan"))
        rows.append({
            "language": r.language, "tokenizer": r.tokenizer, "paired_model": repo,
            "mean_tokens_per_sentence": r.mean_tokens_per_sentence,
            "tokens_vs_english": r.tokens_vs_english,
            "kv_per_token_own_model_fp16_B": own,
            "kv_per_sentence_own_model_KiB": r.mean_tokens_per_sentence * own / 1024,
            "kv_doc_own_model_MiB": DOC_SENTENCES * r.mean_tokens_per_sentence * own / 2**20,
            "kv_per_token_fixed_fp16_B": kv16[FIXED],
            "kv_per_sentence_fixed_model_KiB": r.mean_tokens_per_sentence * kv16[FIXED] / 1024,
            "kv_doc_fixed_model_MiB": DOC_SENTENCES * r.mean_tokens_per_sentence * kv16[FIXED] / 2**20,
            **run_metadata("arithmetic: step1 tokens x step8 KV/token", "fp16 KV", 1,
                           f"{DOC_SENTENCES} FLORES-like sentences", "none"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "09_language_kv_cost.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[["language", "tokenizer", "mean_tokens_per_sentence", "tokens_vs_english",
              "kv_per_sentence_own_model_KiB", "kv_doc_own_model_MiB", "kv_doc_fixed_model_MiB"]]
          .round(2).to_string(index=False))
    plot()


def plot():
    import matplotlib.pyplot as plt
    from src import plotstyle
    plotstyle.apply()
    df = pd.read_csv(RESULTS / "09_language_kv_cost.csv")
    langs = ["English", "Hindi", "Marathi"]
    toks = list(TOK2MODEL)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, title in [
        (axes[0], "kv_doc_own_model_MiB", f"A) Each tokenizer with its own model\n(KV for {DOC_SENTENCES:,} sentences, fp16)"),
        (axes[1], "kv_doc_fixed_model_MiB", f"B) Tokenizer effect only: all priced at Qwen2.5-0.5B KV/token\n(KV for {DOC_SENTENCES:,} sentences, fp16)"),
    ]:
        vals = {t: dict(zip(df[df.tokenizer == t].language, df[df.tokenizer == t][col])) for t in toks}
        plotstyle.grouped_bars(ax, langs, toks, vals, order=toks, fmt="{:.0f}")
        ax.set_title(title, loc="left")
        ax.set_ylabel("MiB of KV cache")
    fig.suptitle("Same meaning, different language: KV-cache memory (tokens from step 1 x KV/token from step 8)",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.TEXT2)
    fig.tight_layout()
    fig.savefig(PLOTS / "09_language_kv_cost.png", dpi=150)
    print("saved plots/09_language_kv_cost.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot()
    else:
        main()
