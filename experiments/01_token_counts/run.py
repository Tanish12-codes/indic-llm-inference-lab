"""Step 1 - Token-count study.

Same ~200 sentences (FLORES+ dev split) in English, Hindi and Marathi, split by
three tokenizers. We measure:
  fertility      = tokens / words      (how many pieces one word becomes)
  chars_per_token = characters / tokens (how much text one token carries)
and verify decode(encode(text)) == text for every sentence.

Run (from the project root):
    .venv\\Scripts\\python.exe experiments\\01_token_counts\\run.py
Outputs:
    results/01_token_counts_per_sentence.csv
    results/01_token_counts_summary.csv
    results/01_hindi_split.txt
    plots/01_fertility_chars_per_token.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.common import PLOTS, RESULTS, run_metadata  # noqa: E402

LANGS = {"English": "eng_Latn", "Hindi": "hin_Deva", "Marathi": "mar_Deva"}
TOKENIZERS = {
    "GPT-2": "gpt2",
    "Qwen2.5": "Qwen/Qwen2.5-0.5B",
    "Param2": "bharatgenai/Param2-17B-A2.4B-Thinking",  # tokenizer files only - never the model
}
DATASET = "openlanguagedata/flores_plus"


def load_flores(n: int, split: str = "dev") -> pd.DataFrame:
    """Return a DataFrame with columns id, English, Hindi, Marathi (aligned by FLORES id)."""
    from datasets import load_dataset

    frames = {}
    for lang, code in LANGS.items():
        try:
            ds = load_dataset(DATASET, code, split=split)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "gated" in msg.lower() or "401" in msg or "403" in msg or "authenticat" in msg.lower():
                sys.exit(
                    "\nFLORES+ is gated. Please:\n"
                    "  1) open https://huggingface.co/datasets/openlanguagedata/flores_plus and accept the terms\n"
                    "  2) run  .venv\\Scripts\\hf auth login   (paste a READ token)\n"
                    f"Original error: {msg[:300]}"
                )
            raise
        df = ds.to_pandas()
        frames[lang] = df[["id", "text"]].rename(columns={"text": lang})
    out = frames["English"]
    for lang in ["Hindi", "Marathi"]:
        out = out.merge(frames[lang], on="id", how="inner")
    out = out.sort_values("id").head(n).reset_index(drop=True)
    return out


def load_tokenizer(repo: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(repo)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"\nCould not load tokenizer {repo}: {type(e).__name__}: {str(e)[:400]}\n"
            "If it says 'gated' or 401: open the model page on huggingface.co, accept the licence, "
            "and make sure you are logged in (hf auth login).\n"
            "If it asks for trust_remote_code: stop and review the remote code before enabling it."
        ) from e


def pieces(tok, ids):
    """How the text is cut: each token decoded on its own.
    A '�' (replacement char) means the token holds only PART of a character's UTF-8 bytes."""
    return [tok.decode([i], clean_up_tokenization_spaces=False) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    sents = load_flores(args.n, args.split)
    print(f"Loaded {len(sents)} aligned sentences from FLORES+ ({args.split}).")

    rows = []
    split_report = []
    hindi_example = sents.loc[0, "Hindi"]
    for tname, repo in TOKENIZERS.items():
        tok = load_tokenizer(repo)
        print(f"\n[{tname}] {repo}  vocab size = {len(tok):,}")
        for lang in LANGS:
            for _, r in sents.iterrows():
                text = r[lang]
                ids = tok.encode(text, add_special_tokens=False)
                back = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                rows.append({
                    "sentence_id": r["id"], "language": lang, "tokenizer": tname, "repo": repo,
                    "vocab_size": len(tok), "chars": len(text), "utf8_bytes": len(text.encode("utf-8")),
                    "words": len(text.split()), "tokens": len(ids), "roundtrip_ok": back == text,
                })
        ids = tok.encode(hindi_example, add_special_tokens=False)
        ps = pieces(tok, ids)
        split_report.append(f"=== {tname} ({repo}) : {len(ids)} tokens for {len(hindi_example.split())} words ===")
        split_report.append(" | ".join(repr(p)[1:-1] for p in ps))
        split_report.append("raw token strings: " + " ".join(tok.convert_ids_to_tokens(ids)[:60]))
        split_report.append("")

    df = pd.DataFrame(rows)
    meta = run_metadata(model="tokenizers only", precision="n/a", batch="n/a",
                        seq_len=f"{args.n} sentences", device="cpu")
    for k, v in meta.items():
        df[k] = v
    df.to_csv(RESULTS / "01_token_counts_per_sentence.csv", index=False, encoding="utf-8")

    g = df.groupby(["language", "tokenizer"], sort=False)
    summary = g.agg(sentences=("tokens", "size"), words=("words", "sum"), chars=("chars", "sum"),
                    tokens=("tokens", "sum"), roundtrip_ok=("roundtrip_ok", "mean"),
                    vocab_size=("vocab_size", "first")).reset_index()
    summary["fertility_tokens_per_word"] = summary["tokens"] / summary["words"]
    summary["chars_per_token"] = summary["chars"] / summary["tokens"]
    summary["mean_tokens_per_sentence"] = summary["tokens"] / summary["sentences"]
    eng = summary[summary.language == "English"].set_index("tokenizer")["tokens"]
    summary["tokens_vs_english"] = summary.apply(lambda r: r["tokens"] / eng[r["tokenizer"]], axis=1)
    summary["roundtrip_ok_pct"] = 100 * summary.pop("roundtrip_ok")
    for k, v in meta.items():
        summary[k] = v
    summary.to_csv(RESULTS / "01_token_counts_summary.csv", index=False, encoding="utf-8")

    header = [f"Hindi sentence (FLORES+ id {sents.loc[0, 'id']}):", hindi_example,
              f"chars={len(hindi_example)}  words={len(hindi_example.split())}  "
              f"utf8 bytes={len(hindi_example.encode('utf-8'))}", "",
              "Pieces are separated by ' | '. A \\ufffd piece holds only part of a character's bytes.", ""]
    (RESULTS / "01_hindi_split.txt").write_text("\n".join(header + split_report), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n" + summary[["language", "tokenizer", "fertility_tokens_per_word", "chars_per_token",
                          "mean_tokens_per_sentence", "tokens_vs_english", "roundtrip_ok_pct"]]
          .round(3).to_string(index=False))
    print("\n" + "\n".join(header[:3] + split_report))

    make_plot()


def make_plot():
    """Grouped bar chart, read back from the CSV (never from in-memory numbers)."""
    import matplotlib.pyplot as plt
    from src import plotstyle
    plotstyle.apply()
    s = pd.read_csv(RESULTS / "01_token_counts_summary.csv")
    langs = list(LANGS)
    toks = list(TOKENIZERS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, title, ylab in [
        (axes[0], "fertility_tokens_per_word", "Fertility: tokens per word (lower = cheaper)", "tokens / word"),
        (axes[1], "chars_per_token", "Characters carried per token (higher = cheaper)", "characters / token"),
    ]:
        vals = {t: dict(zip(s[s.tokenizer == t].language, s[s.tokenizer == t][col])) for t in toks}
        plotstyle.grouped_bars(ax, langs, toks, vals, order=toks)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
    fig.suptitle(f"Same {int(s.sentences.iloc[0])} FLORES+ sentences, three tokenizers",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.TEXT2)
    fig.tight_layout()
    fig.savefig(PLOTS / "01_fertility_chars_per_token.png", dpi=150)
    print(f"\nSaved {PLOTS / '01_fertility_chars_per_token.png'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        make_plot()
    else:
        main()
