"""Why does decode(encode(text)) != text for some Hindi sentences under Qwen2.5?
Prints, for each failure, the exact characters that changed and whether Unicode
normalisation (NFC/NFKC) explains it. No sentence text is saved to disk.
    .venv\\Scripts\\python.exe experiments\\01_token_counts\\check_roundtrip.py
"""
import sys
import unicodedata as ud
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("exp01", Path(__file__).with_name("run.py"))
exp01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp01)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print("Qwen normalizer:", tok.backend_tokenizer.normalizer)
sents = exp01.load_flores(200)
fails = 0
explained = {"NFC": 0, "NFKC": 0, "other": 0}
for _, r in sents.iterrows():
    text = r["Hindi"]
    back = tok.decode(tok.encode(text, add_special_tokens=False), clean_up_tokenization_spaces=False)
    if back == text:
        continue
    fails += 1
    if back == ud.normalize("NFC", text):
        explained["NFC"] += 1
    elif back == ud.normalize("NFKC", text):
        explained["NFKC"] += 1
    else:
        explained["other"] += 1
    if fails <= 5:
        def names(chars):
            return sorted({"U+%04X %s" % (ord(c), ud.name(c, "?")) for c in chars})
        a = names(c for c in text if c not in back)
        b = names(c for c in back if c not in text)
        print(f"id {r['id']}: len {len(text)} -> {len(back)}; only in original: {a}; only in decoded: {b}")
print(f"\n{fails} failures; explained by: {explained}")
