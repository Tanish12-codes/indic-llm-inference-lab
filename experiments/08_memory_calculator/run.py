"""Step 8 - Memory calculator vs measurement.

GPT-2 and Qwen2.5-0.5B: calculator (from config.json) AND measured on the GPU
  - weights: sum of parameter bytes, and CUDA memory before/after loading
  - KV per token: run a forward pass with use_cache=True on N tokens, sum the cache tensors' bytes / N
Param2 (17B MoE): calculator + hub-published parameter count only. We NEVER download its weights.

    .venv\\Scripts\\python.exe experiments\\08_memory_calculator\\run.py
Outputs: results/08_memory_calculator.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.common import RESULTS, get_device, run_metadata  # noqa: E402
from src.memory_model import print_report, report  # noqa: E402

MEASURE = {"gpt2": torch.float32, "Qwen/Qwen2.5-0.5B": torch.float16}
CONFIG_ONLY = ["bharatgenai/Param2-17B-A2.4B-Thinking"]
N_TOKENS = 512


def cache_tensors(pkv):
    """Works across transformers versions (Cache.layers / key_cache lists / legacy tuples)."""
    if hasattr(pkv, "layers"):
        for layer in pkv.layers:
            yield layer.keys
            yield layer.values
    elif hasattr(pkv, "key_cache"):
        yield from pkv.key_cache
        yield from pkv.value_cache
    else:
        for kv in pkv:
            yield from kv


def measure(repo, dtype, dev):
    from transformers import AutoModelForCausalLM
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        m0 = torch.cuda.memory_allocated()
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(dev).eval()
    w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_params = sum(p.numel() for p in model.parameters())
    w_cuda = torch.cuda.memory_allocated() - m0 if dev.type == "cuda" else float("nan")
    ids = torch.randint(100, 1000, (1, N_TOKENS), device=dev)
    with torch.no_grad():
        if dev.type == "cuda":
            torch.cuda.synchronize()
            m1 = torch.cuda.memory_allocated()
        out = model(ids, use_cache=True)
        pkv = out.past_key_values
        del out                                   # drop logits so only the cache stays alive
        if dev.type == "cuda":
            torch.cuda.synchronize()
            kv_cuda = torch.cuda.memory_allocated() - m1
        else:
            kv_cuda = float("nan")
        kv_tensors = sum(t.numel() * t.element_size() for t in cache_tensors(pkv))
    del model, pkv
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"measured_params": n_params, "measured_weight_bytes": w_bytes,
            "measured_weight_cuda_bytes": w_cuda, "measured_dtype": str(dtype).replace("torch.", ""),
            "measured_kv_per_token_bytes": kv_tensors / N_TOKENS,
            "measured_kv_per_token_cuda_bytes": kv_cuda / N_TOKENS, "measure_tokens": N_TOKENS}


def main():
    dev = get_device()
    rows = []
    for repo, dtype in MEASURE.items():
        r = report(repo)
        print_report(r)
        r.update(measure(repo, dtype, dev))
        b = 4 if dtype == torch.float32 else 2
        r["calc_kv_per_token_at_measured_dtype"] = r["kv_per_token_fp32_bytes" if b == 4 else "kv_per_token_fp16/bf16_bytes"]
        r["calc_weight_bytes_at_measured_dtype"] = r["params_total_est"] * b
        print(f"   MEASURED ({r['measured_dtype']}): params {r['measured_params'] / 1e6:.1f} M, "
              f"weights {r['measured_weight_bytes'] / 2**20:.1f} MiB (CUDA delta {r['measured_weight_cuda_bytes'] / 2**20:.1f} MiB), "
              f"KV/token {r['measured_kv_per_token_bytes']:.0f} B vs calculator {r['calc_kv_per_token_at_measured_dtype']:.0f} B "
              f"(CUDA delta {r['measured_kv_per_token_cuda_bytes']:.0f} B)")
        r.update(run_metadata(repo, r["measured_dtype"], 1, N_TOKENS, dev.type))
        rows.append(r)
    for repo in CONFIG_ONLY:
        try:
            r = report(repo)
        except Exception as e:  # noqa: BLE001
            print(f"\nCould not read {repo} config ({type(e).__name__}: {str(e)[:200]}). "
                  "If gated: accept the licence on huggingface.co and log in.")
            continue
        print_report(r)
        print("   (arithmetic from the published config only - weights never downloaded)")
        r.update(run_metadata(repo, "n/a (config arithmetic)", "n/a", "n/a", "none"))
        rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS / "08_memory_calculator.csv", index=False)
    print(f"\nsaved {RESULTS / '08_memory_calculator.csv'}")


if __name__ == "__main__":
    main()
