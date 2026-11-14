"""Environment check: OS, Python, GPU, disk, and every library the project needs.

Run:  .venv\\Scripts\\python.exe scripts\\env_check.py
Writes the same report to results/env_check.txt.
"""
import importlib
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lines = []


def say(s=""):
    print(s)
    lines.append(s)


say("=== Indic LLM Inference Lab: environment check ===")
say(f"OS:      {platform.system()} {platform.release()} ({platform.version()})")
say(f"Python:  {sys.version.split()[0]}  ({sys.executable})")
in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
say(f"venv:    {'yes' if in_venv else 'NO - activate .venv first'}")
total, used, free = shutil.disk_usage(ROOT)
say(f"Disk:    {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB (drive holding the project)")
say("")

missing = []
for mod in ["torch", "transformers", "tokenizers", "datasets", "huggingface_hub",
            "safetensors", "sentencepiece", "pandas", "matplotlib", "pytest"]:
    try:
        m = importlib.import_module(mod)
        say(f"  [ok] {mod:16s} {getattr(m, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        say(f"  [MISSING] {mod:16s} {type(e).__name__}: {e}")
        missing.append(mod)
say("")

try:
    import torch
    say(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    say(f"torch built with CUDA:     {torch.version.cuda}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        say(f"GPU:     {p.name}, {p.total_memory / 2**30:.2f} GiB VRAM, "
            f"compute capability {p.major}.{p.minor}, {p.multi_processor_count} SMs")
        free_b, total_b = torch.cuda.mem_get_info()
        say(f"VRAM free right now: {free_b / 2**30:.2f} GiB")
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        say(f"Tiny GPU matmul test: ok ({y:.1f})")
    else:
        say("!! No CUDA GPU visible to torch. Most likely the CPU-only torch wheel was installed.")
        say("   Fix: .venv\\Scripts\\pip uninstall -y torch ; "
            ".venv\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cu124")
except ImportError:
    pass

try:
    from huggingface_hub import whoami
    who = whoami()
    say(f"Hugging Face login: yes ({who.get('name')})")
except Exception:  # noqa: BLE001
    say("Hugging Face login: NO  -> run:  .venv\\Scripts\\hf auth login   (or huggingface-cli login)")

say("")
say("Missing packages: " + (", ".join(missing) if missing else "none"))
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "env_check.txt").write_text("\n".join(lines), encoding="utf-8")
