"""Small helpers shared by every experiment: paths, device choice, run metadata.

Every results CSV gets the same metadata columns (GPU, torch version, model,
precision, batch, seq length, date) so a number can never lose its context.
"""
from __future__ import annotations

import datetime as _dt
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)


def get_device(prefer: str = "cuda"):
    import torch
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "none (CPU)"


def torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "not installed"


def run_metadata(model: str, precision: str, batch: int | str, seq_len: int | str,
                 device: str | None = None) -> dict:
    """Columns that go on every row of every results CSV."""
    return {
        "gpu": gpu_name(),
        "device": device or ("cuda" if gpu_name() != "none (CPU)" else "cpu"),
        "torch_version": torch_version(),
        "model": model,
        "precision": precision,
        "batch": batch,
        "seq_len": seq_len,
        "date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "os": f"{platform.system()} {platform.release()}",
    }


def set_seed(seed: int = 0):
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fmt_bytes(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PiB"
