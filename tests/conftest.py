import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Exact FP32 maths for correctness tests (TF32 would round matmul inputs to ~10 bits).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="session")
def hf_gpt2():
    from transformers import GPT2LMHeadModel
    try:
        m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    except TypeError:
        m = GPT2LMHeadModel.from_pretrained("gpt2")
    return m.eval()


@pytest.fixture(scope="session")
def our_gpt2():
    from src.gpt2_scratch import GPT2
    return GPT2.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")
