"""Memory calculator: read any Hugging Face config.json -> weight memory and KV-cache per token.

    python -m src.memory_model gpt2 Qwen/Qwen2.5-0.5B bharatgenai/Param2-17B-A2.4B-Thinking
    python -m src.memory_model path/to/config.json

Handles the usual naming zoo:
    layers   : num_hidden_layers | n_layer | num_layers
    hidden   : hidden_size | n_embd | d_model
    q heads  : num_attention_heads | n_head
    KV heads : num_key_value_heads   (missing -> = q heads, i.e. MHA; multi_query -> 1)
    head_dim : head_dim              (missing/None -> hidden / q heads)
    MoE      : num_experts | num_local_experts | n_routed_experts, moe_intermediate_size,
               n_shared_experts | shared_expert_intermediate_size, first_k_dense_replace,
               decoder_sparse_step, num_experts_per_tok
    MLA      : kv_lora_rank (+ qk_rope_head_dim) - DeepSeek-style compressed KV
Parameter counts are an ESTIMATE from the config (biases/norms approximated); when the hub
publishes safetensors metadata we print that exact count beside it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

BYTES = {"fp32": 4, "fp16/bf16": 2, "int8": 1, "int4": 0.5}


def load_config(src: str) -> dict:
    p = Path(src)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    if p.is_dir() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(src, "config.json")).read_text(encoding="utf-8"))


def _get(c: dict, *names, default=None):
    for n in names:
        if c.get(n) is not None:
            return c[n]
    return default


@dataclass
class ModelShape:
    name: str
    model_type: str
    n_layer: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    intermediate: int
    tied: bool
    gated_mlp: bool
    n_experts: int = 0
    experts_per_tok: int = 0
    moe_intermediate: int = 0
    shared_intermediate: int = 0
    n_moe_layers: int = 0
    n_positions: int = 0
    mla_kv_dim: int = 0          # compressed KV width per layer for MLA models (0 = not MLA)
    notes: list = field(default_factory=list)


def parse(cfg: dict, name: str = "?") -> ModelShape:
    if "text_config" in cfg and "num_hidden_layers" not in cfg:     # multimodal wrappers
        cfg = {**cfg, **cfg["text_config"]}
    notes = []
    mt = cfg.get("model_type", "?")
    L = _get(cfg, "num_hidden_layers", "n_layer", "num_layers")
    d = _get(cfg, "hidden_size", "n_embd", "d_model")
    H = _get(cfg, "num_attention_heads", "n_head")
    kv = _get(cfg, "num_key_value_heads", "num_kv_heads", "n_head_kv")
    if kv is None:
        kv = 1 if cfg.get("multi_query") else H
        notes.append(f"num_key_value_heads missing -> {'MQA (1)' if kv == 1 else 'MHA (= query heads)'}")
    hd = _get(cfg, "head_dim")
    if hd is None:
        hd = d // H
        notes.append("head_dim missing -> hidden_size / num_attention_heads")
    inter = _get(cfg, "intermediate_size", "n_inner", "ffn_dim") or 4 * d
    act = str(_get(cfg, "hidden_act", "activation_function", default=""))
    gated = mt not in ("gpt2", "gpt_neox", "gptj", "opt", "bloom", "falcon") and (
        "silu" in act or "swiglu" in act or "gelu" in act and mt in ("gemma", "gemma2", "gemma3"))
    tied = bool(cfg.get("tie_word_embeddings", True))
    s = ModelShape(name, mt, L, d, H, kv, hd, cfg["vocab_size"], inter, tied, gated, notes=notes,
                   n_positions=_get(cfg, "n_positions", "max_position_embeddings", default=0))

    ne = _get(cfg, "num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts", default=0) or 0
    if ne:
        s.n_experts = ne
        s.experts_per_tok = _get(cfg, "num_experts_per_tok", "moe_top_k", "top_k", default=0) or 0
        s.moe_intermediate = _get(cfg, "moe_intermediate_size", "expert_intermediate_size", default=inter)
        n_sh = _get(cfg, "n_shared_experts", "num_shared_experts", default=0) or 0
        sh_i = _get(cfg, "shared_expert_intermediate_size", default=0) or 0
        s.shared_intermediate = sh_i or n_sh * s.moe_intermediate
        dense_first = _get(cfg, "first_k_dense_replace", "num_dense_layers", default=0) or 0
        step = _get(cfg, "decoder_sparse_step", "moe_layer_freq", default=1) or 1
        only = set(cfg.get("mlp_only_layers") or [])
        s.n_moe_layers = sum(1 for i in range(L) if i >= dense_first and (i + 1) % step == 0 and i not in only)
        notes.append(f"MoE: {ne} experts, top-{s.experts_per_tok}, {s.n_moe_layers}/{L} MoE layers, "
                     f"shared-expert width {s.shared_intermediate}")
    if cfg.get("kv_lora_rank"):
        s.mla_kv_dim = cfg["kv_lora_rank"] + (cfg.get("qk_rope_head_dim") or 0)
        notes.append(f"MLA: cache stores a {s.mla_kv_dim}-wide latent per layer instead of K and V heads")
    return s


def param_counts(s: ModelShape) -> dict:
    d, L, V = s.hidden, s.n_layer, s.vocab
    attn = d * s.n_heads * s.head_dim + 2 * d * s.n_kv_heads * s.head_dim + s.n_heads * s.head_dim * d
    if s.model_type == "gpt2":
        attn += 4 * d                                   # c_attn + c_proj biases
    elif s.model_type.startswith("qwen2"):
        attn += s.n_heads * s.head_dim + 2 * s.n_kv_heads * s.head_dim   # Qwen2 has q/k/v biases
    norms = (4 if s.model_type == "gpt2" else 2) * d
    mats = 3 if s.gated_mlp else 2
    dense_mlp = mats * d * s.intermediate + ((s.intermediate + d) if s.model_type == "gpt2" else 0)
    embed = V * d + (0 if s.tied else V * d)
    if s.model_type == "gpt2":
        embed += s.n_positions * d                      # learned position embeddings
    total = embed + L * (attn + norms) + d
    active = total
    if s.n_experts:
        expert = 3 * d * s.moe_intermediate
        shared = 3 * d * s.shared_intermediate
        router = d * s.n_experts
        moe_layer_all = s.n_experts * expert + shared + router
        moe_layer_act = s.experts_per_tok * expert + shared + router
        n_dense = L - s.n_moe_layers
        total += n_dense * dense_mlp + s.n_moe_layers * moe_layer_all
        active += n_dense * dense_mlp + s.n_moe_layers * moe_layer_act
    else:
        total += L * dense_mlp
        active += L * dense_mlp
    return {"params_total_est": total, "params_active_est": active}


def kv_bytes_per_token(s: ModelShape, bytes_per_elem: float = 2) -> float:
    if s.mla_kv_dim:
        return s.n_layer * s.mla_kv_dim * bytes_per_elem
    return 2 * s.n_layer * s.n_kv_heads * s.head_dim * bytes_per_elem


def published_param_count(repo: str):
    """Exact count from the hub's safetensors metadata (no weights downloaded). None if unavailable."""
    try:
        from huggingface_hub import get_safetensors_metadata
        meta = get_safetensors_metadata(repo)
        return int(sum(meta.parameter_count.values()))
    except Exception:  # noqa: BLE001
        return None


def report(src: str, check_hub: bool = True) -> dict:
    s = parse(load_config(src), src)
    pc = param_counts(s)
    pub = published_param_count(src) if check_hub and not Path(src).exists() else None
    row = {**{k: v for k, v in asdict(s).items() if k != "notes"}, **pc,
           "params_published_safetensors": pub, "notes": "; ".join(s.notes)}
    base = pub or pc["params_total_est"]
    for k, b in BYTES.items():
        row[f"weights_{k}_bytes"] = base * b
        row[f"kv_per_token_{k}_bytes"] = kv_bytes_per_token(s, b)
    return row


def print_report(r: dict):
    gib = 2**30
    print(f"\n=== {r['name']}  ({r['model_type']}) ===")
    print(f"layers={r['n_layer']} hidden={r['hidden']} q_heads={r['n_heads']} kv_heads={r['n_kv_heads']} "
          f"head_dim={r['head_dim']} vocab={r['vocab']:,} tied_embeddings={r['tied']}")
    if r["notes"]:
        print("notes: " + r["notes"])
    print(f"params (estimated from config): total {r['params_total_est'] / 1e6:,.1f} M, "
          f"active/token {r['params_active_est'] / 1e6:,.1f} M")
    if r["params_published_safetensors"]:
        print(f"params (hub safetensors metadata): {r['params_published_safetensors'] / 1e6:,.1f} M")
    print("weights: " + ", ".join(f"{k} {r[f'weights_{k}_bytes'] / gib:.2f} GiB" for k in BYTES))
    kv16 = r["kv_per_token_fp16/bf16_bytes"]
    print(f"KV cache per token: {kv16 / 1024:.1f} KiB in fp16  "
          f"(= 2 x {r['n_layer']} layers x {r['n_kv_heads']} KV heads x {r['head_dim']} dims x 2 bytes)"
          if not r["mla_kv_dim"] else f"KV cache per token (MLA latent): {kv16 / 1024:.1f} KiB in fp16")
    for n in (1_000, 32_000):
        print(f"   {n:>6,} tokens x batch 1 -> {kv16 * n / gib:.3f} GiB")


if __name__ == "__main__":
    for src in sys.argv[1:] or ["gpt2", "Qwen/Qwen2.5-0.5B"]:
        print_report(report(src))
