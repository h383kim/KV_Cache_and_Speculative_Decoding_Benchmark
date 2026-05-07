from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVDims:
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int


def _first_attr(config, names, default=None):
    for n in names:
        if hasattr(config, n) and getattr(config, n) is not None:
            return getattr(config, n)
    return default


def extract_kv_dims(config) -> KVDims:
    """Extract the dimensions needed for KV cache memory accounting.

    Handles MHA (num_kv_heads == num_attention_heads), GQA, and MQA
    (num_kv_heads < num_attention_heads, identified via num_key_value_heads).
    """
    num_layers = _first_attr(config, ["num_hidden_layers", "n_layer", "num_layers"])
    num_attention_heads = _first_attr(config, ["num_attention_heads", "n_head"])
    hidden_size = _first_attr(config, ["hidden_size", "n_embd", "d_model"])
    num_kv_heads = _first_attr(
        config,
        ["num_key_value_heads", "num_kv_heads"],
        default=num_attention_heads,
    )
    head_dim = _first_attr(config, ["head_dim"]) or (hidden_size // num_attention_heads)

    if None in (num_layers, num_attention_heads, hidden_size):
        raise ValueError(f"could not infer KV dims from config: {config}")

    return KVDims(
        num_layers=int(num_layers),
        num_attention_heads=int(num_attention_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        hidden_size=int(hidden_size),
    )
