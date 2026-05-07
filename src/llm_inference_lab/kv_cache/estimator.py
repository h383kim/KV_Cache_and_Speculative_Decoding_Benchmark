from __future__ import annotations

import torch

from ..models.config_utils import extract_kv_dims
from ..utils.device import bytes_per_element


def estimate_kv_cache_memory(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """Return estimated KV cache memory in bytes.

    Stores both K and V, hence the factor of 2.
    """
    return (
        2
        * num_layers
        * batch_size
        * seq_len
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )


def estimate_for_model(model, batch_size: int, seq_len: int, dtype: torch.dtype) -> int:
    dims = extract_kv_dims(model.config)
    return estimate_kv_cache_memory(
        num_layers=dims.num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        num_kv_heads=dims.num_kv_heads,
        head_dim=dims.head_dim,
        bytes_per_element=bytes_per_element(dtype),
    )
