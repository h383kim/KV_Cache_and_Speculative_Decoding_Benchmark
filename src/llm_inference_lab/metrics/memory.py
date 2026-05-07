from __future__ import annotations

from typing import Optional

import torch


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: torch.device) -> Optional[float]:
    """Return peak allocated memory in MB on CUDA, else None.

    MPS lacks a comparable peak-memory API in stable torch, so callers should
    fall back to the theoretical KV-cache estimator when this returns None.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None
