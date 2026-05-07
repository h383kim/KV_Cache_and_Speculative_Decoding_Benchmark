from __future__ import annotations

import time

import torch

from ..utils.device import sync


class Timer:
    """Context manager that records wall-clock elapsed time in milliseconds.

    Synchronizes the device on enter and exit so GPU work is fully measured.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        sync(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        sync(self.device)
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
