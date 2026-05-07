import time

import torch

from llm_inference_lab.metrics.latency import Timer
from llm_inference_lab.metrics.memory import peak_memory_mb


def test_timer_records_nonzero_elapsed():
    with Timer(torch.device("cpu")) as t:
        time.sleep(0.01)
    assert t.elapsed_ms >= 9.0  # allow small jitter


def test_peak_memory_none_on_cpu():
    assert peak_memory_mb(torch.device("cpu")) is None
