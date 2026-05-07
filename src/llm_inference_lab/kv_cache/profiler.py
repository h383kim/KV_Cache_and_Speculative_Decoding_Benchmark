from __future__ import annotations

from typing import Any, Dict

import torch

from ..metrics.latency import Timer
from ..metrics.memory import peak_memory_mb, reset_peak
from ..models.loader import LoadedModel
from .estimator import estimate_for_model


def _make_synthetic_input(
    tokenizer, prompt_len: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Build a deterministic synthetic prompt of exact token length.

    We avoid real text so that varying prompt_len does not require finding a
    string that tokenizes to that length. Token ids are clamped to a safe
    range and the first id is the BOS/EOS so the model sees a sensible start.
    """
    vocab = tokenizer.vocab_size
    safe_id = (tokenizer.bos_token_id or tokenizer.eos_token_id or 0)
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(low=0, high=max(vocab - 1, 1), size=(batch_size, prompt_len), generator=g)
    ids[:, 0] = safe_id
    return ids.to(device)


@torch.no_grad()
def run_kv_benchmark(
    loaded: LoadedModel,
    prompt_len: int,
    generation_len: int,
    batch_size: int,
    use_cache: bool,
) -> Dict[str, Any]:
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device
    dtype = loaded.dtype

    input_ids = _make_synthetic_input(tokenizer, prompt_len, batch_size, device)
    reset_peak(device)

    # Warm up once so the first measurement isn't dominated by lazy init.
    _ = model(input_ids=input_ids[:, :1], use_cache=False)

    if use_cache:
        with Timer(device) as t_prefill:
            out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        decode_times_ms = []
        for _ in range(generation_len):
            with Timer(device) as t_step:
                out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            decode_times_ms.append(t_step.elapsed_ms)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    else:
        # Reproduce the no-cache path: full re-forward each step.
        with Timer(device) as t_prefill:
            out = model(input_ids=input_ids, use_cache=False)
        running = torch.cat(
            [input_ids, out.logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1
        )

        decode_times_ms = []
        for _ in range(generation_len):
            with Timer(device) as t_step:
                out = model(input_ids=running, use_cache=False)
            decode_times_ms.append(t_step.elapsed_ms)
            running = torch.cat(
                [running, out.logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1
            )

    prefill_ms = t_prefill.elapsed_ms
    avg_decode_ms = sum(decode_times_ms) / max(len(decode_times_ms), 1)
    total_ms = prefill_ms + sum(decode_times_ms)
    total_new_tokens = generation_len * batch_size
    tokens_per_second = total_new_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

    final_seq_len = prompt_len + generation_len
    estimated_kv_bytes = estimate_for_model(model, batch_size, final_seq_len, dtype)

    return {
        "model": loaded.name,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "generation_len": generation_len,
        "dtype": str(dtype).replace("torch.", ""),
        "use_cache": use_cache,
        "device": device.type,
        "prefill_latency_ms": round(prefill_ms, 3),
        "avg_decode_latency_ms": round(avg_decode_ms, 3),
        "total_latency_ms": round(total_ms, 3),
        "tokens_per_second": round(tokens_per_second, 3),
        "estimated_kv_cache_mb": round(estimated_kv_bytes / (1024 * 1024), 3),
        "peak_gpu_memory_mb": peak_memory_mb(device),
    }
