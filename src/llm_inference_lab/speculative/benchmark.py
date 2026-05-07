from __future__ import annotations

from typing import Any, Dict

import torch

from ..models.loader import LoadedModel
from .greedy import greedy_baseline, speculative_greedy


def run_spec_benchmark(
    target: LoadedModel,
    draft: LoadedModel,
    prompt: str,
    draft_steps: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)

    baseline = greedy_baseline(target, prompt_ids, max_new_tokens)
    spec = speculative_greedy(target, draft, prompt_ids, draft_steps, max_new_tokens)

    speedup = baseline["total_latency_ms"] / spec["total_latency_ms"] if spec["total_latency_ms"] > 0 else 0.0

    baseline_text = target.tokenizer.decode(baseline["output_ids"], skip_special_tokens=True)
    spec_text = target.tokenizer.decode(spec["output_ids"], skip_special_tokens=True)

    return {
        "target_model": target.name,
        "draft_model": draft.name,
        "prompt": prompt,
        "draft_steps": draft_steps,
        "max_new_tokens": max_new_tokens,
        "device": target.device.type,
        "dtype": str(target.dtype).replace("torch.", ""),
        "baseline": {
            **{k: v for k, v in baseline.items() if k != "output_ids"},
            "output_text": baseline_text,
        },
        "speculative": {
            **{k: v for k, v in spec.items() if k != "output_ids"},
            "output_text": spec_text,
        },
        "speedup": round(speedup, 3),
        "outputs_match": baseline["output_ids"] == spec["output_ids"],
    }
