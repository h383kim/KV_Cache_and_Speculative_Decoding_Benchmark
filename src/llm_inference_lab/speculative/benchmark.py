from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..models.loader import LoadedModel
from .greedy import greedy_baseline, speculative_greedy
from .sampling import sampling_baseline, speculative_sampling


def run_spec_benchmark(
    target: LoadedModel,
    draft: LoadedModel,
    prompt: str,
    draft_steps: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)

    if temperature <= 0:
        mode = "greedy"
        baseline = greedy_baseline(target, prompt_ids, max_new_tokens)
        spec = speculative_greedy(target, draft, prompt_ids, draft_steps, max_new_tokens)
        # Greedy decoding is deterministic, so baseline and spec outputs must match.
        outputs_match = baseline["output_ids"] == spec["output_ids"]
    else:
        mode = "sampling"
        baseline = sampling_baseline(target, prompt_ids, max_new_tokens, temperature, top_p)
        spec = speculative_sampling(
            target, draft, prompt_ids, draft_steps, max_new_tokens, temperature, top_p
        )
        # Sampling is stochastic — outputs WILL differ token by token even if both are correct.
        # The correctness guarantee is distributional, not exact equality.
        outputs_match = None

    speedup = baseline["total_latency_ms"] / spec["total_latency_ms"] if spec["total_latency_ms"] > 0 else 0.0

    baseline_text = target.tokenizer.decode(baseline["output_ids"], skip_special_tokens=True)
    spec_text = target.tokenizer.decode(spec["output_ids"], skip_special_tokens=True)

    return {
        "target_model": target.name,
        "draft_model": draft.name,
        "prompt": prompt,
        "draft_steps": draft_steps,
        "max_new_tokens": max_new_tokens,
        "mode": mode,
        "temperature": temperature,
        "top_p": top_p,
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
        "outputs_match": outputs_match,
    }
