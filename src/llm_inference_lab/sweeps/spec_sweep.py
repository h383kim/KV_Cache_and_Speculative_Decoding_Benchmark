from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict

from ..models.loader import load_model_and_tokenizer
from ..speculative.benchmark import run_spec_benchmark
from ..utils.device import dtype_from_str, pick_device
from ..utils.seed import set_seed
from .configs import SpecSweepConfig


def _flatten_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested baseline/speculative dicts into one flat row for JSONL."""
    flat = {k: v for k, v in result.items() if k not in {"baseline", "speculative"}}
    for prefix, inner in (("baseline", result["baseline"]), ("spec", result["speculative"])):
        for k, v in inner.items():
            flat[f"{prefix}_{k}"] = v
    return flat


def run_spec_sweep(config: SpecSweepConfig) -> Path:
    """Run a speculative-decoding sweep over (prompt, draft_steps, max_new_tokens)."""
    set_seed(config.seed)
    device = pick_device()
    dtype = dtype_from_str(config.dtype)
    target = load_model_and_tokenizer(config.target_model, dtype, device)
    draft = load_model_and_tokenizer(config.draft_model, dtype, device)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"

    combos = list(
        itertools.product(config.prompts, config.draft_steps, config.max_new_tokens)
    )
    total = len(combos)

    with out_path.open("w") as f:
        for i, (prompt, k, max_new) in enumerate(combos, start=1):
            short = prompt[:32].replace("\n", " ")
            print(
                f"[spec-sweep {i}/{total}] k={k} max_new={max_new} prompt={short!r}",
                file=sys.stderr,
                flush=True,
            )
            result = run_spec_benchmark(
                target=target,
                draft=draft,
                prompt=prompt,
                draft_steps=k,
                max_new_tokens=max_new,
            )
            f.write(json.dumps(_flatten_row(result)) + "\n")
            f.flush()

    print(f"[spec-sweep] wrote {total} rows to {out_path}", file=sys.stderr)
    return out_path
