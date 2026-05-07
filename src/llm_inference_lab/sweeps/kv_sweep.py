from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from ..kv_cache.profiler import run_kv_benchmark
from ..models.loader import load_model_and_tokenizer
from ..utils.device import dtype_from_str, pick_device
from ..utils.seed import set_seed
from .configs import KVSweepConfig


def run_kv_sweep(config: KVSweepConfig) -> Path:
    """Run a KV-cache benchmark across the configured matrix.

    Loads the model+tokenizer ONCE (model and dtype are scalar in this config) and
    iterates the cartesian product of prompt_len, generation_len, batch_size,
    use_cache. Each row is appended to a JSONL file.
    """
    set_seed(config.seed)
    device = pick_device()
    dtype = dtype_from_str(config.dtype)
    loaded = load_model_and_tokenizer(config.model, dtype, device)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"

    combos = list(
        itertools.product(
            config.prompt_lengths,
            config.generation_lengths,
            config.batch_sizes,
            config.use_cache_options,
        )
    )
    total = len(combos)

    with out_path.open("w") as f:
        for i, (prompt_len, gen_len, batch, use_cache) in enumerate(combos, start=1):
            print(
                f"[kv-sweep {i}/{total}] prompt_len={prompt_len} gen_len={gen_len} "
                f"batch={batch} use_cache={use_cache}",
                file=sys.stderr,
                flush=True,
            )
            row = run_kv_benchmark(
                loaded=loaded,
                prompt_len=prompt_len,
                generation_len=gen_len,
                batch_size=batch,
                use_cache=use_cache,
            )
            f.write(json.dumps(row) + "\n")
            f.flush()

    print(f"[kv-sweep] wrote {total} rows to {out_path}", file=sys.stderr)
    return out_path
