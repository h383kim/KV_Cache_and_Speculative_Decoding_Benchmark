from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .kv_cache.profiler import run_kv_benchmark
from .models.loader import load_model_and_tokenizer
from .speculative.benchmark import run_spec_benchmark
from .utils.device import dtype_from_str, pick_device
from .utils.seed import set_seed

app = typer.Typer(add_completion=False, help="LLM inference benchmarking toolkit.")


def _emit(result: dict, output: Optional[Path]) -> None:
    text = json.dumps(result, indent=2)
    typer.echo(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)


@app.command("kv-benchmark")
def kv_benchmark(
    model: str = typer.Option(..., help="HuggingFace model id (e.g. distilgpt2, gpt2)."),
    prompt_len: int = typer.Option(128, help="Synthetic prompt length in tokens."),
    generation_len: int = typer.Option(32, help="Number of new tokens to generate."),
    batch_size: int = typer.Option(1),
    dtype: str = typer.Option("fp32", help="fp32, fp16, or bf16."),
    use_cache: bool = typer.Option(True),
    seed: int = typer.Option(0),
    output: Optional[Path] = typer.Option(None, help="Optional path to write the result JSON."),
) -> None:
    set_seed(seed)
    device = pick_device()
    torch_dtype = dtype_from_str(dtype)
    loaded = load_model_and_tokenizer(model, torch_dtype, device)
    result = run_kv_benchmark(
        loaded=loaded,
        prompt_len=prompt_len,
        generation_len=generation_len,
        batch_size=batch_size,
        use_cache=use_cache,
    )
    _emit(result, output)


@app.command("spec-decode")
def spec_decode(
    target_model: str = typer.Option(..., help="Target (large/accurate) model id."),
    draft_model: str = typer.Option(..., help="Draft (small/fast) model id."),
    prompt: str = typer.Option(..., help="Prompt text."),
    draft_steps: int = typer.Option(4, help="Number of draft tokens proposed per iteration."),
    max_new_tokens: int = typer.Option(64),
    dtype: str = typer.Option("fp32"),
    seed: int = typer.Option(0),
    output: Optional[Path] = typer.Option(None),
) -> None:
    set_seed(seed)
    device = pick_device()
    torch_dtype = dtype_from_str(dtype)
    target = load_model_and_tokenizer(target_model, torch_dtype, device)
    draft = load_model_and_tokenizer(draft_model, torch_dtype, device)
    result = run_spec_benchmark(
        target=target,
        draft=draft,
        prompt=prompt,
        draft_steps=draft_steps,
        max_new_tokens=max_new_tokens,
    )
    _emit(result, output)


if __name__ == "__main__":
    app()
