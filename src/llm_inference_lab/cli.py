from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .kv_cache.profiler import run_kv_benchmark
from .models.loader import load_model_and_tokenizer
from .reports.plots import plot_kv_sweep, plot_spec_sweep
from .speculative.benchmark import run_spec_benchmark
from .sweeps.configs import load_kv_sweep, load_spec_sweep
from .sweeps.kv_sweep import run_kv_sweep
from .sweeps.spec_sweep import run_spec_sweep
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


@app.command("kv-sweep")
def kv_sweep(
    config: Path = typer.Option(..., help="Path to a KV sweep YAML config."),
) -> None:
    cfg = load_kv_sweep(config)
    out = run_kv_sweep(cfg)
    typer.echo(str(out))


@app.command("spec-sweep")
def spec_sweep(
    config: Path = typer.Option(..., help="Path to a speculative-decoding sweep YAML config."),
) -> None:
    cfg = load_spec_sweep(config)
    out = run_spec_sweep(cfg)
    typer.echo(str(out))


@app.command("plot")
def plot(
    kind: str = typer.Option(..., help="kv or spec"),
    input: Path = typer.Option(..., help="Path to a sweep results.jsonl."),
    output: Path = typer.Option(Path("reports/figures"), help="Directory to write PNGs to."),
) -> None:
    if kind == "kv":
        paths = plot_kv_sweep(input, output)
    elif kind == "spec":
        paths = plot_spec_sweep(input, output)
    else:
        raise typer.BadParameter("kind must be 'kv' or 'spec'")
    for p in paths:
        typer.echo(str(p))


if __name__ == "__main__":
    app()
