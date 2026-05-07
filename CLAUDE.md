# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM inference benchmarking toolkit for two core optimizations: **KV cache profiling** and **speculative decoding**. The full design spec is in `kv_cache_speculative_decoding_benchmark_design_doc.md`. The package is named `llm_inference_lab` and lives under `src/`.

## Common Commands

```bash
# Run KV cache benchmark for a single config
python -m llm_inference_lab kv-benchmark \
  --model gpt2 --prompt-len 512 --generation-len 128 \
  --batch-size 1 --dtype fp16 --use-cache true \
  --output results/kv_gpt2.json

# Sweep over prompt lengths, batch sizes, dtypes (driven by YAML)
python -m llm_inference_lab kv-sweep --config configs/kv_sweep.yaml

# Speculative decoding single run
python -m llm_inference_lab spec-decode \
  --target-model gpt2 --draft-model distilgpt2 \
  --prompt "Explain how transformer inference works." \
  --draft-steps 4 --max-new-tokens 128 \
  --output results/spec_decode.json

# Speculative decoding sweep
python -m llm_inference_lab spec-sweep --config configs/spec_sweep.yaml

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_kv_estimator.py
```

## Architecture

```
src/llm_inference_lab/
  cli.py                  # Typer CLI entrypoints: kv-benchmark, kv-sweep, spec-decode, spec-sweep
  models/
    loader.py             # HuggingFace model + tokenizer loading
    config_utils.py       # Extract num_layers, num_heads, num_kv_heads, head_dim from model config
  kv_cache/
    estimator.py          # estimate_kv_cache_memory() — theoretical formula, no hardware needed
    profiler.py           # Separate prefill vs per-token decode timing; manual decoding loop
    benchmark.py          # Orchestrates sweeps, saves JSONL/CSV traces
  speculative/
    greedy.py             # Draft token generation + accept/reject loop (greedy only)
    verifier.py           # Target model forward pass over prompt+draft tokens
    benchmark.py          # Baseline vs speculative latency comparison
  metrics/
    latency.py            # Timing utilities (CUDA sync before timing on GPU)
    memory.py             # Peak GPU memory via torch.cuda.max_memory_allocated; theoretical estimator wrapper
    summary.py            # Aggregate JSONL → summary stats
  reports/
    plots.py              # Matplotlib figures → reports/figures/
    tables.py             # Pandas summary tables
  utils/
    device.py             # Device selection (CUDA → MPS → CPU)
    logging.py            # Structured logging
    seed.py               # torch.manual_seed for reproducibility
configs/
  kv_sweep.yaml           # Sweep matrix: prompt_lengths, batch_sizes, dtypes, use_cache_options
  spec_sweep.yaml         # Sweep matrix: draft_steps, prompts, max_new_tokens
results/                  # JSON/JSONL/CSV benchmark outputs
reports/figures/          # Saved matplotlib plots
notebooks/                # Jupyter analysis notebooks
tests/
  test_kv_estimator.py    # Unit tests for estimate_kv_cache_memory formula
  test_speculative_greedy.py
  test_metrics.py
```

## Key Design Decisions

**Manual decoding loop**: Use `model(..., past_key_values=past_kv)` directly instead of `model.generate()` so prefill and per-token decode latencies can be measured separately. `transformers.generate` hides these boundaries.

**KV cache memory formula**:
```python
memory_bytes = 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_element
# Factor of 2 for K and V
# MHA: num_kv_heads == num_attention_heads
# GQA/MQA: num_kv_heads < num_attention_heads
```

**Speculative decoding (greedy)**: Draft model proposes `k` tokens → target model verifies in one forward pass → accept matching tokens greedily → on first mismatch, use target's token and restart draft. Track `acceptance_rate = accepted / proposed` and `speedup = baseline_latency / speculative_latency`.

**Tokenizer constraint**: Draft and target models must share the same tokenizer (start with `distilgpt2` → `gpt2`).

**GPU memory timing**: Always call `torch.cuda.synchronize()` before and after timed regions on CUDA to get accurate wall-clock measurements.

## Sweep Config Schema

KV sweep (`configs/kv_sweep.yaml`):
```yaml
model: gpt2
dtype: fp16
prompt_lengths: [128, 512, 1024, 2048]
generation_lengths: [32, 128]
batch_sizes: [1, 2, 4]
use_cache_options: [true, false]
output_dir: results/kv_sweep
```

Spec sweep (`configs/spec_sweep.yaml`):
```yaml
target_model: gpt2
draft_model: distilgpt2
prompts: [...]
draft_steps: [1, 2, 4, 8]
max_new_tokens: [32, 64, 128]
temperature: 0.0
output_dir: results/spec_sweep
```

## Tech Stack

- Python, PyTorch, Transformers (HuggingFace)
- NumPy, Pandas for data handling
- Matplotlib for plots
- Typer for CLI, Pydantic for config validation
- Small models for fast iteration: `distilgpt2`, `gpt2`, `TinyLlama`
