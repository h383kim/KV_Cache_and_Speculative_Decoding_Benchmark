# Project 2 Design Doc: KV Cache & Speculative Decoding Benchmark

## 1. Project Summary

Build a transformer inference benchmarking and visualization toolkit focused on two core inference optimizations:

1. **KV cache behavior during prefill and decode**
2. **Speculative decoding with draft-target model verification**

The goal is to understand and demonstrate why LLM inference is expensive, how KV cache affects memory usage, and when speculative decoding improves generation speed.

This project should be more systems-oriented than product-oriented.

It should answer questions like:

```text
How does KV cache memory grow with sequence length?
Why is prefill compute-heavy but decode memory-bandwidth-heavy?
How does batch size affect KV cache memory?
When does speculative decoding speed up inference?
How does draft model quality affect acceptance rate?
When can speculative decoding hurt performance?
```

---

## 2. Main Goal

Build a benchmark toolkit that measures and visualizes:

```text
prefill latency
decode latency
KV cache memory growth
tokens/sec
batch size effects
context length effects
speculative decoding acceptance rate
speedup vs baseline decoding
draft/target model trade-offs
```

This project should show strong understanding of:

```text
transformer inference internals
attention computation
KV cache mechanics
autoregressive decoding
speculative decoding
performance profiling
benchmark design
```

---

## 3. Target Resume Framing

Possible resume title:

> Built an LLM inference benchmark toolkit analyzing KV-cache memory growth, prefill/decode latency, and speculative decoding speedups.

Possible resume bullets:

```text
Implemented a transformer inference benchmarking toolkit profiling prefill latency, decode latency, KV-cache memory growth, and tokens/sec across context lengths, batch sizes, and model sizes.

Built a speculative decoding implementation with draft-target verification, measuring acceptance rate, speedup, output equivalence, and failure modes across generation workloads.

Developed visualizations explaining how sequence length, batch size, number of layers, hidden size, and attention heads affect KV-cache memory and decode-time bottlenecks.
```

---

## 4. System Architecture

```text
CLI / Notebook / Dashboard
    |
    v
Benchmark Runner
    |
    +--> Model Loader
    |       - target model
    |       - draft model
    |
    +--> KV Cache Profiler
    |       - prefill timing
    |       - decode timing
    |       - memory tracking
    |       - theoretical memory estimator
    |
    +--> Speculative Decoder
    |       - draft token generation
    |       - target verification
    |       - acceptance/rejection logic
    |
    +--> Metrics Collector
    |       - latency
    |       - tokens/sec
    |       - acceptance rate
    |       - speedup
    |       - memory
    |
    +--> Result Store
            - JSONL logs
            - CSV summaries
            - plots
```

---

## 5. Suggested Tech Stack

Core:

```text
Python
PyTorch
Transformers
NumPy
Pandas
Matplotlib
Typer or argparse
Pydantic
```

Optional:

```text
Streamlit
FastAPI
Docker
nvidia-ml-py
torch.profiler
```

For small model experiments:

```text
distilgpt2
gpt2
TinyLlama
Qwen small models
Pythia small models
```

The project should work on small models first so it is easy to reproduce.

---

## 6. Part A: KV Cache Profiler

### 6.1 Concept

During autoregressive generation, the model repeatedly computes attention.

Without KV cache:

```text
At every new token, recompute keys and values for all previous tokens.
```

With KV cache:

```text
Store previous keys and values.
For each new token, only compute query/key/value for the new token.
Reuse previous K/V tensors.
```

This reduces repeated computation but increases memory usage.

---

### 6.2 What to Measure

For a given model and prompt:

```text
prefill latency
decode latency per token
total generation latency
tokens/sec
peak GPU memory
KV cache estimated memory
actual memory usage
```

Vary:

```text
sequence length: 128, 512, 1024, 2048, 4096
batch size: 1, 2, 4, 8
generation length: 32, 64, 128, 256
dtype: fp32, fp16, bf16 if supported
use_cache: true vs false
```

---

### 6.3 Theoretical KV Cache Memory Formula

Implement a function:

```python
def estimate_kv_cache_memory(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """
    Return estimated KV cache memory in bytes.

    KV cache stores both K and V.

    memory = 2 * num_layers * batch_size * seq_len
             * num_kv_heads * head_dim * bytes_per_element
    """
```

Important: distinguish:

```text
MHA: num_kv_heads = num_attention_heads
GQA/MQA: num_kv_heads < num_attention_heads
```

This connects directly to modern LLM inference.

---

### 6.4 Benchmark Output Example

```json
{
  "model": "gpt2",
  "batch_size": 1,
  "prompt_len": 512,
  "generation_len": 128,
  "dtype": "fp16",
  "use_cache": true,
  "prefill_latency_ms": 85.2,
  "avg_decode_latency_ms": 11.3,
  "total_latency_ms": 1531.0,
  "tokens_per_second": 83.6,
  "estimated_kv_cache_mb": 144.0,
  "peak_gpu_memory_mb": 2210.0
}
```

---

### 6.5 Visualizations

Generate plots:

```text
KV cache memory vs sequence length
KV cache memory vs batch size
decode latency vs sequence length
tokens/sec vs context length
use_cache true vs false latency
estimated vs actual memory
```

Each plot should be saved to:

```text
reports/figures/
```

---

## 7. Part B: Speculative Decoding

### 7.1 Concept

Speculative decoding uses two models:

```text
draft model: small and fast
target model: large and accurate
```

Process:

```text
1. Draft model proposes k tokens.
2. Target model verifies those k tokens in one forward pass.
3. Accept matching tokens.
4. If a token is rejected, sample from target model.
5. Repeat.
```

Goal:

```text
Generate multiple tokens per expensive target-model call.
```

---

### 7.2 Implementation Scope

Implement a simplified speculative decoder.

Start with greedy decoding first.

Models:

```text
target_model = gpt2
draft_model = distilgpt2
```

Or use:

```text
target_model = larger small model
draft_model = smaller compatible model
```

Important: use the same tokenizer when possible.

---

### 7.3 Greedy Speculative Decoding Algorithm

Simplified version:

```text
Input:
    prompt
    target model T
    draft model D
    draft_steps k
    max_new_tokens

Loop until max_new_tokens:
    1. Use D to generate k draft tokens autoregressively.
    2. Run T on prompt + accepted tokens + draft tokens.
    3. For each draft token:
        - Get T's greedy token prediction at that position.
        - If target token == draft token:
            accept draft token.
        - Else:
            reject draft token.
            append target token.
            stop checking remaining draft tokens.
    4. If all k draft tokens accepted:
        optionally append one extra target token.
```

Track:

```text
draft_tokens_proposed
draft_tokens_accepted
acceptance_rate
target_forward_calls
draft_forward_calls
latency
speedup_vs_baseline
```

---

### 7.4 Metrics

For baseline target-only decoding:

```text
total_latency_ms
tokens/sec
target_forward_calls
output_text
```

For speculative decoding:

```text
total_latency_ms
tokens/sec
draft_forward_calls
target_forward_calls
draft_tokens_proposed
draft_tokens_accepted
acceptance_rate
speedup
output_text
```

Formula:

```text
acceptance_rate = accepted_draft_tokens / proposed_draft_tokens
speedup = baseline_latency / speculative_latency
```

---

### 7.5 Experiment Matrix

Vary:

```text
draft_steps k: 1, 2, 4, 8, 16
prompt length: short, medium, long
generation length: 32, 64, 128
temperature: 0 for greedy first
draft model size
target model size
```

Optional later:

```text
temperature > 0
top-p sampling
different domains: code, general text, math, medical-style text
```

---

### 7.6 Expected Findings to Document

The README should explain:

```text
Speculative decoding is faster when the draft model is cheap and close enough to the target model.
High acceptance rate usually means better speedup.
If the draft model is too weak, many tokens are rejected.
If the draft model is too large, draft generation overhead removes the benefit.
Larger draft_steps can help when acceptance is high but hurt when acceptance is low.
```

---

## 8. CLI Design

### KV cache benchmark

```bash
python -m llm_inference_lab kv-benchmark   --model gpt2   --prompt-len 512   --generation-len 128   --batch-size 1   --dtype fp16   --use-cache true   --output results/kv_gpt2.json
```

### Sweep

```bash
python -m llm_inference_lab kv-sweep   --config configs/kv_sweep.yaml
```

### Speculative decoding benchmark

```bash
python -m llm_inference_lab spec-decode   --target-model gpt2   --draft-model distilgpt2   --prompt "Explain how transformer inference works."   --draft-steps 4   --max-new-tokens 128   --output results/spec_decode.json
```

### Speculative sweep

```bash
python -m llm_inference_lab spec-sweep   --config configs/spec_sweep.yaml
```

---

## 9. Config Examples

### KV sweep config

```yaml
model: gpt2
dtype: fp16
prompt_lengths: [128, 512, 1024, 2048]
generation_lengths: [32, 128]
batch_sizes: [1, 2, 4]
use_cache_options: [true, false]
output_dir: results/kv_sweep
```

### Speculative sweep config

```yaml
target_model: gpt2
draft_model: distilgpt2
prompts:
  - "Explain how attention works in transformers."
  - "Write a Python function to merge two sorted lists."
  - "Summarize the benefits of quantization for LLM inference."
draft_steps: [1, 2, 4, 8]
max_new_tokens: [32, 64, 128]
temperature: 0.0
output_dir: results/spec_sweep
```

---

## 10. Repository Structure

```text
llm-inference-systems-lab/
  README.md
  pyproject.toml
  Dockerfile

  src/
    llm_inference_lab/
      __init__.py

      cli.py

      models/
        loader.py
        config_utils.py

      kv_cache/
        estimator.py
        profiler.py
        benchmark.py

      speculative/
        greedy.py
        verifier.py
        benchmark.py

      metrics/
        latency.py
        memory.py
        summary.py

      reports/
        plots.py
        tables.py

      utils/
        device.py
        logging.py
        seed.py

  configs/
    kv_sweep.yaml
    spec_sweep.yaml

  results/
    .gitkeep

  reports/
    figures/
    summary.md

  notebooks/
    kv_cache_analysis.ipynb
    speculative_decoding_analysis.ipynb

  tests/
    test_kv_estimator.py
    test_speculative_greedy.py
    test_metrics.py
```

---

## 11. Implementation Milestones

### Milestone 1: KV Cache Memory Estimator

Tasks:

```text
Read model config
Extract num_layers, num_heads, num_kv_heads, hidden size, head_dim
Implement memory formula
Unit test estimator
Print memory table for different seq lengths
```

Deliverable:

```text
CLI command that estimates KV cache memory.
```

---

### Milestone 2: Basic Generation Profiler

Tasks:

```text
Load model and tokenizer
Run generation with use_cache=True
Run generation with use_cache=False
Measure latency and tokens/sec
Measure GPU memory if CUDA available
Save JSON result
```

Deliverable:

```text
Benchmark result comparing cache vs no-cache.
```

---

### Milestone 3: Prefill vs Decode Timing

Tasks:

```text
Separate first forward pass from token-by-token decoding
Measure prefill latency
Measure per-token decode latency
Track generated tokens
Save detailed traces
```

Deliverable:

```text
CSV/JSONL trace with one row per generated token.
```

---

### Milestone 4: KV Cache Sweep + Plots

Tasks:

```text
Sweep prompt length, batch size, generation length
Save results
Generate plots
Write analysis summary
```

Deliverable:

```text
reports/kv_cache_analysis.md
```

---

### Milestone 5: Baseline Greedy Decoder

Tasks:

```text
Implement manual greedy decoding using target model
Compare output with transformers.generate
Measure latency
```

Deliverable:

```text
baseline decoder used for fair comparison.
```

---

### Milestone 6: Speculative Greedy Decoder

Tasks:

```text
Implement draft-token generation
Implement target verification
Implement accept/reject logic
Track acceptance rate
Compare latency against baseline
```

Deliverable:

```text
working speculative decoding implementation.
```

---

### Milestone 7: Speculative Sweep + Analysis

Tasks:

```text
Vary draft_steps
Vary prompt types
Vary generation length
Measure speedup
Plot acceptance rate vs speedup
Document failure cases
```

Deliverable:

```text
reports/speculative_decoding_analysis.md
```

---

## 12. Stretch Goals

```text
Support sampling-based speculative decoding
Support Medusa-style multi-token heads conceptually
Add EAGLE-style draft approximation experiment
Add paged KV cache simulator
Add prefix caching simulator
Add continuous batching simulator
Add Streamlit dashboard
Add torch.profiler traces
Add NVIDIA GPU utilization logging
```

---

## 13. Main Technical Risks

### Risk 1: Speculative decoding with different tokenizers

Mitigation:

```text
Use draft and target models with the same tokenizer family.
Start with GPT-2 and DistilGPT-2.
```

### Risk 2: Hard to beat baseline speed

Mitigation:

```text
The goal is analysis, not necessarily guaranteed speedup.
Document when speculative decoding helps and when it does not.
Small models may show less dramatic speedup because overhead dominates.
```

### Risk 3: GPU memory not available

Mitigation:

```text
Support CPU benchmarking.
Record memory where available.
Keep theoretical KV cache estimator independent of hardware.
```

### Risk 4: Transformers generate hides internals

Mitigation:

```text
Implement manual decoding loop for measurement.
Use model(..., past_key_values=...) directly.
```

---

## 14. What Makes This Resume-Strong

This project demonstrates:

```text
Low-level understanding of autoregressive decoding
KV cache mechanics
Prefill vs decode phases
Memory/latency trade-offs
Speculative decoding implementation
Benchmark design
Performance analysis
```

This is especially relevant for:

```text
ML infrastructure engineer
AI inference engineer
LLM systems engineer
ML performance engineer
Applied ML engineer working on model serving
```
