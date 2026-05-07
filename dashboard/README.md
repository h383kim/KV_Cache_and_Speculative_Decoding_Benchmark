# Dashboard

Streamlit interface for the inference benchmarks.

## Run locally

```bash
uv pip install -e ".[dev,dashboard]"
uv run streamlit run dashboard/app.py
```

The dashboard has three tabs:

1. **Live KV benchmark** — pick a model, prompt length, generation length, batch size, dtype, and `use_cache`. Runs `run_kv_benchmark` and renders prefill / decode latency, tokens/sec, and the theoretical KV-cache footprint.
2. **Live speculative decoding** — pick a target + draft model, a prompt, draft steps `k`, and `max_new_tokens`. Runs both the target-only baseline and the greedy speculative decoder, shows speedup, acceptance rate, target-call reduction, and the two output texts side by side.
3. **Sweep viewer** — pick (or upload) a `results.jsonl` produced by `kv-sweep` or `spec-sweep` and regenerate every figure from `reports/plots.py`.

Models are cached via `@st.cache_resource`, so reloading the same model within a session is instant.

## Deploy on Streamlit Cloud (free)

1. Push to GitHub (already done).
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud), sign in with GitHub, "New app".
3. Repo: `h383kim/KV_Cache_and_Speculative_Decoding_Benchmark`. Branch: `main`. Main file: `dashboard/app.py`.
4. Add a `requirements.txt` shim if needed (Streamlit Cloud reads this rather than `pyproject.toml` extras). On a CPU-only Streamlit Cloud runner, expect the live tabs to be slow; the sweep viewer is fast either way.
