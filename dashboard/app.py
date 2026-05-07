"""Streamlit dashboard for the LLM inference benchmarking toolkit.

Run locally:
    uv pip install -e ".[dev,dashboard]"
    uv run streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

# Make `src/llm_inference_lab` importable when launched via `streamlit run` from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_inference_lab.kv_cache.profiler import run_kv_benchmark  # noqa: E402
from llm_inference_lab.models.loader import load_model_and_tokenizer  # noqa: E402
from llm_inference_lab.reports.plots import plot_kv_sweep, plot_spec_sweep  # noqa: E402
from llm_inference_lab.speculative.benchmark import run_spec_benchmark  # noqa: E402
from llm_inference_lab.utils.device import dtype_from_str, pick_device  # noqa: E402

st.set_page_config(page_title="LLM Inference Lab", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def cached_load(name: str, dtype_str: str, device_str: str):
    return load_model_and_tokenizer(name, dtype_from_str(dtype_str), torch.device(device_str))


def _device_label() -> str:
    d = pick_device()
    return d.type


tab_kv, tab_spec, tab_sweep = st.tabs(
    ["Live KV benchmark", "Live speculative decoding", "Sweep viewer"]
)

# ---------------------------------------------------------------------------
# Tab 1: Live KV benchmark
# ---------------------------------------------------------------------------
with tab_kv:
    st.header("KV cache profiler")
    st.caption(
        "Manual prefill + per-token decode loop. Compares cache-on vs cache-off and "
        "shows the theoretical KV-cache memory footprint."
    )
    with st.form("kv_form"):
        c1, c2, c3 = st.columns(3)
        model_name = c1.text_input("model", "distilgpt2")
        prompt_len = c1.number_input("prompt_len (tokens)", 16, 4096, 128, step=16)
        generation_len = c2.number_input("generation_len", 1, 512, 32)
        batch_size = c2.number_input("batch_size", 1, 8, 1)
        dtype_str = c3.selectbox("dtype", ["fp32", "fp16", "bf16"], index=0)
        use_cache = c3.checkbox("use_cache", value=True)
        submitted = st.form_submit_button("Run benchmark")

    if submitted:
        loaded = cached_load(model_name, dtype_str, _device_label())
        with st.spinner("Running…"):
            result = run_kv_benchmark(
                loaded=loaded,
                prompt_len=int(prompt_len),
                generation_len=int(generation_len),
                batch_size=int(batch_size),
                use_cache=use_cache,
            )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("prefill (ms)", f"{result['prefill_latency_ms']:.1f}")
        m2.metric("avg decode (ms/tok)", f"{result['avg_decode_latency_ms']:.2f}")
        m3.metric("tokens/sec", f"{result['tokens_per_second']:.1f}")
        m4.metric("KV cache (MB)", f"{result['estimated_kv_cache_mb']:.2f}")
        st.bar_chart(
            pd.DataFrame(
                {"latency_ms": [result["prefill_latency_ms"], result["avg_decode_latency_ms"]]},
                index=["prefill (1 forward)", "avg decode (per token)"],
            )
        )
        with st.expander("Full result JSON"):
            st.code(json.dumps(result, indent=2), language="json")

# ---------------------------------------------------------------------------
# Tab 2: Live speculative decoding
# ---------------------------------------------------------------------------
with tab_spec:
    st.header("Greedy speculative decoding")
    st.caption(
        "Draft model proposes k tokens; target verifies in one forward pass; "
        "outputs are guaranteed identical to the target-only baseline."
    )
    with st.form("spec_form"):
        c1, c2 = st.columns(2)
        target_name = c1.text_input("target model", "gpt2")
        draft_name = c1.text_input("draft model", "distilgpt2")
        prompt = c1.text_area(
            "prompt", "Once upon a time, in a land far away,", height=80
        )
        draft_steps = c2.slider("draft_steps (k)", 1, 16, 4)
        max_new_tokens = c2.slider("max_new_tokens", 8, 256, 32, step=8)
        dtype_str = c2.selectbox("dtype ", ["fp32", "fp16", "bf16"], index=0)
        submitted = st.form_submit_button("Run speculative decoding")

    if submitted:
        target = cached_load(target_name, dtype_str, _device_label())
        draft = cached_load(draft_name, dtype_str, _device_label())
        with st.spinner("Running baseline + speculative…"):
            result = run_spec_benchmark(
                target=target,
                draft=draft,
                prompt=prompt,
                draft_steps=int(draft_steps),
                max_new_tokens=int(max_new_tokens),
            )

        b = result["baseline"]
        s = result["speculative"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("speedup", f"{result['speedup']:.2f}×")
        c2.metric("acceptance rate", f"{s['acceptance_rate']:.0%}")
        c3.metric(
            "target calls",
            f"{s['target_forward_calls']}",
            delta=f"{s['target_forward_calls'] - b['target_forward_calls']} vs baseline",
            delta_color="inverse",
        )
        c4.metric(
            "outputs match",
            "yes" if result["outputs_match"] else "no",
        )

        st.subheader("Forward-call breakdown")
        st.bar_chart(
            pd.DataFrame(
                {
                    "baseline": [b["target_forward_calls"], 0],
                    "speculative": [s["target_forward_calls"], s["draft_forward_calls"]],
                },
                index=["target", "draft"],
            )
        )

        st.subheader("Generated text")
        col_a, col_b = st.columns(2)
        col_a.markdown("**Baseline (target only)**")
        col_a.text_area("baseline output", b["output_text"], height=160, label_visibility="collapsed")
        col_b.markdown("**Speculative (target + draft)**")
        col_b.text_area("spec output", s["output_text"], height=160, label_visibility="collapsed")

        with st.expander("Full result JSON"):
            st.code(json.dumps(result, indent=2), language="json")

# ---------------------------------------------------------------------------
# Tab 3: Sweep viewer
# ---------------------------------------------------------------------------
with tab_sweep:
    st.header("Sweep viewer")
    st.caption(
        "Pick a sweep results.jsonl from the `results/` folder (or upload one) and "
        "regenerate the analysis figures."
    )

    available = sorted((REPO_ROOT / "results").rglob("*.jsonl"))
    options = ["<upload>"] + [str(p.relative_to(REPO_ROOT)) for p in available]
    choice = st.selectbox("Choose a JSONL", options)
    upload = st.file_uploader("…or upload one", type=["jsonl"]) if choice == "<upload>" else None
    kind = st.radio("Sweep kind", ["kv", "spec"], horizontal=True)

    if st.button("Render plots"):
        if choice != "<upload>":
            jsonl_path = REPO_ROOT / choice
        elif upload is not None:
            jsonl_path = REPO_ROOT / "results" / "_upload.jsonl"
            jsonl_path.write_bytes(upload.getvalue())
        else:
            st.error("Pick a file or upload one.")
            st.stop()

        out_dir = REPO_ROOT / "reports" / "figures" / f"_dashboard_{kind}"
        with st.spinner("Generating figures…"):
            paths = (plot_kv_sweep if kind == "kv" else plot_spec_sweep)(jsonl_path, out_dir)

        df = pd.read_json(jsonl_path, lines=True)
        st.subheader(f"Raw rows: {len(df)}")
        st.dataframe(df, use_container_width=True)

        st.subheader("Figures")
        for p in paths:
            st.image(str(p), caption=p.name, use_container_width=True)
