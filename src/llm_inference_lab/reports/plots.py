from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # no GUI; safe inside Streamlit and headless CI
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_kv_sweep(jsonl_path: str | Path, out_dir: str | Path) -> List[Path]:
    df = pd.read_json(jsonl_path, lines=True)
    df["seq_len"] = df["prompt_len"] + df["generation_len"]
    out_dir = Path(out_dir)
    paths: List[Path] = []

    # 1) KV cache memory vs sequence length, one line per batch_size.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cache_df = df[df["use_cache"] == True].sort_values("seq_len")  # noqa: E712
    for batch, sub in cache_df.groupby("batch_size"):
        agg = sub.groupby("seq_len", as_index=False)["estimated_kv_cache_mb"].mean()
        ax.plot(agg["seq_len"], agg["estimated_kv_cache_mb"], marker="o", label=f"batch={batch}")
    ax.set_xlabel("sequence length (prompt + generated)")
    ax.set_ylabel("estimated KV cache (MB)")
    ax.set_title(f"KV cache memory vs sequence length — {df['model'].iloc[0]}")
    ax.legend()
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir, "kv_memory_vs_seq_len.png"))

    # 2) Decode latency vs sequence length, by use_cache flag (batch_size=1 slice for clarity).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub_b1 = df[df["batch_size"] == df["batch_size"].min()].sort_values("seq_len")
    for use_cache_val, sub in sub_b1.groupby("use_cache"):
        label = "use_cache=True" if use_cache_val else "use_cache=False"
        agg = sub.groupby("seq_len", as_index=False)["avg_decode_latency_ms"].mean()
        ax.plot(agg["seq_len"], agg["avg_decode_latency_ms"], marker="o", label=label)
    ax.set_xlabel("sequence length")
    ax.set_ylabel("avg decode latency (ms / token)")
    ax.set_title("Decode latency: KV cache on vs off")
    ax.legend()
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir, "decode_latency_vs_seq_len.png"))

    # 3) Tokens/sec vs sequence length, by use_cache.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for use_cache_val, sub in sub_b1.groupby("use_cache"):
        label = "use_cache=True" if use_cache_val else "use_cache=False"
        agg = sub.groupby("seq_len", as_index=False)["tokens_per_second"].mean()
        ax.plot(agg["seq_len"], agg["tokens_per_second"], marker="o", label=label)
    ax.set_xlabel("sequence length")
    ax.set_ylabel("tokens / second")
    ax.set_title("Throughput: KV cache on vs off")
    ax.legend()
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir, "tokens_per_second_vs_seq_len.png"))

    # 4) Prefill vs avg-decode latency (grouped bar by prompt_len, use_cache=True only).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub_cache = cache_df[cache_df["batch_size"] == cache_df["batch_size"].min()]
    grp = (
        sub_cache.groupby("prompt_len", as_index=False)
        .agg(prefill=("prefill_latency_ms", "mean"), decode=("avg_decode_latency_ms", "mean"))
        .sort_values("prompt_len")
    )
    x = range(len(grp))
    width = 0.4
    ax.bar([i - width / 2 for i in x], grp["prefill"], width=width, label="prefill (1 forward)")
    ax.bar([i + width / 2 for i in x], grp["decode"], width=width, label="avg decode (per token)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grp["prompt_len"])
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Prefill vs decode: compute-heavy vs bandwidth-heavy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    paths.append(_save(fig, out_dir, "prefill_vs_decode_latency.png"))

    return paths


def plot_spec_sweep(jsonl_path: str | Path, out_dir: str | Path) -> List[Path]:
    df = pd.read_json(jsonl_path, lines=True)
    out_dir = Path(out_dir)
    paths: List[Path] = []

    # 1) Acceptance rate vs draft_steps, one line per prompt (averaged over max_new_tokens).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for prompt, sub in df.groupby("prompt"):
        agg = sub.groupby("draft_steps", as_index=False)["spec_acceptance_rate"].mean()
        label = (prompt[:30] + "…") if len(prompt) > 30 else prompt
        ax.plot(agg["draft_steps"], agg["spec_acceptance_rate"], marker="o", label=label)
    ax.set_xlabel("draft steps (k)")
    ax.set_ylabel("acceptance rate")
    ax.set_title("Acceptance rate vs draft steps")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir, "acceptance_rate_vs_draft_steps.png"))

    # 2) Speedup vs draft_steps, one line per prompt.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for prompt, sub in df.groupby("prompt"):
        agg = sub.groupby("draft_steps", as_index=False)["speedup"].mean()
        label = (prompt[:30] + "…") if len(prompt) > 30 else prompt
        ax.plot(agg["draft_steps"], agg["speedup"], marker="o", label=label)
    ax.axhline(1.0, linestyle="--", color="grey", linewidth=1, label="baseline (1.0×)")
    ax.set_xlabel("draft steps (k)")
    ax.set_ylabel("speedup vs baseline")
    ax.set_title("Speedup vs draft steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir, "speedup_vs_draft_steps.png"))

    # 3) Target forward calls: baseline vs speculative (averaged across prompts).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grp = df.groupby("draft_steps", as_index=False).agg(
        baseline=("baseline_target_forward_calls", "mean"),
        spec=("spec_target_forward_calls", "mean"),
    )
    x = range(len(grp))
    width = 0.4
    ax.bar([i - width / 2 for i in x], grp["baseline"], width=width, label="baseline")
    ax.bar([i + width / 2 for i in x], grp["spec"], width=width, label="speculative")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grp["draft_steps"])
    ax.set_xlabel("draft steps (k)")
    ax.set_ylabel("target forward calls (avg)")
    ax.set_title("Target forward calls: baseline vs speculative")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    paths.append(_save(fig, out_dir, "target_calls_reduction.png"))

    return paths
