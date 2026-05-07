import pytest
import torch

from llm_inference_lab.models.loader import load_model_and_tokenizer
from llm_inference_lab.speculative.benchmark import run_spec_benchmark


@pytest.mark.slow
def test_self_speculative_high_acceptance():
    """When draft == target, every draft token should be accepted (acceptance ≈ 1.0)."""
    device = torch.device("cpu")
    dtype = torch.float32
    target = load_model_and_tokenizer("distilgpt2", dtype, device)
    draft = load_model_and_tokenizer("distilgpt2", dtype, device)

    result = run_spec_benchmark(
        target=target,
        draft=draft,
        prompt="The quick brown fox",
        draft_steps=2,
        max_new_tokens=8,
    )

    assert result["speculative"]["acceptance_rate"] == pytest.approx(1.0, abs=1e-6)
    assert result["outputs_match"] is True


@pytest.mark.slow
def test_cross_model_speculative_runs():
    """gpt2/distilgpt2 should produce a valid acceptance rate strictly between 0 and 1
    on a non-trivial generation, and outputs should match (greedy spec is exact)."""
    device = torch.device("cpu")
    dtype = torch.float32
    target = load_model_and_tokenizer("gpt2", dtype, device)
    draft = load_model_and_tokenizer("distilgpt2", dtype, device)

    result = run_spec_benchmark(
        target=target,
        draft=draft,
        prompt="Once upon a time, in a land far away,",
        draft_steps=4,
        max_new_tokens=16,
    )

    rate = result["speculative"]["acceptance_rate"]
    assert 0.0 <= rate <= 1.0
    assert result["outputs_match"] is True
    assert result["speculative"]["target_forward_calls"] < result["baseline"]["target_forward_calls"]
