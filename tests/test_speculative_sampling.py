import pytest
import torch

from llm_inference_lab.models.loader import load_model_and_tokenizer
from llm_inference_lab.speculative.benchmark import run_spec_benchmark
from llm_inference_lab.speculative.sampling import _filter_logits


def test_filter_logits_temperature_only():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    probs = _filter_logits(logits, temperature=1.0, top_p=None)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]))
    assert torch.argmax(probs).item() == 3  # highest logit still wins


def test_filter_logits_top_p_keeps_at_least_one():
    # All probabilities equal -> top-p=0.1 should still keep at least one token.
    logits = torch.zeros(1, 8)
    probs = _filter_logits(logits, temperature=1.0, top_p=0.1)
    assert (probs > 0).sum().item() >= 1
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]))


def test_filter_logits_top_p_zeros_tail():
    # Sharp distribution: top token holds nearly all mass.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    probs = _filter_logits(logits, temperature=1.0, top_p=0.5)
    # The top token has > 0.99 mass, so top-p=0.5 should keep ONLY it.
    assert (probs > 0).sum().item() == 1
    assert probs.argmax().item() == 0


@pytest.mark.slow
def test_sampling_spec_runs_and_acceptance_in_range():
    """Smoke test: sampling spec produces valid output with a sensible acceptance rate."""
    torch.manual_seed(123)
    device = torch.device("cpu")
    dtype = torch.float32
    target = load_model_and_tokenizer("distilgpt2", dtype, device)
    draft = load_model_and_tokenizer("distilgpt2", dtype, device)

    result = run_spec_benchmark(
        target=target,
        draft=draft,
        prompt="The quick brown fox",
        draft_steps=4,
        max_new_tokens=16,
        temperature=1.0,
        top_p=0.9,
    )

    assert result["mode"] == "sampling"
    assert result["outputs_match"] is None  # sampling: not expected to match exactly
    rate = result["speculative"]["acceptance_rate"]
    assert 0.0 <= rate <= 1.0
    # When draft == target, sampling should accept most of the time (q ≈ p).
    assert rate > 0.5
    # Output text decodes to a non-empty string.
    assert len(result["speculative"]["output_text"]) > 0


@pytest.mark.slow
def test_sampling_spec_low_temperature_high_acceptance():
    """At very low temperature, sampling spec should behave near-greedy (high acceptance when self-spec)."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    dtype = torch.float32
    target = load_model_and_tokenizer("distilgpt2", dtype, device)
    draft = load_model_and_tokenizer("distilgpt2", dtype, device)

    result = run_spec_benchmark(
        target=target,
        draft=draft,
        prompt="Hello world",
        draft_steps=2,
        max_new_tokens=8,
        temperature=0.01,
        top_p=None,
    )
    # Low temperature + same model -> q and p nearly identical -> acceptance ≈ 1.
    assert result["speculative"]["acceptance_rate"] > 0.9
