"""Sampling-based speculative decoding (Leviathan et al. 2023).

Differs from `speculative_greedy` only in the accept criterion:
- Draft samples each token x ~ q, where q is the (top-p, temperature-scaled) draft distribution.
- Target produces filtered distribution p at the same positions.
- Accept with probability min(1, p(x)/q(x)).
- On rejection: sample correction from the residual (p - q)+ / sum((p - q)+).

The residual-sampling step is what guarantees the final output distribution exactly matches p.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..metrics.latency import Timer
from ..models.loader import LoadedModel
from .greedy import _check_compatible_tokenizers


def _filter_logits(
    logits: torch.Tensor, temperature: float, top_p: Optional[float]
) -> torch.Tensor:
    """Convert logits at one position into a filtered probability distribution.

    Shape: input [B, V] -> output [B, V] summing to 1 along V.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use speculative_greedy for greedy decoding")
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p is None or top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Drop positions strictly past the cutoff but always keep at least the top token.
    mask = cumulative > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def _sample_from(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token id from a [B, V] distribution. Returns [B, 1]."""
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def sampling_baseline(
    target: LoadedModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    """Target-only sampling baseline used to time the sampling speedup."""
    device = target.device
    model = target.model
    target_forward_calls = 0

    with Timer(device) as t:
        out = model(input_ids=prompt_ids, use_cache=True)
        target_forward_calls += 1
        past = out.past_key_values
        probs = _filter_logits(out.logits[:, -1, :], temperature, top_p)
        next_token = _sample_from(probs)
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            target_forward_calls += 1
            past = out.past_key_values
            probs = _filter_logits(out.logits[:, -1, :], temperature, top_p)
            next_token = _sample_from(probs)
            generated.append(next_token)

    output_ids = torch.cat([prompt_ids] + generated, dim=1)
    total_ms = t.elapsed_ms
    tps = max_new_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0
    return {
        "output_ids": output_ids[0].tolist(),
        "total_latency_ms": round(total_ms, 3),
        "tokens_per_second": round(tps, 3),
        "target_forward_calls": target_forward_calls,
    }


@torch.no_grad()
def speculative_sampling(
    target: LoadedModel,
    draft: LoadedModel,
    prompt_ids: torch.Tensor,
    draft_steps: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    """Distribution-preserving speculative decoding (batch_size=1)."""
    if prompt_ids.size(0) != 1:
        raise ValueError("speculative_sampling supports batch_size=1 only")
    _check_compatible_tokenizers(target, draft)

    device = target.device
    t_model = target.model
    d_model = draft.model

    target_forward_calls = 0
    draft_forward_calls = 0
    proposed = 0
    accepted_total = 0

    with Timer(device) as t:
        # Prefill target: sample first token from filtered target distribution at the last prompt position.
        out = t_model(input_ids=prompt_ids, use_cache=True)
        target_forward_calls += 1
        target_past = out.past_key_values
        first_probs = _filter_logits(out.logits[:, -1, :], temperature, top_p)
        first_token = _sample_from(first_probs)

        # Prefill draft.
        out_d = d_model(input_ids=prompt_ids, use_cache=True)
        draft_forward_calls += 1
        draft_past = out_d.past_key_values

        generated_tokens = [first_token]
        last_token = first_token

        while len(generated_tokens) < max_new_tokens:
            # Phase 1: draft samples k tokens, recording q at each position.
            drafts = []
            draft_probs = []
            cur = last_token
            for _ in range(draft_steps):
                out_d = d_model(input_ids=cur, past_key_values=draft_past, use_cache=True)
                draft_forward_calls += 1
                draft_past = out_d.past_key_values
                q = _filter_logits(out_d.logits[:, -1, :], temperature, top_p)  # [1, V]
                cur = _sample_from(q)
                drafts.append(cur)
                draft_probs.append(q)
            # Extra pass so drafts[-1] enters the draft cache (matches greedy bookkeeping).
            out_d = d_model(input_ids=cur, past_key_values=draft_past, use_cache=True)
            draft_forward_calls += 1
            draft_past = out_d.past_key_values

            drafts_tensor = torch.cat(drafts, dim=1)  # [1, k]
            Q = torch.cat(draft_probs, dim=0)  # [k, V]

            # Phase 2: target verifies in one forward pass.
            target_input = torch.cat([last_token, drafts_tensor], dim=1)  # [1, k+1]
            out = t_model(input_ids=target_input, past_key_values=target_past, use_cache=True)
            target_forward_calls += 1
            target_past = out.past_key_values
            P = torch.stack(
                [_filter_logits(out.logits[:, i, :], temperature, top_p)[0] for i in range(draft_steps + 1)],
                dim=0,
            )  # [k+1, V]

            # Modified rejection sampling, left to right.
            accepted = 0
            correction: Optional[torch.Tensor] = None
            for i in range(draft_steps):
                x_i = int(drafts_tensor[0, i].item())
                p_i = float(P[i, x_i].item())
                q_i = float(Q[i, x_i].item())
                accept_prob = min(1.0, p_i / max(q_i, 1e-12))
                if float(torch.rand(1, device="cpu").item()) < accept_prob:
                    accepted += 1
                    continue
                # Reject: sample correction from (p - q)+ renormalized.
                residual = torch.clamp(P[i] - Q[i], min=0.0)
                total = float(residual.sum().item())
                if total <= 1e-12:
                    correction = _sample_from(P[i].unsqueeze(0))  # fallback to target dist
                else:
                    residual = residual / total
                    correction = _sample_from(residual.unsqueeze(0))
                break

            if accepted == draft_steps:
                bonus = _sample_from(P[draft_steps].unsqueeze(0))  # [1, 1]
                new_tokens = torch.cat([drafts_tensor, bonus], dim=1)
            else:
                assert correction is not None
                new_tokens = (
                    torch.cat([drafts_tensor[:, :accepted], correction], dim=1)
                    if accepted > 0
                    else correction
                )
                drop = draft_steps - accepted
                target_past.crop(target_past.get_seq_length() - drop)
                draft_past.crop(draft_past.get_seq_length() - drop)

            for i in range(new_tokens.size(1)):
                generated_tokens.append(new_tokens[:, i : i + 1])
                if len(generated_tokens) >= max_new_tokens:
                    break

            proposed += draft_steps
            accepted_total += accepted
            last_token = generated_tokens[-1]

    generated_tokens = generated_tokens[:max_new_tokens]
    output_ids = torch.cat([prompt_ids] + generated_tokens, dim=1)
    total_ms = t.elapsed_ms
    tps = max_new_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return {
        "output_ids": output_ids[0].tolist(),
        "total_latency_ms": round(total_ms, 3),
        "tokens_per_second": round(tps, 3),
        "draft_forward_calls": draft_forward_calls,
        "target_forward_calls": target_forward_calls,
        "draft_tokens_proposed": proposed,
        "draft_tokens_accepted": accepted_total,
        "acceptance_rate": round(accepted_total / max(proposed, 1), 4),
        "temperature": temperature,
        "top_p": top_p,
    }
