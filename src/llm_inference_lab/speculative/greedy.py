from __future__ import annotations

from typing import Any, Dict

import torch

from ..metrics.latency import Timer
from ..models.loader import LoadedModel


def _check_compatible_tokenizers(target: LoadedModel, draft: LoadedModel) -> None:
    if target.model.config.vocab_size != draft.model.config.vocab_size:
        raise ValueError(
            "target and draft vocab sizes differ "
            f"({target.model.config.vocab_size} vs {draft.model.config.vocab_size}); "
            "speculative decoding requires a shared tokenizer."
        )
    probe = "Hello, world! 12345"
    a = target.tokenizer.encode(probe, add_special_tokens=False)
    b = draft.tokenizer.encode(probe, add_special_tokens=False)
    if a != b:
        raise ValueError(
            "target and draft tokenizers disagree on a probe string; "
            "speculative decoding requires identical tokenization."
        )


@torch.no_grad()
def greedy_baseline(
    target: LoadedModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Manual greedy decode using only the target model. Used as the speedup baseline."""
    device = target.device
    model = target.model
    target_forward_calls = 0

    with Timer(device) as t:
        out = model(input_ids=prompt_ids, use_cache=True)
        target_forward_calls += 1
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            target_forward_calls += 1
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
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
def speculative_greedy(
    target: LoadedModel,
    draft: LoadedModel,
    prompt_ids: torch.Tensor,
    draft_steps: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Greedy speculative decoding (Leviathan et al.-style, deterministic verification).

    Restricted to batch_size=1 for the MVP — different examples within a batch can
    accept different prefix lengths, which complicates cache cropping.
    """
    if prompt_ids.size(0) != 1:
        raise ValueError("speculative_greedy supports batch_size=1 only in the MVP")
    _check_compatible_tokenizers(target, draft)

    device = target.device
    t_model = target.model
    d_model = draft.model

    target_forward_calls = 0
    draft_forward_calls = 0
    proposed = 0
    accepted_total = 0

    with Timer(device) as t:
        # Prefill target on the prompt — out.logits[:, -1] gives our first generated token.
        out = t_model(input_ids=prompt_ids, use_cache=True)
        target_forward_calls += 1
        target_past = out.past_key_values  # length P
        first_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]

        # Prefill draft on the prompt so it has matching cache.
        out_d = d_model(input_ids=prompt_ids, use_cache=True)
        draft_forward_calls += 1
        draft_past = out_d.past_key_values  # length P

        generated_tokens = [first_token]  # list of [1, 1] tensors
        last_token = first_token  # not yet seen by either KV cache

        while len(generated_tokens) < max_new_tokens:
            # Phase 1: draft proposes draft_steps tokens autoregressively.
            # We then run one extra forward pass on drafts[-1] so that ALL k drafts
            # (including the last one) end up in the draft KV cache. This makes the
            # crop math uniform across the all-accepted and partial-accept cases.
            drafts = []
            cur = last_token
            for _ in range(draft_steps):
                out_d = d_model(input_ids=cur, past_key_values=draft_past, use_cache=True)
                draft_forward_calls += 1
                draft_past = out_d.past_key_values
                cur = out_d.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                drafts.append(cur)
            # Place drafts[-1] into the draft cache (prediction is discarded).
            out_d = d_model(input_ids=cur, past_key_values=draft_past, use_cache=True)
            draft_forward_calls += 1
            draft_past = out_d.past_key_values
            drafts_tensor = torch.cat(drafts, dim=1)  # [1, k]

            # Phase 2: target verifies in one forward pass.
            # Feed [last_token, drafts[0], ..., drafts[k-1]] (k+1 tokens) so we get
            # k+1 predictions, one for each verification slot plus a bonus.
            target_input = torch.cat([last_token, drafts_tensor], dim=1)  # [1, k+1]
            out = t_model(input_ids=target_input, past_key_values=target_past, use_cache=True)
            target_forward_calls += 1
            target_past = out.past_key_values  # length P + (current generated) + (k+1)
            target_argmax = out.logits.argmax(dim=-1)  # [1, k+1]

            match = target_argmax[:, :draft_steps] == drafts_tensor  # [1, k]
            match_row = match[0]

            if bool(match_row.all()):
                accepted = draft_steps
                bonus = target_argmax[:, draft_steps : draft_steps + 1]  # [1, 1]
                new_tokens = torch.cat([drafts_tensor, bonus], dim=1)  # [1, k+1]
            else:
                accepted = int(torch.argmin(match_row.int()).item())  # first False
                correction = target_argmax[:, accepted : accepted + 1]  # [1, 1]
                new_tokens = (
                    torch.cat([drafts_tensor[:, :accepted], correction], dim=1)
                    if accepted > 0
                    else correction
                )
                # Crop both caches back to the prefix that was actually accepted.
                # Target forward added k+1 positions (last_token + all k drafts); we want to keep
                # last_token + drafts[0..accepted-1], i.e. (accepted+1) of those. Drop = k - accepted.
                # Draft cache (after the extra pass that wrote drafts[k-1]) has the same shape:
                # last_token + drafts[0..k-1], same drop.
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

            # If we just appended the correction/bonus, that token has NOT been processed by the
            # draft model yet. The draft cache currently ends at the last accepted draft (or at the
            # prompt if accepted==0). On the next iteration we'll feed `last_token` (the correction
            # or bonus) into the draft, which extends the cache correctly.

    generated_tokens = generated_tokens[:max_new_tokens]
    output_ids = torch.cat([prompt_ids] + generated_tokens, dim=1)
    total_ms = t.elapsed_ms
    tps = max_new_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0
    acceptance_rate = accepted_total / max(proposed, 1)

    return {
        "output_ids": output_ids[0].tolist(),
        "total_latency_ms": round(total_ms, 3),
        "tokens_per_second": round(tps, 3),
        "draft_forward_calls": draft_forward_calls,
        "target_forward_calls": target_forward_calls,
        "draft_tokens_proposed": proposed,
        "draft_tokens_accepted": accepted_total,
        "acceptance_rate": round(acceptance_rate, 4),
    }
