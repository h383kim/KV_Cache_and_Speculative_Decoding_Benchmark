from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    name: str
    dtype: torch.dtype
    device: torch.device


def load_model_and_tokenizer(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device).eval()
    return LoadedModel(model=model, tokenizer=tokenizer, name=name, dtype=dtype, device=device)
