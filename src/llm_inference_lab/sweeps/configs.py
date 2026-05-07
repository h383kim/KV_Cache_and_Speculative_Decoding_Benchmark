from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field


class KVSweepConfig(BaseModel):
    model: str
    dtype: str = "fp32"
    prompt_lengths: List[int] = Field(..., min_length=1)
    generation_lengths: List[int] = Field(..., min_length=1)
    batch_sizes: List[int] = Field(default=[1], min_length=1)
    use_cache_options: List[bool] = Field(default=[True], min_length=1)
    output_dir: str = "results/kv_sweep"
    seed: int = 0


class SpecSweepConfig(BaseModel):
    target_model: str
    draft_model: str
    prompts: List[str] = Field(..., min_length=1)
    draft_steps: List[int] = Field(..., min_length=1)
    max_new_tokens: List[int] = Field(..., min_length=1)
    dtype: str = "fp32"
    output_dir: str = "results/spec_sweep"
    seed: int = 0


def load_kv_sweep(path: str | Path) -> KVSweepConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return KVSweepConfig(**raw)


def load_spec_sweep(path: str | Path) -> SpecSweepConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return SpecSweepConfig(**raw)
