from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm_inference_lab.sweeps.configs import (
    KVSweepConfig,
    SpecSweepConfig,
    load_kv_sweep,
    load_spec_sweep,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_kv_sweep_yaml_roundtrips():
    cfg = load_kv_sweep(REPO_ROOT / "configs" / "kv_sweep.yaml")
    assert cfg.model == "distilgpt2"
    assert cfg.prompt_lengths == [64, 128, 256]
    assert cfg.use_cache_options == [True, False]


def test_spec_sweep_yaml_roundtrips():
    cfg = load_spec_sweep(REPO_ROOT / "configs" / "spec_sweep.yaml")
    assert cfg.target_model == "gpt2"
    assert cfg.draft_model == "distilgpt2"
    assert len(cfg.prompts) == 3


def test_kv_sweep_rejects_missing_required(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"model": "gpt2"}))  # missing prompt_lengths/generation_lengths
    with pytest.raises(ValidationError):
        load_kv_sweep(bad)


def test_kv_sweep_rejects_empty_lists():
    with pytest.raises(ValidationError):
        KVSweepConfig(
            model="gpt2",
            prompt_lengths=[],
            generation_lengths=[16],
        )


def test_spec_sweep_rejects_bad_types(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({
            "target_model": "gpt2",
            "draft_model": "distilgpt2",
            "prompts": ["a"],
            "draft_steps": "not a list",
            "max_new_tokens": [32],
        })
    )
    with pytest.raises(ValidationError):
        load_spec_sweep(bad)
