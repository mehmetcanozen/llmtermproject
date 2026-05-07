from __future__ import annotations

import random

import numpy as np

from src.config import load_config, load_schema, sample_run_manifest, sample_run_output, validate_with_schema
from src.utils.determinism import set_global_seed


def test_default_config_loads_required_sections() -> None:
    config = load_config()
    for section in ("project", "paths", "model", "retrieval", "prompt", "gate", "verifier", "evaluation"):
        assert section in config


def test_sample_manifest_validates_against_schema() -> None:
    config = load_config()
    manifest = sample_run_manifest(config)
    validate_with_schema(manifest, load_schema("run_manifest.schema.json"))


def test_sample_output_validates_against_schema() -> None:
    config = load_config()
    output = sample_run_output(config)
    validate_with_schema(output, load_schema("run_output.schema.json"))


def test_seed_helper_sets_python_numpy_and_torch_without_crashing() -> None:
    report = set_global_seed(1234)
    first_random = random.random()
    first_numpy = float(np.random.random())
    set_global_seed(1234)
    assert random.random() == first_random
    assert float(np.random.random()) == first_numpy
    assert report.seed == 1234
    assert report.python_hash_seed == "1234"
    assert report.numpy_seeded is True
