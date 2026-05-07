from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
SCHEMA_DIR = REPO_ROOT / "configs" / "schemas"

REQUIRED_CONFIG_SECTIONS = (
    "project",
    "paths",
    "model",
    "retrieval",
    "prompt",
    "gate",
    "verifier",
    "evaluation",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {config_path}")
    return payload


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(path)
    missing = [section for section in REQUIRED_CONFIG_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Config is missing required sections: {missing}")
    return config


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in file: {json_path}")
    return payload


def load_schema(name: str) -> dict[str, Any]:
    return load_json(SCHEMA_DIR / name)


def validate_with_schema(instance: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors)
        raise ValueError(details)


def sample_run_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": "contract_validation_phase01",
        "created_at": "2026-04-25T00:00:00+03:00",
        "phase": "01_Repo_Scaffold_and_Config",
        "system": "validation",
        "model": {
            "model_id": config["model"]["bringup_model_id"],
            "local_path": config["model"]["local_bringup_path"],
            "quantized": bool(config["model"]["quantization"]["load_in_4bit"]),
            "attn_implementation": config["model"]["attention"]["implementation"],
        },
        "dataset": {
            "name": "contract_sample",
            "split": "none",
            "example_count": 1,
        },
        "config_path": "configs/default.yaml",
        "seed": int(config["project"]["seed"]),
        "input_paths": ["configs/default.yaml"],
        "output_paths": ["artifacts/phase01/validation_sample.json"],
        "notes": ["Schema contract sample used by scripts/validate_config.py."],
    }


def sample_run_output(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": "contract_validation_phase01",
        "example_id": "sample_asqa_0001",
        "dataset": "asqa",
        "system": "baseline",
        "question": "Which source supports the answer?",
        "retrieved_passages": [
            {
                "label": "P1",
                "passage_id": "asqa_train_sample_0001",
                "text": "The cited passage contains the answer used by the model.",
                "rank": 1,
                "score": 1.0,
            }
        ],
        "answer": "The answer is supported by the retrieved passage [P1].",
        "abstained": False,
        "citations": [{"sentence_index": 0, "labels": ["P1"]}],
        "generation": {
            "model_id": config["model"]["bringup_model_id"],
            "temperature": float(config["prompt"]["temperature"]),
            "do_sample": bool(config["prompt"]["do_sample"]),
            "max_new_tokens": int(config["prompt"]["max_new_tokens"]),
        },
        "metadata": {"contract_sample": True},
    }


def validate_config_contract(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = sample_run_manifest(config)
    output = sample_run_output(config)
    validate_with_schema(manifest, load_schema("run_manifest.schema.json"))
    validate_with_schema(output, load_schema("run_output.schema.json"))
    return {
        "config_sections": list(config.keys()),
        "manifest_run_id": manifest["run_id"],
        "output_example_id": output["example_id"],
    }
