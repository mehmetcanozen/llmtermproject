from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = Path(r"C:\AI\models")
DEFAULT_DATA_ROOT = Path(r"C:\AI\data")
DEFAULT_ALCE_PATH = REPO_ROOT / "external" / "ALCE_reference"


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def find_model_dir(models_root: Path, markers: list[str], required_files: list[str]) -> Path:
    require(models_root.exists(), f"Models root not found: {models_root}")
    normalized_markers = [normalize_name(marker) for marker in markers]
    candidates: list[Path] = []
    for child in models_root.iterdir():
        if not child.is_dir():
            continue
        normalized_child = normalize_name(child.name)
        if all(marker in normalized_child for marker in normalized_markers):
            if all((child / file_name).exists() for file_name in required_files):
                candidates.append(child)
    require(bool(candidates), f"Could not find a model directory under {models_root} for markers {markers}")
    return sorted(candidates)[0]


def find_asqa_files(data_root: Path) -> tuple[Path, Path]:
    require(data_root.exists(), f"Data root not found: {data_root}")
    train_candidates = sorted(data_root.rglob("train-*.parquet"))
    dev_candidates = sorted(data_root.rglob("dev-*.parquet"))
    require(bool(train_candidates), f"Could not find ASQA train parquet under {data_root}")
    require(bool(dev_candidates), f"Could not find ASQA dev parquet under {data_root}")
    return train_candidates[0], dev_candidates[0]


def torch_bitsandbytes_smoke() -> dict:
    import bitsandbytes as bnb
    import torch

    require(torch.cuda.is_available(), "CUDA is not available")
    props = torch.cuda.get_device_properties(0)
    result = {
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "bitsandbytes_version": bnb.__version__,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
    }
    require(result["compute_capability"] == "12.0", f"Expected compute capability 12.0, got {result['compute_capability']}")
    return result


def qwen_smoke(model_path: Path, use_4bit: bool) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="eager",
    )

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply with exactly five words about citations."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=12,
        do_sample=False,
        temperature=0.0,
    )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    require(bool(decoded), "Qwen 3B smoke test produced empty output")
    return {
        "model_path": str(model_path),
        "used_4bit": use_4bit,
        "generated_text": decoded,
    }


def embedding_smoke(model_path: Path) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path))
    embedding = model.encode(["citation support check"], normalize_embeddings=True)
    shape = list(embedding.shape)
    require(shape[0] == 1, f"Expected one embedding row, got {shape}")
    return {
        "model_path": str(model_path),
        "embedding_shape": shape,
    }


def asqa_smoke(train_path: Path, dev_path: Path) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(train_path),
            "dev": str(dev_path),
        },
    )
    train_count = len(dataset["train"])
    dev_count = len(dataset["dev"])
    require(train_count > 0, "ASQA train split is empty")
    require(dev_count > 0, "ASQA dev split is empty")
    return {
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "train_examples": train_count,
        "dev_examples": dev_count,
    }


def alce_reference_smoke(alce_path: Path) -> dict:
    require(alce_path.exists(), f"ALCE reference path not found: {alce_path}")
    required_paths = [
        alce_path / "README.md",
        alce_path / "eval.py",
        alce_path / "configs",
        alce_path / "prompts",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    require(not missing, f"ALCE reference folder is missing expected files: {missing}")
    return {
        "alce_path": str(alce_path),
        "has_git_dir": (alce_path / ".git").exists(),
    }


def print_result(name: str, payload: dict) -> None:
    print(f"[PASS] {name}")
    for key, value in payload.items():
        print(f"  - {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local preflight smoke checks for the project.")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--alce-path", type=Path, default=DEFAULT_ALCE_PATH)
    parser.add_argument("--qwen3b-path", type=Path)
    parser.add_argument("--bge-path", type=Path)
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--dev-parquet", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--no-4bit", action="store_true", help="Load the Qwen smoke model without 4-bit quantization.")
    args = parser.parse_args()

    results: dict[str, dict] = {}
    try:
        qwen3b_path = args.qwen3b_path or find_model_dir(
            args.models_root,
            markers=["qwen", "2.5", "3b", "instruct"],
            required_files=["config.json", "generation_config.json", "tokenizer.json"],
        )
        bge_path = args.bge_path or find_model_dir(
            args.models_root,
            markers=["bge", "small", "en", "v1.5"],
            required_files=["config.json", "modules.json"],
        )
        train_path, dev_path = (
            (args.train_parquet, args.dev_parquet)
            if args.train_parquet and args.dev_parquet
            else find_asqa_files(args.data_root)
        )

        discovered = {
            "qwen3b_path": str(qwen3b_path),
            "bge_path": str(bge_path),
            "train_parquet": str(train_path),
            "dev_parquet": str(dev_path),
            "alce_path": str(args.alce_path),
        }
        print("[INFO] Discovered local assets")
        for key, value in discovered.items():
            print(f"  - {key}: {value}")
        results["discovered_paths"] = discovered

        results["torch_bitsandbytes"] = torch_bitsandbytes_smoke()
        print_result("Torch and bitsandbytes smoke", results["torch_bitsandbytes"])

        results["qwen3b_generation"] = qwen_smoke(qwen3b_path, use_4bit=not args.no_4bit)
        print_result("Qwen 3B generation smoke", results["qwen3b_generation"])

        results["embedding_model"] = embedding_smoke(bge_path)
        print_result("Embedding model smoke", results["embedding_model"])

        results["asqa_dataset"] = asqa_smoke(train_path, dev_path)
        print_result("ASQA parquet smoke", results["asqa_dataset"])

        results["alce_reference"] = alce_reference_smoke(args.alce_path)
        print_result("ALCE reference smoke", results["alce_reference"])

        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"[INFO] Wrote JSON report to {args.json_out}")

        print("[PASS] All preflight smoke checks completed successfully")
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": "failed",
                "error": str(exc),
                "partial_results": results,
            }
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[INFO] Wrote partial JSON report to {args.json_out}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
