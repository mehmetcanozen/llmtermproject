from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_CONFIG_PATH, load_config, load_schema
from src.data_loading import ensure_list, normalize_nested, text_key, validate_records, write_json, write_jsonl


SEED = 20260416


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def clean_qa_pair(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": pair.get("question") or "",
        "context": pair.get("context"),
        "short_answers": [str(answer) for answer in ensure_list(pair.get("short_answers")) if answer is not None],
        "wikipage": pair.get("wikipage"),
    }


def clean_wikipage(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": page.get("title"),
        "url": page.get("url"),
    }


def clean_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    knowledge = []
    for item in ensure_list(annotation.get("knowledge")):
        if not isinstance(item, dict):
            continue
        knowledge.append(
            {
                "content": item.get("content"),
                "wikipage": item.get("wikipage"),
            }
        )
    return {
        "long_answer": annotation.get("long_answer"),
        "knowledge": knowledge,
    }


def normalize_split(split: str, parquet_path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(parquet_path)
    records: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        row_dict = normalize_nested(row.to_dict())
        qa_pairs = [clean_qa_pair(pair) for pair in ensure_list(row_dict.get("qa_pairs")) if isinstance(pair, dict)]
        wikipages = [clean_wikipage(page) for page in ensure_list(row_dict.get("wikipages")) if isinstance(page, dict)]
        annotations = [
            clean_annotation(annotation)
            for annotation in ensure_list(row_dict.get("annotations"))
            if isinstance(annotation, dict)
        ]
        gold_long_answers = [
            annotation["long_answer"]
            for annotation in annotations
            if isinstance(annotation.get("long_answer"), str) and annotation["long_answer"].strip()
        ]
        sample_id = str(row_dict.get("sample_id"))
        records.append(
            {
                "record_type": "asqa_example",
                "example_id": f"asqa_{split}_{row_index:05d}",
                "source_split": split,
                "source_row_index": int(row_index),
                "sample_id": sample_id,
                "ambiguous_question": row_dict.get("ambiguous_question") or "",
                "qa_pair_count": len(qa_pairs),
                "qa_pairs": qa_pairs,
                "wikipages": wikipages,
                "annotations": annotations,
                "gold_long_answers": gold_long_answers,
            }
        )
    return records


def build_passages(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicate_count = 0
    skipped_empty = 0
    for record in records:
        for annotation_index, annotation in enumerate(record["annotations"]):
            for knowledge_index, knowledge in enumerate(annotation["knowledge"]):
                title = knowledge.get("wikipage")
                text = knowledge.get("content")
                if not isinstance(text, str) or not text.strip():
                    skipped_empty += 1
                    continue
                dedup_key = f"{text_key(title)}::{text_key(text)}"
                if dedup_key in seen:
                    duplicate_count += 1
                    continue
                passage_id = f"asqa_passage_{len(passages):06d}_{stable_hash(dedup_key)}"
                seen[dedup_key] = passage_id
                passages.append(
                    {
                        "record_type": "asqa_passage",
                        "passage_id": passage_id,
                        "source_example_id": record["example_id"],
                        "source_split": record["source_split"],
                        "source_row_index": record["source_row_index"],
                        "title": title,
                        "text": " ".join(text.split()),
                        "provenance": {
                            "annotation_index": annotation_index,
                            "knowledge_index": knowledge_index,
                            "dedup_key": stable_hash(dedup_key),
                        },
                    }
                )
    stats = {
        "dedup_rule": "lowercase whitespace-normalized title plus lowercase whitespace-normalized text",
        "duplicate_passages_skipped": duplicate_count,
        "empty_knowledge_items_skipped": skipped_empty,
        "unique_passages": len(passages),
    }
    return passages, stats


def qa_bucket(record: dict[str, Any]) -> str:
    count = int(record["qa_pair_count"])
    if 2 <= count <= 3:
        return "2-3"
    if 4 <= count <= 5:
        return "4-5"
    return "6+"


def sample_records(records: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = sorted(rng.sample(records, size), key=lambda record: record["example_id"])
    return selected


def stratified_dev_eval(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    targets = {"2-3": 100, "4-5": 70, "6+": 30}
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for bucket, target in targets.items():
        candidates = [record for record in records if qa_bucket(record) == bucket]
        if len(candidates) < target:
            raise RuntimeError(f"Not enough ASQA dev examples for bucket {bucket}: {len(candidates)} < {target}")
        selected.extend(rng.sample(candidates, target))
    return sorted(selected, key=lambda record: record["example_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize ASQA parquet files and freeze Phase 02 splits.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    config = load_config(args.config)
    train_path = Path(config["paths"]["asqa_train_parquet"])
    dev_path = Path(config["paths"]["asqa_dev_parquet"])

    train_records = normalize_split("train", train_path)
    dev_records = normalize_split("dev", dev_path)
    all_records = train_records + dev_records
    passages, passage_stats = build_passages(all_records)

    calibration = sample_records(train_records, 100, args.seed)
    dev_eval = stratified_dev_eval(dev_records, args.seed)

    schema = load_schema("asqa_record.schema.json")
    validate_records(train_records, schema, "asqa train normalized")
    validate_records(dev_records, schema, "asqa dev normalized")
    validate_records(passages, schema, "asqa passage")

    paths = {
        "train_full": REPO_ROOT / "data" / "asqa" / "normalized" / "train_full.jsonl",
        "dev_full": REPO_ROOT / "data" / "asqa" / "normalized" / "dev_full.jsonl",
        "passages": REPO_ROOT / "data" / "asqa" / "corpus" / "passages.jsonl",
        "calibration": REPO_ROOT / "data" / "asqa" / "splits" / "train_calibration_100.jsonl",
        "dev_eval": REPO_ROOT / "data" / "asqa" / "splits" / "dev_eval_200.jsonl",
        "manifest": REPO_ROOT / "data" / "asqa" / "manifests" / "dataset_manifest.json",
    }
    write_jsonl(paths["train_full"], train_records)
    write_jsonl(paths["dev_full"], dev_records)
    write_jsonl(paths["passages"], passages)
    write_jsonl(paths["calibration"], calibration)
    write_jsonl(paths["dev_eval"], dev_eval)

    manifest = {
        "dataset": "asqa",
        "phase": "02_Data_Pipelines_and_Dataset_Contracts",
        "seed": args.seed,
        "source_files": {
            "train_parquet": str(train_path),
            "dev_parquet": str(dev_path),
        },
        "observed_schema": {
            "columns": ["ambiguous_question", "qa_pairs", "wikipages", "annotations", "sample_id"],
            "nested_fields": {
                "qa_pairs": ["context", "question", "short_answers", "wikipage"],
                "wikipages": ["title", "url"],
                "annotations": ["knowledge", "long_answer"],
                "annotations.knowledge": ["content", "wikipage"],
            },
        },
        "counts": {
            "train_examples": len(train_records),
            "dev_examples": len(dev_records),
            "corpus_passages": len(passages),
            "train_calibration_100": len(calibration),
            "dev_eval_200": len(dev_eval),
            "dev_eval_buckets": dict(Counter(qa_bucket(record) for record in dev_eval)),
        },
        "passage_build": passage_stats,
        "schemas": {
            "asqa_record": "configs/schemas/asqa_record.schema.json",
        },
        "outputs": {key: str(path.relative_to(REPO_ROOT)) for key, path in paths.items() if key != "manifest"},
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps({"status": "passed", "manifest": str(paths["manifest"]), "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
