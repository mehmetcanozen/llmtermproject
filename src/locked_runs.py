from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.attention_gate import GateSettings, generate_with_attention_gate
from src.config import load_schema, validate_with_schema
from src.data_loading import read_jsonl
from src.generation import GenerationSettings, generate_repair_plus_verifier_record, generate_run_record, stable_jsonl
from src.retrieval import HybridRetriever, RetrievalDefaults, encode_texts, load_chunks, load_dense_index


ASQA_SPLITS = {
    "train_calibration_100": Path("data/asqa/splits/train_calibration_100.jsonl"),
    "dev_eval_200": Path("data/asqa/splits/dev_eval_200.jsonl"),
}
FINANCE_SPLITS = {
    "finance_full_100": Path("data/finance/generated/questions.jsonl"),
}
EXISTING_CANDIDATES = {
    ("asqa", "dev_eval_200"): Path("outputs/retrieval/asqa_candidates.jsonl"),
    ("finance", "finance_full_100"): Path("outputs/retrieval/finance_candidates.jsonl"),
}


def validate_retrieval_artifacts(repo_root: Path, dataset: str) -> dict[str, Any]:
    retrieval_dir = repo_root / "outputs" / "retrieval"
    chunks_path = retrieval_dir / f"{dataset}_chunks.jsonl"
    dense_path = retrieval_dir / f"{dataset}_dense_embeddings.npz"
    missing = [str(path.relative_to(repo_root)) for path in (chunks_path, dense_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing retrieval artifact(s) for {dataset}: {', '.join(missing)}")
    empty = [str(path.relative_to(repo_root)) for path in (chunks_path, dense_path) if path.stat().st_size <= 0]
    if empty:
        raise ValueError(f"Empty retrieval artifact(s) for {dataset}: {', '.join(empty)}. Rebuild with scripts/build_retrieval_index.py.")
    chunks = load_chunks(chunks_path)
    chunk_ids, embeddings = load_dense_index(dense_path)
    if len(chunks) != int(embeddings.shape[0]):
        raise ValueError(
            f"Retrieval artifact mismatch for {dataset}: {len(chunks)} chunks but {int(embeddings.shape[0])} embedding rows"
        )
    if len(chunk_ids) != len(chunks):
        raise ValueError(
            f"Retrieval artifact mismatch for {dataset}: {len(chunk_ids)} chunk ids but {len(chunks)} chunks"
        )
    return {
        "dataset": dataset,
        "chunks": len(chunks),
        "embedding_rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0,
    }


@dataclass(frozen=True)
class LockedRunRequest:
    system: str
    model_size: str
    dataset: str
    split: str
    output_dir: Path
    limit: int | None = None
    start: int = 0
    resume: bool = True
    prompt_passage_count: int = 3
    collect_traces: bool = False
    abort_on_gate: bool = True
    store_layer_scores: bool = False
    distractor: bool = False
    run_tag: str = "locked"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def defaults_from_config(config: dict[str, Any]) -> RetrievalDefaults:
    retrieval = config["retrieval"]
    return RetrievalDefaults(
        dense_top_k=int(retrieval["dense_top_k"]),
        bm25_top_k=int(retrieval["bm25_top_k"]),
        prompt_top_k=int(retrieval["prompt_top_k"]),
        dense_weight=float(retrieval["dense_weight"]),
        bm25_weight=float(retrieval["bm25_weight"]),
        chunk_tokens_max=int(retrieval["chunk_tokens_max"]),
        chunk_overlap_tokens=int(retrieval["chunk_overlap_tokens"]),
    )


def asqa_support_map(chunks: list[dict[str, Any]]) -> dict[str, set[str]]:
    support: dict[str, set[str]] = {}
    for chunk in chunks:
        example_id = chunk.get("source_example_id")
        if example_id:
            support.setdefault(example_id, set()).add(chunk["parent_passage_id"])
    return support


def split_path(repo_root: Path, dataset: str, split: str) -> Path:
    if dataset == "asqa":
        if split not in ASQA_SPLITS:
            raise ValueError(f"Unsupported ASQA split: {split}")
        return repo_root / ASQA_SPLITS[split]
    if dataset == "finance":
        if split not in FINANCE_SPLITS:
            raise ValueError(f"Unsupported finance split: {split}")
        return repo_root / FINANCE_SPLITS[split]
    raise ValueError(f"Unsupported dataset: {dataset}")


def query_records_from_split(repo_root: Path, dataset: str, split: str, support: dict[str, set[str]] | None = None) -> list[dict[str, Any]]:
    records = read_jsonl(split_path(repo_root, dataset, split))
    if dataset == "asqa":
        support = support or {}
        return [
            {
                "dataset": "asqa",
                "example_id": record["example_id"],
                "question": record["ambiguous_question"],
                "expected_passage_ids": sorted(support.get(record["example_id"], set())),
                "metadata": {
                    "split": split,
                    "qa_pair_count": record.get("qa_pair_count"),
                    "qa_pairs": record.get("qa_pairs", []),
                    "gold_long_answers": record.get("gold_long_answers", []),
                },
            }
            for record in records
        ]
    return [
        {
            "dataset": "finance",
            "example_id": record["example_id"],
            "question": record["question"],
            "expected_passage_ids": record.get("expected_passage_ids", []),
            "metadata": {
                "split": split,
                "question_type": record.get("question_type"),
                "answerable": record.get("answerable"),
                "company_name": record.get("company_name"),
                "period": record.get("period"),
                "metric_type": record.get("metric_type"),
            },
        }
        for record in records
    ]


def load_retriever(repo_root: Path, dataset: str, defaults: RetrievalDefaults) -> HybridRetriever:
    retrieval_dir = repo_root / "outputs" / "retrieval"
    validate_retrieval_artifacts(repo_root, dataset)
    chunks = load_chunks(retrieval_dir / f"{dataset}_chunks.jsonl")
    _, embeddings = load_dense_index(retrieval_dir / f"{dataset}_dense_embeddings.npz")
    return HybridRetriever(chunks, embeddings, defaults)


def retrieve_records(
    queries: list[dict[str, Any]],
    retriever: HybridRetriever,
    embedding_model_path: Path,
    query_instruction: str,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    embeddings = encode_texts(
        embedding_model_path,
        [query_instruction + record["question"] for record in queries],
        batch_size=batch_size,
        device=device,
    )
    outputs = []
    for query, embedding in zip(queries, embeddings):
        result = retriever.retrieve(query["question"], embedding)
        top_parent_ids = [candidate["parent_passage_id"] for candidate in result["merged_top3"]]
        expected = set(query.get("expected_passage_ids", []))
        outputs.append(
            {
                **query,
                "merged_top3": result["merged_top3"],
                "raw_dense_candidates": result["dense_candidates"],
                "raw_bm25_candidates": result["bm25_candidates"],
                "hit_top3": bool(expected and expected.intersection(top_parent_ids)),
            }
        )
    return outputs


def load_or_build_candidates(
    repo_root: Path,
    config: dict[str, Any],
    dataset: str,
    split: str,
    *,
    use_existing: bool = True,
    retrieval_batch_size: int = 64,
    retrieval_device: str = "cpu",
) -> list[dict[str, Any]]:
    existing = EXISTING_CANDIDATES.get((dataset, split))
    if use_existing and existing and (repo_root / existing).exists():
        validate_retrieval_artifacts(repo_root, dataset)
        return read_jsonl(repo_root / existing)

    defaults = defaults_from_config(config)
    retriever = load_retriever(repo_root, dataset, defaults)
    support = asqa_support_map(retriever.chunks) if dataset == "asqa" else None
    queries = query_records_from_split(repo_root, dataset, split, support)
    records = retrieve_records(
        queries,
        retriever,
        Path(config["retrieval"]["local_embedding_path"]),
        config["retrieval"].get("query_instruction", ""),
        retrieval_batch_size,
        retrieval_device,
    )
    if existing:
        output_path = repo_root / existing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(stable_jsonl(records), encoding="utf-8")
    return records


def choose_distractor(record: dict[str, Any]) -> dict[str, Any]:
    expected = set(record.get("expected_passage_ids") or [])
    prompt_ids = {candidate["parent_passage_id"] for candidate in record.get("merged_top3", [])}
    for source_key in ("raw_dense_candidates", "raw_bm25_candidates"):
        for candidate in record.get(source_key, []) or []:
            passage_id = candidate.get("parent_passage_id")
            if passage_id and passage_id not in expected and passage_id not in prompt_ids:
                return {**candidate, "distractor_source_list": source_key}
    for candidate in record.get("merged_top3", []) or []:
        passage_id = candidate.get("parent_passage_id")
        if passage_id and passage_id not in expected:
            return {**candidate, "distractor_source_list": "merged_top3_fallback"}
    fallback = (record.get("raw_dense_candidates") or record.get("raw_bm25_candidates") or record.get("merged_top3") or [None])[0]
    if fallback:
        return {**fallback, "distractor_source_list": "fallback_first_available"}
    raise ValueError(f"No prompt candidate available for {record.get('example_id')}")


def with_optional_distractor(record: dict[str, Any], enabled: bool) -> dict[str, Any]:
    if not enabled:
        return record
    distractor = choose_distractor(record)
    distractor = {**distractor, "score": max(0.0, min(1.0, float(distractor.get("score", 0.0))))}
    merged = list(record["merged_top3"][:3]) + [distractor]
    return {
        **record,
        "merged_top3": merged,
        "metadata": {
            **record.get("metadata", {}),
            "distractor_enabled": True,
            "distractor_parent_passage_id": distractor.get("parent_passage_id"),
            "distractor_source_list": distractor.get("distractor_source_list"),
        },
    }


def selected_candidates(candidates: list[dict[str, Any]], *, start: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    end = None if limit is None else start + limit
    return candidates[start:end]


def prediction_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record["example_id"] for record in read_jsonl(path)}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def validate_prediction(record: dict[str, Any]) -> None:
    validate_with_schema(record, load_schema("run_output.schema.json"))
    if not record.get("metadata", {}).get("final_format_valid", True):
        raise ValueError(f"Final answer failed format validation for {record['example_id']}")


def request_run_id(request: LockedRunRequest) -> str:
    suffixes = [request.system, request.dataset, request.split, request.model_size, request.run_tag]
    if request.distractor:
        suffixes.append("distractor")
    return "_".join(suffixes)


def run_locked_generation(
    *,
    repo_root: Path,
    config_path: Path,
    config: dict[str, Any],
    request: LockedRunRequest,
    tokenizer: Any,
    model: Any,
    generation_settings: GenerationSettings,
    gate_settings: GateSettings | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    run_id = request_run_id(request)
    output_dir = request.output_dir if request.output_dir.is_absolute() else repo_root / request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    statuses_path = output_dir / "statuses.jsonl"
    traces_path = output_dir / "attention_traces.jsonl"
    selected = [
        with_optional_distractor(candidate, request.distractor)
        for candidate in selected_candidates(candidates, start=request.start, limit=request.limit)
    ]
    already_done = read_existing_ids(predictions_path) if request.resume else set()
    if not request.resume:
        for path in (predictions_path, statuses_path, traces_path):
            if path.exists():
                path.unlink()

    started_at = time.perf_counter()
    completed = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    finance_gold: dict[str, dict[str, Any]] = {}
    asqa_gold: dict[str, dict[str, Any]] = {}
    if request.system == "repair_plus_verifier":
        from src.verifier import load_asqa_gold, load_finance_gold

        finance_gold = load_finance_gold(read_jsonl(repo_root / FINANCE_SPLITS["finance_full_100"]))
        asqa_gold = load_asqa_gold(
            read_jsonl(repo_root / ASQA_SPLITS["train_calibration_100"])
            + read_jsonl(repo_root / ASQA_SPLITS["dev_eval_200"])
        )
    for index, candidate in enumerate(selected, start=request.start):
        if candidate["example_id"] in already_done:
            skipped += 1
            continue
        item_started = time.perf_counter()
        try:
            if request.system == "baseline":
                prediction = generate_run_record(
                    tokenizer,
                    model,
                    candidate,
                    generation_settings,
                    run_id,
                    "baseline",
                    prompt_passage_count=request.prompt_passage_count,
                )
                trace = None
            elif request.system == "gate_only":
                if gate_settings is None:
                    raise ValueError("gate_settings are required for gate_only")
                prediction, trace = generate_with_attention_gate(
                    tokenizer,
                    model,
                    candidate,
                    generation_settings,
                    gate_settings,
                    run_id,
                    abort_on_gate=request.abort_on_gate,
                    store_layer_scores=request.store_layer_scores,
                    prompt_passage_count=request.prompt_passage_count,
                )
            elif request.system == "repair_plus_verifier":
                prediction = generate_repair_plus_verifier_record(
                    tokenizer,
                    model,
                    candidate,
                    generation_settings,
                    run_id,
                    finance_gold=finance_gold,
                    asqa_gold=asqa_gold,
                    prompt_passage_count=request.prompt_passage_count,
                )
                trace = None
            else:
                raise ValueError(f"Unsupported generation system: {request.system}")
            validate_prediction(prediction)
            elapsed = time.perf_counter() - item_started
            append_jsonl(predictions_path, prediction)
            if trace is not None and request.collect_traces:
                append_jsonl(traces_path, trace)
            append_jsonl(
                statuses_path,
                {
                    "example_id": candidate["example_id"],
                    "dataset": candidate["dataset"],
                    "status": "completed",
                    "index": index,
                    "elapsed_seconds": round(elapsed, 3),
                    "created_at": utc_now(),
                },
            )
            completed += 1
        except Exception as exc:  # pragma: no cover - exercised by runtime failures
            failure = {
                "example_id": candidate.get("example_id"),
                "dataset": candidate.get("dataset"),
                "status": "failed",
                "index": index,
                "error": f"{type(exc).__name__}: {exc}",
                "created_at": utc_now(),
            }
            failures.append(failure)
            append_jsonl(statuses_path, failure)
            raise

    final_predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    expected_ids = [candidate["example_id"] for candidate in selected]
    full_split_ids = [candidate["example_id"] for candidate in candidates]
    completed_ids = [record["example_id"] for record in final_predictions if record["example_id"] in set(expected_ids)]
    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "phase": "07_Calibration_and_Evaluation" if request.split == "train_calibration_100" else "08_Experiments_Figures_and_Report_Assets",
        "system": request.system,
        "model": {
            "model_id": generation_settings.model_id,
            "local_path": generation_settings.model_path,
            "quantized": generation_settings.load_in_4bit,
            "attn_implementation": generation_settings.attn_implementation,
        },
        "dataset": {"name": request.dataset, "split": request.split, "example_count": len(selected)},
        "config_path": str(config_path),
        "seed": int(config["project"]["seed"]),
        "input_paths": [str(split_path(repo_root, request.dataset, request.split).relative_to(repo_root))],
        "output_paths": [
            str(predictions_path.relative_to(repo_root)),
            str(statuses_path.relative_to(repo_root)),
        ] + ([str(traces_path.relative_to(repo_root))] if traces_path.exists() else []),
        "notes": [
            f"prompt_passage_count={request.prompt_passage_count}",
            f"limit={request.limit}",
            f"start={request.start}",
            f"distractor={request.distractor}",
            f"abort_on_gate={request.abort_on_gate}",
        ],
        "generation_settings": asdict(generation_settings),
        "gate_settings": None if gate_settings is None else asdict(gate_settings),
        "requested_example_ids": expected_ids,
        "full_split_example_count": len(full_split_ids),
        "selected_example_count": len(expected_ids),
        "completed_example_ids": sorted(set(completed_ids)),
        "completed_count": len(set(completed_ids)),
        "skipped_existing_count": skipped,
        "new_completed_count": completed,
        "failure_count": len(failures),
        "selected_slice_complete": set(completed_ids) >= set(expected_ids),
        "formal_split_complete": request.limit is None and request.start == 0 and set(completed_ids) >= set(full_split_ids),
        "predictions_sha256": prediction_sha256(predictions_path),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    validate_with_schema(
        {key: manifest[key] for key in ("run_id", "created_at", "phase", "system", "model", "dataset", "config_path", "seed", "input_paths", "output_paths", "notes")},
        load_schema("run_manifest.schema.json"),
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def stable_records_hash(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(stable_jsonl(records).encode("utf-8")).hexdigest()
