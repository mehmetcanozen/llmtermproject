from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_CONFIG_PATH, load_config, load_schema, validate_with_schema
from src.data_loading import read_jsonl, write_json
from src.generation import generate_run_record, generation_settings_from_config, load_generation_model, stable_jsonl
from src.retrieval import HybridRetriever, RetrievalDefaults, encode_texts, load_chunks, load_dense_index
from src.utils.determinism import set_global_seed


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


def build_smoke_candidates(config: dict[str, Any], batch_size: int, device: str) -> list[dict[str, Any]]:
    out_dir = REPO_ROOT / "outputs" / "retrieval"
    defaults = defaults_from_config(config)
    embedding_model_path = Path(config["retrieval"]["local_embedding_path"])
    query_instruction = config["retrieval"].get("query_instruction", "")

    asqa_chunks = load_chunks(out_dir / "asqa_chunks.jsonl")
    _, asqa_embeddings = load_dense_index(out_dir / "asqa_dense_embeddings.npz")
    asqa_retriever = HybridRetriever(asqa_chunks, asqa_embeddings, defaults)
    support = asqa_support_map(asqa_chunks)
    asqa_calibration = read_jsonl(REPO_ROOT / "data" / "asqa" / "splits" / "train_calibration_100.jsonl")[:10]
    asqa_queries = [
        {
            "dataset": "asqa",
            "example_id": record["example_id"],
            "question": record["ambiguous_question"],
            "expected_passage_ids": sorted(support.get(record["example_id"], set())),
            "metadata": {"qa_pair_count": record["qa_pair_count"], "split": "train_calibration_100"},
        }
        for record in asqa_calibration
    ]
    asqa_candidates = retrieve_records(asqa_queries, asqa_retriever, embedding_model_path, query_instruction, batch_size, device)

    finance_chunks = load_chunks(out_dir / "finance_chunks.jsonl")
    _, finance_embeddings = load_dense_index(out_dir / "finance_dense_embeddings.npz")
    finance_retriever = HybridRetriever(finance_chunks, finance_embeddings, defaults)
    finance_questions = read_jsonl(REPO_ROOT / "data" / "finance" / "generated" / "questions.jsonl")[:10]
    finance_queries = [
        {
            "dataset": "finance",
            "example_id": record["example_id"],
            "question": record["question"],
            "expected_passage_ids": record["expected_passage_ids"],
            "metadata": {
                "question_type": record["question_type"],
                "answerable": record["answerable"],
                "company_name": record["company_name"],
                "period": record["period"],
                "metric_type": record["metric_type"],
            },
        }
        for record in finance_questions
    ]
    finance_candidates = retrieve_records(
        finance_queries, finance_retriever, embedding_model_path, query_instruction, batch_size, device
    )
    return asqa_candidates + finance_candidates


def write_jsonl_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_format_report(records: list[dict[str, Any]], repeat_matched: bool, output_path: Path) -> dict[str, Any]:
    raw_failures = [record for record in records if not record["metadata"]["raw_format_valid"]]
    final_failures = [record for record in records if not record["metadata"]["final_format_valid"]]
    abstentions = [record for record in records if record["abstained"]]
    summary = {
        "example_count": len(records),
        "repeat_matched": repeat_matched,
        "raw_format_failures": len(raw_failures),
        "final_format_failures": len(final_failures),
        "final_abstentions": len(abstentions),
        "raw_failure_rate": len(raw_failures) / len(records) if records else 0.0,
        "final_failure_rate": len(final_failures) / len(records) if records else 0.0,
    }
    lines = [
        "# Baseline Smoke Format Report",
        "",
        f"- Examples: `{summary['example_count']}`",
        f"- Repeated run byte-for-byte match: `{repeat_matched}`",
        f"- Raw format failures: `{summary['raw_format_failures']}`",
        f"- Final format failures after deterministic format fallback: `{summary['final_format_failures']}`",
        f"- Final abstentions: `{summary['final_abstentions']}`",
        "",
        "## Smoke Set IDs",
        "",
    ]
    for record in records:
        lines.append(
            f"- `{record['dataset']}` `{record['example_id']}` raw_valid=`{record['metadata']['raw_format_valid']}` abstained=`{record['abstained']}`"
        )
    if raw_failures:
        lines.extend(["", "## Raw Format Failures", ""])
        for record in raw_failures:
            lines.append(
                f"- `{record['example_id']}` errors=`{record['metadata']['raw_format_errors']}` raw=`{record['metadata']['raw_answer']}`"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def validate_predictions(records: list[dict[str, Any]]) -> None:
    schema = load_schema("run_output.schema.json")
    for record in records:
        validate_with_schema(record, schema)
        if not record["metadata"]["final_format_valid"]:
            raise ValueError(f"Final answer failed format validation for {record['example_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed Phase 04 baseline generation smoke.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-size", choices=["3b", "7b"], default="3b")
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument("--retrieval-device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    run_id = f"baseline_smoke_{args.model_size}_phase04"
    output_dir = REPO_ROOT / "outputs" / "runs" / "baseline_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_smoke_candidates(config, args.retrieval_batch_size, args.retrieval_device)
    if len(candidates) != 20:
        raise RuntimeError(f"Expected 20 smoke candidates, got {len(candidates)}")

    settings = generation_settings_from_config(config, args.model_size)
    tokenizer, model = load_generation_model(settings)
    records_first = [generate_run_record(tokenizer, model, candidate, settings, run_id, "baseline") for candidate in candidates]
    set_global_seed(seed)
    records_second = [generate_run_record(tokenizer, model, candidate, settings, run_id, "baseline") for candidate in candidates]

    first_text = stable_jsonl(records_first)
    second_text = stable_jsonl(records_second)
    repeat_matched = first_text == second_text
    validate_predictions(records_first)
    if not repeat_matched:
        raise RuntimeError("Repeated baseline smoke run did not match byte-for-byte")

    write_jsonl_text(output_dir / "predictions.jsonl", first_text)
    write_jsonl_text(output_dir / "predictions_rerun.jsonl", second_text)
    summary = render_format_report(records_first, repeat_matched, output_dir / "format_report.md")
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "04_Baseline_RAG_Generation",
        "system": "baseline",
        "model": {
            "model_id": settings.model_id,
            "local_path": settings.model_path,
            "quantized": settings.load_in_4bit,
            "attn_implementation": settings.attn_implementation,
        },
        "dataset": {"name": "mixed", "split": "asqa_train_calibration_100_plus_finance_first10", "example_count": 20},
        "config_path": str(args.config),
        "seed": seed,
        "input_paths": [
            "data/asqa/splits/train_calibration_100.jsonl",
            "data/finance/generated/questions.jsonl",
            "outputs/retrieval/asqa_dense_embeddings.npz",
            "outputs/retrieval/finance_dense_embeddings.npz",
        ],
        "output_paths": [
            "outputs/runs/baseline_smoke/predictions.jsonl",
            "outputs/runs/baseline_smoke/predictions_rerun.jsonl",
            "outputs/runs/baseline_smoke/format_report.md",
        ],
        "generation_settings": settings.__dict__,
        "prompt_contract": "src.prompting.ANSWER_CONTRACT",
        "smoke_example_ids": [record["example_id"] for record in records_first],
        "format_summary": summary,
        "predictions_sha256": hashlib.sha256(first_text.encode("utf-8")).hexdigest(),
    }
    validate_with_schema({key: manifest[key] for key in ("run_id", "created_at", "phase", "system", "model", "dataset", "config_path", "seed", "input_paths", "output_paths")}, load_schema("run_manifest.schema.json"))
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps({"status": "passed", "summary": summary, "manifest": str(output_dir / "run_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
