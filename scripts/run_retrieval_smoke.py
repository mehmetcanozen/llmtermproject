from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data_loading import read_jsonl, write_json, write_jsonl
from src.retrieval import HybridRetriever, RetrievalDefaults, encode_texts, load_chunks, load_dense_index


def defaults_from_config(config: dict) -> RetrievalDefaults:
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
    support: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        source_example_id = chunk.get("source_example_id")
        if source_example_id:
            support[source_example_id].add(chunk["parent_passage_id"])
    return support


def select_finance_inspection(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = {
        "exact_numeric": 5,
        "wrong_period_trap": 5,
        "near_duplicate_issuer_trap": 5,
        "retrieval_collision_distractor": 5,
    }
    selected: list[dict[str, Any]] = []
    for question_type, count in targets.items():
        group = [question for question in questions if question["question_type"] == question_type and question["answerable"]]
        selected.extend(group[:count])
    return selected


def build_query_records(asqa_eval: list[dict[str, Any]], finance_questions: list[dict[str, Any]], support: dict[str, set[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    asqa_inspection = [record for record in asqa_eval if support.get(record["example_id"])][:10]
    finance_inspection = select_finance_inspection(finance_questions)
    inspection_ids = {record["example_id"] for record in asqa_inspection + finance_inspection}

    asqa_queries = [
        {
            "dataset": "asqa",
            "example_id": record["example_id"],
            "question": record["ambiguous_question"],
            "expected_passage_ids": sorted(support.get(record["example_id"], set())),
            "inspection_member": record["example_id"] in inspection_ids,
            "metadata": {"qa_pair_count": record["qa_pair_count"]},
        }
        for record in asqa_eval
    ]
    finance_queries = [
        {
            "dataset": "finance",
            "example_id": record["example_id"],
            "question": record["question"],
            "expected_passage_ids": record["expected_passage_ids"],
            "inspection_member": record["example_id"] in inspection_ids,
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
    return asqa_queries, finance_queries, inspection_ids


def run_queries(queries: list[dict[str, Any]], retriever: HybridRetriever, model_path: Path, query_instruction: str, batch_size: int, device: str) -> list[dict[str, Any]]:
    query_texts = [query_instruction + query["question"] for query in queries]
    query_embeddings = encode_texts(model_path, query_texts, batch_size=batch_size, device=device)
    outputs: list[dict[str, Any]] = []
    for query, embedding in zip(queries, query_embeddings):
        result = retriever.retrieve(query["question"], embedding)
        top_parent_ids = [candidate["parent_passage_id"] for candidate in result["merged_top3"]]
        expected = set(query["expected_passage_ids"])
        hit = bool(expected and expected.intersection(top_parent_ids))
        outputs.append(
            {
                **query,
                "raw_dense_candidates": result["dense_candidates"],
                "raw_bm25_candidates": result["bm25_candidates"],
                "merged_top3": result["merged_top3"],
                "hit_top3": hit,
            }
        )
    return outputs


def render_inspection_sheet(asqa_outputs: list[dict[str, Any]], finance_outputs: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    inspected = [record for record in asqa_outputs + finance_outputs if record["inspection_member"]]
    asqa_hits = sum(1 for record in inspected if record["dataset"] == "asqa" and record["hit_top3"])
    finance_hits = sum(1 for record in inspected if record["dataset"] == "finance" and record["hit_top3"])
    total_hits = asqa_hits + finance_hits
    summary = {
        "inspected": len(inspected),
        "asqa_inspected": sum(1 for record in inspected if record["dataset"] == "asqa"),
        "finance_inspected": sum(1 for record in inspected if record["dataset"] == "finance"),
        "asqa_hits": asqa_hits,
        "finance_hits": finance_hits,
        "total_hits": total_hits,
        "passed": len(inspected) == 30 and total_hits >= 21 and asqa_hits >= 6 and finance_hits >= 15,
    }

    lines = [
        "# Phase 03 Manual Retrieval Inspection",
        "",
        "Inspection method: provenance-assisted manual sheet. A row is marked relevant when at least one top-3 chunk comes from the expected ASQA source example or the expected synthetic finance passage, with snippets shown for review.",
        "",
        "## Summary",
        "",
        f"- Inspected examples: `{summary['inspected']}`",
        f"- ASQA top-3 support hits: `{summary['asqa_hits']}/10`",
        f"- Finance top-3 support hits: `{summary['finance_hits']}/20`",
        f"- Overall top-3 support hits: `{summary['total_hits']}/30`",
        f"- Phase 03 retrieval gate: `{'passed' if summary['passed'] else 'failed'}`",
        "",
        "## Rows",
        "",
        "| # | Dataset | Example ID | Relevant in top 3 | Expected passage IDs | Top 3 parent IDs | Question | Top snippet |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for index, record in enumerate(inspected, start=1):
        expected = ", ".join(record["expected_passage_ids"][:3])
        top_ids = ", ".join(candidate["parent_passage_id"] for candidate in record["merged_top3"])
        snippet = record["merged_top3"][0]["text"][:180].replace("|", "/") if record["merged_top3"] else ""
        question = record["question"][:140].replace("|", "/")
        lines.append(
            f"| {index} | {record['dataset']} | `{record['example_id']}` | `{record['hit_top3']}` | `{expected}` | `{top_ids}` | {question} | {snippet} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 03 retrieval smoke and create the 30-example inspection sheet.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    defaults = defaults_from_config(config)
    out_dir = REPO_ROOT / "outputs" / "retrieval"
    model_path = Path(config["retrieval"]["local_embedding_path"])

    asqa_chunks = load_chunks(out_dir / "asqa_chunks.jsonl")
    finance_chunks = load_chunks(out_dir / "finance_chunks.jsonl")
    _, asqa_embeddings = load_dense_index(out_dir / "asqa_dense_embeddings.npz")
    _, finance_embeddings = load_dense_index(out_dir / "finance_dense_embeddings.npz")

    asqa_retriever = HybridRetriever(asqa_chunks, asqa_embeddings, defaults)
    finance_retriever = HybridRetriever(finance_chunks, finance_embeddings, defaults)

    asqa_eval = read_jsonl(REPO_ROOT / "data" / "asqa" / "splits" / "dev_eval_200.jsonl")
    finance_questions = read_jsonl(REPO_ROOT / "data" / "finance" / "generated" / "questions.jsonl")
    asqa_queries, finance_queries, _ = build_query_records(asqa_eval, finance_questions, asqa_support_map(asqa_chunks))

    query_instruction = config["retrieval"].get("query_instruction", "")
    asqa_outputs = run_queries(asqa_queries, asqa_retriever, model_path, query_instruction, args.batch_size, args.device)
    finance_outputs = run_queries(finance_queries, finance_retriever, model_path, query_instruction, args.batch_size, args.device)

    write_jsonl(out_dir / "asqa_candidates.jsonl", asqa_outputs)
    write_jsonl(out_dir / "finance_candidates.jsonl", finance_outputs)
    summary = render_inspection_sheet(asqa_outputs, finance_outputs, out_dir / "manual_inspection_phase03.md")
    manifest = {
        "phase": "03_Retrieval_System",
        "defaults": defaults.__dict__,
        "query_instruction_used": bool(query_instruction),
        "candidate_files": {
            "asqa": "outputs/retrieval/asqa_candidates.jsonl",
            "finance": "outputs/retrieval/finance_candidates.jsonl",
        },
        "inspection_sheet": "outputs/retrieval/manual_inspection_phase03.md",
        "inspection_summary": summary,
        "asqa_candidate_count": len(asqa_outputs),
        "finance_candidate_count": len(finance_outputs),
        "finance_candidate_types": dict(Counter(record["metadata"]["question_type"] for record in finance_outputs)),
    }
    write_json(out_dir / "retrieval_smoke_manifest.json", manifest)
    print(json.dumps({"status": "passed" if summary["passed"] else "failed", "summary": summary}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
