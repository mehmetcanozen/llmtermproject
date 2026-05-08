from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from src.locked_runs import validate_retrieval_artifacts
from src.retrieval import HybridRetriever, RetrievalDefaults, build_chunk_records


def test_chunk_records_preserve_parent_passage_id() -> None:
    records = [
        {
            "passage_id": "p1",
            "source_example_id": "ex1",
            "source_split": "dev",
            "title": "Example Page",
            "text": " ".join(["support"] * 230),
        }
    ]
    chunks = build_chunk_records(records, "asqa", RetrievalDefaults(chunk_tokens_max=120, chunk_overlap_tokens=20))
    assert len(chunks) >= 2
    assert {chunk["parent_passage_id"] for chunk in chunks} == {"p1"}
    assert {chunk["source_example_id"] for chunk in chunks} == {"ex1"}


def test_hybrid_retriever_returns_three_prompt_chunks() -> None:
    chunks = [
        {"chunk_id": f"c{i}", "parent_passage_id": f"p{i}", "source_example_id": None, "title": None, "text": text}
        for i, text in enumerate(["alpha revenue", "beta income", "gamma cash", "delta risk"])
    ]
    embeddings = np.eye(4, dtype=np.float32)
    retriever = HybridRetriever(chunks, embeddings, RetrievalDefaults(dense_top_k=4, bm25_top_k=4, prompt_top_k=3))
    result = retriever.retrieve("alpha revenue", np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert len(result["merged_top3"]) == 3
    assert result["merged_top3"][0]["parent_passage_id"] == "p0"


def test_retrieval_validation_rejects_zero_byte_embeddings() -> None:
    repo_root = Path("outputs") / "test" / f"retrieval_validation_{uuid4().hex}"
    retrieval_dir = repo_root / "outputs" / "retrieval"
    retrieval_dir.mkdir(parents=True)
    (retrieval_dir / "asqa_chunks.jsonl").write_text(
        '{"chunk_id":"c0","parent_passage_id":"p0","text":"support"}\n',
        encoding="utf-8",
    )
    (retrieval_dir / "asqa_dense_embeddings.npz").write_bytes(b"")
    try:
        validate_retrieval_artifacts(repo_root, "asqa")
    except ValueError as exc:
        assert "Empty retrieval artifact" in str(exc)
    else:  # pragma: no cover - keeps assertion message clear
        raise AssertionError("zero-byte retrieval embeddings should fail validation")
