from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data_loading import read_jsonl, write_json
from src.retrieval import RetrievalDefaults, build_chunk_records, encode_texts, save_chunks, save_dense_index


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


def build_one(dataset: str, source_path: Path, chunk_path: Path, embedding_path: Path, model_path: Path, defaults: RetrievalDefaults, batch_size: int, device: str) -> dict:
    records = read_jsonl(source_path)
    chunks = build_chunk_records(records, dataset, defaults)
    save_chunks(chunk_path, chunks)
    embeddings = encode_texts(model_path, [chunk["text"] for chunk in chunks], batch_size=batch_size, device=device)
    save_dense_index(embedding_path, [chunk["chunk_id"] for chunk in chunks], embeddings)
    return {
        "dataset": dataset,
        "source_records": len(records),
        "chunks": len(chunks),
        "chunk_path": str(chunk_path.relative_to(REPO_ROOT)),
        "embedding_path": str(embedding_path.relative_to(REPO_ROOT)),
        "embedding_shape": list(embeddings.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 03 chunk files and dense retrieval indexes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    defaults = defaults_from_config(config)
    model_path = Path(config["retrieval"]["local_embedding_path"])
    out_dir = REPO_ROOT / "outputs" / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        build_one(
            "asqa",
            REPO_ROOT / "data" / "asqa" / "corpus" / "passages.jsonl",
            out_dir / "asqa_chunks.jsonl",
            out_dir / "asqa_dense_embeddings.npz",
            model_path,
            defaults,
            args.batch_size,
            args.device,
        ),
        build_one(
            "finance",
            REPO_ROOT / "data" / "finance" / "corpus" / "passages.jsonl",
            out_dir / "finance_chunks.jsonl",
            out_dir / "finance_dense_embeddings.npz",
            model_path,
            defaults,
            args.batch_size,
            args.device,
        ),
    ]
    manifest = {
        "phase": "03_Retrieval_System",
        "embedding_model": config["retrieval"]["embedding_model_id"],
        "embedding_model_path": str(model_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "chunking": {
            "chunk_tokens_max": defaults.chunk_tokens_max,
            "chunk_overlap_tokens": defaults.chunk_overlap_tokens,
        },
        "results": results,
    }
    write_json(out_dir / "retrieval_index_manifest.json", manifest)
    print(json.dumps({"status": "passed", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
