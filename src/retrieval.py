from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.data_loading import read_jsonl, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class RetrievalDefaults:
    dense_top_k: int = 5
    bm25_top_k: int = 5
    prompt_top_k: int = 3
    dense_weight: float = 0.60
    bm25_weight: float = 0.40
    chunk_tokens_max: int = 220
    chunk_overlap_tokens: int = 40


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def chunk_by_words(text: str, max_tokens: int = 220, overlap_tokens: int = 40) -> list[dict[str, Any]]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [{"text": " ".join(words), "token_start": 0, "token_end": len(words)}]

    chunks: list[dict[str, Any]] = []
    step = max(1, max_tokens - overlap_tokens)
    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        if end - start < max_tokens // 2 and chunks:
            start = max(0, len(words) - max_tokens)
            end = len(words)
        chunks.append({"text": " ".join(words[start:end]), "token_start": start, "token_end": end})
        if end == len(words):
            break
        start += step
    return chunks


def make_search_text(title: str | None, text: str) -> str:
    if title:
        return f"{title}. {text}"
    return text


def build_chunk_records(records: list[dict[str, Any]], dataset: str, defaults: RetrievalDefaults) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for record in records:
        parent_passage_id = record["passage_id"]
        title = record.get("title") or record.get("company_name")
        base_text = record["text"]
        search_text = make_search_text(title, base_text)
        for local_index, chunk in enumerate(
            chunk_by_words(search_text, defaults.chunk_tokens_max, defaults.chunk_overlap_tokens)
        ):
            chunk_id = f"{dataset}_chunk_{len(chunks):06d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "dataset": dataset,
                    "parent_passage_id": parent_passage_id,
                    "source_example_id": record.get("source_example_id"),
                    "source_split": record.get("source_split"),
                    "title": title,
                    "text": chunk["text"],
                    "token_start": chunk["token_start"],
                    "token_end": chunk["token_end"],
                    "local_chunk_index": local_index,
                }
            )
    return chunks


def save_chunks(path: str | Path, chunks: list[dict[str, Any]]) -> None:
    write_jsonl(path, chunks)


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def encode_texts(model_path: str | Path, texts: list[str], batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def save_dense_index(path: str | Path, chunk_ids: list[str], embeddings: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, chunk_ids=np.asarray(chunk_ids), embeddings=embeddings.astype(np.float32))


def load_dense_index(path: str | Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(path)
    return [str(item) for item in payload["chunk_ids"].tolist()], np.asarray(payload["embeddings"], dtype=np.float32)


def minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return {key: (1.0 if hi > 0 else 0.0) for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        defaults: RetrievalDefaults,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count {len(chunks)} does not match embedding rows {len(embeddings)}")
        self.chunks = chunks
        self.embeddings = embeddings
        self.defaults = defaults
        self.tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def retrieve(self, query: str, query_embedding: np.ndarray) -> dict[str, Any]:
        query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        dense_scores_array = self.embeddings @ query_vector
        dense_top = top_indices(dense_scores_array, self.defaults.dense_top_k)
        dense_raw = {index: float(dense_scores_array[index]) for index in dense_top}

        bm25_scores_array = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        bm25_top = top_indices(bm25_scores_array, self.defaults.bm25_top_k)
        bm25_raw = {index: float(bm25_scores_array[index]) for index in bm25_top if bm25_scores_array[index] > 0}

        dense_norm = minmax_normalize(dense_raw)
        bm25_norm = minmax_normalize(bm25_raw)
        candidate_indices = sorted(set(dense_raw) | set(bm25_raw))
        merged = []
        for index in candidate_indices:
            dense_score = dense_norm.get(index, 0.0)
            bm25_score = bm25_norm.get(index, 0.0)
            final_score = self.defaults.dense_weight * dense_score + self.defaults.bm25_weight * bm25_score
            merged.append(format_candidate(self.chunks[index], final_score, dense_raw.get(index), bm25_raw.get(index)))
        merged.sort(key=lambda item: (-item["score"], item["chunk_id"]))
        return {
            "dense_candidates": [
                format_candidate(self.chunks[index], float(dense_scores_array[index]), float(dense_scores_array[index]), None)
                for index in dense_top
            ],
            "bm25_candidates": [
                format_candidate(self.chunks[index], float(bm25_scores_array[index]), None, float(bm25_scores_array[index]))
                for index in bm25_top
            ],
            "merged_top3": merged[: self.defaults.prompt_top_k],
        }


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if len(scores) == 0:
        return []
    top_k = min(top_k, len(scores))
    indices = np.argpartition(-scores, top_k - 1)[:top_k]
    return sorted((int(index) for index in indices), key=lambda index: float(scores[index]), reverse=True)


def format_candidate(
    chunk: dict[str, Any],
    score: float,
    dense_score: float | None,
    bm25_score: float | None,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "parent_passage_id": chunk["parent_passage_id"],
        "source_example_id": chunk.get("source_example_id"),
        "title": chunk.get("title"),
        "score": round(float(score), 6),
        "dense_score": None if dense_score is None else round(float(dense_score), 6),
        "bm25_score": None if bm25_score is None else round(float(bm25_score), 6),
        "text": chunk["text"],
    }
