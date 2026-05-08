from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


ABSTENTION = "INSUFFICIENT_SUPPORT"
ANSWER_CONTRACT = """You are a question-answering system that must use only the provided passages.

Rules:
- Answer only with information supported by the passages.
- Every factual sentence must end with one or more citations like [P1] or [P2].
- If the passages do not contain enough evidence, output exactly: INSUFFICIENT_SUPPORT
- Do not use outside knowledge."""

USER_PROMPT_TEMPLATE = """Question:
{question}

Passages:
{passages}

Answer style:
- {answer_style}
- Do not explain why other passages are irrelevant.
- Put citation markers at the end of the sentence.

Answer:"""

REPAIR_PROMPT_TEMPLATE = """Question:
{question}

Passages:
{passages}

Previous answer:
{failed_answer}

Verifier errors:
{verifier_errors}

Repair instructions:
- Use only the passages above.
- Fix only claims that can be directly supported by the cited passage text.
- {answer_style}
- Every factual sentence must end with one or more citation markers.
- If the passages do not fully support a corrected answer, output exactly: INSUFFICIENT_SUPPORT

Answer:"""

CITATION_RE = re.compile(r"\[(P[1-4])\]")
SENTENCE_END_CITATION_RE = re.compile(r"(?:\s*\[P[1-4]\])+\s*[.!?]?\s*$")


@dataclass(frozen=True)
class FormatValidation:
    valid: bool
    abstained: bool
    citations: list[dict[str, Any]]
    errors: list[str]
    checked_sentences: int


def prepare_prompt_passages(candidates: list[dict[str, Any]], required_count: int = 3) -> list[dict[str, Any]]:
    if len(candidates) < required_count:
        raise ValueError(f"Expected at least {required_count} retrieval candidates, got {len(candidates)}")
    passages: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:required_count], start=1):
        passages.append(
            {
                "label": f"P{index}",
                "passage_id": candidate["parent_passage_id"],
                "text": " ".join(str(candidate.get("text", "")).split()),
                "rank": index,
                "score": float(candidate.get("score", 0.0)),
                "title": candidate.get("title"),
                "chunk_id": candidate.get("chunk_id"),
            }
        )
    return passages


def passage_prompt_text(passage: dict[str, Any]) -> str:
    title = passage.get("title")
    text = str(passage["text"])
    prefix = f"{title}. " if title and not text.startswith(str(title)) else ""
    return f"{prefix}{text}"


def render_passage_block(passages: list[dict[str, Any]]) -> str:
    lines = []
    for passage in passages:
        lines.append(f"[{passage['label']}] {passage_prompt_text(passage)}")
    return "\n".join(lines)


def answer_style_for_dataset(dataset: str | None) -> str:
    if dataset == "asqa":
        return "If answerable, use one to three concise sentences to cover the question's distinct parts."
    if dataset == "finance":
        return "If answerable, use one concise sentence with the company, period, metric, and exact value."
    return "If answerable, use one concise sentence."


def build_messages(question: str, passages: list[dict[str, Any]], *, dataset: str | None = None) -> list[dict[str, str]]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        passages=render_passage_block(passages),
        answer_style=answer_style_for_dataset(dataset),
    )
    return [
        {"role": "system", "content": ANSWER_CONTRACT},
        {"role": "user", "content": user_prompt},
    ]


def build_repair_messages(
    question: str,
    passages: list[dict[str, Any]],
    *,
    failed_answer: str,
    verifier_errors: list[str],
    dataset: str | None = None,
) -> list[dict[str, str]]:
    error_text = "\n".join(f"- {error}" for error in verifier_errors) if verifier_errors else "- verifier_failed"
    user_prompt = REPAIR_PROMPT_TEMPLATE.format(
        question=question,
        passages=render_passage_block(passages),
        failed_answer=failed_answer,
        verifier_errors=error_text,
        answer_style=answer_style_for_dataset(dataset),
    )
    return [
        {"role": "system", "content": ANSWER_CONTRACT},
        {"role": "user", "content": user_prompt},
    ]


def build_chat_prompt(tokenizer: Any, question: str, passages: list[dict[str, Any]], *, dataset: str | None = None) -> str:
    return tokenizer.apply_chat_template(
        build_messages(question, passages, dataset=dataset),
        tokenize=False,
        add_generation_prompt=True,
    )


def build_repair_chat_prompt(
    tokenizer: Any,
    question: str,
    passages: list[dict[str, Any]],
    *,
    failed_answer: str,
    verifier_errors: list[str],
    dataset: str | None = None,
) -> str:
    return tokenizer.apply_chat_template(
        build_repair_messages(
            question,
            passages,
            failed_answer=failed_answer,
            verifier_errors=verifier_errors,
            dataset=dataset,
        ),
        tokenize=False,
        add_generation_prompt=True,
    )


def split_answer_sentences(answer: str) -> list[str]:
    normalized = " ".join(answer.strip().split())
    if not normalized:
        return []
    if normalized == ABSTENTION:
        return [normalized]
    # Keep bracket citations attached to the sentence they close.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", normalized)
    return [part.strip() for part in parts if part.strip()]


def extract_citation_labels(text: str) -> list[str]:
    return CITATION_RE.findall(text)


def validate_answer_format(answer: str, allowed_labels: set[str] | None = None) -> FormatValidation:
    allowed = allowed_labels or {"P1", "P2", "P3", "P4"}
    stripped = answer.strip()
    if stripped == ABSTENTION:
        return FormatValidation(valid=True, abstained=True, citations=[], errors=[], checked_sentences=0)
    if stripped.startswith(ABSTENTION):
        return FormatValidation(
            valid=False,
            abstained=False,
            citations=[],
            errors=["malformed_abstention"],
            checked_sentences=1,
        )

    errors: list[str] = []
    citations: list[dict[str, Any]] = []
    sentences = split_answer_sentences(stripped)
    if not sentences:
        errors.append("empty_answer")
    for index, sentence in enumerate(sentences):
        labels = extract_citation_labels(sentence)
        if not labels:
            errors.append(f"sentence_{index}_missing_citation")
        if labels and not SENTENCE_END_CITATION_RE.search(sentence):
            errors.append(f"sentence_{index}_citation_not_at_end")
        invalid = sorted(label for label in set(labels) if label not in allowed)
        if invalid:
            errors.append(f"sentence_{index}_invalid_citation_labels:{','.join(invalid)}")
        citations.append({"sentence_index": index, "labels": labels})
    return FormatValidation(
        valid=not errors,
        abstained=False,
        citations=citations,
        errors=errors,
        checked_sentences=len(sentences),
    )


def normalize_citation_surface(answer: str) -> str:
    stripped = answer.strip()
    if stripped == ABSTENTION:
        return stripped
    if stripped.startswith(ABSTENTION):
        return ABSTENTION

    # Common local-model typo: [CP1] instead of [P1].
    normalized = re.sub(r"\[CP([1-4])\]", r"[P\1]", stripped)
    sentences = split_answer_sentences(normalized)
    if not sentences:
        return normalized

    fixed_sentences: list[str] = []
    changed = False
    for sentence in sentences:
        labels = extract_citation_labels(sentence)
        if not labels or SENTENCE_END_CITATION_RE.search(sentence):
            fixed_sentences.append(sentence)
            continue
        unique_labels = list(dict.fromkeys(labels))
        without_citations = CITATION_RE.sub("", sentence).strip()
        without_citations = re.sub(r"\s+([,.!?])", r"\1", without_citations)
        without_citations = without_citations.rstrip()
        while without_citations.endswith((".", "!", "?")):
            without_citations = without_citations[:-1].rstrip()
        fixed_sentences.append(f"{without_citations} {''.join(f'[{label}]' for label in unique_labels)}.")
        changed = True
    return " ".join(fixed_sentences) if changed else normalized


def enforce_format_or_abstain(answer: str, allowed_labels: set[str] | None = None) -> tuple[str, FormatValidation, FormatValidation]:
    raw_validation = validate_answer_format(answer, allowed_labels)
    if raw_validation.valid:
        return answer.strip(), raw_validation, raw_validation
    normalized_answer = normalize_citation_surface(answer)
    normalized_validation = validate_answer_format(normalized_answer, allowed_labels)
    if normalized_validation.valid:
        return normalized_answer, raw_validation, normalized_validation
    final_answer = ABSTENTION
    final_validation = validate_answer_format(final_answer, allowed_labels)
    return final_answer, raw_validation, final_validation


def map_passage_token_spans(tokenizer: Any, chat_prompt: str, passages: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    encoded = tokenizer(chat_prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    spans: dict[str, dict[str, int]] = {}
    cursor = 0
    for passage in passages:
        marker = f"[{passage['label']}]"
        marker_index = chat_prompt.find(marker, cursor)
        if marker_index < 0:
            raise ValueError(f"Could not find passage marker {marker} in prompt")
        text = passage_prompt_text(passage)
        text_start = chat_prompt.find(text, marker_index)
        if text_start < 0:
            raise ValueError(f"Could not find passage text for {marker} in prompt")
        text_end = text_start + len(text)
        token_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > text_start and start < text_end
        ]
        if not token_indices:
            raise ValueError(f"Could not map token span for {marker}")
        spans[passage["label"]] = {
            "char_start": text_start,
            "char_end": text_end,
            "token_start": min(token_indices),
            "token_end_exclusive": max(token_indices) + 1,
        }
        cursor = text_end
    return spans
