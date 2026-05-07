from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.prompting import ABSTENTION, extract_citation_labels, split_answer_sentences, validate_answer_format


ANY_CITATION_RE = re.compile(r"\[([A-Za-z]*P?\d+(?:-[A-Za-z]*P?\d+)?)\]")
EXPLICIT_NUMBER_RE = re.compile(r"\b(?:\d{1,4}(?:,\d{3})*|\d+(?:\.\d+)?)\b")
QUOTED_SPAN_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
METRIC_LABELS = {
    "revenue": "revenue",
    "operating_income": "operating income",
    "cash_reserve": "cash reserve",
}


@dataclass(frozen=True)
class VerificationSummary:
    passed: bool
    abstained: bool
    citation_format_valid: bool
    support_passed: bool
    exact_answer_correct: bool | None
    correct_citation: bool | None
    false_attribution: bool
    sentence_count: int
    errors: list[str]
    asqa_short_answer_coverage: float | None = None
    asqa_short_answer_hits: int | None = None
    asqa_short_answer_total: int | None = None


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.lower().split())


def answer_text_without_citations(answer: str) -> str:
    return normalize_text(ANY_CITATION_RE.sub("", answer))


def label_to_passage_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {passage["label"]: passage for passage in record.get("retrieved_passages", [])}


def malformed_citation_tokens(answer: str) -> list[str]:
    labels = set(extract_citation_labels(answer))
    malformed = []
    for match in ANY_CITATION_RE.findall(answer):
        if match not in labels:
            malformed.append(match)
    return malformed


def cited_passage_text(labels: list[str], passages_by_label: dict[str, dict[str, Any]]) -> str:
    return " ".join(passages_by_label[label]["text"] for label in labels if label in passages_by_label)


def explicit_anchors(sentence: str) -> list[str]:
    anchors = EXPLICIT_NUMBER_RE.findall(sentence)
    for match in QUOTED_SPAN_RE.findall(sentence):
        anchors.extend(item for item in match if item)
    return anchors


def verify_citation_structure(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], bool]:
    answer = record.get("answer", "")
    passages_by_label = label_to_passage_map(record)
    allowed_labels = set(passages_by_label)
    format_result = validate_answer_format(answer, allowed_labels=allowed_labels)
    errors = list(format_result.errors)
    malformed = malformed_citation_tokens(answer)
    if malformed:
        errors.append("malformed_citation:" + ",".join(malformed))
    sentence_results: list[dict[str, Any]] = []
    if answer.strip() == ABSTENTION:
        return sentence_results, errors, not errors
    sentences = split_answer_sentences(answer)
    for index, sentence in enumerate(sentences):
        labels = extract_citation_labels(sentence)
        missing = not labels
        invalid = sorted(label for label in labels if label not in allowed_labels)
        sentence_errors = []
        if missing:
            sentence_errors.append("missing_citation")
        if invalid:
            sentence_errors.append("citation_outside_retrieved_set:" + ",".join(invalid))
        sentence_results.append(
            {
                "sentence_index": index,
                "sentence": sentence,
                "labels": labels,
                "cited_passage_ids": [
                    passages_by_label[label]["passage_id"] for label in labels if label in passages_by_label
                ],
                "structure_passed": not sentence_errors,
                "errors": sentence_errors,
            }
        )
        errors.extend(f"sentence_{index}_{error}" for error in sentence_errors)
    return sentence_results, errors, not errors


def verify_asqa_proxy(record: dict[str, Any], sentence_results: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    passages_by_label = label_to_passage_map(record)
    errors: list[str] = []
    for result in sentence_results:
        labels = result["labels"]
        cited_text = normalize_text(cited_passage_text(labels, passages_by_label))
        missing_anchors = []
        for anchor in explicit_anchors(result["sentence"]):
            if normalize_text(anchor) not in cited_text:
                missing_anchors.append(anchor)
        result["asqa_proxy"] = {
            "explicit_anchors": explicit_anchors(result["sentence"]),
            "missing_anchors": missing_anchors,
            "passed": not missing_anchors,
        }
        if missing_anchors:
            errors.append(f"sentence_{result['sentence_index']}_missing_anchor:" + ",".join(missing_anchors))
    return not errors, errors


def verify_asqa_short_answers(record: dict[str, Any], asqa_gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold = asqa_gold.get(record["example_id"], {})
    qa_pairs = gold.get("qa_pairs") or record.get("metadata", {}).get("source_metadata", {}).get("qa_pairs") or []
    answer_norm = answer_text_without_citations(record.get("answer", ""))
    hits = 0
    total = 0
    matched_pairs = []
    for pair in qa_pairs:
        short_answers = pair.get("short_answers") or []
        if not short_answers:
            continue
        total += 1
        matched = [short for short in short_answers if normalize_text(short) and normalize_text(short) in answer_norm]
        if matched:
            hits += 1
        matched_pairs.append(
            {
                "question": pair.get("question"),
                "short_answers": short_answers,
                "matched_short_answers": matched,
            }
        )
    return {
        "short_answer_hits": hits,
        "short_answer_total": total,
        "short_answer_coverage": None if total == 0 else hits / total,
        "matched_pairs": matched_pairs,
    }


def verify_finance_exact(record: dict[str, Any], finance_gold: dict[str, dict[str, Any]]) -> tuple[bool, bool, list[str], dict[str, Any]]:
    example_id = record["example_id"]
    gold = finance_gold.get(example_id)
    if gold is None:
        return False, False, ["missing_finance_gold"], {}
    if record.get("answer", "").strip() == ABSTENTION:
        expected_clean_abstention = not gold.get("answerable", False)
        return expected_clean_abstention, expected_clean_abstention, [], {
            "gold_answerable": gold.get("answerable", False),
            "expected_passage_ids": gold.get("expected_passage_ids", []),
        }

    answer_norm = normalize_text(record.get("answer", ""))
    metric_label = METRIC_LABELS.get(gold["metric_type"], gold["metric_type"].replace("_", " "))
    checks = {
        "company_name": normalize_text(gold["company_name"]) in answer_norm,
        "period": normalize_text(gold["period"]) in answer_norm,
        "metric_type": normalize_text(metric_label) in answer_norm,
        "numeric_value": normalize_text(gold["gold_answer"]) in answer_norm,
    }
    passages_by_label = label_to_passage_map(record)
    cited_passage_ids = {
        passage["passage_id"]
        for result in record.get("citations", [])
        for label in result.get("labels", [])
        for passage in [passages_by_label.get(label)]
        if passage
    }
    expected_ids = set(gold.get("expected_passage_ids", []))
    checks["cited_passage_id"] = bool(expected_ids and expected_ids.intersection(cited_passage_ids))
    errors = [f"finance_{name}_mismatch" for name, passed in checks.items() if not passed]
    exact_correct = all(checks.values())
    correct_citation = checks["cited_passage_id"]
    details = {
        "gold_answerable": gold.get("answerable", False),
        "company_name": gold["company_name"],
        "period": gold["period"],
        "metric_type": gold["metric_type"],
        "gold_answer": gold["gold_answer"],
        "expected_passage_ids": sorted(expected_ids),
        "cited_passage_ids": sorted(cited_passage_ids),
        "checks": checks,
    }
    return exact_correct, correct_citation, errors, details


def verify_record(
    record: dict[str, Any],
    finance_gold: dict[str, dict[str, Any]] | None = None,
    asqa_gold: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    finance_gold = finance_gold or {}
    asqa_gold = asqa_gold or {}
    answer = record.get("answer", "")
    abstained = answer.strip() == ABSTENTION
    sentence_results, structure_errors, structure_passed = verify_citation_structure(record)
    support_passed = structure_passed
    support_errors: list[str] = []
    exact_answer_correct: bool | None = None
    correct_citation: bool | None = None
    domain_details: dict[str, Any] = {}
    asqa_short_answer_coverage: float | None = None
    asqa_short_answer_hits: int | None = None
    asqa_short_answer_total: int | None = None

    if not abstained and structure_passed:
        if record["dataset"] == "finance":
            exact_answer_correct, correct_citation, finance_errors, domain_details = verify_finance_exact(
                record, finance_gold
            )
            support_passed = exact_answer_correct
            support_errors.extend(finance_errors)
        elif record["dataset"] == "asqa":
            support_passed, support_errors = verify_asqa_proxy(record, sentence_results)
            exact_answer_correct = None
            correct_citation = None
            short_answer_details = verify_asqa_short_answers(record, asqa_gold)
            asqa_short_answer_coverage = short_answer_details["short_answer_coverage"]
            asqa_short_answer_hits = short_answer_details["short_answer_hits"]
            asqa_short_answer_total = short_answer_details["short_answer_total"]
            domain_details = short_answer_details
    elif abstained and record["dataset"] == "finance":
        gold = finance_gold.get(record["example_id"])
        if gold is not None:
            exact_answer_correct = not gold.get("answerable", False)
            correct_citation = None
            domain_details = {
                "gold_answerable": gold.get("answerable", False),
                "expected_passage_ids": gold.get("expected_passage_ids", []),
            }

    errors = structure_errors + support_errors
    citation_format_valid = not structure_errors
    false_attribution = (not abstained) and (not support_passed)
    summary = VerificationSummary(
        passed=not errors,
        abstained=abstained,
        citation_format_valid=citation_format_valid,
        support_passed=support_passed,
        exact_answer_correct=exact_answer_correct,
        correct_citation=correct_citation,
        false_attribution=false_attribution,
        sentence_count=len(sentence_results),
        errors=errors,
        asqa_short_answer_coverage=asqa_short_answer_coverage,
        asqa_short_answer_hits=asqa_short_answer_hits,
        asqa_short_answer_total=asqa_short_answer_total,
    )
    return {
        "run_id": record.get("run_id"),
        "example_id": record["example_id"],
        "dataset": record["dataset"],
        "system": record["system"],
        "summary": summary.__dict__,
        "sentences": sentence_results,
        "domain_details": domain_details,
    }


def load_finance_gold(questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {question["example_id"]: question for question in questions}


def load_asqa_gold(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["example_id"]: record for record in records}
