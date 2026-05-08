from __future__ import annotations

from src.prompting import (
    ABSTENTION,
    build_repair_messages,
    enforce_format_or_abstain,
    prepare_prompt_passages,
    validate_answer_format,
)


def test_format_validator_accepts_cited_sentence() -> None:
    result = validate_answer_format("The answer is supported by the first passage [P1].")
    assert result.valid is True
    result = validate_answer_format("The answer is supported by the first passage. [P1]")
    assert result.valid is True
    assert result.citations == [{"sentence_index": 0, "labels": ["P1"]}]


def test_format_validator_accepts_abstention() -> None:
    result = validate_answer_format(ABSTENTION)
    assert result.valid is True
    assert result.abstained is True
    assert validate_answer_format("INSUFFICIENT_SUPPORT [P1]").valid is False


def test_prepare_prompt_passages_labels_top_three() -> None:
    candidates = [
        {"parent_passage_id": f"p{i}", "text": f"text {i}", "score": 1.0 - i * 0.1}
        for i in range(4)
    ]
    passages = prepare_prompt_passages(candidates)
    assert [passage["label"] for passage in passages] == ["P1", "P2", "P3"]
    assert [passage["rank"] for passage in passages] == [1, 2, 3]


def test_format_enforcer_repositions_existing_citation() -> None:
    final_answer, raw_validation, final_validation = enforce_format_or_abstain(
        "[P1] The value was 236 million fictional credits."
    )
    assert raw_validation.valid is False
    assert final_validation.valid is True
    assert final_answer == "The value was 236 million fictional credits [P1]."


def test_asqa_multi_sentence_format_requires_each_sentence_cited() -> None:
    result = validate_answer_format("First supported fact [P1]. Second unsupported sentence.")
    assert result.valid is False
    assert "sentence_1_missing_citation" in result.errors


def test_finance_repair_prompt_does_not_receive_gold_answer() -> None:
    passages = prepare_prompt_passages(
        [
            {"parent_passage_id": "p1", "text": "Acme FY2025 revenue was 42 credits.", "score": 1.0},
            {"parent_passage_id": "p2", "text": "Acme FY2024 revenue was 39 credits.", "score": 0.8},
            {"parent_passage_id": "p3", "text": "Beta FY2025 revenue was 77 credits.", "score": 0.4},
        ]
    )
    messages = build_repair_messages(
        "What was Acme revenue in FY2025?",
        passages,
        failed_answer="Acme FY2025 revenue was 41 credits [P1].",
        verifier_errors=["wrong_numeric_value"],
        dataset="finance",
    )
    prompt = messages[1]["content"]
    assert "wrong_numeric_value" in prompt
    assert "42 credits" in prompt
    assert "999 credits" not in prompt
