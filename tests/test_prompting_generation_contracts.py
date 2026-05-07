from __future__ import annotations

from src.prompting import ABSTENTION, enforce_format_or_abstain, prepare_prompt_passages, validate_answer_format


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
