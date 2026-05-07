from __future__ import annotations

from src.verifier import verify_record


FINANCE_GOLD = {
    "fin_q_002": {
        "example_id": "fin_q_002",
        "company_name": "Brindle Vale Robotics",
        "period": "Q3 FY2025",
        "metric_type": "cash_reserve",
        "gold_answer": "236 million fictional credits",
        "answerable": True,
        "expected_passage_ids": ["fin_passage_002_02"],
    },
    "fin_q_075": {
        "example_id": "fin_q_075",
        "company_name": "Glimmer Nocturne Works",
        "period": "Q4 FY2026",
        "metric_type": "revenue",
        "gold_answer": "INSUFFICIENT_SUPPORT",
        "answerable": False,
        "expected_passage_ids": [],
    },
}


def finance_record(answer: str, labels: list[str] | None = None, passage_id: str = "fin_passage_002_02") -> dict:
    labels = labels if labels is not None else ["P1"]
    return {
        "run_id": "test",
        "example_id": "fin_q_002",
        "dataset": "finance",
        "system": "baseline",
        "question": "What was the cash reserve for Brindle Vale Robotics in Q3 FY2025?",
        "retrieved_passages": [
            {
                "label": "P1",
                "passage_id": passage_id,
                "text": "Brindle Vale Robotics. Fictional issuer disclosure for Brindle Vale Robotics in Q3 FY2025: cash reserve was 236 million fictional credits.",
                "rank": 1,
                "score": 1.0,
            }
        ],
        "answer": answer,
        "abstained": answer == "INSUFFICIENT_SUPPORT",
        "citations": [{"sentence_index": 0, "labels": labels}] if labels else [],
        "generation": {"model_id": "test", "temperature": 0.0, "do_sample": False, "max_new_tokens": 20},
    }


def test_missing_citation_is_rejected() -> None:
    verdict = verify_record(
        finance_record("Brindle Vale Robotics had 236 million fictional credits in Q3 FY2025.", labels=[]),
        FINANCE_GOLD,
    )
    assert verdict["summary"]["passed"] is False
    assert any("missing_citation" in error for error in verdict["summary"]["errors"])


def test_malformed_citation_is_rejected() -> None:
    verdict = verify_record(
        finance_record("Brindle Vale Robotics had 236 million fictional credits in Q3 FY2025 [P9].", labels=[]),
        FINANCE_GOLD,
    )
    assert verdict["summary"]["passed"] is False
    assert any("malformed_citation" in error for error in verdict["summary"]["errors"])


def test_wrong_passage_citation_is_rejected() -> None:
    verdict = verify_record(
        finance_record(
            "Brindle Vale Robotics had a cash reserve of 236 million fictional credits in Q3 FY2025 [P1].",
            passage_id="fin_passage_002_00",
        ),
        FINANCE_GOLD,
    )
    assert verdict["summary"]["passed"] is False
    assert "finance_cited_passage_id_mismatch" in verdict["summary"]["errors"]


def test_finance_structured_mismatch_is_rejected() -> None:
    verdict = verify_record(
        finance_record("Brindle Vale Robotics had a cash reserve of 999 million fictional credits in Q3 FY2025 [P1]."),
        FINANCE_GOLD,
    )
    assert verdict["summary"]["passed"] is False
    assert "finance_numeric_value_mismatch" in verdict["summary"]["errors"]


def test_abstention_is_handled_cleanly() -> None:
    record = finance_record("INSUFFICIENT_SUPPORT", labels=[])
    record["example_id"] = "fin_q_075"
    verdict = verify_record(record, FINANCE_GOLD)
    assert verdict["summary"]["abstained"] is True
    assert verdict["summary"]["citation_format_valid"] is True


def test_multiple_citations_on_one_sentence() -> None:
    record = finance_record(
        "Brindle Vale Robotics had a cash reserve of 236 million fictional credits in Q3 FY2025 [P1][P2].",
        labels=["P1", "P2"],
    )
    record["retrieved_passages"].append(
        {
            "label": "P2",
            "passage_id": "fin_passage_002_01",
            "text": "Brindle Vale Robotics. Q2 FY2025 cash reserve was 224 million fictional credits.",
            "rank": 2,
            "score": 0.9,
        }
    )
    verdict = verify_record(record, FINANCE_GOLD)
    assert verdict["sentences"][0]["labels"] == ["P1", "P2"]
    assert verdict["summary"]["passed"] is True


def test_asqa_number_anchor_proxy() -> None:
    record = {
        "run_id": "test",
        "example_id": "asqa_1",
        "dataset": "asqa",
        "system": "baseline",
        "question": "When?",
        "retrieved_passages": [
            {
                "label": "P1",
                "passage_id": "asqa_passage",
                "text": "The event occurred in 1987.",
                "rank": 1,
                "score": 1.0,
            }
        ],
        "answer": "The event occurred in 1987 [P1].",
        "abstained": False,
        "citations": [{"sentence_index": 0, "labels": ["P1"]}],
        "generation": {"model_id": "test", "temperature": 0.0, "do_sample": False, "max_new_tokens": 20},
    }
    assert verify_record(record)["summary"]["passed"] is True
    record["answer"] = "The event occurred in 1999 [P1]."
    assert verify_record(record)["summary"]["passed"] is False


def test_asqa_short_answer_coverage_is_reported() -> None:
    record = {
        "run_id": "test",
        "example_id": "asqa_2",
        "dataset": "asqa",
        "system": "baseline",
        "question": "How many?",
        "retrieved_passages": [
            {
                "label": "P1",
                "passage_id": "asqa_passage",
                "text": "Virginia had six parks in 1936 and 38 parks in 2016.",
                "rank": 1,
                "score": 1.0,
            }
        ],
        "answer": "Virginia had six parks in 1936 and 38 parks in 2016 [P1].",
        "abstained": False,
        "citations": [{"sentence_index": 0, "labels": ["P1"]}],
        "generation": {"model_id": "test", "temperature": 0.0, "do_sample": False, "max_new_tokens": 20},
    }
    gold = {
        "asqa_2": {
            "qa_pairs": [
                {"question": "How many in 1936?", "short_answers": ["six", "6"]},
                {"question": "How many in 2016?", "short_answers": ["38"]},
            ]
        }
    }
    verdict = verify_record(record, asqa_gold=gold)
    assert verdict["summary"]["asqa_short_answer_coverage"] == 1.0
    assert verdict["summary"]["asqa_short_answer_hits"] == 2
