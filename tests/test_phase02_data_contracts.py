from __future__ import annotations

from scripts.build_finance_dataset import build_dataset


def test_finance_dataset_composition_and_reproducibility() -> None:
    _, questions_a, manifest_a = build_dataset(20260416)
    _, questions_b, manifest_b = build_dataset(20260416)
    assert manifest_a["counts"]["composition"] == {
        "exact_numeric": 40,
        "wrong_period_trap": 20,
        "near_duplicate_issuer_trap": 15,
        "unanswerable": 15,
        "retrieval_collision_distractor": 10,
    }
    assert [(item["example_id"], item["gold_answer"]) for item in questions_a] == [
        (item["example_id"], item["gold_answer"]) for item in questions_b
    ]
    assert manifest_a["determinism_check"]["same_example_ids_and_gold_answers"] is True
    assert manifest_b["determinism_check"]["same_example_ids_and_gold_answers"] is True
