from __future__ import annotations

from pathlib import Path

from src.final_assets import Canvas, build_distractor_probe_rows, metric_fieldnames, package_generated_distractor_rows, choose_distractor


def test_choose_distractor_avoids_expected_and_existing_top3() -> None:
    record = {
        "expected_passage_ids": ["gold"],
        "merged_top3": [
            {"parent_passage_id": "gold", "score": 1.0},
            {"parent_passage_id": "top2", "score": 0.8},
            {"parent_passage_id": "top3", "score": 0.5},
        ],
        "raw_dense_candidates": [
            {"parent_passage_id": "gold", "score": 0.9, "text": "gold"},
            {"parent_passage_id": "top2", "score": 0.7, "text": "top2"},
            {"parent_passage_id": "distractor", "score": 0.6, "text": "distractor"},
        ],
        "raw_bm25_candidates": [],
    }
    distractor = choose_distractor(record)
    assert distractor["parent_passage_id"] == "distractor"
    assert distractor["source_list"] == "raw_dense_candidates"


def test_distractor_probe_adds_one_prompt_passage_for_each_system() -> None:
    candidate = {
        "dataset": "finance",
        "example_id": "fin_q_001",
        "question": "What was the revenue?",
        "expected_passage_ids": ["gold"],
        "hit_top3": True,
        "merged_top3": [
            {"parent_passage_id": "gold", "score": 1.0, "title": "Issuer", "text": "gold"},
            {"parent_passage_id": "near", "score": 0.5, "title": "Issuer", "text": "near"},
            {"parent_passage_id": "other", "score": 0.4, "title": "Other", "text": "other"},
        ],
        "raw_dense_candidates": [{"parent_passage_id": "extra", "score": 0.7, "title": "Issuer", "text": "extra"}],
        "raw_bm25_candidates": [],
    }
    rows = build_distractor_probe_rows([], [candidate], asqa_count=0, finance_count=1)
    assert len(rows) == 4
    assert {row["system"] for row in rows} == {
        "baseline",
        "gate_only",
        "gate_plus_verifier",
        "repair_plus_verifier",
    }
    assert all(row["added_distractor_count"] == 1 for row in rows)
    assert all(row["probe_prompt_passage_count"] == 4 for row in rows)


def test_generated_distractor_metric_rows_are_packaged_as_stress_test() -> None:
    rows = package_generated_distractor_rows(
        [{"system": "baseline", "dataset": "asqa", "example_count": "200"}]
    )
    assert rows[0]["phase08_package_scope"] == "generated_distractor_stress_test"
    assert rows[0]["formal_full_eval_pass"] is False
    assert "generated_distractor_metrics" not in metric_fieldnames([])
    assert "phase08_package_scope" in metric_fieldnames(rows)


def test_canvas_writes_valid_png() -> None:
    output = Path("outputs/test/final_assets_canvas.png")
    canvas = Canvas(width=80, height=40)
    canvas.text(4, 4, "OK", scale=2)
    canvas.rect(4, 25, 50, 30, (42, 157, 143))
    canvas.save_png(output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
