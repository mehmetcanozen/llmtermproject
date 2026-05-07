from __future__ import annotations

from pathlib import Path

from src.final_assets import package_metric_rows
from src.locked_runs import choose_distractor, query_records_from_split, selected_candidates, split_path, with_optional_distractor


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_split_builders_do_not_truncate_eval_sets() -> None:
    asqa = query_records_from_split(REPO_ROOT, "asqa", "dev_eval_200", support={})
    finance = query_records_from_split(REPO_ROOT, "finance", "finance_full_100")
    assert len(asqa) == 200
    assert len(finance) == 100
    assert split_path(REPO_ROOT, "asqa", "dev_eval_200").exists()
    assert split_path(REPO_ROOT, "finance", "finance_full_100").exists()


def test_selected_candidates_only_truncates_when_limit_is_explicit() -> None:
    rows = [{"example_id": f"ex_{index}"} for index in range(5)]
    assert len(selected_candidates(rows, limit=None)) == 5
    assert [row["example_id"] for row in selected_candidates(rows, start=1, limit=2)] == ["ex_1", "ex_2"]


def test_distractor_adds_fourth_prompt_passage_without_replacing_top3() -> None:
    record = {
        "example_id": "fin_q_001",
        "expected_passage_ids": ["gold"],
        "metadata": {},
        "merged_top3": [
            {"parent_passage_id": "gold", "score": 1.0, "text": "gold"},
            {"parent_passage_id": "near", "score": 0.7, "text": "near"},
            {"parent_passage_id": "other", "score": 0.4, "text": "other"},
        ],
        "raw_dense_candidates": [{"parent_passage_id": "extra", "score": 0.9, "text": "extra"}],
        "raw_bm25_candidates": [],
    }
    updated = with_optional_distractor(record, True)
    assert [row["parent_passage_id"] for row in updated["merged_top3"]] == ["gold", "near", "other", "extra"]
    assert updated["metadata"]["distractor_enabled"] is True


def test_distractor_falls_back_when_all_candidates_are_expected() -> None:
    record = {
        "example_id": "asqa_dev_edge",
        "expected_passage_ids": ["gold"],
        "merged_top3": [{"parent_passage_id": "gold", "score": 1.0, "text": "gold"}],
        "raw_dense_candidates": [{"parent_passage_id": "gold", "score": 1.0, "text": "gold"}],
        "raw_bm25_candidates": [],
    }
    distractor = choose_distractor(record)
    assert distractor["parent_passage_id"] == "gold"
    assert distractor["distractor_source_list"] == "fallback_first_available"


def test_final_package_keeps_formal_flag_false_for_proxy_rows() -> None:
    rows = package_metric_rows(
        [{"system": "baseline", "dataset": "asqa"}],
        formal_full_eval_pass=False,
        scope_note="proxy only",
    )
    assert rows[0]["formal_full_eval_pass"] is False
    assert rows[0]["scope_note"] == "proxy only"


def test_calibration_script_defaults_to_train_calibration_not_dev_eval() -> None:
    source = (REPO_ROOT / "scripts" / "calibrate_gate.py").read_text(encoding="utf-8")
    assert "train_calibration_100" in source
    assert "dev_eval_200" not in source
