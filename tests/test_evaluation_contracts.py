from __future__ import annotations

from src.evaluation import (
    ExampleScore,
    infer_eval_scope,
    paired_bootstrap_delta,
    paired_wilcoxon_delta,
    score_from_verdict,
    score_with_simulated_gate,
    simulate_gate_trace,
)


def test_distractor_locked_paths_have_distractor_eval_scope() -> None:
    scope = infer_eval_scope(
        "outputs/runs/locked/baseline_asqa_dev_eval_200_3b_distractor3b_distractor/predictions.jsonl",
        "baseline_asqa_dev_eval_200_3b_distractor3b_distractor",
    )
    assert scope == "locked_distractor_probe"


def test_gate_trace_replay_triggers_after_consecutive_low_windows() -> None:
    trace = {
        "tokens": [
            {"token_index": 0, "ignored_for_gate": True, "support_score": 0.0},
            {"token_index": 1, "ignored_for_gate": False, "support_score": 0.01},
            {"token_index": 2, "ignored_for_gate": False, "support_score": 0.02},
            {"token_index": 3, "ignored_for_gate": False, "support_score": 0.03},
            {"token_index": 4, "ignored_for_gate": False, "support_score": 0.04},
            {"token_index": 5, "ignored_for_gate": False, "support_score": 0.01},
        ]
    }
    result = simulate_gate_trace(trace, threshold=0.10, consecutive_failures=2, rolling_window=4)
    assert result["triggered"]
    assert result["trigger_token_index"] == 5


def test_gate_trace_replay_uses_requested_tail_layers() -> None:
    trace = {
        "tokens": [
            {
                "token_index": 0,
                "ignored_for_gate": False,
                "support_score": 0.01,
                "layer_passage_scores": [
                    {"P1": 0.50},
                    {"P1": 0.50},
                    {"P1": 0.01},
                    {"P1": 0.01},
                ],
            },
            {
                "token_index": 1,
                "ignored_for_gate": False,
                "support_score": 0.01,
                "layer_passage_scores": [
                    {"P1": 0.50},
                    {"P1": 0.50},
                    {"P1": 0.01},
                    {"P1": 0.01},
                ],
            },
        ]
    }
    assert simulate_gate_trace(trace, threshold=0.10, consecutive_failures=1, rolling_window=2, tail_layers=2)["triggered"] is True
    assert simulate_gate_trace(trace, threshold=0.10, consecutive_failures=1, rolling_window=2, tail_layers=4)["triggered"] is False
    assert simulate_gate_trace(trace, threshold=0.10, consecutive_failures=1, rolling_window=2, tail_layers=1)["triggered"] is True


def test_gate_plus_verifier_projection_rejects_false_attribution() -> None:
    verdict = {
        "run_id": "gate_smoke_7b_phase05",
        "example_id": "fin_q_002",
        "dataset": "finance",
        "system": "gate_only",
        "source_file": "outputs/runs/gate_smoke_7b_check/predictions.jsonl",
        "sentences": [],
        "domain_details": {"gold_answerable": True},
        "summary": {
            "abstained": False,
            "citation_format_valid": True,
            "support_passed": False,
            "exact_answer_correct": False,
            "correct_citation": True,
            "false_attribution": True,
            "sentence_count": 1,
            "passed": False,
        },
    }
    score = score_from_verdict(verdict, prediction=None, apply_verifier=True)
    assert score.system == "gate_plus_verifier"
    assert score.abstained
    assert score.rejected_by_verifier
    assert not score.false_attribution
    assert score.exact_answer_correct is False


def test_simulated_gate_abstention_keeps_unanswerable_exactness() -> None:
    score = ExampleScore(
        example_id="fin_q_075",
        dataset="finance",
        system="gate_only",
        model_size="3b",
        run_id="gate_smoke_3b_phase05",
        source_file="outputs/runs/gate_smoke/predictions.jsonl",
        eval_scope="targeted_gate_smoke_12_3b",
        artifact_mode="recorded_prediction",
        abstained=False,
        citation_format_valid=True,
        support_passed=False,
        false_attribution=True,
        exact_answer_correct=False,
        correct_citation=False,
        answerable=False,
        retrieval_hit=None,
        sentence_count=1,
        support_sentence_count=0,
        support_sentence_passes=0,
    )
    simulated = score_with_simulated_gate(score, triggered=True)
    assert simulated.abstained
    assert simulated.exact_answer_correct is True
    assert not simulated.false_attribution


def test_paired_bootstrap_delta_is_deterministic() -> None:
    left = [0.0, 1.0, 0.0, 1.0]
    right = [1.0, 1.0, 0.0, 1.0]
    first = paired_bootstrap_delta(left, right, samples=100, seed=7, confidence_level=0.95)
    second = paired_bootstrap_delta(left, right, samples=100, seed=7, confidence_level=0.95)
    assert first == second
    assert first["n_pairs"] == 4
    assert first["delta"] == 0.25
    assert first["ci_low"] <= first["delta"] <= first["ci_high"]


def test_wilcoxon_delta_handles_zero_and_nonzero_pairs() -> None:
    assert paired_wilcoxon_delta([1.0, 0.0], [1.0, 0.0])["p_value"] == 1.0
    result = paired_wilcoxon_delta([0.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    assert result["n_pairs"] == 3
    assert result["note"] in {"ok", "all_zero_differences"}
