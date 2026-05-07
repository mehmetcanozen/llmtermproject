from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.evaluation import (
    build_example_scores,
    calibration_metrics,
    example_metric,
    read_json,
    read_jsonl,
    score_with_simulated_gate,
    simulate_gate_trace,
    write_csv,
)


def repo_relative(path: Path) -> str:
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    return str(resolved.resolve().relative_to(REPO_ROOT))


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]
TAIL_LAYERS = [2, 4]
CONSECUTIVE_FAILURES = [2, 3]


def mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def baseline_overlap_metrics(scores: list[Any], gate_example_ids: set[str], dataset: str) -> dict[str, Any]:
    overlap = [
        score
        for score in scores
        if score.system == "baseline"
        and score.model_size == "3b"
        and score.dataset == dataset
        and score.example_id in gate_example_ids
    ]
    return {
        "baseline_overlap_example_count": len(overlap),
        "baseline_answer_coverage_overlap": mean_bool([example_metric(score, "answer_coverage") == 1.0 for score in overlap]),
        "baseline_unsupported_non_abstained_overlap": mean_bool(
            [(not score.abstained) and score.false_attribution for score in overlap]
        ),
    }


def evaluate_candidate(
    *,
    stage: str,
    traces: list[dict[str, Any]],
    base_scores_by_id: dict[str, Any],
    threshold: float,
    tail_layers: int,
    consecutive_failures: int,
    rolling_window: int,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    simulated_scores = []
    triggered_ids = []
    for trace in traces:
        result = simulate_gate_trace(
            trace,
            threshold=threshold,
            consecutive_failures=consecutive_failures,
            rolling_window=rolling_window,
            tail_layers=tail_layers,
        )
        if result["triggered"]:
            triggered_ids.append(trace["example_id"])
        simulated_scores.append(
            score_with_simulated_gate(base_scores_by_id[trace["example_id"]], triggered=result["triggered"])
        )
    metrics = calibration_metrics(simulated_scores)
    coverage = metrics["answer_coverage"]
    baseline_coverage = baseline_metrics["baseline_answer_coverage_overlap"]
    unsupported = metrics["unsupported_non_abstained_rate"]
    baseline_unsupported = baseline_metrics["baseline_unsupported_non_abstained_overlap"]
    coverage_margin = None if coverage is None or baseline_coverage is None else coverage - baseline_coverage
    reduces_unsupported = (
        False
        if unsupported is None or baseline_unsupported is None
        else unsupported < baseline_unsupported
    )
    passes_constraints = bool(
        coverage is not None
        and baseline_coverage is not None
        and coverage >= baseline_coverage - 0.10
        and (metrics["abstention_rate"] or 0.0) <= 0.35
    )
    return {
        "stage": stage,
        "threshold": threshold,
        "tail_layers": tail_layers,
        "consecutive_failures": consecutive_failures,
        "rolling_window": rolling_window,
        "trace_tail_layers_source": "per_layer" if traces and traces[0].get("tokens", [{}])[0].get("layer_passage_scores") else 4,
        "tail_layers_evaluation_note": "native_per_layer_trace" if traces and traces[0].get("tokens", [{}])[0].get("layer_passage_scores") else "legacy_recorded_tail4",
        "simulated_triggered_count": len(triggered_ids),
        "simulated_triggered_ids": triggered_ids,
        **metrics,
        **baseline_metrics,
        "coverage_margin_from_baseline": coverage_margin,
        "reduces_unsupported_vs_baseline_overlap": reduces_unsupported,
        "passes_plan_selection_constraints": passes_constraints,
    }


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    threshold_distance = abs(float(row["threshold"]) - 0.10)
    return (
        row["unsupported_non_abstained_rate"] if row["unsupported_non_abstained_rate"] is not None else 1.0,
        -(row["answer_coverage"] if row["answer_coverage"] is not None else -1.0),
        row["abstention_rate"] if row["abstention_rate"] is not None else 1.0,
        0 if int(row["tail_layers"]) == 4 else 1,
        threshold_distance,
        0 if int(row["consecutive_failures"]) == 3 else 1,
    )


def choose_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_b = [row for row in rows if row["stage"] == "B"]
    pool = stage_b or rows
    ranked = sorted(pool, key=rank_key)
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
    return ranked[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the attention gate from saved train-calibration traces.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--verifier-artifact", type=Path, default=REPO_ROOT / "outputs" / "evaluation" / "verifier_verdicts.json")
    parser.add_argument("--trace-file", type=Path, default=REPO_ROOT / "outputs" / "runs" / "locked" / "gate_only_asqa_train_calibration_100_3b_calibration" / "attention_traces.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "calibration")
    args = parser.parse_args()

    config = load_config(args.config)
    verifier_path = args.verifier_artifact
    if not verifier_path.exists():
        verifier_path = REPO_ROOT / "artifacts" / "verifier" / "verifier_examples.json"
    verifier_artifact = read_json(verifier_path)
    scores = build_example_scores(REPO_ROOT, verifier_artifact, include_gate_plus_verifier=False)
    trace_file = args.trace_file
    if not trace_file.exists():
        trace_file = REPO_ROOT / "outputs" / "runs" / "gate_smoke" / "attention_traces.jsonl"
    traces = read_jsonl(trace_file)
    trace_ids = {trace["example_id"] for trace in traces}
    trace_dataset = traces[0].get("dataset", "finance") if traces else "unknown"
    base_scores_by_id = {
        score.example_id: score
        for score in scores
        if score.system == "gate_only" and score.model_size == "3b" and score.example_id in trace_ids
    }
    missing = [trace["example_id"] for trace in traces if trace["example_id"] not in base_scores_by_id]
    if missing:
        raise ValueError(f"Trace examples missing verifier scores: {missing}")

    rolling_window = int(config["gate"]["rolling_window"])
    gate_example_ids = set(base_scores_by_id)
    baseline_metrics = baseline_overlap_metrics(scores, gate_example_ids, trace_dataset)
    rows: list[dict[str, Any]] = []
    stage_a_traces = traces[:60]
    for threshold in THRESHOLDS:
        rows.append(
            evaluate_candidate(
                stage="A",
                traces=stage_a_traces,
                base_scores_by_id=base_scores_by_id,
                threshold=threshold,
                tail_layers=4,
                consecutive_failures=3,
                rolling_window=rolling_window,
                baseline_metrics=baseline_metrics,
            )
        )
    top_thresholds = [
        row["threshold"]
        for row in sorted([row for row in rows if row["stage"] == "A"], key=lambda row: (
            row["unsupported_non_abstained_rate"] if row["unsupported_non_abstained_rate"] is not None else 1.0,
            -(row["answer_coverage"] if row["answer_coverage"] is not None else -1.0),
            abs(float(row["threshold"]) - 0.10),
        ))[:2]
    ]
    for threshold in top_thresholds:
        for tail_layers in TAIL_LAYERS:
            for consecutive_failures in CONSECUTIVE_FAILURES:
                rows.append(
                    evaluate_candidate(
                        stage="B",
                        traces=traces,
                        base_scores_by_id=base_scores_by_id,
                        threshold=threshold,
                        tail_layers=tail_layers,
                        consecutive_failures=consecutive_failures,
                        rolling_window=rolling_window,
                        baseline_metrics=baseline_metrics,
                    )
                )

    chosen = choose_row(rows)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "selection_rank",
        "threshold",
        "tail_layers",
        "consecutive_failures",
        "rolling_window",
        "trace_tail_layers_source",
        "tail_layers_evaluation_note",
        "example_count",
        "answerable_count",
        "unanswerable_count",
        "simulated_triggered_count",
        "simulated_triggered_ids",
        "unsupported_non_abstained_rate",
        "answer_coverage",
        "abstention_rate",
        "answerable_abstention_rate",
        "unanswerable_abstention_rate",
        "baseline_overlap_example_count",
        "baseline_answer_coverage_overlap",
        "baseline_unsupported_non_abstained_overlap",
        "coverage_margin_from_baseline",
        "reduces_unsupported_vs_baseline_overlap",
        "passes_plan_selection_constraints",
    ]
    write_csv(output_dir / "calibration_results.csv", rows, fieldnames)

    status = (
        "proxy_selected_and_constraints_met"
        if chosen["passes_plan_selection_constraints"] and chosen["reduces_unsupported_vs_baseline_overlap"]
        else "proxy_selected_constraints_not_fully_satisfied"
    )
    chosen_payload = {
        "phase": "07_Calibration_and_Evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "calibration_mode": "artifact_trace_replay_smoke",
        "chosen_gate_config": {
            "threshold": float(chosen["threshold"]),
            "tail_layers": int(chosen["tail_layers"]),
            "rolling_window": int(chosen["rolling_window"]),
            "consecutive_failures": int(chosen["consecutive_failures"]),
            "skip_initial_generated_tokens": int(config["gate"]["skip_initial_generated_tokens"]),
        },
        "why_selected": {
            "selection_rule": "lowest unsupported non-abstained rate, then highest answer coverage, then lower abstention, tie-breaking toward Phase 05 runtime settings",
            "selection_rank": chosen.get("selection_rank"),
            "unsupported_non_abstained_rate": chosen["unsupported_non_abstained_rate"],
            "answer_coverage": chosen["answer_coverage"],
            "abstention_rate": chosen["abstention_rate"],
            "baseline_answer_coverage_overlap": chosen["baseline_answer_coverage_overlap"],
            "baseline_unsupported_non_abstained_overlap": chosen["baseline_unsupported_non_abstained_overlap"],
            "passes_plan_selection_constraints": chosen["passes_plan_selection_constraints"],
            "reduces_unsupported_vs_baseline_overlap": chosen["reduces_unsupported_vs_baseline_overlap"],
        },
        "source_artifacts": {
            "verifier_artifact": repo_relative(verifier_path),
            "trace_file": repo_relative(trace_file),
            "calibration_results": "outputs/calibration/calibration_results.csv",
        },
        "limitations": [
            f"Trace dataset: {trace_dataset}; trace count: {len(traces)}.",
            "Stage A uses the first 60 traces; Stage B uses all traces and sweeps the top 2 Stage A thresholds.",
            "Tail-layer sweeps are exact only when traces contain layer_passage_scores; otherwise legacy smoke traces remain proxy rows.",
            "The chosen config is acceptable for final evaluation only if this file was produced from train_calibration_100 traces.",
        ],
    }
    (output_dir / "chosen_gate_config.yaml").write_text(
        yaml.safe_dump(chosen_payload, sort_keys=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "chosen_gate_config": chosen_payload["chosen_gate_config"],
                "calibration_results": str(output_dir / "calibration_results.csv"),
                "chosen_config": str(output_dir / "chosen_gate_config.yaml"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
