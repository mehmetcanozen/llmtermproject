from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.evaluation import (
    aggregate_metric_rows,
    build_example_scores,
    confidence_intervals,
    read_json,
    read_jsonl,
    write_csv,
)


METRIC_FIELDNAMES = [
    "system",
    "dataset",
    "model_size",
    "run_id",
    "eval_scope",
    "source_file",
    "artifact_mode",
    "example_count",
    "rejected_by_verifier_count",
    "answerable_count",
    "unanswerable_count",
    "answer_coverage",
    "citation_format_rate",
    "support_proxy_sentence_rate",
    "unsupported_non_abstained_rate",
    "abstention_rate",
    "retrieval_hit_rate",
    "exact_answer_accuracy",
    "correct_citation_rate",
    "false_attribution_rate",
    "abstention_rate_unanswerable",
    "asqa_short_answer_coverage",
]


def split_ids(path: Path) -> list[str]:
    return [record["example_id"] for record in read_jsonl(path)]


def score_ids_by_group(scores: list[Any]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    for score in scores:
        key = f"{score.dataset}|{score.system}|{score.model_size}|{score.eval_scope}|{score.artifact_mode}"
        grouped.setdefault(key, set()).add(score.example_id)
    return {key: sorted(value) for key, value in sorted(grouped.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Phase 07 metric tables and confidence intervals.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--verifier-artifact", type=Path, default=REPO_ROOT / "outputs" / "evaluation" / "verifier_verdicts.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_config = config["evaluation"]
    seed = int(eval_config.get("calibration_seed", config["project"]["seed"]))
    bootstrap_samples = int(eval_config.get("bootstrap_samples", 1000))
    confidence_level = float(eval_config.get("confidence_level", 0.95))
    verifier_path = args.verifier_artifact
    if not verifier_path.exists():
        verifier_path = REPO_ROOT / "artifacts" / "verifier" / "verifier_examples.json"
    verifier_artifact = read_json(verifier_path)
    scores = build_example_scores(REPO_ROOT, verifier_artifact, include_gate_plus_verifier=True)
    rows = aggregate_metric_rows(scores)
    intervals = confidence_intervals(
        scores,
        samples=bootstrap_samples,
        seed=seed,
        confidence_level=confidence_level,
    )

    output_dir = args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metric_tables.csv", rows, METRIC_FIELDNAMES)
    (output_dir / "confidence_intervals.json").write_text(
        json.dumps(intervals, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )

    expected_eval_splits = {
        "train_calibration_100": split_ids(REPO_ROOT / "data" / "asqa" / "splits" / "train_calibration_100.jsonl"),
        "dev_eval_200": split_ids(REPO_ROOT / "data" / "asqa" / "splits" / "dev_eval_200.jsonl"),
        "finance_full_100": split_ids(REPO_ROOT / "data" / "finance" / "generated" / "questions.jsonl"),
    }
    actual_scored_ids = score_ids_by_group(scores)
    dev_ids = set(expected_eval_splits["dev_eval_200"])
    finance_ids = set(expected_eval_splits["finance_full_100"])
    full_3b_groups = {
        key: set(ids)
        for key, ids in actual_scored_ids.items()
        if "|3b|" in key and "|recorded_prediction" in key
    }
    full_3b_asqa = all(
        full_3b_groups.get(f"asqa|{system}|3b|locked_fixed_split|recorded_prediction", set()) >= dev_ids
        for system in ("baseline", "gate_only")
    )
    full_3b_finance = all(
        full_3b_groups.get(f"finance|{system}|3b|locked_fixed_split|recorded_prediction", set()) >= finance_ids
        for system in ("baseline", "gate_only")
    )
    repair_full_3b_asqa = full_3b_groups.get(
        "asqa|repair_plus_verifier|3b|locked_fixed_split|recorded_prediction",
        set(),
    ) >= dev_ids
    repair_full_3b_finance = full_3b_groups.get(
        "finance|repair_plus_verifier|3b|locked_fixed_split|recorded_prediction",
        set(),
    ) >= finance_ids
    full_7b_groups = {
        key: set(ids)
        for key, ids in actual_scored_ids.items()
        if "|7b|" in key and "|recorded_prediction" in key
    }
    repair_full_7b_asqa = full_7b_groups.get(
        "asqa|repair_plus_verifier|7b|locked_fixed_split|recorded_prediction",
        set(),
    ) >= dev_ids
    repair_full_7b_finance = full_7b_groups.get(
        "finance|repair_plus_verifier|7b|locked_fixed_split|recorded_prediction",
        set(),
    ) >= finance_ids
    formal_full_eval_pass = bool(full_3b_asqa and full_3b_finance)
    repair_plus_full_eval_pass = bool(
        repair_full_3b_asqa and repair_full_3b_finance and repair_full_7b_asqa and repair_full_7b_finance
    )
    split_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_eval_splits": expected_eval_splits,
        "actual_scored_prediction_ids_by_group": actual_scored_ids,
        "formal_full_eval_pass": formal_full_eval_pass,
        "repair_plus_full_eval_pass": repair_plus_full_eval_pass,
        "repair_plus_scope_warning": None if repair_plus_full_eval_pass else (
            "Metric tables do not yet include full repair_plus_verifier 3B and 7B locked predictions "
            "on both dev_eval_200 and finance_full_100."
        ),
        "scope_warning": None if formal_full_eval_pass else (
            "Metric tables do not yet include full 3B locked predictions for baseline and gate_only "
            "on both dev_eval_200 and finance_full_100."
        ),
    }
    (output_dir / "eval_split_ids.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "phase": "07_Calibration_and_Evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verifier_artifact": str(verifier_path),
        "metric_tables": str(output_dir / "metric_tables.csv"),
        "confidence_intervals": str(output_dir / "confidence_intervals.json"),
        "eval_split_ids": str(output_dir / "eval_split_ids.json"),
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "row_count": len(rows),
        "systems_present": sorted({row["system"] for row in rows}),
        "datasets_present": sorted({row["dataset"] for row in rows}),
        "formal_full_eval_pass": formal_full_eval_pass,
        "repair_plus_full_eval_pass": repair_plus_full_eval_pass,
        "full_3b_asqa": full_3b_asqa,
        "full_3b_finance": full_3b_finance,
        "repair_full_3b_asqa": repair_full_3b_asqa,
        "repair_full_3b_finance": repair_full_3b_finance,
        "repair_full_7b_asqa": repair_full_7b_asqa,
        "repair_full_7b_finance": repair_full_7b_finance,
        "scope_warning": split_payload["scope_warning"],
        "repair_plus_scope_warning": split_payload["repair_plus_scope_warning"],
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed" if formal_full_eval_pass else "passed_with_scope_warning",
                "metric_rows": len(rows),
                "systems_present": manifest["systems_present"],
                "metric_tables": manifest["metric_tables"],
                "confidence_intervals": manifest["confidence_intervals"],
                "formal_full_eval_pass": formal_full_eval_pass,
                "repair_plus_full_eval_pass": repair_plus_full_eval_pass,
                "scope_warning": manifest["scope_warning"],
                "repair_plus_scope_warning": manifest["repair_plus_scope_warning"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
