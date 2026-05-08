from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PRIMARY_METRICS = {
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
}


@dataclass(frozen=True)
class ExampleScore:
    example_id: str
    dataset: str
    system: str
    model_size: str
    run_id: str
    source_file: str
    eval_scope: str
    artifact_mode: str
    abstained: bool
    citation_format_valid: bool
    support_passed: bool
    false_attribution: bool
    exact_answer_correct: bool | None
    correct_citation: bool | None
    answerable: bool | None
    retrieval_hit: bool | None
    sentence_count: int
    support_sentence_count: int
    support_sentence_passes: int
    asqa_short_answer_coverage: float | None = None
    rejected_by_verifier: bool = False


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def infer_model_size(run_id: str, source_file: str = "") -> str:
    text = f"{run_id} {source_file}".lower()
    if "7b" in text:
        return "7b"
    if "3b" in text:
        return "3b"
    return "unknown"


def infer_eval_scope(source_file: str, run_id: str) -> str:
    normalized = source_file.replace("\\", "/").lower()
    if "distractor" in normalized:
        return "locked_distractor_probe"
    if "locked" in normalized:
        return "locked_fixed_split"
    if "baseline_smoke" in normalized:
        return "baseline_smoke_mixed_train10_finance10"
    if "gate_smoke_7b_check" in normalized or "7b" in run_id.lower():
        return "targeted_gate_smoke_12_transfer_7b"
    if "gate_smoke" in normalized:
        return "targeted_gate_smoke_12_3b"
    return "unknown_artifact_scope"


def prediction_index(repo_root: Path, source_files: Iterable[str]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sorted(set(source_files)):
        path = repo_root / source
        if not path.exists():
            path = repo_root / source.replace("\\", "/")
        for record in read_jsonl(path):
            index[(record["run_id"], record["example_id"])] = record
    return index


def finance_answerable(verdict: dict[str, Any], prediction: dict[str, Any] | None) -> bool | None:
    details = verdict.get("domain_details") or {}
    if "gold_answerable" in details:
        return bool(details["gold_answerable"])
    if prediction:
        metadata = prediction.get("metadata", {}).get("source_metadata", {})
        if "answerable" in metadata:
            return bool(metadata["answerable"])
    return None


def retrieval_hit(prediction: dict[str, Any] | None) -> bool | None:
    if not prediction:
        return None
    metadata = prediction.get("metadata", {})
    expected = metadata.get("expected_passage_ids") or []
    if not expected:
        return None
    value = metadata.get("retrieval_hit_top3")
    return None if value is None else bool(value)


def support_sentence_counts(verdict: dict[str, Any]) -> tuple[int, int]:
    sentences = verdict.get("sentences") or []
    if verdict.get("dataset") != "asqa":
        return 0, 0
    count = 0
    passed = 0
    for sentence in sentences:
        if "asqa_proxy" not in sentence:
            continue
        count += 1
        passed += int(bool(sentence["asqa_proxy"].get("passed")))
    return count, passed


def score_from_verdict(
    verdict: dict[str, Any],
    prediction: dict[str, Any] | None = None,
    *,
    apply_verifier: bool = False,
) -> ExampleScore:
    summary = verdict["summary"]
    rejected = apply_verifier and (not summary["abstained"]) and (not summary["passed"])
    answerable = finance_answerable(verdict, prediction) if verdict["dataset"] == "finance" else None
    abstained = bool(summary["abstained"]) or rejected
    exact_answer_correct = summary.get("exact_answer_correct")
    correct_citation = summary.get("correct_citation")
    false_attribution = bool(summary.get("false_attribution", False))
    support_passed = bool(summary.get("support_passed", False))
    citation_format_valid = bool(summary.get("citation_format_valid", False))
    sentence_count = int(summary.get("sentence_count", 0))
    support_sentence_count, support_sentence_passes = support_sentence_counts(verdict)
    asqa_short_answer_coverage = summary.get("asqa_short_answer_coverage")

    if rejected:
        false_attribution = False
        support_passed = True
        citation_format_valid = True
        sentence_count = 0
        support_sentence_count = 0
        support_sentence_passes = 0
        correct_citation = None
        if verdict["dataset"] == "finance":
            exact_answer_correct = False if answerable is not False else True
        else:
            exact_answer_correct = None
        asqa_short_answer_coverage = None

    source_file = verdict.get("source_file", "")
    run_id = verdict.get("run_id", "")
    system = "gate_plus_verifier" if apply_verifier else verdict.get("system", "unknown")
    return ExampleScore(
        example_id=verdict["example_id"],
        dataset=verdict["dataset"],
        system=system,
        model_size=infer_model_size(run_id, source_file),
        run_id=run_id,
        source_file=source_file,
        eval_scope=infer_eval_scope(source_file, run_id),
        artifact_mode="verifier_rejection_projection" if apply_verifier else "recorded_prediction",
        abstained=abstained,
        citation_format_valid=citation_format_valid,
        support_passed=support_passed,
        false_attribution=false_attribution,
        exact_answer_correct=exact_answer_correct,
        correct_citation=correct_citation,
        answerable=answerable,
        retrieval_hit=retrieval_hit(prediction),
        sentence_count=sentence_count,
        support_sentence_count=support_sentence_count,
        support_sentence_passes=support_sentence_passes,
        asqa_short_answer_coverage=asqa_short_answer_coverage,
        rejected_by_verifier=rejected,
    )


def build_example_scores(
    repo_root: Path,
    verifier_artifact: dict[str, Any],
    *,
    include_gate_plus_verifier: bool = True,
) -> list[ExampleScore]:
    verdicts = verifier_artifact["verdicts"]
    predictions = prediction_index(repo_root, [verdict["source_file"] for verdict in verdicts])
    scores: list[ExampleScore] = []
    for verdict in verdicts:
        prediction = predictions.get((verdict["run_id"], verdict["example_id"]))
        scores.append(score_from_verdict(verdict, prediction))
        if include_gate_plus_verifier and verdict.get("system") == "gate_only":
            scores.append(score_from_verdict(verdict, prediction, apply_verifier=True))
    return scores


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def mean_bool(values: Iterable[bool | None]) -> float | None:
    clean = [bool(value) for value in values if value is not None]
    return safe_rate(sum(clean), len(clean))


def mean_numeric(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return safe_rate(sum(clean), len(clean))


def example_metric(score: ExampleScore, metric: str) -> float | None:
    if metric == "answer_coverage":
        if score.dataset == "asqa":
            return float((not score.abstained) and score.support_passed and score.citation_format_valid)
        if score.dataset == "finance":
            return float(score.answerable is True and score.exact_answer_correct is True)
    if metric == "citation_format_rate":
        return float(score.citation_format_valid)
    if metric == "unsupported_non_abstained_rate":
        return float((not score.abstained) and score.false_attribution)
    if metric == "abstention_rate":
        return float(score.abstained)
    if metric == "retrieval_hit_rate":
        return None if score.retrieval_hit is None else float(score.retrieval_hit)
    if metric == "exact_answer_accuracy":
        return None if score.exact_answer_correct is None else float(score.exact_answer_correct)
    if metric == "correct_citation_rate":
        return None if score.correct_citation is None else float(score.correct_citation)
    if metric == "false_attribution_rate":
        return float(score.false_attribution)
    if metric == "abstention_rate_unanswerable":
        return None if score.answerable is not False else float(score.abstained)
    if metric == "asqa_short_answer_coverage":
        return score.asqa_short_answer_coverage
    if metric == "support_proxy_sentence_rate":
        return None
    raise KeyError(f"Unknown metric: {metric}")


def aggregate_group(scores: list[ExampleScore]) -> dict[str, Any]:
    first = scores[0]
    support_sentence_count = sum(score.support_sentence_count for score in scores)
    support_sentence_passes = sum(score.support_sentence_passes for score in scores)
    row: dict[str, Any] = {
        "system": first.system,
        "dataset": first.dataset,
        "model_size": first.model_size,
        "run_id": first.run_id,
        "eval_scope": first.eval_scope,
        "source_file": first.source_file,
        "artifact_mode": first.artifact_mode,
        "example_count": len(scores),
        "rejected_by_verifier_count": sum(score.rejected_by_verifier for score in scores),
        "answerable_count": sum(score.answerable is True for score in scores),
        "unanswerable_count": sum(score.answerable is False for score in scores),
    }
    for metric in sorted(PRIMARY_METRICS):
        if metric == "support_proxy_sentence_rate":
            row[metric] = safe_rate(support_sentence_passes, support_sentence_count)
            continue
        if metric == "asqa_short_answer_coverage":
            row[metric] = mean_numeric(example_metric(score, metric) for score in scores)
            continue
        row[metric] = mean_bool(example_metric(score, metric) for score in scores)
    return row


def aggregate_metric_rows(scores: list[ExampleScore]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[ExampleScore]] = defaultdict(list)
    for score in scores:
        groups[
            (
                score.system,
                score.dataset,
                score.model_size,
                score.run_id,
                score.source_file,
                score.artifact_mode,
            )
        ].append(score)
    rows = [aggregate_group(group_scores) for group_scores in groups.values()]
    rows.sort(key=lambda row: (row["dataset"], row["system"], row["model_size"], row["run_id"], row["artifact_mode"]))
    return rows


def paired_arrays(
    scores: list[ExampleScore],
    *,
    dataset: str,
    left_system: str,
    right_system: str,
    metric: str,
    left_model_size: str | None = None,
    right_model_size: str | None = None,
) -> tuple[list[float], list[float], list[str]]:
    left: dict[str, float] = {}
    right: dict[str, float] = {}
    for score in scores:
        if score.dataset != dataset:
            continue
        value = example_metric(score, metric)
        if value is None:
            continue
        if score.system == left_system and (left_model_size is None or score.model_size == left_model_size):
            left[score.example_id] = value
        if score.system == right_system and (right_model_size is None or score.model_size == right_model_size):
            right[score.example_id] = value
    ids = sorted(set(left).intersection(right))
    return [left[example_id] for example_id in ids], [right[example_id] for example_id in ids], ids


def paired_bootstrap_delta(
    left_values: list[float],
    right_values: list[float],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if len(left_values) != len(right_values):
        raise ValueError("paired bootstrap requires equal-length arrays")
    n = len(left_values)
    if n == 0:
        return {"n_pairs": 0, "delta": None, "ci_low": None, "ci_high": None}
    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    deltas = right - left
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample_indices = rng.integers(0, n, size=n)
        draws[index] = float(deltas[sample_indices].mean())
    alpha = 1.0 - confidence_level
    return {
        "n_pairs": n,
        "delta": float(deltas.mean()),
        "ci_low": float(np.quantile(draws, alpha / 2.0)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
    }


def paired_wilcoxon_delta(left_values: list[float], right_values: list[float]) -> dict[str, Any]:
    if len(left_values) != len(right_values):
        raise ValueError("Wilcoxon requires equal-length arrays")
    n = len(left_values)
    if n == 0:
        return {"n_pairs": 0, "statistic": None, "p_value": None, "note": "no_pairs"}
    differences = np.asarray(right_values, dtype=np.float64) - np.asarray(left_values, dtype=np.float64)
    if np.allclose(differences, 0.0):
        return {"n_pairs": n, "statistic": 0.0, "p_value": 1.0, "note": "all_zero_differences"}
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
    except Exception as exc:  # pragma: no cover - depends on scipy edge cases
        return {"n_pairs": n, "statistic": None, "p_value": None, "note": f"{type(exc).__name__}: {exc}"}
    return {"n_pairs": n, "statistic": float(result.statistic), "p_value": float(result.pvalue), "note": "ok"}


def confidence_intervals(
    scores: list[ExampleScore],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    comparisons = []
    for dataset in sorted({score.dataset for score in scores}):
        metrics = (
            ["answer_coverage", "unsupported_non_abstained_rate", "abstention_rate", "asqa_short_answer_coverage"]
            if dataset == "asqa"
            else ["exact_answer_accuracy", "unsupported_non_abstained_rate", "false_attribution_rate", "abstention_rate"]
        )
        for left, right in (
            ("baseline", "gate_only"),
            ("baseline", "gate_plus_verifier"),
            ("gate_only", "gate_plus_verifier"),
            ("baseline", "repair_plus_verifier"),
            ("gate_plus_verifier", "repair_plus_verifier"),
        ):
            comparisons.append(
                {
                    "name": f"{left}_3b_vs_{right}_3b_{dataset}_overlap",
                    "dataset": dataset,
                    "left_system": left,
                    "right_system": right,
                    "left_model_size": "3b",
                    "right_model_size": "3b",
                    "metrics": metrics,
                }
            )
        if any(score.dataset == dataset and score.model_size == "7b" for score in scores):
            for left, right in (
                ("gate_only", "gate_plus_verifier"),
                ("gate_plus_verifier", "repair_plus_verifier"),
            ):
                comparisons.append(
                    {
                        "name": f"{left}_7b_vs_{right}_7b_{dataset}_transfer",
                        "dataset": dataset,
                        "left_system": left,
                        "right_system": right,
                        "left_model_size": "7b",
                        "right_model_size": "7b",
                        "metrics": metrics,
                    }
                )
    output: dict[str, Any] = {
        "method": "paired bootstrap percentile interval",
        "samples": samples,
        "seed": seed,
        "confidence_level": confidence_level,
        "comparisons": [],
    }
    for comparison in comparisons:
        metric_rows = []
        paired_ids: list[str] = []
        for metric in comparison["metrics"]:
            left, right, ids = paired_arrays(
                scores,
                dataset=comparison["dataset"],
                left_system=comparison["left_system"],
                right_system=comparison["right_system"],
                metric=metric,
                left_model_size=comparison["left_model_size"],
                right_model_size=comparison["right_model_size"],
            )
            if ids:
                paired_ids = ids
            result = paired_bootstrap_delta(
                left,
                right,
                samples=samples,
                seed=seed,
                confidence_level=confidence_level,
            )
            metric_rows.append({"metric": metric, **result, "wilcoxon": paired_wilcoxon_delta(left, right)})
        output["comparisons"].append(
            {
                **comparison,
                "paired_example_ids": paired_ids,
                "metric_intervals": metric_rows,
            }
        )
    return output


def simulate_gate_trace(
    trace: dict[str, Any],
    *,
    threshold: float,
    consecutive_failures: int,
    rolling_window: int,
    tail_layers: int = 4,
) -> dict[str, Any]:
    rolling_scores: list[float] = []
    failures = 0
    for token in trace.get("tokens", []):
        if token.get("ignored_for_gate"):
            continue
        support_score = token_support_score(token, tail_layers=tail_layers)
        rolling_scores.append(support_score)
        window = rolling_scores[-rolling_window:]
        failed = len(window) >= rolling_window and (sum(window) / len(window)) < threshold
        failures = failures + 1 if failed else 0
        if failures >= consecutive_failures:
            return {
                "triggered": True,
                "trigger_token_index": token["token_index"],
                "trigger_support_score": support_score,
            }
    return {"triggered": False, "trigger_token_index": None, "trigger_support_score": None}


def token_support_score(token: dict[str, Any], *, tail_layers: int) -> float:
    layer_scores = token.get("layer_passage_scores")
    if not layer_scores:
        return float(token["support_score"])
    selected = layer_scores[-tail_layers:] if tail_layers > 0 else layer_scores
    values = [float(score) for layer in selected for score in layer.values()]
    return max(values) if values else float(token["support_score"])


def score_with_simulated_gate(score: ExampleScore, *, triggered: bool) -> ExampleScore:
    if not triggered:
        return score
    exact = score.exact_answer_correct
    if score.dataset == "finance":
        exact = True if score.answerable is False else False
    return ExampleScore(
        **{
            **score.__dict__,
            "abstained": True,
            "citation_format_valid": True,
            "support_passed": True,
            "false_attribution": False,
            "exact_answer_correct": exact,
            "correct_citation": None,
            "sentence_count": 0,
            "support_sentence_count": 0,
            "support_sentence_passes": 0,
            "asqa_short_answer_coverage": None,
        }
    )


def calibration_metrics(scores: list[ExampleScore]) -> dict[str, Any]:
    answerable = [score for score in scores if score.answerable is True]
    unanswerable = [score for score in scores if score.answerable is False]
    return {
        "example_count": len(scores),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "unsupported_non_abstained_rate": mean_bool(
            (not score.abstained) and score.false_attribution for score in scores
        ),
        "answer_coverage": mean_bool(example_metric(score, "answer_coverage") == 1.0 for score in scores),
        "abstention_rate": mean_bool(score.abstained for score in scores),
        "answerable_abstention_rate": mean_bool(score.abstained for score in answerable),
        "unanswerable_abstention_rate": mean_bool(score.abstained for score in unanswerable),
        "asqa_short_answer_coverage": mean_numeric(score.asqa_short_answer_coverage for score in scores),
    }
