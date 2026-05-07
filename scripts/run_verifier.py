from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loading import read_jsonl, write_json
from src.verifier import load_asqa_gold, load_finance_gold, verify_record


def default_prediction_sources() -> list[Path]:
    locked = sorted((REPO_ROOT / "outputs" / "runs" / "locked").glob("*/predictions.jsonl"))
    if locked:
        preferred = [
            path for path in locked
            if "smoke" not in path.parent.name.lower()
            and ("dev_eval_200" in path.parent.name or "finance_full_100" in path.parent.name)
        ]
        if preferred:
            return preferred
        non_smoke = [path for path in locked if "smoke" not in path.parent.name.lower()]
        return non_smoke or locked
    return [
        REPO_ROOT / "outputs" / "runs" / "baseline_smoke" / "predictions.jsonl",
        REPO_ROOT / "outputs" / "runs" / "gate_smoke" / "predictions.jsonl",
        REPO_ROOT / "outputs" / "runs" / "gate_smoke_7b_check" / "predictions.jsonl",
    ]


def as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify saved prediction JSONL files.")
    parser.add_argument("--source", type=Path, action="append", help="Prediction JSONL file. May be passed more than once.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "evaluation" / "verifier_verdicts.json")
    args = parser.parse_args()

    sources = args.source or default_prediction_sources()
    sources = [path if path.is_absolute() else REPO_ROOT / path for path in sources]
    existing_sources = [path for path in sources if path.exists()]
    if not existing_sources:
        raise FileNotFoundError("No prediction sources found")

    finance_gold = load_finance_gold(read_jsonl(REPO_ROOT / "data" / "finance" / "generated" / "questions.jsonl"))
    asqa_gold = load_asqa_gold(
        read_jsonl(REPO_ROOT / "data" / "asqa" / "splits" / "train_calibration_100.jsonl")
        + read_jsonl(REPO_ROOT / "data" / "asqa" / "splits" / "dev_eval_200.jsonl")
    )
    verdicts = []
    source_counts = {}
    for source in existing_sources:
        records = read_jsonl(source)
        source_key = as_repo_relative(source)
        source_counts[source_key] = len(records)
        for record in records:
            verdict = verify_record(record, finance_gold=finance_gold, asqa_gold=asqa_gold)
            verdict["source_file"] = source_key
            verdicts.append(verdict)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verdict_count": len(verdicts),
        "source_counts": source_counts,
        "passed": sum(1 for verdict in verdicts if verdict["summary"]["passed"]),
        "failed": sum(1 for verdict in verdicts if not verdict["summary"]["passed"]),
        "abstained": sum(1 for verdict in verdicts if verdict["summary"]["abstained"]),
        "false_attribution": sum(1 for verdict in verdicts if verdict["summary"]["false_attribution"]),
    }
    payload = {"summary": summary, "verdicts": verdicts}
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    write_json(output, payload)
    print(json.dumps({"status": "passed", "summary": summary, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
