from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loading import read_jsonl, write_json
from src.verifier import load_asqa_gold, load_finance_gold, verify_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic verifier smoke on saved Phase 04/05 outputs.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "verifier" / "verifier_examples.json")
    args = parser.parse_args()

    finance_gold = load_finance_gold(read_jsonl(REPO_ROOT / "data" / "finance" / "generated" / "questions.jsonl"))
    asqa_gold = load_asqa_gold(
        read_jsonl(REPO_ROOT / "data" / "asqa" / "splits" / "train_calibration_100.jsonl")
        + read_jsonl(REPO_ROOT / "data" / "asqa" / "splits" / "dev_eval_200.jsonl")
    )
    sources = [
        REPO_ROOT / "outputs" / "runs" / "baseline_smoke" / "predictions.jsonl",
        REPO_ROOT / "outputs" / "runs" / "gate_smoke" / "predictions.jsonl",
        REPO_ROOT / "outputs" / "runs" / "gate_smoke_7b_check" / "predictions.jsonl",
    ]
    verdicts = []
    source_counts = {}
    for source in sources:
        if not source.exists():
            continue
        records = read_jsonl(source)
        source_counts[str(source.relative_to(REPO_ROOT))] = len(records)
        for record in records:
            verdict = verify_record(record, finance_gold=finance_gold, asqa_gold=asqa_gold)
            verdict["source_file"] = str(source.relative_to(REPO_ROOT))
            verdicts.append(verdict)

    summary = {
        "verdict_count": len(verdicts),
        "source_counts": source_counts,
        "passed": sum(1 for verdict in verdicts if verdict["summary"]["passed"]),
        "failed": sum(1 for verdict in verdicts if not verdict["summary"]["passed"]),
        "abstained": sum(1 for verdict in verdicts if verdict["summary"]["abstained"]),
        "false_attribution": sum(1 for verdict in verdicts if verdict["summary"]["false_attribution"]),
    }
    payload = {"summary": summary, "verdicts": verdicts}
    write_json(args.output, payload)
    print(json.dumps({"status": "passed", "summary": summary, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
