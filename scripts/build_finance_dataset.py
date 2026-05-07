from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_schema
from src.data_loading import validate_records, write_json, write_jsonl


SEED = 20260416
ABSTAIN = "INSUFFICIENT_SUPPORT"
METRICS = {
    "revenue": "revenue",
    "operating_income": "operating income",
    "cash_reserve": "cash reserve",
}
PERIODS = ["Q1 FY2025", "Q2 FY2025", "Q3 FY2025"]

ISSUERS = [
    "Aster Quill Systems",
    "Aster Quiln Systems",
    "Brindle Vale Robotics",
    "Brindle Vane Robotics",
    "Cobalt Nera Foods",
    "Cobalt Nira Foods",
    "Dovetail Orbis Labs",
    "Dovetail Orbix Labs",
    "Eldin Quartz Media",
    "Eldin Quarte Media",
    "Fennel Arc Devices",
    "Glimmer Nocturne Works",
    "Harbor Vellum Tools",
    "Ivory Lattice Energy",
    "Juniper Solace Textiles",
    "Kestrel Umber Engines",
    "Luma Wisp Analytics",
    "Morrow Sable Components",
    "Nacre Fable Holdings",
    "Orchid Zephyr Foundry",
    "Peregrine Alloy Studio",
    "Quasar Meadow Logistics",
    "Rune Copper Fabricators",
    "Saffron Echo Materials",
    "Trellis Opal Instruments",
    "Umber Finch Networks",
    "Velvet Ion Dynamics",
    "Warden Pearl Devices",
]


def issuer_id(index: int) -> str:
    return f"issuer_{index:03d}"


def metric_answer(value: int) -> str:
    return f"{value} million fictional credits"


def build_passages(seed: int) -> list[dict]:
    rng = random.Random(seed)
    passages: list[dict] = []
    for issuer_index, company_name in enumerate(ISSUERS):
        for period_index, period in enumerate(PERIODS):
            base = 90 + issuer_index * 17 + period_index * 11
            metric_values = {
                "revenue": base + rng.randint(1, 35),
                "operating_income": base // 2 + rng.randint(1, 21),
                "cash_reserve": base + 60 + rng.randint(1, 45),
            }
            passage_id = f"fin_passage_{issuer_index:03d}_{period_index:02d}"
            metric_text = ", ".join(
                f"{label} was {metric_answer(metric_values[key])}" for key, label in METRICS.items()
            )
            passages.append(
                {
                    "record_type": "finance_passage",
                    "passage_id": passage_id,
                    "company_name": company_name,
                    "issuer_id": issuer_id(issuer_index),
                    "period": period,
                    "metric_values": metric_values,
                    "text": (
                        f"Fictional issuer disclosure for {company_name} in {period}: "
                        f"{metric_text}. These values are synthetic and not tied to a real company."
                    ),
                }
            )
    return passages


def by_company_period(passages: list[dict]) -> dict[tuple[str, str], dict]:
    return {(passage["company_name"], passage["period"]): passage for passage in passages}


def make_question(
    question_type: str,
    index: int,
    company_name: str,
    period: str,
    metric_type: str,
    passage: dict | None,
    answerable: bool = True,
    metadata: dict | None = None,
) -> dict:
    label = METRICS[metric_type]
    gold_answer = metric_answer(passage["metric_values"][metric_type]) if answerable and passage else ABSTAIN
    expected = [passage["passage_id"]] if answerable and passage else []
    return {
        "record_type": "finance_question",
        "example_id": f"fin_q_{index:03d}",
        "question_type": question_type,
        "question": f"What was the {label} for {company_name} in {period}?",
        "gold_answer": gold_answer,
        "answerable": answerable,
        "expected_passage_ids": expected,
        "metric_type": metric_type,
        "company_name": company_name,
        "period": period,
        "metadata": metadata or {},
    }


def generate_questions(passages: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    lookup = by_company_period(passages)
    questions: list[dict] = []

    def append(question_type: str, company_index: int, period: str, metric: str, answerable: bool = True, meta=None):
        company = ISSUERS[company_index]
        passage = lookup.get((company, period))
        questions.append(make_question(question_type, len(questions), company, period, metric, passage, answerable, meta))

    for i in range(40):
        append("exact_numeric", i % len(ISSUERS), PERIODS[i % len(PERIODS)], list(METRICS)[i % len(METRICS)])

    for i in range(20):
        company_index = (i + 3) % len(ISSUERS)
        target_period = PERIODS[i % len(PERIODS)]
        conflicting_period = PERIODS[(i + 1) % len(PERIODS)]
        append(
            "wrong_period_trap",
            company_index,
            target_period,
            list(METRICS)[(i + 1) % len(METRICS)],
            True,
            {"conflicting_period": conflicting_period},
        )

    near_duplicate_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    for i in range(15):
        target, distractor = near_duplicate_pairs[i % len(near_duplicate_pairs)]
        append(
            "near_duplicate_issuer_trap",
            target,
            PERIODS[(i + 1) % len(PERIODS)],
            list(METRICS)[(i + 2) % len(METRICS)],
            True,
            {"near_duplicate_company": ISSUERS[distractor]},
        )

    for i in range(15):
        company_index = (i + 11) % len(ISSUERS)
        metric = list(METRICS)[i % len(METRICS)]
        missing_period = "Q4 FY2026" if i % 2 == 0 else "FY2024"
        append("unanswerable", company_index, missing_period, metric, False, {"reason": "period_not_in_corpus"})

    collision_groups = [
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23, 24],
    ]
    for i in range(10):
        group = collision_groups[i % len(collision_groups)]
        target = rng.choice(group)
        distractors = [ISSUERS[item] for item in group if item != target]
        append(
            "retrieval_collision_distractor",
            target,
            PERIODS[(i + 2) % len(PERIODS)],
            list(METRICS)[i % len(METRICS)],
            True,
            {"distractor_companies": distractors},
        )

    return questions


def build_dataset(seed: int) -> tuple[list[dict], list[dict], dict]:
    passages = build_passages(seed)
    questions = generate_questions(passages, seed)
    composition = Counter(question["question_type"] for question in questions)
    manifest = {
        "dataset": "finance",
        "phase": "02_Data_Pipelines_and_Dataset_Contracts",
        "seed": seed,
        "fictional_only": True,
        "issuer_count": len(ISSUERS),
        "periods": PERIODS,
        "metrics": list(METRICS),
        "counts": {
            "passages": len(passages),
            "questions": len(questions),
            "composition": dict(composition),
            "answerable": sum(1 for question in questions if question["answerable"]),
            "unanswerable": sum(1 for question in questions if not question["answerable"]),
        },
        "schemas": {
            "finance_record": "configs/schemas/finance_record.schema.json",
        },
    }
    repeat_questions = generate_questions(build_passages(seed), seed)
    manifest["determinism_check"] = {
        "same_example_ids_and_gold_answers": [
            (question["example_id"], question["gold_answer"]) for question in questions
        ]
        == [(question["example_id"], question["gold_answer"]) for question in repeat_questions],
    }
    return passages, questions, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic synthetic finance stress dataset.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    passages, questions, manifest = build_dataset(args.seed)
    schema = load_schema("finance_record.schema.json")
    validate_records(passages, schema, "finance passage")
    validate_records(questions, schema, "finance question")
    if manifest["counts"]["composition"] != {
        "exact_numeric": 40,
        "wrong_period_trap": 20,
        "near_duplicate_issuer_trap": 15,
        "unanswerable": 15,
        "retrieval_collision_distractor": 10,
    }:
        raise RuntimeError(f"Unexpected finance composition: {manifest['counts']['composition']}")
    if not manifest["determinism_check"]["same_example_ids_and_gold_answers"]:
        raise RuntimeError("Finance determinism check failed")

    paths = {
        "passages": REPO_ROOT / "data" / "finance" / "corpus" / "passages.jsonl",
        "questions": REPO_ROOT / "data" / "finance" / "generated" / "questions.jsonl",
        "manifest": REPO_ROOT / "data" / "finance" / "manifests" / "dataset_manifest.json",
    }
    write_jsonl(paths["passages"], passages)
    write_jsonl(paths["questions"], questions)
    manifest["outputs"] = {key: str(path.relative_to(REPO_ROOT)) for key, path in paths.items() if key != "manifest"}
    write_json(paths["manifest"], manifest)
    print(json.dumps({"status": "passed", "manifest": str(paths["manifest"]), "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
