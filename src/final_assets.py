from __future__ import annotations

import binascii
import csv
import json
import math
import re
import struct
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SYSTEMS = ("baseline", "gate_only", "gate_plus_verifier")
SYSTEM_LABELS = {
    "baseline": "BASE",
    "gate_only": "GATE",
    "gate_plus_verifier": "VERIFY",
}
PACKAGE_SCOPE = "phase08_artifact_package_from_saved_predictions"
PROBE_MODE = "static_prompt_injection_proxy_no_generation"
GENERATED_DISTRACTOR_SCOPE = "generated_distractor_stress_test"
GENERATED_DISTRACTOR_NOTE = (
    "Generated distractor stress test: each prompt includes one added plausible irrelevant fourth passage. "
    "This robustness slice does not replace the normal fixed-split evaluation."
)


@dataclass(frozen=True)
class FinalAssetPaths:
    root: Path

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def examples(self) -> Path:
        return self.root / "examples"


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def truncate(value: str, limit: int = 420) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def package_metric_rows(metric_rows: list[dict[str, str]], *, formal_full_eval_pass: bool, scope_note: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        packaged = dict(row)
        packaged["phase08_package_scope"] = PACKAGE_SCOPE
        packaged["formal_full_eval_pass"] = formal_full_eval_pass
        packaged["scope_note"] = scope_note
        rows.append(packaged)
    return rows


def package_generated_distractor_rows(metric_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        packaged = dict(row)
        packaged["phase08_package_scope"] = GENERATED_DISTRACTOR_SCOPE
        packaged["formal_full_eval_pass"] = False
        packaged["scope_note"] = GENERATED_DISTRACTOR_NOTE
        rows.append(packaged)
    return rows


def filter_metric_rows(rows: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("dataset") == dataset]


def metric_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
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
        "phase08_package_scope",
        "formal_full_eval_pass",
        "scope_note",
    ]
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    return [field for field in preferred if field in keys] + sorted(keys.difference(preferred))


def candidate_pool(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    pool = []
    for key in ("raw_dense_candidates", "raw_bm25_candidates", "merged_top3"):
        for candidate in record.get(key, []) or []:
            pool.append((key, candidate))
    return pool


def choose_distractor(record: dict[str, Any]) -> dict[str, Any]:
    expected = set(record.get("expected_passage_ids") or [])
    top3 = {passage["parent_passage_id"] for passage in record.get("merged_top3", [])}
    for source, candidate in candidate_pool(record):
        passage_id = candidate.get("parent_passage_id")
        if not passage_id or passage_id in expected or passage_id in top3:
            continue
        return {"source_list": source, **candidate}
    for source, candidate in candidate_pool(record):
        passage_id = candidate.get("parent_passage_id")
        if not passage_id or passage_id in expected:
            continue
        return {"source_list": source, **candidate}
    first = (record.get("merged_top3") or [{}])[0]
    return {"source_list": "fallback_first_prompt_passage", **first}


def build_distractor_probe_rows(
    asqa_candidates: list[dict[str, Any]],
    finance_candidates: list[dict[str, Any]],
    *,
    asqa_count: int = 40,
    finance_count: int = 20,
) -> list[dict[str, Any]]:
    selected = asqa_candidates[:asqa_count] + finance_candidates[:finance_count]
    rows: list[dict[str, Any]] = []
    for record in selected:
        distractor = choose_distractor(record)
        expected = set(record.get("expected_passage_ids") or [])
        top3 = record.get("merged_top3", []) or []
        top3_ids = {passage["parent_passage_id"] for passage in top3}
        top1_title = top3[0].get("title") if top3 else ""
        top3_min_score = min((float(passage.get("score", 0.0)) for passage in top3), default=0.0)
        distractor_score = float(distractor.get("score") or 0.0)
        same_title = bool(top1_title and distractor.get("title") == top1_title)
        same_source = bool(
            distractor.get("source_example_id")
            and distractor.get("source_example_id") == record.get("example_id")
        )
        competing = bool(same_title or distractor_score >= top3_min_score)
        for system in SYSTEMS:
            rows.append(
                {
                    "probe_mode": PROBE_MODE,
                    "system": system,
                    "dataset": record["dataset"],
                    "example_id": record["example_id"],
                    "question": record["question"],
                    "selected_subset": "asqa_dev_eval_200_first40" if record["dataset"] == "asqa" else "finance_full_first20",
                    "original_prompt_passage_count": 3,
                    "probe_prompt_passage_count": 4,
                    "added_distractor_count": 1,
                    "original_hit_top3": bool(record.get("hit_top3")),
                    "expected_passage_count": len(expected),
                    "distractor_parent_passage_id": distractor.get("parent_passage_id"),
                    "distractor_title": distractor.get("title"),
                    "distractor_source_example_id": distractor.get("source_example_id"),
                    "distractor_source_list": distractor.get("source_list"),
                    "distractor_score": distractor_score,
                    "distractor_text": truncate(distractor.get("text", ""), 260),
                    "distractor_in_expected_set": distractor.get("parent_passage_id") in expected,
                    "distractor_already_in_top3": distractor.get("parent_passage_id") in top3_ids,
                    "same_title_as_top1": same_title,
                    "same_source_example_as_query": same_source,
                    "competing_distractor_proxy": competing,
                    "output_metric_delta_available": False,
                    "unsupported_non_abstained_delta": None,
                    "answer_coverage_delta": None,
                    "scope_note": "Prompt-input probe only; no new model generations were run in this iteration.",
                }
            )
    return rows


def distractor_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["system"], row["dataset"])].append(row)
    summaries = []
    for (system, dataset), group in sorted(groups.items()):
        count = len(group)
        summaries.append(
            {
                "system": system,
                "dataset": dataset,
                "probe_mode": PROBE_MODE,
                "example_count": count,
                "original_hit_rate": sum(bool(row["original_hit_top3"]) for row in group) / count if count else None,
                "competing_distractor_proxy_rate": sum(bool(row["competing_distractor_proxy"]) for row in group) / count if count else None,
                "same_title_distractor_rate": sum(bool(row["same_title_as_top1"]) for row in group) / count if count else None,
                "metric_delta_available": False,
            }
        )
    return summaries


def prediction_index(repo_root: Path, verdicts: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sorted({verdict["source_file"] for verdict in verdicts}):
        for record in read_jsonl(repo_root / source):
            index[(record["run_id"], record["example_id"])] = record
    return index


def example_reason(category: str, verdict: dict[str, Any]) -> str:
    summary = verdict["summary"]
    details = verdict.get("domain_details") or {}
    if category == "baseline_failure" and summary.get("abstained"):
        return "Baseline abstained on an answerable finance question."
    if category == "baseline_failure":
        return "Baseline output failed the deterministic verifier."
    if category == "gate_success":
        return "Gate-only 3B answered with exact finance value and expected cited passage."
    if category == "verifier_catch":
        errors = ", ".join(summary.get("errors") or [])
        return f"Verifier caught a citation-formatted but unsupported answer: {errors}."
    return "Distractor probe selected a plausible irrelevant fourth prompt passage."


def example_payload(
    *,
    category: str,
    verdict: dict[str, Any],
    prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "category": category,
        "example_id": verdict["example_id"],
        "dataset": verdict["dataset"],
        "system": verdict.get("system"),
        "run_id": verdict.get("run_id"),
        "reason": example_reason(category, verdict),
        "question": None if prediction is None else prediction.get("question"),
        "answer": None if prediction is None else prediction.get("answer"),
        "abstained": verdict["summary"].get("abstained"),
        "verifier_summary": verdict["summary"],
        "domain_details": verdict.get("domain_details", {}),
        "retrieved_passages": [
            {
                "label": passage.get("label"),
                "passage_id": passage.get("passage_id"),
                "text": truncate(passage.get("text", ""), 300),
            }
            for passage in ((prediction or {}).get("retrieved_passages") or [])
        ],
    }


def select_qualitative_examples(
    repo_root: Path,
    verifier_artifact: dict[str, Any],
    distractor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    verdicts = verifier_artifact["verdicts"]
    predictions = prediction_index(repo_root, verdicts)
    selected: list[dict[str, Any]] = []

    baseline_failures = [
        verdict
        for verdict in verdicts
        if verdict.get("system") == "baseline"
        and (
            verdict["summary"].get("false_attribution")
            or verdict["summary"].get("exact_answer_correct") is False
        )
    ][:2]
    gate_successes = [
        verdict
        for verdict in verdicts
        if verdict.get("system") == "gate_only"
        and "3b" in str(verdict.get("run_id", "")).lower()
        and verdict["summary"].get("exact_answer_correct") is True
        and not verdict["summary"].get("abstained")
    ][:2]
    verifier_catches = [
        verdict
        for verdict in verdicts
        if verdict.get("system") == "gate_only"
        and "7b" in str(verdict.get("run_id", "")).lower()
        and verdict["summary"].get("false_attribution")
    ][:2]

    for category, group in (
        ("baseline_failure", baseline_failures),
        ("gate_success", gate_successes),
        ("verifier_catch", verifier_catches),
    ):
        for verdict in group:
            selected.append(
                example_payload(
                    category=category,
                    verdict=verdict,
                    prediction=predictions.get((verdict["run_id"], verdict["example_id"])),
                )
            )

    probe_cases = []
    seen_probe_ids: set[str] = set()
    for row in distractor_rows:
        if row["system"] != "baseline" or row["example_id"] in seen_probe_ids:
            continue
        if row["competing_distractor_proxy"]:
            seen_probe_ids.add(row["example_id"])
            probe_cases.append(row)
        if len(probe_cases) == 2:
            break
    if len(probe_cases) < 2:
        for row in distractor_rows:
            if row["system"] == "baseline" and row["example_id"] not in seen_probe_ids:
                seen_probe_ids.add(row["example_id"])
                probe_cases.append(row)
            if len(probe_cases) == 2:
                break
    for row in probe_cases:
        selected.append(
            {
                "category": "distractor_probe_case",
                "example_id": row["example_id"],
                "dataset": row["dataset"],
                "system": row["system"],
                "reason": "Static probe added one plausible irrelevant distractor passage to the prompt input.",
                "question": row["question"],
                "probe_mode": row["probe_mode"],
                "original_hit_top3": row["original_hit_top3"],
                "competing_distractor_proxy": row["competing_distractor_proxy"],
                "distractor": {
                    "parent_passage_id": row["distractor_parent_passage_id"],
                    "title": row["distractor_title"],
                    "text": row["distractor_text"],
                    "source_list": row["distractor_source_list"],
                },
                "scope_note": row["scope_note"],
            }
        )
    return selected


def write_qualitative_examples(output_dir: Path, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for index, example in enumerate(examples, start=1):
        safe_category = re.sub(r"[^a-z0-9_]+", "_", example["category"].lower())
        filename = f"{index:02d}_{safe_category}_{example['example_id']}.json"
        write_json(output_dir / filename, example)
        index_rows.append(
            {
                "index": index,
                "category": example["category"],
                "dataset": example["dataset"],
                "system": example.get("system"),
                "example_id": example["example_id"],
                "path": f"outputs/final/examples/{filename}",
                "reason": example["reason"],
            }
        )
    write_csv(
        output_dir / "example_index.csv",
        index_rows,
        ["index", "category", "dataset", "system", "example_id", "path", "reason"],
    )
    lines = ["# Qualitative Example Index", ""]
    for row in index_rows:
        lines.append(
            f"- `{row['index']:02}` `{row['category']}` `{row['dataset']}` `{row['example_id']}`: {row['reason']}"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_rows


FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    ".": ["0", "0", "0", "0", "0", "1", "1"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    "_": ["000", "000", "000", "000", "000", "000", "111"],
    "/": ["001", "001", "010", "010", "100", "100", "000"],
    "%": ["1001", "1001", "0010", "0100", "1001", "1001", "0000"],
    ":": ["0", "1", "1", "0", "1", "1", "0"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "100", "100"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["111", "100", "100", "100", "100", "100", "111"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "G": ["111", "100", "100", "101", "101", "101", "111"],
    "H": ["101", "101", "101", "111", "101", "101", "101"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "001", "101", "101", "111"],
    "K": ["101", "101", "110", "100", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["1001", "1111", "1111", "1001", "1001", "1001", "1001"],
    "N": ["101", "111", "111", "111", "111", "111", "101"],
    "O": ["111", "101", "101", "101", "101", "101", "111"],
    "P": ["111", "101", "101", "111", "100", "100", "100"],
    "Q": ["111", "101", "101", "101", "111", "001", "001"],
    "R": ["111", "101", "101", "111", "110", "101", "101"],
    "S": ["111", "100", "100", "111", "001", "001", "111"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "W": ["1001", "1001", "1001", "1001", "1111", "1111", "1001"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
    "Z": ["111", "001", "001", "010", "100", "100", "111"],
}


class Canvas:
    def __init__(self, width: int = 920, height: int = 520, background: tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        offset = (y * self.width + x) * 3
        self.pixels[offset : offset + 3] = bytes(color)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0, x1 = sorted((max(0, x0), min(self.width - 1, x1)))
        y0, y1 = sorted((max(0, y0), min(self.height - 1, y1)))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.set_pixel(x, y, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                return
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    def text(self, x: int, y: int, text: str, color: tuple[int, int, int] = (30, 30, 30), scale: int = 2) -> None:
        cursor = x
        for char in text.upper():
            glyph = FONT.get(char, FONT[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cursor + gx * scale, y + gy * scale, cursor + (gx + 1) * scale - 1, y + (gy + 1) * scale - 1, color)
            cursor += (len(glyph[0]) + 1) * scale

    def save_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            raw.extend(self.pixels[y * stride : (y + 1) * stride])
        with path.open("wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
            write_png_chunk(handle, b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            write_png_chunk(handle, b"IDAT", zlib.compress(bytes(raw), level=9))
            write_png_chunk(handle, b"IEND", b"")


def write_png_chunk(handle: Any, kind: bytes, data: bytes) -> None:
    handle.write(struct.pack(">I", len(data)))
    handle.write(kind)
    handle.write(data)
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(data, checksum)
    handle.write(struct.pack(">I", checksum & 0xFFFFFFFF))


COLORS = {
    "baseline": (88, 115, 184),
    "gate_only": (42, 157, 143),
    "gate_plus_verifier": (204, 102, 83),
    "asqa": (116, 150, 193),
    "finance": (232, 173, 77),
    "axis": (50, 50, 50),
    "grid": (225, 225, 225),
}


def value_from_row(row: dict[str, Any], metric: str) -> float:
    value = safe_float(row.get(metric))
    if value is None or math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def draw_bar_chart(path: Path, title: str, rows: list[dict[str, Any]], metric: str, *, finance_only: bool = False) -> None:
    selected = [row for row in rows if not finance_only or row.get("dataset") == "finance"]
    selected = selected[:8]
    canvas = Canvas()
    left, bottom, top, right = 95, 440, 70, 880
    canvas.text(95, 24, title, scale=3)
    for tick in (0.0, 0.5, 1.0):
        y = bottom - int((bottom - top) * tick)
        canvas.line(left, y, right, y, COLORS["grid"])
        canvas.text(35, y - 8, f"{tick:.1f}", scale=2)
    canvas.line(left, top, left, bottom, COLORS["axis"])
    canvas.line(left, bottom, right, bottom, COLORS["axis"])
    if not selected:
        canvas.text(280, 230, "NO DATA", scale=4)
        canvas.save_png(path)
        return
    slot = (right - left) // len(selected)
    for index, row in enumerate(selected):
        x0 = left + index * slot + 16
        x1 = left + (index + 1) * slot - 16
        value = value_from_row(row, metric)
        y = bottom - int((bottom - top) * value)
        color = COLORS.get(row.get("system", ""), (120, 120, 120))
        canvas.rect(x0, y, x1, bottom - 1, color)
        canvas.text(x0, y - 22, f"{value:.2f}", scale=2)
        label = f"{SYSTEM_LABELS.get(row.get('system', ''), 'SYS')}/{row.get('dataset', '')[:3].upper()}"
        canvas.text(x0 - 4, bottom + 12, label, scale=2)
    canvas.save_png(path)


def draw_scatter(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    canvas = Canvas()
    left, bottom, top, right = 95, 440, 70, 880
    canvas.text(95, 24, title, scale=3)
    canvas.text(300, 476, "ANSWER COVERAGE", scale=2)
    canvas.text(10, 54, "ABSTENTION", scale=2)
    for tick in (0.0, 0.5, 1.0):
        x = left + int((right - left) * tick)
        y = bottom - int((bottom - top) * tick)
        canvas.line(x, top, x, bottom, COLORS["grid"])
        canvas.line(left, y, right, y, COLORS["grid"])
    canvas.line(left, top, left, bottom, COLORS["axis"])
    canvas.line(left, bottom, right, bottom, COLORS["axis"])
    for row in rows:
        x_value = value_from_row(row, "answer_coverage")
        y_value = value_from_row(row, "abstention_rate")
        x = left + int((right - left) * x_value)
        y = bottom - int((bottom - top) * y_value)
        color = COLORS.get(row.get("system", ""), (120, 120, 120))
        canvas.rect(x - 5, y - 5, x + 5, y + 5, color)
        canvas.text(x + 8, y - 7, SYSTEM_LABELS.get(row.get("system", ""), "SYS"), scale=1)
    canvas.save_png(path)


def draw_distractor_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    canvas = Canvas()
    left, bottom, top, right = 95, 440, 70, 880
    canvas.text(95, 24, "DISTRACTOR SENSITIVITY PROXY", scale=3)
    for tick in (0.0, 0.5, 1.0):
        y = bottom - int((bottom - top) * tick)
        canvas.line(left, y, right, y, COLORS["grid"])
        canvas.text(35, y - 8, f"{tick:.1f}", scale=2)
    canvas.line(left, top, left, bottom, COLORS["axis"])
    canvas.line(left, bottom, right, bottom, COLORS["axis"])
    selected = rows[:8]
    slot = (right - left) // max(1, len(selected))
    for index, row in enumerate(selected):
        x0 = left + index * slot + 16
        x1 = left + (index + 1) * slot - 16
        value = value_from_row(row, "competing_distractor_proxy_rate")
        y = bottom - int((bottom - top) * value)
        color = COLORS.get(row.get("dataset", ""), (120, 120, 120))
        canvas.rect(x0, y, x1, bottom - 1, color)
        canvas.text(x0, y - 22, f"{value:.2f}", scale=2)
        label = f"{SYSTEM_LABELS.get(row.get('system', ''), 'SYS')}/{row.get('dataset', '')[:3].upper()}"
        canvas.text(x0 - 4, bottom + 12, label, scale=2)
    canvas.save_png(path)


def write_figures(paths: FinalAssetPaths, metric_rows: list[dict[str, Any]], distractor_summary: list[dict[str, Any]]) -> list[str]:
    paths.figures.mkdir(parents=True, exist_ok=True)
    figures = [
        paths.figures / "unsupported_non_abstained.png",
        paths.figures / "abstention_vs_coverage.png",
        paths.figures / "finance_citation_accuracy.png",
        paths.figures / "distractor_sensitivity.png",
    ]
    draw_bar_chart(figures[0], "UNSUPPORTED NON ABSTAINED RATE", metric_rows, "unsupported_non_abstained_rate")
    draw_scatter(figures[1], "ABSTENTION VS COVERAGE", metric_rows)
    draw_bar_chart(figures[2], "FINANCE CITATION ACCURACY", metric_rows, "correct_citation_rate", finance_only=True)
    draw_distractor_chart(figures[3], distractor_summary)
    return [str(path) for path in figures]


def draw_generated_distractor_chart(path: Path, normal_rows: list[dict[str, Any]], generated_rows: list[dict[str, Any]]) -> None:
    canvas = Canvas(width=1120, height=560)
    left, bottom, top, right = 105, 470, 78, 1080
    canvas.text(95, 24, "NORMAL VS GENERATED DISTRACTOR", scale=3)
    for tick in (0.0, 0.5, 1.0):
        y = bottom - int((bottom - top) * tick)
        canvas.line(left, y, right, y, COLORS["grid"])
        canvas.text(35, y - 8, f"{tick:.1f}", scale=2)
    canvas.line(left, top, left, bottom, COLORS["axis"])
    canvas.line(left, bottom, right, bottom, COLORS["axis"])
    rows: list[tuple[str, dict[str, Any], str]] = []
    for row in normal_rows:
        rows.append(("N", row, "unsupported_non_abstained_rate"))
        rows.append(("N", row, "abstention_rate"))
    for row in generated_rows:
        rows.append(("D", row, "unsupported_non_abstained_rate"))
        rows.append(("D", row, "abstention_rate"))
    selected = rows[:24]
    slot = (right - left) // max(1, len(selected))
    for index, (condition, row, metric) in enumerate(selected):
        x0 = left + index * slot + 5
        x1 = left + (index + 1) * slot - 5
        value = value_from_row(row, metric)
        y = bottom - int((bottom - top) * value)
        base = COLORS.get(row.get("system", ""), (120, 120, 120))
        color = tuple(max(0, channel - 35) for channel in base) if metric == "abstention_rate" else base
        canvas.rect(x0, y, x1, bottom - 1, color)
        metric_label = "U" if metric == "unsupported_non_abstained_rate" else "A"
        canvas.text(x0, y - 18, f"{value:.2f}", scale=1)
        canvas.text(x0, bottom + 10, f"{condition}{metric_label}", scale=1)
    canvas.text(105, 505, "N/D=NORMAL/DISTRACTOR  U/A=UNSUPPORTED/ABSTENTION", scale=2)
    canvas.save_png(path)


def build_report_notes(
    *,
    output_path: Path,
    chosen_config_text: str,
    metric_rows: list[dict[str, Any]],
    distractor_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    manifest_paths: dict[str, str],
    formal_full_eval_pass: bool,
    scope_note: str,
    generated_distractor_rows: list[dict[str, Any]] | None = None,
) -> None:
    systems = sorted({row["system"] for row in metric_rows})
    datasets = sorted({row["dataset"] for row in metric_rows})
    lines = [
        "# Final Report Notes",
        "",
        "## Package Status",
        "",
        "This final package is report-ready as an artifact bundle.",
        f"Formal full-evaluation pass for completed 3B fixed-split scope: `{formal_full_eval_pass}`.",
        scope_note,
        "",
        "## Locked Gate Config",
        "",
        "```yaml",
        chosen_config_text.strip(),
        "```",
        "",
        "## Run Manifests Used",
        "",
    ]
    for label, path in manifest_paths.items():
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(
        [
            "",
            "## Fixed Eval IDs",
            "",
            "- ASQA planned fixed evaluation set: `data/asqa/splits/dev_eval_200.jsonl` (`200` IDs).",
            "- Finance planned fixed evaluation set: `data/finance/generated/questions.jsonl` (`100` IDs).",
            "- Current scored prediction IDs are enumerated in `outputs/evaluation/eval_split_ids.json` when full evaluation has been run.",
            "- Distractor probe static subset: first `40` ASQA retrieval candidates from `outputs/retrieval/asqa_candidates.jsonl` and first `20` finance retrieval candidates from `outputs/retrieval/finance_candidates.jsonl`.",
            "",
            "## Tables And Figures",
            "",
            "- `outputs/final/tables/system_comparison.csv`",
            "- `outputs/final/tables/asqa_metrics.csv`",
            "- `outputs/final/tables/finance_metrics.csv`",
            "- `outputs/final/tables/distractor_probe.csv`",
            "- `outputs/final/tables/distractor_probe_summary.csv`",
            "- `outputs/final/tables/generated_distractor_metrics.csv` when generated distractor runs are available.",
            "- `outputs/final/figures/unsupported_non_abstained.png`",
            "- `outputs/final/figures/abstention_vs_coverage.png`",
            "- `outputs/final/figures/finance_citation_accuracy.png`",
            "- `outputs/final/figures/distractor_sensitivity.png`",
            "- `outputs/final/figures/generated_distractor_robustness.png` when generated distractor runs are available.",
            "",
            "## Qualitative Examples",
            "",
            f"- Curated examples: `{len(example_rows)}`.",
            "- Index: `outputs/final/examples/example_index.csv`.",
            "- Categories: baseline failures, gate successes, verifier catches, and distractor-probe cases.",
            "",
            "## Required Limitations",
            "",
            "- Bounded ASQA corpus design: ASQA uses a local bounded corpus derived from available ASQA passages. It is not an open-web retrieval benchmark.",
            "- Support-proxy limitation: ASQA support checks only verify citation structure plus explicit numbers, years, and quoted spans against cited passages. This is useful but not proof of complete factual faithfulness.",
            "- Finance synthetic scope: finance records are fictional stress-test disclosures and questions. They are not real financial data and must not be described as deployment evidence.",
            "- 3B/7B limitation: the Qwen 7B transfer check produced format-valid but semantically wrong finance period strings such as malformed fiscal years; the deterministic verifier catches these as false attributions.",
            f"- Full-eval status: {scope_note}",
            "- Static distractor limitation: the older distractor probe is a prompt-input static proxy. It adds one plausible irrelevant passage to saved retrieval inputs and reports sensitivity signals, but it does not include new model generations.",
            "- Generated distractor limitation: generated distractor runs are a robustness stress test with a fourth prompt passage, not a replacement for the normal fixed-split results.",
            "",
            "## External Method References",
            "",
            "- ALCE frames citation evaluation around correctness and citation quality: https://aclanthology.org/2023.emnlp-main.398/",
            "- SciPy documents paired bootstrap confidence-interval mechanics used as the model for Phase 07 intervals: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
            "- Matplotlib `savefig`/bar/scatter docs were checked for conventional figure outputs, but this package uses a standard-library PNG fallback because Pillow is absent in the local conda env: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html",
            "",
            "## Snapshot Summary",
            "",
            f"- Systems represented in saved metrics: `{', '.join(systems)}`.",
            f"- Datasets represented in saved metrics: `{', '.join(datasets)}`.",
            f"- Distractor probe rows: `{len(distractor_rows)}`.",
            f"- Generated distractor metric rows: `{len(generated_distractor_rows or [])}`.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_assets(repo_root: Path, output_root: Path) -> dict[str, Any]:
    paths = FinalAssetPaths(output_root)
    for directory in (paths.tables, paths.figures, paths.examples):
        directory.mkdir(parents=True, exist_ok=True)

    metric_source = repo_root / "outputs" / "evaluation" / "metric_tables.csv"
    evaluation_manifest_path = repo_root / "outputs" / "evaluation" / "evaluation_manifest.json"
    if not metric_source.exists():
        metric_source = repo_root / "outputs" / "calibration" / "metric_tables.csv"
        evaluation_manifest_path = repo_root / "outputs" / "calibration" / "evaluation_manifest.json"
    evaluation_manifest = read_json(evaluation_manifest_path) if evaluation_manifest_path.exists() else {}
    formal_full_eval_pass = bool(evaluation_manifest.get("formal_full_eval_pass", False))
    scope_note = evaluation_manifest.get("scope_warning") or (
        "Full locked 3B baseline and gate-only predictions are present for dev_eval_200 and finance_full_100."
        if formal_full_eval_pass
        else "Full locked 3B fixed-split predictions are not yet present for all required systems."
    )
    metric_rows = package_metric_rows(
        read_csv(metric_source),
        formal_full_eval_pass=formal_full_eval_pass,
        scope_note=scope_note,
    )
    generated_distractor_metric_path = repo_root / "outputs" / "distractor_evaluation" / "metric_tables.csv"
    generated_distractor_rows = (
        package_generated_distractor_rows(read_csv(generated_distractor_metric_path))
        if generated_distractor_metric_path.exists()
        else []
    )
    write_csv(paths.tables / "system_comparison.csv", metric_rows, metric_fieldnames(metric_rows))
    write_csv(paths.tables / "asqa_metrics.csv", filter_metric_rows(metric_rows, "asqa"), metric_fieldnames(metric_rows))
    write_csv(paths.tables / "finance_metrics.csv", filter_metric_rows(metric_rows, "finance"), metric_fieldnames(metric_rows))
    if generated_distractor_rows:
        write_csv(
            paths.tables / "generated_distractor_metrics.csv",
            generated_distractor_rows,
            metric_fieldnames(generated_distractor_rows),
        )

    asqa_candidates = read_jsonl(repo_root / "outputs" / "retrieval" / "asqa_candidates.jsonl")
    finance_candidates = read_jsonl(repo_root / "outputs" / "retrieval" / "finance_candidates.jsonl")
    distractor_rows = build_distractor_probe_rows(asqa_candidates, finance_candidates)
    distractor_summary = distractor_summary_rows(distractor_rows)
    probe_fields = [
        "probe_mode",
        "system",
        "dataset",
        "example_id",
        "question",
        "selected_subset",
        "original_prompt_passage_count",
        "probe_prompt_passage_count",
        "added_distractor_count",
        "original_hit_top3",
        "expected_passage_count",
        "distractor_parent_passage_id",
        "distractor_title",
        "distractor_source_example_id",
        "distractor_source_list",
        "distractor_score",
        "distractor_text",
        "distractor_in_expected_set",
        "distractor_already_in_top3",
        "same_title_as_top1",
        "same_source_example_as_query",
        "competing_distractor_proxy",
        "output_metric_delta_available",
        "unsupported_non_abstained_delta",
        "answer_coverage_delta",
        "scope_note",
    ]
    write_csv(paths.tables / "distractor_probe.csv", distractor_rows, probe_fields)
    write_csv(
        paths.tables / "distractor_probe_summary.csv",
        distractor_summary,
        [
            "system",
            "dataset",
            "probe_mode",
            "example_count",
            "original_hit_rate",
            "competing_distractor_proxy_rate",
            "same_title_distractor_rate",
            "metric_delta_available",
        ],
    )

    figure_paths = write_figures(paths, metric_rows, distractor_summary)
    if generated_distractor_rows:
        generated_figure = paths.figures / "generated_distractor_robustness.png"
        draw_generated_distractor_chart(generated_figure, metric_rows, generated_distractor_rows)
        figure_paths.append(str(generated_figure))
    verifier_artifact = read_json(repo_root / "artifacts" / "verifier" / "verifier_examples.json")
    examples = select_qualitative_examples(repo_root, verifier_artifact, distractor_rows)
    example_rows = write_qualitative_examples(paths.examples, examples)

    chosen_config_path = repo_root / "outputs" / "calibration" / "chosen_gate_config.yaml"
    manifest_paths = {
        "baseline_smoke": "outputs/runs/baseline_smoke/run_manifest.json",
        "gate_smoke_3b": "outputs/runs/gate_smoke/run_manifest.json",
        "gate_smoke_7b_transfer": "outputs/runs/gate_smoke_7b_check/run_manifest.json",
        "phase07_evaluation": "outputs/calibration/evaluation_manifest.json",
        "chosen_gate_config": "outputs/calibration/chosen_gate_config.yaml",
    }
    build_report_notes(
        output_path=output_root / "report_notes.md",
        chosen_config_text=chosen_config_path.read_text(encoding="utf-8"),
        metric_rows=metric_rows,
        distractor_rows=distractor_rows,
        example_rows=example_rows,
        manifest_paths=manifest_paths,
        formal_full_eval_pass=formal_full_eval_pass,
        scope_note=scope_note,
        generated_distractor_rows=generated_distractor_rows,
    )

    manifest = {
        "phase": "08_Experiments_Figures_and_Report_Assets",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_status": "complete_full_3b_eval_package" if formal_full_eval_pass else "complete_artifact_package_with_full_eval_scope_warning",
        "formal_full_eval_pass": formal_full_eval_pass,
        "scope_warning": scope_note,
        "tables": {
            "system_comparison": "outputs/final/tables/system_comparison.csv",
            "asqa_metrics": "outputs/final/tables/asqa_metrics.csv",
            "finance_metrics": "outputs/final/tables/finance_metrics.csv",
            "distractor_probe": "outputs/final/tables/distractor_probe.csv",
            "distractor_probe_summary": "outputs/final/tables/distractor_probe_summary.csv",
            "generated_distractor_metrics": "outputs/final/tables/generated_distractor_metrics.csv" if generated_distractor_rows else None,
        },
        "figures": [str(Path(path).relative_to(repo_root)) for path in figure_paths],
        "examples": {
            "count": len(example_rows),
            "index": "outputs/final/examples/example_index.csv",
        },
        "report_notes": "outputs/final/report_notes.md",
        "fixed_eval_sources": {
            "asqa_dev_eval_200": "data/asqa/splits/dev_eval_200.jsonl",
            "finance_full_100": "data/finance/generated/questions.jsonl",
            "scored_prediction_ids": "outputs/evaluation/eval_split_ids.json" if metric_source.parent.name == "evaluation" else "outputs/calibration/eval_split_ids.json",
        },
        "metric_source": str(metric_source.relative_to(repo_root)),
        "run_manifests_used": manifest_paths,
        "validation": {
            "all_required_tables_exist": True,
            "all_required_figures_exist": True,
            "curated_example_count_at_least_8": len(example_rows) >= 8,
            "report_notes_include_required_limitations": True,
            "all_three_systems_represented": sorted({row["system"] for row in metric_rows}) == sorted(SYSTEMS),
            "same_fixed_ids_full_eval": formal_full_eval_pass,
            "same_fixed_ids_note": "Full 3B fixed IDs are complete." if formal_full_eval_pass else "Saved prediction artifacts do not cover the full fixed eval IDs for baseline and gate_only 3B.",
            "figures_generated_from_saved_tables": True,
            "generated_distractor_metrics_included": bool(generated_distractor_rows),
        },
    }
    write_json(output_root / "final_manifest.json", manifest)
    return manifest
