from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "outputs" / "final"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def optional_csv(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def fmt_percent(value: str | None) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value) * 100:.1f}%"


def md_table(rows: list[list[str]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def comparison_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    output = []
    for row in rows:
        output.append(
            [
                row["system"],
                row["dataset"],
                row["model_size"],
                row["example_count"],
                fmt_percent(row.get("answer_coverage")),
                fmt_percent(row.get("unsupported_non_abstained_rate")),
                fmt_percent(row.get("abstention_rate")),
                fmt_percent(row.get("correct_citation_rate")),
                row["artifact_mode"],
            ]
        )
    return output


def distractor_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    return [
        [
            row["system"],
            row["dataset"],
            row["example_count"],
            fmt_percent(row.get("original_hit_rate")),
            fmt_percent(row.get("competing_distractor_proxy_rate")),
            fmt_percent(row.get("same_title_distractor_rate")),
            row["metric_delta_available"],
        ]
        for row in rows
    ]


def repair_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    return [
        [
            row["dataset"],
            row["model_size"],
            row["example_count"],
            row["repair_attempted_count"],
            row["accepted_after_repair_count"],
            row["repair_rejected_abstained_count"],
            fmt_percent(row.get("repair_salvage_rate")),
            row["unsupported_accepted_after_repair_count"],
        ]
        for row in rows
    ]


def example_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    return [
        [
            row["index"],
            row["category"],
            row["dataset"],
            row["system"],
            row["example_id"],
            f"[JSON]({Path(row['path']).relative_to('outputs/final').as_posix()})",
            row["reason"],
        ]
        for row in rows
    ]


def figure_block(filename: str, title: str, caption: str) -> list[str]:
    return [
        f"### {title}",
        "",
        f"![{title}](figures/{filename})",
        "",
        f"*{caption}*",
        "",
    ]


def write_report() -> dict[str, Any]:
    manifest = read_json(FINAL_DIR / "final_manifest.json")
    comparison = read_csv(FINAL_DIR / "tables" / "system_comparison.csv")
    generated_distractor = optional_csv(FINAL_DIR / "tables" / "generated_distractor_metrics.csv")
    repair_salvage = optional_csv(FINAL_DIR / "tables" / "repair_salvage.csv")
    distractor = read_csv(FINAL_DIR / "tables" / "distractor_probe_summary.csv")
    examples = read_csv(FINAL_DIR / "examples" / "example_index.csv")
    report_notes = (FINAL_DIR / "report_notes.md").read_text(encoding="utf-8")
    chosen_config = (REPO_ROOT / "outputs" / "calibration" / "chosen_gate_config.yaml").read_text(encoding="utf-8").strip()
    generated_at = datetime.now(timezone.utc).isoformat()
    formal_full_eval_pass = bool(manifest["formal_full_eval_pass"])
    scope_warning = manifest.get("scope_warning") or "Full locked 3B fixed-split evaluation artifacts are available."

    lines: list[str] = [
        "# Deterministic Citation Enforcement in RAG",
        "",
        "**Final Course Project Report Package**",
        "",
        f"- Generated: `{generated_at}`",
        f"- Package status: `{manifest['package_status']}`",
        f"- Formal full-evaluation pass: `{manifest['formal_full_eval_pass']}`",
        "",
        f"> Evaluation scope note: {scope_warning}",
        "",
        "## Abstract",
        "",
        "This project implements a local retrieval-augmented generation prototype that asks whether an "
        "inference-time attention gate, deterministic verifier, and one-step evidence-only repair loop can reduce unsupported cited answers "
        "without retraining the language model. The system compares a baseline RAG generator, a gate-only "
        "variant, a gate-plus-verifier variant, and a repair-plus-verifier variant. ASQA is used as a bounded local-corpus citation task, "
        "while a synthetic finance dataset provides exact-answer stress tests for numeric and period-specific claims.",
        "",
        "The current artifact bundle demonstrates working retrieval, deterministic generation, "
        "token-level passage-directed attention tracing, deterministic verification, calibration/evaluation "
        "machinery, report tables, figures, qualitative examples, and generated distractor stress-test support when available. Results are interpreted at the scope "
        "declared by the package manifest.",
        "",
        "## Research Question",
        "",
        "Can a local RAG system reduce unsupported cited answers by combining a passage-directed attention gate "
        "with a deterministic post-generation verifier, while preserving answer coverage on supported examples?",
        "",
        "## System Variants",
        "",
        "- `baseline`: deterministic RAG generation with required sentence-level citations.",
        "- `gate_only`: baseline plus token-level passage-directed self-attention tracing and abstention gate.",
        "- `gate_plus_verifier`: gate outputs projected through the deterministic verifier; unsupported outputs are rejected.",
        "- `repair_plus_verifier`: generate, verify, attempt one evidence-only repair when needed, then accept only if the verifier passes.",
        "",
        "## Locked Gate Configuration",
        "",
        "```yaml",
        chosen_config,
        "```",
        "",
        "## Evaluation Scope",
        "",
        "- Planned fixed ASQA evaluation source: `data/asqa/splits/dev_eval_200.jsonl`.",
        "- Planned fixed finance evaluation source: `data/finance/generated/questions.jsonl`.",
        "- Actually scored IDs: `outputs/evaluation/eval_split_ids.json` when the locked evaluation has been run.",
        f"- Final tables in this report carry `formal_full_eval_pass={str(formal_full_eval_pass).lower()}`.",
        "",
        "## Main Results",
        "",
    ]
    lines.extend(
        md_table(
            comparison_rows(comparison),
            [
                "System",
                "Dataset",
                "Model",
                "N",
                "Answer Coverage",
                "Unsupported Non-Abstained",
                "Abstention",
                "Correct Citation",
                "Artifact Mode",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Meaningfulness",
            "",
            "The most meaningful result is verifier-bounded citation safety. The verifier rejects answers whose citation format looks valid but whose cited content fails deterministic support checks. The repair-plus-verifier path then asks a stronger question: can the system recover some answer coverage without relaxing the final safety boundary?",
            "",
            "The project does not show a broad answer-quality breakthrough. Finance exact-answer accuracy remains limited, and ASQA support is evaluated with a deterministic proxy rather than a human factuality audit. The right claim is narrow but useful: the system makes unsupported cited answers more auditable and easier to suppress.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(
        figure_block(
            "unsupported_non_abstained.png",
            "Unsupported Non-Abstained Rate",
            "Lower is better. The 7B gate-only transfer slice shows unsupported cited period errors; the verifier projection removes those false attributions by abstaining.",
        )
    )
    lines.extend(
        figure_block(
            "abstention_vs_coverage.png",
            "Abstention vs Answer Coverage",
            "The report highlights the central tradeoff: stricter rejection reduces unsupported outputs but can increase abstention.",
        )
    )
    lines.extend(
        figure_block(
            "finance_citation_accuracy.png",
            "Finance Citation Accuracy",
            "Finance checks require exact company, metric, period, numeric value, and expected cited passage ID.",
        )
    )
    lines.extend(
        figure_block(
            "safety_vs_coverage_frontier.png",
            "Safety vs Coverage Frontier",
            "Each point plots answer coverage against unsupported non-abstained rate. The intended best region is high coverage with low unsupported output.",
        )
    )
    lines.extend(
        figure_block(
            "repair_funnel.png",
            "Repair Funnel",
            "Counts how many repair-plus-verifier examples passed initially, needed repair, were salvaged, or still abstained.",
        )
    )
    lines.extend(
        figure_block(
            "distractor_sensitivity.png",
            "Distractor Sensitivity Proxy",
            "Static probe summary after adding one plausible irrelevant passage to the prompt input subset. No new model generations are included in this proxy.",
        )
    )
    if generated_distractor:
        lines.extend(
            figure_block(
                "generated_distractor_robustness.png",
                "Generated Distractor Robustness",
                "Real generated runs with a fourth plausible irrelevant passage. This stress test is separate from the normal fixed-split evaluation.",
            )
        )
        lines.extend(["## Generated Distractor Stress Test", ""])
        lines.extend(
            md_table(
                comparison_rows(generated_distractor),
                [
                    "System",
                    "Dataset",
                    "Model",
                    "N",
                    "Answer Coverage",
                    "Unsupported Non-Abstained",
                    "Abstention",
                    "Correct Citation",
                    "Artifact Mode",
                ],
            )
        )
        lines.extend([""])
    if repair_salvage:
        lines.extend(["## Repair Salvage", ""])
        lines.extend(
            md_table(
                repair_rows(repair_salvage),
                [
                    "Dataset",
                    "Model",
                    "N",
                    "Repair Attempts",
                    "Accepted Repairs",
                    "Repair Abstains",
                    "Repair Salvage Rate",
                    "Unsupported Accepted Repairs",
                ],
            )
        )
        lines.extend([""])
    lines.extend(
        [
            "## Distractor Probe Summary",
            "",
        ]
    )
    lines.extend(
        md_table(
            distractor_rows(distractor),
            [
                "System",
                "Dataset",
                "N",
                "Original Hit Rate",
                "Competing Distractor Proxy",
                "Same-Title Distractor",
                "Output Delta Available",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Qualitative Examples",
            "",
        ]
    )
    lines.extend(
        md_table(
            example_rows(examples),
            ["#", "Category", "Dataset", "System", "Example ID", "File", "Why It Matters"],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The evidence supports a careful engineering conclusion rather than a broad benchmark claim. "
            "The pipeline can enforce citation format, capture attention traces, verify finance answers exactly, "
            "and reject observed transfer failure modes where outputs carry right-looking citations with wrong details. "
            "Claims should stay within the fixed-split scope recorded in the manifest.",
            "",
            "The ASQA portion should be read as bounded local-corpus evidence. The support proxy checks explicit "
            "numbers, years, and quoted spans against cited passages, but it does not prove full semantic faithfulness. "
            "The finance portion is stronger for exact checks, but it is synthetic and intentionally fictional.",
            "",
            "## Limitations",
            "",
            f"- Full-eval status: {scope_warning}",
            "- ASQA is evaluated against a bounded local corpus, not open-web retrieval.",
            "- ASQA support verification is a deterministic proxy, not a human factuality audit.",
            "- Finance examples are synthetic stress tests and not real financial evidence.",
            "- The Qwen 7B transfer showed malformed period strings such as `FY2225`; the verifier catches them, but this increases abstention.",
            "- The static distractor probe is prompt-input analysis, not a regenerated output comparison.",
            "- Generated distractor stress-test rows, when present, are robustness evidence and do not replace the normal fixed-split evaluation.",
            "",
            "## Reproducibility",
            "",
            "Run with the local conda environment:",
            "",
            "```powershell",
            "$env:PYTHONNOUSERSITE='1'",
            "& 'C:\\Users\\omehm\\anaconda3\\envs\\llm-citation\\python.exe' scripts\\run_verifier.py",
            "& 'C:\\Users\\omehm\\anaconda3\\envs\\llm-citation\\python.exe' scripts\\run_eval_suite.py",
            "& 'C:\\Users\\omehm\\anaconda3\\envs\\llm-citation\\python.exe' scripts\\build_final_assets.py",
            "& 'C:\\Users\\omehm\\anaconda3\\envs\\llm-citation\\python.exe' scripts\\build_markdown_report.py",
            "& 'C:\\Users\\omehm\\anaconda3\\envs\\llm-citation\\python.exe' -m pytest tests -q",
            "```",
            "",
            "Primary artifacts:",
            "",
            "- `outputs/final/final_manifest.json`",
            "- `outputs/final/tables/system_comparison.csv`",
            "- `outputs/final/tables/coverage_safety_summary.csv`",
            "- `outputs/final/tables/repair_salvage.csv`",
            "- `outputs/final/tables/distractor_probe.csv`",
            "- `outputs/final/tables/generated_distractor_metrics.csv`",
            "- `outputs/final/figures/`",
            "- `outputs/final/examples/`",
            "- `outputs/final/report_notes.md`",
            "",
            "## References",
            "",
            "- Gao et al. 2023, ALCE: https://aclanthology.org/2023.emnlp-main.398/",
            "- Stelmakh et al. 2022, ASQA: https://aclanthology.org/2022.emnlp-main.566/",
            "- Jain and Wallace 2019, Attention is not Explanation: https://aclanthology.org/N19-1357/",
            "- Wiegreffe and Pinter 2019, Attention is not not Explanation: https://aclanthology.org/D19-1002/",
            "- Ding et al. 2025, attention attribution: https://aclanthology.org/2025.findings-acl.21/",
            "- Huang et al. 2025, retrieval heads and faithfulness: https://aclanthology.org/2025.acl-long.826/",
            "- Hugging Face generation docs: https://huggingface.co/docs/transformers/main_classes/text_generation",
            "- Qwen2.5-7B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "- SciPy bootstrap documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
            "- SciPy Wilcoxon documentation: https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.stats.wilcoxon.html",
            "",
            "## Appendix: Report Notes Snapshot",
            "",
            "<details>",
            "<summary>Open detailed notes</summary>",
            "",
            report_notes,
            "",
            "</details>",
            "",
        ]
    )

    report_path = FINAL_DIR / "FINAL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    asset_index_lines = [
        "# Report Asset Index",
        "",
        f"- Generated: `{generated_at}`",
        "- Main report: `outputs/final/FINAL_REPORT.md`",
        "- Notes: `outputs/final/report_notes.md`",
        "- Manifest: `outputs/final/final_manifest.json`",
        "",
        "## Figures",
        "",
        "- `outputs/final/figures/unsupported_non_abstained.png`",
        "- `outputs/final/figures/abstention_vs_coverage.png`",
        "- `outputs/final/figures/finance_citation_accuracy.png`",
        "- `outputs/final/figures/safety_vs_coverage_frontier.png`",
        "- `outputs/final/figures/repair_funnel.png`",
        "- `outputs/final/figures/distractor_sensitivity.png`",
        "- `outputs/final/figures/generated_distractor_robustness.png`" if generated_distractor else "",
        "",
        "## Tables",
        "",
        "- `outputs/final/tables/system_comparison.csv`",
        "- `outputs/final/tables/asqa_metrics.csv`",
        "- `outputs/final/tables/finance_metrics.csv`",
        "- `outputs/final/tables/coverage_safety_summary.csv`",
        "- `outputs/final/tables/repair_salvage.csv`",
        "- `outputs/final/tables/distractor_probe.csv`",
        "- `outputs/final/tables/distractor_probe_summary.csv`",
        "- `outputs/final/tables/generated_distractor_metrics.csv`" if generated_distractor else "",
        "",
        "## Examples",
        "",
        "- `outputs/final/examples/example_index.csv`",
        "- `outputs/final/examples/*.json`",
        "",
    ]
    asset_index_path = FINAL_DIR / "REPORT_ASSET_INDEX.md"
    asset_index_path.write_text("\n".join(asset_index_lines), encoding="utf-8")

    report_manifest = {
        "created_at": generated_at,
        "report": "outputs/final/FINAL_REPORT.md",
        "asset_index": "outputs/final/REPORT_ASSET_INDEX.md",
        "figures": manifest["figures"],
        "tables": manifest["tables"],
        "examples": manifest["examples"],
        "formal_full_eval_pass": manifest["formal_full_eval_pass"],
        "scope_warning": manifest["scope_warning"],
    }
    report_manifest_path = FINAL_DIR / "report_manifest.json"
    report_manifest_path.write_text(json.dumps(report_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return report_manifest


def main() -> int:
    if not (FINAL_DIR / "final_manifest.json").exists():
        print("Missing outputs/final/final_manifest.json. Run scripts/build_final_assets.py first.", file=sys.stderr)
        return 1
    manifest = write_report()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
