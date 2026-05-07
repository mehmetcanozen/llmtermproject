# Deterministic Citation Enforcement in RAG

Course project for Large Language Models. This repository implements a local retrieval-augmented generation pipeline that compares three systems on citation-grounded answering:

- `baseline`
- `gate_only`
- `gate_plus_verifier`

The main question is whether an inference-time attention gate and a deterministic verifier can reduce unsupported cited answers without retraining the language model.

## What Is In The Repo

- `src/`: core retrieval, prompting, generation, gate, verifier, evaluation, and final-asset code
- `scripts/`: reproducible entrypoints for data prep, smoke checks, evaluation, and report generation
- `data/`: small fixed ASQA splits/manifests and the synthetic finance dataset used by the project
- `outputs/final/`: final figures, tables, qualitative examples, and the detailed artifact report
- `docs/meta/`: proposal, submission criteria, and implementation notes
- `docs/plans/` and `docs/progress/`: phase-by-phase engineering trail

## Implemented System

The implemented project is a local, Windows-first RAG prototype built around:

- `Qwen2.5-3B-Instruct` for the locked final evaluation path
- a bounded local retrieval setup with dense + BM25 hybrid ranking
- deterministic generation with required sentence-level citations
- a passage-directed attention support gate
- a deterministic verifier that rejects unsupported citation-formatted answers

The proposal originally mentioned a Llama-3 + LangChain/Chroma direction, but the final implementation was adjusted toward a more stable and reproducible local pipeline in this repo.

## Final Result Snapshot

The packaged evaluation status for the GitHub repo is in `outputs/final/final_manifest.json`.

- `formal_full_eval_pass: true`
- locked 3B evaluation covers:
  - `data/asqa/splits/dev_eval_200.jsonl`
  - `data/finance/generated/questions.jsonl`

Headline results from `outputs/final/tables/system_comparison.csv`:

| System | Dataset | Answer Coverage | Unsupported Non-Abstained | Abstention |
| --- | --- | --- | --- | --- |
| baseline | ASQA | 49.5% | 3.5% | 47.0% |
| gate_only | ASQA | 54.5% | 2.5% | 43.0% |
| gate_plus_verifier | ASQA | 54.5% | 0.0% | 45.5% |
| baseline | Finance | 47.0% | 1.0% | 52.0% |
| gate_only | Finance | 47.0% | 2.0% | 51.0% |
| gate_plus_verifier | Finance | 47.0% | 0.0% | 53.0% |

Interpretation: the verifier is the strongest safety mechanism in the final package. It removes unsupported non-abstained outputs in the saved evaluation artifacts, with the usual cost of slightly higher abstention.

Generated distractor stress-test results are packaged separately in `outputs/final/tables/generated_distractor_metrics.csv` and `outputs/final/figures/generated_distractor_robustness.png`. In that regenerated 3B stress test, each prompt includes one additional plausible but irrelevant fourth passage:

| System | Dataset | Unsupported Non-Abstained | Abstention |
| --- | --- | --- | --- |
| baseline | ASQA distractor | 2.0% | 44.0% |
| gate_only | ASQA distractor | 2.5% | 42.5% |
| gate_plus_verifier | ASQA distractor | 0.0% | 45.0% |
| baseline | Finance distractor | 17.0% | 39.0% |
| gate_only | Finance distractor | 17.0% | 44.0% |
| gate_plus_verifier | Finance distractor | 0.0% | 61.0% |

Interpretation: the generated distractor evidence strengthens the meaningful project claim. The verifier does not broadly improve answer quality, but it does convert unsupported cited answers into abstentions under a harder distractor condition.

## How To Run

Use the existing conda environment. Full regeneration expects the local model/data paths in `configs/default.yaml`; bulky regenerated retrieval and run artifacts are intentionally ignored by Git.

```powershell
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\omehm\anaconda3\envs\llm-citation\python.exe' scripts\run_verifier.py
& 'C:\Users\omehm\anaconda3\envs\llm-citation\python.exe' scripts\run_eval_suite.py
& 'C:\Users\omehm\anaconda3\envs\llm-citation\python.exe' scripts\build_final_assets.py
& 'C:\Users\omehm\anaconda3\envs\llm-citation\python.exe' scripts\build_markdown_report.py
& 'C:\Users\omehm\anaconda3\envs\llm-citation\python.exe' -m pytest
```

## Dependencies

- Python: `3.11`
- Main environment lock: `artifacts/preflight/requirements-lock.txt`
- Key libraries: `torch`, `transformers`, `sentence-transformers`, `rank-bm25`, `scipy`, `pytest`

Local model and dataset paths are configured in `configs/default.yaml`.

## Submission-Oriented Files

- Short report draft: `docs/submission/SUBMISSION_REPORT_DRAFT.md`
- Full artifact report: `outputs/final/FINAL_REPORT.md`
- Report asset index: `outputs/final/REPORT_ASSET_INDEX.md`
- Final package manifest: `outputs/final/final_manifest.json`
- Generated distractor stress-test metrics: `outputs/final/tables/generated_distractor_metrics.csv`
- Generated distractor robustness figure: `outputs/final/figures/generated_distractor_robustness.png`
- Sample outputs: `outputs/final/examples/`

## Important Limitations

- ASQA is evaluated as a bounded local-corpus task, not open-web retrieval.
- The ASQA support check is a deterministic proxy, not a human factuality audit.
- The finance dataset is synthetic and should be described as a controlled stress test.
- The normal fixed split remains the authoritative final evaluation. The generated distractor experiment is a separate robustness stress test and should not be described as replacing the formal fixed-split result.
- The older static distractor proxy remains diagnostic only; the generated distractor stress test is the regenerated evidence.

## Testing Notes

`pytest.ini` keeps test discovery inside `tests/` and routes temporary pytest files into `.pytest_tmp/`, so running `pytest` no longer litters the repo root with `pytest-cache-files-*` folders.
