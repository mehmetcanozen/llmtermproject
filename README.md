# Deterministic Citation Enforcement in RAG

Course project for a Large Language Models class. This repository implements a local retrieval-augmented generation pipeline for citation-grounded answering.

The final project question is:

> Can a RAG system recover more useful cited answers while still rejecting unsupported citations?

The strongest final system is `repair_plus_verifier`: generate a cited answer, verify support, attempt one evidence-only repair if needed, verify again, and abstain if the repaired answer still fails.

## What Is Included

- `src/`: retrieval, prompting, generation, attention gate, verifier, evaluation, and final asset code
- `scripts/`: reproducible command-line entrypoints
- `tests/`: unit and contract tests
- `configs/default.yaml`: local configuration used by the original Windows machine
- `configs/local.example.yaml`: template for other machines
- `data/`: small fixed ASQA splits/manifests and the synthetic finance dataset
- `outputs/final/`: final lightweight tables, figures, examples, and report notes
- `outputs/evaluation/`: saved verifier/evaluation artifacts for artifact-level reproduction
- `docs/RUNNING.md`: fresh-clone setup and reproduction guide
- `docs/submission/`: submission guide and IEEE-style report package

Large/generated assets are intentionally not included: model weights, ASQA parquet files, dense embedding `.npz` files, and full generation run folders.

## Final Result Snapshot

The fixed evaluation uses:

- ASQA: `data/asqa/splits/dev_eval_200.jsonl` with 200 examples
- Finance: `data/finance/generated/questions.jsonl` with 100 examples

Headline fixed-split results:

| System | Dataset | Coverage | Unsupported Non-Abstained | Abstention | Extra Metric |
| --- | --- | ---: | ---: | ---: | --- |
| baseline 3B | ASQA | 49.5% | 3.5% | 47.0% | short-answer coverage 27.5% |
| gate+verifier 3B | ASQA | 54.5% | 0.0% | 45.5% | short-answer coverage 26.1% |
| repair+verifier 3B | ASQA | 53.0% | 0.0% | 47.0% | short-answer coverage 33.5% |
| baseline 3B | Finance | 47.0% | 1.0% | 52.0% | exact accuracy 62.0% |
| gate+verifier 3B | Finance | 47.0% | 0.0% | 53.0% | exact accuracy 62.0% |
| repair+verifier 3B | Finance | 65.0% | 0.0% | 35.0% | exact accuracy 80.0% |

Repair salvage:

- ASQA 3B: 24 repair attempts, 11 accepted repairs, 0 unsupported accepted repairs
- Finance 3B: 42 repair attempts, 41 accepted repairs, 0 unsupported accepted repairs

Generated distractor stress test:

- ASQA: 40 examples, 55.0% answer coverage, 0.0% unsupported non-abstained
- Finance: 20 examples, 85.0% exact accuracy, 0.0% unsupported non-abstained

Interpretation: repair helps recover supported answers, but the deterministic verifier remains the final acceptance boundary.

## Quick Start

For a fresh clone, use the full guide:

- `docs/RUNNING.md`

Short version:

```powershell
git clone https://github.com/mehmetcanozen/llmtermproject.git
cd llmtermproject

conda create -n llm-citation python=3.11 -y
conda activate llm-citation
python -m pip install -r requirements.txt

$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests -q --basetemp outputs\test\pytest_tmp
python scripts\run_eval_suite.py `
  --verifier-artifact outputs\evaluation\verifier_verdicts.json `
  --output-dir outputs\test\readiness_eval
```

Expected evaluation summary includes:

- `formal_full_eval_pass: true`
- `repair_plus_full_eval_pass: true`
- `systems_present` includes `repair_plus_verifier`

## Full Local Regeneration

Full model regeneration requires local assets that are too large for GitHub:

- Qwen2.5-3B-Instruct
- Qwen2.5-7B-Instruct, optional for the 7B comparison
- BAAI/bge-small-en-v1.5
- ASQA train/dev parquet files

Copy and edit the local config template:

```powershell
Copy-Item configs\local.example.yaml configs\local.yaml
notepad configs\local.yaml
```

Then run:

```powershell
python scripts\prepare_asqa.py --config configs\local.yaml
python scripts\build_retrieval_index.py --config configs\local.yaml --device cpu --batch-size 64
python scripts\run_locked_generation.py --config configs\local.yaml --system repair_plus_verifier --dataset finance --split finance_full_100 --model-size 3b --limit 3 --run-tag smoke3 --no-resume
python scripts\run_locked_generation.py --config configs\local.yaml --system repair_plus_verifier --dataset asqa --split dev_eval_200 --model-size 3b --limit 3 --run-tag smoke3 --no-resume
```

See `docs/RUNNING.md` for the full fixed-split workflow.

## Dependencies

- Python 3.11
- `requirements.txt` for artifact-level verification, tests, and saved metrics
- `requirements-full.txt` for local retrieval/generation with Qwen and BGE
- `artifacts/preflight/requirements-lock.txt` for the original local environment snapshot
- No paid/cloud API is required

Core libraries include `rank-bm25`, `numpy`, `pandas`, `scipy`, `jsonschema`, and `pytest`. Full local generation additionally uses `torch`, `transformers`, `sentence-transformers`, `accelerate`, and `bitsandbytes`.

## Sample Inputs and Outputs

Sample inputs:

- `data/asqa/splits/dev_eval_200.jsonl`
- `data/finance/generated/questions.jsonl`

Sample outputs:

- `outputs/final/examples/`
- `outputs/final/tables/system_comparison.csv`
- `outputs/final/tables/repair_salvage.csv`
- `outputs/final/tables/generated_distractor_metrics.csv`
- `outputs/final/figures/`

Submission materials:

- `docs/submission/SUBMISSION_GUIDE.md`
- `docs/submission/final_report_ieee/final_report.pdf`
- `docs/submission/final_report_ieee/final_report.tex`
- `docs/submission/final_report_ieee/final_report_print.html`

## Limitations

- ASQA is evaluated as a bounded local-corpus task, not open-web retrieval.
- The ASQA verifier is a deterministic proxy, not a human factuality judge.
- The finance dataset is synthetic and should be described as a controlled stress test.
- The generated distractor run is additional robustness evidence; it does not replace the fixed split.
- Full regeneration requires local model/data paths configured by the user.
