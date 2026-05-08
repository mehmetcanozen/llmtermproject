# Running the Project From a Fresh Clone

This repository supports two levels of reproduction.

1. **Artifact-level verification**: no local LLM weights are needed. This checks tests, saved metrics, final tables, figures, and report assets.
2. **Full local regeneration**: requires local Qwen, BGE, and ASQA parquet files. This can rerun retrieval and generation.

The repository intentionally does not include model weights, dense embedding files, full ASQA corpora, or full generation run folders.

## 1. Clone and Install

```powershell
git clone https://github.com/mehmetcanozen/llmtermproject.git
cd llmtermproject

conda create -n llm-citation python=3.11 -y
conda activate llm-citation
python -m pip install -r requirements.txt
```

If you want full local LLM generation, install the PyTorch build that matches your CUDA/driver, then install the extra model dependencies:

```powershell
python -m pip install -r requirements-full.txt
```

## 2. Artifact-Level Verification

These commands should work from a fresh clone without downloading local LLM weights.

```powershell
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'

python -m pytest tests -q --basetemp outputs\test\pytest_tmp
python scripts\run_eval_suite.py `
  --verifier-artifact outputs\evaluation\verifier_verdicts.json `
  --output-dir outputs\test\readiness_eval
```

Expected high-level result from `run_eval_suite.py`:

- `formal_full_eval_pass: true`
- `repair_plus_full_eval_pass: true`
- `systems_present` includes `repair_plus_verifier`

The final lightweight result package is already included under:

- `outputs/final/tables/`
- `outputs/final/figures/`
- `outputs/final/examples/`
- `outputs/final/FINAL_REPORT.md`

## 3. Submission Report Assets

The submission guide and IEEE-style report live in:

- `docs/submission/SUBMISSION_GUIDE.md`
- `docs/submission/final_report_ieee/final_report.tex`
- `docs/submission/final_report_ieee/final_report_print.html`
- `docs/submission/final_report_ieee/final_report.pdf`

To rebuild the clean report figures on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\docs\submission\final_report_ieee\build_report_figures.ps1
```

## 4. Full Local Regeneration

Full generation requires local files that are not stored in Git:

- Qwen2.5 3B Instruct model directory
- Qwen2.5 7B Instruct model directory, optional but used for the final comparison
- BAAI BGE small English embedding model directory
- ASQA train/dev parquet files

Start by copying the local config template:

```powershell
Copy-Item configs\local.example.yaml configs\local.yaml
notepad configs\local.yaml
```

Edit the paths in `configs/local.yaml`, then run:

```powershell
python scripts\prepare_asqa.py --config configs\local.yaml
python scripts\build_retrieval_index.py --config configs\local.yaml --device cpu --batch-size 64
```

Smoke-test the repair system:

```powershell
python scripts\run_locked_generation.py --config configs\local.yaml --system repair_plus_verifier --dataset finance --split finance_full_100 --model-size 3b --limit 3 --run-tag smoke3 --no-resume
python scripts\run_locked_generation.py --config configs\local.yaml --system repair_plus_verifier --dataset asqa --split dev_eval_200 --model-size 3b --limit 3 --run-tag smoke3 --no-resume
```

After generation, verify and evaluate the saved predictions:

```powershell
python scripts\run_verifier.py --output outputs\evaluation\verifier_verdicts.json
python scripts\run_eval_suite.py
```

## 5. What Is Included vs. Generated

Included in Git:

- Source code in `src/`
- Entrypoint scripts in `scripts/`
- Tests in `tests/`
- Small fixed splits and finance data in `data/`
- Final tables, figures, examples, and report assets in `outputs/final/`
- Saved verifier/evaluation artifacts needed for artifact-level verification
- Small retrieval candidate files needed by final asset tooling

Generated locally and not included in Git:

- Dense embedding `.npz` files
- Full retrieval chunk files
- Full model prediction run folders
- Downloaded model weights
- Full ASQA corpus/parquet files
