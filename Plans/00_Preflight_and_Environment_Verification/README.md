# Phase 00 - Preflight And Environment Verification

## Objective

Create and validate the Windows + Conda environment that the rest of the project will use. This phase exists to prove the machine, driver, CUDA stack, and local-model path are workable before any project code is written.

## Inputs

- [`../README.md`](../README.md)
- [`../SETUP_AND_DOWNLOADS.md`](../SETUP_AND_DOWNLOADS.md)
- Access to the local machine, installed tools, and pre-downloaded models

## In Scope

- Create or activate the `llm-citation` Conda environment
- Install Torch using the Windows CUDA command documented in `SETUP_AND_DOWNLOADS.md`
- Install the direct Python dependencies
- Verify CUDA visibility
- Verify bitsandbytes import
- Verify the expected compute capability
- Run one real deterministic Qwen 3B smoke generation
- Record exact versions, local asset paths, and install commands

## Out Of Scope

- Repo scaffolding beyond minimal artifact storage
- Retrieval logic
- Dataset normalization
- Evaluation code
- Report writing

## Tasks

1. Confirm `nvidia-smi` works and shows the expected GPU.
2. Create or activate `llm-citation`.
3. Install Torch using the Windows CUDA command in `SETUP_AND_DOWNLOADS.md`.
4. Install the project package set from `SETUP_AND_DOWNLOADS.md`.
5. Run `python scripts\preflight_smoke.py --json-out artifacts\preflight\smoke_test_report.json`.
6. If the default 4-bit smoke fails, run `python scripts\preflight_smoke.py --no-4bit --json-out artifacts\preflight\smoke_test_report.json` once as a fallback diagnostic.
7. Run `python scripts\preflight_risk_checks.py --json-out artifacts\preflight\risk_check_report.json`.
8. Record exact versions for Python, Torch, transformers, accelerate, bitsandbytes, sentence-transformers, datasets, and chromadb.
9. Save the preflight report artifacts.

## Deliverables

- `artifacts/preflight/environment_report.md`
- `artifacts/preflight/environment_report.json`
- `artifacts/preflight/risk_check_report.json`
- `artifacts/preflight/qwen3b_smoke.txt`
- `artifacts/preflight/requirements-lock.txt`

If `artifacts/preflight/` does not exist yet, create only that minimal path for this phase.

## Validation

- Run `nvidia-smi`
- Run `python scripts\preflight_smoke.py --json-out artifacts\preflight\smoke_test_report.json`
- Run `python scripts\preflight_risk_checks.py --json-out artifacts\preflight\risk_check_report.json`
- Confirm the environment report lists the exact package versions actually installed

## Pass Criteria

This phase passes only if all of the following are true:

- `torch.cuda.is_available()` is `True`
- `bitsandbytes` imports successfully
- The visible device is the RTX 5070
- The reported compute capability is `12.0`
- Qwen 3B loads and generates deterministic text without crashing
- Eager `output_attentions=True` returns usable attention tensors
- The environment report and lock file are saved

If any one of those checks fails, the phase does not pass.

## Stop-and-Ask Conditions

- The live PyTorch selector does not offer a Windows CUDA build that matches the local driver
- The GPU reports a compute capability other than `12.0`
- Qwen 3B works only without CUDA
- The environment can run non-quantized inference but not 4-bit inference after one documented retry
- A required local model folder is missing or corrupted

## Required Progress Entry

Create a progress file named like `YYYY-MM-DD_HHMM_phase-00_iter-01.md` in `Progress/`.

The entry must include:

- The exact Torch install command used
- The exact versions installed
- The smoke and risk-check script commands run
- Whether deterministic generation matched across two attempts
- Whether eager `output_attentions=True` returned usable tensors
- Any blocker or fallback decision

## Next Phase Handoff

The next phase must receive:

- The exact environment name
- The exact package versions
- The exact local model and data locations
- Any known Windows-specific caveats
- The saved preflight artifacts
