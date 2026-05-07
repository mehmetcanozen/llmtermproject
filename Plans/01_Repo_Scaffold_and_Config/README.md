# Phase 01 - Repo Scaffold And Config

## Objective

Create the project skeleton, configuration contract, and reproducibility utilities. This phase establishes the minimal structure that every later phase depends on.

## Inputs

- Successful Phase 00 handoff
- The preflight artifacts and verified environment

## In Scope

- Create the repo directory skeleton
- Define a single configuration format
- Define machine-readable schemas for run manifests and run outputs
- Add deterministic seed utilities
- Add a small validation entrypoint that proves configs and schemas load correctly

## Out Of Scope

- Real retrieval logic
- Real generation logic
- Dataset loading beyond placeholder contracts
- Calibration
- Final experiments

## Tasks

1. Create this project structure:

```text
artifacts/
  preflight/
configs/
  schemas/
data/
  asqa/
    raw/
    normalized/
    corpus/
    splits/
    manifests/
  finance/
    raw/
    generated/
    corpus/
    manifests/
outputs/
  retrieval/
  runs/
  calibration/
  final/
scripts/
src/
  __init__.py
  config.py
  data_loading.py
  retrieval.py
  prompting.py
  generation.py
  attention_gate.py
  verifier.py
  evaluation.py
  utils/
tests/
```

Treat `external/ALCE_reference` as reference-only space. Do not make the main runtime depend on code inside that ALCE copy.

2. Use `YAML` for human-edited configs.
3. Use `JSON Schema` for validating run manifests and run outputs.
4. Add deterministic utilities that set Python, NumPy, and Torch seeds together.
5. Define one base experiment config with sections for model, retrieval, prompt, gate, verifier, and evaluation.
6. Add a small validation script that loads the config and validates one sample manifest and one sample output object.

## Deliverables

- `configs/default.yaml`
- `configs/schemas/run_manifest.schema.json`
- `configs/schemas/run_output.schema.json`
- `src/config.py`
- `src/utils/determinism.py`
- `scripts/validate_config.py`
- `tests/test_config_contracts.py`

## Validation

- Load `configs/default.yaml` successfully
- Validate a sample manifest against `run_manifest.schema.json`
- Validate a sample output object against `run_output.schema.json`
- Run one test that confirms the seed helper sets all relevant seeds without crashing

## Pass Criteria

This phase passes only if all of the following are true:

- The scaffold exists
- `configs/default.yaml` loads cleanly
- Both schemas validate their sample objects
- The validation script exits successfully
- The determinism helper exists and is exercised by at least one test

## Stop-and-Ask Conditions

- The environment cannot import a required dependency from Phase 00
- The chosen config contract cannot represent all three system variants cleanly
- A later phase clearly needs a different artifact layout than the one defined here

## Required Progress Entry

Create a progress file named like `YYYY-MM-DD_HHMM_phase-01_iter-01.md`.

The entry must include:

- The exact scaffold created
- The chosen config format and why it was chosen
- The names of the schema files
- The validation commands run
- Any contract decisions that later phases must not break

## Next Phase Handoff

The next phase must receive:

- The final scaffold
- The config schema
- The run output schema
- The determinism helper contract
- The validation script path and expected command
