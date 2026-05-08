# Repair Plus Verifier Overhaul Runbook

This runbook captures the local commands for the high-impact overhaul path. It assumes the Windows conda environment and local model/data paths already configured in `configs/default.yaml`.

## 1. Rebuild Data And Retrieval

```powershell
cd C:\SoftwareProjects\LLMTermProject
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$py='C:\Users\omehm\anaconda3\envs\llm-citation\python.exe'

& $py scripts\prepare_asqa.py
& $py scripts\build_retrieval_index.py --device cpu --batch-size 64
```

The retrieval gate in `src/locked_runs.py` now fails generation if dense embedding files are missing, zero bytes, or do not match chunk counts.

## 2. Repair Plus Verifier Smoke

```powershell
& $py scripts\run_locked_generation.py --system repair_plus_verifier --dataset finance --split finance_full_100 --model-size 3b --limit 3 --run-tag smoke3 --no-resume
& $py scripts\run_locked_generation.py --system repair_plus_verifier --dataset asqa --split dev_eval_200 --model-size 3b --limit 3 --run-tag smoke3 --no-resume

& $py scripts\run_verifier.py `
  --source outputs\runs\locked\repair_plus_verifier_finance_finance_full_100_3b_smoke3\predictions.jsonl `
  --source outputs\runs\locked\repair_plus_verifier_asqa_dev_eval_200_3b_smoke3\predictions.jsonl `
  --output outputs\test\repair_plus_verifier_smoke_verdicts.json
```

Expected smoke condition: verifier summary has `false_attribution: 0`.

## 3. Full Fixed-Split Comparison

Run these when the machine can work for a longer period. The baseline and gate-only 3B runs already exist in the current package, but these commands make the comparison reproducible after a retrieval rebuild.

```powershell
& $py scripts\run_locked_generation.py --system baseline --dataset asqa --split dev_eval_200 --model-size 3b --run-tag overhaul --no-resume
& $py scripts\run_locked_generation.py --system baseline --dataset finance --split finance_full_100 --model-size 3b --run-tag overhaul --no-resume

& $py scripts\run_locked_generation.py --system gate_only --dataset asqa --split dev_eval_200 --model-size 3b --run-tag overhaul --gate-config outputs\calibration\chosen_gate_config.yaml --collect-traces --store-layer-scores --no-resume
& $py scripts\run_locked_generation.py --system gate_only --dataset finance --split finance_full_100 --model-size 3b --run-tag overhaul --gate-config outputs\calibration\chosen_gate_config.yaml --collect-traces --store-layer-scores --no-resume

& $py scripts\run_locked_generation.py --system repair_plus_verifier --dataset asqa --split dev_eval_200 --model-size 3b --run-tag overhaul --no-resume
& $py scripts\run_locked_generation.py --system repair_plus_verifier --dataset finance --split finance_full_100 --model-size 3b --run-tag overhaul --no-resume

& $py scripts\run_locked_generation.py --system repair_plus_verifier --dataset asqa --split dev_eval_200 --model-size 7b --run-tag overhaul --no-resume
& $py scripts\run_locked_generation.py --system repair_plus_verifier --dataset finance --split finance_full_100 --model-size 7b --run-tag overhaul --no-resume
```

## 4. Generated Distractor Stress Test

```powershell
& $py scripts\run_locked_generation.py --system repair_plus_verifier --dataset asqa --split dev_eval_200 --model-size 3b --limit 40 --run-tag generated_distractor --prompt-passage-count 4 --distractor --no-resume
& $py scripts\run_locked_generation.py --system repair_plus_verifier --dataset finance --split finance_full_100 --model-size 3b --limit 20 --run-tag generated_distractor --prompt-passage-count 4 --distractor --no-resume
```

## 5. Rebuild Verification, Metrics, And Final Assets

```powershell
& $py scripts\run_verifier.py
& $py scripts\run_eval_suite.py
& $py scripts\build_final_assets.py
& $py scripts\build_markdown_report.py
& $py -m pytest tests -q --basetemp outputs\test\pytest_tmp
```

Acceptance target: `pytest` passes, `outputs\evaluation\evaluation_manifest.json` has `formal_full_eval_pass: true`, and the best repair-plus-verifier row has unsupported non-abstained rate no worse than gate-plus-verifier while improving at least one coverage or exact-answer metric.
