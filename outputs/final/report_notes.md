# Final Report Notes

## Package Status

This final package is report-ready as an artifact bundle.
Formal full-evaluation pass for completed 3B fixed-split scope: `True`.
Full locked 3B baseline and gate-only predictions are present for dev_eval_200 and finance_full_100.

## Locked Gate Config

```yaml
phase: 07_Calibration_and_Evaluation
created_at: '2026-04-27T01:02:36.105865+00:00'
status: proxy_selected_constraints_not_fully_satisfied
calibration_mode: artifact_trace_replay_smoke
chosen_gate_config:
  threshold: 0.1
  tail_layers: 4
  rolling_window: 4
  consecutive_failures: 3
  skip_initial_generated_tokens: 8
why_selected:
  selection_rule: lowest unsupported non-abstained rate, then highest answer coverage,
    then lower abstention, tie-breaking toward Phase 05 runtime settings
  selection_rank: 1
  unsupported_non_abstained_rate: 0.02
  answer_coverage: 0.6
  abstention_rate: 0.38
  baseline_answer_coverage_overlap: 0.64
  baseline_unsupported_non_abstained_overlap: 0.01
  passes_plan_selection_constraints: false
  reduces_unsupported_vs_baseline_overlap: false
source_artifacts:
  verifier_artifact: outputs\evaluation\calibration_verifier_verdicts.json
  trace_file: outputs\runs\locked\gate_only_asqa_train_calibration_100_3b_calibration\attention_traces.jsonl
  calibration_results: outputs/calibration/calibration_results.csv
limitations:
- 'Trace dataset: asqa; trace count: 100.'
- Stage A uses the first 60 traces; Stage B uses all traces and sweeps the top 2 Stage
  A thresholds.
- Tail-layer sweeps are exact only when traces contain layer_passage_scores; otherwise
  legacy smoke traces remain proxy rows.
- The chosen config is acceptable for final evaluation only if this file was produced
  from train_calibration_100 traces.
```

## Run Manifests Used

- `baseline_smoke`: `outputs/runs/baseline_smoke/run_manifest.json`
- `gate_smoke_3b`: `outputs/runs/gate_smoke/run_manifest.json`
- `gate_smoke_7b_transfer`: `outputs/runs/gate_smoke_7b_check/run_manifest.json`
- `phase07_evaluation`: `outputs/calibration/evaluation_manifest.json`
- `chosen_gate_config`: `outputs/calibration/chosen_gate_config.yaml`

## Fixed Eval IDs

- ASQA planned fixed evaluation set: `data/asqa/splits/dev_eval_200.jsonl` (`200` IDs).
- Finance planned fixed evaluation set: `data/finance/generated/questions.jsonl` (`100` IDs).
- Current scored prediction IDs are enumerated in `outputs/evaluation/eval_split_ids.json` when full evaluation has been run.
- Distractor probe static subset: first `40` ASQA retrieval candidates from `outputs/retrieval/asqa_candidates.jsonl` and first `20` finance retrieval candidates from `outputs/retrieval/finance_candidates.jsonl`.

## Tables And Figures

- `outputs/final/tables/system_comparison.csv`
- `outputs/final/tables/asqa_metrics.csv`
- `outputs/final/tables/finance_metrics.csv`
- `outputs/final/tables/coverage_safety_summary.csv`
- `outputs/final/tables/repair_salvage.csv`
- `outputs/final/tables/distractor_probe.csv`
- `outputs/final/tables/distractor_probe_summary.csv`
- `outputs/final/tables/generated_distractor_metrics.csv` when generated distractor runs are available.
- `outputs/final/figures/unsupported_non_abstained.png`
- `outputs/final/figures/abstention_vs_coverage.png`
- `outputs/final/figures/safety_vs_coverage_frontier.png`
- `outputs/final/figures/repair_funnel.png`
- `outputs/final/figures/finance_citation_accuracy.png`
- `outputs/final/figures/distractor_sensitivity.png`
- `outputs/final/figures/generated_distractor_robustness.png` when generated distractor runs are available.

## Qualitative Examples

- Curated examples: `8`.
- Index: `outputs/final/examples/example_index.csv`.
- Categories: baseline failures, gate successes, verifier catches, repair outcomes when available, and distractor-probe cases.

## Required Limitations

- Bounded ASQA corpus design: ASQA uses a local bounded corpus derived from available ASQA passages. It is not an open-web retrieval benchmark.
- Support-proxy limitation: ASQA support checks only verify citation structure plus explicit numbers, years, and quoted spans against cited passages. This is useful but not proof of complete factual faithfulness.
- Finance synthetic scope: finance records are fictional stress-test disclosures and questions. They are not real financial data and must not be described as deployment evidence.
- 3B/7B limitation: the Qwen 7B transfer check produced format-valid but semantically wrong finance period strings such as malformed fiscal years; the deterministic verifier catches these as false attributions.
- Full-eval status: Full locked 3B baseline and gate-only predictions are present for dev_eval_200 and finance_full_100.
- Static distractor limitation: the older distractor probe is a prompt-input static proxy. It adds one plausible irrelevant passage to saved retrieval inputs and reports sensitivity signals, but it does not include new model generations.
- Generated distractor limitation: generated distractor runs are a robustness stress test with a fourth prompt passage, not a replacement for the normal fixed-split results.

## External Method References

- ALCE frames citation evaluation around correctness and citation quality: https://aclanthology.org/2023.emnlp-main.398/
- SciPy documents paired bootstrap confidence-interval mechanics used as the model for Phase 07 intervals: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- Matplotlib `savefig`/bar/scatter docs were checked for conventional figure outputs, but this package uses a standard-library PNG fallback because Pillow is absent in the local conda env: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html

## Snapshot Summary

- Systems represented in saved metrics: `baseline, gate_only, gate_plus_verifier, repair_plus_verifier`.
- Datasets represented in saved metrics: `asqa, finance`.
- Distractor probe rows: `240`.
- Generated distractor metric rows: `2`.
- Repair salvage rows: `4`.
