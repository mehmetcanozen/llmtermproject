# Phase 07 - Calibration And Evaluation

## Objective

Choose a practical gate configuration and compute the core evaluation metrics with confidence intervals, using a calibration strategy that is realistic on local hardware.

## Inputs

- Successful Phase 06 handoff
- `train_calibration_100`
- `dev_eval_200`
- Finance question set
- Working baseline, gate, and verifier

## In Scope

- Gate calibration
- System comparison metrics
- Confidence intervals
- Result tables for the chosen configuration

## Out Of Scope

- Final figures and report packaging
- Huge brute-force sweeps
- Full-dev benchmark runs unless explicitly added later

## Tasks

1. Calibrate on `train_calibration_100`, not on the dev evaluation split.
2. Use a staged search to keep runtime manageable.

Stage A:

- Model: Qwen 3B
- Fix layers=`4`, consecutive_failures=`3`
- Sweep thresholds: `0.05`, `0.10`, `0.15`, `0.20`, `0.25`
- Use the first `60` calibration examples
- Keep the best `2` thresholds by lowest unsupported non-abstained rate

Stage B:

- Model: Qwen 3B
- Use the full `train_calibration_100`
- Evaluate the top `2` thresholds from Stage A
- Sweep layers in `{2, 4}`
- Sweep consecutive failures in `{2, 3}`

3. Choose the winning gate config by this rule:

- minimize unsupported non-abstained rate
- subject to ASQA answer coverage staying within `10` percentage points of baseline
- subject to abstention rate staying at or below `35%`

4. Transfer the chosen gate settings to Qwen 7B.
5. If 7B calibration is unstable or too slow, use the chosen 3B settings directly for 7B final evaluation and document that limitation.
6. Evaluate the three systems on:

- `dev_eval_200`
- full finance question set

7. Required metrics:

ASQA:

- `answer_coverage`
- `citation_format_rate`
- `support_proxy_sentence_rate`
- `unsupported_non_abstained_rate`
- `abstention_rate`
- `retrieval_hit_rate`

Finance:

- `exact_answer_accuracy`
- `correct_citation_rate`
- `false_attribution_rate`
- `abstention_rate_unanswerable`
- `retrieval_hit_rate`

8. Compute paired bootstrap `95%` confidence intervals for the primary deltas between systems.
9. Save the chosen config and the metric tables.

## Deliverables

- `scripts/calibrate_gate.py`
- `scripts/run_eval_suite.py`
- `outputs/calibration/calibration_results.csv`
- `outputs/calibration/chosen_gate_config.yaml`
- `outputs/calibration/metric_tables.csv`
- `outputs/calibration/confidence_intervals.json`

## Validation

- Confirm calibration uses only `train_calibration_100`
- Confirm evaluation uses `dev_eval_200` and the finance question set
- Confirm the chosen config file matches the best-performing row in the calibration results
- Confirm confidence intervals were actually computed and saved

## Pass Criteria

This phase passes only if all of the following are true:

- A single chosen gate config is saved
- Metric tables exist for all three systems
- Confidence intervals are saved
- The chosen config reduces unsupported non-abstained outputs relative to baseline
- The chosen config keeps ASQA answer coverage within `10` points of baseline

## Stop-and-Ask Conditions

- Every candidate config collapses into high abstention
- No candidate config beats baseline on unsupported non-abstained rate
- 7B runtime is so unstable that even the transferred config cannot complete the eval set

## Required Progress Entry

Create a progress file named like `YYYY-MM-DD_HHMM_phase-07_iter-01.md`.

The entry must include:

- The calibration sweep actually run
- The chosen gate config and why it won
- The main metric deltas
- The confidence interval path
- Any 3B-to-7B transfer limitation

## Next Phase Handoff

The next phase must receive:

- The chosen gate config
- The metric tables
- The confidence intervals
- The exact eval split IDs
- The main error modes discovered during calibration
