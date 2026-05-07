# Phase 08 - Experiments, Figures, And Report Assets

## Objective

Create the final experiment package: run the locked system comparison, execute a small distractor-passage probe, save report-ready tables and figures, and curate qualitative examples that explain the results honestly.

## Inputs

- Successful Phase 07 handoff
- Locked gate config
- Metric tables and confidence intervals
- Stable run scripts for all three systems

## In Scope

- Final locked comparison on the fixed evaluation sets
- Distractor-passage probe
- Tables, figures, and qualitative examples
- Final narrative notes for the report

## Out Of Scope

- New architecture changes
- New datasets
- New calibration sweeps
- Last-minute framework migrations

## Tasks

1. Run the locked three-system comparison on:

- `dev_eval_200`
- full finance question set

2. Do not retune anything during this phase.
3. Run a distractor-passage probe on a fixed subset:

- `40` ASQA examples from `dev_eval_200`
- `20` finance examples

4. For the probe, add exactly one plausible but irrelevant distractor chunk to the prompt and re-run the same locked systems.
5. Save the final tables:

- overall system comparison
- ASQA-only metrics
- finance-only metrics
- distractor-probe deltas

6. Save the final figures:

- unsupported non-abstained rate by system
- abstention vs answer coverage tradeoff
- finance citation accuracy by system
- distractor sensitivity summary

7. Curate at least `8` qualitative examples:

- `2` baseline failures
- `2` gate successes
- `2` verifier catches
- `2` distractor-probe cases

8. Write short report notes that disclose:

- bounded ASQA corpus design
- support-proxy limitations
- finance synthetic-dataset scope
- any 3B/7B limitation actually encountered

## Deliverables

- `outputs/final/tables/system_comparison.csv`
- `outputs/final/tables/distractor_probe.csv`
- `outputs/final/figures/unsupported_non_abstained.png`
- `outputs/final/figures/abstention_vs_coverage.png`
- `outputs/final/figures/finance_citation_accuracy.png`
- `outputs/final/figures/distractor_sensitivity.png`
- `outputs/final/examples/`
- `outputs/final/report_notes.md`

## Validation

- Confirm all three systems were run on the same fixed example IDs
- Confirm figures are generated from saved tables, not from ad hoc notebook state
- Confirm qualitative examples reference actual saved outputs
- Confirm `report_notes.md` includes the required limitations

## Pass Criteria

This phase passes only if all of the following are true:

- Final comparison tables exist
- Final figures exist
- The distractor-probe results are saved
- At least `8` qualitative examples are curated
- The report notes disclose the main limitations honestly

## Stop-and-Ask Conditions

- A locked system can no longer reproduce the saved Phase 07 metrics
- The final probe requires retuning to run
- A missing artifact from an earlier phase makes the final package impossible to reproduce

## Required Progress Entry

Create a progress file named like `YYYY-MM-DD_HHMM_phase-08_iter-01.md`.

The entry must include:

- The final run manifests used
- The fixed eval IDs used
- The saved table and figure paths
- The curated example count
- The final limitation notes

## Next Phase Handoff

This is the final build phase. The handoff package must contain:

- The final tables
- The final figures
- The curated examples
- The report notes
- The exact locked config and manifests needed to reproduce the package
