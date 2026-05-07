# Phase 04 - Baseline RAG Generation

## Objective

Implement the deterministic baseline generation path that uses the retriever outputs, formats citations consistently, and writes schema-valid run outputs.

## Inputs

- Successful Phase 03 handoff
- Retrieval outputs for both corpora
- Valid config and schemas from Phase 01

## In Scope

- Prompt construction
- Deterministic baseline generation
- Citation-format validation
- Structured run output saving
- Reproducibility checks

## Out Of Scope

- Attention gating
- Verifier logic beyond format checks
- Hyperparameter calibration
- Final figures

## Tasks

1. Use this fixed answer contract:

```text
You are a question-answering system that must use only the provided passages.

Rules:
- Answer only with information supported by the passages.
- Every factual sentence must end with one or more citations like [P1] or [P2].
- If the passages do not contain enough evidence, output exactly: INSUFFICIENT_SUPPORT
- Do not use outside knowledge.
```

2. Build prompts with exactly `3` retrieved passages labeled `[P1]`, `[P2]`, `[P3]`.
3. Use deterministic decoding:

- `temperature=0`
- `do_sample=False`
- `max_new_tokens=160`

4. Save one structured output record per example.
5. Add a format validator that checks:

- The answer is either `INSUFFICIENT_SUPPORT` or citation-bearing text
- Every non-empty factual sentence ends in at least one passage citation marker
- All cited passage IDs are in the retrieved set

6. Run a fixed 20-example smoke run:

- `10` ASQA examples from `train_calibration_100`
- `10` finance examples from `questions.jsonl`

7. Re-run the same smoke set with the same seed and compare outputs byte-for-byte.

## Deliverables

- `src/prompting.py`
- `src/generation.py`
- `scripts/run_baseline_smoke.py`
- `outputs/runs/baseline_smoke/run_manifest.json`
- `outputs/runs/baseline_smoke/predictions.jsonl`
- `outputs/runs/baseline_smoke/format_report.md`

## Validation

- Validate `predictions.jsonl` against the run output schema
- Confirm the smoke run covers exactly `20` examples
- Compare the two deterministic runs and confirm identical outputs

## Pass Criteria

This phase passes only if all of the following are true:

- The smoke run outputs are schema-valid
- The smoke run covers exactly `20` examples
- The two repeated runs match exactly
- Every non-abstained factual sentence passes the citation-format validator

If the formatting fails, refine the prompt or output handling before moving on.

## Stop-and-Ask Conditions

- The baseline model repeatedly ignores the citation contract after one prompt refinement iteration
- The model produces unstable outputs with the same seed
- The run output schema cannot represent actual generation results cleanly

## Required Progress Entry

Create a progress file named like `YYYY-MM-DD_HHMM_phase-04_iter-01.md`.

The entry must include:

- The exact prompt template used
- The generation settings
- The smoke set IDs
- Whether the repeated runs matched
- The format failure rate, if any

## Next Phase Handoff

The next phase must receive:

- The final prompt contract
- The baseline smoke outputs
- The reproducibility result
- The format validator behavior
- Any recurring malformed-output pattern
