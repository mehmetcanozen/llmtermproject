# Phase 06 - Deterministic Verifier

## Objective

Implement the deterministic verifier that checks citation structure for both domains and performs strict structured checks for the synthetic finance set.

## Inputs

- Successful Phase 05 handoff
- Baseline and gated outputs
- Finance dataset with expected passage IDs and exact answers

## In Scope

- Sentence splitting
- Citation marker parsing
- Passage mapping
- Finance structured checks
- ASQA citation-structure and anchor checks
- Unit tests

## Out Of Scope

- Large learned verifier models
- Paid API verification
- Final confidence intervals

## Tasks

1. Split answers into sentences while preserving bracket citations.
2. Extract all citation markers of the form `[P<number>]`.
3. Reject any factual sentence that has no citation marker.
4. Reject any citation marker that points outside the retrieved passage set.
5. Finance verifier checks must be exact for:

- company name
- quarter or period
- metric type
- numeric value
- cited passage ID

6. ASQA verifier checks must be explicit but modest:

- every factual sentence has at least one citation
- every cited passage exists in the retrieved set
- every explicit number, year, or quoted span in the sentence appears in at least one cited passage

7. Record verifier outputs per sentence and per example.
8. Add unit tests for:

- malformed citations
- missing citations
- wrong passage citations
- finance structured mismatches
- abstention handling
- multiple citations on one sentence

## Deliverables

- `src/verifier.py`
- `tests/test_verifier.py`
- `scripts/run_verifier_smoke.py`
- `artifacts/verifier/verifier_examples.json`

## Validation

- Run the verifier unit test file
- Run a small verifier smoke script on both ASQA and finance outputs
- Confirm the verifier emits machine-readable verdicts

## Pass Criteria

This phase passes only if all of the following are true:

- Unit tests pass for malformed citations
- Unit tests pass for missing citations
- Unit tests pass for wrong passage citations
- Unit tests pass for finance structured mismatches
- Abstention outputs are handled cleanly

## Stop-and-Ask Conditions

- Sentence splitting corrupts citation markers
- The verifier contract conflicts with the run output schema
- A required finance field is missing from the dataset contract

## Required Progress Entry

Create a progress file named like `YYYY-MM-DD_HHMM_phase-06_iter-01.md`.

The entry must include:

- The sentence-splitting rule
- The citation parsing rule
- The exact finance checks performed
- The ASQA support-proxy checks performed
- The unit test command and outcome

## Next Phase Handoff

The next phase must receive:

- The verifier output schema
- The test results
- The finance exact-check contract
- The ASQA support-proxy contract
- Any known false-positive or false-negative pattern
