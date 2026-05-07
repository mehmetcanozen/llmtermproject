# Phase 05 - Attention Gate

## Objective

Implement the inference-time grounding gate that monitors passage-directed self-attention during generation and abstains when support stays too weak.

## Inputs

- Successful Phase 04 handoff
- Working baseline generation path
- Stable eager-attention model loading path from Phase 00

## In Scope

- Prompt span mapping
- Autoregressive decode loop with attention capture
- Per-token support scoring
- Rolling gate logic
- Attention trace saving

## Out Of Scope

- Full calibration sweep
- Final verifier
- Final result package

## Tasks

1. Map each prompt passage to exact token spans before generation starts.
2. Replace the one-shot generation call with a token-by-token decode loop.
3. Request attentions on each decode step and use `use_cache=True`.
4. Score support per token by:

- taking the last `4` layers
- averaging across heads
- summing attention mass landing on each passage span
- using the maximum passage score as the token support score

5. Ignore or downweight these tokens for gate decisions:

- punctuation-only tokens
- bracket and citation marker tokens
- the first `8` generated tokens

6. Use this default gate:

- rolling window: `4`
- threshold: `0.10`
- consecutive failures: `3`

7. On gate failure, replace the answer with exactly `INSUFFICIENT_SUPPORT`.
8. Save a token-level attention trace per example for debugging.
9. Run a targeted 12-example gate smoke set:

- `6` known weak-support examples
- `6` clearly supported examples

10. Use Qwen 3B first. Move the same logic to 7B only after the 3B path works.

## Deliverables

- `src/attention_gate.py`
- `scripts/run_gate_smoke.py`
- `outputs/runs/gate_smoke/predictions.jsonl`
- `outputs/runs/gate_smoke/attention_traces.jsonl`
- `outputs/runs/gate_smoke/gate_report.md`

## Validation

- Confirm every smoke example saves an attention trace
- Confirm each trace records token text, token index, passage scores, and gate decision
- Confirm supported and weak-support examples are clearly labeled in the smoke report

## Pass Criteria

This phase passes only if all of the following are true:

- Attention traces are saved for all `12` smoke examples
- At least `4` of the `6` weak-support examples abstain
- At least `4` of the `6` clearly supported examples complete without abstaining
- The gating logic runs on Qwen 3B without crashing

Moving the same logic to 7B is part of the phase, but 3B must pass first.

## Stop-and-Ask Conditions

- The chosen backend does not return usable attentions
- The eager path causes repeated memory failure
- The gate abstains on nearly every example
- The support trace appears disconnected from passage spans

## Required Progress Entry

Create a progress file named like `YYYY-MM-DD_HHMM_phase-05_iter-01.md`.

The entry must include:

- The token-span mapping rule
- The gate hyperparameters used
- The smoke set composition
- The weak-support and supported-case outcomes
- Any backend-specific caveat

## Next Phase Handoff

The next phase must receive:

- The final gate implementation path
- The attention trace format
- The default hyperparameters
- The smoke report
- Any known over-abstention pattern
