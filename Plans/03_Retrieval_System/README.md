# Phase 03 - Retrieval System

## Objective

Build a hybrid retriever over the bounded local corpora and prove that it retrieves supportable passages before any attention gating is attempted.

## Inputs

- Successful Phase 02 handoff
- Bounded ASQA corpus
- Synthetic finance corpus and questions
- Verified environment with embedding model cached

## In Scope

- Passage chunking
- Dense retrieval with `BAAI/bge-small-en-v1.5`
- BM25 retrieval over the same chunk set
- Simple score merging
- Retrieval inspection tooling and manual review

## Out Of Scope

- Generation
- Attention gating
- Final evaluation
- External search or large remote corpora

## Tasks

1. Chunk corpus passages to `180-220` tokens with `40` token overlap.
2. Create chunk IDs that preserve parent passage provenance.
3. Dense retrieval defaults:

- Model: `BAAI/bge-small-en-v1.5`
- Top-k: `5`
- CPU first

4. BM25 retrieval defaults:

- Top-k: `5`

5. Merge dense and BM25 results by chunk ID.
6. Normalize scores into `[0, 1]` per query.
7. Use this final score:

- `0.60 * dense_norm + 0.40 * bm25_norm`

8. Keep top `3` chunks for prompting.
9. Save both raw candidate lists and final merged lists.
10. Run a manual inspection set of exactly `30` questions:

- `10` from `data/asqa/splits/dev_eval_200.jsonl`
- `20` from `data/finance/generated/questions.jsonl`

11. Build a human-readable inspection sheet that records whether at least one clearly relevant support chunk appears in the top `3`.

## Deliverables

- `src/retrieval.py`
- `scripts/build_retrieval_index.py`
- `scripts/run_retrieval_smoke.py`
- `outputs/retrieval/asqa_candidates.jsonl`
- `outputs/retrieval/finance_candidates.jsonl`
- `outputs/retrieval/manual_inspection_phase03.md`

## Validation

- Confirm chunk records preserve parent passage IDs
- Confirm the retriever returns exactly `3` prompt-ready chunks per query
- Confirm the inspection sheet exists and names the inspected example IDs

## Pass Criteria

This phase passes only if all of the following are true:

- Retrieval runs successfully on both corpora
- The manual inspection sheet covers exactly `30` questions
- A clearly relevant support chunk appears in the top `3` for at least `21` of the `30` inspected questions
- At least `6` of the `10` ASQA inspections pass
- At least `15` of the `20` finance inspections pass

If the pass rate is below those thresholds, improve retrieval before moving on.

## Stop-and-Ask Conditions

- The embedding model cannot load or encode reliably
- Chunking destroys enough context that support becomes obviously fragmented
- ASQA retrieval stays below `6/10` after one retrieval-tuning iteration
- Any proposed fix requires leaving the bounded local corpus design

## Required Progress Entry

Create a progress file named like `YYYY-MM-DD_HHMM_phase-03_iter-01.md`.

The entry must include:

- Chunking settings
- Dense and BM25 top-k values
- The merge formula
- The inspection results
- The strongest retrieval failure pattern observed

## Next Phase Handoff

The next phase must receive:

- The chunked corpora
- The merged retrieval outputs
- The exact retrieval defaults
- The manual inspection sheet
- Any known retrieval failure categories
