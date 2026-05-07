# Phase 02 - Data Pipelines And Dataset Contracts

## Objective

Create the bounded local data layer for the project: normalized ASQA data, a bounded ASQA passage corpus, a reproducible synthetic finance dataset, and manifests that freeze the evaluation splits.

## Inputs

- Successful Phase 01 handoff
- Cached ASQA dataset
- The implementation guide and local docs

## In Scope

- Inspect the actual ASQA schema
- Normalize ASQA into project-local JSONL files
- Build a bounded ASQA corpus from ASQA-derived knowledge passages
- Freeze fixed calibration and evaluation subsets
- Generate a deterministic synthetic finance corpus and question set
- Save dataset manifests and schemas

## Out Of Scope

- Full-web or Wikipedia-scale retrieval
- External financial data ingestion
- Human annotation workflows
- Final scoring

## Tasks

1. Normalize ASQA into these local artifacts:

- `data/asqa/normalized/train_full.jsonl`
- `data/asqa/normalized/dev_full.jsonl`

2. Build a bounded ASQA passage corpus from ASQA-derived knowledge passages and preserve provenance per passage:

- `passage_id`
- `source_example_id`
- `source_split`
- `title` when available
- `text`

3. Deduplicate ASQA passages by normalized title plus normalized text.
4. Freeze a fixed calibration split:

- `data/asqa/splits/train_calibration_100.jsonl`
- Sampling rule: fixed random seed `20260416`
- Size: exactly `100`

5. Freeze a fixed evaluation split:

- `data/asqa/splits/dev_eval_200.jsonl`
- Sampling rule: fixed random seed `20260416`
- Stratify by `len(qa_pairs)` buckets: `2-3`, `4-5`, `6+`
- Target counts: `100`, `70`, `30`

6. Generate the synthetic finance corpus with fixed seed `20260416`.
7. Use only fictional issuers and fictional values.
8. Build `100` finance questions with this minimum composition:

- `40` exact numeric answerable questions
- `20` wrong-period or conflicting-period traps
- `15` near-duplicate issuer-name traps
- `15` unanswerable questions
- `10` retrieval-collision distractor cases

9. Each finance question must include:

- `example_id`
- `question`
- `gold_answer`
- `answerable`
- `expected_passage_ids`
- `metric_type`
- `company_name`
- `period`

10. Save manifests that document counts, seeds, source files, and generation settings.

## Deliverables

- `scripts/prepare_asqa.py`
- `scripts/build_finance_dataset.py`
- `data/asqa/corpus/passages.jsonl`
- `data/asqa/manifests/dataset_manifest.json`
- `data/finance/corpus/passages.jsonl`
- `data/finance/generated/questions.jsonl`
- `data/finance/manifests/dataset_manifest.json`
- `configs/schemas/asqa_record.schema.json`
- `configs/schemas/finance_record.schema.json`

## Validation

- Validate the normalized files against their schemas
- Re-run the finance generator and confirm identical example IDs and identical gold answers
- Confirm the split files have the exact intended counts
- Confirm the ASQA corpus contains provenance fields for every passage

## Pass Criteria

This phase passes only if all of the following are true:

- ASQA normalized files exist
- The bounded ASQA corpus exists and is schema-valid
- `train_calibration_100` and `dev_eval_200` exist with the exact intended counts
- The finance dataset regenerates identically with the same seed
- The manifests document counts, seeds, and source provenance

## Stop-and-Ask Conditions

- The cached ASQA schema differs materially from what the pipeline expects
- The bounded ASQA corpus ends up too sparse to support retrieval
- Any finance generator path introduces real company names or real filings
- The evaluation split cannot satisfy the requested stratified counts

## Required Progress Entry

Create a progress file named like `YYYY-MM-DD_HHMM_phase-02_iter-01.md`.

The entry must include:

- The normalized ASQA schema actually observed
- The exact split sizes created
- The deduplication rule used for ASQA passages
- The finance dataset composition counts
- The deterministic seed and reproduction result

## Next Phase Handoff

The next phase must receive:

- The bounded ASQA passage corpus
- The finance corpus and question file
- The fixed calibration and evaluation splits
- The dataset manifests
- The schema paths for both domains
