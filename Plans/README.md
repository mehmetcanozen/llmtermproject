# Deterministic Citation Enforcement Plan Pack

This folder is the operational contract for the project. A future AI agent should treat it as the working playbook for building, evaluating, and documenting the course project.

## Project Summary

Build a local, reproducible RAG system that answers only from retrieved passages, cites every factual sentence, and abstains when support is weak. The project compares three systems:

1. `baseline`
2. `gate_only`
3. `gate_plus_verifier`

The main question is:

> Can an inference-time grounding gate plus a deterministic post-generation verifier reduce unsupported cited answers without retraining the model?

## Honest Framing

The project must be described honestly in all code, docs, tables, and final writing.

- The ASQA track is a bounded local corpus experiment built from ASQA-derived knowledge passages, not a full open-web retrieval system.
- Passage attribution in decoder-only models must be described as passage-directed self-attention over prompt tokens.
- Internal support signals and verifier proxies are useful engineering signals, but they are not proof of faithfulness.
- The finance track is a synthetic, controlled stress test, not a real financial deployment benchmark.
- The final system is a course-project prototype, not a production system and not safe for legal, financial, or compliance use.

## Source Of Truth Order

When instructions conflict, use this order:

1. Direct operator or user instructions for the current task
2. The current phase README in this folder
3. This file
4. The implementation guide markdown file in `Docs/`
5. `Docs/20210808020LLMProposal.pdf`
6. Current official vendor or model documentation for time-sensitive install or runtime details

Do not silently inherit assumptions from older notes if a phase README says otherwise.

## Project Management Approach

Use phase-gated Kanban.

- Work flows in order from Phase 00 to Phase 08.
- Keep `WIP = 1` inside a phase. Finish one auditable objective before starting another.
- Do not pull the next phase until the current phase has passed its gate.
- If a phase fails, iterate inside the same phase until it passes or a real blocker is reached.
- Every iteration must end with a progress entry in `C:\SoftwareProjects\LLMTermProject\Progress`.
- Prefer small, evidence-backed iterations over large speculative rewrites.

This is the best fit here because the project has changing research details, hard technical dependencies, and a strong need for explicit stop conditions.

## Hardware And Runtime Defaults

- OS: Windows first
- Environment: Conda
- GPU: GeForce RTX 5070, 12 GB VRAM, expected compute capability `12.0`
- CPU: Core Ultra 7 265
- RAM: 64 GB
- Bring-up model: `Qwen/Qwen2.5-3B-Instruct`
- Main experiment model: `Qwen/Qwen2.5-7B-Instruct`
- Embedding model: `BAAI/bge-small-en-v1.5`
- Coding style: plain Python, minimal abstractions, local-first evaluation

If 7B never reaches stable deterministic inference on Windows after a documented retry path, continue with 3B and report that limitation explicitly rather than hiding it.

## Technical Defaults

- Python `3.11`
- Deterministic decoding: `temperature=0`, `do_sample=False`
- Prompt passage count: `3`
- Prompt length target: roughly `900-1200` input tokens
- Generation cap: `max_new_tokens <= 160`
- Fixed abstention string: `INSUFFICIENT_SUPPORT`
- Retrieval: hybrid dense plus BM25 over a bounded local corpus
- Evaluation: local metrics first, with ALCE used only as a reference point for metric design

## Best Practices For The Agent

- Read the current phase README and the previous phase handoff before changing anything.
- Keep the stack simple. Do not add framework layers unless a phase explicitly needs them.
- Record exact commands, package versions, model IDs, and local asset paths.
- Lock versions only after a successful smoke test.
- Start with 3B, then move to 7B only after the smaller path is stable.
- Keep the ASQA corpus bounded and disclose that design choice in all results.
- Keep the finance set fictional and reproducible.
- Save machine-readable artifacts, not only human-readable notes.
- Prefer one fixed schema per artifact type.
- Never overclaim what the support signal means.
- Never skip retrieval sanity checks.
- Never skip verifier unit tests.

## Progress Logging Rules

After every iteration, create a new markdown file under `C:\SoftwareProjects\LLMTermProject\Progress`.

- File name format: `YYYY-MM-DD_HHMM_phase-##_iter-##.md`
- Example: `2026-04-16_2130_phase-03_iter-02.md`
- Use the template at [`_templates/PROGRESS_ENTRY_TEMPLATE.md`](./_templates/PROGRESS_ENTRY_TEMPLATE.md)
- One file per iteration
- Do not overwrite earlier progress files
- Include real evidence: commands run, files changed, test outcomes, blockers, and the next planned action

If an iteration fails, still write the progress entry and mark the blocker clearly.

## External References To Respect

- NVIDIA CUDA GPU table: <https://developer.nvidia.com/cuda/gpus>
- bitsandbytes installation docs: <https://huggingface.co/docs/bitsandbytes/installation>
- Qwen2.5 model card: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- PyTorch install page: <https://pytorch.org/get-started/locally/>
- PyTorch blog: <https://pytorch.org/blog/>
- ALCE repo: <https://github.com/princeton-nlp/ALCE>
- Atlassian Kanban guidance: <https://www.atlassian.com/agile/project-management/kanban-principles>
- Anthropic agent guidance: <https://www.anthropic.com/engineering/building-effective-agents>

## Phase Index

| Phase | Folder | Main Outcome | Pass Gate |
|---|---|---|---|
| 00 | [`00_Preflight_and_Environment_Verification`](./00_Preflight_and_Environment_Verification/README.md) | Stable Windows + Conda environment with a real model smoke test | GPU visible, bitsandbytes imports, compute capability is `12.0`, and Qwen 3B generates deterministically |
| 01 | [`01_Repo_Scaffold_and_Config`](./01_Repo_Scaffold_and_Config/README.md) | Repo skeleton, config contract, schemas, determinism utilities | Config loads, schemas validate, scaffold exists |
| 02 | [`02_Data_Pipelines_and_Dataset_Contracts`](./02_Data_Pipelines_and_Dataset_Contracts/README.md) | ASQA normalization, bounded ASQA corpus, synthetic finance dataset, manifests | Fixed splits exist, finance data regenerates identically, schemas documented |
| 03 | [`03_Retrieval_System`](./03_Retrieval_System/README.md) | Hybrid retrieval over bounded corpora with manual inspection | Relevant support appears in top-3 for at least 21 of 30 inspection cases |
| 04 | [`04_Baseline_RAG_Generation`](./04_Baseline_RAG_Generation/README.md) | Deterministic baseline generation with structured outputs | 20-example smoke run is reproducible and citation-format valid |
| 05 | [`05_Attention_Gate`](./05_Attention_Gate/README.md) | Token-level support scoring and abstention gate | Attention traces are recoverable and targeted weak-support cases can trigger abstention |
| 06 | [`06_Deterministic_Verifier`](./06_Deterministic_Verifier/README.md) | Deterministic verifier and unit tests | Malformed, missing, and wrong-passages cases are caught by tests |
| 07 | [`07_Calibration_and_Evaluation`](./07_Calibration_and_Evaluation/README.md) | Calibrated gate settings plus metrics and confidence intervals | Chosen config reduces unsupported non-abstained outputs without collapsing correctness |
| 08 | [`08_Experiments_Figures_and_Report_Assets`](./08_Experiments_Figures_and_Report_Assets/README.md) | Final comparisons, tables, figures, qualitative examples | Final result package is complete, reproducible, and report-ready |

## Required Reading Order For A Fresh Agent

1. This file
2. [`SETUP_AND_DOWNLOADS.md`](./SETUP_AND_DOWNLOADS.md)
3. The current phase README
4. The most recent file in `Progress/`
5. The implementation guide in `Docs/`

Do not start coding before you know which phase is active.
