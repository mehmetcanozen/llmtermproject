# Submission Guide: Report, GitHub, and 5-10 Minute Project Video

Project: Deterministic Citation Enforcement in Retrieval-Augmented Generation  
Group member: Mehmet Can Ozen, Student ID 20210808020  
Repository: https://github.com/mehmetcanozen/llmtermproject

This guide is built directly around the course submission criteria in `docs/meta/SubmissionCriteria.txt`: final report PDF, GitHub repository, and a 5-10 minute project video with a short demo. The strongest final project claim is:

> A verifier-bounded repair RAG system can recover useful answer coverage while keeping unsupported accepted citations at 0.0% on the fixed split and on a generated distractor stress test.

## What To Submit

Submit these items:

1. Final report PDF.
   - Use the IEEE-style source in `docs/submission/final_report_ieee/final_report.tex`.
   - The report already includes the required sections: title and group member, abstract, problem/motivation, data, methodology, experiments/results, limitations, conclusion, and references.
   - If you compile on Overleaf, upload the whole `docs/submission/final_report_ieee` folder so the figures resolve.

2. GitHub repository link.
   - Use: `https://github.com/mehmetcanozen/llmtermproject`
   - Mention that bulky regenerated outputs are ignored, while lightweight final tables, figures, examples, and source code are included.

3. Project video.
   - Length: 5-10 minutes.
   - Recommended target: 7-8 minutes. This is long enough to show the project properly without rambling.
   - Submit one video for the group.

## Video Storyline

Use this as the core video arc:

1. Problem: RAG answers can include citations that look authoritative but do not support the claim.
2. Task: build a local RAG pipeline and measure citation safety on ASQA plus a synthetic finance stress test.
3. Method: compare baseline RAG, attention gate, verifier projection, and repair-plus-verifier.
4. Result: repair-plus-verifier 3B reaches 0.0% unsupported non-abstained answers while improving finance exact accuracy and ASQA short-answer coverage.
5. Demo: show a prediction, verifier output, final tables/figures, and the generated distractor robustness slice.
6. Limitations: ASQA support is proxy-based, finance is synthetic, and 7B became too conservative.

## Recommended 7-8 Minute Video Timeline

### 0:00-0:30 Opening

Say:

> My project is about citation hallucination in retrieval-augmented generation. The failure mode I focus on is when an LLM gives an answer with a citation marker, but the cited passage does not actually support the claim. This is risky because the answer looks auditable even when it is wrong.

Show:

- `README.md`
- Project title
- Repository root

Do not spend much time on the file tree yet.

### 0:30-1:15 Project Goal

Say:

> The goal is not simply to make the model answer more often. The goal is to make accepted cited answers safer. The final system is verifier-bounded: it can repair an answer once, but it only accepts the repair if the deterministic verifier passes.

Show:

- `outputs/final/FINAL_REPORT.md`
- The "Research Question" or "System Variants" section

Key phrase to use:

> The repair step is allowed to recover coverage, but the verifier remains the final acceptance boundary.

### 1:15-2:00 Data And Task

Say:

> I evaluate on two sources. ASQA gives ambiguous natural-language questions, but I use it as a bounded local-corpus citation task. The finance dataset is synthetic and controlled, so exact company, period, metric, numeric value, and cited passage can be checked deterministically.

Show:

- `data/asqa/splits/dev_eval_200.jsonl`
- `data/finance/generated/questions.jsonl`
- `configs/default.yaml`

Mention:

- ASQA fixed evaluation: 200 examples.
- Finance fixed evaluation: 100 examples.
- Generated distractor stress test: 40 ASQA and 20 finance examples with a fourth irrelevant passage added.

Important limitation:

> ASQA is not evaluated as open-web retrieval here. It is a bounded local-corpus experiment.

### 2:00-3:15 Method

Say:

> The pipeline starts with hybrid retrieval: dense BGE embeddings plus BM25. The generator is deterministic Qwen2.5-Instruct. Every factual sentence must end with a citation marker. Then the verifier checks citation format and support. For finance, it checks exact value and citation. For ASQA, it checks citation structure plus explicit anchors and short-answer coverage.

Show these files:

- `src/retrieval.py`
- `src/prompting.py`
- `src/generation.py`
- `src/verifier.py`

Then explain the four variants:

- `baseline`: deterministic RAG with citations.
- `gate_only`: adds the attention gate.
- `gate_plus_verifier`: projects gate outputs through verifier rejection.
- `repair_plus_verifier`: generate, verify, repair once if needed, verify again, otherwise abstain.

Use a simple verbal flow:

```text
retrieve passages -> generate cited answer -> verify
if verifier fails -> repair using same evidence -> verify again
if still fails -> INSUFFICIENT_SUPPORT
```

### 3:15-4:15 Demo

Best low-risk demo: do not rerun the full model live. Show saved artifacts and optionally run a small verifier command.

Show:

- `outputs/runs/locked/repair_plus_verifier_finance_finance_full_100_3b_overhaul/predictions.jsonl`
- `outputs/evaluation/verifier_verdicts.json`
- `outputs/final/examples/`

Good demo command:

```powershell
$py='C:\Users\omehm\anaconda3\envs\llm-citation\python.exe'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& $py scripts\run_eval_suite.py
```

This command is fast and prints:

- `formal_full_eval_pass: true`
- `repair_plus_full_eval_pass: true`
- systems include `repair_plus_verifier`

If you want to show a model run, use only a tiny one-example or three-example smoke run, not a full split.

### 4:15-6:00 Results

Show:

- `outputs/final/tables/system_comparison.csv`
- `outputs/final/tables/repair_salvage.csv`
- `outputs/final/tables/generated_distractor_metrics.csv`
- `outputs/final/figures/safety_vs_coverage_frontier.png`
- `outputs/final/figures/repair_funnel.png`
- `outputs/final/figures/generated_distractor_robustness.png`

Say these numbers clearly:

Fixed split:

- Baseline ASQA unsupported non-abstained: 3.5%.
- Gate-plus-verifier ASQA unsupported non-abstained: 0.0%.
- Repair-plus-verifier 3B ASQA unsupported non-abstained: 0.0%.
- Repair-plus-verifier 3B ASQA short-answer coverage: 33.5%, versus 26.1% for gate-plus-verifier.
- Baseline finance exact accuracy: 62.0%.
- Repair-plus-verifier 3B finance exact accuracy: 80.0%.
- Repair-plus-verifier 3B finance answer coverage: 65.0%, versus 47.0% for baseline and gate-plus-verifier.

Repair funnel:

- Finance 3B: 42 repair attempts, 41 accepted repairs, 0 unsupported accepted repairs.
- ASQA 3B: 24 repair attempts, 11 accepted repairs, 0 unsupported accepted repairs.

Generated distractor stress test:

- ASQA: 40 examples, 0.0% unsupported non-abstained, 55.0% answer coverage.
- Finance: 20 examples, 0.0% unsupported non-abstained, 85.0% exact accuracy.

Interpretation:

> The best system is repair-plus-verifier 3B. It keeps the same zero-unsupported accepted-answer safety boundary as the verifier, but recovers useful coverage, especially in finance.

### 6:00-7:00 Limitations

Say:

> The ASQA verifier is a deterministic proxy, not a human factuality judge. It checks citation structure and explicit anchors, but not full semantic truth. The finance dataset is synthetic by design, so it is a controlled stress test rather than real financial evidence. Finally, Qwen 7B was more conservative here: it also had 0.0% unsupported accepted answers, but it abstained much more, especially on finance.

Show:

- `outputs/final/report_notes.md`
- Limitations section in the report

Do not hide the limitations. They make the project look more honest and technically mature.

### 7:00-7:45 Closing

Say:

> The final result is that a verifier-bounded repair loop can improve answer usefulness without giving up citation safety. The system does not prove universal factuality, but it demonstrates a practical local RAG safety pattern: unsupported cited claims become abstentions, and supported repair opportunities can be recovered.

Show:

- `outputs/final/final_manifest.json`
- `outputs/final/FINAL_REPORT.md`
- GitHub repository page if available

End with:

> The code, final figures, tables, examples, and reproducibility scripts are all included in the repository.

## Screen Recording Checklist

Before recording:

- Close unrelated windows and notifications.
- Use a readable terminal font.
- Zoom editor to around 110-125%.
- Keep the repository open at `C:\SoftwareProjects\LLMTermProject`.
- Open these files/tabs in advance:
  - `README.md`
  - `src/generation.py`
  - `src/verifier.py`
  - `outputs/final/tables/system_comparison.csv`
  - `outputs/final/tables/repair_salvage.csv`
  - `outputs/final/tables/generated_distractor_metrics.csv`
  - `outputs/final/figures/safety_vs_coverage_frontier.png`
  - `outputs/final/figures/repair_funnel.png`
  - `outputs/final/figures/generated_distractor_robustness.png`
  - `outputs/final/final_manifest.json`

During recording:

- Do not scroll through huge JSON files for too long.
- When showing code, point to function names rather than reading every line.
- Keep claims tied to visible artifacts.
- Use exact terms: "unsupported non-abstained" and "abstention".
- Say "synthetic finance stress test", not "real financial benchmark".

After recording:

- Check that the video is 5-10 minutes.
- Check that your voice is audible.
- Check that code and tables are readable.
- Include the GitHub link in the submitted text field or report.

## Short Script You Can Read

Use this if you want a tight narration:

> This project studies citation hallucination in retrieval-augmented generation. I built a local RAG system where generated answers must cite retrieved passages. The key issue is that a citation marker can look trustworthy even when the cited passage does not support the claim. To address that, I implemented a deterministic verifier and then extended the system with a repair-plus-verifier path. The repair path generates an answer, verifies it, attempts one evidence-only repair if verification fails, and accepts the result only if the verifier passes. Otherwise it outputs `INSUFFICIENT_SUPPORT`.
>
> I evaluated on a bounded ASQA split and a synthetic finance dataset. The finance task is useful because exact company, period, metric, numeric value, and cited passage can be checked deterministically. The strongest result is the 3B repair-plus-verifier system. On the full fixed split it keeps unsupported accepted answers at 0.0%, improves finance exact accuracy from 62.0% to 80.0%, and improves ASQA short-answer coverage compared with the verifier-only projection. In the generated distractor stress test, where each prompt includes an additional irrelevant passage, the same system still has 0.0% unsupported non-abstained answers.
>
> The limitations are important. The ASQA support check is a proxy rather than a human factuality audit, and the finance dataset is synthetic. Still, the project demonstrates a practical safety pattern for local RAG: the model can recover useful supported answers through repair, but unsupported cited answers are converted into explicit abstentions.

## Final Submission Checklist

- Final report PDF created from `docs/submission/final_report_ieee/final_report.tex`.
- Report includes group member name: Mehmet Can Ozen.
- GitHub repo is live and linked.
- Video is 5-10 minutes.
- Video covers:
  - project goal,
  - data/task,
  - LLM method,
  - experiments/results,
  - limitations,
  - demo.
- Do one final repository check:

```powershell
$py='C:\Users\omehm\anaconda3\envs\llm-citation\python.exe'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& $py scripts\run_eval_suite.py
& $py -m pytest tests -q --basetemp outputs\test\pytest_tmp
```

Expected:

- `formal_full_eval_pass: true`
- `repair_plus_full_eval_pass: true`
- `43 passed`
