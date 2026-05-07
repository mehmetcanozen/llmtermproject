# Deterministic Citation Enforcement in Retrieval-Augmented Generation

**Group Members:** Mehmet Can Ozen (`20210808020`) and `ADD_OTHER_MEMBER_NAMES_HERE`

## Abstract

This project studies a specific RAG failure mode: a model gives an answer with citation markers even though the cited passages do not actually support the claim. We built a local retrieval-augmented generation pipeline and compared three variants: a baseline generator, an attention-gated generator, and an attention-gated generator with a deterministic verifier. The goal was to see whether unsupported cited answers could be reduced without retraining the language model. We evaluated the system on a bounded ASQA-derived corpus and on a synthetic finance dataset designed to stress exact numeric and period-sensitive claims. In the final saved evaluation package, the verifier-based system reduced unsupported non-abstained outputs to `0.0%` on both datasets. We also ran a generated distractor stress test where each prompt included one additional plausible but irrelevant passage; there too, the verifier projection reduced unsupported cited answers to `0.0%`, with a clear abstention cost.

## Problem Definition and Motivation

RAG systems are often presented as safer than plain language models because they retrieve evidence before answering. However, retrieval alone does not guarantee grounded behavior. A model can still produce a confident answer with citation markers that look correct while misreading the source text, mixing details from different passages, or citing a passage that does not support the claim. This is especially risky in domains such as finance, law, or policy, where a wrong answer with a formal-looking citation can appear more trustworthy than an uncited hallucination.

Our project focuses on this citation-grounding problem. Instead of only improving prompts, we tried to add explicit inference-time controls. The main idea was to use internal support signals during generation and then apply a deterministic post-check that can reject unsupported outputs.

## Dataset / Data Source

We used two datasets.

First, we built a bounded local corpus from ASQA-style question answering data. This part of the project is useful for citation-grounded long-form QA, but it should be described honestly: it is not an open-web retrieval benchmark. Retrieval happens only over the local corpus prepared inside the repository.

Second, we created a synthetic finance dataset. These examples are intentionally fictional, but they are useful because they allow strict checking of company name, fiscal period, metric type, numeric value, and cited passage identity. This made the finance set a good stress test for citation hallucination and exact-support verification.

For the locked final evaluation, the project uses `200` ASQA evaluation examples from `data/asqa/splits/dev_eval_200.jsonl` and `100` finance examples from `data/finance/generated/questions.jsonl`.

## LLM Methodology

The final implementation differs slightly from the early proposal. While the proposal discussed a Llama-based setup, the implemented project uses local Qwen2.5 models and a custom Python pipeline that was more stable in the available Windows environment.

The system has three variants:

1. `baseline`: deterministic RAG generation with retrieved passages and required sentence-level citations.
2. `gate_only`: the baseline plus a passage-directed attention support gate. During generation, the system tracks whether the produced tokens still appear to attend to the retrieved support passages.
3. `gate_plus_verifier`: the gate output is passed through a deterministic verifier that checks citation structure and support conditions, then rejects unsupported outputs by abstaining.

Retrieval is hybrid dense-plus-BM25 ranking over the local corpus. Prompt construction uses the top passages, and generation is deterministic (`temperature=0`, `do_sample=False`). The verifier is rule-based rather than learned. On finance examples, it checks exact fields such as company, period, metric, value, and cited passage. On ASQA examples, it uses a support proxy based on citation presence, cited-passage existence, and whether explicit numbers, years, or quoted spans appear in the cited text.

## Experiments and Results

The final package in `outputs/evaluation/evaluation_manifest.json` reports `formal_full_eval_pass: true`, which means the saved locked 3B evaluation artifacts are complete for the intended final split.

The most important results are:

- On **ASQA**, unsupported non-abstained rate improved from `3.5%` in the baseline to `2.5%` with the gate, and to `0.0%` with gate plus verifier.
- On **ASQA**, answer coverage also improved from `49.5%` to `54.5%` when the gate was introduced, and the verifier kept the same coverage while increasing abstention slightly.
- On **Finance**, unsupported non-abstained rate was `1.0%` for the baseline, `2.0%` for gate-only, and `0.0%` for gate plus verifier.
- On **Finance**, exact answer accuracy was `62.0%` for all three systems in the saved package, showing that the verifier improved safety mainly by rejecting unsupported outputs rather than by making the model more knowledgeable.
- In the **generated ASQA distractor** stress test, unsupported non-abstained rate was `2.0%` for baseline, `2.5%` for gate-only, and `0.0%` for gate plus verifier.
- In the **generated Finance distractor** stress test, unsupported non-abstained or false-attribution rate was `17.0%` for both baseline and gate-only, and `0.0%` for gate plus verifier. This stronger safety result came with abstention increasing to `61.0%`.

Overall, the strongest conclusion is that the deterministic verifier is the most effective safety layer in this repository. The attention gate alone gives mixed results, but the gate-plus-verifier configuration consistently removes unsupported cited outputs in the saved evaluation artifacts.

We also produced figures, comparison tables, and qualitative examples in `outputs/final/`. The qualitative cases are useful because they show concrete failure modes rather than only aggregate metrics.

The most meaningful project output is therefore not a broad answer-quality breakthrough. It is a robustness result about citation safety: under both the normal fixed split and a harder generated distractor condition, the verifier catches unsupported cited answers and converts them into explicit abstentions.

## Discussion and Limitations

This project should be interpreted as an engineering prototype, not as a universal claim about faithfulness in RAG.

The ASQA part is limited by the bounded local-corpus design. The support checks there are deterministic proxies and do not replace human factuality evaluation. The finance dataset is synthetic, which makes exact checking easier but means the results should not be described as deployment evidence. The generated distractor experiment is a separate robustness stress test and does not replace the normal fixed-split evaluation. The older static distractor proxy in the artifact package should be treated as diagnostic only.

Another important limitation is that the verifier mostly improves safety through abstention. In other words, it is better at stopping risky outputs than at increasing underlying answer-generation quality. That is still useful, but it should be described clearly.

## Conclusion

This project shows that inference-time controls can make a RAG system more conservative and more auditable. In the saved final evaluation package, the best safety result came from combining retrieval, deterministic generation, an attention-based support signal, and a deterministic verifier that rejects unsupported cited answers. The generated distractor stress test makes the result more meaningful because the same verifier behavior appears when prompts contain an additional plausible irrelevant passage. The final system does not solve factuality in a general sense, but it demonstrates a practical way to turn some citation hallucinations into explicit abstentions instead of silently incorrect answers.

## References

1. Gao et al., 2023. ALCE.
2. Stelmakh et al., 2022. ASQA.
3. Jain and Wallace, 2019. Attention is not Explanation.
4. Wiegreffe and Pinter, 2019. Attention is not not Explanation.
5. Ding et al., 2025. Attention attribution for source tracing.
