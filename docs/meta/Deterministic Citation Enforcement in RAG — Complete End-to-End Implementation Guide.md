# Deterministic Citation Enforcement in RAG
## Revised End-to-End Guide for the LLM Course Project

Mehmet Can Ozen - Student ID: 20210808020  
Revision date: 2026-04-14  
Target machine: GeForce RTX 5070 12 GB + Windows + local development

---

## 1. Executive Summary

Your project idea is strong, but the old guide was trying to be a publishable systems paper and a coding tutorial at the same time. That made it overconfident in a few places, especially around hardware assumptions, attention-based attribution claims, and evaluation.

This revised guide is built for your actual goal:

1. Finish a technically solid course project.
2. Make a defensible claim that survives instructor scrutiny.
3. Keep the implementation realistic on an RTX 5070 with 12 GB VRAM.

The project should be framed as:

> Can an inference-time grounding gate, based on passage-directed attention plus post-generation verification, reduce unsupported cited answers in single-document or small-context RAG without any fine-tuning?

That wording is strong, honest, and achievable.

---

## 2. What Needed Fixing

| Old guide issue | Why it is risky | Revised position |
|---|---|---|
| Treated the RTX 5070 like `sm_100` | Current NVIDIA tables list GeForce RTX 5070 as compute capability 12.0, not 10.0 | Assume Blackwell `CC 12.0` and verify with a smoke test |
| Treated decoder-only Llama attention as "cross-attention" | Llama/Qwen-style causal LMs do not have encoder-decoder cross-attention in normal RAG prompting | Measure passage-directed **self-attention** over prompt tokens |
| Treated attention as if it proves grounding | Attention-based evidence is useful, but the literature does not support claiming it is a proof of faithful attribution | Use attention as an online signal, then add explicit citation verification |
| Used fuzzy string matching as the core citation metric | Too weak for an attribution project | Use ALCE-style citation evaluation on ASQA and exact/structured scoring on finance |
| Hard-coded software versions that are already aging | This is fragile, especially on a new GPU generation | Pin only after your first successful run; build around currently supported toolchains |
| Used a very ambitious one-model / one-threshold story | Too brittle and too easy to break in practice | Compare `baseline`, `gate only`, and `gate + verifier` |
| Overclaimed likely outcomes like `>95% citation accuracy` | Not defensible before you run the system | Use relative-improvement success criteria instead of inflated absolute promises |

---

## 3. Final Project Scope

### 3.1 Core Research Question

Does an inference-time grounding mechanism reduce citation hallucination in RAG, without retraining the model?

### 3.2 Recommended Claim

Your final report should claim:

- passage-directed attention can be used as an online grounding signal during generation;
- a calibrated attention gate can reduce unsupported non-abstained answers;
- attention alone is not enough, so a post-generation verifier materially improves reliability.

Do **not** claim:

- that attention proves faithfulness;
- that your method solves citation faithfulness in general;
- that the system is reliable for legal or financial deployment.

### 3.3 Deliverable You Should Actually Build

Build and compare three systems:

1. `Vanilla RAG baseline`
2. `Attention-gated RAG`
3. `Attention-gated RAG + deterministic verifier`

This is much stronger than only comparing a baseline against one constrained system, because it separates:

- the value of the online gate;
- the value of the verifier;
- the tradeoff between abstention and citation quality.

---

## 4. What the Literature Supports

These are the ideas your guide should stand on:

- Lewis et al. introduced RAG as a retrieval-conditioned generation setup rather than a pure parametric-memory setup.
- Jain and Wallace argue attention should not automatically be treated as explanation.
- Wiegreffe and Pinter argue the interpretation question is more nuanced, but they still do not justify treating attention as hard proof.
- Gao et al. introduced ALCE, which is the right direction for automatic citation evaluation because it measures correctness and citation quality rather than crude lexical overlap.
- Bohnet et al. and the Attributed-QA release formalize answer-plus-attribution evaluation and provide AutoAIS-style support checking.
- Wallat et al. argue that correctness and faithfulness are not the same thing in RAG attribution, which is exactly why your final system should include a verifier and a faithfulness diagnostic.
- Xu et al. show that citation evaluation should ideally move toward fine-grained, claim-level metrics rather than only sentence-level citation presence.

The key consequence is simple:

> Attention can be one signal in your system, but it should not be the only mechanism you rely on to argue that a citation is trustworthy.

---

## 5. Recommended Project Design

### 5.1 System Overview

Use a standard RAG pipeline, then add two reliability layers:

1. `Retriever`
2. `Generator`
3. `Online grounding gate`
4. `Post-generation verifier`

The control flow is:

```text
question
  -> retrieve top-k passages
  -> build prompt with numbered passages
  -> generate answer token by token with KV cache
  -> monitor passage-directed self-attention
  -> abort if support stays too low
  -> verify each cited sentence against its cited passage
  -> return answer or abstain
```

### 5.2 Why This Version Is Better

- It matches your proposal's "deterministic enforcement" idea.
- It is still novel enough for a course project.
- It avoids the biggest conceptual weakness of the old guide: acting like attention weights are the whole story.

---

## 6. Model and Hardware Plan for RTX 5070 12 GB

### 6.1 Important Hardware Correction

Current NVIDIA CUDA GPU tables list `GeForce RTX 5070` under compute capability `12.0`.

That means your smoke test should expect `major == 12`, not `10`.

### 6.2 Recommended Model Choice

Use this order:

1. `Qwen/Qwen2.5-7B-Instruct` as the main model
2. `Qwen/Qwen2.5-3B-Instruct` as the fallback if attention extraction is too slow or memory is too tight

Why this is better than the old Llama choice:

- Qwen2.5 is not gated, so you avoid Hugging Face access friction.
- The model card explicitly recommends current `transformers` support.
- For a class project, removing access friction matters more than choosing a brand-name checkpoint.

If you already have Llama access and want to keep it, that is fine, but do not make the project depend on gated downloads.

### 6.3 Practical VRAM Advice

Even with 4-bit quantization, your real enemy is not only model weights. It is:

- long prompts,
- returned attention tensors,
- KV cache growth,
- and repeated forward passes in the custom decode loop.

So for the first working version:

- keep `top_k = 3`;
- chunk retrieved passages to around `180-250` tokens each;
- cap the prompt at about `900-1200` total input tokens;
- use `max_new_tokens <= 160` for evaluation;
- use `temperature = 0`.

That recommendation is an engineering inference for your hardware budget. It is safer than starting with huge prompts.

### 6.4 Windows vs WSL2

Start on Windows. Do **not** force a WSL2 migration unless something actually breaks.

Use WSL2 only if:

- `bitsandbytes` fails to load the correct CUDA kernels;
- Windows wheels lag behind the versions you need;
- or your Python environment becomes unstable.

As of the currently visible bitsandbytes installation docs, recent Windows builds already target modern architectures including `sm120` for newer CUDA toolkits, so the old "Windows is hopeless" assumption is no longer the best default.

---

## 7. Software Stack

Keep the stack small and boring.

### 7.1 Recommended Core Dependencies

- `torch`
- `torchvision`
- `torchaudio`
- `transformers`
- `accelerate`
- `bitsandbytes`
- `sentence-transformers`
- `chromadb`
- `rank-bm25`
- `datasets`
- `evaluate`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `python-dotenv`
- `pytest`

### 7.2 What to Avoid

Avoid making the project depend heavily on LangChain unless you already have working code that needs it.

Reason:

- LangChain adds abstraction noise;
- its import paths move often;
- and it is not helping the scientific core of your project.

For this project, plain Python plus a small number of direct libraries is more bulletproof.

### 7.3 Versioning Strategy

Do not blindly pin old versions from the previous guide.

Use this process instead:

1. Install the current officially supported PyTorch build for Windows and CUDA from the PyTorch selector.
2. Install the latest stable `transformers`, `accelerate`, and `bitsandbytes`.
3. Run a smoke test.
4. Only after a successful run, freeze the environment with:

```bash
pip freeze > requirements-lock.txt
```

That gives you a reproducible environment without guessing stale versions in advance.

---

## 8. Environment Setup

### 8.1 Create the Environment

```bash
conda create -n llm-citation python=3.11 -y
conda activate llm-citation
```

### 8.2 Install PyTorch

Use the current official PyTorch command from the selector for `Windows + Pip + Python + CUDA`.

For a Conda-first setup, install PyTorch through conda so the CUDA runtime aligns with your environment:

```bash
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.4
```

As of 2026-04-14, this setup targets modern RTX 5070 stacks. If your selector suggests a specific CUDA build for your machine, use that instead.

### 8.3 Install the Rest

```bash
pip install --upgrade pip
pip install -U transformers accelerate bitsandbytes sentence-transformers chromadb rank-bm25 datasets evaluate pandas scikit-learn scipy matplotlib seaborn python-dotenv pytest
```

### 8.4 Smoke Test

Run a tiny script before writing model code:

```python
import torch
import bitsandbytes as bnb

assert torch.cuda.is_available(), "CUDA is not available"
props = torch.cuda.get_device_properties(0)
print(props.name)
print("compute capability:", f"{props.major}.{props.minor}")
print("torch:", torch.__version__)
print("bnb:", bnb.__version__)
```

What you want:

- the GPU is visible;
- compute capability prints as `12.0`;
- `bitsandbytes` imports cleanly.

If `bitsandbytes` imports but quantized loading still misbehaves, test model loading directly before moving on.

---

## 9. Recommended Repository Layout

Use something closer to this:

```text
llm_citation_project/
  Docs/
  data/
    asqa/
    finance/
  outputs/
    runs/
    figures/
    tables/
  src/
    config.py
    data_loading.py
    retrieval.py
    prompting.py
    generation.py
    attention_gate.py
    verifier.py
    evaluation.py
    utils.py
  scripts/
    smoke_test.py
    build_finance_dataset.py
    calibrate_gate.py
    run_asqa_baseline.py
    run_asqa_gated.py
    run_asqa_gated_verified.py
    run_finance_eval.py
    plot_results.py
  tests/
```

This keeps the project focused on the experiment rather than on framework plumbing.

---

## 10. Data Plan

### 10.1 Primary Dataset: ASQA

Use ASQA as your main benchmark because:

- it is a real long-form QA dataset;
- it has ambiguous factoid questions;
- and it is already used in citation-focused evaluation work such as ALCE.

Recommended split usage:

- use a small slice of `train` for calibration only;
- use `dev` for your reported results;
- leave `test` untouched unless your course setup explicitly allows it.

Do not calibrate on the same split you report as final.

### 10.2 Secondary Dataset: Synthetic Finance Set

Keep the synthetic finance dataset, but change its role.

It should be a **controlled stress test**, not your only serious evaluation.

Use it for cases like:

- conflicting quarterly numbers;
- near-duplicate company names;
- unanswerable questions;
- exact-number citation support;
- passage retrieval collisions.

This is valuable because it gives you a clean place to test deterministic behavior.

### 10.3 Optional Faithfulness Probe

Add one lightweight diagnostic inspired by recent attribution-faithfulness work:

- retrieve the normal top-k passages;
- then inject one irrelevant but plausible distractor passage;
- rerun generation;
- measure how often the cited support changes in a way that is not justified.

This gives you a very nice discussion section:

- correctness may stay the same;
- faithfulness may still fail;
- your gate and verifier may reduce that gap.

That is a much more mature story than "attention thresholding solved citation hallucination."

---

## 11. Retrieval Design

### 11.1 Retriever

Use a hybrid retriever:

- dense retrieval with a sentence embedding model;
- lexical retrieval with BM25;
- merge the results;
- rerank by a simple weighted score.

### 11.2 Embedding Model

Use a compact embedding model on CPU.

Recommended:

- `BAAI/bge-small-en-v1.5`

This is a better fit than a very large embedding model because:

- the project bottleneck is generation and evaluation, not dense embedding quality;
- CPU embedding is enough here;
- and smaller embeddings reduce friction.

### 11.3 Retrieval Settings

Recommended starting values:

- dense top-k: `5`
- BM25 top-k: `5`
- merged top-k for prompting: `3`
- chunk size: `180-250` tokens
- chunk overlap: `30-50` tokens

### 11.4 Retrieval Sanity Check

Before working on the gate, manually inspect `30` examples:

- `10` ASQA examples
- `20` finance examples

If retrieval is bad, your attribution experiment is meaningless. Fix retrieval first.

---

## 12. Prompt Design

Keep the prompt strict and simple.

### 12.1 Prompt Template

```text
You are a question-answering system that must use only the provided passages.

Rules:
- Answer only with information supported by the passages.
- Every factual sentence must end with one or more citations like [P1] or [P2].
- If the passages do not contain enough evidence, output exactly: INSUFFICIENT_SUPPORT
- Do not use outside knowledge.

Question:
{question}

Passages:
[P1] {passage_1}
[P2] {passage_2}
[P3] {passage_3}

Answer:
```

### 12.2 Why This Prompt Works Better

- It gives you deterministic citation markers.
- It makes sentence-level verification easier.
- It gives the model a valid abstention path before the gate has to intervene.

### 12.3 Generation Settings

For experiments, keep decoding deterministic:

- `temperature = 0`
- `do_sample = False`
- `top_p = None`
- `max_new_tokens = 120-160`

This is the right choice for a scientific comparison.

---

## 13. The Grounding Gate

### 13.1 Important Conceptual Correction

For a decoder-only model, what you are measuring is not classic cross-attention to an external encoder. It is the amount of **self-attention from the current generated token to the token span of each retrieved passage inside the prompt**.

Call it:

> passage-directed self-attention

That wording is accurate.

### 13.2 What to Measure

At each decode step:

1. run the next-token forward pass with `use_cache=True`;
2. request `output_attentions=True`;
3. average attention over the last few layers and heads;
4. sum the attention mass landing on each passage span;
5. use the maximum passage support, or the support of the currently cited passage if you track citations explicitly.

### 13.3 Practical Scoring Rule

Use a rolling score, not a single-token trigger.

Recommended first version:

- aggregate over the last `4` transformer layers;
- compute support for each generated token;
- maintain a rolling average over the last `4` generated content tokens;
- trigger abstention only if the rolling score stays below threshold for `3` consecutive checks.

This is better than a raw single-token cutoff because punctuation, stopwords, and formatting tokens are noisy.

### 13.4 When Not to Apply the Gate

Skip or downweight gate checks for:

- punctuation-only tokens;
- citation bracket tokens like `[` `]` `P1`;
- the first few generated tokens;
- the exact abstention phrase if generation is already moving there.

### 13.5 Recommended Initial Hyperparameters

Start here:

- tail layers: `4`
- threshold: `0.10`
- rolling window: `4`
- consecutive failures: `3`
- minimum generated tokens before abort: `8`

Then tune on a calibration set.

### 13.6 What the Gate Should Output

Use one fixed abstention string:

```text
INSUFFICIENT_SUPPORT
```

Do not generate long refusal prose. A fixed token sequence is easier to score.

---

## 14. Why the Verifier Is Mandatory

The verifier is the part that turns your project from "interesting attention hack" into "serious citation-reliability study."

### 14.1 What the Verifier Does

After generation:

1. split the answer into sentences;
2. extract each sentence's citation markers;
3. map each marker to the cited passage;
4. reject the answer if a factual sentence has no citation;
5. reject or flag the answer if the cited passage does not support the sentence.

### 14.2 Two Verification Modes

Use different verification rules for the two datasets.

#### ASQA

Do not rely on fuzzy string overlap.

Instead:

- score final outputs with the ALCE evaluation code where possible;
- optionally add sentence-to-passage NLI as a local analysis tool if you have time.

#### Finance

Use deterministic structured checks:

- exact company match;
- exact quarter and year match;
- exact numeric support for revenue or earnings fields;
- citation must point to the correct source passage.

This is where your "deterministic enforcement" story is strongest.

### 14.3 Best Three-System Comparison

The comparison you should report is:

1. `Baseline`
2. `Gate only`
3. `Gate + verifier`

This will let you show whether:

- the gate helps on its own;
- the verifier catches additional unsupported outputs;
- or the verifier alone does most of the work.

---

## 15. Evaluation Plan

### 15.1 Primary Metrics

Use these as your main reported metrics:

| Metric | Meaning | Where to use it |
|---|---|---|
| Answer correctness | Whether the answer is correct | ASQA and finance |
| Citation quality | Whether cited passages support the answer | ASQA |
| Unsupported non-abstained rate | Bad answers that slipped through without abstaining | ASQA and finance |
| Abstention rate | How often the system refuses | ASQA and finance |
| Retrieval hit rate | Whether a supporting passage was retrieved in top-k | ASQA and finance |

### 15.2 Recommended ASQA Evaluation

Use ALCE-style evaluation rather than inventing your own citation metric.

That is the most important evaluation upgrade in this revised guide.

If using the ALCE repository directly is too heavy for your local workflow, at minimum structure your evaluation to mirror its categories:

- correctness
- citation quality
- optional fluency

### 15.3 Recommended Finance Evaluation

Report:

- exact answer accuracy
- correct citation rate
- false attribution rate
- abstention rate on unanswerable questions

Because the finance set is synthetic, you control the ground truth and can score this cleanly.

### 15.4 Optional Fine-Grained Citation Metric

If time remains, add a small extension based on the ALiiCE idea:

- decompose the answer into atomic claims;
- check whether each claim has a valid citation;
- report citation precision and recall at the claim level.

This is optional, not required.

---

## 16. Calibration Plan

Do not overcomplicate calibration.

### 16.1 Calibration Set

Use:

- `150-200` ASQA training examples, or
- a smaller slice like `100` if runtime is tight.

### 16.2 What to Tune

Tune only a small set:

- threshold: `[0.05, 0.10, 0.15, 0.20, 0.25]`
- tail layers: `[2, 4, 8]`
- consecutive failures: `[2, 3, 4]`

### 16.3 Selection Rule

Select the configuration that:

1. minimizes unsupported non-abstained answers;
2. while keeping answer correctness within `10` percentage points of baseline;
3. and keeping abstention rate reasonable.

That is much better than maximizing one metric in isolation.

---

## 17. Experimental Plan

### Experiment A: Retrieval sanity

Goal:

- confirm top-k retrieval actually includes relevant passages.

### Experiment B: Main ASQA comparison

Compare:

1. baseline
2. gate only
3. gate + verifier

Report:

- answer correctness
- citation quality
- abstention rate
- unsupported non-abstained rate

### Experiment C: Finance stress test

Measure:

- exact numeric correctness
- correct citation accuracy
- false attribution rate
- abstention on unanswerables

### Experiment D: Faithfulness probe

Add a distractor passage and observe:

- citation changes
- correctness changes
- verifier rejections

This experiment can become one of your most interesting qualitative findings.

---

## 18. Statistical Analysis

For per-example comparisons between systems, use:

- paired bootstrap confidence intervals;
- and either paired Wilcoxon or approximate randomization.

If your instructor is not strict about significance testing, paired bootstrap CIs alone are already useful and easy to explain.

Do not overcomplicate this part.

---

## 19. Realistic Success Criteria

Do not promise `>95%` citation accuracy before you run anything.

Use relative, defensible goals instead.

| Outcome tier | Recommended interpretation |
|---|---|
| Minimum success | Gate or gate+verifier reduces unsupported non-abstained answers relative to baseline |
| Strong success | Reliability improves with only a modest drop in answer correctness |
| Excellent success | Gate+verifier improves citation quality clearly and is more robust in the distractor-passage probe |

For the finance set, a good concrete target is:

- `>= 90%` correct citation precision on answerable numeric questions
- strong abstention behavior on unanswerable questions

For ASQA, do not set a fake absolute target. Report the observed baseline and the relative gain.

---

## 20. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Quantized model does not behave correctly on your Windows stack | Medium | High | Verify with a tiny real model load before building anything else |
| Attention extraction is too slow on 7B | Medium | Medium | Debug on 3B first, then scale up |
| Long prompts cause memory issues | High | High | Keep top-k small and prompt length capped |
| Retrieval quality is weak | Medium | High | Run manual retrieval inspection before gate experiments |
| Gate over-abstains | High | Medium | Calibrate on a small train slice |
| Citation formatting drifts | Medium | Medium | Use deterministic prompt instructions and a verifier |
| Results look good on correctness but weak on faithfulness | Medium | Medium | Report that honestly and use the distractor probe |
| Bibliography contains unverified references | Medium | Medium | Only cite papers you can verify directly |

---

## 21. What Not to Do

- Do not describe decoder-only prompt attention as cross-attention.
- Do not use fuzzy string matching as the main attribution metric.
- Do not rely on one single threshold result without ablations.
- Do not evaluate only on a synthetic dataset.
- Do not promise legal or financial safety.
- Do not claim attention proves faithful use of the source.
- Do not build the whole project around a gated model download if an ungated alternative works.

---

## 22. Recommended Writing Angle for the Final Report

The strongest final story is:

1. RAG systems can still hallucinate citations.
2. A decoder's passage-directed self-attention gives a useful online signal of grounding.
3. That signal can be used to gate generation and force abstention.
4. However, recent attribution work shows that correctness and faithfulness differ.
5. Therefore, the best practical student system is a two-stage design: online gate plus post-generation verifier.

That is a much stronger and more mature paper narrative than "attention thresholding solves citation hallucination."

---

## 23. Suggested Week-by-Week Plan

| Week | Goal |
|---|---|
| 1 | Environment setup, smoke test, tiny quantized model load |
| 2 | Retrieval pipeline and prompt template |
| 3 | Baseline generation and finance dataset construction |
| 4 | Attention-gated decoding loop |
| 5 | Verifier implementation |
| 6 | Calibration on ASQA train slice |
| 7 | Final ASQA dev and finance experiments |
| 8 | Figures, tables, qualitative examples, report writing |

---

## 24. References You Should Actually Use

These are verified sources that support the revised guide:

1. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"  
   https://arxiv.org/abs/2005.11401

2. Stelmakh et al., "ASQA: Factoid Questions Meet Long-Form Answers"  
   https://aclanthology.org/2022.emnlp-main.566/

3. Gao et al., "Enabling Large Language Models to Generate Text with Citations"  
   https://aclanthology.org/2023.emnlp-main.398/

4. Bohnet et al., "Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models" and the Attributed-QA release  
   https://github.com/google-research-datasets/Attributed-QA  
   https://arxiv.org/abs/2212.08037

5. Jain and Wallace, "Attention is not Explanation"  
   https://aclanthology.org/N19-1357/

6. Wiegreffe and Pinter, "Attention is not not Explanation"  
   https://aclanthology.org/D19-1002/

7. Wallat et al., "Correctness is not Faithfulness in RAG Attributions"  
   https://arxiv.org/abs/2412.18004

8. Xu et al., "ALiiCE: Evaluating Positional Fine-grained Citation Generation"  
   https://aclanthology.org/2025.naacl-long.23/

9. NVIDIA CUDA GPU compute capability table  
   https://developer.nvidia.com/cuda-gpus

10. PyTorch install and previous versions pages  
    https://pytorch.org/get-started/locally  
    https://pytorch.org/get-started/previous-versions

11. bitsandbytes installation guide  
    https://huggingface.co/docs/bitsandbytes/main/en/installation

12. Qwen2.5 model card  
    https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

13. Transformers attention backend documentation  
    https://huggingface.co/docs/transformers

---

## 25. Final Recommendation

If you follow only one high-level recipe, use this one:

1. Start with `Qwen2.5-3B` for debugging.
2. Move to `Qwen2.5-7B` once the pipeline works.
3. Keep prompts short.
4. Compare `baseline`, `gate only`, and `gate + verifier`.
5. Use ASQA as the main benchmark and finance as the controlled stress test.
6. Evaluate citation quality with ALCE-style metrics, not fuzzy overlap.
7. Describe the mechanism as passage-directed self-attention, not cross-attention.
8. Present attention as a grounding signal, not a proof of faithfulness.

That version of the project is ambitious enough to be impressive, but realistic enough to finish well.
