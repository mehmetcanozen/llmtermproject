from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.prompting import ABSTENTION, build_chat_prompt, build_repair_chat_prompt, enforce_format_or_abstain, prepare_prompt_passages


@dataclass(frozen=True)
class GenerationSettings:
    model_id: str
    model_path: str
    max_new_tokens: int = 160
    temperature: float = 0.0
    do_sample: bool = False
    load_in_4bit: bool = True
    attn_implementation: str = "eager"


def generation_settings_from_config(config: dict[str, Any], model_size: str = "3b") -> GenerationSettings:
    if model_size == "7b":
        model_id = config["model"]["main_model_id"]
        model_path = config["model"]["local_main_path"]
    else:
        model_id = config["model"]["bringup_model_id"]
        model_path = config["model"]["local_bringup_path"]
    return GenerationSettings(
        model_id=model_id,
        model_path=model_path,
        max_new_tokens=int(config["prompt"]["max_new_tokens"]),
        temperature=float(config["prompt"]["temperature"]),
        do_sample=bool(config["prompt"]["do_sample"]),
        load_in_4bit=bool(config["model"]["quantization"]["load_in_4bit"]),
        attn_implementation=config["model"]["attention"]["implementation"],
    )


def load_generation_model(settings: GenerationSettings):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = None
    if settings.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_path,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation=settings.attn_implementation,
    )
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def deterministic_generate(tokenizer: Any, model: Any, prompt: str, settings: GenerationSettings) -> str:
    import torch

    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    kwargs = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
        "temperature": settings.temperature,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        outputs = model.generate(**inputs, **kwargs)
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def retrieved_passages_for_record(candidate_record: dict[str, Any], required_count: int = 3) -> list[dict[str, Any]]:
    return prepare_prompt_passages(candidate_record["merged_top3"], required_count=required_count)


def generate_run_record(
    tokenizer: Any,
    model: Any,
    candidate_record: dict[str, Any],
    settings: GenerationSettings,
    run_id: str,
    system: str = "baseline",
    prompt_passage_count: int = 3,
) -> dict[str, Any]:
    passages = retrieved_passages_for_record(candidate_record, required_count=prompt_passage_count)
    prompt = build_chat_prompt(tokenizer, candidate_record["question"], passages, dataset=candidate_record["dataset"])
    raw_answer = deterministic_generate(tokenizer, model, prompt, settings)
    final_answer, raw_validation, final_validation = enforce_format_or_abstain(
        raw_answer,
        allowed_labels={passage["label"] for passage in passages},
    )
    return {
        "run_id": run_id,
        "example_id": candidate_record["example_id"],
        "dataset": candidate_record["dataset"],
        "system": system,
        "question": candidate_record["question"],
        "retrieved_passages": [
            {
                "label": passage["label"],
                "passage_id": passage["passage_id"],
                "text": passage["text"],
                "rank": passage["rank"],
                "score": passage["score"],
            }
            for passage in passages
        ],
        "answer": final_answer,
        "abstained": final_validation.abstained,
        "citations": final_validation.citations,
        "generation": {
            "model_id": settings.model_id,
            "temperature": settings.temperature,
            "do_sample": settings.do_sample,
            "max_new_tokens": settings.max_new_tokens,
        },
        "metadata": {
            "raw_answer": raw_answer,
            "raw_format_valid": raw_validation.valid,
            "raw_format_errors": raw_validation.errors,
            "final_format_valid": final_validation.valid,
            "retrieval_hit_top3": candidate_record.get("hit_top3"),
            "expected_passage_ids": candidate_record.get("expected_passage_ids", []),
            "source_metadata": candidate_record.get("metadata", {}),
        },
    }


def _prediction_record(
    *,
    run_id: str,
    example_id: str,
    dataset: str,
    system: str,
    question: str,
    passages: list[dict[str, Any]],
    answer: str,
    validation: Any,
    settings: GenerationSettings,
    metadata: dict[str, Any],
    verifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "run_id": run_id,
        "example_id": example_id,
        "dataset": dataset,
        "system": system,
        "question": question,
        "retrieved_passages": [
            {
                "label": passage["label"],
                "passage_id": passage["passage_id"],
                "text": passage["text"],
                "rank": passage["rank"],
                "score": passage["score"],
            }
            for passage in passages
        ],
        "answer": answer,
        "abstained": validation.abstained,
        "citations": validation.citations,
        "generation": {
            "model_id": settings.model_id,
            "temperature": settings.temperature,
            "do_sample": settings.do_sample,
            "max_new_tokens": settings.max_new_tokens,
        },
        "metadata": metadata,
    }
    if verifier is not None:
        record["verifier"] = verifier
    return record


def generate_repair_plus_verifier_record(
    tokenizer: Any,
    model: Any,
    candidate_record: dict[str, Any],
    settings: GenerationSettings,
    run_id: str,
    *,
    finance_gold: dict[str, dict[str, Any]],
    asqa_gold: dict[str, dict[str, Any]],
    prompt_passage_count: int = 3,
) -> dict[str, Any]:
    from src.verifier import verify_record

    passages = retrieved_passages_for_record(candidate_record, required_count=prompt_passage_count)
    allowed_labels = {passage["label"] for passage in passages}
    prompt = build_chat_prompt(tokenizer, candidate_record["question"], passages, dataset=candidate_record["dataset"])
    raw_answer = deterministic_generate(tokenizer, model, prompt, settings)
    initial_answer, raw_validation, initial_validation = enforce_format_or_abstain(raw_answer, allowed_labels=allowed_labels)
    base_metadata = {
        "strategy": "repair_plus_verifier",
        "raw_answer": raw_answer,
        "raw_format_valid": raw_validation.valid,
        "raw_format_errors": raw_validation.errors,
        "final_format_valid": initial_validation.valid,
        "retrieval_hit_top3": candidate_record.get("hit_top3"),
        "expected_passage_ids": candidate_record.get("expected_passage_ids", []),
        "source_metadata": candidate_record.get("metadata", {}),
        "repair_attempted": False,
        "repair_errors": [],
        "accepted_after_repair": False,
    }
    initial_record = _prediction_record(
        run_id=run_id,
        example_id=candidate_record["example_id"],
        dataset=candidate_record["dataset"],
        system="repair_plus_verifier",
        question=candidate_record["question"],
        passages=passages,
        answer=initial_answer,
        validation=initial_validation,
        settings=settings,
        metadata=base_metadata,
    )
    initial_verdict = verify_record(initial_record, finance_gold=finance_gold, asqa_gold=asqa_gold)
    if initial_verdict["summary"]["passed"]:
        initial_record["verifier"] = {
            "initial_summary": initial_verdict["summary"],
            "final_summary": initial_verdict["summary"],
        }
        initial_record["metadata"]["strategy"] = "initial_verified"
        return initial_record

    repair_prompt = build_repair_chat_prompt(
        tokenizer,
        candidate_record["question"],
        passages,
        failed_answer=raw_answer or initial_answer,
        verifier_errors=initial_verdict["summary"]["errors"],
        dataset=candidate_record["dataset"],
    )
    repair_raw = deterministic_generate(tokenizer, model, repair_prompt, settings)
    repaired_answer, repair_raw_validation, repair_validation = enforce_format_or_abstain(
        repair_raw,
        allowed_labels=allowed_labels,
    )
    repair_metadata = {
        **base_metadata,
        "strategy": "repair_attempt",
        "repair_attempted": True,
        "repair_errors": initial_verdict["summary"]["errors"],
        "initial_answer": initial_answer,
        "raw_repair_answer": repair_raw,
        "repair_raw_format_valid": repair_raw_validation.valid,
        "repair_raw_format_errors": repair_raw_validation.errors,
        "final_format_valid": repair_validation.valid,
    }
    repaired_record = _prediction_record(
        run_id=run_id,
        example_id=candidate_record["example_id"],
        dataset=candidate_record["dataset"],
        system="repair_plus_verifier",
        question=candidate_record["question"],
        passages=passages,
        answer=repaired_answer,
        validation=repair_validation,
        settings=settings,
        metadata=repair_metadata,
    )
    repair_verdict = verify_record(repaired_record, finance_gold=finance_gold, asqa_gold=asqa_gold)
    if repair_verdict["summary"]["passed"]:
        repaired_record["metadata"]["strategy"] = "accepted_after_repair"
        repaired_record["metadata"]["accepted_after_repair"] = True
        repaired_record["verifier"] = {
            "initial_summary": initial_verdict["summary"],
            "repair_summary": repair_verdict["summary"],
            "final_summary": repair_verdict["summary"],
        }
        return repaired_record

    abstention_validation = enforce_format_or_abstain(ABSTENTION, allowed_labels=allowed_labels)[2]
    final_record = _prediction_record(
        run_id=run_id,
        example_id=candidate_record["example_id"],
        dataset=candidate_record["dataset"],
        system="repair_plus_verifier",
        question=candidate_record["question"],
        passages=passages,
        answer=ABSTENTION,
        validation=abstention_validation,
        settings=settings,
        metadata={
            **repair_metadata,
            "strategy": "repair_rejected_abstained",
            "accepted_after_repair": False,
            "rejected_repair_errors": repair_verdict["summary"]["errors"],
        },
        verifier={
            "initial_summary": initial_verdict["summary"],
            "repair_summary": repair_verdict["summary"],
            "final_summary": verify_record(
                {
                    **repaired_record,
                    "answer": ABSTENTION,
                    "abstained": True,
                    "citations": [],
                },
                finance_gold=finance_gold,
                asqa_gold=asqa_gold,
            )["summary"],
        },
    )
    return final_record


def stable_jsonl(records: list[dict[str, Any]]) -> str:
    import json

    return "".join(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n" for record in records)
