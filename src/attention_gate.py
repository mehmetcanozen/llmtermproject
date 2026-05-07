from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.generation import GenerationSettings
from src.prompting import (
    ABSTENTION,
    build_chat_prompt,
    enforce_format_or_abstain,
    map_passage_token_spans,
    prepare_prompt_passages,
)


PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)
CITATION_FRAGMENT_RE = re.compile(r"^\[?P?[1-4]?\]?$")


@dataclass(frozen=True)
class GateSettings:
    tail_layers: int = 4
    rolling_window: int = 4
    threshold: float = 0.10
    consecutive_failures: int = 3
    skip_initial_generated_tokens: int = 8


def gate_settings_from_config(config: dict[str, Any]) -> GateSettings:
    gate = config["gate"]
    return GateSettings(
        tail_layers=int(gate["tail_layers"]),
        rolling_window=int(gate["rolling_window"]),
        threshold=float(gate["threshold"]),
        consecutive_failures=int(gate["consecutive_failures"]),
        skip_initial_generated_tokens=int(gate["skip_initial_generated_tokens"]),
    )


def is_ignored_gate_token(token_text: str) -> bool:
    stripped = token_text.strip()
    if not stripped:
        return True
    if PUNCTUATION_RE.match(stripped):
        return True
    if CITATION_FRAGMENT_RE.match(stripped):
        return True
    return False


def layer_attention_scores_to_passages(attentions: tuple[Any, ...], passage_spans: dict[str, dict[str, int]]) -> list[dict[str, float]]:
    if not attentions:
        raise ValueError("No attentions returned by model")
    layer_scores: list[dict[str, float]] = []
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        # Shape: batch, heads, query_length, key_length. Decode steps use query_length=1.
        attention_vector = layer_attention[0, :, -1, :].float().mean(dim=0)
        key_length = int(attention_vector.shape[0])
        scores_for_layer: dict[str, float] = {}
        for label, span in passage_spans.items():
            start = max(0, min(int(span["token_start"]), key_length))
            end = max(start, min(int(span["token_end_exclusive"]), key_length))
            scores_for_layer[label] = float(attention_vector[start:end].sum().item()) if end > start else 0.0
        layer_scores.append(scores_for_layer)
    if not layer_scores:
        raise ValueError("No usable attention tensors returned by model")
    return layer_scores


def aggregate_tail_layer_scores(layer_scores: list[dict[str, float]], passage_labels: list[str], tail_layers: int) -> dict[str, float]:
    selected = layer_scores[-tail_layers:] if tail_layers > 0 else layer_scores
    return {
        label: max(scores_for_layer.get(label, 0.0) for scores_for_layer in layer_scores)
        for label in passage_labels
    } if not selected else {
        label: max(scores_for_layer.get(label, 0.0) for scores_for_layer in selected)
        for label in passage_labels
    }


def score_attention_to_passages(attentions: tuple[Any, ...], passage_spans: dict[str, dict[str, int]], tail_layers: int) -> dict[str, float]:
    layer_scores = layer_attention_scores_to_passages(attentions, passage_spans)
    return aggregate_tail_layer_scores(layer_scores, list(passage_spans), tail_layers)


def generate_with_attention_gate(
    tokenizer: Any,
    model: Any,
    candidate_record: dict[str, Any],
    generation_settings: GenerationSettings,
    gate_settings: GateSettings,
    run_id: str,
    *,
    abort_on_gate: bool = True,
    store_layer_scores: bool = False,
    prompt_passage_count: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    passages = prepare_prompt_passages(candidate_record["merged_top3"], required_count=prompt_passage_count)
    prompt = build_chat_prompt(tokenizer, candidate_record["question"], passages)
    passage_spans = map_passage_token_spans(tokenizer, prompt, passages)

    encoded = tokenizer([prompt], return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    generated_ids: list[int] = []
    trace_tokens: list[dict[str, Any]] = []
    rolling_scores: list[float] = []
    consecutive_failures = 0
    gate_triggered = False
    gate_trigger_index: int | None = None

    with torch.inference_mode():
        outputs = model(
            **encoded,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

        for token_index in range(generation_settings.max_new_tokens):
            step_outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            token_id = int(next_token[0, 0].item())
            generated_ids.append(token_id)
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            layer_scores = layer_attention_scores_to_passages(step_outputs.attentions, passage_spans)
            passage_scores = aggregate_tail_layer_scores(layer_scores, list(passage_spans), gate_settings.tail_layers)
            max_support = max(passage_scores.values()) if passage_scores else 0.0
            ignored = is_ignored_gate_token(token_text) or token_index < gate_settings.skip_initial_generated_tokens
            rolling_average = None
            gate_failed = False
            if not ignored:
                rolling_scores.append(max_support)
                window = rolling_scores[-gate_settings.rolling_window :]
                rolling_average = sum(window) / len(window)
                gate_failed = len(window) >= gate_settings.rolling_window and rolling_average < gate_settings.threshold
                consecutive_failures = consecutive_failures + 1 if gate_failed else 0
            token_trace = {
                "token_index": token_index,
                "token_id": token_id,
                "token_text": token_text,
                "passage_scores": {label: round(score, 6) for label, score in passage_scores.items()},
                "support_score": round(max_support, 6),
                "ignored_for_gate": ignored,
                "rolling_average": None if rolling_average is None else round(rolling_average, 6),
                "gate_failed": gate_failed,
                "consecutive_failures": consecutive_failures,
                "gate_triggered_here": False,
            }
            if store_layer_scores:
                token_trace["layer_passage_scores"] = [
                    {label: round(score, 6) for label, score in scores.items()}
                    for scores in layer_scores
                ]
            trace_tokens.append(token_trace)
            if consecutive_failures >= gate_settings.consecutive_failures:
                gate_triggered = True
                gate_trigger_index = token_index
                trace_tokens[-1]["gate_triggered_here"] = True
                if abort_on_gate:
                    break
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            past_key_values = step_outputs.past_key_values
            next_token = step_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if gate_triggered:
        final_answer = ABSTENTION
        raw_validation = enforce_format_or_abstain(raw_answer, {passage["label"] for passage in passages})[1]
        final_validation = enforce_format_or_abstain(final_answer, {passage["label"] for passage in passages})[2]
    else:
        final_answer, raw_validation, final_validation = enforce_format_or_abstain(
            raw_answer,
            allowed_labels={passage["label"] for passage in passages},
        )

    prediction = {
        "run_id": run_id,
        "example_id": candidate_record["example_id"],
        "dataset": candidate_record["dataset"],
        "system": "gate_only",
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
            "model_id": generation_settings.model_id,
            "temperature": generation_settings.temperature,
            "do_sample": generation_settings.do_sample,
            "max_new_tokens": generation_settings.max_new_tokens,
        },
        "gate": {
            "triggered": gate_triggered,
            "trigger_token_index": gate_trigger_index,
            "settings": gate_settings.__dict__,
        },
        "metadata": {
            "raw_answer": raw_answer,
            "raw_format_valid": raw_validation.valid,
            "raw_format_errors": raw_validation.errors,
            "final_format_valid": final_validation.valid,
            "retrieval_hit_top3": candidate_record.get("hit_top3"),
            "expected_passage_ids": candidate_record.get("expected_passage_ids", []),
            "source_metadata": candidate_record.get("metadata", {}),
            "gate_label": candidate_record.get("gate_label"),
        },
    }
    trace = {
        "run_id": run_id,
        "example_id": candidate_record["example_id"],
        "dataset": candidate_record["dataset"],
        "gate_label": candidate_record.get("gate_label"),
        "question": candidate_record["question"],
        "passage_spans": passage_spans,
        "gate_settings": gate_settings.__dict__,
        "gate_triggered": gate_triggered,
        "trigger_token_index": gate_trigger_index,
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "tokens": trace_tokens,
    }
    return prediction, trace
