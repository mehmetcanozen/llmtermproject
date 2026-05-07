from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from preflight_smoke import DEFAULT_MODELS_ROOT, find_model_dir, require


def count_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def load_qwen(model_path: Path, use_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, messages: list[dict], max_new_tokens: int = 20) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def determinism_check(tokenizer, model) -> dict:
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply with exactly five words about citations."},
    ]
    run_1 = generate_text(tokenizer, model, messages)
    run_2 = generate_text(tokenizer, model, messages)
    matched = run_1 == run_2
    require(matched, "Determinism check failed: repeated runs produced different outputs")
    return {
        "prompt": messages[-1]["content"],
        "run_1": run_1,
        "run_2": run_2,
        "matched": matched,
    }


def strict_instruction_following_check(tokenizer, model) -> dict:
    messages = [
        {"role": "system", "content": "You are a precise assistant."},
        {
            "role": "user",
            "content": "Reply with exactly five words about citations. Use plain words only and no numbering.",
        },
    ]
    generated = generate_text(tokenizer, model, messages)
    words = count_words(generated)
    passed = len(words) == 5
    return {
        "prompt": messages[-1]["content"],
        "generated_text": generated,
        "word_count": len(words),
        "words": words,
        "passed": passed,
        "severity": "advisory",
    }


def attention_output_check(tokenizer, model) -> dict:
    import torch

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Summarize the need for citations in one sentence."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        prompt_outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    require(prompt_outputs.attentions is not None, "Prompt forward pass did not return attentions")
    require(len(prompt_outputs.attentions) > 0, "Prompt attentions are empty")
    prompt_last = prompt_outputs.attentions[-1]
    require(prompt_last is not None, "Last prompt-layer attention is None")
    require(prompt_last.ndim == 4, f"Expected 4D prompt attentions, got shape {tuple(prompt_last.shape)}")
    require(torch.isfinite(prompt_last).all().item(), "Prompt attentions contain non-finite values")

    prompt_last_token_mean_mass = float(prompt_last[:, :, -1, :].sum(dim=-1).mean().item())

    next_token = prompt_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.inference_mode():
        decode_outputs = model(
            input_ids=next_token,
            past_key_values=prompt_outputs.past_key_values,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    require(decode_outputs.attentions is not None, "Decode-step forward pass did not return attentions")
    require(len(decode_outputs.attentions) > 0, "Decode-step attentions are empty")
    decode_last = decode_outputs.attentions[-1]
    require(decode_last is not None, "Last decode-layer attention is None")
    require(decode_last.ndim == 4, f"Expected 4D decode attentions, got shape {tuple(decode_last.shape)}")
    require(torch.isfinite(decode_last).all().item(), "Decode attentions contain non-finite values")

    decode_last_token_mean_mass = float(decode_last[:, :, -1, :].sum(dim=-1).mean().item())
    require(
        0.95 <= prompt_last_token_mean_mass <= 1.05,
        f"Prompt attention mass looks invalid: {prompt_last_token_mean_mass}",
    )
    require(
        0.95 <= decode_last_token_mean_mass <= 1.05,
        f"Decode attention mass looks invalid: {decode_last_token_mean_mass}",
    )

    return {
        "prompt_attentions_layers": len(prompt_outputs.attentions),
        "prompt_last_layer_shape": list(prompt_last.shape),
        "prompt_last_token_mean_mass": prompt_last_token_mean_mass,
        "decode_attentions_layers": len(decode_outputs.attentions),
        "decode_last_layer_shape": list(decode_last.shape),
        "decode_last_token_mean_mass": decode_last_token_mean_mass,
        "used_eager_backend": True,
        "usable_for_token_level_gate": True,
    }


def print_result(prefix: str, payload: dict) -> None:
    print(prefix)
    for key, value in payload.items():
        print(f"  - {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deeper verification checks for determinism and attention extraction.")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--qwen3b-path", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--no-4bit", action="store_true", help="Load the model without 4-bit quantization.")
    parser.add_argument(
        "--strict-instruction",
        action="store_true",
        help="Treat the exact-five-words instruction-following check as a blocking failure.",
    )
    args = parser.parse_args()

    results: dict[str, dict] = {}
    try:
        qwen3b_path = args.qwen3b_path or find_model_dir(
            args.models_root,
            markers=["qwen", "2.5", "3b", "instruct"],
            required_files=["config.json", "generation_config.json", "tokenizer.json"],
        )
        results["discovered_paths"] = {"qwen3b_path": str(qwen3b_path)}
        print("[INFO] Discovered local assets")
        print(f"  - qwen3b_path: {qwen3b_path}")

        tokenizer, model = load_qwen(qwen3b_path, use_4bit=not args.no_4bit)
        results["model_load"] = {
            "model_path": str(qwen3b_path),
            "used_4bit": not args.no_4bit,
            "attn_implementation": "eager",
        }
        print_result("[PASS] Model load", results["model_load"])

        results["determinism_check"] = determinism_check(tokenizer, model)
        print_result("[PASS] Determinism check", results["determinism_check"])

        results["attention_output_check"] = attention_output_check(tokenizer, model)
        print_result("[PASS] Attention output check", results["attention_output_check"])

        results["instruction_following_check"] = strict_instruction_following_check(tokenizer, model)
        if results["instruction_following_check"]["passed"]:
            print_result("[PASS] Exact-five-words instruction check", results["instruction_following_check"])
        else:
            print_result("[WARN] Exact-five-words instruction check", results["instruction_following_check"])
            if args.strict_instruction:
                raise RuntimeError(
                    "Strict instruction-following check failed: model did not return exactly five words."
                )

        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"[INFO] Wrote JSON report to {args.json_out}")

        print("[PASS] Deep verification checks completed successfully")
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": "failed",
                "error": str(exc),
                "partial_results": results,
            }
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[INFO] Wrote partial JSON report to {args.json_out}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
