from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attention_gate import gate_settings_from_config
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.generation import generation_settings_from_config, load_generation_model
from src.locked_runs import LockedRunRequest, load_or_build_candidates, run_locked_generation
from src.utils.determinism import set_global_seed


def gate_settings_from_yaml(path: Path):
    import yaml
    from src.attention_gate import GateSettings

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = payload.get("chosen_gate_config", payload)
    return GateSettings(
        tail_layers=int(settings["tail_layers"]),
        rolling_window=int(settings["rolling_window"]),
        threshold=float(settings["threshold"]),
        consecutive_failures=int(settings["consecutive_failures"]),
        skip_initial_generated_tokens=int(settings["skip_initial_generated_tokens"]),
    )


def default_output_dir(system: str, dataset: str, split: str, model_size: str, run_tag: str, distractor: bool) -> Path:
    root = REPO_ROOT / "outputs" / "runs" / "locked"
    suffix = f"{system}_{dataset}_{split}_{model_size}_{run_tag}"
    if distractor:
        suffix += "_distractor"
    return root / suffix


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable locked generation on fixed project splits.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--system", choices=["baseline", "gate_only", "repair_plus_verifier"], required=True)
    parser.add_argument("--model-size", choices=["3b", "7b"], default="3b")
    parser.add_argument("--dataset", choices=["asqa", "finance"], required=True)
    parser.add_argument("--split", choices=["train_calibration_100", "dev_eval_200", "finance_full_100"], required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--run-tag", default="locked")
    parser.add_argument("--prompt-passage-count", type=int, choices=[3, 4], default=3)
    parser.add_argument("--collect-traces", action="store_true")
    parser.add_argument("--no-gate-abort", action="store_true")
    parser.add_argument("--store-layer-scores", action="store_true")
    parser.add_argument("--gate-config", type=Path, help="Optional chosen_gate_config.yaml for gate_only runs.")
    parser.add_argument("--distractor", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument("--retrieval-device", default="cpu")
    parser.add_argument("--rebuild-candidates", action="store_true")
    args = parser.parse_args()

    if args.dataset == "asqa" and args.split == "finance_full_100":
        parser.error("ASQA runs cannot use finance_full_100")
    if args.dataset == "finance" and args.split != "finance_full_100":
        parser.error("Finance runs must use finance_full_100")
    if args.distractor and args.prompt_passage_count != 4:
        parser.error("--distractor requires --prompt-passage-count 4")
    if args.system != "gate_only" and (args.collect_traces or args.store_layer_scores or args.no_gate_abort):
        parser.error("Trace/gate options only apply to gate_only")

    config = load_config(args.config)
    set_global_seed(int(config["project"]["seed"]))
    settings = generation_settings_from_config(config, args.model_size)
    if args.max_new_tokens is not None:
        settings = type(settings)(**{**settings.__dict__, "max_new_tokens": args.max_new_tokens})
    if args.system == "gate_only" and args.gate_config:
        gate_path = args.gate_config if args.gate_config.is_absolute() else REPO_ROOT / args.gate_config
        gate_settings = gate_settings_from_yaml(gate_path)
    else:
        gate_settings = gate_settings_from_config(config) if args.system == "gate_only" else None

    candidates = load_or_build_candidates(
        REPO_ROOT,
        config,
        args.dataset,
        args.split,
        use_existing=not args.rebuild_candidates,
        retrieval_batch_size=args.retrieval_batch_size,
        retrieval_device=args.retrieval_device,
    )
    output_dir = args.output_dir or default_output_dir(
        args.system,
        args.dataset,
        args.split,
        args.model_size,
        args.run_tag,
        args.distractor,
    )
    request = LockedRunRequest(
        system=args.system,
        model_size=args.model_size,
        dataset=args.dataset,
        split=args.split,
        output_dir=output_dir,
        limit=args.limit,
        start=args.start,
        resume=not args.no_resume,
        prompt_passage_count=args.prompt_passage_count,
        collect_traces=args.collect_traces,
        abort_on_gate=not args.no_gate_abort,
        store_layer_scores=args.store_layer_scores,
        distractor=args.distractor,
        run_tag=args.run_tag,
    )

    tokenizer, model = load_generation_model(settings)
    manifest = run_locked_generation(
        repo_root=REPO_ROOT,
        config_path=args.config,
        config=config,
        request=request,
        tokenizer=tokenizer,
        model=model,
        generation_settings=settings,
        gate_settings=gate_settings,
        candidates=candidates,
    )
    print(
        json.dumps(
            {
                "status": "passed" if manifest["failure_count"] == 0 else "failed",
                "run_id": manifest["run_id"],
                "completed_count": manifest["completed_count"],
                "formal_split_complete": manifest["formal_split_complete"],
                "manifest": str((Path(manifest["output_paths"][0]).parent / "run_manifest.json").as_posix()),
            },
            indent=2,
        )
    )
    return 0 if manifest["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
