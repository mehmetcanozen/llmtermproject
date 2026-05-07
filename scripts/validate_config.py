from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_CONFIG_PATH, validate_config_contract
from src.utils.determinism import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the project config and Phase 01 JSON schemas.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = validate_config_contract(args.config)
    seed_report = set_global_seed(args.seed)
    payload = {
        "status": "passed",
        "config": str(args.config),
        "contract": result,
        "seed_report": seed_report.__dict__,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
