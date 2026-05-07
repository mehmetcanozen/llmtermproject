from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.final_assets import build_final_assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 08 final tables, figures, examples, and report notes.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "final")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir)
    manifest = build_final_assets(REPO_ROOT, output_dir)
    print(
        json.dumps(
            {
                "status": manifest["package_status"],
                "formal_full_eval_pass": manifest["formal_full_eval_pass"],
                "tables": manifest["tables"],
                "figures": manifest["figures"],
                "example_count": manifest["examples"]["count"],
                "report_notes": manifest["report_notes"],
                "scope_warning": manifest["scope_warning"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
