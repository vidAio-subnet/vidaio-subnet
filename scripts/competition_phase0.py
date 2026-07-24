#!/usr/bin/env python3
"""Run local competition Phase 0 gates without importing the validator stack."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vidaio_subnet_core" / "competition" / "phase0.py"


def _load_phase0_module():
    spec = importlib.util.spec_from_file_location("vidaio_competition_phase0", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Phase 0 module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path (parent directory must already exist)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless every gate has passed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase0 = _load_phase0_module()
    report = phase0.phase0_report(phase0.run_local_phase0_gates())
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)

    if args.output:
        output = args.output.expanduser().resolve()
        if not output.parent.is_dir():
            raise FileNotFoundError(f"report parent does not exist: {output.parent}")
        output.write_text(rendered + "\n", encoding="utf-8")

    if report["overall_status"] == "failed":
        return 1
    if args.strict and not report["production_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
