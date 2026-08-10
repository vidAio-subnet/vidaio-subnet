#!/usr/bin/env python3
"""Perform explicit, audited repairs of competition evaluation state."""

from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for package_name, package_path in (
    ("vidaio_subnet_core", ROOT / "vidaio_subnet_core"),
    ("vidaio_subnet_core.competition", ROOT / "vidaio_subnet_core" / "competition"),
):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

from vidaio_subnet_core.competition.repository import (  # noqa: E402
    INFRASTRUCTURE_FAILURE_REASON_CODES,
    CompetitionRepository,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    requeue = subparsers.add_parser(
        "requeue-infrastructure",
        help="preserve and requeue exhausted validator-infrastructure attempts",
    )
    requeue.add_argument("--competition-id", required=True)
    requeue.add_argument(
        "--database-url", default="sqlite:///video_subnet_validator.db"
    )
    requeue.add_argument(
        "--reason-code",
        action="append",
        choices=sorted(INFRASTRUCTURE_FAILURE_REASON_CODES),
        help="limit repair to a reason code; repeat to select several",
    )
    requeue.add_argument("--actor", default="competition-repair-cli")
    requeue.add_argument(
        "--apply",
        action="store_true",
        help="required acknowledgement that this mutates competition state",
    )
    args = parser.parse_args()
    if not args.apply:
        parser.error("requeue-infrastructure requires --apply")

    repository = CompetitionRepository(args.database_url)
    reason_codes = frozenset(args.reason_code or INFRASTRUCTURE_FAILURE_REASON_CODES)
    result = repository.requeue_infrastructure_failures(
        args.competition_id,
        reason_codes=reason_codes,
        now=datetime.now(timezone.utc),
        actor=args.actor,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
