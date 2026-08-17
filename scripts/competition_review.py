#!/usr/bin/env python3
"""Inspect competition rankings and record audited human-review decisions."""

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

from vidaio_subnet_core.competition.repository import CompetitionRepository  # noqa: E402


def _review_dict(review) -> dict[str, object]:
    return {
        "id": review.id,
        "competition_id": review.competition_id,
        "operator_identity": review.operator_identity,
        "review_type": review.review_type,
        "contenders": json.loads(review.contenders_json),
        "decision": json.loads(review.decision_json),
        "reason": review.reason,
        "supersedes_review_id": review.supersedes_review_id,
        "superseded": bool(review.superseded),
        "integrity_hash": review.integrity_hash,
        "created_at": review.created_at,
    }


def _add_common_mutation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--competition-id", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="required acknowledgement that this records a review and mutates state",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default="sqlite:///video_subnet_validator.db")
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser(
        "list",
        help="show the provisional/final ranking packet and audit review history",
    )
    listing.add_argument("--competition-id", required=True)

    readability = subparsers.add_parser(
        "resolve-readability",
        help="accept or reject a contender paused in REVIEW_REQUIRED",
    )
    _add_common_mutation_arguments(readability)
    readability.add_argument("--hotkey", required=True)
    readability.add_argument("--decision", required=True, choices=("accept", "reject"))

    disqualify = subparsers.add_parser(
        "disqualify",
        help="exclude a contender and recalculate scores when already scored",
    )
    _add_common_mutation_arguments(disqualify)
    disqualify.add_argument("--hotkey", required=True)

    tie = subparsers.add_parser(
        "order-exact-tie",
        help="order every contender in one exact rounded-score tie group",
    )
    _add_common_mutation_arguments(tie)
    tie.add_argument(
        "--hotkey",
        dest="ordered_hotkeys",
        action="append",
        required=True,
        help="repeat in the desired first-to-last order",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = CompetitionRepository(args.database_url)
    if args.command == "list":
        print(
            json.dumps(
                repository.competition_review_packet(args.competition_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.apply:
        raise SystemExit(f"{args.command} requires --apply")
    now = datetime.now(timezone.utc)
    if args.command == "resolve-readability":
        review = repository.resolve_readability_review(
            args.competition_id,
            args.hotkey,
            accepted=args.decision == "accept",
            operator_identity=args.operator,
            reason=args.reason,
            now=now,
        )
    elif args.command == "disqualify":
        review = repository.disqualify_contender(
            args.competition_id,
            args.hotkey,
            operator_identity=args.operator,
            reason=args.reason,
            now=now,
        )
    else:
        review = repository.order_exact_tie(
            args.competition_id,
            args.ordered_hotkeys,
            operator_identity=args.operator,
            reason=args.reason,
            now=now,
        )
    print(json.dumps(_review_dict(review), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
