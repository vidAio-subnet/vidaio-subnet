#!/usr/bin/env python3
"""Validate a competition manifest, warmup media, and optional contender source."""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep this local security check independent from the full validator/Bittensor stack.
for package_name, package_path in (
    ("vidaio_subnet_core", ROOT / "vidaio_subnet_core"),
    ("vidaio_subnet_core.competition", ROOT / "vidaio_subnet_core" / "competition"),
):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

from vidaio_subnet_core.competition.config import (  # noqa: E402
    CompetitionManifest,
    load_manifest,
)
from vidaio_subnet_core.competition.qualification import (  # noqa: E402
    preflight_warmup_fixture,
)
from vidaio_subnet_core.competition.validation import (  # noqa: E402
    RepositoryStaticValidator,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--repository", type=Path, help="Also run static repository validation"
    )
    parser.add_argument(
        "--validate-boss",
        action="store_true",
        help=(
            "Require and statically validate the boss SDK export configured by "
            "boss.repository_path in the supplied manifest"
        ),
    )
    return parser.parse_args(argv)


def validate_manifest_boss(
    manifest: CompetitionManifest,
    repository_root: Path,
) -> dict[str, object]:
    """Validate the configured boss identity, export shape, and source policy."""

    boss_path = manifest.boss.repository_path
    boss_hotkey = manifest.boss.boss_hotkey
    if boss_path is None or boss_hotkey is None:
        raise ValueError(
            "--validate-boss requires boss.repository_path and boss.boss_hotkey "
            "in the supplied manifest"
        )

    # Reuse the manifest's containment and complete-SDK-export checks before
    # applying the same objective static policy used for miner submissions.
    manifest.validate_runtime_paths(repository_root)
    resolved_boss = (repository_root.resolve(strict=True) / boss_path).resolve(
        strict=True
    )
    report = RepositoryStaticValidator().validate(resolved_boss)
    return {
        "boss_hotkey": boss_hotkey,
        "repository_path": boss_path.as_posix(),
        "resolved_repository_path": str(resolved_boss),
        "repository": report.as_dict(),
        "status": report.status.value,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    media = preflight_warmup_fixture(manifest, ROOT)
    statuses = ["ACCEPTED"]
    result: dict[str, object] = {
        "competition_id": manifest.competition_id,
        "manifest_digest": manifest.digest(),
        "warmup": media.__dict__,
        "status": "ACCEPTED",
    }
    if args.repository:
        report = RepositoryStaticValidator().validate(args.repository)
        result["repository"] = report.as_dict()
        statuses.append(report.status.value)
    if args.validate_boss:
        boss = validate_manifest_boss(manifest, ROOT)
        result["boss"] = boss
        statuses.append(str(boss["status"]))
    status_priority = {"ACCEPTED": 0, "REVIEW_REQUIRED": 1, "REJECTED": 2}
    result["status"] = max(statuses, key=status_priority.__getitem__)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ACCEPTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
