#!/usr/bin/env python3
"""Prepare, verify, upload, and seal a competition evaluation dataset."""

from __future__ import annotations

import argparse
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import only the competition package when this file is executed directly. The
# repository's top-level package eagerly imports the validator stack, which is
# unrelated to dataset preparation and may not be installed in an operator CLI
# environment.
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
from vidaio_subnet_core.competition.dataset import (  # noqa: E402
    DatasetError,
    EvaluationIndex,
    ModalVolumeStore,
    prepare_index,
    validate_local_index,
)
from vidaio_subnet_core.competition.repository import (  # noqa: E402
    CompetitionRepository,
)


def _index(path: str) -> EvaluationIndex:
    return EvaluationIndex.model_validate_json(Path(path).read_text(encoding="utf-8"))


def ensure_manifest_registered(
    repository: CompetitionRepository,
    manifest: CompetitionManifest,
    *,
    now: datetime,
    actor: str,
) -> bool:
    """Register a missing manifest without silently replacing an existing one."""

    existing = repository.get(manifest.competition_id)
    if existing is None:
        repository.insert_manifest(manifest, now=now, actor=actor)
        return True
    if existing.manifest_digest != manifest.digest():
        raise RuntimeError(
            "database contains a different manifest revision for "
            f"{manifest.competition_id}; use the validator's registered manifest"
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--source-dir", required=True)
    prepare.add_argument("--index", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--source-dir", required=True)
    validate.add_argument("--index", required=True)

    upload = subparsers.add_parser("upload")
    upload.add_argument("--manifest", required=True)
    upload.add_argument("--source-dir", required=True)
    upload.add_argument("--index", required=True)
    upload.add_argument("--environment", default="main")

    seal = subparsers.add_parser("seal")
    seal.add_argument("--manifest", required=True)
    seal.add_argument("--index", required=True)
    seal.add_argument("--environment", default="main")
    seal.add_argument("--database-url", default="sqlite:///video_subnet_validator.db")
    seal.add_argument("--actor", default="competition-dataset-cli")

    args = parser.parse_args()
    manifest = load_manifest(args.manifest)

    if args.command == "prepare":
        evaluation_index = prepare_index(manifest, Path(args.source_dir))
        Path(args.index).write_text(
            evaluation_index.normalized_json() + "\n", encoding="utf-8"
        )
        print(
            f"Prepared {len(evaluation_index.items)} item(s); "
            f"index checksum={evaluation_index.digest()}"
        )
        return 0

    evaluation_index = _index(args.index)
    evaluation_index.validate_for_manifest(manifest)
    if args.command == "validate":
        validate_local_index(evaluation_index, manifest, Path(args.source_dir))
        print(
            f"Validated {len(evaluation_index.items)} item(s); "
            f"index checksum={evaluation_index.digest()}"
        )
        return 0

    store = ModalVolumeStore(environment_name=args.environment)
    if args.command == "upload":
        store.upload(
            manifest,
            evaluation_index,
            Path(args.source_dir),
        )
        print(
            f"Uploaded and read-back verified {len(evaluation_index.items)} item(s) "
            f"to {manifest.evaluation_input_volume_name}"
        )
        return 0

    remote_index = store.load_index(manifest)
    if remote_index.digest() != evaluation_index.digest():
        raise RuntimeError("remote and local evaluation indexes differ")
    repository = CompetitionRepository(args.database_url)
    now = datetime.now(timezone.utc)
    registered = ensure_manifest_registered(
        repository,
        manifest,
        now=now,
        actor=args.actor,
    )
    digest = repository.seal_evaluation_dataset(
        manifest.competition_id,
        evaluation_index,
        now=now,
        actor=args.actor,
    )
    if registered:
        print(
            f"Registered {manifest.competition_id} as SCHEDULED in {args.database_url}"
        )
    print(
        f"Sealed {len(evaluation_index.items)} item(s) for "
        f"{manifest.competition_id}; index checksum={digest}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetError as exc:
        print(f"Dataset validation failed:\n{exc}", file=sys.stderr)
        raise SystemExit(2) from None
