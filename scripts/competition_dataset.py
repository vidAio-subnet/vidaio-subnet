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
    DatasetValidationIssue,
    EvaluationIndex,
    ModalVolumeStore,
    exclude_invalid_sources,
    format_validation_issues,
    local_index_validation_issues,
    prepare_index_candidates,
    validate_local_index,
)
from vidaio_subnet_core.competition.repository import (  # noqa: E402
    CompetitionRepository,
)


def _index(path: str) -> EvaluationIndex:
    return EvaluationIndex.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _write_index(path: str, index: EvaluationIndex) -> None:
    Path(path).write_text(index.normalized_json() + "\n", encoding="utf-8")


def _add_yes_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--yes",
        action="store_true",
        help="disqualify flagged entries without interactive confirmation",
    )


def _prompt_yes_no(question: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        print(question + "yes", file=sys.stderr)
        return True
    while True:
        try:
            response = input(question).strip().lower()
        except EOFError:
            print("no", file=sys.stderr)
            return False
        if response == "yes":
            return True
        if response == "no":
            return False
        print("Please type yes or no.", file=sys.stderr)


def _review_issues(
    index: EvaluationIndex | None,
    issues: tuple[DatasetValidationIssue, ...],
    *,
    manifest: CompetitionManifest,
    command: str,
    assume_yes: bool,
    index_path: str,
) -> EvaluationIndex:
    if issues:
        print(format_validation_issues(issues), file=sys.stderr)
        count = sum(len(issue.evaluation_ids) for issue in issues)
        question = (
            f"Disqualify {count} evaluation entry/entries from {len(issues)} "
            f"source(s) and continue {command}? Type yes or no: "
        )
        if not _prompt_yes_no(question, assume_yes=assume_yes):
            raise DatasetError(
                f"operator declined to continue {command} with flagged entries"
            )
    if index is None:
        raise DatasetError("all evaluation sources are invalid")
    if issues and command != "prepare":
        index = exclude_invalid_sources(index, issues, manifest)
    if issues:
        print(
            f"Disqualified {count} evaluation entry/entries; "
            f"{len(index.items)} remain.",
            file=sys.stderr,
        )
    if issues or command == "prepare":
        _write_index(index_path, index)
    return index


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

    commands = {}
    for name in ("prepare", "validate", "upload"):
        command = commands[name] = subparsers.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--source-dir", required=True)
        command.add_argument("--index", required=True)
        _add_yes_argument(command)

    upload = commands["upload"]
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
        evaluation_index, issues = prepare_index_candidates(
            manifest, Path(args.source_dir)
        )
        evaluation_index = _review_issues(
            evaluation_index,
            issues,
            manifest=manifest,
            command="prepare",
            assume_yes=args.yes,
            index_path=args.index,
        )
        print(
            f"Prepared {len(evaluation_index.items)} item(s); "
            f"index checksum={evaluation_index.digest()}"
        )
        return 0

    evaluation_index = _index(args.index)
    if args.command in {"validate", "upload"}:
        issues = local_index_validation_issues(
            evaluation_index, manifest, Path(args.source_dir)
        )
        evaluation_index = _review_issues(
            evaluation_index,
            issues,
            manifest=manifest,
            command=args.command,
            assume_yes=args.yes,
            index_path=args.index,
        )
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

    remote_index = store.load_index(manifest, validate_for_manifest=False)
    if remote_index.digest() != evaluation_index.digest():
        raise RuntimeError("remote and local evaluation indexes differ")
    repository = CompetitionRepository(args.database_url)
    issues = store.validation_issues(manifest, remote_index)
    if issues:
        raise DatasetError(
            format_validation_issues(issues)
            + "\nThe uploaded dataset is immutable; prepare and upload a new "
            "competition ID and Volume."
        )
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
