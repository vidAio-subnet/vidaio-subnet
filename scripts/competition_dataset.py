#!/usr/bin/env python3
"""Prepare, validate, upload, and seal a competition evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    format_validation_issues,
    prepare_index_candidates,
)
from vidaio_subnet_core.competition.repository import (  # noqa: E402
    CompetitionRepository,
)


def _index(path: str) -> EvaluationIndex:
    return EvaluationIndex.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _write_index(path: str, index: EvaluationIndex) -> None:
    Path(path).write_text(index.normalized_json() + "\n", encoding="utf-8")


class SourceFileSnapshot(BaseModel):
    """Cheap identity check for a source already hashed and probed by prepare."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_source_path: str
    size_bytes: int = Field(ge=0)
    device: int
    inode: int
    modified_ns: int
    changed_ns: int


class DatasetPipelineReceipt(BaseModel):
    """Local proof that the required dataset pipeline stages ran in order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    stage: Literal["prepared", "validated", "uploaded", "sealed"]
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_root: str
    sources: tuple[SourceFileSnapshot, ...]
    environment: str | None = None
    volume_name: str | None = None
    updated_at: datetime

    def normalized_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )


def _receipt_path(index_path: str) -> Path:
    path = Path(index_path)
    return path.with_name(path.name + ".pipeline.json")


def _write_receipt(index_path: str, receipt: DatasetPipelineReceipt) -> None:
    destination = _receipt_path(index_path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(receipt.normalized_json() + "\n")
        temporary = Path(stream.name)
    temporary.replace(destination)


def _load_receipt(index_path: str) -> DatasetPipelineReceipt:
    path = _receipt_path(index_path)
    try:
        return DatasetPipelineReceipt.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise DatasetError(
            f"pipeline receipt is missing: {path}; run prepare first"
        ) from exc
    except Exception as exc:
        raise DatasetError(f"pipeline receipt is invalid: {path}: {exc}") from exc


def _source_snapshots(
    index: EvaluationIndex, source_root: Path
) -> tuple[SourceFileSnapshot, ...]:
    root = source_root.resolve()
    snapshots = []
    logical_paths = sorted({item.logical_source_path for item in index.items})
    for logical_source_path in logical_paths:
        parts = PurePosixPath(logical_source_path).parts
        if not parts or parts[0] != "inputs":
            raise DatasetError(
                f"source path must be below inputs/: {logical_source_path}"
            )
        path = root.joinpath(*parts[1:])
        try:
            stat = path.stat()
        except OSError as exc:
            raise DatasetError(f"could not stat source file {path}: {exc}") from exc
        if not path.is_file():
            raise DatasetError(f"source file is missing: {path}")
        snapshots.append(
            SourceFileSnapshot(
                logical_source_path=logical_source_path,
                size_bytes=stat.st_size,
                device=stat.st_dev,
                inode=stat.st_ino,
                modified_ns=stat.st_mtime_ns,
                changed_ns=stat.st_ctime_ns,
            )
        )
    return tuple(snapshots)


def _require_receipt(
    index_path: str,
    *,
    manifest: CompetitionManifest,
    index: EvaluationIndex,
    allowed_stages: tuple[str, ...],
    required_stage: str,
) -> DatasetPipelineReceipt:
    receipt = _load_receipt(index_path)
    if receipt.stage not in allowed_stages:
        raise DatasetError(
            f"pipeline is at stage {receipt.stage!r}; required prior stage is "
            f"{required_stage!r}"
        )
    if receipt.manifest_digest != manifest.digest():
        raise DatasetError("pipeline receipt does not match the manifest; rerun prepare")
    if receipt.index_digest != index.digest():
        raise DatasetError("pipeline receipt does not match the index; rerun prepare")
    return receipt


def _require_unchanged_sources(
    receipt: DatasetPipelineReceipt,
    index: EvaluationIndex,
    source_root: Path,
) -> tuple[SourceFileSnapshot, ...]:
    resolved_root = str(source_root.resolve())
    if receipt.source_root != resolved_root:
        raise DatasetError(
            "source directory does not match the prepared dataset: "
            f"prepared={receipt.source_root} current={resolved_root}"
        )
    snapshots = _source_snapshots(index, source_root)
    if snapshots != receipt.sources:
        raise DatasetError(
            "one or more source files changed after preparation; rerun prepare"
        )
    return snapshots


def _advance_receipt(
    receipt: DatasetPipelineReceipt,
    stage: Literal["validated", "uploaded", "sealed"],
    *,
    environment: str | None = None,
    volume_name: str | None = None,
) -> DatasetPipelineReceipt:
    return receipt.model_copy(
        update={
            "stage": stage,
            "environment": environment,
            "volume_name": volume_name,
            "updated_at": datetime.now(timezone.utc),
        }
    )


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
    assume_yes: bool,
    index_path: str,
) -> EvaluationIndex:
    if issues:
        print(format_validation_issues(issues), file=sys.stderr)
        count = sum(len(issue.evaluation_ids) for issue in issues)
        question = (
            f"Disqualify {count} evaluation entry/entries from {len(issues)} "
            "source(s) and continue prepare? Type yes or no: "
        )
        if not _prompt_yes_no(question, assume_yes=assume_yes):
            raise DatasetError(
                "operator declined to continue prepare with flagged entries"
            )
    if index is None:
        raise DatasetError("all evaluation sources are invalid")
    if issues:
        print(
            f"Disqualified {count} evaluation entry/entries; "
            f"{len(index.items)} remain.",
            file=sys.stderr,
        )
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
    _add_yes_argument(commands["prepare"])

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
            assume_yes=args.yes,
            index_path=args.index,
        )
        receipt = DatasetPipelineReceipt(
            stage="prepared",
            manifest_digest=manifest.digest(),
            index_digest=evaluation_index.digest(),
            source_root=str(Path(args.source_dir).resolve()),
            sources=_source_snapshots(
                evaluation_index,
                Path(args.source_dir),
            ),
            updated_at=datetime.now(timezone.utc),
        )
        _write_receipt(args.index, receipt)
        print(
            f"Prepared {len(evaluation_index.items)} item(s); "
            f"index checksum={evaluation_index.digest()}; "
            f"pipeline receipt={_receipt_path(args.index)}"
        )
        return 0

    evaluation_index = _index(args.index)
    if args.command == "validate":
        receipt = _require_receipt(
            args.index,
            manifest=manifest,
            index=evaluation_index,
            allowed_stages=("prepared", "validated"),
            required_stage="prepared",
        )
        _require_unchanged_sources(
            receipt,
            evaluation_index,
            Path(args.source_dir),
        )
        evaluation_index.validate_for_manifest(manifest)
        _write_receipt(
            args.index,
            _advance_receipt(receipt, "validated"),
        )
        print(
            f"Validated {len(evaluation_index.items)} item(s); "
            f"index checksum={evaluation_index.digest()}"
        )
        return 0

    if args.command == "upload":
        receipt = _require_receipt(
            args.index,
            manifest=manifest,
            index=evaluation_index,
            allowed_stages=("validated", "uploaded"),
            required_stage="validated",
        )
        _require_unchanged_sources(
            receipt,
            evaluation_index,
            Path(args.source_dir),
        )
        if receipt.stage == "uploaded" and (
            receipt.environment != args.environment
            or receipt.volume_name != manifest.evaluation_input_volume_name
        ):
            raise DatasetError(
                "dataset was already uploaded to a different environment or Volume"
            )
        store = ModalVolumeStore(environment_name=args.environment)
        store.upload(
            manifest,
            evaluation_index,
            Path(args.source_dir),
            validate_local=False,
        )
        _write_receipt(
            args.index,
            _advance_receipt(
                receipt,
                "uploaded",
                environment=args.environment,
                volume_name=manifest.evaluation_input_volume_name,
            ),
        )
        print(
            f"Uploaded and read-back verified {len(evaluation_index.items)} item(s) "
            f"to {manifest.evaluation_input_volume_name}"
        )
        return 0

    receipt = _require_receipt(
        args.index,
        manifest=manifest,
        index=evaluation_index,
        allowed_stages=("uploaded", "sealed"),
        required_stage="uploaded",
    )
    if (
        receipt.environment != args.environment
        or receipt.volume_name != manifest.evaluation_input_volume_name
    ):
        raise DatasetError(
            "upload receipt does not match the requested environment and Volume"
        )
    evaluation_index.validate_for_manifest(manifest)
    store = ModalVolumeStore(environment_name=args.environment)
    remote_index = store.load_index(manifest, validate_for_manifest=False)
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
    _write_receipt(
        args.index,
        _advance_receipt(
            receipt,
            "sealed",
            environment=args.environment,
            volume_name=manifest.evaluation_input_volume_name,
        ),
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
