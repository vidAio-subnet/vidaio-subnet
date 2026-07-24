"""Private S3 backup of final contender repository snapshots."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from botocore.config import Config as BotoConfig
from loguru import logger
from sqlalchemy.engine import make_url

from .config import CompetitionManifest
from .intake import (
    boss_contender_id,
    make_repository_read_only,
    pinned_repository_source,
)
from .phase0 import SecretRedactor
from .state import CompetitionState
from .validation import RepositoryStaticValidator, write_validation_report


LOG_REDACTOR = SecretRedactor()
Clock = Callable[[], datetime]


class CompetitionArtifactBackupError(RuntimeError):
    """Raised when final submission snapshots cannot be privately backed up."""


class CompetitionDatabaseBackupError(RuntimeError):
    """Raised when a completed competition database cannot be backed up."""


@dataclass(frozen=True)
class CompetitionArtifactBackupResult:
    bucket: str
    prefix: str
    archive_key: str
    inventory_key: str
    archive_sha256: str
    archive_size_bytes: int
    contender_count: int

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}/"


@dataclass(frozen=True)
class CompetitionDatabaseBackupResult:
    bucket: str
    key: str
    sha256: str
    size_bytes: int

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


class CompetitionArtifactBackupService:
    """Archive final submission artifacts and upload them without public URLs."""

    def __init__(
        self,
        repository: Any,
        *,
        artifact_root: Path,
        bucket: str,
        key_prefix: str = "competition_artifacts",
        s3_client: Any | None = None,
        region_name: str = "us-east-1",
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        database_url: str,
        repository_root: Path | None = None,
        actor: str = "validator:competition-artifact-backup",
        clock: Clock | None = None,
    ) -> None:
        self.repository = repository
        self.artifact_root = artifact_root
        self.bucket = bucket.strip()
        self.key_prefix = key_prefix.strip("/")
        self.actor = actor
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.repository_root = (repository_root or Path.cwd()).resolve()
        self.boss_validator = RepositoryStaticValidator()
        if not self.bucket:
            raise ValueError("competition artifact backup bucket is required")
        if not self.key_prefix or ".." in self.key_prefix.split("/"):
            raise ValueError("competition artifact backup prefix is invalid")
        parsed_database_url = make_url(database_url)
        if parsed_database_url.get_backend_name() != "sqlite":
            raise ValueError("competition database backup requires SQLite")
        if (
            not parsed_database_url.database
            or parsed_database_url.database == ":memory:"
        ):
            raise ValueError("competition database backup requires a SQLite file")
        self.database_path = Path(parsed_database_url.database).expanduser()
        if not self.database_path.is_absolute():
            self.database_path = (Path.cwd() / self.database_path).resolve()
        self.s3 = s3_client or self._create_client(
            region_name=region_name,
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )

    @staticmethod
    def _create_client(
        *,
        region_name: str,
        endpoint_url: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
    ) -> Any:
        import boto3

        normalized_endpoint = (endpoint_url or "").strip() or None
        if normalized_endpoint and not normalized_endpoint.startswith(
            ("http://", "https://")
        ):
            normalized_endpoint = f"https://{normalized_endpoint}"
        kwargs: dict[str, Any] = {
            "region_name": region_name,
            "config": BotoConfig(signature_version="s3v4"),
        }
        if normalized_endpoint:
            kwargs["endpoint_url"] = normalized_endpoint
        if access_key_id:
            kwargs["aws_access_key_id"] = access_key_id
        if secret_access_key:
            kwargs["aws_secret_access_key"] = secret_access_key
        return boto3.client("s3", **kwargs)

    def backup(self, competition_id: str) -> CompetitionArtifactBackupResult:
        competition = self.repository.get(competition_id)
        if competition is None:
            raise KeyError(competition_id)
        if competition.status != CompetitionState.FINALIZING_SUBMISSIONS.value:
            raise CompetitionArtifactBackupError(
                "submission artifacts can only be backed up during finalisation"
            )
        if competition.submission_backup_status == "COMPLETED":
            return CompetitionArtifactBackupResult(
                bucket=competition.submission_backup_bucket,
                prefix=competition.submission_backup_prefix,
                archive_key=competition.submission_backup_archive_key,
                inventory_key=competition.submission_backup_inventory_key,
                archive_sha256=competition.submission_backup_checksum,
                archive_size_bytes=competition.submission_backup_size_bytes,
                contender_count=competition.submission_backup_contender_count,
            )

        digest_label = competition.manifest_digest[:16]
        prefix = (
            f"{self.key_prefix}/{competition_id}/submission-snapshots/{digest_label}"
        )
        archive_name = f"competition_artifacts-{competition_id}.tar.gz"
        archive_key = f"{prefix}/{archive_name}"
        inventory_key = f"{prefix}/inventory.json"

        try:
            self._ensure_boss_snapshot(competition)
            contenders = [
                contender
                for contender in self.repository.list_contenders(competition_id)
                if contender.pinned_commit_sha
            ]
            self._verify_bucket_private()
            with tempfile.TemporaryDirectory(
                prefix=f"competition-backup-{competition_id}-"
            ) as temp:
                archive = Path(temp) / archive_name
                self._create_archive(competition_id, contenders, archive)
                archive_size = archive.stat().st_size
                archive_checksum = _sha256_file(archive)
                inventory = self._inventory(
                    competition,
                    contenders,
                    archive_key=archive_key,
                    archive_size=archive_size,
                    archive_checksum=archive_checksum,
                    created_at=self.clock(),
                )
                self.s3.upload_file(
                    str(archive),
                    self.bucket,
                    archive_key,
                    ExtraArgs={
                        "ContentType": "application/gzip",
                        "Metadata": {"sha256": archive_checksum},
                    },
                )
                self._verify_object(
                    archive_key,
                    archive_size,
                    expected_checksum=archive_checksum,
                )
                inventory_body = (
                    json.dumps(inventory, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8")
                inventory_checksum = hashlib.sha256(inventory_body).hexdigest()
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=inventory_key,
                    Body=inventory_body,
                    ContentType="application/json",
                    Metadata={"sha256": inventory_checksum},
                )
                self._verify_object(
                    inventory_key,
                    len(inventory_body),
                    expected_checksum=inventory_checksum,
                )
        except Exception as exc:
            detail = LOG_REDACTOR.redact_text(str(exc))[:500]
            self.repository.record_submission_backup_failure(
                competition_id,
                bucket=self.bucket,
                prefix=prefix,
                detail=detail,
                now=self.clock(),
                actor=self.actor,
            )
            raise CompetitionArtifactBackupError(detail) from exc

        result = CompetitionArtifactBackupResult(
            bucket=self.bucket,
            prefix=prefix,
            archive_key=archive_key,
            inventory_key=inventory_key,
            archive_sha256=archive_checksum,
            archive_size_bytes=archive_size,
            contender_count=len(contenders),
        )
        self.repository.record_submission_backup_success(
            competition_id,
            bucket=result.bucket,
            prefix=result.prefix,
            archive_key=result.archive_key,
            inventory_key=result.inventory_key,
            checksum=result.archive_sha256,
            size_bytes=result.archive_size_bytes,
            contender_count=result.contender_count,
            now=self.clock(),
            actor=self.actor,
        )
        logger.info(
            "Competition submission snapshots backed up privately: "
            "competition_id={} contenders={} path={}",
            competition_id,
            result.contender_count,
            result.s3_uri,
        )
        return result

    def _ensure_boss_snapshot(self, competition: Any) -> None:
        manifest = CompetitionManifest.model_validate_json(competition.manifest_json)
        if manifest.boss.repository_path is None:
            return

        payout_hotkey = manifest.boss.boss_hotkey
        assert payout_hotkey is not None
        contender_id = boss_contender_id(payout_hotkey)
        destination = pinned_repository_source(
            self.artifact_root, manifest.competition_id, contender_id
        ).parent
        existing = self.repository.get_contender(manifest.competition_id, contender_id)
        if existing is not None:
            if not existing.is_boss:
                raise CompetitionArtifactBackupError(
                    f"boss contender identity is already in use: {contender_id}"
                )
            if not (destination / "source").is_dir():
                raise CompetitionArtifactBackupError(
                    f"boss contender snapshot is missing: {contender_id}"
                )
            return

        source = (self.repository_root / manifest.boss.repository_path).resolve(
            strict=True
        )
        if source == self.repository_root or not source.is_relative_to(
            self.repository_root
        ):
            raise CompetitionArtifactBackupError(
                "boss repository_path resolves outside the validator repository"
            )
        if not source.is_dir():
            raise CompetitionArtifactBackupError(
                f"boss repository_path is not a directory: {source}"
            )

        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(
            tempfile.mkdtemp(prefix=f".{contender_id}-", dir=destination.parent)
        )
        try:
            staged_source = staged / "source"
            shutil.copytree(source, staged_source, symlinks=True)
            validation = self.boss_validator.validate(staged_source)
            tree_sha = validation.repository_tree_sha256
            snapshot_time = self.clock()
            repository_display = f"manifest:{manifest.boss.repository_path.as_posix()}"
            repository_url_hash = hashlib.sha256(
                repository_display.encode("utf-8")
            ).hexdigest()
            (staged / "submission.json").write_text(
                json.dumps(
                    {
                        "competition_id": manifest.competition_id,
                        "contender_hotkey": contender_id,
                        "payout_hotkey": payout_hotkey,
                        "is_boss": True,
                        "repository_display": repository_display,
                        "repository_url_hash": repository_url_hash,
                        "pinned_commit_sha": tree_sha,
                        "pinned_tree_sha": tree_sha,
                        "latest_commit_time": snapshot_time.isoformat(),
                        "cloned_at": snapshot_time.isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            write_validation_report(validation, staged / "validation_report.json")
            os.replace(staged, destination)

            submitted_identity = self.repository.get_contender(
                manifest.competition_id, payout_hotkey
            )
            try:
                self.repository.record_pinned_contender(
                    competition_id=manifest.competition_id,
                    hotkey=contender_id,
                    repository_url_hash=repository_url_hash,
                    repository_display=repository_display,
                    pinned_commit_sha=tree_sha,
                    pinned_tree_sha=tree_sha,
                    latest_commit_time=snapshot_time.isoformat(),
                    validation=validation,
                    now=snapshot_time,
                    actor=self.actor,
                    uid_snapshot=getattr(submitted_identity, "uid_snapshot", None),
                    coldkey_snapshot=getattr(
                        submitted_identity, "coldkey_snapshot", None
                    ),
                    is_boss=True,
                )
            except Exception:
                shutil.rmtree(destination, ignore_errors=True)
                raise
            make_repository_read_only(destination / "source")
        except Exception:
            shutil.rmtree(staged, ignore_errors=True)
            raise

    def backup_database(
        self, competition_id: str
    ) -> CompetitionDatabaseBackupResult:
        """Upload a consistent SQLite snapshot after competition completion."""

        competition = self.repository.get(competition_id)
        if competition is None:
            raise KeyError(competition_id)
        if competition.status != CompetitionState.COMPLETED.value:
            raise CompetitionDatabaseBackupError(
                "database backup requires a completed competition"
            )
        existing = self.repository.database_backup_record(competition_id)
        if existing is not None:
            return CompetitionDatabaseBackupResult(
                bucket=existing["bucket"],
                key=existing["key"],
                sha256=existing["sha256"],
                size_bytes=existing["size_bytes"],
            )

        prefix = (
            f"{self.key_prefix}/{competition_id}/final/"
            f"{competition.manifest_digest[:16]}"
        )
        key = f"{prefix}/{self.database_path.name}"
        try:
            self._verify_bucket_private()
            with tempfile.TemporaryDirectory(
                prefix=f"competition-database-backup-{competition_id}-"
            ) as temp:
                snapshot = Path(temp) / self.database_path.name
                self._create_database_snapshot(snapshot)
                size_bytes = snapshot.stat().st_size
                checksum = _sha256_file(snapshot)
                self.s3.upload_file(
                    str(snapshot),
                    self.bucket,
                    key,
                    ExtraArgs={
                        "ContentType": "application/vnd.sqlite3",
                        "Metadata": {"sha256": checksum},
                    },
                )
                self._verify_object(
                    key,
                    size_bytes,
                    expected_checksum=checksum,
                )
        except Exception as exc:
            detail = LOG_REDACTOR.redact_text(str(exc))[:500]
            raise CompetitionDatabaseBackupError(detail) from exc

        result = CompetitionDatabaseBackupResult(
            bucket=self.bucket,
            key=key,
            sha256=checksum,
            size_bytes=size_bytes,
        )
        self.repository.record_database_backup_success(
            competition_id,
            bucket=result.bucket,
            key=result.key,
            checksum=result.sha256,
            size_bytes=result.size_bytes,
            now=self.clock(),
            actor=self.actor,
        )
        logger.info(
            "Completed competition SQLite database backed up privately: "
            "competition_id={} path={}",
            competition_id,
            result.s3_uri,
        )
        return result

    def _create_database_snapshot(self, destination: Path) -> None:
        if not self.database_path.is_file():
            raise CompetitionDatabaseBackupError(
                f"competition SQLite database is missing: {self.database_path}"
            )
        source_uri = f"{self.database_path.as_uri()}?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source:
            with sqlite3.connect(destination) as snapshot:
                source.backup(snapshot)
                integrity = snapshot.execute("PRAGMA integrity_check").fetchone()
                if integrity != ("ok",):
                    raise CompetitionDatabaseBackupError(
                        f"SQLite snapshot integrity check failed: {integrity!r}"
                    )

    def _create_archive(
        self, competition_id: str, contenders: list[Any], destination: Path
    ) -> None:
        manifest_root = self.artifact_root / competition_id
        if not manifest_root.is_dir():
            raise CompetitionArtifactBackupError(
                f"competition artifact directory is missing: {manifest_root}"
            )
        archive_root = Path("competition_artifacts") / competition_id
        with destination.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
                with tarfile.open(
                    fileobj=zipped, mode="w", dereference=False
                ) as archive:
                    archive.add(manifest_root, arcname=archive_root, recursive=True)
                    for contender in contenders:
                        if (
                            not contender.hotkey
                            or contender.hotkey in {".", ".."}
                            or "/" in contender.hotkey
                            or "\\" in contender.hotkey
                        ):
                            raise CompetitionArtifactBackupError(
                                f"invalid contender archive identity: {contender.hotkey}"
                            )
                        contender_root = pinned_repository_source(
                            self.artifact_root, competition_id, contender.hotkey
                        ).parent
                        if not contender_root.is_dir():
                            raise CompetitionArtifactBackupError(
                                "pinned contender artifact directory is missing: "
                                f"{contender.hotkey}"
                            )
                        archive.add(
                            contender_root,
                            arcname=(archive_root / "contenders" / contender.hotkey),
                            recursive=True,
                        )

    def _verify_bucket_private(self) -> None:
        acl = self.s3.get_bucket_acl(Bucket=self.bucket)
        self._reject_public_acl_grants(acl, f"bucket {self.bucket}")

    def _verify_object(
        self, key: str, expected_size: int, *, expected_checksum: str
    ) -> None:
        metadata = self.s3.head_object(Bucket=self.bucket, Key=key)
        observed_size = int(metadata.get("ContentLength", -1))
        if observed_size != expected_size:
            raise CompetitionArtifactBackupError(
                f"S3 object size mismatch for {key}: "
                f"expected {expected_size}, observed {observed_size}"
            )
        observed_checksum = metadata.get("Metadata", {}).get("sha256")
        if observed_checksum != expected_checksum:
            raise CompetitionArtifactBackupError(
                f"S3 object checksum metadata mismatch for {key}"
            )
        acl = self.s3.get_object_acl(Bucket=self.bucket, Key=key)
        self._reject_public_acl_grants(acl, f"object {key}")

    @staticmethod
    def _reject_public_acl_grants(acl: dict[str, Any], target: str) -> None:
        public_groups = {
            "http://acs.amazonaws.com/groups/global/AllUsers",
            "http://acs.amazonaws.com/groups/global/AuthenticatedUsers",
        }
        for grant in acl.get("Grants", []):
            grantee = grant.get("Grantee", {})
            if grantee.get("URI") in public_groups:
                raise CompetitionArtifactBackupError(
                    f"S3 {target} has a public ACL grant"
                )

    @staticmethod
    def _inventory(
        competition: Any,
        contenders: list[Any],
        *,
        archive_key: str,
        archive_size: int,
        archive_checksum: str,
        created_at: datetime,
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "competition_id": competition.competition_id,
            "manifest_digest": competition.manifest_digest,
            "created_at": created_at.astimezone(timezone.utc).isoformat(),
            "visibility": "PRIVATE",
            "archive": {
                "key": archive_key,
                "size_bytes": archive_size,
                "sha256": archive_checksum,
            },
            "contenders": [
                {
                    "hotkey": contender.hotkey,
                    "payout_hotkey": (
                        competition.boss_hotkey
                        if contender.is_boss
                        else contender.hotkey
                    ),
                    "is_boss": bool(contender.is_boss),
                    "solution_type": "boss" if contender.is_boss else "submission",
                    "submission_revision": contender.submission_revision,
                    "pinned_commit_sha": contender.pinned_commit_sha,
                    "pinned_tree_sha": contender.pinned_tree_sha,
                    "validation_status": contender.validation_status,
                    "reason_code": contender.reason_code,
                }
                for contender in contenders
            ],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CompetitionArtifactBackupError",
    "CompetitionArtifactBackupResult",
    "CompetitionArtifactBackupService",
    "CompetitionDatabaseBackupError",
    "CompetitionDatabaseBackupResult",
]
