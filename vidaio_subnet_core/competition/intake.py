"""Credential-safe repository polling, cloning, pinning, and validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from .phase0 import SecretRedactor
from .validation import (
    RepositoryStaticValidator,
    ValidationReport,
    write_validation_report,
)

if TYPE_CHECKING:
    from .repository import CompetitionRepository


GITHUB_REPOSITORY_PATTERN = re.compile(
    r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\.git$"
)
BOSS_CONTENDER_SUFFIX = "-boss"


def boss_contender_id(hotkey: str) -> str:
    """Return the internal contender identity for the manifest boss snapshot."""

    return f"{hotkey}{BOSS_CONTENDER_SUFFIX}"


class IntakeError(RuntimeError):
    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


@dataclass
class RepositorySubmission:
    competition_id: str
    contender_hotkey: str
    repository_url: str
    github_pat: str = field(repr=False)
    nonce: str = ""


@dataclass(frozen=True)
class PinnedRepository:
    competition_id: str
    contender_hotkey: str
    source_path: Path
    commit_sha: str
    tree_sha: str
    committer_time: str
    repository_url_hash: str
    repository_display: str
    cloned_at: str
    validation: ValidationReport
    previous_artifact_path: Path | None = field(
        default=None, repr=False, compare=False
    )


class GitTransport(Protocol):
    def clone_and_pin(
        self,
        repository_url: str,
        github_pat: str,
        destination: Path,
    ) -> tuple[str, str, str]: ...


class SecureGitTransport:
    """Clone once using askpass; never place credentials in argv or Git config."""

    def __init__(
        self, *, max_transfer_bytes: int = 2_000_000_000, timeout_seconds: int = 600
    ) -> None:
        self.max_transfer_bytes = max_transfer_bytes
        self.timeout_seconds = timeout_seconds

    def clone_and_pin(
        self, repository_url: str, github_pat: str, destination: Path
    ) -> tuple[str, str, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="competition-askpass-", dir=destination.parent
        ) as temp:
            askpass = Path(temp) / "askpass.sh"
            askpass.write_text(
                '#!/bin/sh\ncase "$1" in *Username*) printf \'%s\' "$VIDAIO_GIT_USERNAME" ;; *) printf \'%s\' "$VIDAIO_GIT_PASSWORD" ;; esac\n',
                encoding="utf-8",
            )
            askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            env = {
                **os.environ,
                "GIT_ASKPASS": str(askpass),
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_SYSTEM": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "VIDAIO_GIT_USERNAME": "x-access-token",
                "VIDAIO_GIT_PASSWORD": github_pat,
            }
            command = [
                "git",
                "-c",
                "credential.helper=",
                "-c",
                "core.hooksPath=/dev/null",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--no-recurse-submodules",
                repository_url,
                str(destination),
            ]
            self._run_bounded(command, destination, env, github_pat)

        self._git(destination, "remote", "remove", "origin", secret=github_pat)
        self._git(
            destination, "config", "core.hooksPath", "/dev/null", secret=github_pat
        )
        commit = self._git(destination, "rev-parse", "HEAD", secret=github_pat).strip()
        tree = self._git(
            destination, "rev-parse", "HEAD^{tree}", secret=github_pat
        ).strip()
        committer_time = self._git(
            destination, "show", "-s", "--format=%cI", "HEAD", secret=github_pat
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit) or not re.fullmatch(
            r"[0-9a-f]{40,64}", tree
        ):
            raise IntakeError(
                "GIT_PIN_INVALID", "Git returned an invalid commit or tree SHA"
            )
        return commit, tree, committer_time

    def _run_bounded(
        self, command: list[str], size_root: Path, env: dict[str, str], secret: str
    ) -> None:
        started = time.monotonic()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        while process.poll() is None:
            if time.monotonic() - started > self.timeout_seconds:
                process.kill()
                process.communicate()
                raise IntakeError(
                    "GIT_CLONE_TIMEOUT", "repository clone exceeded its time limit"
                )
            if _directory_size(size_root) > self.max_transfer_bytes:
                process.kill()
                process.communicate()
                raise IntakeError(
                    "GIT_TRANSFER_TOO_LARGE",
                    "repository clone exceeded its transfer limit",
                )
            time.sleep(0.05)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            detail = SecretRedactor([secret]).redact_text((stderr or stdout).strip())
            raise IntakeError("GIT_CLONE_FAILED", detail[:500] or "git clone failed")

    @staticmethod
    def _git(repository: Path, *args: str, secret: str) -> str:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "credential.helper=",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        if result.returncode != 0:
            detail = SecretRedactor([secret]).redact_text(
                (result.stderr or result.stdout).strip()
            )
            raise IntakeError("GIT_PIN_FAILED", detail[:500] or "Git pinning failed")
        return result.stdout


class RepositoryIntake:
    def __init__(
        self,
        artifact_root: Path,
        *,
        transport: GitTransport | None = None,
        validator: RepositoryStaticValidator | None = None,
    ) -> None:
        self.artifact_root = artifact_root
        self.transport = transport or SecureGitTransport()
        self.validator = validator or RepositoryStaticValidator()

    def accept(self, submission: RepositorySubmission) -> PinnedRepository:
        try:
            return self._accept(submission, replace_existing=False)
        finally:
            submission.github_pat = ""

    def replace(self, submission: RepositorySubmission) -> PinnedRepository:
        """Stage and atomically install a replacement for a collected submission."""

        try:
            return self._accept(submission, replace_existing=True)
        finally:
            submission.github_pat = ""

    def commit(self, pinned: PinnedRepository) -> None:
        """Discard the previous artifact after SQLite accepts the new revision."""

        if pinned.previous_artifact_path is not None:
            _remove_tree(pinned.previous_artifact_path)

    def rollback(self, pinned: PinnedRepository) -> None:
        """Restore the prior artifact when SQLite rejects the replacement."""

        contender_root = pinned.source_path.parent
        _remove_tree(contender_root)
        if pinned.previous_artifact_path is not None:
            os.replace(pinned.previous_artifact_path, contender_root)

    def _accept(
        self, submission: RepositorySubmission, *, replace_existing: bool
    ) -> PinnedRepository:
        match = GITHUB_REPOSITORY_PATTERN.fullmatch(submission.repository_url)
        if match is None:
            raise IntakeError(
                "REPOSITORY_URL_INVALID",
                "only canonical HTTPS GitHub .git URLs are accepted",
            )
        if not submission.github_pat:
            raise IntakeError("GITHUB_PAT_MISSING", "a raw GitHub PAT is required")
        competition_slug = _safe_slug(submission.competition_id)
        hotkey_slug = _safe_slug(submission.contender_hotkey)
        contender_root = (
            self.artifact_root / competition_slug / "contenders" / hotkey_slug
        )
        if contender_root.exists() and not replace_existing:
            raise IntakeError(
                "SUBMISSION_ALREADY_PINNED",
                "a finalized submission is immutable and cannot be cloned again",
            )
        contender_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{hotkey_slug}-", dir=contender_root.parent)
        )
        source = temporary / "source"
        previous_artifact: Path | None = None
        try:
            commit, tree, committer_time = self.transport.clone_and_pin(
                submission.repository_url, submission.github_pat, source
            )
            report = self.validator.validate(source)
            owner, repository = match.groups()
            cloned_at = datetime.now(timezone.utc).isoformat()
            safe_submission = {
                "competition_id": submission.competition_id,
                "contender_hotkey": submission.contender_hotkey,
                "nonce": submission.nonce,
                "repository_display": f"github.com/{owner}/{repository}",
                "repository_url_hash": hashlib.sha256(
                    submission.repository_url.encode()
                ).hexdigest(),
                "pinned_commit_sha": commit,
                "pinned_tree_sha": tree,
                "latest_commit_time": committer_time,
                "cloned_at": cloned_at,
            }
            (temporary / "submission.json").write_text(
                json.dumps(safe_submission, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_validation_report(report, temporary / "validation_report.json")
            make_repository_read_only(source)
            if contender_root.exists():
                previous_artifact = temporary.with_name(
                    f"{temporary.name}-previous"
                )
                os.replace(contender_root, previous_artifact)
            try:
                os.replace(temporary, contender_root)
            except Exception:
                if previous_artifact is not None:
                    os.replace(previous_artifact, contender_root)
                    previous_artifact = None
                raise
            return PinnedRepository(
                submission.competition_id,
                submission.contender_hotkey,
                contender_root / "source",
                commit,
                tree,
                committer_time,
                safe_submission["repository_url_hash"],
                safe_submission["repository_display"],
                cloned_at,
                report,
                previous_artifact,
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


class CompetitionSubmissionIntakeService:
    """Finalize one authenticated poll response into artifacts and SQLite."""

    def __init__(
        self,
        intake: RepositoryIntake,
        repository: "CompetitionRepository",
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.intake = intake
        self.repository = repository
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def finalize(
        self,
        submission: RepositorySubmission,
        *,
        expected_competition_id: str,
        expected_hotkey: str,
        expected_nonce: str,
        actor: str,
        uid_snapshot: int | None = None,
        coldkey_snapshot: str | None = None,
    ) -> PinnedRepository:
        try:
            validate_polled_submission(
                submission,
                expected_competition_id=expected_competition_id,
                expected_hotkey=expected_hotkey,
                expected_nonce=expected_nonce,
            )
        except Exception:
            submission.github_pat = ""
            raise
        pinned = self.intake.replace(submission)
        try:
            self.repository.record_pinned_contender(
                competition_id=pinned.competition_id,
                hotkey=pinned.contender_hotkey,
                repository_url_hash=pinned.repository_url_hash,
                repository_display=pinned.repository_display,
                pinned_commit_sha=pinned.commit_sha,
                pinned_tree_sha=pinned.tree_sha,
                latest_commit_time=pinned.committer_time,
                validation=pinned.validation,
                now=self.clock(),
                actor=actor,
                uid_snapshot=uid_snapshot,
                coldkey_snapshot=coldkey_snapshot,
            )
        except Exception:
            self.intake.rollback(pinned)
            raise
        else:
            self.intake.commit(pinned)
        return pinned


def validate_polled_submission(
    submission: RepositorySubmission,
    *,
    expected_competition_id: str,
    expected_hotkey: str,
    expected_nonce: str,
) -> None:
    if submission.competition_id != expected_competition_id:
        raise IntakeError(
            "COMPETITION_ID_MISMATCH", "submission belongs to another competition"
        )
    if submission.contender_hotkey != expected_hotkey:
        raise IntakeError(
            "HOTKEY_MISMATCH", "response hotkey does not match the queried axon"
        )
    if submission.nonce != expected_nonce:
        raise IntakeError(
            "SUBMISSION_REPLAY", "submission nonce is stale or unexpected"
        )


def _safe_slug(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")[:48] or "contender"
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:12]}"


def pinned_repository_source(
    artifact_root: Path, competition_id: str, contender_hotkey: str
) -> Path:
    """Return the deterministic source path created by repository intake."""

    return (
        artifact_root
        / _safe_slug(competition_id)
        / "contenders"
        / _safe_slug(contender_hotkey)
        / "source"
    )


def _directory_size(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                total += (Path(directory) / name).lstat().st_size
            except FileNotFoundError:
                pass
    return total


def make_repository_read_only(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        for name in names:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(
                    stat.S_IRUSR
                    | stat.S_IXUSR
                    | stat.S_IRGRP
                    | stat.S_IXGRP
                    | stat.S_IROTH
                    | stat.S_IXOTH
                )
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        for name in names:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        Path(directory).chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    shutil.rmtree(root)


__all__ = [
    "GITHUB_REPOSITORY_PATTERN",
    "CompetitionSubmissionIntakeService",
    "GitTransport",
    "IntakeError",
    "PinnedRepository",
    "RepositoryIntake",
    "RepositorySubmission",
    "SecureGitTransport",
    "pinned_repository_source",
    "validate_polled_submission",
]
