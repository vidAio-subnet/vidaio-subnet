"""Validator-owned image build boundaries for competition contenders.

The validator never imports contender Python.  The live Modal path binds every
immutable image to the pinned source tree and explicitly records that Modal did
not provide final-image size attestation.  Deployments with a quota-attesting
builder can use :class:`TrustedImageBuilder` for the stricter evidence contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .phase0 import IMAGE_SIZE_LIMIT_BYTES, enforce_image_size


SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_TREE_SHA = re.compile(r"^[0-9a-f]{40,64}$")
MODAL_BUILD_TIMEOUT_SECONDS = 10 * 60


def _dev_mode_enabled() -> bool:
    return os.getenv("DEV_MODE", "False").strip().lower() == "true"


class BuildReason(str, Enum):
    ACCEPTED = "ACCEPTED"
    TRUSTED_BUILDER_UNAVAILABLE = "TRUSTED_BUILDER_UNAVAILABLE"
    UNTRUSTED_BUILDER = "UNTRUSTED_BUILDER"
    SOURCE_EVIDENCE_MISMATCH = "SOURCE_EVIDENCE_MISMATCH"
    IMAGE_ID_MISSING = "IMAGE_ID_MISSING"
    IMAGE_DIGEST_INVALID = "IMAGE_DIGEST_INVALID"
    IMAGE_SIZE_EVIDENCE_MISSING = "IMAGE_SIZE_EVIDENCE_MISSING"
    IMAGE_SIZE_LIMIT_EXCEEDED = "IMAGE_SIZE_LIMIT_EXCEEDED"
    BUILD_QUOTA_NOT_ENFORCED = "BUILD_QUOTA_NOT_ENFORCED"
    BUILD_QUOTA_MISMATCH = "BUILD_QUOTA_MISMATCH"
    BUILD_INPUT_INVALID = "BUILD_INPUT_INVALID"
    BUILD_TIMEOUT = "BUILD_TIMEOUT"
    BUILD_INFRASTRUCTURE_ERROR = "BUILD_INFRASTRUCTURE_ERROR"


class TrustedBuildError(RuntimeError):
    def __init__(self, reason_code: BuildReason, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code.value}: {detail}")


@dataclass(frozen=True)
class BuildRequest:
    competition_id: str
    hotkey: str
    source_root: Path
    pinned_tree_sha: str
    size_limit_bytes: int = IMAGE_SIZE_LIMIT_BYTES
    modal_build_timeout_seconds: float = MODAL_BUILD_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.source_root.is_dir():
            raise ValueError("source_root must be a pinned contender directory")
        if not GIT_TREE_SHA.fullmatch(self.pinned_tree_sha):
            raise ValueError("pinned_tree_sha must be a hexadecimal Git tree SHA")
        if self.size_limit_bytes != IMAGE_SIZE_LIMIT_BYTES:
            raise ValueError("competition images use the fixed 25 GB limit")
        if self.modal_build_timeout_seconds <= 0:
            raise ValueError("modal_build_timeout_seconds must be positive")


@dataclass(frozen=True)
class BuildEvidence:
    builder_id: str
    source_tree_sha: str
    image_id: str
    image_digest: str
    image_size_bytes: int
    quota_limit_bytes: int
    quota_enforced_during_build: bool
    built_at: datetime
    security_level: str = "TRUSTED"
    image_size_measurement: str = "FINAL_IMAGE"

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["built_at"] = self.built_at.astimezone(timezone.utc).isoformat()
        return payload


class TrustedBuildBackend(Protocol):
    @property
    def builder_id(self) -> str: ...

    def build(self, request: BuildRequest) -> BuildEvidence: ...


class ModalImageBuildBackend:
    """Build a pinned contender directly on Modal.

    Modal returns an immutable image ID but does not attest the final image size
    or enforce the competition's hostile-build disk quota. Evidence from this
    backend records that limitation rather than presenting it as trusted size
    evidence. Operators must acknowledge this when enabling competition
    execution in any environment.
    """

    builder_id = "modal-direct-unattested"

    def __init__(
        self,
        *,
        environment_name: str = "dev",
        app_name: str = "vidaio-competition-image-builder",
        modal_api: Any = None,
    ) -> None:
        self._subprocess_builds = modal_api is None
        if modal_api is None:
            try:
                import modal as modal_api
            except ImportError as exc:
                raise RuntimeError(
                    "Modal SDK is required for competition image builds"
                ) from exc
        self.modal = modal_api
        self.environment_name = environment_name
        self.app_name = app_name

    def _build_app_name(self, request: BuildRequest) -> str:
        digest = hashlib.sha256(
            f"{request.competition_id}:{request.hotkey}:{request.pinned_tree_sha}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
        competition = re.sub(r"[^a-z0-9-]+", "-", request.competition_id.lower())
        competition = competition.strip("-")[:24] or "competition"
        return f"{self.app_name[:24]}-{competition}-{digest}"[:63]

    def _stop_timed_out_app(self, app_name: str) -> str | None:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "app",
                    "stop",
                    "--yes",
                    "--env",
                    self.environment_name,
                    app_name,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            return f"remote App stop failed: {type(exc).__name__}: {str(exc)[:500]}"
        if completed.returncode != 0:
            detail = (
                completed.stderr or completed.stdout or "Modal CLI returned failure"
            )
            return f"remote App stop failed: {detail.strip()[:500]}"
        return None

    def _build_in_subprocess(
        self,
        request: BuildRequest,
        dockerfile: Path,
        compression_root: Path,
    ) -> str:
        worker = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "competition_modal_build.py"
        )
        app_name = self._build_app_name(request)
        with tempfile.TemporaryDirectory(prefix="competition-modal-build-") as temp:
            result_path = Path(temp) / "result.json"
            command = [
                sys.executable,
                str(worker),
                "--environment",
                self.environment_name,
                "--app-name",
                app_name,
                "--dockerfile",
                str(dockerfile),
                "--context-dir",
                str(compression_root),
                "--result",
                str(result_path),
            ]
            stream_output = _dev_mode_enabled()
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=None if stream_output else subprocess.PIPE,
                    stderr=None if stream_output else subprocess.PIPE,
                    text=True,
                    timeout=request.modal_build_timeout_seconds,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
            except subprocess.TimeoutExpired as exc:
                stop_failure = self._stop_timed_out_app(app_name)
                detail = (
                    "Modal image build exceeded "
                    f"{request.modal_build_timeout_seconds:g} seconds "
                    f"(app={app_name})"
                )
                if stop_failure:
                    detail = f"{detail}; {stop_failure}"
                raise TrustedBuildError(
                    BuildReason.BUILD_TIMEOUT,
                    detail,
                ) from exc
            if completed.returncode != 0:
                detail = completed.stderr or completed.stdout or "build worker failed"
                raise RuntimeError(detail.strip()[:1000])
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                image_id = str(payload["image_id"])
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Modal build worker returned no image ID") from exc
            if not image_id:
                raise RuntimeError("Modal returned no immutable image ID")
            return image_id

    def build(self, request: BuildRequest) -> BuildEvidence:
        compression_root = request.source_root / "miner" / "compression"
        dockerfile = compression_root / "Dockerfile"
        requirements = compression_root / "requirements.txt"
        service = compression_root / "app.py"
        for required in (dockerfile, requirements, service):
            if not required.is_file():
                raise TrustedBuildError(
                    BuildReason.BUILD_INPUT_INVALID,
                    f"pinned contender is missing {required.relative_to(request.source_root)}",
                )
        try:
            if self._subprocess_builds:
                image_id = self._build_in_subprocess(
                    request, dockerfile, compression_root
                )
            else:
                image_id = self._build_in_process(dockerfile, compression_root)
        except TrustedBuildError:
            raise
        except Exception as exc:
            raise TrustedBuildError(
                BuildReason.BUILD_INFRASTRUCTURE_ERROR, str(exc)[:1000]
            ) from exc

        binding_digest = hashlib.sha256(
            f"{request.pinned_tree_sha}:{image_id}".encode("utf-8")
        ).hexdigest()
        return BuildEvidence(
            builder_id=self.builder_id,
            source_tree_sha=request.pinned_tree_sha,
            image_id=image_id,
            image_digest=f"sha256:{binding_digest}",
            image_size_bytes=0,
            quota_limit_bytes=request.size_limit_bytes,
            quota_enforced_during_build=False,
            built_at=datetime.now(timezone.utc),
            security_level="MODAL_UNATTESTED",
            image_size_measurement="UNAVAILABLE",
        )

    def _build_in_process(self, dockerfile: Path, compression_root: Path) -> str:
        """Test/injected backend path; production builds use process isolation."""

        try:
            app = self.modal.App.lookup(
                self.app_name,
                environment_name=self.environment_name,
                create_if_missing=True,
            )
            image = self.modal.Image.from_dockerfile(
                dockerfile,
                context_dir=compression_root,
                secrets=[],
            )
            output_context = nullcontext()
            if _dev_mode_enabled():
                enable_output = getattr(self.modal, "enable_output", None)
                if callable(enable_output):
                    output_context = enable_output()
            with output_context:
                built_image = image.build(app)
            image_id = str(
                getattr(built_image, "object_id", "") or getattr(image, "object_id", "")
            )
            if not image_id:
                raise RuntimeError("Modal returned no immutable image ID")
        except Exception as exc:
            raise RuntimeError(str(exc)[:1000]) from exc
        return image_id


class ModalImageBuilder:
    """Verify source binding and limitations of a direct Modal build."""

    def __init__(self, backend: ModalImageBuildBackend) -> None:
        self.backend = backend

    def build(self, request: BuildRequest) -> BuildEvidence:
        evidence = self.backend.build(request)
        if evidence.builder_id != ModalImageBuildBackend.builder_id:
            raise TrustedBuildError(
                BuildReason.UNTRUSTED_BUILDER,
                "Modal evidence came from an unexpected builder",
            )
        if (
            evidence.security_level != "MODAL_UNATTESTED"
            or evidence.image_size_measurement != "UNAVAILABLE"
        ):
            raise TrustedBuildError(
                BuildReason.IMAGE_SIZE_EVIDENCE_MISSING,
                "Modal evidence must state that image size is unavailable",
            )
        if evidence.source_tree_sha != request.pinned_tree_sha:
            raise TrustedBuildError(
                BuildReason.SOURCE_EVIDENCE_MISMATCH,
                "Modal image is not bound to the pinned contender tree",
            )
        if not evidence.image_id.strip():
            raise TrustedBuildError(
                BuildReason.IMAGE_ID_MISSING, "Modal returned no image ID"
            )
        if not SHA256_DIGEST.fullmatch(evidence.image_digest):
            raise TrustedBuildError(
                BuildReason.IMAGE_DIGEST_INVALID,
                "Modal image binding digest is invalid",
            )
        if evidence.quota_enforced_during_build:
            raise TrustedBuildError(
                BuildReason.BUILD_QUOTA_MISMATCH,
                "Modal backend unexpectedly claimed trusted quota enforcement",
            )
        return evidence


class TrustedImageBuilder:
    """Accept only size-attested immutable images from configured builders."""

    def __init__(
        self,
        backend: TrustedBuildBackend,
        *,
        trusted_builder_ids: frozenset[str],
    ) -> None:
        self.backend = backend
        self.trusted_builder_ids = trusted_builder_ids

    def build(self, request: BuildRequest) -> BuildEvidence:
        if self.backend.builder_id not in self.trusted_builder_ids:
            raise TrustedBuildError(
                BuildReason.UNTRUSTED_BUILDER,
                f"builder {self.backend.builder_id!r} is not allowlisted",
            )
        evidence = self.backend.build(request)
        self.verify(request, evidence)
        return evidence

    def verify(self, request: BuildRequest, evidence: BuildEvidence) -> None:
        if evidence.builder_id != self.backend.builder_id:
            raise TrustedBuildError(
                BuildReason.UNTRUSTED_BUILDER,
                "evidence builder identity does not match the selected backend",
            )
        if evidence.source_tree_sha != request.pinned_tree_sha:
            raise TrustedBuildError(
                BuildReason.SOURCE_EVIDENCE_MISMATCH,
                "build evidence is not bound to the pinned contender tree",
            )
        if not evidence.image_id.strip():
            raise TrustedBuildError(
                BuildReason.IMAGE_ID_MISSING, "builder returned no immutable image ID"
            )
        if not SHA256_DIGEST.fullmatch(evidence.image_digest):
            raise TrustedBuildError(
                BuildReason.IMAGE_DIGEST_INVALID,
                "builder returned an invalid image digest",
            )
        if (
            evidence.security_level != "TRUSTED"
            or evidence.image_size_measurement != "FINAL_IMAGE"
        ):
            raise TrustedBuildError(
                BuildReason.IMAGE_SIZE_EVIDENCE_MISSING,
                "trusted evidence must contain a final-image size measurement",
            )
        if not evidence.quota_enforced_during_build:
            raise TrustedBuildError(
                BuildReason.BUILD_QUOTA_NOT_ENFORCED,
                "the 25 GB limit was measured after build but not enforced during it",
            )
        if evidence.quota_limit_bytes != request.size_limit_bytes:
            raise TrustedBuildError(
                BuildReason.BUILD_QUOTA_MISMATCH,
                "builder quota does not equal the manifest's fixed 25 GB limit",
            )
        if isinstance(evidence.image_size_bytes, bool) or not isinstance(
            evidence.image_size_bytes, int
        ):
            raise TrustedBuildError(
                BuildReason.IMAGE_SIZE_EVIDENCE_MISSING,
                "builder returned no integer image-size measurement",
            )
        try:
            enforce_image_size(evidence.image_size_bytes, request.size_limit_bytes)
        except (TypeError, ValueError) as exc:
            raise TrustedBuildError(
                BuildReason.IMAGE_SIZE_LIMIT_EXCEEDED, str(exc)
            ) from exc


class CompetitionBuildService:
    """Bind the trusted builder to pinned contender and SQLite state."""

    def __init__(
        self,
        repository: Any,
        builder: Any,
        *,
        actor: str,
        accepted_build_status: str = "ACCEPTED",
        clock=lambda: datetime.now(timezone.utc),
    ) -> None:
        self.repository = repository
        self.builder = builder
        self.actor = actor
        self.accepted_build_status = accepted_build_status
        self.clock = clock

    def build_contender(
        self,
        competition_id: str,
        hotkey: str,
        source_root: Path,
        modal_build_timeout_seconds: float = MODAL_BUILD_TIMEOUT_SECONDS,
    ) -> BuildEvidence:
        contender = self.repository.get_contender(competition_id, hotkey)
        if contender is None or not contender.pinned_tree_sha:
            raise TrustedBuildError(
                BuildReason.SOURCE_EVIDENCE_MISMATCH,
                "contender source was not immutably pinned before build",
            )
        try:
            request = BuildRequest(
                competition_id=competition_id,
                hotkey=hotkey,
                source_root=source_root,
                pinned_tree_sha=contender.pinned_tree_sha,
                modal_build_timeout_seconds=modal_build_timeout_seconds,
            )
            evidence = self.builder.build(request)
        except TrustedBuildError as exc:
            if exc.reason_code == BuildReason.BUILD_INFRASTRUCTURE_ERROR:
                self.repository.record_build_retryable_failure(
                    competition_id=competition_id,
                    hotkey=hotkey,
                    reason_code=exc.reason_code.value,
                    detail=str(exc),
                    now=self.clock(),
                    actor=self.actor,
                )
            else:
                self.repository.record_build_rejection(
                    competition_id=competition_id,
                    hotkey=hotkey,
                    reason_code=exc.reason_code.value,
                    detail=str(exc),
                    now=self.clock(),
                    actor=self.actor,
                )
            raise
        except Exception as exc:
            wrapped = TrustedBuildError(
                BuildReason.BUILD_INFRASTRUCTURE_ERROR, str(exc)[:1000]
            )
            self.repository.record_build_retryable_failure(
                competition_id=competition_id,
                hotkey=hotkey,
                reason_code=wrapped.reason_code.value,
                detail=str(wrapped),
                now=self.clock(),
                actor=self.actor,
            )
            raise wrapped from exc
        self.repository.record_build_evidence(
            competition_id=competition_id,
            hotkey=hotkey,
            image_id=evidence.image_id,
            image_digest=evidence.image_digest,
            image_size_bytes=evidence.image_size_bytes,
            evidence=evidence.as_dict(),
            build_status=self.accepted_build_status,
            now=self.clock(),
            actor=self.actor,
        )
        return evidence


__all__ = [
    "BuildEvidence",
    "BuildReason",
    "BuildRequest",
    "CompetitionBuildService",
    "ModalImageBuildBackend",
    "ModalImageBuilder",
    "MODAL_BUILD_TIMEOUT_SECONDS",
    "TrustedBuildBackend",
    "TrustedBuildError",
    "TrustedImageBuilder",
]
