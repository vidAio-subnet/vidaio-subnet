"""Restart-safe competition invitation and repository-submission polling."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from loguru import logger

from vidaio_subnet_core.protocol import (
    CompetitionInvitationProtocol,
    CompetitionSubmissionProtocol,
    CompetitionSubmissionReviewStatus,
    CompetitionSubmissionStatus,
    CompetitionType,
)

from .config import CompetitionManifest
from .intake import (
    CompetitionSubmissionIntakeService,
    RepositorySubmission,
)
from .models import Competition
from .phase0 import SecretRedactor
from .repository import CompetitionRepository
from .state import CompetitionState


@dataclass(frozen=True)
class CompetitionMinerEndpoint:
    """Point-in-time chain identity and opaque transport target for one miner."""

    uid: int
    hotkey: str
    coldkey: str | None
    transport: Any = field(repr=False)


class CompetitionForwarder(Protocol):
    async def __call__(
        self,
        endpoint: CompetitionMinerEndpoint,
        synapse: CompetitionInvitationProtocol | CompetitionSubmissionProtocol,
        timeout_seconds: float,
    ) -> Any | None: ...


Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EnrollmentResponseError(RuntimeError):
    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(detail)


class CompetitionEnrollmentDispatcher:
    """Send invitations and poll opted-in miners without retaining credentials."""

    def __init__(
        self,
        repository: CompetitionRepository,
        intake: CompetitionSubmissionIntakeService,
        forward: CompetitionForwarder,
        *,
        owner_id: str,
        timeout_seconds: float = 30,
        max_concurrent_requests: int = 32,
        clock: Clock = utc_now,
    ) -> None:
        self.repository = repository
        self.intake = intake
        self.forward = forward
        self.owner_id = owner_id
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._clock = clock

    async def run_once(
        self,
        competition: Competition,
        endpoints: list[CompetitionMinerEndpoint],
    ) -> None:
        now = self._normalized_now()
        if competition.status != CompetitionState.ENROLLING.value:
            return
        manifest = CompetitionManifest.model_validate_json(competition.manifest_json)
        if now >= manifest.contender_finalisation_time:
            return

        endpoints_by_hotkey = {endpoint.hotkey: endpoint for endpoint in endpoints}
        candidates = [
            (endpoint.hotkey, endpoint.uid, endpoint.coldkey)
            for endpoint in endpoints_by_hotkey.values()
        ]
        created = await asyncio.to_thread(
            self.repository.ensure_invitation_candidates,
            competition.competition_id,
            candidates,
            now=now,
            actor=self.owner_id,
        )
        retry_before = now - manifest.contender_ping_interval
        due_invitations = await asyncio.to_thread(
            self.repository.list_due_invitation_candidates,
            competition.competition_id,
            retry_before=retry_before,
        )
        invitation_tasks = [
            self._invite(competition, manifest, endpoints_by_hotkey[row.hotkey], row.hotkey)
            for row in due_invitations
            if row.hotkey in endpoints_by_hotkey
        ]
        if invitation_tasks:
            logger.info(
                "Sending competition invitations: id={} miners={}",
                competition.competition_id,
                len(invitation_tasks),
            )
            await asyncio.gather(*invitation_tasks)

        poll_now = self._normalized_now()
        poll_retry_before = poll_now - manifest.contender_ping_interval
        due_submissions = await asyncio.to_thread(
            self.repository.list_due_submission_candidates,
            competition.competition_id,
            retry_before=poll_retry_before,
        )
        intake_semaphore = asyncio.Semaphore(manifest.max_parallel_contenders)
        submission_tasks = [
            self._poll_submission(
                competition,
                manifest,
                endpoints_by_hotkey[row.hotkey],
                row.hotkey,
                row.uid_snapshot,
                row.coldkey_snapshot,
                row.validation_status,
                row.reason_code,
                row.reason_detail,
                row.pinned_commit_sha,
                row.submission_revision or 0,
                intake_semaphore,
            )
            for row in due_submissions
            if row.hotkey in endpoints_by_hotkey
        ]
        if submission_tasks:
            logger.info(
                "Polling competition submissions: id={} participating_miners={}",
                competition.competition_id,
                len(submission_tasks),
            )
            await asyncio.gather(*submission_tasks)

        cycle_logger = (
            logger.info
            if created or invitation_tasks or submission_tasks
            else logger.debug
        )
        cycle_logger(
            "Competition enrollment cycle: id={} endpoints={} new_candidates={} "
            "invitations={} submission_polls={}",
            competition.competition_id,
            len(endpoints_by_hotkey),
            created,
            len(invitation_tasks),
            len(submission_tasks),
        )

    async def _invite(
        self,
        competition: Competition,
        manifest: CompetitionManifest,
        endpoint: CompetitionMinerEndpoint,
        hotkey: str,
    ) -> None:
        now = self._normalized_now()
        retry_before = now - manifest.contender_ping_interval
        claimed = await asyncio.to_thread(
            self.repository.claim_invitation_attempt,
            competition.competition_id,
            hotkey,
            retry_before=retry_before,
            now=now,
        )
        if not claimed:
            return

        nonce = secrets.token_urlsafe(24)
        synapse = CompetitionInvitationProtocol(
            competition_id=competition.competition_id,
            competition_type=CompetitionType.COMPRESSION,
            manifest_digest=competition.manifest_digest,
            registration_deadline=manifest.contender_finalisation_time,
            invitation_nonce=nonce,
        )
        try:
            async with self._semaphore:
                response = await self.forward(
                    endpoint, synapse, self.timeout_seconds
                )
            self._validate_endpoint_hotkey(response, endpoint)
            self._validate_manifest_binding(response, competition)
            invitation = response.invitation_response
            if invitation.competition_id != competition.competition_id:
                raise EnrollmentResponseError(
                    "COMPETITION_ID_MISMATCH",
                    "invitation response belongs to another competition",
                )
            if invitation.echo_nonce != nonce:
                raise EnrollmentResponseError(
                    "INVITATION_REPLAY",
                    "invitation response nonce is stale or unexpected",
                )
            if invitation.participating and (
                invitation.supported_competition_type != CompetitionType.COMPRESSION
            ):
                raise EnrollmentResponseError(
                    "COMPETITION_TYPE_MISMATCH",
                    "miner opted in without confirming compression support",
                )
            response_time = self._normalized_now()
            if response_time >= manifest.contender_finalisation_time:
                raise EnrollmentResponseError(
                    "ENROLLMENT_WINDOW_CLOSED",
                    "invitation response arrived after contender finalisation",
                )
            await asyncio.to_thread(
                self.repository.record_invitation_response,
                competition.competition_id,
                hotkey,
                participating=invitation.participating,
                refusal_reason=invitation.refusal_reason,
                now=response_time,
                actor=self.owner_id,
            )
            logger.info(
                "Competition invitation response: id={} uid={} hotkey={} participating={}",
                competition.competition_id,
                endpoint.uid,
                hotkey,
                invitation.participating,
            )
        except Exception as exc:
            await self._record_error(
                competition.competition_id,
                hotkey,
                stage="INVITATION",
                exc=exc,
            )

    async def _poll_submission(
        self,
        competition: Competition,
        manifest: CompetitionManifest,
        endpoint: CompetitionMinerEndpoint,
        hotkey: str,
        uid_snapshot: int | None,
        coldkey_snapshot: str | None,
        validation_status: str | None,
        reason_code: str | None,
        reason_detail: str | None,
        pinned_commit_sha: str | None,
        submission_revision: int,
        intake_semaphore: asyncio.Semaphore,
    ) -> None:
        now = self._normalized_now()
        retry_before = now - manifest.contender_ping_interval
        claimed = await asyncio.to_thread(
            self.repository.claim_submission_poll,
            competition.competition_id,
            hotkey,
            retry_before=retry_before,
            now=now,
        )
        if not claimed:
            return

        nonce = secrets.token_urlsafe(24)
        try:
            feedback_status = CompetitionSubmissionReviewStatus(
                validation_status or "NOT_RECEIVED"
            )
        except ValueError:
            feedback_status = CompetitionSubmissionReviewStatus.NOT_RECEIVED
        synapse = CompetitionSubmissionProtocol(
            competition_id=competition.competition_id,
            manifest_digest=competition.manifest_digest,
            request_nonce=nonce,
            requested_at=now,
            last_submission_status=feedback_status,
            last_submission_reason_code=reason_code,
            last_submission_reason_detail=reason_detail,
            last_pinned_commit_sha=pinned_commit_sha,
            submission_revision=submission_revision,
        )
        credential = ""
        response = None
        try:
            async with self._semaphore:
                response = await self.forward(
                    endpoint, synapse, self.timeout_seconds
                )
            self._validate_endpoint_hotkey(response, endpoint)
            self._validate_manifest_binding(response, competition)
            submission_response = response.submission_response
            credential = submission_response.github_pat
            if submission_response.competition_id != competition.competition_id:
                raise EnrollmentResponseError(
                    "COMPETITION_ID_MISMATCH",
                    "submission response belongs to another competition",
                )
            if submission_response.echo_nonce != nonce:
                raise EnrollmentResponseError(
                    "SUBMISSION_REPLAY",
                    "submission response nonce is stale or unexpected",
                )
            response_time = self._normalized_now()
            if response_time >= manifest.contender_finalisation_time:
                raise EnrollmentResponseError(
                    "SUBMISSION_WINDOW_CLOSED",
                    "submission response arrived after contender finalisation",
                )
            if submission_response.status == CompetitionSubmissionStatus.READY:
                submission = RepositorySubmission(
                    competition_id=competition.competition_id,
                    contender_hotkey=hotkey,
                    repository_url=submission_response.repository_url,
                    github_pat=credential,
                    nonce=nonce,
                )
                async with intake_semaphore:
                    pinned = await asyncio.to_thread(
                        self.intake.finalize,
                        submission,
                        expected_competition_id=competition.competition_id,
                        expected_hotkey=hotkey,
                        expected_nonce=nonce,
                        actor=self.owner_id,
                        uid_snapshot=uid_snapshot,
                        coldkey_snapshot=coldkey_snapshot,
                    )
                credential = ""
                logger.info(
                    "Competition submission pinned: id={} uid={} hotkey={} repository={}",
                    competition.competition_id,
                    endpoint.uid,
                    hotkey,
                    pinned.repository_display,
                )
            else:
                await asyncio.to_thread(
                    self.repository.record_submission_poll_result,
                    competition.competition_id,
                    hotkey,
                    status=submission_response.status.value,
                    reason=submission_response.reason,
                    now=response_time,
                    actor=self.owner_id,
                )
                logger.info(
                    "Competition submission response: id={} uid={} hotkey={} status={}",
                    competition.competition_id,
                    endpoint.uid,
                    hotkey,
                    submission_response.status.value,
                )
        except Exception as exc:
            await self._record_error(
                competition.competition_id,
                hotkey,
                stage="SUBMISSION_POLL",
                exc=exc,
                secrets_to_redact=(credential,),
            )
        finally:
            credential = ""
            if response is not None:
                submission_response = getattr(response, "submission_response", None)
                if submission_response is not None:
                    submission_response.github_pat = ""

    @staticmethod
    def _validate_endpoint_hotkey(
        response: Any | None, endpoint: CompetitionMinerEndpoint
    ) -> None:
        if response is None:
            raise EnrollmentResponseError("NO_RESPONSE", "miner returned no response")
        response_axon = getattr(response, "axon", None)
        response_hotkey = str(getattr(response_axon, "hotkey", "") or "")
        if not response_hotkey or response_hotkey != endpoint.hotkey:
            raise EnrollmentResponseError(
                "HOTKEY_MISMATCH",
                "response hotkey does not match the queried axon",
            )

    @staticmethod
    def _validate_manifest_binding(response: Any, competition: Competition) -> None:
        if (
            getattr(response, "competition_id", "") != competition.competition_id
            or getattr(response, "manifest_digest", "")
            != competition.manifest_digest
        ):
            raise EnrollmentResponseError(
                "MANIFEST_BINDING_MISMATCH",
                "response is bound to a stale or different competition manifest",
            )

    async def _record_error(
        self,
        competition_id: str,
        hotkey: str,
        *,
        stage: str,
        exc: Exception,
        secrets_to_redact: tuple[str, ...] = (),
    ) -> None:
        redactor = SecretRedactor(secrets_to_redact)
        detail = redactor.redact_text(str(exc))[:500]
        reason_code = getattr(exc, "reason_code", type(exc).__name__.upper())
        await asyncio.to_thread(
            self.repository.record_enrollment_error,
            competition_id,
            hotkey,
            stage=stage,
            reason_code=str(reason_code),
            detail=detail,
            now=self._normalized_now(),
            actor=self.owner_id,
        )
        logger.warning(
            "Competition enrollment request failed: id={} hotkey={} stage={} "
            "reason_code={} detail={}",
            competition_id,
            hotkey,
            stage,
            reason_code,
            detail,
        )

    def _normalized_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("competition enrollment clock must be timezone-aware")
        return now.astimezone(timezone.utc)


__all__ = [
    "CompetitionEnrollmentDispatcher",
    "CompetitionForwarder",
    "CompetitionMinerEndpoint",
    "EnrollmentResponseError",
]
