from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from bittensor import Synapse
from typing import Any, Optional, List
from enum import Enum, IntEnum
from datetime import datetime, timedelta, timezone
import re

class Version(BaseModel):
    major: int
    minor: int
    patch: int
    
class ContentLength(IntEnum):
    """
    Enumeration of allowed video content lengths in seconds.
    These represent the maximum duration of video content that miners can process efficiently.
    """
    FIVE = 5
    TEN = 10
    # TWENTY = 20    
    # FORTY = 40  
    # EIGHTY = 80    
    # ONE_SIXTY = 160  
    # THREE_TWENTY = 320 

class TaskType(IntEnum):
    """
    Enumeration of allowed task types that miners can handle.
    These represent the types of video processing tasks that miners can warrant.
    """
    COMPRESSION = 1
    UPSCALING = 2

class UpscalingMinerPayload(BaseModel):
    reference_video_urls: List[str] = Field(
        description="The URLs of the reference videos to be optimized",
        default_factory=list,
    )
    reference_video_url: str = Field(
        description="Legacy scalar URL of the first reference video",
        default="",
    )
    maximum_optimized_size_mb: int = Field(
        description="The maximum size of the optimized video in MB",
        default=100,
        gt=0,
    )
    task_types: List[str] = Field(
        description="The types of tasks: HD24K, SD2HD, SD24K, 4K28K",
        default_factory=list,
    )
    task_type: str = Field(
        description="Legacy scalar task type for the first reference video",
        default="HD24K",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            if not data.get("reference_video_urls") and data.get("reference_video_url"):
                data["reference_video_urls"] = [data["reference_video_url"]]
            if not data.get("reference_video_url") and data.get("reference_video_urls"):
                data["reference_video_url"] = data["reference_video_urls"][0]
            if not data.get("task_types") and data.get("task_type"):
                data["task_types"] = [data["task_type"]]
            if not data.get("task_type") and data.get("task_types"):
                data["task_type"] = data["task_types"][0]
        return data

    @model_validator(mode="after")
    def sync_legacy_fields(self):
        if not self.reference_video_urls and self.reference_video_url:
            self.reference_video_urls = [self.reference_video_url]
        if self.reference_video_urls:
            self.reference_video_url = self.reference_video_urls[0]
        if not self.task_types and self.task_type:
            self.task_types = [self.task_type]
        if self.task_types:
            self.task_type = self.task_types[0]
        return self

class CompressionMinerPayload(BaseModel):
    reference_video_urls: List[str] = Field(
        description="The URLs of the reference videos to be compressed",
        default_factory=list,
    )
    reference_video_url: str = Field(
        description="Legacy scalar URL of the first reference video",
        default="",
    )
    vmaf_threshold: float = Field(
        description="The VMAF threshold for quality control during compression",
        default=90.0,
        ge=0.0,
        le=100.0,
    )
    vmaf_thresholds: List[float] = Field(
        description="Per-video VMAF thresholds for batched compression payloads",
        default_factory=list,
    )
    target_codec: str = Field(
        description="The target codec for compression (e.g., av1, hevc, h264, vp9)",
        default="av1",
    )
    target_codecs: List[str] = Field(
        description="Per-video target codecs for batched compression payloads",
        default_factory=list,
    )
    codec_mode: str = Field(
        description="Codec mode: CBR (Constant Bitrate), VBR (Variable Bitrate), or CRF (Constant Rate Factor)",
        default="CRF",
    )
    codec_modes: List[str] = Field(
        description="Per-video codec modes for batched compression payloads",
        default_factory=list,
    )
    target_bitrate: float = Field(
        description="Target bitrate in Mbps (megabits per second)",
        default=10.0,
        gt=0.0,
    )
    target_bitrates: List[float] = Field(
        description="Per-video target bitrates for batched compression payloads",
        default_factory=list,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            if not data.get("reference_video_urls") and data.get("reference_video_url"):
                data["reference_video_urls"] = [data["reference_video_url"]]
            if not data.get("reference_video_url") and data.get("reference_video_urls"):
                data["reference_video_url"] = data["reference_video_urls"][0]
        return data

    @model_validator(mode="after")
    def sync_legacy_fields(self):
        if not self.reference_video_urls and self.reference_video_url:
            self.reference_video_urls = [self.reference_video_url]
        if self.reference_video_urls:
            self.reference_video_url = self.reference_video_urls[0]
        return self


class MinerResponse(BaseModel):
    optimized_video_urls: List[str] = Field(
        description="The URLs of the processed videos (compressed/upscaled)",
        default_factory=list,
    )
    optimized_video_url: str = Field(
        description="Legacy scalar URL of the first processed video",
        default="",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            if not data.get("optimized_video_urls") and data.get("optimized_video_url"):
                data["optimized_video_urls"] = [data["optimized_video_url"]]
            if not data.get("optimized_video_url") and data.get("optimized_video_urls"):
                data["optimized_video_url"] = data["optimized_video_urls"][0]
        return data

    @model_validator(mode="after")
    def sync_legacy_fields(self):
        if not self.optimized_video_urls and self.optimized_video_url:
            self.optimized_video_urls = [self.optimized_video_url]
        if self.optimized_video_urls:
            self.optimized_video_url = self.optimized_video_urls[0]
        return self


# ---------------------------------------------------------------------------
# Polling-based organic protocol models
# ---------------------------------------------------------------------------

class JobKickoffResponse(BaseModel):
    """Returned by miner on job kick-off — confirms whether the job was accepted."""
    accepted: bool = Field(
        description="Whether the miner accepted the job",
        default=False,
    )


class PollResponse(BaseModel):
    """Returned by miner on each poll request."""
    job_id: str = Field(
        description="The job_id being polled",
        default="",
    )
    status: str = Field(
        description="Job status: 'processing' | 'completed' | 'failed'",
        default="unknown",
    )
    optimized_video_urls: List[str] = Field(
        description="Populated with result URLs once status is 'completed'",
        default_factory=list,
    )
    optimized_video_url: str = Field(
        description="Legacy scalar URL of the first processed video",
        default="",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            if not data.get("optimized_video_urls") and data.get("optimized_video_url"):
                data["optimized_video_urls"] = [data["optimized_video_url"]]
            if not data.get("optimized_video_url") and data.get("optimized_video_urls"):
                data["optimized_video_url"] = data["optimized_video_urls"][0]
        return data

    @model_validator(mode="after")
    def sync_legacy_fields(self):
        if not self.optimized_video_urls and self.optimized_video_url:
            self.optimized_video_urls = [self.optimized_video_url]
        if self.optimized_video_urls:
            self.optimized_video_url = self.optimized_video_urls[0]
        return self


class ScoringPayload(BaseModel):
    reference_video_url: str = Field(
        description="The URL of the reference video",
        default="",
        min_length=1, 
    )
    optimized_video_url: str = Field(
        description="The URL of the processed video (compressed/upscaled)",
        default="",
        min_length=1,
    )


class ScoringResponse(BaseModel):
    score: float = Field(
        description="Quality score of the processed video",
        default=0.0,
        ge=0.0,  
        le=1.0,  
    )


class VideoCompressionProtocol(Synapse):
    """Protocol for video compression operations."""
    
    version: Optional[Version] = None
    round_id: Optional[str] = None
    
    miner_payload: CompressionMinerPayload = Field(
        description="The payload for the compression miner. Cannot be modified after initialization.",
        default_factory=CompressionMinerPayload,
        frozen=True,
    )
    miner_response: MinerResponse = Field(
        description="The response from the miner",
        default_factory=MinerResponse,
    )

    @property
    def scoring_payload(self) -> ScoringPayload:
        """Generate scoring payload from miner payload and response."""
        return ScoringPayload(
            reference_video_url=self.miner_payload.reference_video_url,
            optimized_video_url=self.miner_response.optimized_video_url,
        )


class VideoUpscalingProtocol(Synapse):
    """Protocol for video upscaling operations."""
    
    version: Optional[Version] = None

    round_id: Optional[str] = None
    
    miner_payload: UpscalingMinerPayload = Field(
        description="The payload for the upscaling miner. Cannot be modified after initialization.",
        default_factory=UpscalingMinerPayload,
        frozen=True,
    )
    miner_response: MinerResponse = Field(
        description="The response from the miner",
        default_factory=MinerResponse,
    )

    @property
    def scoring_payload(self) -> ScoringPayload:
        """Generate scoring payload from miner payload and response."""
        return ScoringPayload(
            reference_video_url=self.miner_payload.reference_video_url,
            optimized_video_url=self.miner_response.optimized_video_url,
        )


class LengthCheckProtocol(Synapse):
    """
    Protocol for verifying and enforcing maximum content length constraints.
    
    This protocol ensures that content processing requests don't exceed the miner's
    capacity to handle content within a reasonable timeframe. Miners can specify
    their maximum supported content length from the predefined options.
    
    Attributes:
        version (Optional[Version]): The version of the protocol implementation.
        max_content_length (ContentLength): Maximum content length that
            miners can process, must be one of the predefined values (5, 10, or 20).
    """
    
    version: Optional[Version] = None
    max_content_length: ContentLength = Field(
        description="Maximum content length miner can process (5, 10, or 20)",
        default=ContentLength.FIVE
    )


class TaskWarrantProtocol(Synapse):
    """
    Protocol for verifying and warranting task types that miners can handle.
    
    This protocol ensures that miners can specify which types of video processing
    tasks they are capable of handling. This helps in task distribution and
    ensures miners only receive tasks they can process.
    
    Attributes:
        version (Optional[Version]): The version of the protocol implementation.
        warrant_task (Optional[TaskType]): The type of task the miner can handle,
            must be one of the predefined values (COMPRESSION or UPSCALING).
            Will be None if miner doesn't respond, allowing fallback to performance history.
    """
    
    version: Optional[Version] = None
    warrant_task: Optional[TaskType] = Field(
        description="Type of task miner can handle: COMPRESSION or UPSCALING",
        default=None
    )


# ---------------------------------------------------------------------------
# Polling-based organic compression protocols
# ---------------------------------------------------------------------------

class VideoCompressionJobProtocol(Synapse):
    """Phase-1 organic compression kick-off.

    The validator assigns a ``job_id`` before sending. The miner uses it as
    the key for the async job so both sides share the same identifier.
    The ``miner_response`` field is populated by the validator after the
    poll phase completes, so downstream code can access
    ``synapse.miner_response.optimized_video_url`` unchanged.
    """

    job_id: str = Field(
        description="Validator-assigned UUID for this job",
        default="",
    )
    miner_payload: CompressionMinerPayload = Field(
        description="Compression parameters for the miner job",
        default_factory=CompressionMinerPayload,
        frozen=True,
    )
    job_response: JobKickoffResponse = Field(
        description="Miner's ack",
        default_factory=JobKickoffResponse,
    )
    miner_response: MinerResponse = Field(
        description="Filled by validator after polling completes (for downstream compatibility)",
        default_factory=MinerResponse,
    )


class VideoCompressionPollProtocol(Synapse):
    """Phase-2 organic compression poll.

    Validator sends the ``job_id`` from Phase-1; miner returns current
    status and, when complete, the ``optimized_video_url``.
    """

    job_id: str = Field(
        description="The job_id returned during the kick-off phase",
        default="",
    )
    poll_response: PollResponse = Field(
        description="Miner's current status + optional result URL",
        default_factory=PollResponse,
    )


# ---------------------------------------------------------------------------
# Polling-based organic upscaling protocols
# ---------------------------------------------------------------------------

class VideoUpscalingJobProtocol(Synapse):
    """Phase-1 organic upscaling kick-off.

    The validator assigns a ``job_id`` before sending. The miner uses it as
    the key for the async job so both sides share the same identifier.
    The ``miner_response`` field is populated by the validator after the
    poll phase completes, so downstream code can access
    ``synapse.miner_response.optimized_video_url`` unchanged.
    """

    job_id: str = Field(
        description="Validator-assigned UUID for this job",
        default="",
    )
    miner_payload: UpscalingMinerPayload = Field(
        description="Upscaling parameters for the miner job",
        default_factory=UpscalingMinerPayload,
        frozen=True,
    )
    job_response: JobKickoffResponse = Field(
        description="Miner's ack",
        default_factory=JobKickoffResponse,
    )
    miner_response: MinerResponse = Field(
        description="Filled by validator after polling completes (for downstream compatibility)",
        default_factory=MinerResponse,
    )


class VideoUpscalingPollProtocol(Synapse):
    """Phase-2 organic upscaling poll.

    Validator sends the ``job_id`` from Phase-1; miner returns current
    status and, when complete, the ``optimized_video_url``.
    """

    job_id: str = Field(
        description="The job_id returned during the kick-off phase",
        default="",
    )
    poll_response: PollResponse = Field(
        description="Miner's current status + optional result URL",
        default_factory=PollResponse,
    )


# ---------------------------------------------------------------------------
# Validator-owned compression competition protocols
# ---------------------------------------------------------------------------

class CompetitionType(str, Enum):
    COMPRESSION = "COMPRESSION"


class CompetitionSubmissionStatus(str, Enum):
    NOT_READY = "NOT_READY"
    READY = "READY"
    WITHDRAWN = "WITHDRAWN"


class CompetitionSubmissionReviewStatus(str, Enum):
    NOT_RECEIVED = "NOT_RECEIVED"
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class CompetitionInvitationResponse(BaseModel):
    competition_id: str = ""
    echo_nonce: str = ""
    participating: bool = False
    supported_competition_type: CompetitionType | None = None
    refusal_reason: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_participation(self):
        if self.participating:
            if self.supported_competition_type != CompetitionType.COMPRESSION:
                raise ValueError("participating miners must confirm COMPRESSION support")
            if self.refusal_reason:
                raise ValueError("a participating miner cannot include a refusal reason")
        return self


_GITHUB_REPOSITORY_PATTERN = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$"
)

_COMPETITION_TRANSPORT_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_COMPETITION_REQUEST_MAX_AGE = timedelta(minutes=5)
_COMPETITION_REQUEST_FUTURE_TOLERANCE = timedelta(seconds=30)


class CompetitionSubmissionResponse(BaseModel):
    competition_id: str = ""
    echo_nonce: str = ""
    status: CompetitionSubmissionStatus = CompetitionSubmissionStatus.NOT_READY
    repository_url: str = ""
    github_pat: str = Field(
        default="",
        repr=False,
        description="Raw short-lived read-only PAT; never persist or log this field",
    )
    commit_hint: str | None = Field(default=None, max_length=64)
    reason: str | None = Field(default=None, max_length=256)

    @field_validator("repository_url")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        if value and not _GITHUB_REPOSITORY_PATTERN.fullmatch(value):
            raise ValueError("repository_url must be a GitHub HTTPS .git URL")
        return value

    @model_validator(mode="after")
    def validate_status_fields(self):
        if self.status == CompetitionSubmissionStatus.READY:
            if not self.repository_url or not self.github_pat:
                raise ValueError("READY submissions require repository_url and github_pat")
        elif self.repository_url or self.github_pat:
            raise ValueError("non-ready submissions cannot include repository credentials")
        return self


class CompetitionInvitationProtocol(Synapse):
    """Validator invitation and miner opt-in response for one manifest revision."""

    protocol_version: int = Field(default=2, ge=2)
    competition_id: str = Field(
        default="",
        pattern=r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$"
    )
    competition_type: CompetitionType = CompetitionType.COMPRESSION
    manifest_digest: str = Field(default="", pattern=r"^[a-f0-9]{64}$")
    registration_deadline: datetime = _COMPETITION_TRANSPORT_EPOCH
    invitation_nonce: str = Field(default="", min_length=16, max_length=128)
    invitation_response: CompetitionInvitationResponse = Field(
        default_factory=CompetitionInvitationResponse
    )

    @field_serializer("registration_deadline")
    def serialize_registration_deadline(self, value: datetime) -> str:
        """Keep Bittensor's header serializer on JSON-native wire values."""

        return value.isoformat()

    @model_validator(mode="after")
    def validate_response_binding(self):
        response = self.invitation_response
        if response.competition_id and response.competition_id != self.competition_id:
            raise ValueError("invitation response belongs to another competition")
        if response.echo_nonce and response.echo_nonce != self.invitation_nonce:
            raise ValueError("invitation response nonce does not match the request")
        if self.registration_deadline.tzinfo is None:
            raise ValueError("registration_deadline must be timezone-aware")
        return self

    def is_open_invitation(self, now: datetime) -> bool:
        """Reject transport defaults and invitations outside their UTC window."""

        if now.tzinfo is None or now.utcoffset() is None:
            return False
        return bool(
            self.competition_id
            and self.manifest_digest
            and self.invitation_nonce
            and now.astimezone(timezone.utc)
            <= self.registration_deadline.astimezone(timezone.utc)
        )


class CompetitionSubmissionProtocol(Synapse):
    """Validator submission poll and raw-PAT miner response.

    The request nonce and manifest digest prevent a response from being accepted
    for a stale or different competition. The PAT is deliberately wire-visible,
    but omitted from repr; consumers must immediately move it to the ephemeral
    Git askpass boundary and persist only a credential-free record.
    """

    protocol_version: int = Field(default=3, ge=2)
    competition_id: str = Field(
        default="",
        pattern=r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$"
    )
    manifest_digest: str = Field(default="", pattern=r"^[a-f0-9]{64}$")
    request_nonce: str = Field(default="", min_length=16, max_length=128)
    requested_at: datetime = _COMPETITION_TRANSPORT_EPOCH
    last_submission_status: CompetitionSubmissionReviewStatus = (
        CompetitionSubmissionReviewStatus.NOT_RECEIVED
    )
    last_submission_reason_code: str | None = Field(default=None, max_length=128)
    last_submission_reason_detail: str | None = Field(default=None, max_length=500)
    last_pinned_commit_sha: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{40,64}$"
    )
    submission_revision: int = Field(default=0, ge=0)
    submission_response: CompetitionSubmissionResponse = Field(
        default_factory=CompetitionSubmissionResponse
    )

    @field_serializer("requested_at")
    def serialize_requested_at(self, value: datetime) -> str:
        """Keep Bittensor's header serializer on JSON-native wire values."""

        return value.isoformat()

    @model_validator(mode="after")
    def validate_response_binding(self):
        response = self.submission_response
        if response.competition_id and response.competition_id != self.competition_id:
            raise ValueError("submission response belongs to another competition")
        if response.echo_nonce and response.echo_nonce != self.request_nonce:
            raise ValueError("submission response nonce does not match the request")
        if self.requested_at.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")
        return self

    def is_fresh_request(self, now: datetime) -> bool:
        """Reject transport defaults, stale polls, and implausibly future polls."""

        if now.tzinfo is None or now.utcoffset() is None:
            return False
        normalized_now = now.astimezone(timezone.utc)
        requested_at = self.requested_at.astimezone(timezone.utc)
        age = normalized_now - requested_at
        return bool(
            self.competition_id
            and self.manifest_digest
            and self.request_nonce
            and -_COMPETITION_REQUEST_FUTURE_TOLERANCE
            <= age
            <= _COMPETITION_REQUEST_MAX_AGE
        )
