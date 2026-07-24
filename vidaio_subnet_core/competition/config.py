"""Versioned competition manifest and runtime configuration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


COMPETITION_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?$")
VOLUME_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
ALLOWED_GPU_TYPES = frozenset({"L4", "L40S", "RTX-PRO-6000"})


def parse_duration(value: Any) -> timedelta:
    if isinstance(value, timedelta):
        return value
    if not isinstance(value, str):
        raise TypeError("duration must be a timedelta or a string such as '30m'")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", value)
    if match:
        amount = Decimal(match.group(1))
        multiplier = {
            "s": Decimal(1),
            "m": Decimal(60),
            "h": Decimal(3600),
            "d": Decimal(86400),
        }[match.group(2)]
        seconds = amount * multiplier
    else:
        # Pydantic serializes timedeltas in normalized manifests as ISO-8601.
        iso_match = re.fullmatch(
            r"P(?:(\d+(?:\.\d+)?)D)?"
            r"(?:T(?:(\d+(?:\.\d+)?)H)?"
            r"(?:(\d+(?:\.\d+)?)M)?"
            r"(?:(\d+(?:\.\d+)?)S)?)?",
            value,
            flags=re.IGNORECASE,
        )
        if not iso_match or not any(iso_match.groups()):
            raise ValueError(
                "duration must use s/m/h/d shorthand or an ISO-8601 duration"
            )
        days, hours, minutes, iso_seconds = (
            Decimal(part or "0") for part in iso_match.groups()
        )
        seconds = (
            days * Decimal(86400)
            + hours * Decimal(3600)
            + minutes * Decimal(60)
            + iso_seconds
        )
    if seconds <= 0:
        raise ValueError("duration must be positive")
    return timedelta(seconds=float(seconds))


def calculate_length_weight(
    duration_seconds: float,
    max_video_length_seconds: float,
    length_weight_exponent: float = 1.0,
) -> float:
    """Return the manifest-defined normalized weight for one video duration."""

    values = (
        duration_seconds,
        max_video_length_seconds,
        length_weight_exponent,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("length weight inputs must be finite numbers")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if max_video_length_seconds <= 0:
        raise ValueError("max_video_length_seconds must be positive")
    if duration_seconds > max_video_length_seconds:
        raise ValueError("duration_seconds exceeds max_video_length_seconds")
    if not 0 < length_weight_exponent <= 10:
        raise ValueError("length_weight_exponent must be in (0, 10]")
    base_weight = math.log1p(duration_seconds) / math.log1p(
        max_video_length_seconds
    )
    return base_weight**length_weight_exponent


class BossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_path: Path | None = None
    boss_hotkey: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("repository_path")
    @classmethod
    def validate_repository_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        if value.is_absolute() or value == Path(".") or ".." in value.parts:
            raise ValueError(
                "boss repository_path must be a non-root path relative to the "
                "validator repository"
            )
        return value

    @field_validator("boss_hotkey")
    @classmethod
    def validate_boss_hotkey(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("boss_hotkey cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_boss_identity(self) -> "BossConfig":
        if (self.repository_path is None) != (self.boss_hotkey is None):
            raise ValueError(
                "boss repository_path and boss_hotkey must either both be set "
                "or both be null"
            )
        return self


class ScoringFactors(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quality: Decimal = Decimal("0.6")
    cost_efficiency: Decimal = Decimal("0.25")
    length_coverage: Decimal = Decimal("0.15")
    runtime: Decimal = Decimal("0")

    @property
    def media_score(self) -> Decimal:
        """Absolute media-score weight; `quality` is the legacy manifest key."""

        return self.quality

    @model_validator(mode="after")
    def validate_weights(self) -> "ScoringFactors":
        values = (
            self.quality,
            self.cost_efficiency,
            self.length_coverage,
            self.runtime,
        )
        if any(not value.is_finite() for value in values):
            raise ValueError("scoring factors must be finite")
        if any(value < 0 for value in values):
            raise ValueError("scoring factors cannot be negative")
        if self.runtime != 0:
            raise ValueError(
                "runtime must have zero direct score weight because no runtime "
                "score component is defined"
            )
        if abs(sum(values) - Decimal(1)) > Decimal("1e-12"):
            raise ValueError("scoring factors must sum to one")
        return self


class CompetitionManifest(BaseModel):
    """Immutable, normalized source of truth for one competition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    scoring_version: str = Field(
        default="3",
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    competition_id: str
    competition_type: Literal["COMPRESSION"] = "COMPRESSION"
    competition_start_time: datetime
    contender_ping_interval: timedelta
    contender_finalisation_time: datetime
    human_review_deadline: datetime
    competition_end_time: datetime

    required_routes: tuple[str, ...] = ("/compress",)
    allowed_gpus: tuple[str, ...]
    max_cpu_cores: int = Field(default=32, ge=1, le=32)
    requested_cpu_cores: int = Field(ge=1, le=32)
    container_size_limit_gb: Literal[25] = 25
    modal_build_timeout: timedelta = Field(default=timedelta(minutes=10))
    evaluation_batch_size: int = Field(default=5, ge=1, le=5)
    evaluation_batched_run_timeout: timedelta
    scoring_batched_run_timeout: timedelta = Field(default=timedelta(minutes=5))
    min_video_length: timedelta
    max_video_length: timedelta
    length_weight_exponent: float = Field(
        default=1.0,
        gt=0,
        le=10,
        allow_inf_nan=False,
    )
    required_output_codec: Literal["AV1"] = "AV1"
    vmaf_threshold: float = Field(ge=0, le=100)
    vmaf_sample_count: int = Field(gt=0)
    minimum_compression_ratio: float = Field(default=1.25, ge=1, le=100)
    scoring_seed: int
    warmup_input_path: Path
    boss: BossConfig = Field(default_factory=BossConfig)
    evaluation_input_volume_name: str
    evaluation_reference_volume_name: str | None = None
    evaluation_index_path: str
    output_volume_prefix: str
    max_parallel_contenders: int = Field(default=4, ge=1, le=32)
    max_attempts_per_item: int = Field(default=2, ge=1, le=10)
    scoring_factors: ScoringFactors = Field(default_factory=ScoringFactors)
    cost_floor_usd: Decimal = Field(default=Decimal("0.000001"), gt=0)
    score_precision: int = Field(default=8, ge=1, le=12)

    @field_validator(
        "contender_ping_interval",
        "modal_build_timeout",
        "evaluation_batched_run_timeout",
        "scoring_batched_run_timeout",
        "min_video_length",
        "max_video_length",
        mode="before",
    )
    @classmethod
    def parse_manifest_duration(cls, value: Any) -> timedelta:
        return parse_duration(value)

    @field_validator(
        "competition_start_time",
        "contender_finalisation_time",
        "human_review_deadline",
        "competition_end_time",
    )
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("competition timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("competition_id")
    @classmethod
    def validate_competition_id(cls, value: str) -> str:
        if not COMPETITION_ID_PATTERN.fullmatch(value):
            raise ValueError("competition_id is not a valid lowercase slug")
        return value

    @field_validator(
        "evaluation_input_volume_name",
        "output_volume_prefix",
    )
    @classmethod
    def validate_volume_name(cls, value: str) -> str:
        if not VOLUME_NAME_PATTERN.fullmatch(value):
            raise ValueError("invalid Modal volume name")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "CompetitionManifest":
        if self.required_routes != ("/compress",):
            raise ValueError("compression competitions require exactly /compress")
        if self.competition_start_time.weekday() != 3:
            raise ValueError("competition_start_time must be a Thursday")
        if not (
            self.competition_start_time
            < self.contender_finalisation_time
            <= self.human_review_deadline
            < self.competition_end_time
        ):
            raise ValueError("competition timestamps are out of order")
        if self.contender_ping_interval >= (
            self.contender_finalisation_time - self.competition_start_time
        ):
            raise ValueError("contender ping interval must fit inside enrollment")
        if self.requested_cpu_cores > self.max_cpu_cores:
            raise ValueError("requested CPU exceeds the manifest hard limit")
        if not self.allowed_gpus or not set(self.allowed_gpus) <= ALLOWED_GPU_TYPES:
            raise ValueError("allowed_gpus contains an unsupported GPU")
        if self.evaluation_batched_run_timeout > timedelta(hours=24):
            raise ValueError(
                "evaluation batch timeout exceeds the platform safety ceiling"
            )
        if self.scoring_batched_run_timeout > timedelta(hours=24):
            raise ValueError("scoring timeout exceeds the platform safety ceiling")
        if self.min_video_length > self.max_video_length:
            raise ValueError("min_video_length cannot exceed max_video_length")
        if self.evaluation_reference_volume_name is not None:
            raise ValueError(
                "compression scoring uses the immutable input as its reference; "
                "evaluation_reference_volume_name must be omitted"
            )
        if not self.evaluation_index_path.startswith("/"):
            raise ValueError("evaluation_index_path must be absolute inside the Volume")
        return self

    def normalized_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return hashlib.sha256(self.normalized_json().encode("utf-8")).hexdigest()

    def validate_runtime_paths(self, repository_root: Path) -> None:
        repository_root = repository_root.resolve(strict=True)
        warmup = self.warmup_input_path
        if not warmup.is_absolute():
            warmup = repository_root / warmup
        if not warmup.is_file():
            raise ValueError(f"warmup fixture is missing: {warmup}")

        boss_path = self.boss.repository_path
        if boss_path is None:
            return
        configured_boss = repository_root / boss_path
        try:
            resolved_boss = configured_boss.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(
                f"boss SDK export directory is missing: {configured_boss}"
            ) from exc
        if repository_root not in resolved_boss.parents:
            raise ValueError(
                "boss repository_path resolves outside the validator repository"
            )
        if not resolved_boss.is_dir():
            raise ValueError(f"boss SDK export is not a directory: {resolved_boss}")

        required_paths = (
            ".vidaio-sdk-export",
            "competition_solution.json",
            "requirements.txt",
            "miner/modal_workers.py",
            "scripts/competition_modal_build.py",
            "competitions/fixtures/compression_warmup_input.mp4",
        )
        missing = [
            relative
            for relative in required_paths
            if not (resolved_boss / relative).is_file()
        ]
        if missing:
            raise ValueError(
                "boss repository_path is not a complete competition_sdk.py export; "
                f"missing: {', '.join(missing)}"
            )


class CompetitionConfig(BaseSettings):
    """Validator-owned runtime settings. Disabled unless explicitly enabled."""

    model_config = SettingsConfigDict(
        env_prefix="COMPETITION_",
        extra="ignore",
        case_sensitive=False,
    )

    mode_enabled: bool = False
    database_url: str = "sqlite:///video_subnet_validator.db"
    artifact_root: Path = Path("competition_artifacts")
    manifest_glob: str = "competitions/manifests/*.json"
    scheduler_interval_seconds: float = Field(default=30, gt=0, le=300)
    lease_ttl_seconds: int = Field(default=120, ge=10, le=3600)
    network_timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_concurrent_requests: int = Field(default=32, ge=1, le=256)
    artifact_backup_bucket: str = Field(
        default_factory=lambda: os.getenv("BUCKET_NAME", "").strip()
    )
    artifact_backup_prefix: str = "competition_artifacts"
    artifact_backup_region: str = Field(
        default_factory=lambda: os.getenv("AWS_REGION", "us-east-1").strip()
        or "us-east-1"
    )
    artifact_backup_endpoint_url: str | None = Field(
        default_factory=lambda: os.getenv("BUCKET_COMPATIBLE_ENDPOINT", "").strip()
        or None
    )
    artifact_backup_access_key_id: str | None = Field(
        default_factory=lambda: os.getenv(
            "BUCKET_COMPATIBLE_ACCESS_KEY", ""
        ).strip()
        or None
    )
    artifact_backup_secret_access_key: str | None = Field(
        default_factory=lambda: os.getenv(
            "BUCKET_COMPATIBLE_SECRET_KEY", ""
        ).strip()
        or None
    )
    execution_enabled: bool = False
    build_backend: Literal["disabled", "modal"] = "disabled"
    accept_modal_build_without_size_attestation: bool = False
    modal_environment: str = Field(default="dev", min_length=1, max_length=64)
    owner_id: str = Field(
        default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}"
    )

    @model_validator(mode="after")
    def validate_execution_settings(self) -> "CompetitionConfig":
        if self.mode_enabled and not self.artifact_backup_bucket.strip():
            raise ValueError(
                "competition mode requires COMPETITION_ARTIFACT_BACKUP_BUCKET "
                "or BUCKET_NAME"
            )
        prefix_parts = self.artifact_backup_prefix.strip("/").split("/")
        if not self.artifact_backup_prefix.strip("/") or ".." in prefix_parts:
            raise ValueError("competition artifact backup prefix is invalid")
        if not self.execution_enabled:
            return self
        if not self.mode_enabled:
            raise ValueError(
                "competition execution requires COMPETITION_MODE_ENABLED=true"
            )
        if self.build_backend == "disabled":
            raise ValueError(
                "competition execution requires an explicitly configured build backend"
            )
        if (
            self.build_backend == "modal"
            and not self.accept_modal_build_without_size_attestation
        ):
            raise ValueError(
                "Modal builds require "
                "COMPETITION_ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION=true"
            )
        return self


def load_manifest(path: Path | str) -> CompetitionManifest:
    manifest_path = Path(path)
    raw_text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        raw = json.loads(raw_text)
    elif manifest_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML competition manifests"
            ) from exc
        raw = yaml.safe_load(raw_text)
    else:
        raise ValueError("competition manifest must be JSON or YAML")
    if not isinstance(raw, dict):
        raise ValueError("competition manifest root must be an object")
    return CompetitionManifest.model_validate(raw)
