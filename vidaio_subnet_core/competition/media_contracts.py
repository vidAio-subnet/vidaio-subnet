"""Task-neutral media roles and inactive future competition contracts.

Compression currently has two physical media files: the immutable source is both
the contender input and quality reference, and the contender produces an output.
Future upscaling needs three physical files and must not conflate their roles:
the high-resolution ground truth, its downsampled reference (the contender input),
and the contender's processed output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CompetitionMediaRole(str, Enum):
    GROUND_TRUTH = "GROUND_TRUTH"
    REFERENCE = "REFERENCE"
    MINER_PROCESSED = "MINER_PROCESSED"


class IndexedMediaAsset(BaseModel):
    """Immutable metadata for one trusted dataset asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    codec: str = Field(min_length=1, max_length=32)
    pixel_format: str = Field(min_length=1, max_length=32)
    sample_aspect_ratio: str = Field(min_length=1, max_length=32)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("media path must be relative and cannot traverse")
        return str(path)

    def sandbox_path(self, mount_root: str = "/evaluation-inputs") -> str:
        return f"{mount_root.rstrip('/')}/{self.relative_path}"


class UpscalingEvaluationIndexItemStub(BaseModel):
    """Reserved three-file dataset contract for a future upscaling adapter.

    This type is intentionally not accepted by ``CompetitionManifest`` or the
    live dispatcher yet. It fixes the future data roles before implementation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    evaluation_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"
    )
    ground_truth: IndexedMediaAsset
    reference: IndexedMediaAsset

    @model_validator(mode="after")
    def validate_asset_relationship(self) -> "UpscalingEvaluationIndexItemStub":
        if self.ground_truth.relative_path == self.reference.relative_path:
            raise ValueError("ground truth and downsampled reference must be distinct")
        if (
            self.ground_truth.width < self.reference.width
            or self.ground_truth.height < self.reference.height
        ):
            raise ValueError("ground truth cannot be smaller than its reference")
        if (
            self.ground_truth.width == self.reference.width
            and self.ground_truth.height == self.reference.height
        ):
            raise ValueError("upscaling reference must be spatially downsampled")
        duration_tolerance = max(0.2, self.ground_truth.duration_seconds * 0.01)
        if (
            abs(self.ground_truth.duration_seconds - self.reference.duration_seconds)
            > duration_tolerance
        ):
            raise ValueError("ground truth and reference durations must match")
        if self.ground_truth.frame_count != self.reference.frame_count:
            raise ValueError("ground truth and reference frame counts must match")
        if self.ground_truth.sample_aspect_ratio != self.reference.sample_aspect_ratio:
            raise ValueError("ground truth and reference aspect ratios must match")
        return self

    @property
    def contender_input_path(self) -> str:
        """The miner receives only the downsampled reference path."""

        return self.reference.sandbox_path()

    @property
    def trusted_quality_reference_path(self) -> str:
        """The future trusted scorer compares output against ground truth."""

        return self.ground_truth.sandbox_path("/evaluation-ground-truth")


@dataclass(frozen=True)
class CompetitionScoringMedia:
    """Explicit scorer inputs shared by current and future task adapters."""

    reference_video: bytes
    miner_processed_video: bytes
    ground_truth_video: bytes | None = None

    @classmethod
    def compression(
        cls, reference_video: bytes, miner_processed_video: bytes
    ) -> "CompetitionScoringMedia":
        return cls(
            reference_video=reference_video,
            miner_processed_video=miner_processed_video,
        )

    @classmethod
    def upscaling(
        cls,
        *,
        ground_truth_video: bytes,
        downsampled_reference_video: bytes,
        miner_processed_video: bytes,
    ) -> "CompetitionScoringMedia":
        return cls(
            ground_truth_video=ground_truth_video,
            reference_video=downsampled_reference_video,
            miner_processed_video=miner_processed_video,
        )


@runtime_checkable
class CompetitionTaskAdapter(Protocol):
    """Extension boundary used when another competition type is activated."""

    competition_type: str
    required_route: str

    def score_media(
        self,
        manifest: object,
        item: object,
        media: CompetitionScoringMedia,
        *,
        runtime_seconds: float,
        allocated_gpu_type: str,
        allocated_gpu_count: int,
        allocated_cpu_cores: float,
    ): ...


class UpscalingCompetitionAdapterStub:
    """Non-active placeholder; activation requires its own manifest and scorer."""

    competition_type = "UPSCALING"
    required_route = "/upscale"
    index_item_type = UpscalingEvaluationIndexItemStub

    def score_media(
        self,
        manifest: object,
        item: UpscalingEvaluationIndexItemStub,
        media: CompetitionScoringMedia,
        *,
        runtime_seconds: float,
        allocated_gpu_type: str,
        allocated_gpu_count: int,
        allocated_cpu_cores: float,
    ):
        del (
            manifest,
            item,
            media,
            runtime_seconds,
            allocated_gpu_type,
            allocated_gpu_count,
            allocated_cpu_cores,
        )
        raise NotImplementedError(
            "upscaling competition execution is reserved but not implemented"
        )


__all__ = [
    "CompetitionMediaRole",
    "CompetitionScoringMedia",
    "CompetitionTaskAdapter",
    "IndexedMediaAsset",
    "UpscalingCompetitionAdapterStub",
    "UpscalingEvaluationIndexItemStub",
]
