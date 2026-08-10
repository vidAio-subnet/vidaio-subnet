"""Strict, local-filesystem-only route contracts for competition contenders."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


INPUT_ROOT = PurePosixPath("/evaluation-inputs")
OUTPUT_ROOT = PurePosixPath("/output")


def _validate_sandbox_path(value: str, root: PurePosixPath) -> str:
    if not value or "://" in value or "\x00" in value:
        raise ValueError("path must be a non-empty local filesystem path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("path must be absolute and cannot contain traversal")
    if path != root and root not in path.parents:
        raise ValueError(f"path must be below {root}")
    return str(path)


class CompetitionCompressionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"
    )
    input_path: str
    output_path: str
    codec: Literal["AV1"] = "AV1"
    codec_mode: Literal["CRF", "VBR"] = "CRF"
    target_bitrate: Literal[5_000_000, 8_000_000, 10_000_000] | None = None
    vmaf_threshold: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_rate_control(self) -> "CompetitionCompressionItem":
        if self.codec_mode == "VBR" and self.target_bitrate is None:
            raise ValueError("VBR competition item requires target_bitrate")
        if self.codec_mode == "CRF" and self.target_bitrate is not None:
            raise ValueError("CRF competition item cannot set target_bitrate")
        return self

    @field_validator("input_path")
    @classmethod
    def validate_input_path(cls, value: str) -> str:
        return _validate_sandbox_path(value, INPUT_ROOT)

    @field_validator("output_path")
    @classmethod
    def validate_output_path(cls, value: str) -> str:
        normalized = _validate_sandbox_path(value, OUTPUT_ROOT)
        if not normalized.lower().endswith(".mp4"):
            raise ValueError("competition output must be an MP4 path")
        return normalized


class CompetitionCompressionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    competition_id: str = Field(min_length=1, max_length=64)
    hotkey: str = Field(min_length=1, max_length=128)
    batch_id: str = Field(min_length=1, max_length=128)
    items: tuple[CompetitionCompressionItem, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_items_and_outputs(self) -> "CompetitionCompressionRequest":
        input_ids = [item.evaluation_id for item in self.items]
        outputs = [item.output_path for item in self.items]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("evaluation_id values must be unique within a batch")
        if len(outputs) != len(set(outputs)):
            raise ValueError("output_path values must be unique within a batch")
        return self


class CompetitionCompressionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: str | None = None


class CompetitionCompressionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[CompetitionCompressionResult, ...]


__all__ = [
    "CompetitionCompressionItem",
    "CompetitionCompressionRequest",
    "CompetitionCompressionResponse",
    "CompetitionCompressionResult",
    "INPUT_ROOT",
    "OUTPUT_ROOT",
]
