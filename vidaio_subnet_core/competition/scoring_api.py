"""HTTP contracts and client for batched competition compression scoring."""

from __future__ import annotations

from decimal import Decimal
from pathlib import PurePosixPath
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import CompetitionManifest, VOLUME_NAME_PATTERN
from .dataset import CompressionEvaluationIndexItem, EvaluationIndexItem
from .scoring import AggregateResult, ScoredItem
from .timeouts import competition_scoring_timeout_seconds


class CompetitionScoringBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item: CompressionEvaluationIndexItem | EvaluationIndexItem
    output_path: str
    runtime_seconds: float = Field(ge=0)
    allocated_gpu_type: str = Field(min_length=1, max_length=64)
    allocated_gpu_count: int = Field(gt=0)
    allocated_cpu_cores: float = Field(gt=0)

    @field_validator("output_path")
    @classmethod
    def validate_output_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("output_path must be relative and cannot traverse")
        if path.parts[0] != "evaluations":
            raise ValueError("competition output_path must be below evaluations/")
        return str(path)


class CompetitionScoringBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: CompetitionManifest
    modal_environment: str = Field(min_length=1, max_length=64)
    input_volume_name: str
    output_volume_name: str
    items: tuple[CompetitionScoringBatchItem, ...] = Field(
        min_length=1,
        max_length=5,
    )

    @field_validator("input_volume_name", "output_volume_name")
    @classmethod
    def validate_volume_name(cls, value: str) -> str:
        if not VOLUME_NAME_PATTERN.fullmatch(value):
            raise ValueError("invalid Modal volume name")
        return value

    @model_validator(mode="after")
    def validate_items(self) -> "CompetitionScoringBatchRequest":
        if self.input_volume_name != self.manifest.evaluation_input_volume_name:
            raise ValueError("input volume must match the competition manifest")
        evaluation_ids = [value.item.evaluation_id for value in self.items]
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise ValueError("competition scoring evaluation IDs must be unique")
        return self


class CompetitionScoredItemPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_checksum: str
    output_checksum: str
    input_size_bytes: int
    output_size_bytes: int
    vmaf_score: float | None
    compression_ratio: float | None
    media_score: float | None = None
    media_compression_component: float | None = None
    media_vmaf_component: float | None = None
    media_score_reason: str | None = None
    runtime_seconds: float
    estimated_cost_usd: Decimal
    cost_attribution_method: str

    @classmethod
    def from_scored_item(cls, value: ScoredItem) -> "CompetitionScoredItemPayload":
        return cls(**value.__dict__)

    def to_scored_item(self) -> ScoredItem:
        return ScoredItem(**self.model_dump())


class CompetitionScoringBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_id: str
    status: Literal["SCORED", "FAILED"]
    reason_code: str | None = None
    retryable: bool = False
    metrics: CompetitionScoredItemPayload | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "CompetitionScoringBatchResult":
        if self.status == "SCORED":
            if self.metrics is None or self.reason_code is not None or self.retryable:
                raise ValueError("scored results require metrics and no failure fields")
        elif not self.reason_code:
            raise ValueError("failed results require a reason_code")
        if self.status == "FAILED" and self.retryable:
            raise ValueError("failed results cannot be marked retryable")
        return self


class CompetitionScoringBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[CompetitionScoringBatchResult, ...]


class CompetitionAggregateHistory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    hotkey: str
    evaluation_id: str
    status: Literal["SCORED", "FAILED"]
    length_weight: float
    vmaf_threshold: float | None = None
    vmaf_score: float | None = None
    compression_rate: float | None = None
    media_score: float | None = None
    media_compression_component: float | None = None
    media_vmaf_component: float | None = None
    estimated_cost_usd: Decimal | None = None
    reconciled_cost_usd: Decimal | None = None


class CompetitionAggregateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: CompetitionManifest
    histories: tuple[CompetitionAggregateHistory, ...] = Field(min_length=1)


class CompetitionAggregatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hotkey: str
    media_score_aggregate: float
    quality_aggregate: float
    cost_efficiency_aggregate: float
    length_coverage: float
    final_score: float
    average_vmaf: float
    average_compression_ratio: float
    estimated_cost_usd: Decimal
    successful_items: int
    failed_items: int

    @classmethod
    def from_aggregate(cls, value: AggregateResult) -> "CompetitionAggregatePayload":
        return cls(
            **value.__dict__,
            quality_aggregate=value.media_score_aggregate,
        )


class CompetitionHistoryComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    history_id: int
    media_score: float
    compression: float
    vmaf_quality: float
    cost_efficiency: float
    completion: float


class CompetitionAggregateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    aggregates: tuple[CompetitionAggregatePayload, ...]
    components: tuple[CompetitionHistoryComponent, ...]


class CompetitionScoringClient:
    """Async client used by the competition validator process."""

    def __init__(
        self,
        base_url: str,
        *,
        modal_environment: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.modal_environment = modal_environment
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(base_url=base_url)

    async def score_batch(
        self,
        manifest: CompetitionManifest,
        *,
        output_volume_name: str,
        items: tuple[CompetitionScoringBatchItem, ...],
    ) -> tuple[CompetitionScoringBatchResult, ...]:
        request = CompetitionScoringBatchRequest(
            manifest=manifest,
            modal_environment=self.modal_environment,
            input_volume_name=manifest.evaluation_input_volume_name,
            output_volume_name=output_volume_name,
            items=items,
        )
        timeout = competition_scoring_timeout_seconds(
            (value.item for value in items),
            minimum_timeout_seconds=(
                manifest.scoring_batched_run_timeout.total_seconds()
            ),
        )
        response = await self.client.post(
            "/score_compression_competition",
            json=request.model_dump(mode="json"),
            timeout=timeout,
        )
        response.raise_for_status()
        payload = CompetitionScoringBatchResponse.model_validate(response.json())
        expected = [value.item.evaluation_id for value in items]
        actual = [value.evaluation_id for value in payload.results]
        if actual != expected:
            raise RuntimeError(
                "competition scoring response items do not match the request"
            )
        return payload.results

    async def score_aggregates(
        self,
        manifest: CompetitionManifest,
        histories: tuple[CompetitionAggregateHistory, ...],
    ) -> CompetitionAggregateResponse:
        request = CompetitionAggregateRequest(
            manifest=manifest,
            histories=histories,
        )
        response = await self.client.post(
            "/score_compression_competition_aggregates",
            json=request.model_dump(mode="json"),
            timeout=300.0,
        )
        response.raise_for_status()
        return CompetitionAggregateResponse.model_validate(response.json())

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()


__all__ = [
    "CompetitionAggregateHistory",
    "CompetitionAggregatePayload",
    "CompetitionAggregateRequest",
    "CompetitionAggregateResponse",
    "CompetitionHistoryComponent",
    "CompetitionScoredItemPayload",
    "CompetitionScoringBatchItem",
    "CompetitionScoringBatchRequest",
    "CompetitionScoringBatchResponse",
    "CompetitionScoringBatchResult",
    "CompetitionScoringClient",
]
