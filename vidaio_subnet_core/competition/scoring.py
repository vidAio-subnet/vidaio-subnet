"""Deterministic compression validation, item scoring, and aggregation."""

from __future__ import annotations

import hashlib
import math
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterable

from services.scoring.scoring_function import calculate_compression_score

from .config import CompetitionManifest, calculate_length_weight
from .dataset import EvaluationIndexItem
from .media_contracts import CompetitionScoringMedia
from .pricing import (
    COST_ATTRIBUTION_METHOD,
    GPU_PRICE_PER_SECOND_USD,
    SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD,
    estimate_sandbox_cost,
)
from .qualification import FfprobeMediaInspector, QualificationError


class ItemScoringError(RuntimeError):
    def __init__(
        self,
        reason_code: str,
        detail: str,
        *,
        metrics: ScoredItem | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.metrics = metrics
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True)
class ScoredItem:
    input_checksum: str
    output_checksum: str
    input_size_bytes: int
    output_size_bytes: int
    vmaf_score: float | None
    compression_ratio: float | None
    runtime_seconds: float
    estimated_cost_usd: Decimal
    media_score: float | None = None
    media_compression_component: float | None = None
    media_vmaf_component: float | None = None
    media_score_reason: str | None = None
    cost_attribution_method: str = COST_ATTRIBUTION_METHOD


def estimate_item_cost(
    runtime_seconds: float,
    *,
    allocated_gpu_type: str,
    allocated_gpu_count: int,
    allocated_cpu_cores: float,
) -> Decimal:
    try:
        return estimate_sandbox_cost(
            allocated_gpu_type=allocated_gpu_type,
            allocated_gpu_count=allocated_gpu_count,
            allocated_cpu_cores=allocated_cpu_cores,
            runtime_seconds=runtime_seconds,
        )
    except ValueError as exc:
        raise ItemScoringError("COST_RATE_MISSING", str(exc)) from exc


def _default_vmaf(
    source: Path,
    output: Path,
    sample_count: int,
    frame_count: int,
    scoring_seed: int,
) -> float:
    from services.scoring.vmaf_metric import vmaf_metric_ffmpeg

    n_subsample = max(1, frame_count // max(1, sample_count))
    deterministic_offset = scoring_seed % n_subsample
    return float(
        vmaf_metric_ffmpeg(
            str(output),
            str(source),
            skip_frames=deterministic_offset,
            n_subsample=n_subsample,
            neg_model=True,
        )
    )


class CompetitionItemScorer:
    competition_type = "COMPRESSION"
    required_route = "/compress"

    def __init__(
        self,
        *,
        inspector: FfprobeMediaInspector | None = None,
        vmaf: Callable[[Path, Path, int, int, int], float] = _default_vmaf,
    ) -> None:
        self.inspector = inspector or FfprobeMediaInspector()
        self.vmaf = vmaf

    def score_media(
        self,
        manifest: CompetitionManifest,
        item: EvaluationIndexItem,
        media: CompetitionScoringMedia,
        *,
        runtime_seconds: float,
        allocated_gpu_type: str,
        allocated_gpu_count: int,
        allocated_cpu_cores: float,
    ) -> ScoredItem:
        """Score the active compression media contract.

        Compression has no separate ground-truth asset: its immutable reference
        is both the contender input and quality reference. A non-null ground
        truth therefore fails closed until an upscaling adapter is activated.
        """

        if media.ground_truth_video is not None:
            raise ItemScoringError(
                "COMPETITION_MEDIA_CONTRACT_MISMATCH",
                "compression scoring does not accept a separate ground truth",
            )
        return self.score(
            manifest,
            item,
            media.reference_video,
            media.miner_processed_video,
            runtime_seconds=runtime_seconds,
            allocated_gpu_type=allocated_gpu_type,
            allocated_gpu_count=allocated_gpu_count,
            allocated_cpu_cores=allocated_cpu_cores,
        )

    def score(
        self,
        manifest: CompetitionManifest,
        item: EvaluationIndexItem,
        source_bytes: bytes,
        output_bytes: bytes,
        *,
        runtime_seconds: float,
        allocated_gpu_type: str,
        allocated_gpu_count: int,
        allocated_cpu_cores: float,
    ) -> ScoredItem:
        source_checksum = hashlib.sha256(source_bytes).hexdigest()
        output_checksum = hashlib.sha256(output_bytes).hexdigest()
        compression_ratio = (
            len(source_bytes) / len(output_bytes) if output_bytes else None
        )
        estimated_cost = estimate_item_cost(
            runtime_seconds,
            allocated_gpu_type=allocated_gpu_type,
            allocated_gpu_count=allocated_gpu_count,
            allocated_cpu_cores=allocated_cpu_cores,
        )

        def measured(
            vmaf_score: float | None = None,
            *,
            media_score: float | None = None,
            media_compression_component: float | None = None,
            media_vmaf_component: float | None = None,
            media_score_reason: str | None = None,
        ) -> ScoredItem:
            return ScoredItem(
                input_checksum=source_checksum,
                output_checksum=output_checksum,
                input_size_bytes=len(source_bytes),
                output_size_bytes=len(output_bytes),
                vmaf_score=vmaf_score,
                compression_ratio=compression_ratio,
                runtime_seconds=runtime_seconds,
                estimated_cost_usd=estimated_cost,
                media_score=media_score,
                media_compression_component=media_compression_component,
                media_vmaf_component=media_vmaf_component,
                media_score_reason=media_score_reason,
            )

        def reject(
            reason_code: str,
            detail: str,
            *,
            vmaf_score: float | None = None,
            media_score: float | None = None,
            media_compression_component: float | None = None,
            media_vmaf_component: float | None = None,
            media_score_reason: str | None = None,
        ) -> None:
            raise ItemScoringError(
                reason_code,
                detail,
                metrics=measured(
                    vmaf_score,
                    media_score=media_score,
                    media_compression_component=media_compression_component,
                    media_vmaf_component=media_vmaf_component,
                    media_score_reason=media_score_reason,
                ),
            )

        if len(source_bytes) != item.size_bytes:
            reject("INPUT_SIZE_MISMATCH", item.evaluation_id)
        if source_checksum != item.sha256:
            reject("INPUT_CHECKSUM_MISMATCH", item.evaluation_id)
        if not output_bytes:
            reject("OUTPUT_EMPTY", item.evaluation_id)

        with tempfile.TemporaryDirectory(prefix="competition-score-") as temp:
            root = Path(temp)
            source = root / "source.mp4"
            output = root / "output.mp4"
            source.write_bytes(source_bytes)
            output.write_bytes(output_bytes)
            try:
                source_media = self.inspector.inspect(source)
            except QualificationError as exc:
                raise ItemScoringError(
                    exc.reason_code, str(exc), metrics=measured()
                ) from exc
            try:
                output_media = self.inspector.inspect(output)
            except QualificationError as exc:
                raise ItemScoringError(
                    "OUTPUT_MEDIA_INVALID", str(exc), metrics=measured()
                ) from exc

            if (source_media.width, source_media.height) != (item.width, item.height):
                reject("INPUT_DIMENSIONS_MISMATCH", item.evaluation_id)
            if source_media.frame_count != item.frame_count:
                reject("INPUT_FRAME_COUNT_MISMATCH", item.evaluation_id)
            source_duration_tolerance = max(0.2, item.duration_seconds * 0.005)
            if (
                abs(source_media.duration_seconds - item.duration_seconds)
                > source_duration_tolerance
            ):
                reject("INPUT_DURATION_MISMATCH", item.evaluation_id)
            if source_media.codec != item.codec:
                reject("INPUT_CODEC_MISMATCH", item.evaluation_id)
            if source_media.pixel_format != item.pixel_format:
                reject("INPUT_PIXEL_FORMAT_MISMATCH", item.evaluation_id)
            if source_media.sample_aspect_ratio != item.sample_aspect_ratio:
                reject("INPUT_ASPECT_RATIO_MISMATCH", item.evaluation_id)

            violations: list[tuple[str, str]] = []
            if len(output_bytes) >= len(source_bytes):
                violations.append(("OUTPUT_NOT_SMALLER", item.evaluation_id))
            if output_media.codec.lower() != "av1":
                violations.append(("OUTPUT_CODEC_INVALID", output_media.codec))
            if "mp4" not in output_media.container.lower():
                violations.append(("OUTPUT_CONTAINER_INVALID", output_media.container))
            dimensions_match = (output_media.width, output_media.height) == (
                source_media.width,
                source_media.height,
            )
            if not dimensions_match:
                violations.append(("OUTPUT_DIMENSIONS_CHANGED", item.evaluation_id))
            duration_tolerance = max(0.5, source_media.duration_seconds * 0.01)
            duration_matches = (
                abs(output_media.duration_seconds - source_media.duration_seconds)
                <= duration_tolerance
            )
            if not duration_matches:
                violations.append(("OUTPUT_DURATION_CHANGED", item.evaluation_id))
            if not output_media.pixel_format.lower().startswith("yuv"):
                violations.append(
                    ("OUTPUT_PIXEL_FORMAT_INVALID", output_media.pixel_format)
                )
            if output_media.sample_aspect_ratio != source_media.sample_aspect_ratio:
                violations.append(("OUTPUT_ASPECT_RATIO_CHANGED", item.evaluation_id))
            frame_tolerance = max(1, round(item.frame_count * 0.01))
            if (
                output_media.frame_count is None
                or abs(output_media.frame_count - item.frame_count) > frame_tolerance
            ):
                violations.append(("OUTPUT_FRAME_COUNT_CHANGED", item.evaluation_id))

            assert compression_ratio is not None
            if compression_ratio < manifest.minimum_compression_ratio:
                violations.append(
                    ("COMPRESSION_RATIO_TOO_LOW", f"{compression_ratio:.6f}")
                )

            vmaf_score = None
            if dimensions_match and duration_matches:
                try:
                    vmaf_score = self.vmaf(
                        source,
                        output,
                        manifest.vmaf_sample_count,
                        item.frame_count,
                        manifest.scoring_seed,
                    )
                except Exception:
                    if not violations:
                        raise
            if violations:
                reason_code, detail = violations[0]
                reject(reason_code, detail, vmaf_score=vmaf_score)
            assert vmaf_score is not None
            required_vmaf = float(
                getattr(item, "vmaf_threshold", manifest.vmaf_threshold)
            )
            if not math.isfinite(vmaf_score):
                reject(
                    "VMAF_INVALID",
                    f"{vmaf_score:.6f}",
                    vmaf_score=vmaf_score,
                )
            (
                media_score,
                media_compression_component,
                media_vmaf_component,
                media_score_reason,
            ) = calculate_compression_score(
                vmaf_score=vmaf_score,
                compression_rate=len(output_bytes) / len(source_bytes),
                vmaf_threshold=required_vmaf,
            )
            if media_score <= 0:
                reject(
                    "MEDIA_SCORE_ZERO",
                    media_score_reason,
                    vmaf_score=vmaf_score,
                    media_score=media_score,
                    media_compression_component=media_compression_component,
                    media_vmaf_component=media_vmaf_component,
                    media_score_reason=media_score_reason,
                )

        return measured(
            vmaf_score,
            media_score=media_score,
            media_compression_component=media_compression_component,
            media_vmaf_component=media_vmaf_component,
            media_score_reason=media_score_reason,
        )


@dataclass(frozen=True)
class AggregateResult:
    hotkey: str
    media_score_aggregate: float
    cost_efficiency_aggregate: float
    length_coverage: float
    final_score: float
    average_vmaf: float
    average_compression_ratio: float
    estimated_cost_usd: Decimal
    successful_items: int
    failed_items: int

    @property
    def quality_aggregate(self) -> float:
        """Compatibility name for the absolute media-score aggregate."""

        return self.media_score_aggregate


def length_weight(duration_seconds: float, manifest: CompetitionManifest) -> float:
    return calculate_length_weight(
        duration_seconds,
        manifest.max_video_length.total_seconds(),
        manifest.length_weight_exponent,
    )


def compute_aggregates(
    manifest: CompetitionManifest,
    histories: Iterable[object],
) -> tuple[
    list[AggregateResult],
    dict[int, tuple[float, float, float, float, float]],
]:
    rows = list(histories)
    terminal = [row for row in rows if row.status in {"SCORED", "FAILED"}]
    by_hotkey: dict[str, list[object]] = {}
    for row in terminal:
        by_hotkey.setdefault(row.hotkey, []).append(row)

    components: dict[int, tuple[float, float, float, float, float]] = {}
    valid_media: dict[int, tuple[float, float, float]] = {}
    effective_costs: dict[int, Decimal] = {}
    minimum_costs: dict[str, Decimal] = {}
    for row in terminal:
        if (
            row.status != "SCORED"
            or row.vmaf_score is None
            or row.compression_rate is None
            or float(row.compression_rate) <= 0
            or float(row.compression_rate) < manifest.minimum_compression_ratio
        ):
            components[row.id] = (0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        (
            media_score,
            compression_component,
            vmaf_component,
            _reason,
        ) = calculate_compression_score(
            vmaf_score=float(row.vmaf_score),
            compression_rate=1.0 / float(row.compression_rate),
            vmaf_threshold=float(
                getattr(row, "vmaf_threshold", None) or manifest.vmaf_threshold
            ),
        )
        if media_score <= 0:
            components[row.id] = (0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        cost = max(
            Decimal(str(row.reconciled_cost_usd or row.estimated_cost_usd or 0)),
            manifest.cost_floor_usd,
        )
        valid_media[row.id] = (
            float(media_score),
            float(compression_component),
            float(vmaf_component),
        )
        effective_costs[row.id] = cost
        prior_minimum = minimum_costs.get(row.evaluation_id)
        if prior_minimum is None or cost < prior_minimum:
            minimum_costs[row.evaluation_id] = cost

    for row in terminal:
        media = valid_media.get(row.id)
        if media is None:
            continue
        cost_efficiency = min(
            1.0,
            float(minimum_costs[row.evaluation_id] / effective_costs[row.id]),
        )
        components[row.id] = (
            media[0],
            media[1],
            media[2],
            cost_efficiency,
            1.0,
        )

    aggregates = []
    for hotkey, contender_rows in by_hotkey.items():
        total_weight = sum(float(row.length_weight) for row in contender_rows)
        if total_weight <= 0:
            raise ValueError(f"non-positive evaluation weight for {hotkey}")
        media_score = (
            sum(
                float(row.length_weight) * components[row.id][0]
                for row in contender_rows
            )
            / total_weight
        )
        cost = (
            sum(
                float(row.length_weight) * components[row.id][3]
                for row in contender_rows
            )
            / total_weight
        )
        coverage = (
            sum(
                float(row.length_weight) * components[row.id][4]
                for row in contender_rows
            )
            / total_weight
        )
        successful = [row for row in contender_rows if row.status == "SCORED"]
        final = (
            float(manifest.scoring_factors.media_score) * media_score
            + float(manifest.scoring_factors.cost_efficiency) * cost
            + float(manifest.scoring_factors.length_coverage) * coverage
        )
        aggregates.append(
            AggregateResult(
                hotkey=hotkey,
                media_score_aggregate=media_score,
                cost_efficiency_aggregate=cost,
                length_coverage=coverage,
                final_score=round(final, manifest.score_precision),
                average_vmaf=(
                    sum(float(row.vmaf_score) for row in successful) / len(successful)
                    if successful
                    else 0.0
                ),
                average_compression_ratio=(
                    sum(float(row.compression_rate) for row in successful)
                    / len(successful)
                    if successful
                    else 0.0
                ),
                estimated_cost_usd=sum(
                    (
                        Decimal(str(row.estimated_cost_usd or 0))
                        for row in contender_rows
                    ),
                    Decimal(0),
                ),
                successful_items=len(successful),
                failed_items=len(contender_rows) - len(successful),
            )
        )
    return aggregates, components


__all__ = [
    "AggregateResult",
    "CompetitionItemScorer",
    "COST_ATTRIBUTION_METHOD",
    "GPU_PRICE_PER_SECOND_USD",
    "ItemScoringError",
    "SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD",
    "ScoredItem",
    "compute_aggregates",
    "estimate_item_cost",
    "length_weight",
]
