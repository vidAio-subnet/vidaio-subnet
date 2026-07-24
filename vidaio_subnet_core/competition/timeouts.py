"""Workload-derived deadlines for competition execution and scoring."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from typing import Protocol


REFERENCE_4K_PIXELS = 3840 * 2160
ENCODING_CHUNK_SECONDS = 10 * 60
ENCODING_4K_CHUNK_RUNTIME_SECONDS = 2 * 60
ENCODING_PARALLEL_VIDEOS = 4
VMAF_4K_FRAMES_PER_SECOND = 200
PROCESSING_TIMEOUT_GRACE_SECONDS = 2 * 60
BATCH_LEASE_GRACE_SECONDS = 2 * 60


class VideoWorkload(Protocol):
    duration_seconds: float
    width: int
    height: int
    frame_count: int


def _items(values: Iterable[VideoWorkload]) -> tuple[VideoWorkload, ...]:
    items = tuple(values)
    if not items:
        raise ValueError("at least one video is required to calculate a timeout")
    return items


def _resolution_scale(item: VideoWorkload) -> float:
    return (item.width * item.height) / REFERENCE_4K_PIXELS


def estimated_encoding_runtime_seconds(
    values: Iterable[VideoWorkload],
) -> float:
    """Estimate a four-lane NVENC chunk schedule from duration and resolution."""

    video_runtimes = []
    for item in _items(values):
        scale = _resolution_scale(item)
        chunk_count = math.ceil(item.duration_seconds / ENCODING_CHUNK_SECONDS)
        video_runtime = max(
            ENCODING_4K_CHUNK_RUNTIME_SECONDS,
            chunk_count
            * ENCODING_4K_CHUNK_RUNTIME_SECONDS
            * scale,
        )
        video_runtimes.append(video_runtime)

    lane_loads = [0.0] * min(
        ENCODING_PARALLEL_VIDEOS,
        len(video_runtimes),
    )
    heapq.heapify(lane_loads)
    for runtime in sorted(video_runtimes, reverse=True):
        lightest = heapq.heappop(lane_loads)
        heapq.heappush(lane_loads, lightest + runtime)
    return max(lane_loads)


def competition_execution_timeout_seconds(
    values: Iterable[VideoWorkload],
    *,
    minimum_timeout_seconds: float = 0,
) -> int:
    """Return the miner deadline, never below the manifest's batch minimum."""

    estimate = estimated_encoding_runtime_seconds(values)
    return math.ceil(
        max(
            estimate + PROCESSING_TIMEOUT_GRACE_SECONDS,
            minimum_timeout_seconds,
        )
    )


def competition_execution_lease_seconds(
    values: Iterable[VideoWorkload],
    *,
    minimum_timeout_seconds: float = 0,
) -> int:
    """Return the database lease, including post-invocation recovery grace."""

    return (
        competition_execution_timeout_seconds(
            values,
            minimum_timeout_seconds=minimum_timeout_seconds,
        )
        + BATCH_LEASE_GRACE_SECONDS
    )


def estimated_vmaf_runtime_seconds(values: Iterable[VideoWorkload]) -> float:
    """Estimate sequential VMAF work at 200 FPS for 4K frames."""

    return sum(
        item.frame_count / VMAF_4K_FRAMES_PER_SECOND * _resolution_scale(item)
        for item in _items(values)
    )


def competition_scoring_timeout_seconds(
    values: Iterable[VideoWorkload],
    *,
    minimum_timeout_seconds: float = 0,
) -> int:
    """Return the scorer deadline, never below the manifest batch minimum."""

    estimate = estimated_vmaf_runtime_seconds(values)
    return math.ceil(
        max(
            estimate + PROCESSING_TIMEOUT_GRACE_SECONDS,
            minimum_timeout_seconds,
        )
    )


__all__ = [
    "BATCH_LEASE_GRACE_SECONDS",
    "ENCODING_4K_CHUNK_RUNTIME_SECONDS",
    "ENCODING_CHUNK_SECONDS",
    "ENCODING_PARALLEL_VIDEOS",
    "PROCESSING_TIMEOUT_GRACE_SECONDS",
    "REFERENCE_4K_PIXELS",
    "VMAF_4K_FRAMES_PER_SECOND",
    "competition_execution_lease_seconds",
    "competition_execution_timeout_seconds",
    "competition_scoring_timeout_seconds",
    "estimated_encoding_runtime_seconds",
    "estimated_vmaf_runtime_seconds",
]
