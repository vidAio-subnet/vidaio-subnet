"""Warmup qualification contract used before a contender can be evaluated."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import CompetitionManifest
from .contracts import (
    CompetitionCompressionItem,
    CompetitionCompressionRequest,
    CompetitionCompressionResponse,
)


class QualificationClient(Protocol):
    def health(self) -> dict[str, Any]: ...
    def compress(
        self, request: CompetitionCompressionRequest
    ) -> CompetitionCompressionResponse: ...


class QualificationError(RuntimeError):
    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    duration_seconds: float
    codec: str
    container: str
    pixel_format: str
    sample_aspect_ratio: str
    size_bytes: int
    frame_count: int | None = None


class FfprobeMediaInspector:
    def __init__(self, executable: str = "ffprobe") -> None:
        self.executable = executable

    def inspect(self, path: Path) -> MediaInfo:
        result = subprocess.run(
            [
                self.executable,
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,pix_fmt,"
                "sample_aspect_ratio,duration,nb_read_frames,nb_frames:"
                "format=format_name,duration",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise QualificationError("MEDIA_PROBE_FAILED", result.stderr.strip()[:300])
        try:
            payload = json.loads(result.stdout)
            stream = next(
                item for item in payload["streams"] if item.get("codec_type") == "video"
            )
            duration = stream.get("duration") or payload["format"].get("duration")
            return MediaInfo(
                width=int(stream["width"]),
                height=int(stream["height"]),
                duration_seconds=float(duration),
                codec=str(stream["codec_name"]),
                container=str(payload["format"]["format_name"]),
                pixel_format=str(stream.get("pix_fmt", "")),
                sample_aspect_ratio=str(stream.get("sample_aspect_ratio", "")),
                size_bytes=path.stat().st_size,
                frame_count=int(
                    stream.get("nb_read_frames") or stream.get("nb_frames")
                ),
            )
        except (
            KeyError,
            StopIteration,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise QualificationError(
                "MEDIA_PROBE_INVALID", "ffprobe returned incomplete media metadata"
            ) from exc


@dataclass(frozen=True)
class QualificationResult:
    health_ok: bool
    input_media: MediaInfo
    output_media: MediaInfo
    output_path: str


class WarmupQualifier:
    def __init__(
        self,
        client: QualificationClient,
        inspector: FfprobeMediaInspector | None = None,
    ) -> None:
        self.client = client
        self.inspector = inspector or FfprobeMediaInspector()

    def qualify(
        self,
        manifest: CompetitionManifest,
        *,
        repository_root: Path,
        contender_hotkey: str,
        sandbox_input_path: str = "/evaluation-inputs/warmup.mp4",
        sandbox_output_path: str = "/output/warmup/output.mp4",
    ) -> QualificationResult:
        manifest.validate_runtime_paths(repository_root)
        fixture = manifest.warmup_input_path
        if not fixture.is_absolute():
            fixture = repository_root / fixture
        input_media = self.inspector.inspect(fixture)
        if (
            input_media.duration_seconds <= 0
            or input_media.duration_seconds > 5.5
            or input_media.width > 1280
            or input_media.height > 720
        ):
            raise QualificationError(
                "WARMUP_FIXTURE_INVALID",
                "warmup fixture must be at most 5.5 seconds and 720p",
            )
        health = self.client.health()
        if health.get("status") != "ok":
            raise QualificationError(
                "HEALTH_CHECK_FAILED", "contender /health did not return status=ok"
            )
        local_io = health.get("competition_local_io", {})
        if local_io.get("remote_io_disabled") is not True:
            raise QualificationError(
                "REMOTE_IO_ENABLED",
                "competition service did not prove remote I/O is disabled",
            )
        request = CompetitionCompressionRequest(
            competition_id=manifest.competition_id,
            hotkey=contender_hotkey,
            batch_id="qualification-warmup",
            items=(
                CompetitionCompressionItem(
                    evaluation_id="warmup",
                    input_path=sandbox_input_path,
                    output_path=sandbox_output_path,
                    codec="AV1",
                    vmaf_threshold=manifest.vmaf_threshold,
                ),
            ),
        )
        response = self.client.compress(request)
        if len(response.results) != 1:
            raise QualificationError(
                "COMPRESSION_RESPONSE_INVALID",
                "warmup response does not match its request",
            )
        item = response.results[0]
        if item.output_path != sandbox_output_path:
            raise QualificationError(
                "WARMUP_COMPRESSION_FAILED",
                "warmup response did not return the assigned output path",
            )
        output_media = self.inspector.inspect(Path(sandbox_output_path))
        if output_media.codec != "av1" or "mp4" not in output_media.container:
            raise QualificationError(
                "OUTPUT_MEDIA_INVALID", "warmup output must be AV1 in MP4"
            )
        if (
            output_media.width != input_media.width
            or output_media.height != input_media.height
        ):
            raise QualificationError(
                "OUTPUT_DIMENSIONS_INVALID",
                "warmup compression must preserve input dimensions",
            )
        if output_media.size_bytes >= input_media.size_bytes:
            raise QualificationError(
                "OUTPUT_NOT_COMPRESSED",
                "warmup output must be smaller than its input",
            )
        duration_tolerance = max(0.2, input_media.duration_seconds * 0.02)
        if (
            abs(output_media.duration_seconds - input_media.duration_seconds)
            > duration_tolerance
        ):
            raise QualificationError(
                "OUTPUT_DURATION_INVALID",
                "warmup output duration does not match its input",
            )
        if not output_media.pixel_format.startswith(
            "yuv"
        ) or output_media.sample_aspect_ratio not in ("", "1:1"):
            raise QualificationError(
                "OUTPUT_MEDIA_INVALID", "warmup output must use YUV square pixels"
            )
        return QualificationResult(True, input_media, output_media, sandbox_output_path)


def preflight_warmup_fixture(
    manifest: CompetitionManifest, repository_root: Path
) -> MediaInfo:
    manifest.validate_runtime_paths(repository_root)
    path = manifest.warmup_input_path
    if not path.is_absolute():
        path = repository_root / path
    media = FfprobeMediaInspector().inspect(path)
    if (
        media.duration_seconds <= 0
        or media.duration_seconds > 5.5
        or media.width > 1280
        or media.height > 720
    ):
        raise QualificationError(
            "WARMUP_FIXTURE_INVALID",
            "warmup fixture must be at most 5.5 seconds and 720p",
        )
    return media


__all__ = [
    "FfprobeMediaInspector",
    "MediaInfo",
    "QualificationClient",
    "QualificationError",
    "QualificationResult",
    "WarmupQualifier",
    "preflight_warmup_fixture",
]
