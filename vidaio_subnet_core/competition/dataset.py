"""Immutable competition dataset indexes and Modal Volume access."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .batching import canonical_batch_assignments
from .config import CompetitionManifest


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetValidationIssue:
    """One source and every evaluation variant made unsafe by that source."""

    source_path: str
    evaluation_ids: tuple[str, ...]
    reasons: tuple[str, ...]


def format_validation_issues(
    issues: tuple[DatasetValidationIssue, ...] | list[DatasetValidationIssue],
) -> str:
    lines = [f"{len(issues)} invalid evaluation source(s):"]
    for issue in issues:
        lines.append(
            f"  - source={issue.source_path} "
            f"evaluations={','.join(issue.evaluation_ids)}"
        )
        lines.extend(f"      {reason}" for reason in issue.reasons)
    return "\n".join(lines)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvaluationIndexItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"
    )
    source_path: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    codec: str = Field(min_length=1, max_length=32)
    pixel_format: str = Field(min_length=1, max_length=32)
    sample_aspect_ratio: str = Field(min_length=1, max_length=32)

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("source_path must be relative and cannot traverse")
        return str(path)

    @property
    def logical_source_path(self) -> str:
        """Return the path visible inside a batch-scoped input mount."""

        path = PurePosixPath(self.source_path)
        if len(path.parts) >= 3 and path.parts[0] == "batches":
            return str(PurePosixPath(*path.parts[2:]))
        return str(path)

    @property
    def sandbox_path(self) -> str:
        return f"/evaluation-inputs/{self.logical_source_path}"


COMPRESSION_VMAF_THRESHOLDS = (85.0, 89.0, 93.0)
COMPRESSION_VBR_BITRATES_BPS = (5_000_000, 8_000_000, 10_000_000)


class CompressionEvaluationIndexItem(EvaluationIndexItem):
    """One query variant referencing an immutable source video."""

    codec_mode: Literal["CRF", "VBR"]
    vmaf_threshold: float = Field(ge=0, le=100)
    target_bitrate: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_rate_control(self) -> "CompressionEvaluationIndexItem":
        if self.codec_mode == "VBR":
            if self.target_bitrate not in COMPRESSION_VBR_BITRATES_BPS:
                raise ValueError("VBR query requires an approved target bitrate")
        elif self.target_bitrate is not None:
            raise ValueError("CRF query cannot set target_bitrate")
        return self


_EVALUATION_ITEM_ADAPTER = TypeAdapter(
    CompressionEvaluationIndexItem | EvaluationIndexItem
)


def parse_evaluation_index_item_json(
    value: str,
) -> CompressionEvaluationIndexItem | EvaluationIndexItem:
    return _EVALUATION_ITEM_ADAPTER.validate_json(value)


class EvaluationIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=2, ge=1, le=2)
    competition_id: str = Field(min_length=1, max_length=64)
    items: tuple[CompressionEvaluationIndexItem | EvaluationIndexItem, ...] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def unique_items(self) -> "EvaluationIndex":
        ids = [item.evaluation_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation IDs must be unique")
        sources: dict[str, EvaluationIndexItem] = {}
        variants: set[tuple[str, str, float, int | None]] = set()
        for item in self.items:
            logical_source_path = item.logical_source_path
            previous = sources.setdefault(logical_source_path, item)
            if (previous.size_bytes, previous.sha256) != (
                item.size_bytes,
                item.sha256,
            ):
                raise ValueError(
                    "evaluation variants sharing a source path must have identical "
                    "size and checksum"
                )
            if previous is not item and (
                not isinstance(previous, CompressionEvaluationIndexItem)
                or not isinstance(item, CompressionEvaluationIndexItem)
            ):
                raise ValueError(
                    "evaluation source paths must be unique unless they are "
                    "compression query variants"
                )
            if isinstance(item, CompressionEvaluationIndexItem):
                query = (
                    logical_source_path,
                    item.codec_mode,
                    item.vmaf_threshold,
                    item.target_bitrate,
                )
                if query in variants:
                    raise ValueError("compression query variants must be unique")
                variants.add(query)
        if self.schema_version == 1 and variants:
            raise ValueError(
                "compression query variants require index schema_version 2"
            )
        return self

    def normalized_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return sha256_bytes(self.normalized_json().encode("utf-8"))

    def validate_for_manifest(self, manifest: CompetitionManifest) -> None:
        if self.competition_id != manifest.competition_id:
            raise DatasetError("dataset index competition_id does not match manifest")
        minimum = manifest.min_video_length.total_seconds()
        maximum = manifest.max_video_length.total_seconds()
        duration_violations = []
        assignments = self.canonical_batch_assignments(manifest)
        invalid_paths = []
        for item in self.items:
            batch_index, _position = assignments[item.evaluation_id]
            logical_path = PurePosixPath(item.logical_source_path)
            expected = f"batches/{batch_index:06d}/{logical_path}"
            if not logical_path.parts or logical_path.parts[0] != "inputs":
                invalid_paths.append(
                    f"  - evaluation_id={item.evaluation_id} "
                    f"source must be below inputs/: {item.source_path}"
                )
                continue
            if item.source_path != expected:
                invalid_paths.append(
                    f"  - evaluation_id={item.evaluation_id} "
                    f"expected={expected} observed={item.source_path}"
                )
        if invalid_paths:
            raise DatasetError(
                "evaluation source paths do not match their canonical batch:\n"
                + "\n".join(invalid_paths)
            )
        unique_sources = {item.logical_source_path: item for item in self.items}
        for item in unique_sources.values():
            if not minimum <= item.duration_seconds <= maximum:
                duration_violations.append(
                    f"  - evaluation_id={item.evaluation_id} "
                    f"source={item.source_path} "
                    f"duration={item.duration_seconds:.3f}s"
                )
        if duration_violations:
            raise DatasetError(
                f"{len(duration_violations)} video(s) are outside the manifest "
                f"duration range [{minimum:.3f}s, {maximum:.3f}s]:\n"
                + "\n".join(duration_violations)
            )

    def canonical_batch_assignments(
        self, manifest: CompetitionManifest
    ) -> dict[str, tuple[int, int]]:
        return canonical_batch_assignments(
            (
                (item.evaluation_id, item.logical_source_path)
                for item in self.items
            ),
            manifest.evaluation_batch_size,
        )

    def with_batch_scoped_source_paths(
        self, manifest: CompetitionManifest
    ) -> "EvaluationIndex":
        assignments = self.canonical_batch_assignments(manifest)
        items = tuple(
            type(item).model_validate(
                {
                    **item.model_dump(mode="python"),
                    "source_path": (
                        f"batches/{assignments[item.evaluation_id][0]:06d}/"
                        f"{item.logical_source_path}"
                    ),
                }
            )
            for item in self.items
        )
        return self.model_copy(update={"items": items})


def _probe_video(path: Path, executable: str = "ffprobe") -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                executable,
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
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DatasetError(f"could not probe {path}: {exc}") from exc
    if result.returncode != 0:
        raise DatasetError(f"ffprobe failed for {path}: {result.stderr[:300]}")
    if result.stderr.strip():
        raise DatasetError(
            f"ffprobe reported media errors for {path}: {result.stderr[:300]}"
        )
    try:
        payload = json.loads(result.stdout)
        video_streams = [
            value for value in payload["streams"] if value.get("codec_type") == "video"
        ]
        if len(video_streams) != 1:
            raise DatasetError(
                f"expected exactly one video stream in {path}, found "
                f"{len(video_streams)}"
            )
        stream = video_streams[0]
        format_name = str(payload["format"]["format_name"])
        if "mp4" not in format_name.lower():
            raise DatasetError(
                f"evaluation source is not an MP4 container: {path} ({format_name})"
            )
        duration = stream.get("duration") or payload["format"]["duration"]
        frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
        return {
            "duration_seconds": float(duration),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "frame_count": int(frame_count),
            "codec": str(stream["codec_name"]),
            "pixel_format": str(stream.get("pix_fmt") or "unknown"),
            "sample_aspect_ratio": str(stream.get("sample_aspect_ratio") or "unknown"),
        }
    except (
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise DatasetError(f"ffprobe returned incomplete metadata for {path}") from exc


_UNKNOWN_MEDIA_VALUES = {"", "unknown", "n/a", "na", "none", "null"}
_MEDIA_FIELDS = (
    "duration_seconds",
    "width",
    "height",
    "frame_count",
    "codec",
    "pixel_format",
    "sample_aspect_ratio",
)


def _metadata_validation_reasons(
    metadata: dict[str, Any], manifest: CompetitionManifest
) -> list[str]:
    reasons = []
    duration = metadata.get("duration_seconds")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration):
        reasons.append(f"duration is not finite: {duration!r}")
    elif duration <= 0:
        reasons.append(f"duration must be positive: {duration!r}")
    else:
        minimum = manifest.min_video_length.total_seconds()
        maximum = manifest.max_video_length.total_seconds()
        if not minimum <= duration <= maximum:
            reasons.append(
                f"duration {duration:.6f}s is outside manifest range "
                f"[{minimum:.6f}s, {maximum:.6f}s]"
            )

    for field in ("width", "height", "frame_count"):
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            reasons.append(f"{field} must be a positive integer: {value!r}")

    for field in ("codec", "pixel_format"):
        value = str(metadata.get(field, "")).strip()
        if value.lower() in _UNKNOWN_MEDIA_VALUES:
            reasons.append(f"{field} is missing or unknown: {value!r}")

    sar = str(metadata.get("sample_aspect_ratio", "")).strip()
    try:
        numerator_text, denominator_text = sar.split(":", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
        if numerator <= 0 or denominator <= 0:
            raise ValueError
    except (TypeError, ValueError):
        reasons.append(
            f"sample_aspect_ratio must be a positive rational such as 1:1: {sar!r}"
        )
    return reasons


def _item_metadata(item: EvaluationIndexItem) -> dict[str, Any]:
    return {field: getattr(item, field) for field in _MEDIA_FIELDS}


def _indexed_metadata_reasons(
    item: EvaluationIndexItem,
    metadata: dict[str, Any],
    manifest: CompetitionManifest,
) -> list[str]:
    reasons = _metadata_validation_reasons(_item_metadata(item), manifest)
    duration = float(metadata["duration_seconds"])
    duration_tolerance = max(0.2, float(item.duration_seconds) * 0.005)
    if abs(duration - float(item.duration_seconds)) > duration_tolerance:
        reasons.append(
            "duration metadata mismatch: "
            f"index={item.duration_seconds:.6f}s probed={duration:.6f}s "
            f"tolerance={duration_tolerance:.6f}s"
        )
    for field in (
        "width",
        "height",
        "frame_count",
        "codec",
        "pixel_format",
        "sample_aspect_ratio",
    ):
        indexed = getattr(item, field)
        probed = metadata[field]
        if indexed != probed:
            reasons.append(
                f"{field} metadata mismatch: index={indexed!r} probed={probed!r}"
            )
    return reasons


def _source_validation_reasons(
    items: tuple[EvaluationIndexItem, ...],
    manifest: CompetitionManifest,
    probe_path: Path,
    measured_size: int,
    measured_checksum: str,
    *,
    ffprobe_executable: str,
    location: str = "",
) -> list[str]:
    reasons = []
    for item in items:
        if measured_size != item.size_bytes:
            reasons.append(
                f"{location}size mismatch for {item.evaluation_id}: "
                f"index={item.size_bytes} actual={measured_size}"
            )
        if measured_checksum != item.sha256:
            reasons.append(
                f"{location}checksum mismatch for {item.evaluation_id}: "
                f"index={item.sha256} actual={measured_checksum}"
            )
    try:
        metadata = _probe_video(probe_path, ffprobe_executable)
    except DatasetError as exc:
        return [*reasons, str(exc)]
    reasons.extend(_metadata_validation_reasons(metadata, manifest))
    for item in items:
        reasons.extend(_indexed_metadata_reasons(item, metadata, manifest))
    return reasons


def _issue(
    source_path: str,
    items: tuple[EvaluationIndexItem, ...],
    reasons: list[str],
) -> DatasetValidationIssue:
    return DatasetValidationIssue(
        source_path=source_path,
        evaluation_ids=tuple(item.evaluation_id for item in items),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _items_by_logical_source(
    index: EvaluationIndex,
) -> dict[str, tuple[EvaluationIndexItem, ...]]:
    grouped: dict[str, list[EvaluationIndexItem]] = {}
    for item in index.items:
        grouped.setdefault(item.logical_source_path, []).append(item)
    return {path: tuple(items) for path, items in grouped.items()}


def prepare_index_candidates(
    manifest: CompetitionManifest,
    source_root: Path,
    *,
    ffprobe_executable: str = "ffprobe",
) -> tuple[EvaluationIndex | None, tuple[DatasetValidationIssue, ...]]:
    """Probe every source and return a clean candidate plus collected failures."""

    source_root = source_root.resolve()
    paths = sorted(path for path in source_root.rglob("*.mp4") if path.is_file())
    if not paths:
        raise DatasetError(f"no MP4 inputs found below {source_root}")
    items = []
    issues = []
    query_random = random.Random(manifest.scoring_seed)
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(source_root).as_posix()
        source_id = f"input-{index:05d}"
        codec_mode = query_random.choice(("CRF", "VBR"))
        threshold = query_random.choice(COMPRESSION_VMAF_THRESHOLDS)
        threshold_label = f"{int(threshold):02d}"
        if codec_mode == "CRF":
            evaluation_id = f"{source_id}-crf-vmaf{threshold_label}"
            bitrate = None
        else:
            bitrate = query_random.choice(COMPRESSION_VBR_BITRATES_BPS)
            evaluation_id = (
                f"{source_id}-vbr-vmaf{threshold_label}-{bitrate // 1_000_000}mbps"
            )
        source_path = f"inputs/{relative}"

        def flag(*reasons: str) -> DatasetValidationIssue:
            return DatasetValidationIssue(source_path, (evaluation_id,), reasons)

        try:
            metadata = _probe_video(path, ffprobe_executable)
        except DatasetError as exc:
            issues.append(flag(str(exc)))
            continue
        reasons = _metadata_validation_reasons(metadata, manifest)
        if reasons:
            issues.append(flag(*reasons))
            continue
        try:
            size_bytes = path.stat().st_size
            checksum = sha256_file(path)
        except OSError as exc:
            issues.append(flag(f"could not read source file {path}: {exc}"))
            continue
        common = {
            "source_path": source_path,
            "size_bytes": size_bytes,
            "sha256": checksum,
            **metadata,
        }
        try:
            items.append(
                CompressionEvaluationIndexItem(
                    evaluation_id=evaluation_id,
                    codec_mode=codec_mode,
                    vmaf_threshold=threshold,
                    target_bitrate=bitrate,
                    **common,
                )
            )
        except ValueError as exc:
            issues.append(flag(f"generated index item is invalid: {exc}"))
    evaluation_index = (
        EvaluationIndex(
            competition_id=manifest.competition_id,
            items=tuple(items),
        ).with_batch_scoped_source_paths(manifest)
        if items
        else None
    )
    return evaluation_index, tuple(issues)


def prepare_index(
    manifest: CompetitionManifest,
    source_root: Path,
    *,
    ffprobe_executable: str = "ffprobe",
) -> EvaluationIndex:
    evaluation_index, issues = prepare_index_candidates(
        manifest,
        source_root,
        ffprobe_executable=ffprobe_executable,
    )
    if issues:
        raise DatasetError(format_validation_issues(issues))
    if evaluation_index is None:
        raise DatasetError("all evaluation sources are invalid")
    return evaluation_index


def local_index_validation_issues(
    index: EvaluationIndex,
    manifest: CompetitionManifest,
    source_root: Path,
    *,
    ffprobe_executable: str = "ffprobe",
) -> tuple[DatasetValidationIssue, ...]:
    """Reprobe local bytes and compare every scoring-relevant index field."""

    index.validate_for_manifest(manifest)
    issues = []
    source_root = source_root.resolve()
    for source_path, items in _items_by_logical_source(index).items():
        reasons = []
        relative = PurePosixPath(source_path)
        if not relative.parts or relative.parts[0] != "inputs":
            reasons.append("source path must be below inputs/")
            issues.append(_issue(source_path, items, reasons))
            continue
        path = source_root.joinpath(*relative.parts[1:])
        if not path.is_file():
            reasons.append(f"source file is missing: {path}")
            issues.append(_issue(source_path, items, reasons))
            continue
        try:
            measured_size = path.stat().st_size
            measured_checksum = sha256_file(path)
        except OSError as exc:
            reasons.append(f"could not read source file {path}: {exc}")
            issues.append(_issue(source_path, items, reasons))
            continue
        reasons.extend(
            _source_validation_reasons(
                items,
                manifest,
                path,
                measured_size,
                measured_checksum,
                ffprobe_executable=ffprobe_executable,
            )
        )
        if reasons:
            issues.append(_issue(source_path, items, reasons))
    return tuple(issues)


def exclude_invalid_sources(
    index: EvaluationIndex,
    issues: tuple[DatasetValidationIssue, ...] | list[DatasetValidationIssue],
    manifest: CompetitionManifest,
) -> EvaluationIndex:
    excluded = {issue.source_path for issue in issues}
    remaining = tuple(
        item
        for item in index.items
        if item.source_path not in excluded
        and item.logical_source_path not in excluded
    )
    if not remaining:
        raise DatasetError("all evaluation entries would be disqualified")
    return index.model_copy(update={"items": remaining}).with_batch_scoped_source_paths(
        manifest
    )


def validate_local_index(
    index: EvaluationIndex,
    manifest: CompetitionManifest,
    source_root: Path,
    *,
    ffprobe_executable: str = "ffprobe",
) -> None:
    issues = local_index_validation_issues(
        index,
        manifest,
        source_root,
        ffprobe_executable=ffprobe_executable,
    )
    if issues:
        raise DatasetError(format_validation_issues(issues))


class ModalVolumeStore:
    def __init__(self, *, environment_name: str = "main", modal_api: Any = None):
        if modal_api is None:
            try:
                import modal as modal_api
            except ImportError as exc:
                raise DatasetError("Modal SDK is required for dataset access") from exc
        self.modal = modal_api
        self.environment_name = environment_name

    def _volume(self, name: str, *, create_if_missing: bool = False):
        return self.modal.Volume.from_name(
            name,
            environment_name=self.environment_name,
            create_if_missing=create_if_missing,
        )

    def _ensure_environment(self) -> None:
        try:
            environment = self.modal.Environment.from_name(
                self.environment_name,
                create_if_missing=True,
            )
            environment.hydrate()
        except Exception as exc:
            raise DatasetError(
                f"could not create or access Modal environment "
                f"{self.environment_name!r}: {exc}"
            ) from exc

    def _is_volume_not_found(self, error: BaseException) -> bool:
        modal_exception = getattr(self.modal, "exception", None)
        not_found_error = getattr(modal_exception, "NotFoundError", None)
        if not isinstance(not_found_error, type):
            return False
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            if isinstance(current, not_found_error):
                return True
            seen.add(id(current))
            current = current.__cause__ or current.__context__
        return False

    def read_bytes(self, volume_name: str, path: str, *, attempts: int = 5) -> bytes:
        normalized = str(PurePosixPath(path)).lstrip("/")
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return b"".join(self._volume(volume_name).read_file(normalized))
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(2**attempt, 8))
        raise DatasetError(
            f"could not read {volume_name}/{normalized}: {last_error}"
        ) from last_error

    def load_index(
        self,
        manifest: CompetitionManifest,
        *,
        attempts: int = 5,
        validate_for_manifest: bool = True,
    ) -> EvaluationIndex:
        payload = self.read_bytes(
            manifest.evaluation_input_volume_name,
            manifest.evaluation_index_path,
            attempts=attempts,
        )
        try:
            index = EvaluationIndex.model_validate_json(payload)
        except Exception as exc:
            raise DatasetError("evaluation index is invalid") from exc
        if validate_for_manifest:
            index.validate_for_manifest(manifest)
        return index

    def validation_issues(
        self,
        manifest: CompetitionManifest,
        index: EvaluationIndex,
        *,
        ffprobe_executable: str = "ffprobe",
    ) -> tuple[DatasetValidationIssue, ...]:
        """Read back and semantically validate every indexed remote source."""

        if index.competition_id != manifest.competition_id:
            raise DatasetError("dataset index competition_id does not match manifest")
        issues = []
        index.validate_for_manifest(manifest)
        for source_path, items in _items_by_logical_source(index).items():
            reasons = []
            for remote_path in dict.fromkeys(item.source_path for item in items):
                try:
                    payload = self.read_bytes(
                        manifest.evaluation_input_volume_name,
                        remote_path,
                    )
                except DatasetError as exc:
                    reasons.append(str(exc))
                    continue
                with tempfile.NamedTemporaryFile(suffix=".mp4") as source_file:
                    source_file.write(payload)
                    source_file.flush()
                    reasons.extend(
                        _source_validation_reasons(
                            items,
                            manifest,
                            Path(source_file.name),
                            len(payload),
                            sha256_bytes(payload),
                            ffprobe_executable=ffprobe_executable,
                            location=f"remote {remote_path} ",
                        )
                    )
            if reasons:
                issues.append(_issue(source_path, items, reasons))
        return tuple(issues)

    def upload(
        self,
        manifest: CompetitionManifest,
        index: EvaluationIndex,
        source_root: Path,
        *,
        validate_local: bool = True,
    ) -> None:
        if validate_local:
            validate_local_index(index, manifest, source_root)
        self._ensure_environment()
        volume_name = manifest.evaluation_input_volume_name
        try:
            existing = self.load_index(
                manifest,
                attempts=1,
                validate_for_manifest=False,
            )
        except DatasetError as exc:
            existing = None
            if self._is_volume_not_found(exc):
                logger.info(
                    "Modal Volume '{}' was not found in environment '{}'; "
                    "creating it before dataset upload",
                    volume_name,
                    self.environment_name,
                )
            else:
                logger.info(
                    "No existing evaluation dataset index was found in Modal "
                    "Volume '{}' in environment '{}'; the volume will be created "
                    "if it is missing",
                    volume_name,
                    self.environment_name,
                )
        if existing is not None:
            if existing.digest() != index.digest():
                raise DatasetError(
                    "the Volume already contains a different index; the uploaded "
                    "dataset is immutable, so "
                    "use a new competition ID and Volume"
                )
        else:
            volume = self._volume(volume_name, create_if_missing=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json"
            ) as index_file:
                index_file.write(index.normalized_json())
                index_file.flush()
                with volume.batch_upload(force=True) as batch:
                    unique_sources = {item.source_path: item for item in index.items}
                    for item in unique_sources.values():
                        relative = PurePosixPath(item.logical_source_path)
                        local_path = source_root.joinpath(*relative.parts[1:])
                        batch.put_file(local_path, f"/{item.source_path}")
                    batch.put_file(index_file.name, manifest.evaluation_index_path)
            logger.info(
                "Uploaded evaluation dataset to Modal Volume '{}' in environment "
                "'{}' (created automatically if it was missing)",
                volume_name,
                self.environment_name,
            )
        remote_index = self.load_index(manifest)
        if remote_index.digest() != index.digest():
            raise DatasetError(
                "uploaded evaluation index failed read-back verification"
            )
        unique_sources = {item.source_path: item for item in index.items}
        for item in unique_sources.values():
            remote_source = self.read_bytes(
                manifest.evaluation_input_volume_name, item.source_path
            )
            if (
                len(remote_source) != item.size_bytes
                or sha256_bytes(remote_source) != item.sha256
            ):
                raise DatasetError(
                    f"{item.evaluation_id}: uploaded source failed read-back "
                    "verification"
                )


__all__ = [
    "DatasetError",
    "DatasetValidationIssue",
    "COMPRESSION_VBR_BITRATES_BPS",
    "COMPRESSION_VMAF_THRESHOLDS",
    "CompressionEvaluationIndexItem",
    "EvaluationIndex",
    "EvaluationIndexItem",
    "ModalVolumeStore",
    "exclude_invalid_sources",
    "format_validation_issues",
    "local_index_validation_issues",
    "parse_evaluation_index_item_json",
    "prepare_index",
    "prepare_index_candidates",
    "sha256_bytes",
    "sha256_file",
    "validate_local_index",
]
