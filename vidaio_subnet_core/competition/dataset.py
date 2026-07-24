"""Immutable competition dataset indexes and Modal Volume access."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import tempfile
import time
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

from .config import CompetitionManifest


class DatasetError(RuntimeError):
    pass


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
    def sandbox_path(self) -> str:
        return f"/evaluation-inputs/{self.source_path}"


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
            previous = sources.setdefault(item.source_path, item)
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
                    item.source_path,
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
        unique_sources = {item.source_path: item for item in self.items}
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


def _probe_video(path: Path, executable: str = "ffprobe") -> dict[str, Any]:
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,pix_fmt,sample_aspect_ratio,"
            "duration,nb_read_frames,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise DatasetError(f"ffprobe failed for {path}: {result.stderr[:300]}")
    try:
        payload = json.loads(result.stdout)
        stream = next(
            value for value in payload["streams"] if value.get("codec_type") == "video"
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


def prepare_index(
    manifest: CompetitionManifest,
    source_root: Path,
    *,
    ffprobe_executable: str = "ffprobe",
) -> EvaluationIndex:
    source_root = source_root.resolve()
    paths = sorted(path for path in source_root.rglob("*.mp4") if path.is_file())
    if not paths:
        raise DatasetError(f"no MP4 inputs found below {source_root}")
    items = []
    query_random = random.Random(manifest.scoring_seed)
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(source_root).as_posix()
        metadata = _probe_video(path, ffprobe_executable)
        common = {
            "source_path": f"inputs/{relative}",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            **metadata,
        }
        source_id = f"input-{index:05d}"
        codec_mode = query_random.choice(("CRF", "VBR"))
        threshold = query_random.choice(COMPRESSION_VMAF_THRESHOLDS)
        threshold_label = f"{int(threshold):02d}"
        if codec_mode == "CRF":
            items.append(
                CompressionEvaluationIndexItem(
                    evaluation_id=f"{source_id}-crf-vmaf{threshold_label}",
                    codec_mode="CRF",
                    vmaf_threshold=threshold,
                    **common,
                )
            )
        else:
            bitrate = query_random.choice(COMPRESSION_VBR_BITRATES_BPS)
            items.append(
                CompressionEvaluationIndexItem(
                    evaluation_id=(
                        f"{source_id}-vbr-vmaf{threshold_label}-"
                        f"{bitrate // 1_000_000}mbps"
                    ),
                    codec_mode="VBR",
                    vmaf_threshold=threshold,
                    target_bitrate=bitrate,
                    **common,
                )
            )
    evaluation_index = EvaluationIndex(
        competition_id=manifest.competition_id,
        items=tuple(items),
    )
    evaluation_index.validate_for_manifest(manifest)
    return evaluation_index


def validate_local_index(
    index: EvaluationIndex, manifest: CompetitionManifest, source_root: Path
) -> None:
    index.validate_for_manifest(manifest)
    unique_sources = {item.source_path: item for item in index.items}
    for item in unique_sources.values():
        relative = PurePosixPath(item.source_path)
        if not relative.parts or relative.parts[0] != "inputs":
            raise DatasetError(
                f"{item.evaluation_id}: source path must be below inputs/"
            )
        path = source_root.joinpath(*relative.parts[1:])
        if not path.is_file():
            raise DatasetError(f"{item.evaluation_id}: source file is missing: {path}")
        if path.stat().st_size != item.size_bytes or sha256_file(path) != item.sha256:
            raise DatasetError(
                f"{item.evaluation_id}: source checksum/size mismatch: {path}"
            )


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
        self, manifest: CompetitionManifest, *, attempts: int = 5
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
        index.validate_for_manifest(manifest)
        return index

    def upload(
        self,
        manifest: CompetitionManifest,
        index: EvaluationIndex,
        source_root: Path,
    ) -> None:
        validate_local_index(index, manifest, source_root)
        self._ensure_environment()
        volume_name = manifest.evaluation_input_volume_name
        try:
            existing = self.load_index(manifest, attempts=1)
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
                raise DatasetError("the Volume already contains a different index")
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
                        relative = PurePosixPath(item.source_path)
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
    "COMPRESSION_VBR_BITRATES_BPS",
    "COMPRESSION_VMAF_THRESHOLDS",
    "CompressionEvaluationIndexItem",
    "EvaluationIndex",
    "EvaluationIndexItem",
    "ModalVolumeStore",
    "parse_evaluation_index_item_json",
    "prepare_index",
    "sha256_bytes",
    "sha256_file",
    "validate_local_index",
]
