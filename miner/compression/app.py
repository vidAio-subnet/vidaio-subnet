"""
Compression microservice — wraps the ffmpeg binary installed in this container.

Accepts a video path (on shared volume) OR a URL, codec, and quality settings,
runs GPU-accelerated ffmpeg compression, returns the output path or S3 URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import boto3
import httpx
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("compression")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_cleanup_worker())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title="Video Compression Service", lifespan=lifespan)

SHARED_VOLUME_PATH = os.getenv("SHARED_VOLUME_PATH", "/tmp/organic-proxy")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_COMPRESSION", "2"))
MAX_QUEUE_SIZE = int(
    os.getenv("MAX_QUEUE_SIZE_COMPRESSION") or os.getenv("MAX_QUEUE_SIZE", "5")
)
DISABLE_REMOTE_IO = os.getenv("DISABLE_REMOTE_IO", "false").lower() in (
    "1",
    "true",
    "yes",
)
COMPETITION_INPUT_ROOT = os.getenv("COMPETITION_INPUT_ROOT", "/evaluation-inputs")
COMPETITION_OUTPUT_ROOT = os.getenv("COMPETITION_OUTPUT_ROOT", "/output")
DEFAULT_COMPRESSION_CQ = 35
COMPRESSION_CQ_BY_TYPE = {
    "Low": 40,
    "Medium": 35,
    "High": 30,
}

# Storage provider label only. Uploads use one S3-compatible code path.
STORAGE_PROVIDER = os.getenv("MINER_STORAGE_PROVIDER", "s3").lower()
S3_REGION = os.getenv("MINER_STORAGE_S3_REGION", "us-east-1").strip() or "us-east-1"
S3_BUCKET = os.getenv("MINER_STORAGE_S3_BUCKET_NAME", "").strip()
S3_ACCESS_KEY_ID = os.getenv("MINER_STORAGE_S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("MINER_STORAGE_S3_SECRET_ACCESS_KEY", "").strip()
S3_ENDPOINT_URL = (
    os.getenv("MINER_STORAGE_S3_ENDPOINT_URL")
    or os.getenv("MINER_STORAGE_S3_ENDPOINT")
    or ""
).strip()
S3_PRESIGNED_EXPIRY = int(
    os.getenv("MINER_STORAGE_S3_PRESIGNED_EXPIRY")
    or os.getenv("S3_PRESIGNED_EXPIRY")
    or "3600"
)
PRESIGNED_URL_CLEANUP_GRACE_SECONDS = int(
    os.getenv("PRESIGNED_URL_CLEANUP_GRACE_SECONDS", "600")
)
TEMP_FILE_TTL_SECONDS = int(
    os.getenv("MINER_TEMP_FILE_TTL_SECONDS")
    or os.getenv("TEMP_FILE_TTL_SECONDS")
    or str(min(S3_PRESIGNED_EXPIRY, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)
CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("MINER_CLEANUP_INTERVAL_SECONDS")
    or os.getenv("CLEANUP_INTERVAL_SECONDS")
    or "300"
)
CLEANUP_MAX_VOLUME_BYTES = int(
    os.getenv("MINER_CLEANUP_MAX_VOLUME_BYTES")
    or os.getenv("CLEANUP_MAX_VOLUME_BYTES")
    or "9000000000"
)
CLEANUP_MIN_FILE_AGE_SECONDS = int(
    os.getenv("MINER_CLEANUP_MIN_FILE_AGE_SECONDS")
    or os.getenv("CLEANUP_MIN_FILE_AGE_SECONDS")
    or "60"
)
CLEANUP_ENABLED = os.getenv(
    "MINER_CLEANUP_ENABLED", os.getenv("CLEANUP_ENABLED", "true")
).lower() in (
    "1",
    "true",
    "yes",
)
STORAGE_CLEANUP_ENABLED = os.getenv(
    "MINER_STORAGE_CLEANUP_ENABLED", os.getenv("STORAGE_CLEANUP_ENABLED", "true")
).lower() in ("1", "true", "yes")
STORAGE_CLEANUP_PREFIXES = [
    prefix.strip()
    for prefix in os.getenv(
        "MINER_STORAGE_CLEANUP_PREFIXES", "processing/,upscaling/"
    ).split(",")
    if prefix.strip()
]
STORAGE_OBJECT_TTL_SECONDS = int(
    os.getenv("MINER_STORAGE_OBJECT_TTL_SECONDS")
    or os.getenv("STORAGE_OBJECT_TTL_SECONDS")
    or str(min(S3_PRESIGNED_EXPIRY, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)
COMPRESSION_CHUNKING_ENABLED = os.getenv(
    "COMPRESSION_CHUNKING_ENABLED", "true"
).lower() in (
    "1",
    "true",
    "yes",
)
COMPRESSION_CHUNK_MIN_DURATION_SECONDS = int(
    os.getenv("COMPRESSION_CHUNK_MIN_DURATION_SECONDS", "1200")
)
COMPRESSION_CHUNK_TARGET_SECONDS = int(
    os.getenv("COMPRESSION_CHUNK_TARGET_SECONDS", "600")
)
COMPRESSION_CHUNK_PARALLELISM = max(
    1, int(os.getenv("COMPRESSION_CHUNK_PARALLELISM", "2"))
)
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")


def resolve_compression_cq(
    *,
    explicit_cq: int | None = None,
    compression_type: Literal["Low", "Medium", "High"] | None = None,
    vmaf_threshold: float | None = None,
) -> int:
    """Resolve CQ once for inference and competition requests.

    Explicit legacy overrides retain precedence. Otherwise an explicit quality
    tier wins, followed by the VMAF target and the historical medium default.
    """

    if explicit_cq is not None:
        return explicit_cq
    if compression_type is not None:
        return COMPRESSION_CQ_BY_TYPE[compression_type]
    if vmaf_threshold is not None:
        if vmaf_threshold >= 93:
            return COMPRESSION_CQ_BY_TYPE["High"]
        if vmaf_threshold >= 89:
            return COMPRESSION_CQ_BY_TYPE["Medium"]
        return COMPRESSION_CQ_BY_TYPE["Low"]
    return DEFAULT_COMPRESSION_CQ


_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_size = 0
_active_count = 0
_active_file_paths: set[str] = set()
_lock = asyncio.Lock()

# Codec name → nvenc/software encoder mapping
CODEC_MAP = {
    "AV1": "av1_nvenc",
    "H264": "h264_nvenc",
    "H.264": "h264_nvenc",
    "HEVC": "hevc_nvenc",
    "H265": "hevc_nvenc",
    "H.265": "hevc_nvenc",
    "VP9": "libvpx-vp9",
}


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _get_s3_client():
    client_kwargs = {
        "region_name": S3_REGION,
        "aws_access_key_id": S3_ACCESS_KEY_ID or None,
        "aws_secret_access_key": S3_SECRET_ACCESS_KEY or None,
        "config": Config(signature_version="s3v4"),
    }
    if S3_ENDPOINT_URL:
        client_kwargs["endpoint_url"] = S3_ENDPOINT_URL
    return boto3.client("s3", **client_kwargs)


def _storage_config_status() -> dict[str, object]:
    return {
        "provider": STORAGE_PROVIDER,
        "region": S3_REGION,
        "bucket_configured": bool(S3_BUCKET),
        "access_key_configured": bool(S3_ACCESS_KEY_ID),
        "secret_key_configured": bool(S3_SECRET_ACCESS_KEY),
        "endpoint_configured": bool(S3_ENDPOINT_URL),
    }


def _compression_config_status() -> dict[str, object]:
    return {
        "chunking_enabled": COMPRESSION_CHUNKING_ENABLED,
        "chunk_min_duration_seconds": COMPRESSION_CHUNK_MIN_DURATION_SECONDS,
        "chunk_target_seconds": COMPRESSION_CHUNK_TARGET_SECONDS,
        "chunk_parallelism": COMPRESSION_CHUNK_PARALLELISM,
    }


def _validate_s3_config():
    missing = []
    if not S3_BUCKET:
        missing.append("MINER_STORAGE_S3_BUCKET_NAME")
    if not S3_ACCESS_KEY_ID:
        missing.append("MINER_STORAGE_S3_ACCESS_KEY_ID")
    if not S3_SECRET_ACCESS_KEY:
        missing.append("MINER_STORAGE_S3_SECRET_ACCESS_KEY")

    if missing:
        raise RuntimeError(f"Missing storage configuration: {', '.join(missing)}")


async def _download_url(url: str, dest: str):
    log.info(f"Downloading {url[:80]}... → {dest}")
    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
    log.info(f"Downloaded {os.path.getsize(dest) / (1024 * 1024):.1f} MB → {dest}")


def _upload_to_s3(local_path: str, key: str) -> str:
    _validate_s3_config()
    client = _get_s3_client()
    client.upload_file(local_path, S3_BUCKET, key)
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=min(S3_PRESIGNED_EXPIRY, 604800),
    )
    log.info(f"Uploaded to {STORAGE_PROVIDER} storage: s3://{S3_BUCKET}/{key}")
    return url


def _cleanup(*paths: str):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _cleanup_tree(path: str):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _format_process_output(stdout: bytes, stderr: bytes) -> str:
    parts = []
    stdout_msg = stdout.decode(errors="replace").strip()
    stderr_msg = stderr.decode(errors="replace").strip()
    if stdout_msg:
        parts.append(f"stdout:\n{stdout_msg}")
    if stderr_msg:
        parts.append(f"stderr:\n{stderr_msg}")
    return "\n\n".join(parts)


async def _run_process(
    cmd: list[str], task_label: str, step: str
) -> tuple[int | None, bytes, bytes, str]:
    try:
        log.info(f"[{task_label}] {step}: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout, stderr, ""
    except FileNotFoundError:
        return None, b"", b"", f"{cmd[0]} binary not found"
    except OSError as e:
        return None, b"", b"", f"Failed to start {cmd[0]}: {e}"


async def _probe_duration_seconds(path: str, task_label: str) -> float | None:
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    returncode, stdout, stderr, run_error = await _run_process(
        cmd, task_label, "ffprobe duration"
    )
    if returncode != 0 or run_error:
        detail = run_error or stderr.decode(errors="replace").strip()
        log.warning(f"[{task_label}] Failed to probe duration: {detail}")
        return None
    try:
        return float(stdout.decode().strip())
    except ValueError:
        log.warning(f"[{task_label}] ffprobe returned invalid duration: {stdout!r}")
        return None


async def _probe_segment_duration_seconds(path: str, task_label: str) -> float:
    duration = await _probe_duration_seconds(path, task_label)
    if duration is None:
        raise RuntimeError(f"Unable to probe segment duration: {path}")
    return duration


def _format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


async def _log_chunk_seams(segments: list[str], task_label: str) -> list[float]:
    durations = [
        await _probe_segment_duration_seconds(segment, task_label)
        for segment in segments
    ]
    seams: list[float] = []
    elapsed = 0.0
    for duration in durations[:-1]:
        elapsed += duration
        seams.append(elapsed)

    log.info(
        f"[{task_label}] chunk_seams "
        + json.dumps(
            {
                "seam_seconds": [round(seam, 3) for seam in seams],
                "seam_timestamps": [_format_timestamp(seam) for seam in seams],
                "segment_durations_seconds": [
                    round(duration, 3) for duration in durations
                ],
            }
        )
    )
    return seams


def _build_ffmpeg_args(
    local_input: str, output_path: str, req: "CompressRequest", encoder: str
) -> list[str]:
    resolved_cq = resolve_compression_cq(
        explicit_cq=req.cq,
        compression_type=req.compression_type,
        vmaf_threshold=req.vmaf_threshold,
    )
    ffmpeg_args = [
        FFMPEG_BIN,
        "-y",
        "-hwaccel",
        "cuda",
        "-i",
        local_input,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        encoder,
    ]

    if req.codec_mode == "VBR" and req.target_bitrate:
        ffmpeg_args.extend(["-b:v", str(req.target_bitrate)])
    else:
        ffmpeg_args.extend(["-cq", str(resolved_cq)])

    ffmpeg_args.extend(["-preset", req.preset])

    video_filters = []
    if req.target_width and req.target_height:
        w = req.target_width if req.target_width % 2 == 0 else req.target_width - 1
        h = req.target_height if req.target_height % 2 == 0 else req.target_height - 1
        video_filters.append(f"scale={w}:{h}")

    video_filters.append("setsar=1")
    ffmpeg_args.extend(["-vf", ",".join(video_filters)])

    ffmpeg_args.extend(
        ["-c:a", "copy", "-sn", "-dn", "-movflags", "+faststart", output_path]
    )
    return ffmpeg_args


def _should_chunk(req: "CompressRequest", duration_seconds: float | None) -> bool:
    if req.chunked is not None:
        return req.chunked
    return (
        COMPRESSION_CHUNKING_ENABLED
        and duration_seconds is not None
        and duration_seconds >= COMPRESSION_CHUNK_MIN_DURATION_SECONDS
    )


def _concat_file_line(path: str) -> str:
    escaped = path.replace("'", "'\\''")
    return f"file '{escaped}'\n"


async def _split_at_keyframes(
    local_input: str,
    segments_dir: str,
    task_label: str,
    chunk_duration_seconds: int,
) -> list[str]:
    os.makedirs(segments_dir, exist_ok=True)
    segment_pattern = os.path.join(segments_dir, "input_%05d.mp4")
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-y",
        "-i",
        local_input,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-sn",
        "-dn",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_duration_seconds),
        "-reset_timestamps",
        "1",
        "-segment_format",
        "mp4",
        segment_pattern,
    ]
    returncode, stdout, stderr, run_error = await _run_process(
        cmd, task_label, "split segments"
    )
    if returncode != 0 or run_error:
        detail = (
            run_error
            or _format_process_output(stdout, stderr)
            or "segment split failed"
        )
        raise RuntimeError(detail)

    segments = sorted(
        os.path.join(segments_dir, filename)
        for filename in os.listdir(segments_dir)
        if filename.startswith("input_") and filename.endswith(".mp4")
    )
    if not segments:
        raise RuntimeError("segment split produced no files")
    return segments


async def _compress_chunked(
    local_input: str,
    output_path: str,
    req: "CompressRequest",
    encoder: str,
    task_label: str,
) -> None:
    chunk_duration_seconds = (
        req.chunk_duration_seconds or COMPRESSION_CHUNK_TARGET_SECONDS
    )
    parallelism = max(1, req.chunk_parallelism or COMPRESSION_CHUNK_PARALLELISM)
    work_dir = os.path.join(SHARED_VOLUME_PATH, f"{task_label}_chunks")
    encoded_dir = os.path.join(work_dir, "encoded")
    tracked_paths: list[str] = []

    try:
        input_segments = await _split_at_keyframes(
            local_input, work_dir, task_label, chunk_duration_seconds
        )
        if len(input_segments) < 2:
            raise RuntimeError(
                "segment split produced one chunk; falling back to single-pass compression"
            )
        await _log_chunk_seams(input_segments, task_label)

        os.makedirs(encoded_dir, exist_ok=True)
        encoded_segments = [
            os.path.join(encoded_dir, f"encoded_{index:05d}.mp4")
            for index, _ in enumerate(input_segments)
        ]
        tracked_paths = [*input_segments, *encoded_segments]
        await _track_temp_files(*tracked_paths)

        log.info(
            f"[{task_label}] Compressing {len(input_segments)} chunks "
            f"(target={chunk_duration_seconds}s, parallelism={parallelism})"
        )
        semaphore = asyncio.Semaphore(parallelism)

        async def _compress_one(index: int, segment_input: str, segment_output: str):
            async with semaphore:
                cmd = _build_ffmpeg_args(segment_input, segment_output, req, encoder)
                returncode, stdout, stderr, run_error = await _run_process(
                    cmd,
                    task_label,
                    f"compress chunk {index + 1}/{len(input_segments)}",
                )
                if returncode != 0 or run_error:
                    detail = (
                        run_error
                        or _format_process_output(stdout, stderr)
                        or "chunk compression failed"
                    )
                    raise RuntimeError(f"chunk {index + 1} failed: {detail}")
                if not os.path.exists(segment_output):
                    raise RuntimeError(
                        f"chunk {index + 1} output missing: {segment_output}"
                    )

        chunk_results = await asyncio.gather(
            *[
                _compress_one(index, segment_input, encoded_segments[index])
                for index, segment_input in enumerate(input_segments)
            ],
            return_exceptions=True,
        )
        chunk_errors = [
            result for result in chunk_results if isinstance(result, Exception)
        ]
        if chunk_errors:
            raise RuntimeError(str(chunk_errors[0]))

        concat_list = os.path.join(work_dir, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as file:
            for segment in encoded_segments:
                file.write(_concat_file_line(segment))

        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output_path,
        ]
        returncode, stdout, stderr, run_error = await _run_process(
            cmd, task_label, "merge chunks"
        )
        if returncode != 0 or run_error:
            detail = (
                run_error
                or _format_process_output(stdout, stderr)
                or "chunk merge failed"
            )
            raise RuntimeError(detail)
    finally:
        await _untrack_temp_files(*tracked_paths)
        _cleanup_tree(work_dir)


def _cleanup_config_status() -> dict[str, object]:
    return {
        "enabled": CLEANUP_ENABLED,
        "interval_seconds": CLEANUP_INTERVAL_SECONDS,
        "temp_file_ttl_seconds": TEMP_FILE_TTL_SECONDS,
        "max_volume_bytes": CLEANUP_MAX_VOLUME_BYTES,
        "min_file_age_seconds": CLEANUP_MIN_FILE_AGE_SECONDS,
        "presigned_url_expiry_seconds": min(S3_PRESIGNED_EXPIRY, 604800),
        "presigned_url_cleanup_grace_seconds": PRESIGNED_URL_CLEANUP_GRACE_SECONDS,
        "storage_cleanup_enabled": STORAGE_CLEANUP_ENABLED,
        "storage_object_ttl_seconds": STORAGE_OBJECT_TTL_SECONDS,
        "storage_cleanup_prefixes": STORAGE_CLEANUP_PREFIXES,
    }


def _shared_root() -> str:
    return os.path.abspath(SHARED_VOLUME_PATH)


def _normalize_path(path: str) -> str:
    return os.path.abspath(path)


def _is_shared_path(path: str) -> bool:
    try:
        return (
            os.path.commonpath([_shared_root(), _normalize_path(path)])
            == _shared_root()
        )
    except ValueError:
        return False


async def _track_temp_files(*paths: str):
    async with _lock:
        for path in paths:
            if path and _is_shared_path(path):
                _active_file_paths.add(_normalize_path(path))


async def _untrack_temp_files(*paths: str):
    async with _lock:
        for path in paths:
            if path:
                _active_file_paths.discard(_normalize_path(path))


async def _protected_paths_snapshot() -> set[str]:
    async with _lock:
        return set(_active_file_paths)


def _remove_stale_file(path: str, reason: str) -> int:
    try:
        size = os.path.getsize(path)
        os.remove(path)
        log.info(f"Removed {reason} temp file: {path} ({size} bytes)")
        return size
    except FileNotFoundError:
        return 0
    except OSError as e:
        log.warning(f"Failed to remove temp file {path}: {e}")
        return 0


async def _cleanup_shared_volume_once():
    if not CLEANUP_ENABLED or not os.path.isdir(SHARED_VOLUME_PATH):
        return

    now = time.time()
    protected_paths = await _protected_paths_snapshot()
    total_bytes = 0
    candidates: list[tuple[float, str, int]] = []

    for root, _, files in os.walk(SHARED_VOLUME_PATH):
        for filename in files:
            path = os.path.abspath(os.path.join(root, filename))
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue

            total_bytes += stat.st_size
            if path in protected_paths:
                continue

            candidates.append((stat.st_mtime, path, stat.st_size))
            if now - stat.st_mtime >= TEMP_FILE_TTL_SECONDS:
                total_bytes -= _remove_stale_file(path, "expired")

    if CLEANUP_MAX_VOLUME_BYTES <= 0 or total_bytes <= CLEANUP_MAX_VOLUME_BYTES:
        return

    for _, path, size in sorted(candidates):
        if total_bytes <= CLEANUP_MAX_VOLUME_BYTES:
            break
        if path in protected_paths or not os.path.exists(path):
            continue
        try:
            file_age = now - os.path.getmtime(path)
        except FileNotFoundError:
            continue
        if file_age < CLEANUP_MIN_FILE_AGE_SECONDS:
            continue
        total_bytes -= _remove_stale_file(path, "over-quota")


def _storage_cleanup_ready() -> bool:
    return (
        STORAGE_CLEANUP_ENABLED
        and STORAGE_OBJECT_TTL_SECONDS > 0
        and bool(STORAGE_CLEANUP_PREFIXES)
        and bool(S3_BUCKET)
        and bool(S3_ACCESS_KEY_ID)
        and bool(S3_SECRET_ACCESS_KEY)
    )


def _cleanup_expired_storage_objects_once() -> int:
    if not _storage_cleanup_ready():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=STORAGE_OBJECT_TTL_SECONDS)
    client = _get_s3_client()
    deleted = 0

    for prefix in STORAGE_CLEANUP_PREFIXES:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            expired_objects = []
            for obj in page.get("Contents", []):
                last_modified = obj.get("LastModified")
                if last_modified is None:
                    continue
                if last_modified.tzinfo is None:
                    last_modified = last_modified.replace(tzinfo=timezone.utc)
                if last_modified < cutoff:
                    expired_objects.append({"Key": obj["Key"]})

            for index in range(0, len(expired_objects), 1000):
                batch = expired_objects[index : index + 1000]
                if not batch:
                    continue
                response = client.delete_objects(
                    Bucket=S3_BUCKET,
                    Delete={"Objects": batch, "Quiet": True},
                )
                deleted += len(batch) - len(response.get("Errors", []))

    if deleted:
        log.info(f"Deleted {deleted} expired {STORAGE_PROVIDER} object(s)")
    return deleted


async def _cleanup_worker():
    if not CLEANUP_ENABLED and not _storage_cleanup_ready():
        log.info("Cleanup worker disabled")
        return

    log.info(
        "Starting cleanup worker "
        f"(path={SHARED_VOLUME_PATH}, ttl={TEMP_FILE_TTL_SECONDS}s, "
        f"interval={CLEANUP_INTERVAL_SECONDS}s, max_bytes={CLEANUP_MAX_VOLUME_BYTES}, "
        f"storage_ttl={STORAGE_OBJECT_TTL_SECONDS}s)"
    )

    while True:
        try:
            if CLEANUP_ENABLED:
                await _cleanup_shared_volume_once()
            if _storage_cleanup_ready():
                await asyncio.to_thread(_cleanup_expired_storage_objects_once)
        except Exception as e:
            log.warning(f"Cleanup pass failed: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


def _queue_capacity() -> int:
    return MAX_CONCURRENT + MAX_QUEUE_SIZE


def _queue_snapshot_locked() -> dict[str, int]:
    queued_tasks = max(0, _queue_size - _active_count)
    return {
        "max_concurrent": MAX_CONCURRENT,
        "max_queue_size": MAX_QUEUE_SIZE,
        "active_tasks": _active_count,
        "queued_tasks": queued_tasks,
        "total_pending": _queue_size,
        "queue_capacity_remaining": max(0, _queue_capacity() - _queue_size),
    }


async def _queue_snapshot() -> dict[str, int]:
    async with _lock:
        return _queue_snapshot_locked()


@asynccontextmanager
async def _queued_task(task_label: str):
    global _queue_size

    async with _lock:
        if _queue_size >= _queue_capacity():
            snapshot = _queue_snapshot_locked()
            detail = (
                f"Compression queue full: {snapshot['queued_tasks']}/{MAX_QUEUE_SIZE} queued, "
                f"{snapshot['active_tasks']}/{MAX_CONCURRENT} active"
            )
            log.warning(f"[{task_label}] {detail}")
            raise HTTPException(status_code=429, detail=detail)

        _queue_size += 1
        queue_position = max(0, _queue_size - MAX_CONCURRENT)
        snapshot = _queue_snapshot_locked()

    try:
        yield queue_position, snapshot
    finally:
        async with _lock:
            _queue_size = max(0, _queue_size - 1)


@asynccontextmanager
async def _running_task():
    global _active_count

    async with _semaphore:
        async with _lock:
            _active_count += 1
            snapshot = _queue_snapshot_locked()

        try:
            yield snapshot
        finally:
            async with _lock:
                _active_count = max(0, _active_count - 1)


class CompressRequest(BaseModel):
    video_paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Input video path or URL list, up to 5 items",
    )
    output_paths: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Optional caller-owned local output paths",
    )
    task_id: str = Field("", description="Task ID for logging")
    codec: str = Field("AV1", description="Target codec: AV1, H264, HEVC, VP9")
    codec_mode: str = Field("CRF", description="Rate control mode: CRF or VBR")
    cq: Optional[int] = Field(
        None,
        ge=0,
        le=63,
        description="Optional explicit CQ override (lower = higher quality)",
    )
    compression_type: Optional[Literal["Low", "Medium", "High"]] = Field(
        None,
        description="Optional inference quality tier used when cq is omitted",
    )
    preset: str = Field("p4", description="Encoder preset")
    target_bitrate: Optional[int] = Field(
        None, description="Target bitrate in bps (for VBR mode)"
    )
    target_width: Optional[int] = Field(
        None, description="Target width for downscaling"
    )
    target_height: Optional[int] = Field(
        None, description="Target height for downscaling"
    )
    chunked: Optional[bool] = Field(
        None, description="Override automatic long-video chunking"
    )
    chunk_duration_seconds: Optional[int] = Field(
        None, description="Target chunk duration"
    )
    chunk_parallelism: Optional[int] = Field(
        None, description="Parallel chunk encodes per request"
    )
    vmaf_threshold: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Competition quality floor communicated to customized solutions",
    )

    @model_validator(mode="after")
    def validate_output_paths(self):
        if self.output_paths and len(self.output_paths) != len(self.video_paths):
            raise ValueError("output_paths must have the same length as video_paths")
        if len(self.output_paths) != len(set(self.output_paths)):
            raise ValueError("output_paths values must be unique")
        return self


class CompressResponse(BaseModel):
    output_paths: list[str] = Field(
        default_factory=list, description="Per-input local output paths"
    )
    output_urls: list[str] = Field(
        default_factory=list, description="Per-input S3 presigned URLs"
    )
    errors: list[Optional[str]] = Field(
        default_factory=list, description="Per-input errors"
    )
    success: bool
    active_tasks: Optional[int] = None
    queued_tasks: Optional[int] = None


class CompetitionCompressionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    def validate_rate_control(self):
        if self.codec_mode == "VBR" and self.target_bitrate is None:
            raise ValueError("VBR competition item requires target_bitrate")
        if self.codec_mode == "CRF" and self.target_bitrate is not None:
            raise ValueError("CRF competition item cannot set target_bitrate")
        return self


class CompetitionCompressionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition_id: str = Field(min_length=1, max_length=64)
    hotkey: str = Field(min_length=1, max_length=128)
    batch_id: str = Field(min_length=1, max_length=128)
    items: list[CompetitionCompressionItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_unique_values(self):
        if len({item.evaluation_id for item in self.items}) != len(self.items):
            raise ValueError("evaluation_id values must be unique")
        if len({item.output_path for item in self.items}) != len(self.items):
            raise ValueError("output_path values must be unique")
        return self


class CompetitionCompressionResult(BaseModel):
    output_path: str | None = None


class CompetitionCompressionResponse(BaseModel):
    results: list[CompetitionCompressionResult]


@app.get("/health")
async def health():
    snapshot = await _queue_snapshot()
    return {
        "status": "ok",
        **snapshot,
        "storage": _storage_config_status(),
        "cleanup": _cleanup_config_status(),
        "compression": _compression_config_status(),
        "competition_local_io": {
            "remote_io_disabled": DISABLE_REMOTE_IO,
            "input_root": COMPETITION_INPUT_ROOT,
            "output_root": COMPETITION_OUTPUT_ROOT,
        },
    }


@app.get("/queue")
async def queue_status():
    return await _queue_snapshot()


async def _compress_one(
    req: CompressRequest,
    input_video: str,
    task_label: str,
    *,
    requested_output_path: str | None = None,
) -> CompressResponse:
    remote_mode = _is_url(input_video)
    encoder = CODEC_MAP.get(req.codec.upper(), "av1_nvenc")
    resolved_cq = resolve_compression_cq(
        explicit_cq=req.cq,
        compression_type=req.compression_type,
        vmaf_threshold=req.vmaf_threshold,
    )

    if remote_mode and DISABLE_REMOTE_IO:
        return CompressResponse(
            success=False,
            errors=["Remote URL input is disabled for this local-only service"],
        )

    local_input = ""
    output_path = ""
    output_filename = ""
    returncode: int | None = None
    stdout = b""
    stderr = b""
    run_error = ""

    try:
        async with _queued_task(task_label) as (queue_position, snapshot):
            log.info(
                f"[{task_label}] Queued compression "
                f"(codec={encoder}, cq={resolved_cq}, position={queue_position}, "
                f"waiting={snapshot['queued_tasks']}, remote={remote_mode})"
            )

            os.makedirs(SHARED_VOLUME_PATH, exist_ok=True)

            # --- Resolve input to local path ---
            if remote_mode:
                local_input = os.path.join(
                    SHARED_VOLUME_PATH, f"{task_label}_input.mp4"
                )
                await _track_temp_files(local_input)
                try:
                    await _download_url(input_video, local_input)
                except Exception as e:
                    _cleanup(local_input)
                    return CompressResponse(
                        success=False, errors=[f"Failed to download input: {e}"]
                    )
            else:
                local_input = input_video
                if not os.path.exists(local_input):
                    raise HTTPException(
                        status_code=400, detail=f"Input file not found: {local_input}"
                    )
                await _track_temp_files(local_input)

            basename = os.path.splitext(os.path.basename(local_input))[0]
            output_filename = f"{basename}_compressed.mp4"
            output_path = requested_output_path or os.path.join(
                SHARED_VOLUME_PATH, output_filename
            )
            await _track_temp_files(output_path)

            duration_seconds = await _probe_duration_seconds(local_input, task_label)
            use_chunked = _should_chunk(req, duration_seconds)

            async with _running_task() as running_snapshot:
                log.info(
                    f"[{task_label}] Starting compression "
                    f"(active={running_snapshot['active_tasks']}/{MAX_CONCURRENT}, "
                    f"queued={running_snapshot['queued_tasks']}, "
                    f"duration={duration_seconds}, chunked={use_chunked})"
                )

                if use_chunked:
                    try:
                        await _compress_chunked(
                            local_input, output_path, req, encoder, task_label
                        )
                        returncode = 0
                    except RuntimeError as e:
                        if "falling back to single-pass compression" in str(e):
                            log.warning(f"[{task_label}] {e}")
                            cmd = _build_ffmpeg_args(
                                local_input, output_path, req, encoder
                            )
                            returncode, stdout, stderr, run_error = await _run_process(
                                cmd,
                                task_label,
                                "single-pass compression fallback",
                            )
                        else:
                            run_error = str(e)
                            log.error(
                                f"[{task_label}] Chunked compression failed: {run_error}"
                            )
                else:
                    cmd = _build_ffmpeg_args(local_input, output_path, req, encoder)
                    returncode, stdout, stderr, run_error = await _run_process(
                        cmd,
                        task_label,
                        "single-pass compression",
                    )

        snapshot = await _queue_snapshot()
        stats = dict(
            active_tasks=snapshot["active_tasks"], queued_tasks=snapshot["queued_tasks"]
        )

        if returncode is None:
            _cleanup(output_path)
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(
                success=False,
                errors=[run_error or "compression did not start"],
                **stats,
            )

        if returncode != 0:
            err_msg = (
                run_error
                or _format_process_output(stdout, stderr)
                or "compression failed without output"
            )
            log.error(f"[{task_label}] ffmpeg failed (rc={returncode}): {err_msg}")
            _cleanup(output_path)
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, errors=[err_msg], **stats)

        if not os.path.exists(output_path):
            log.error(f"[{task_label}] Output file not found: {output_path}")
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(
                success=False, errors=["Output file not created"], **stats
            )

        # --- Remote mode: upload result to S3, return URL ---
        if remote_mode:
            try:
                s3_key = f"processing/{task_label}/{output_filename}"
                output_url = _upload_to_s3(output_path, s3_key)
                log.info(
                    f"[{task_label}] Compression complete (remote): {output_url[:80]}..."
                )
                return CompressResponse(
                    output_urls=[output_url], errors=[None], success=True, **stats
                )
            except Exception as e:
                log.error(f"[{task_label}] S3 upload failed: {e}")
                return CompressResponse(
                    success=False, errors=[f"S3 upload failed: {e}"], **stats
                )
            finally:
                _cleanup(local_input, output_path)

        log.info(f"[{task_label}] Compression complete (local): {output_path}")
        return CompressResponse(
            output_paths=[output_path], errors=[None], success=True, **stats
        )
    finally:
        await _untrack_temp_files(local_input, output_path)


def _combine_compress_responses(responses: list[CompressResponse]) -> CompressResponse:
    output_paths = [
        response.output_paths[0] if response.output_paths else ""
        for response in responses
    ]
    output_urls = [
        response.output_urls[0] if response.output_urls else ""
        for response in responses
    ]
    errors = [response.errors[0] if response.errors else None for response in responses]
    success = all(response.success for response in responses)
    latest = responses[-1] if responses else None

    return CompressResponse(
        output_paths=output_paths,
        output_urls=output_urls,
        errors=errors,
        success=success,
        active_tasks=latest.active_tasks if latest else None,
        queued_tasks=latest.queued_tasks if latest else None,
    )


def _competition_input_path(raw_path: str) -> Path:
    if _is_url(raw_path) or not os.path.isabs(raw_path):
        raise ValueError("competition input must be an absolute local path")
    root = Path(COMPETITION_INPUT_ROOT).resolve(strict=True)
    path = Path(raw_path).resolve(strict=True)
    if path == root or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("competition input must be a file below /evaluation-inputs")
    return path


def _competition_output_path(raw_path: str) -> Path:
    if (
        _is_url(raw_path)
        or not os.path.isabs(raw_path)
        or not raw_path.lower().endswith(".mp4")
    ):
        raise ValueError("competition output must be an absolute local MP4 path")
    root = Path(COMPETITION_OUTPUT_ROOT).resolve(strict=True)
    path = Path(raw_path)
    if ".." in path.parts:
        raise ValueError("competition output cannot contain traversal")
    lexical_root = Path(COMPETITION_OUTPUT_ROOT).absolute()
    if path == lexical_root or not path.is_relative_to(lexical_root):
        raise ValueError("competition output must be below /output")
    parent = path.parent
    if not parent.is_dir():
        raise ValueError("competition output parent must already exist")
    resolved = path.resolve(strict=False)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("competition output must be below /output")
    current = root
    for part in resolved.relative_to(root).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError("competition output cannot traverse symlinks")
    if path.exists() or path.is_symlink():
        raise ValueError("competition output must not overwrite an existing path")
    return resolved


async def _compress_competition(
    req: CompetitionCompressionRequest,
) -> CompetitionCompressionResponse:
    if not DISABLE_REMOTE_IO:
        raise HTTPException(
            status_code=503, detail="competition route requires DISABLE_REMOTE_IO=true"
        )
    prepared: list[tuple[CompetitionCompressionItem, Path, Path]] = []
    try:
        for item in req.items:
            prepared.append(
                (
                    item,
                    _competition_input_path(item.input_path),
                    _competition_output_path(item.output_path),
                )
            )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log.info(
        "[competition:%s] Batch received (hotkey=%s, items=%s)",
        req.batch_id,
        req.hotkey,
        [item.evaluation_id for item, _, _ in prepared],
    )

    async def run(
        item: CompetitionCompressionItem, input_path: Path, output_path: Path
    ) -> CompetitionCompressionResult:
        legacy = CompressRequest(
            video_paths=[str(input_path)],
            task_id=item.evaluation_id,
            codec=item.codec,
            codec_mode=item.codec_mode,
            target_bitrate=item.target_bitrate,
            vmaf_threshold=item.vmaf_threshold,
        )
        response = await _compress_one(
            legacy,
            str(input_path),
            item.evaluation_id,
            requested_output_path=str(output_path),
        )
        if not response.success:
            log.error(
                "[%s] Competition compression failed: %s",
                item.evaluation_id,
                response.errors[0] if response.errors else "compression failed",
            )
            return CompetitionCompressionResult(output_path=None)
        return CompetitionCompressionResult(output_path=item.output_path)

    results = await asyncio.gather(*(run(*item) for item in prepared))
    log.info(
        "[competition:%s] Batch complete (scored_outputs=%s, failed_outputs=%s)",
        req.batch_id,
        sum(result.output_path is not None for result in results),
        sum(result.output_path is None for result in results),
    )
    return CompetitionCompressionResponse(results=list(results))


@app.post("/compress", response_model=CompressResponse | CompetitionCompressionResponse)
async def compress(req: CompressRequest | CompetitionCompressionRequest):
    if isinstance(req, CompetitionCompressionRequest):
        return await _compress_competition(req)

    input_videos = [video_path.strip() for video_path in req.video_paths]
    if any(not video_path for video_path in input_videos):
        raise HTTPException(status_code=400, detail="video_paths entries are required")

    base_task_id = req.task_id or uuid.uuid4().hex[:8]
    requested_outputs: list[str | None] = [None] * len(input_videos)
    if req.output_paths:
        if not DISABLE_REMOTE_IO:
            raise HTTPException(
                status_code=400,
                detail="caller-owned output_paths require DISABLE_REMOTE_IO=true",
            )
        try:
            requested_outputs = [
                str(_competition_output_path(path)) for path in req.output_paths
            ]
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    responses = await asyncio.gather(
        *[
            _compress_one(
                req,
                input_video,
                base_task_id
                if len(input_videos) == 1
                else f"{base_task_id}-{index + 1}",
                requested_output_path=requested_outputs[index],
            )
            for index, input_video in enumerate(input_videos)
        ]
    )

    return (
        responses[0] if len(responses) == 1 else _combine_compress_responses(responses)
    )
