"""
Compression microservice — wraps the ffmpeg binary installed in this container.

Accepts a video path (on shared volume) OR a URL, codec, and quality settings,
runs GPU-accelerated ffmpeg compression, returns the output path or S3 URL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3
import httpx
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE_COMPRESSION") or os.getenv("MAX_QUEUE_SIZE", "5"))
DISABLE_REMOTE_IO = os.getenv("DISABLE_REMOTE_IO", "false").lower() in ("1", "true", "yes")

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
PRESIGNED_URL_CLEANUP_GRACE_SECONDS = int(os.getenv("PRESIGNED_URL_CLEANUP_GRACE_SECONDS", "600"))
TEMP_FILE_TTL_SECONDS = int(
    os.getenv("MINER_TEMP_FILE_TTL_SECONDS")
    or os.getenv("TEMP_FILE_TTL_SECONDS")
    or str(min(S3_PRESIGNED_EXPIRY, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)
CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("MINER_CLEANUP_INTERVAL_SECONDS") or os.getenv("CLEANUP_INTERVAL_SECONDS") or "300"
)
CLEANUP_MAX_VOLUME_BYTES = int(
    os.getenv("MINER_CLEANUP_MAX_VOLUME_BYTES") or os.getenv("CLEANUP_MAX_VOLUME_BYTES") or "9000000000"
)
CLEANUP_MIN_FILE_AGE_SECONDS = int(
    os.getenv("MINER_CLEANUP_MIN_FILE_AGE_SECONDS") or os.getenv("CLEANUP_MIN_FILE_AGE_SECONDS") or "60"
)
CLEANUP_ENABLED = os.getenv("MINER_CLEANUP_ENABLED", os.getenv("CLEANUP_ENABLED", "true")).lower() in (
    "1",
    "true",
    "yes",
)
STORAGE_CLEANUP_ENABLED = os.getenv(
    "MINER_STORAGE_CLEANUP_ENABLED", os.getenv("STORAGE_CLEANUP_ENABLED", "true")
).lower() in ("1", "true", "yes")
STORAGE_CLEANUP_PREFIXES = [
    prefix.strip()
    for prefix in os.getenv("MINER_STORAGE_CLEANUP_PREFIXES", "processing/,upscaling/").split(",")
    if prefix.strip()
]
STORAGE_OBJECT_TTL_SECONDS = int(
    os.getenv("MINER_STORAGE_OBJECT_TTL_SECONDS")
    or os.getenv("STORAGE_OBJECT_TTL_SECONDS")
    or str(min(S3_PRESIGNED_EXPIRY, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)
COMPRESSION_CHUNKING_ENABLED = os.getenv("COMPRESSION_CHUNKING_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
COMPRESSION_CHUNK_MIN_DURATION_SECONDS = int(os.getenv("COMPRESSION_CHUNK_MIN_DURATION_SECONDS", "1200"))
COMPRESSION_CHUNK_TARGET_SECONDS = int(os.getenv("COMPRESSION_CHUNK_TARGET_SECONDS", "600"))
COMPRESSION_CHUNK_PARALLELISM = max(1, int(os.getenv("COMPRESSION_CHUNK_PARALLELISM", "2")))
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

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
    log.info(f"Downloaded {os.path.getsize(dest) / (1024*1024):.1f} MB → {dest}")


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


async def _run_process(cmd: list[str], task_label: str, step: str) -> tuple[int | None, bytes, bytes, str]:
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
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    returncode, stdout, stderr, run_error = await _run_process(cmd, task_label, "ffprobe duration")
    if returncode != 0 or run_error:
        detail = run_error or stderr.decode(errors="replace").strip()
        log.warning(f"[{task_label}] Failed to probe duration: {detail}")
        return None
    try:
        return float(stdout.decode().strip())
    except ValueError:
        log.warning(f"[{task_label}] ffprobe returned invalid duration: {stdout!r}")
        return None


def _build_ffmpeg_args(local_input: str, output_path: str, req: "CompressRequest", encoder: str) -> list[str]:
    ffmpeg_args = [
        FFMPEG_BIN,
        "-y",
        "-hwaccel", "cuda",
        "-i", local_input,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", encoder,
    ]

    if req.codec_mode == "VBR" and req.target_bitrate:
        ffmpeg_args.extend(["-b:v", str(req.target_bitrate)])
    else:
        ffmpeg_args.extend(["-cq", str(req.cq)])

    ffmpeg_args.extend(["-preset", req.preset])

    if req.target_width and req.target_height:
        w = req.target_width if req.target_width % 2 == 0 else req.target_width - 1
        h = req.target_height if req.target_height % 2 == 0 else req.target_height - 1
        ffmpeg_args.extend(["-vf", f"scale={w}:{h}"])

    ffmpeg_args.extend(["-c:a", "copy", "-sn", "-dn", "-movflags", "+faststart", output_path])
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
        "-i", local_input,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c", "copy",
        "-sn",
        "-dn",
        "-f", "segment",
        "-segment_time", str(chunk_duration_seconds),
        "-reset_timestamps", "1",
        "-segment_format", "mp4",
        segment_pattern,
    ]
    returncode, stdout, stderr, run_error = await _run_process(cmd, task_label, "split segments")
    if returncode != 0 or run_error:
        detail = run_error or _format_process_output(stdout, stderr) or "segment split failed"
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
    chunk_duration_seconds = req.chunk_duration_seconds or COMPRESSION_CHUNK_TARGET_SECONDS
    parallelism = max(1, req.chunk_parallelism or COMPRESSION_CHUNK_PARALLELISM)
    work_dir = os.path.join(SHARED_VOLUME_PATH, f"{task_label}_chunks")
    encoded_dir = os.path.join(work_dir, "encoded")
    tracked_paths: list[str] = []

    try:
        input_segments = await _split_at_keyframes(local_input, work_dir, task_label, chunk_duration_seconds)
        if len(input_segments) < 2:
            raise RuntimeError("segment split produced one chunk; falling back to single-pass compression")

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
                    detail = run_error or _format_process_output(stdout, stderr) or "chunk compression failed"
                    raise RuntimeError(f"chunk {index + 1} failed: {detail}")
                if not os.path.exists(segment_output):
                    raise RuntimeError(f"chunk {index + 1} output missing: {segment_output}")

        chunk_results = await asyncio.gather(
            *[
                _compress_one(index, segment_input, encoded_segments[index])
                for index, segment_input in enumerate(input_segments)
            ],
            return_exceptions=True,
        )
        chunk_errors = [result for result in chunk_results if isinstance(result, Exception)]
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
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        returncode, stdout, stderr, run_error = await _run_process(cmd, task_label, "merge chunks")
        if returncode != 0 or run_error:
            detail = run_error or _format_process_output(stdout, stderr) or "chunk merge failed"
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
        return os.path.commonpath([_shared_root(), _normalize_path(path)]) == _shared_root()
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
    video_path: str = Field(..., description="Local path or URL to input video")
    task_id: str = Field("", description="Task ID for logging")
    codec: str = Field("AV1", description="Target codec: AV1, H264, HEVC, VP9")
    codec_mode: str = Field("CRF", description="Rate control mode: CRF or VBR")
    cq: int = Field(35, description="Constant quality value (lower = higher quality)")
    preset: str = Field("p4", description="Encoder preset")
    target_bitrate: Optional[int] = Field(None, description="Target bitrate in bps (for VBR mode)")
    target_width: Optional[int] = Field(None, description="Target width for downscaling")
    target_height: Optional[int] = Field(None, description="Target height for downscaling")
    chunked: Optional[bool] = Field(None, description="Override automatic long-video chunking")
    chunk_duration_seconds: Optional[int] = Field(None, description="Target chunk duration")
    chunk_parallelism: Optional[int] = Field(None, description="Parallel chunk encodes per request")


class CompressResponse(BaseModel):
    output_path: str = Field("", description="Local path (local mode) or empty (remote mode)")
    output_url: str = Field("", description="S3 presigned URL (remote mode) or empty (local mode)")
    success: bool
    error: Optional[str] = None
    active_tasks: Optional[int] = None
    queued_tasks: Optional[int] = None


@app.get("/health")
async def health():
    snapshot = await _queue_snapshot()
    return {
        "status": "ok",
        **snapshot,
        "storage": _storage_config_status(),
        "cleanup": _cleanup_config_status(),
        "compression": _compression_config_status(),
    }


@app.get("/queue")
async def queue_status():
    return await _queue_snapshot()


@app.post("/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    task_label = req.task_id or uuid.uuid4().hex[:8]
    remote_mode = _is_url(req.video_path)
    encoder = CODEC_MAP.get(req.codec.upper(), "av1_nvenc")

    if remote_mode and DISABLE_REMOTE_IO:
        return CompressResponse(success=False, error="Remote URL input is disabled for this local-only service")

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
                f"(codec={encoder}, cq={req.cq}, position={queue_position}, "
                f"waiting={snapshot['queued_tasks']}, remote={remote_mode})"
            )

            os.makedirs(SHARED_VOLUME_PATH, exist_ok=True)

            # --- Resolve input to local path ---
            if remote_mode:
                local_input = os.path.join(SHARED_VOLUME_PATH, f"{task_label}_input.mp4")
                await _track_temp_files(local_input)
                try:
                    await _download_url(req.video_path, local_input)
                except Exception as e:
                    _cleanup(local_input)
                    return CompressResponse(success=False, error=f"Failed to download input: {e}")
            else:
                local_input = req.video_path
                if not os.path.exists(local_input):
                    raise HTTPException(status_code=400, detail=f"Input file not found: {local_input}")
                await _track_temp_files(local_input)

            basename = os.path.splitext(os.path.basename(local_input))[0]
            output_filename = f"{basename}_compressed.mp4"
            output_path = os.path.join(SHARED_VOLUME_PATH, output_filename)
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
                        await _compress_chunked(local_input, output_path, req, encoder, task_label)
                        returncode = 0
                    except RuntimeError as e:
                        if "falling back to single-pass compression" in str(e):
                            log.warning(f"[{task_label}] {e}")
                            cmd = _build_ffmpeg_args(local_input, output_path, req, encoder)
                            returncode, stdout, stderr, run_error = await _run_process(
                                cmd,
                                task_label,
                                "single-pass compression fallback",
                            )
                        else:
                            run_error = str(e)
                            log.error(f"[{task_label}] Chunked compression failed: {run_error}")
                else:
                    cmd = _build_ffmpeg_args(local_input, output_path, req, encoder)
                    returncode, stdout, stderr, run_error = await _run_process(
                        cmd,
                        task_label,
                        "single-pass compression",
                    )

        snapshot = await _queue_snapshot()
        stats = dict(active_tasks=snapshot["active_tasks"], queued_tasks=snapshot["queued_tasks"])

        if returncode is None:
            _cleanup(output_path)
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, error=run_error or "compression did not start", **stats)

        if returncode != 0:
            err_msg = run_error or _format_process_output(stdout, stderr) or "compression failed without output"
            log.error(f"[{task_label}] ffmpeg failed (rc={returncode}): {err_msg}")
            _cleanup(output_path)
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, error=err_msg, **stats)

        if not os.path.exists(output_path):
            log.error(f"[{task_label}] Output file not found: {output_path}")
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, error="Output file not created", **stats)

        # --- Remote mode: upload result to S3, return URL ---
        if remote_mode:
            try:
                s3_key = f"processing/{task_label}/{output_filename}"
                output_url = _upload_to_s3(output_path, s3_key)
                log.info(f"[{task_label}] Compression complete (remote): {output_url[:80]}...")
                return CompressResponse(output_url=output_url, success=True, **stats)
            except Exception as e:
                log.error(f"[{task_label}] S3 upload failed: {e}")
                return CompressResponse(success=False, error=f"S3 upload failed: {e}", **stats)
            finally:
                _cleanup(local_input, output_path)

        log.info(f"[{task_label}] Compression complete (local): {output_path}")
        return CompressResponse(output_path=output_path, success=True, **stats)
    finally:
        await _untrack_temp_files(local_input, output_path)
