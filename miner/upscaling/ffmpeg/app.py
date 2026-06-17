from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Optional

import boto3
import httpx
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("upscaling-ffmpeg")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_cleanup_worker())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title="FFmpeg Upscaling Service", lifespan=lifespan)

SHARED_VOLUME_PATH = os.getenv("SHARED_VOLUME_PATH", "/tmp/organic-proxy")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_UPSCALING", "2"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE_UPSCALING") or os.getenv("MAX_QUEUE_SIZE", "5"))
FFMPEG_CODEC = os.getenv("FFMPEG_CODEC", "hevc_nvenc")
FFMPEG_PRESET = os.getenv("FFMPEG_PRESET", "p7")
FFMPEG_CQ = int(os.getenv("FFMPEG_CQ", "35"))
DISABLE_REMOTE_IO = os.getenv("DISABLE_REMOTE_IO", "false").lower() in ("1", "true", "yes")

STORAGE_PROVIDER = os.getenv("MINER_STORAGE_PROVIDER", "s3").lower()
S3_REGION = os.getenv("MINER_STORAGE_S3_REGION", "us-east-1").strip() or "us-east-1"
S3_BUCKET = os.getenv("MINER_STORAGE_S3_BUCKET_NAME", "").strip()
S3_ACCESS_KEY_ID = os.getenv("MINER_STORAGE_S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("MINER_STORAGE_S3_SECRET_ACCESS_KEY", "").strip()
S3_ENDPOINT_URL = (os.getenv("MINER_STORAGE_S3_ENDPOINT_URL") or os.getenv("MINER_STORAGE_S3_ENDPOINT") or "").strip()
S3_PRESIGNED_EXPIRY = int(os.getenv("MINER_STORAGE_S3_PRESIGNED_EXPIRY") or os.getenv("S3_PRESIGNED_EXPIRY") or "3600")
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

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_size = 0
_active_count = 0
_active_file_paths: set[str] = set()
_lock = asyncio.Lock()


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _get_s3_client():
    kwargs = {
        "region_name": S3_REGION,
        "aws_access_key_id": S3_ACCESS_KEY_ID or None,
        "aws_secret_access_key": S3_SECRET_ACCESS_KEY or None,
        "config": Config(signature_version="s3v4"),
    }
    if S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = S3_ENDPOINT_URL
    return boto3.client("s3", **kwargs)


def _validate_s3_config():
    missing = [k for k, v in {
        "MINER_STORAGE_S3_BUCKET_NAME": S3_BUCKET,
        "MINER_STORAGE_S3_ACCESS_KEY_ID": S3_ACCESS_KEY_ID,
        "MINER_STORAGE_S3_SECRET_ACCESS_KEY": S3_SECRET_ACCESS_KEY,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing storage configuration: {', '.join(missing)}")


async def _download_url(url: str, dest: str):
    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)


def _upload_to_s3(local_path: str, key: str) -> str:
    _validate_s3_config()
    client = _get_s3_client()
    client.upload_file(local_path, S3_BUCKET, key)
    return client.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=min(S3_PRESIGNED_EXPIRY, 604800))


def _cleanup(*paths: str):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _cleanup_config_status() -> dict[str, object]:
    return {
        "enabled": CLEANUP_ENABLED,
        "interval_seconds": CLEANUP_INTERVAL_SECONDS,
        "temp_file_ttl_seconds": TEMP_FILE_TTL_SECONDS,
        "max_volume_bytes": CLEANUP_MAX_VOLUME_BYTES,
        "min_file_age_seconds": CLEANUP_MIN_FILE_AGE_SECONDS,
        "presigned_url_expiry_seconds": min(S3_PRESIGNED_EXPIRY, 604800),
        "presigned_url_cleanup_grace_seconds": PRESIGNED_URL_CLEANUP_GRACE_SECONDS,
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


async def _cleanup_worker():
    if not CLEANUP_ENABLED:
        log.info("Shared-volume cleanup worker disabled")
        return

    log.info(
        "Starting shared-volume cleanup worker "
        f"(path={SHARED_VOLUME_PATH}, ttl={TEMP_FILE_TTL_SECONDS}s, "
        f"interval={CLEANUP_INTERVAL_SECONDS}s, max_bytes={CLEANUP_MAX_VOLUME_BYTES})"
    )

    while True:
        try:
            await _cleanup_shared_volume_once()
        except Exception as e:
            log.warning(f"Shared-volume cleanup pass failed: {e}")
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
                f"Upscaling queue full: {snapshot['queued_tasks']}/{MAX_QUEUE_SIZE} queued, "
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


class UpscaleRequest(BaseModel):
    video_path: str
    scale: int = Field(..., description="Upscaling factor: 2 or 4")
    task_id: str = ""


class UpscaleResponse(BaseModel):
    output_path: str = ""
    output_url: str = ""
    success: bool
    error: Optional[str] = None
    active_tasks: Optional[int] = None
    queued_tasks: Optional[int] = None


@app.get("/health")
async def health():
    snapshot = await _queue_snapshot()
    return {"status": "ok", **snapshot, "cleanup": _cleanup_config_status()}


@app.get("/queue")
async def queue_status():
    return await _queue_snapshot()


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale(req: UpscaleRequest):
    if req.scale not in (2, 4):
        raise HTTPException(status_code=400, detail="scale must be 2 or 4")

    task = req.task_id or uuid.uuid4().hex[:8]
    remote = _is_url(req.video_path)
    if remote and DISABLE_REMOTE_IO:
        return UpscaleResponse(success=False, error="Remote URL input is disabled for this local-only service")

    local_input = ""
    output_path = ""
    result = None
    run_error = ""

    try:
        async with _queued_task(task) as (queue_position, snapshot):
            log.info(
                f"[{task}] Queued {req.scale}x ffmpeg upscale "
                f"(position={queue_position}, waiting={snapshot['queued_tasks']}, remote={remote})"
            )

            os.makedirs(SHARED_VOLUME_PATH, exist_ok=True)

            local_input = os.path.join(SHARED_VOLUME_PATH, f"{task}_input.mp4") if remote else req.video_path
            await _track_temp_files(local_input)
            if remote:
                try:
                    await _download_url(req.video_path, local_input)
                except Exception as e:
                    _cleanup(local_input)
                    return UpscaleResponse(success=False, error=f"Failed to download input: {e}")
            elif not os.path.exists(local_input):
                raise HTTPException(status_code=400, detail=f"Input file not found: {local_input}")

            base = os.path.splitext(os.path.basename(local_input))[0]
            output_path = os.path.join(SHARED_VOLUME_PATH, f"{base}_upscaled_{req.scale}x.mp4")
            await _track_temp_files(output_path)

            cmd = [
                "ffmpeg", "-y", "-hwaccel", "cuda", "-i", local_input,
                "-vf", f"scale=iw*{req.scale}:ih*{req.scale}:flags=lanczos",
                "-c:v", FFMPEG_CODEC, "-preset", FFMPEG_PRESET, "-cq", str(FFMPEG_CQ),
                "-c:a", "copy", output_path,
            ]

            async with _running_task() as running_snapshot:
                log.info(
                    f"[{task}] Starting ffmpeg upscale "
                    f"(active={running_snapshot['active_tasks']}/{MAX_CONCURRENT}, "
                    f"queued={running_snapshot['queued_tasks']})"
                )
                try:
                    result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
                except FileNotFoundError:
                    run_error = "ffmpeg binary not found"
                    log.error(f"[{task}] {run_error}")
                except OSError as e:
                    run_error = f"Failed to start ffmpeg: {e}"
                    log.error(f"[{task}] {run_error}")

        snapshot = await _queue_snapshot()
        stats = dict(active_tasks=snapshot["active_tasks"], queued_tasks=snapshot["queued_tasks"])

        if result is None:
            _cleanup(output_path)
            if remote:
                _cleanup(local_input)
            return UpscaleResponse(success=False, error=run_error or "ffmpeg did not start", **stats)

        if result.returncode != 0:
            _cleanup(output_path)
            if remote:
                _cleanup(local_input)
            return UpscaleResponse(success=False, error=result.stderr[-1000:], **stats)
        if remote:
            try:
                key = f"upscaling/{task}/{os.path.basename(output_path)}"
                url = _upload_to_s3(output_path, key)
                return UpscaleResponse(success=True, output_url=url, **stats)
            except Exception as e:
                log.error(f"[{task}] S3 upload failed: {e}")
                return UpscaleResponse(success=False, error=f"S3 upload failed: {e}", **stats)
            finally:
                _cleanup(local_input, output_path)
        return UpscaleResponse(success=True, output_path=output_path, **stats)
    finally:
        await _untrack_temp_files(local_input, output_path)
