"""
Upscaling microservice — runs the installed Video2X CLI.

Accepts a video path (on shared volume) OR a URL and scale factor (2 or 4),
runs video2x locally in this service container, and returns the output path or
uploaded S3 URL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
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
log = logging.getLogger("upscaling")

app = FastAPI(title="Video Upscaling Service")

VIDEO2X_BIN = os.getenv("VIDEO2X_BIN", "video2x")
VIDEO2X_GPU = (os.getenv("VIDEO2X_GPU") or os.getenv("VIDEO2X_DEVICE") or "0").strip()
SHARED_VOLUME_PATH = os.getenv("SHARED_VOLUME_PATH", "/tmp/organic-proxy")
DISABLE_REMOTE_IO = os.getenv("DISABLE_REMOTE_IO", "false").lower() in ("1", "true", "yes")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_UPSCALING", "2"))

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

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_size = 0
_active_count = 0
_lock = asyncio.Lock()


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
    """Download a URL (presigned S3 or direct) to local path."""
    log.info(f"Downloading {url[:80]}... → {dest}")
    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
    log.info(f"Downloaded {os.path.getsize(dest) / (1024*1024):.1f} MB → {dest}")


def _upload_to_s3(local_path: str, key: str) -> str:
    """Upload file to storage and return presigned URL."""
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


class UpscaleRequest(BaseModel):
    video_path: str = Field(..., description="Local path or URL to input video")
    scale: int = Field(..., description="Upscaling factor: 2 or 4")
    task_id: str = Field("", description="Task ID for logging")
    model: str = Field("realesr-animevideov3", description="RealESRGAN model name")
    codec: str = Field("hevc_nvenc", description="Output video codec")
    cq: int = Field(35, description="Constant quality value")


class UpscaleResponse(BaseModel):
    output_path: str = Field("", description="Local path (local mode) or empty (remote mode)")
    output_url: str = Field("", description="S3 presigned URL (remote mode) or empty (local mode)")
    success: bool
    error: Optional[str] = None
    queue_position: Optional[int] = None
    active_tasks: Optional[int] = None
    queued_tasks: Optional[int] = None


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "video2x_binary": VIDEO2X_BIN,
        "video2x_gpu": VIDEO2X_GPU,
        "max_concurrent": MAX_CONCURRENT,
        "active_tasks": _active_count,
        "queued_tasks": max(0, _queue_size - _active_count),
        "storage": _storage_config_status(),
    }


@app.get("/queue")
async def queue_status():
    return {
        "max_concurrent": MAX_CONCURRENT,
        "active_tasks": _active_count,
        "queued_tasks": max(0, _queue_size - _active_count),
        "total_pending": _queue_size,
    }


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale(req: UpscaleRequest):
    if req.scale not in (2, 4):
        raise HTTPException(status_code=400, detail="scale must be 2 or 4")

    task_label = req.task_id or uuid.uuid4().hex[:8]
    remote_mode = _is_url(req.video_path)

    if remote_mode and DISABLE_REMOTE_IO:
        return UpscaleResponse(success=False, error="Remote URL input is disabled for this local-only service")

    # --- Resolve input to local path ---
    if remote_mode:
        local_input = os.path.join(SHARED_VOLUME_PATH, f"{task_label}_input.mp4")
        try:
            await _download_url(req.video_path, local_input)
        except Exception as e:
            return UpscaleResponse(success=False, error=f"Failed to download input: {e}")
    else:
        local_input = req.video_path
        if not os.path.exists(local_input):
            raise HTTPException(status_code=400, detail=f"Input file not found: {local_input}")

    basename = os.path.splitext(os.path.basename(local_input))[0]
    output_filename = f"{basename}_upscaled_{req.scale}x.mp4"
    output_path = os.path.join(SHARED_VOLUME_PATH, output_filename)

    cmd = [
        VIDEO2X_BIN,
        "-i", local_input,
        "-o", output_path,
        "-p", "realesrgan",
        "-s", str(req.scale),
    ]
    if VIDEO2X_GPU:
        cmd.extend(["-g", VIDEO2X_GPU])
    cmd.extend([
        "--realesrgan-model", req.model,
        "-c", req.codec,
        "-e", "preset=p7",
        "-e", f"cq={req.cq}",
        "-e", "profile=main",
        "-e", "pix_fmt=yuv420p",
        "-e", "sar=1:1",
        "-e", "color_primaries=bt709",
        "-e", "color_trc=bt709",
        "-e", "colorspace=bt709",
        "-e", "movflags=+faststart",
    ])

    global _queue_size, _active_count

    async with _lock:
        _queue_size += 1
        position = _queue_size
        queued = max(0, _queue_size - _active_count)

    log.info(f"[{task_label}] Queued {req.scale}x upscale (position={position}, waiting={queued}, remote={remote_mode})")

    proc = None
    stderr = b""
    run_error = ""
    try:
        async with _semaphore:
            async with _lock:
                _active_count += 1
                queued_now = max(0, _queue_size - _active_count)

            log.info(f"[{task_label}] Starting {req.scale}x upscale (active={_active_count}/{MAX_CONCURRENT}, queued={queued_now})")
            log.info(f"[{task_label}] Command: {' '.join(cmd)}")

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
            except FileNotFoundError:
                run_error = f"Video2X binary not found: {VIDEO2X_BIN}"
                log.error(f"[{task_label}] {run_error}")
            except OSError as e:
                run_error = f"Failed to start Video2X: {e}"
                log.error(f"[{task_label}] {run_error}")
            finally:
                async with _lock:
                    _active_count -= 1
    finally:
        async with _lock:
            _queue_size -= 1

    stats = dict(active_tasks=_active_count, queued_tasks=max(0, _queue_size - _active_count))

    if proc is None:
        if remote_mode:
            _cleanup(local_input)
        return UpscaleResponse(success=False, error=run_error or "Video2X did not start", **stats)

    if proc.returncode != 0:
        err_msg = stderr.decode(errors="replace").strip()
        log.error(f"[{task_label}] video2x failed (rc={proc.returncode}): {err_msg}")
        if remote_mode:
            _cleanup(local_input)
        return UpscaleResponse(success=False, error=err_msg, **stats)

    if not os.path.exists(output_path):
        log.error(f"[{task_label}] Output file not found after video2x: {output_path}")
        if remote_mode:
            _cleanup(local_input)
        return UpscaleResponse(success=False, error="Output file not created", **stats)

    # --- Remote mode: upload result to S3, return URL ---
    if remote_mode:
        try:
            s3_key = f"processing/{task_label}/{output_filename}"
            output_url = _upload_to_s3(output_path, s3_key)
            log.info(f"[{task_label}] Upscaling complete (remote): {output_url}")
            return UpscaleResponse(output_url=output_url, success=True, **stats)
        except Exception as e:
            log.error(f"[{task_label}] S3 upload failed: {e}")
            return UpscaleResponse(success=False, error=f"S3 upload failed: {e}", **stats)
        finally:
            _cleanup(local_input, output_path)

    log.info(f"[{task_label}] Upscaling complete (local): {output_path}")
    return UpscaleResponse(output_path=output_path, success=True, **stats)


def _cleanup(*paths: str):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
