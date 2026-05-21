from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import uuid
from typing import Optional

import boto3
import httpx
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("upscaling-ffmpeg")

app = FastAPI(title="FFmpeg Upscaling Service")

SHARED_VOLUME_PATH = os.getenv("SHARED_VOLUME_PATH", "/tmp/organic-proxy")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_UPSCALING", "2"))
FFMPEG_CODEC = os.getenv("FFMPEG_CODEC", "av1_nvenc")
FFMPEG_PRESET = os.getenv("FFMPEG_PRESET", "p4")
FFMPEG_CQ = int(os.getenv("FFMPEG_CQ", "25"))
DISABLE_REMOTE_IO = os.getenv("DISABLE_REMOTE_IO", "false").lower() in ("1", "true", "yes")

STORAGE_PROVIDER = os.getenv("ORGANIC_PROXY_STORAGE_PROVIDER", "s3").lower()
S3_REGION = os.getenv("ORGANIC_PROXY_STORAGE_S3_REGION", "us-east-1").strip() or "us-east-1"
S3_BUCKET = os.getenv("ORGANIC_PROXY_STORAGE_S3_BUCKET_NAME", "").strip()
S3_ACCESS_KEY_ID = os.getenv("ORGANIC_PROXY_STORAGE_S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("ORGANIC_PROXY_STORAGE_S3_SECRET_ACCESS_KEY", "").strip()
S3_ENDPOINT_URL = (os.getenv("ORGANIC_PROXY_STORAGE_S3_ENDPOINT_URL") or os.getenv("ORGANIC_PROXY_STORAGE_S3_ENDPOINT") or "").strip()
S3_PRESIGNED_EXPIRY = int(os.getenv("ORGANIC_PROXY_STORAGE_S3_PRESIGNED_EXPIRY") or os.getenv("S3_PRESIGNED_EXPIRY") or "3600")

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_size = 0
_active_count = 0
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
        "ORGANIC_PROXY_STORAGE_S3_BUCKET_NAME": S3_BUCKET,
        "ORGANIC_PROXY_STORAGE_S3_ACCESS_KEY_ID": S3_ACCESS_KEY_ID,
        "ORGANIC_PROXY_STORAGE_S3_SECRET_ACCESS_KEY": S3_SECRET_ACCESS_KEY,
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


class UpscaleRequest(BaseModel):
    video_path: str
    scale: int = Field(..., description="Upscaling factor: 2 or 4")
    task_id: str = ""


class UpscaleResponse(BaseModel):
    output_path: str = ""
    output_url: str = ""
    success: bool
    error: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "max_concurrent": MAX_CONCURRENT, "active_tasks": _active_count, "queued_tasks": max(0, _queue_size - _active_count)}


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale(req: UpscaleRequest):
    global _queue_size, _active_count

    if req.scale not in (2, 4):
        raise HTTPException(status_code=400, detail="scale must be 2 or 4")

    task = req.task_id or uuid.uuid4().hex[:8]
    remote = _is_url(req.video_path)
    if remote and DISABLE_REMOTE_IO:
        return UpscaleResponse(success=False, error="Remote URL input is disabled for this local-only service")

    local_input = os.path.join(SHARED_VOLUME_PATH, f"{task}_input.mp4") if remote else req.video_path
    if remote:
        await _download_url(req.video_path, local_input)
    elif not os.path.exists(local_input):
        raise HTTPException(status_code=400, detail=f"Input file not found: {local_input}")

    base = os.path.splitext(os.path.basename(local_input))[0]
    output_path = os.path.join(SHARED_VOLUME_PATH, f"{base}_upscaled_{req.scale}x.mp4")

    cmd = [
        "ffmpeg", "-y", "-hwaccel", "cuda", "-i", local_input,
        "-vf", f"scale=iw*{req.scale}:ih*{req.scale}:flags=lanczos",
        "-c:v", FFMPEG_CODEC, "-preset", FFMPEG_PRESET, "-cq", str(FFMPEG_CQ),
        "-c:a", "copy", output_path,
    ]

    async with _lock:
        _queue_size += 1
    async with _semaphore:
        async with _lock:
            _active_count += 1
        try:
            result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return UpscaleResponse(success=False, error=result.stderr[-1000:])
            if remote:
                key = f"upscaling/{task}/{os.path.basename(output_path)}"
                url = _upload_to_s3(output_path, key)
                return UpscaleResponse(success=True, output_url=url)
            return UpscaleResponse(success=True, output_path=output_path)
        finally:
            async with _lock:
                _active_count = max(0, _active_count - 1)
                _queue_size = max(0, _queue_size - 1)
