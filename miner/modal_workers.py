from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / "miner" / ".env", override=False)
except ImportError:
    pass

APP_NAME = os.getenv("MODAL_APP_NAME", "vidaio-miner-workers")
SECRET_NAME = os.getenv("MODAL_MINER_SECRET_NAME", "vidaio-miner-secrets")
GPU_TYPE = os.getenv("MODAL_GPU", "RTX-PRO-6000")
SHORT_COMPRESSION_GPU = os.getenv("MODAL_SHORT_COMPRESSION_GPU", "T4")
LONG_COMPRESSION_GPU = os.getenv("MODAL_LONG_COMPRESSION_GPU", GPU_TYPE)
CPU_CORES = float(os.getenv("MODAL_CPU_CORES", "16"))
MAX_CONTAINERS_UPSCALING = int(os.getenv("MODAL_UPSCALING_MAX_CONTAINERS", "5"))
MAX_CONTAINERS_COMPRESSION = int(os.getenv("MODAL_COMPRESSION_MAX_CONTAINERS", "5"))
SCALEDOWN_WINDOW_SECONDS = int(os.getenv("MODAL_SCALEDOWN_WINDOW_SECONDS", "300"))
BUFFER_CONTAINERS = int(os.getenv("MODAL_BUFFER_CONTAINERS", "0"))
UPSCALING_TIMEOUT_SECONDS = int(os.getenv("MODAL_UPSCALING_TIMEOUT_SECONDS", "1800"))
COMPRESSION_TIMEOUT_SECONDS = int(os.getenv("MODAL_COMPRESSION_TIMEOUT_SECONDS", "14400"))
COMPRESSION_PROBE_TIMEOUT_SECONDS = int(os.getenv("MODAL_COMPRESSION_PROBE_TIMEOUT_SECONDS", "60"))
LONG_COMPRESSION_THRESHOLD_SECONDS = int(os.getenv("MODAL_LONG_COMPRESSION_THRESHOLD_SECONDS", "1200"))
LONG_COMPRESSION_CHUNK_SECONDS = int(os.getenv("MODAL_LONG_COMPRESSION_CHUNK_SECONDS", "600"))
LONG_COMPRESSION_CHUNK_PARALLELISM = int(os.getenv("MODAL_LONG_COMPRESSION_CHUNK_PARALLELISM", "4"))
CLEANUP_TIMEOUT_SECONDS = int(os.getenv("MODAL_CLEANUP_TIMEOUT_SECONDS", "900"))
CLEANUP_PERIOD_HOURS = int(os.getenv("MODAL_CLEANUP_PERIOD_HOURS", "1"))
MODAL_SHARED_VOLUME_PATH = os.getenv("MODAL_SHARED_VOLUME_PATH", "/tmp/organic-proxy")

app = modal.App(APP_NAME)
miner_secret = modal.Secret.from_name(SECRET_NAME)

compression_volume = modal.Volume.from_name(
    os.getenv("MODAL_COMPRESSION_VOLUME_NAME", "vidaio-compression-tmp"),
    create_if_missing=True,
)
upscaling_video2x_volume = modal.Volume.from_name(
    os.getenv("MODAL_UPSCALING_VIDEO2X_VOLUME_NAME", "vidaio-upscaling-video2x-tmp"),
    create_if_missing=True,
)
upscaling_ffmpeg_volume = modal.Volume.from_name(
    os.getenv("MODAL_UPSCALING_FFMPEG_VOLUME_NAME", "vidaio-upscaling-ffmpeg-tmp"),
    create_if_missing=True,
)

modal_service_env = {
    "SHARED_VOLUME_PATH": MODAL_SHARED_VOLUME_PATH,
    "DISABLE_REMOTE_IO": "false",
    "MAX_CONCURRENT_UPSCALING": "1",
    "MAX_QUEUE_SIZE_UPSCALING": "0",
    "MAX_CONCURRENT_COMPRESSION": "1",
    "MAX_QUEUE_SIZE_COMPRESSION": "0",
    "MAX_QUEUE_SIZE": "0",
    "MINER_CLEANUP_ENABLED": "false",
    "MINER_STORAGE_CLEANUP_ENABLED": "false",
    "NVIDIA_VISIBLE_DEVICES": "all",
    "NVIDIA_DRIVER_CAPABILITIES": "graphics,video,compute,utility",
}

compression_base_image = modal.Image.from_dockerfile(
    str(REPO_ROOT / "miner/compression/Dockerfile"),
    context_dir=str(REPO_ROOT / "miner/compression"),
).env(modal_service_env)
compression_short_image = compression_base_image.env(
    {
        "COMPRESSION_CHUNKING_ENABLED": "false",
    }
)
compression_long_image = compression_base_image.env(
    {
        "COMPRESSION_CHUNKING_ENABLED": "true",
        "COMPRESSION_CHUNK_MIN_DURATION_SECONDS": "0",
        "COMPRESSION_CHUNK_TARGET_SECONDS": str(LONG_COMPRESSION_CHUNK_SECONDS),
        "COMPRESSION_CHUNK_PARALLELISM": str(LONG_COMPRESSION_CHUNK_PARALLELISM),
    }
)
upscaling_video2x_image = modal.Image.from_dockerfile(
    str(REPO_ROOT / "miner/upscaling/Dockerfile"),
    context_dir=str(REPO_ROOT / "miner/upscaling"),
).env(modal_service_env)
upscaling_ffmpeg_image = modal.Image.from_dockerfile(
    str(REPO_ROOT / "miner/upscaling/ffmpeg/Dockerfile"),
    context_dir=str(REPO_ROOT / "miner/upscaling/ffmpeg"),
).env(modal_service_env)
cleanup_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3")
)
compression_router_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("modal>=1.0.0")
)


def _gpu_worker_options(
    max_containers: int,
    timeout_seconds: int,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    return {
        "gpu": gpu_type or GPU_TYPE,
        "cpu": CPU_CORES,
        "min_containers": 0,
        "max_containers": max_containers,
        "buffer_containers": BUFFER_CONTAINERS,
        "scaledown_window": SCALEDOWN_WINDOW_SECONDS,
        "timeout": timeout_seconds,
        "secrets": [miner_secret],
    }


def _load_service_module():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    import importlib

    return importlib.import_module("app")


def _load_fastapi_app():
    return _load_service_module().app


def _task_id(payload: dict) -> str:
    task_id = str(payload.get("task_id") or "missing-task-id")
    return task_id[:120]


def _modal_input_id() -> str:
    try:
        return modal.current_input_id()
    except Exception:
        return "unavailable"


def _log_task_event(worker: str, payload: dict, event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "worker": worker,
                "task_id": _task_id(payload),
                "modal_input_id": _modal_input_id(),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_with_task_logs(worker: str, payload: dict, callback) -> dict:
    _log_task_event(worker, payload, "started")
    try:
        result = callback()
    except Exception as exc:
        _log_task_event(worker, payload, "exception", error=str(exc))
        raise

    if isinstance(result, dict):
        _log_task_event(
            worker,
            payload,
            "finished",
            success=result.get("success"),
            error=result.get("error"),
            output_url_configured=bool(result.get("output_url")),
        )
    else:
        _log_task_event(worker, payload, "finished", result_type=type(result).__name__)
    return result


def _run_service_request(request_model_name: str, handler_name: str, payload: dict) -> dict:
    import asyncio

    from fastapi import HTTPException

    service_module = _load_service_module()
    request_model = getattr(service_module, request_model_name)
    handler = getattr(service_module, handler_name)

    try:
        response = asyncio.run(handler(request_model(**payload)))
    except HTTPException as exc:
        return {"success": False, "error": str(exc.detail)}

    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return {"success": False, "error": f"Unexpected response type: {type(response).__name__}"}


def _probe_remote_duration_seconds(video_url: str) -> float | None:
    import subprocess

    if not video_url.startswith(("http://", "https://")):
        return None

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_url,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=COMPRESSION_PROBE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        print(f"Duration probe failed: {result.stderr[-1000:]}")
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


@app.function(
    image=compression_router_image,
    secrets=[miner_secret],
    min_containers=0,
    max_containers=MAX_CONTAINERS_COMPRESSION,
    buffer_containers=BUFFER_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=COMPRESSION_TIMEOUT_SECONDS + COMPRESSION_PROBE_TIMEOUT_SECONDS,
)
@modal.concurrent(max_inputs=16, target_inputs=8)
def compress(payload: dict) -> dict:
    def _route():
        duration_seconds = _probe_remote_duration_seconds(str(payload.get("video_path", "")))
        long_video = duration_seconds is None or duration_seconds >= LONG_COMPRESSION_THRESHOLD_SECONDS
        if long_video:
            routed_payload = {
                **payload,
                "chunked": True,
                "chunk_duration_seconds": LONG_COMPRESSION_CHUNK_SECONDS,
                "chunk_parallelism": LONG_COMPRESSION_CHUNK_PARALLELISM,
            }
            _log_task_event(
                "compress",
                payload,
                "routed",
                duration_seconds=duration_seconds,
                threshold_seconds=LONG_COMPRESSION_THRESHOLD_SECONDS,
                routed_worker="compress_rtx_pro_6000",
            )
            return compress_rtx_pro_6000.remote(routed_payload)

        _log_task_event(
            "compress",
            payload,
            "routed",
            duration_seconds=duration_seconds,
            threshold_seconds=LONG_COMPRESSION_THRESHOLD_SECONDS,
            routed_worker="compress_t4",
        )
        return compress_t4.remote({**payload, "chunked": False})

    return _run_with_task_logs("compress", payload, _route)


@app.function(
    image=compression_short_image,
    volumes={MODAL_SHARED_VOLUME_PATH: compression_volume},
    **_gpu_worker_options(MAX_CONTAINERS_COMPRESSION, COMPRESSION_TIMEOUT_SECONDS, SHORT_COMPRESSION_GPU),
)
@modal.concurrent(max_inputs=1)
def compress_t4(payload: dict) -> dict:
    return _run_with_task_logs(
        "compress_t4",
        payload,
        lambda: _run_service_request("CompressRequest", "compress", payload),
    )


@app.function(
    image=compression_long_image,
    volumes={MODAL_SHARED_VOLUME_PATH: compression_volume},
    **_gpu_worker_options(MAX_CONTAINERS_COMPRESSION, COMPRESSION_TIMEOUT_SECONDS, LONG_COMPRESSION_GPU),
)
@modal.concurrent(max_inputs=1)
def compress_rtx_pro_6000(payload: dict) -> dict:
    return _run_with_task_logs(
        "compress_rtx_pro_6000",
        payload,
        lambda: _run_service_request("CompressRequest", "compress", payload),
    )


@app.function(
    image=upscaling_video2x_image,
    volumes={MODAL_SHARED_VOLUME_PATH: upscaling_video2x_volume},
    **_gpu_worker_options(MAX_CONTAINERS_UPSCALING, UPSCALING_TIMEOUT_SECONDS),
)
@modal.concurrent(max_inputs=1)
def upscale_video2x(payload: dict) -> dict:
    return _run_with_task_logs(
        "upscale_video2x",
        payload,
        lambda: _run_service_request("UpscaleRequest", "upscale", payload),
    )


@app.function(
    image=upscaling_ffmpeg_image,
    volumes={MODAL_SHARED_VOLUME_PATH: upscaling_ffmpeg_volume},
    **_gpu_worker_options(MAX_CONTAINERS_UPSCALING, UPSCALING_TIMEOUT_SECONDS),
)
@modal.concurrent(max_inputs=1)
def upscale_ffmpeg(payload: dict) -> dict:
    return _run_with_task_logs(
        "upscale_ffmpeg",
        payload,
        lambda: _run_service_request("UpscaleRequest", "upscale", payload),
    )


@app.function(
    image=compression_long_image,
    volumes={MODAL_SHARED_VOLUME_PATH: compression_volume},
    **_gpu_worker_options(MAX_CONTAINERS_COMPRESSION, COMPRESSION_TIMEOUT_SECONDS, LONG_COMPRESSION_GPU),
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def compression_api():
    return _load_fastapi_app()


@app.function(
    image=compression_short_image,
    volumes={MODAL_SHARED_VOLUME_PATH: compression_volume},
    **_gpu_worker_options(MAX_CONTAINERS_COMPRESSION, COMPRESSION_TIMEOUT_SECONDS, SHORT_COMPRESSION_GPU),
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def compression_t4_api():
    return _load_fastapi_app()


@app.function(
    image=upscaling_video2x_image,
    volumes={MODAL_SHARED_VOLUME_PATH: upscaling_video2x_volume},
    **_gpu_worker_options(MAX_CONTAINERS_UPSCALING, UPSCALING_TIMEOUT_SECONDS),
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def upscaling_video2x_api():
    return _load_fastapi_app()


@app.function(
    image=upscaling_ffmpeg_image,
    volumes={MODAL_SHARED_VOLUME_PATH: upscaling_ffmpeg_volume},
    **_gpu_worker_options(MAX_CONTAINERS_UPSCALING, UPSCALING_TIMEOUT_SECONDS),
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def upscaling_ffmpeg_api():
    return _load_fastapi_app()


def _storage_cleanup_ready() -> bool:
    return all(
        os.getenv(name)
        for name in (
            "MINER_STORAGE_S3_BUCKET_NAME",
            "MINER_STORAGE_S3_ACCESS_KEY_ID",
            "MINER_STORAGE_S3_SECRET_ACCESS_KEY",
        )
    )


def _presigned_expiry_seconds() -> int:
    return int(
        os.getenv("MINER_STORAGE_S3_PRESIGNED_EXPIRY")
        or os.getenv("S3_PRESIGNED_EXPIRY")
        or "3600"
    )


def _cleanup_grace_seconds() -> int:
    return int(os.getenv("PRESIGNED_URL_CLEANUP_GRACE_SECONDS", "600"))


def _storage_object_ttl_seconds() -> int:
    return int(
        os.getenv("MINER_STORAGE_OBJECT_TTL_SECONDS")
        or str(min(_presigned_expiry_seconds(), 604800) + _cleanup_grace_seconds())
    )


def _volume_file_ttl_seconds() -> int:
    return int(
        os.getenv("MODAL_VOLUME_FILE_TTL_SECONDS")
        or os.getenv("MINER_TEMP_FILE_TTL_SECONDS")
        or str(min(_presigned_expiry_seconds(), 604800) + _cleanup_grace_seconds())
    )


def _get_s3_client():
    import boto3
    from botocore.config import Config

    endpoint_url = (
        os.getenv("MINER_STORAGE_S3_ENDPOINT_URL")
        or os.getenv("MINER_STORAGE_S3_ENDPOINT")
        or ""
    ).strip()
    kwargs = {
        "region_name": os.getenv("MINER_STORAGE_S3_REGION", "us-east-1").strip() or "us-east-1",
        "aws_access_key_id": os.getenv("MINER_STORAGE_S3_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("MINER_STORAGE_S3_SECRET_ACCESS_KEY"),
        "config": Config(signature_version="s3v4"),
    }
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)


def _cleanup_expired_storage_objects() -> int:
    ttl_seconds = _storage_object_ttl_seconds()
    if not _storage_cleanup_ready() or ttl_seconds <= 0:
        return 0

    bucket = os.environ["MINER_STORAGE_S3_BUCKET_NAME"]
    prefixes = [
        prefix.strip()
        for prefix in os.getenv("MINER_STORAGE_CLEANUP_PREFIXES", "processing/,upscaling/").split(",")
        if prefix.strip()
    ]
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
    client = _get_s3_client()
    deleted = 0

    for prefix in prefixes:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
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
                    Bucket=bucket,
                    Delete={"Objects": batch, "Quiet": True},
                )
                deleted += len(batch) - len(response.get("Errors", []))

    return deleted


def _cleanup_expired_volume_files(root: Path) -> int:
    ttl_seconds = _volume_file_ttl_seconds()
    if ttl_seconds <= 0 or not root.exists():
        return 0

    cutoff = time.time() - ttl_seconds
    deleted = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"Failed to remove stale Modal volume file {path}: {exc}")
    return deleted


@app.function(
    image=cleanup_image,
    secrets=[miner_secret],
    schedule=modal.Period(hours=CLEANUP_PERIOD_HOURS),
    cpu=1.0,
    min_containers=0,
    max_containers=1,
    timeout=CLEANUP_TIMEOUT_SECONDS,
    volumes={
        "/vol/compression": compression_volume,
        "/vol/upscaling-video2x": upscaling_video2x_volume,
        "/vol/upscaling-ffmpeg": upscaling_ffmpeg_volume,
    },
)
def cleanup_expired_artifacts() -> dict:
    storage_deleted = _cleanup_expired_storage_objects()
    volume_deleted = {
        "compression": _cleanup_expired_volume_files(Path("/vol/compression")),
        "upscaling_video2x": _cleanup_expired_volume_files(Path("/vol/upscaling-video2x")),
        "upscaling_ffmpeg": _cleanup_expired_volume_files(Path("/vol/upscaling-ffmpeg")),
    }

    compression_volume.commit()
    upscaling_video2x_volume.commit()
    upscaling_ffmpeg_volume.commit()

    result = {
        "storage_deleted": storage_deleted,
        "volume_deleted": volume_deleted,
        "storage_object_ttl_seconds": _storage_object_ttl_seconds(),
        "volume_file_ttl_seconds": _volume_file_ttl_seconds(),
    }
    print(json.dumps(result, sort_keys=True))
    return result


@app.local_entrypoint()
def test_worker(
    worker: str,
    video_url: str,
    task_id: str = "manual-test",
    scale: int = 2,
    codec: str = "AV1",
    codec_mode: str = "CRF",
    cq: int = 35,
    target_bitrate: int = 0,
):
    worker_key = worker.strip().lower()
    if worker_key in ("compression", "compress"):
        payload: dict[str, Any] = {
            "video_path": video_url,
            "task_id": task_id,
            "codec": codec,
            "codec_mode": codec_mode.upper(),
            "cq": cq,
        }
        if target_bitrate > 0:
            payload["target_bitrate"] = target_bitrate
        result = compress.remote(payload)
    elif worker_key in ("compression-t4", "compress_t4", "t4"):
        payload = {
            "video_path": video_url,
            "task_id": task_id,
            "codec": codec,
            "codec_mode": codec_mode.upper(),
            "cq": cq,
            "chunked": False,
        }
        if target_bitrate > 0:
            payload["target_bitrate"] = target_bitrate
        result = compress_t4.remote(payload)
    elif worker_key in ("compression-rtx", "compress_rtx", "compress_rtx_pro_6000", "rtx"):
        payload = {
            "video_path": video_url,
            "task_id": task_id,
            "codec": codec,
            "codec_mode": codec_mode.upper(),
            "cq": cq,
            "chunked": True,
            "chunk_duration_seconds": LONG_COMPRESSION_CHUNK_SECONDS,
            "chunk_parallelism": LONG_COMPRESSION_CHUNK_PARALLELISM,
        }
        if target_bitrate > 0:
            payload["target_bitrate"] = target_bitrate
        result = compress_rtx_pro_6000.remote(payload)
    elif worker_key in ("video2x", "upscaling-video2x", "upscale_video2x"):
        result = upscale_video2x.remote(
            {"video_path": video_url, "task_id": task_id, "scale": scale}
        )
    elif worker_key in ("ffmpeg", "upscaling-ffmpeg", "upscale_ffmpeg"):
        result = upscale_ffmpeg.remote(
            {"video_path": video_url, "task_id": task_id, "scale": scale}
        )
    else:
        raise ValueError("worker must be one of: compression, compression-t4, compression-rtx, video2x, ffmpeg")

    print(json.dumps(result, indent=2, sort_keys=True))
