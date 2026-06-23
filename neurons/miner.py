import asyncio
import os
import posixpath
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import boto3
import httpx
import bittensor as bt
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv
from loguru import logger
from vidaio_subnet_core.base.miner import BaseMiner
from vidaio_subnet_core.protocol import (
    VideoUpscalingProtocol,
    LengthCheckProtocol,
    ContentLength,
    VideoCompressionProtocol,
    TaskWarrantProtocol,
    TaskType,
    VideoCompressionJobProtocol,
    VideoCompressionPollProtocol,
    VideoUpscalingJobProtocol,
    VideoUpscalingPollProtocol,
    JobKickoffResponse,
    PollResponse,
)

from vidaio_subnet_core.utilities.version import check_version

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / "miner" / ".env", override=False)

MAX_CONTENT_LEN = ContentLength.FIVE
warrant_task = TaskType.UPSCALING
DEV_MODE = os.getenv("DEV_MODE", "False").lower() == "true"

UPSCALING_SERVICE_URL = os.getenv("MINER_UPSCALING_SERVICE_URL", "http://localhost:8003").rstrip("/")
COMPRESSION_SERVICE_URL = os.getenv("MINER_COMPRESSION_SERVICE_URL", "http://localhost:8004").rstrip("/")
PROCESSING_BACKEND = os.getenv("MINER_PROCESSING_BACKEND", "http").strip().lower()
UPSCALING_SERVICE_TIMEOUT_SECONDS = float(os.getenv("MINER_UPSCALING_SERVICE_TIMEOUT_SECONDS", "1200"))
DEFAULT_COMPRESSION_TIMEOUT_SECONDS = os.getenv(
    "MODAL_COMPRESSION_TIMEOUT_SECONDS" if PROCESSING_BACKEND == "modal" else "MINER_DEFAULT_COMPRESSION_TIMEOUT_SECONDS",
    "14400" if PROCESSING_BACKEND == "modal" else "600",
)
COMPRESSION_SERVICE_TIMEOUT_SECONDS = float(
    os.getenv("MINER_COMPRESSION_SERVICE_TIMEOUT_SECONDS", DEFAULT_COMPRESSION_TIMEOUT_SECONDS)
)
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "vidaio-miner-workers").strip()
MODAL_UPSCALING_FUNCTION = os.getenv("MINER_MODAL_UPSCALING_FUNCTION", "upscale_video2x").strip()
MODAL_COMPRESSION_FUNCTION = os.getenv("MINER_MODAL_COMPRESSION_FUNCTION", "compress").strip()
MINER_DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("MINER_DOWNLOAD_TIMEOUT_SECONDS", "600"))
MINER_UPLOAD_URL_EXPIRY_SECONDS = int(os.getenv("MINER_STORAGE_S3_PRESIGNED_EXPIRY", os.getenv("S3_PRESIGNED_EXPIRY", "3600")))

HOST_SHARED_VOLUME_PATH = Path(
    os.getenv("MINER_SHARED_VOLUME_PATH")
    or os.getenv("MINER_SHARED_DIR", "/tmp/vidaio-miner-video-tmp")
).expanduser()
if not HOST_SHARED_VOLUME_PATH.is_absolute():
    HOST_SHARED_VOLUME_PATH = REPO_ROOT / HOST_SHARED_VOLUME_PATH
CONTAINER_SHARED_VOLUME_PATH = os.getenv("MINER_CONTAINER_SHARED_VOLUME_PATH", "/tmp/organic-proxy").rstrip("/")

S3_REGION = os.getenv("MINER_STORAGE_S3_REGION", "us-east-1").strip() or "us-east-1"
S3_BUCKET = os.getenv("MINER_STORAGE_S3_BUCKET_NAME", "").strip()
S3_ACCESS_KEY_ID = os.getenv("MINER_STORAGE_S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("MINER_STORAGE_S3_SECRET_ACCESS_KEY", "").strip()
S3_ENDPOINT_URL = (
    os.getenv("MINER_STORAGE_S3_ENDPOINT_URL")
    or os.getenv("MINER_STORAGE_S3_ENDPOINT")
    or ""
).strip()
PRESIGNED_URL_CLEANUP_GRACE_SECONDS = int(os.getenv("PRESIGNED_URL_CLEANUP_GRACE_SECONDS", "600"))
SHARED_VOLUME_FILE_TTL_SECONDS = int(
    os.getenv("MINER_TEMP_FILE_TTL_SECONDS")
    or os.getenv("TEMP_FILE_TTL_SECONDS")
    or str(min(MINER_UPLOAD_URL_EXPIRY_SECONDS, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)
SHARED_VOLUME_CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("MINER_CLEANUP_INTERVAL_SECONDS") or os.getenv("CLEANUP_INTERVAL_SECONDS") or "300"
)
SHARED_VOLUME_CLEANUP_MAX_BYTES = int(
    os.getenv("MINER_CLEANUP_MAX_VOLUME_BYTES") or os.getenv("CLEANUP_MAX_VOLUME_BYTES") or "9000000000"
)
SHARED_VOLUME_CLEANUP_MIN_FILE_AGE_SECONDS = int(
    os.getenv("MINER_CLEANUP_MIN_FILE_AGE_SECONDS") or os.getenv("CLEANUP_MIN_FILE_AGE_SECONDS") or "60"
)
SHARED_VOLUME_CLEANUP_ENABLED = os.getenv(
    "MINER_CLEANUP_ENABLED", os.getenv("CLEANUP_ENABLED", "true")
).lower() in ("1", "true", "yes")
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
    or str(min(MINER_UPLOAD_URL_EXPIRY_SECONDS, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)

TASK_TYPE_TO_SCALE = {
    "SD24K": 4,
    "HD24K": 2,
    "SD2HD": 2,
    "4K28K": 2,
}

COMPRESSION_CQ_BY_TYPE = {
    "Low": 40,
    "Medium": 35,
    "High": 30,
}


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


class Miner(BaseMiner):
    def __init__(self, config: dict | None = None) -> None:
        """
        Initializes the Miner instance.
        """
        super().__init__()
        # Shared in-memory job store for polling-based organic requests.
        # Structure: { job_id: {"task": asyncio.Task, "result": str | None, "error": str | None} }
        self._jobs: dict[str, dict] = {}
        self._active_shared_files: set[str] = set()
        self._shared_files_lock = threading.Lock()
        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread: threading.Thread | None = None
        self._start_shared_volume_cleanup_worker()

    def stop_run_thread(self):
        super().stop_run_thread()
        self._stop_shared_volume_cleanup_worker()

    def _normalize_shared_path(self, path: Path) -> str:
        return str(path.resolve())

    def _is_path_in_shared_volume(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(HOST_SHARED_VOLUME_PATH.resolve())
            return True
        except ValueError:
            return False

    def _track_shared_file(self, path: Path | None):
        if path is None or not self._is_path_in_shared_volume(path):
            return
        with self._shared_files_lock:
            self._active_shared_files.add(self._normalize_shared_path(path))

    def _untrack_shared_file(self, path: Path | None):
        if path is None:
            return
        with self._shared_files_lock:
            self._active_shared_files.discard(self._normalize_shared_path(path))

    def _active_shared_files_snapshot(self) -> set[str]:
        with self._shared_files_lock:
            return set(self._active_shared_files)

    def _should_blacklist_non_validator(self, uid: int) -> bool:
        return not DEV_MODE and not self.metagraph.validator_permit[uid]

    def _remove_stale_shared_file(self, path: Path, reason: str) -> int:
        try:
            size = path.stat().st_size
            path.unlink()
            logger.info(f"Removed {reason} shared temp file: {path} ({size} bytes)")
            return size
        except FileNotFoundError:
            return 0
        except OSError as e:
            logger.warning(f"Failed to remove shared temp file {path}: {e}")
            return 0

    def _cleanup_shared_volume_once(self):
        if not SHARED_VOLUME_CLEANUP_ENABLED or not HOST_SHARED_VOLUME_PATH.exists():
            return

        now = time.time()
        protected_paths = self._active_shared_files_snapshot()
        total_bytes = 0
        candidates: list[tuple[float, Path, int]] = []

        for path in HOST_SHARED_VOLUME_PATH.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue

            total_bytes += stat.st_size
            normalized_path = self._normalize_shared_path(path)
            if normalized_path in protected_paths:
                continue

            candidates.append((stat.st_mtime, path, stat.st_size))
            if now - stat.st_mtime >= SHARED_VOLUME_FILE_TTL_SECONDS:
                total_bytes -= self._remove_stale_shared_file(path, "expired")

        if SHARED_VOLUME_CLEANUP_MAX_BYTES <= 0 or total_bytes <= SHARED_VOLUME_CLEANUP_MAX_BYTES:
            return

        for _, path, size in sorted(candidates):
            if total_bytes <= SHARED_VOLUME_CLEANUP_MAX_BYTES:
                break
            normalized_path = self._normalize_shared_path(path)
            if normalized_path in protected_paths or not path.exists():
                continue
            try:
                if now - path.stat().st_mtime < SHARED_VOLUME_CLEANUP_MIN_FILE_AGE_SECONDS:
                    continue
            except FileNotFoundError:
                continue
            total_bytes -= self._remove_stale_shared_file(path, "over-quota")

    def _storage_cleanup_ready(self) -> bool:
        return (
            STORAGE_CLEANUP_ENABLED
            and STORAGE_OBJECT_TTL_SECONDS > 0
            and bool(STORAGE_CLEANUP_PREFIXES)
            and bool(S3_BUCKET)
            and bool(S3_ACCESS_KEY_ID)
            and bool(S3_SECRET_ACCESS_KEY)
        )

    def _cleanup_expired_storage_objects_once(self) -> int:
        if not self._storage_cleanup_ready():
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=STORAGE_OBJECT_TTL_SECONDS)
        client = self._get_s3_client()
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
            logger.info(f"Deleted {deleted} expired storage object(s)")
        return deleted

    def _shared_volume_cleanup_loop(self):
        logger.info(
            "Starting miner cleanup worker "
            f"(path={HOST_SHARED_VOLUME_PATH}, ttl={SHARED_VOLUME_FILE_TTL_SECONDS}s, "
            f"interval={SHARED_VOLUME_CLEANUP_INTERVAL_SECONDS}s, "
            f"max_bytes={SHARED_VOLUME_CLEANUP_MAX_BYTES}, "
            f"storage_ttl={STORAGE_OBJECT_TTL_SECONDS}s)"
        )
        while not self._cleanup_stop_event.is_set():
            try:
                if SHARED_VOLUME_CLEANUP_ENABLED:
                    self._cleanup_shared_volume_once()
                if self._storage_cleanup_ready():
                    self._cleanup_expired_storage_objects_once()
            except Exception as e:
                logger.warning(f"Miner cleanup pass failed: {e}")
            self._cleanup_stop_event.wait(SHARED_VOLUME_CLEANUP_INTERVAL_SECONDS)

    def _start_shared_volume_cleanup_worker(self):
        if not SHARED_VOLUME_CLEANUP_ENABLED and not self._storage_cleanup_ready():
            logger.info("Miner cleanup worker disabled")
            return
        self._cleanup_thread = threading.Thread(
            target=self._shared_volume_cleanup_loop,
            name="miner-shared-volume-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _stop_shared_volume_cleanup_worker(self):
        self._cleanup_stop_event.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=5)

    def _container_shared_path(self, host_path: Path) -> str:
        return f"{CONTAINER_SHARED_VOLUME_PATH}/{host_path.name}"

    def _host_shared_path(self, container_path: str) -> Path:
        shared_mount = posixpath.normpath(CONTAINER_SHARED_VOLUME_PATH)
        normalized_path = posixpath.normpath(container_path)

        if normalized_path == shared_mount or not normalized_path.startswith(f"{shared_mount}/"):
            raise ValueError(f"Service returned path outside shared volume: {container_path}")

        relative_path = posixpath.relpath(normalized_path, shared_mount)
        host_path = (HOST_SHARED_VOLUME_PATH / relative_path).resolve()
        shared_root = HOST_SHARED_VOLUME_PATH.resolve()

        try:
            host_path.relative_to(shared_root)
        except ValueError:
            raise ValueError(f"Mapped host path escapes shared volume: {host_path}") from None

        return host_path

    async def _download_to_shared_volume(self, video_url: str, task_id: str) -> tuple[Path, str]:
        HOST_SHARED_VOLUME_PATH.mkdir(parents=True, exist_ok=True)
        host_path = HOST_SHARED_VOLUME_PATH / f"{task_id}_input.mp4"
        self._track_shared_file(host_path)

        logger.info(f"Downloading validator payload to shared volume: {host_path}")
        timeout = httpx.Timeout(MINER_DOWNLOAD_TIMEOUT_SECONDS)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                async with client.stream("GET", video_url) as response:
                    response.raise_for_status()
                    with open(host_path, "wb") as file:
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                            file.write(chunk)
        except Exception:
            self._cleanup_shared_files(host_path)
            self._untrack_shared_file(host_path)
            raise

        return host_path, self._container_shared_path(host_path)

    def _get_s3_client(self):
        client_kwargs = {
            "region_name": S3_REGION,
            "aws_access_key_id": S3_ACCESS_KEY_ID or None,
            "aws_secret_access_key": S3_SECRET_ACCESS_KEY or None,
            "config": BotoConfig(signature_version="s3v4"),
        }
        if S3_ENDPOINT_URL:
            client_kwargs["endpoint_url"] = S3_ENDPOINT_URL
        return boto3.client("s3", **client_kwargs)

    async def _upload_processed_video(self, host_path: Path, service_name: str, task_id: str) -> str | None:
        missing = []
        if not S3_BUCKET:
            missing.append("MINER_STORAGE_S3_BUCKET_NAME")
        if not S3_ACCESS_KEY_ID:
            missing.append("MINER_STORAGE_S3_ACCESS_KEY_ID")
        if not S3_SECRET_ACCESS_KEY:
            missing.append("MINER_STORAGE_S3_SECRET_ACCESS_KEY")

        if missing:
            logger.error(f"Missing storage configuration: {', '.join(missing)}")
            return None

        if not host_path.exists():
            logger.error(f"Processed file not found on shared volume: {host_path}")
            return None

        object_key = f"processing/{service_name}/{task_id}/{host_path.name}"

        def _upload_and_sign() -> str:
            client = self._get_s3_client()
            client.upload_file(str(host_path), S3_BUCKET, object_key)
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": object_key},
                ExpiresIn=min(MINER_UPLOAD_URL_EXPIRY_SECONDS, 604800),
            )

        logger.info(f"Uploading processed {service_name} output: {object_key}")
        return await asyncio.to_thread(_upload_and_sign)

    def _cleanup_shared_files(self, *paths: Path | None):
        for path in paths:
            self._untrack_shared_file(path)
            if path and path.exists():
                try:
                    path.unlink()
                except OSError as e:
                    logger.warning(f"Failed to remove shared file {path}: {e}")

    def _upscaling_service_payload(self, video_path: str, task_type: str, task_id: str) -> dict:
        scale = TASK_TYPE_TO_SCALE.get(task_type, 2)
        if task_type not in TASK_TYPE_TO_SCALE:
            logger.warning(f"Unknown upscaling task_type={task_type}, defaulting to 2x scale")

        return {
            "video_paths": [video_path],
            "scale": scale,
            "task_id": task_id,
        }

    def _compression_type_from_payload(self, payload) -> str:
        task_data = payload.model_dump() if hasattr(payload, "model_dump") else {}
        compression_type = task_data.get("compression_type") or getattr(payload, "compression_type", None)
        if compression_type in COMPRESSION_CQ_BY_TYPE:
            return compression_type

        vmaf_threshold = float(payload.vmaf_threshold)
        if vmaf_threshold >= 93:
            return "High"
        if vmaf_threshold >= 89:
            return "Medium"
        return "Low"

    def _compression_service_payload(self, payload, video_path: str, task_id: str) -> dict:
        compression_type = self._compression_type_from_payload(payload)
        compression_config = {
            "video_paths": [video_path],
            "task_id": task_id,
            "codec": payload.target_codec,
            "codec_mode": payload.codec_mode.upper(),
            "target_bitrate": int(float(payload.target_bitrate) * 1_000_000),
        }

        if compression_type == "Low":
            compression_config["cq"] = 40
        elif compression_type == "Medium":
            compression_config["cq"] = 35
        elif compression_type == "High":
            compression_config["cq"] = 30

        return compression_config

    def _payload_debug_dump(self, payload) -> dict:
        if hasattr(payload, "model_dump"):
            try:
                return payload.model_dump()
            except Exception:
                pass
        return {
            "reference_video_urls": getattr(payload, "reference_video_urls", None),
            "reference_video_url": getattr(payload, "reference_video_url", None),
            "task_types": getattr(payload, "task_types", None),
            "task_type": getattr(payload, "task_type", None),
        }

    def _payload_reference_video_urls(self, payload) -> list[str]:
        payload_data = self._payload_debug_dump(payload)
        raw_urls = getattr(payload, "reference_video_urls", None) or payload_data.get("reference_video_urls") or []
        if isinstance(raw_urls, str):
            raw_urls = [raw_urls]

        legacy_url = getattr(payload, "reference_video_url", None) or payload_data.get("reference_video_url")
        urls = [str(url).strip() for url in raw_urls if str(url).strip()]
        if not urls and legacy_url and str(legacy_url).strip():
            urls = [str(legacy_url).strip()]

        return urls

    def _payload_task_types(self, payload, query_count: int) -> list[str]:
        if query_count <= 0:
            return []

        payload_data = self._payload_debug_dump(payload)
        raw_task_types = getattr(payload, "task_types", None) or payload_data.get("task_types") or []
        if isinstance(raw_task_types, str):
            raw_task_types = [raw_task_types]

        legacy_task_type = getattr(payload, "task_type", None) or payload_data.get("task_type") or "HD24K"
        if not raw_task_types:
            raw_task_types = [legacy_task_type] * query_count

        task_types = [str(task_type).strip() for task_type in raw_task_types if str(task_type).strip()]
        if not task_types:
            task_types = ["HD24K"] * query_count
        if len(task_types) < query_count:
            task_types.extend([task_types[-1]] * (query_count - len(task_types)))
        return task_types[:query_count]

    async def _post_processing_service(
        self,
        service_name: str,
        service_url: str,
        endpoint: str,
        payload: dict,
        timeout_seconds: float,
    ) -> str | None:
        url = f"{service_url}{endpoint}"
        timeout = httpx.Timeout(timeout_seconds)

        logger.info(f"Forwarding {service_name} request to {url}")
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)

        if response.status_code != 200:
            logger.error(f"{service_name} service error {response.status_code}: {response.text[:1000]}")
            return None

        result = response.json()
        if not result.get("success", False):
            logger.error(f"{service_name} service failed: {result.get('errors')}")
            return None

        output_urls = result.get("output_urls") or []
        output_paths = result.get("output_paths") or []
        processed_video_url = (output_urls[0] if output_urls else None) or (output_paths[0] if output_paths else None)
        if not processed_video_url:
            logger.error(f"{service_name} service returned no output URL/path")
            return None

        logger.info(f"Received {service_name} result from container service")
        return processed_video_url

    async def _call_modal_processing_function(
        self,
        service_name: str,
        function_name: str,
        payload: dict,
        timeout_seconds: float,
    ) -> str | None:
        if not MODAL_APP_NAME or not function_name:
            logger.error(f"Modal {service_name} configuration is missing app or function name")
            return None

        def _remote_call() -> dict:
            try:
                import modal
            except ImportError as exc:
                raise RuntimeError(
                    "The modal package is required when MINER_PROCESSING_BACKEND=modal"
                ) from exc

            function = modal.Function.from_name(MODAL_APP_NAME, function_name)
            return function.remote(payload)

        logger.info(f"Forwarding {service_name} request to Modal: {MODAL_APP_NAME}.{function_name}")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_remote_call),
                timeout=timeout_seconds,
            )
        except Exception as e:
            logger.error(f"Modal {service_name} request failed: {e}")
            return None

        if not isinstance(result, dict):
            logger.error(f"Modal {service_name} returned unexpected result type: {type(result).__name__}")
            return None
        if not result.get("success", False):
            logger.error(f"Modal {service_name} failed: {result.get('errors')}")
            return None

        output_urls = result.get("output_urls") or []
        processed_video_url = output_urls[0] if output_urls else None
        if not processed_video_url:
            logger.error(f"Modal {service_name} returned no output_urls")
            return None
        if not _is_url(processed_video_url):
            logger.error(f"Modal {service_name} returned non-URL output: {processed_video_url}")
            return None

        logger.info(f"Received {service_name} result from Modal")
        return processed_video_url

    async def _forward_upscaling_to_modal(self, payload_url: str, task_type: str, task_id: str) -> str | None:
        return await self._call_modal_processing_function(
            "upscaling",
            MODAL_UPSCALING_FUNCTION,
            self._upscaling_service_payload(payload_url, task_type, task_id),
            UPSCALING_SERVICE_TIMEOUT_SECONDS,
        )

    async def _forward_compression_to_modal(self, payload, task_id: str) -> str | None:
        payload_urls = self._payload_reference_video_urls(payload)
        if not payload_urls:
            logger.error(f"Compression payload missing reference video URLs: {self._payload_debug_dump(payload)}")
            return None

        return await self._call_modal_processing_function(
            "compression",
            MODAL_COMPRESSION_FUNCTION,
            self._compression_service_payload(payload, payload_urls[0], task_id),
            COMPRESSION_SERVICE_TIMEOUT_SECONDS,
        )

    async def _forward_upscaling_payload_to_service(self, payload) -> list[str]:
        urls = self._payload_reference_video_urls(payload)
        if not urls:
            logger.error(f"Upscaling payload missing reference video URLs: {self._payload_debug_dump(payload)}")
            return []

        task_types = self._payload_task_types(payload, len(urls))

        async def _process_one(index: int, payload_url: str, task_type: str) -> str:
            try:
                processed_url = await self._forward_upscaling_to_service(payload_url, task_type)
                return processed_url or ""
            except Exception as e:
                logger.error(f"Failed to process upscaling payload item {index}: {e}")
                return ""

        logger.info(f"Forwarding {len(urls)} upscaling payload item(s) concurrently")
        return await asyncio.gather(
            *[
                _process_one(index, payload_url, task_type)
                for index, (payload_url, task_type) in enumerate(zip(urls, task_types))
            ]
        )

    async def _forward_compression_url_to_service(self, payload, payload_url: str) -> str | None:
        task_id = uuid.uuid4().hex[:12]
        if PROCESSING_BACKEND == "modal":
            return await self._call_modal_processing_function(
                "compression",
                MODAL_COMPRESSION_FUNCTION,
                self._compression_service_payload(payload, payload_url, task_id),
                COMPRESSION_SERVICE_TIMEOUT_SECONDS,
            )

        input_host_path = None
        output_host_path = None
        try:
            input_host_path, input_container_path = await self._download_to_shared_volume(
                payload_url, task_id
            )
            processed_ref = await self._post_processing_service(
                "compression",
                COMPRESSION_SERVICE_URL,
                "/compress",
                self._compression_service_payload(payload, input_container_path, task_id),
                COMPRESSION_SERVICE_TIMEOUT_SECONDS,
            )
            if processed_ref is None:
                return None
            if _is_url(processed_ref):
                return processed_ref

            output_host_path = self._host_shared_path(processed_ref)
            self._track_shared_file(output_host_path)
            return await self._upload_processed_video(output_host_path, "compression", task_id)
        finally:
            self._cleanup_shared_files(input_host_path, output_host_path)

    async def _forward_compression_payload_to_service(self, payload) -> list[str]:
        urls = self._payload_reference_video_urls(payload)
        if not urls:
            logger.error(f"Compression payload missing reference video URLs: {self._payload_debug_dump(payload)}")
            return []

        async def _process_one(index: int, payload_url: str) -> str:
            try:
                item_payload = self._compression_payload_for_index(payload, index)
                processed_url = await self._forward_compression_url_to_service(item_payload, payload_url)
                return processed_url or ""
            except Exception as e:
                logger.error(f"Failed to process compression payload item {index}: {e}")
                return ""

        logger.info(f"Forwarding {len(urls)} compression payload item(s) concurrently")
        return await asyncio.gather(
            *[
                _process_one(index, payload_url)
                for index, payload_url in enumerate(urls)
            ]
        )

    def _compression_payload_for_index(self, payload, index: int):
        updates = {}
        per_item_fields = {
            "vmaf_thresholds": "vmaf_threshold",
            "target_codecs": "target_codec",
            "codec_modes": "codec_mode",
            "target_bitrates": "target_bitrate",
        }

        for list_field, scalar_field in per_item_fields.items():
            values = getattr(payload, list_field, []) or []
            if index < len(values):
                updates[scalar_field] = values[index]

        if updates and hasattr(payload, "model_copy"):
            return payload.model_copy(update=updates)
        return payload

    async def _forward_upscaling_to_service(self, payload_url: str, task_type: str) -> str | None:
        task_id = uuid.uuid4().hex[:12]
        if PROCESSING_BACKEND == "modal":
            return await self._forward_upscaling_to_modal(payload_url, task_type, task_id)

        input_host_path = None
        output_host_path = None
        try:
            input_host_path, input_container_path = await self._download_to_shared_volume(payload_url, task_id)
            processed_ref = await self._post_processing_service(
                "upscaling",
                UPSCALING_SERVICE_URL,
                "/upscale",
                self._upscaling_service_payload(input_container_path, task_type, task_id),
                UPSCALING_SERVICE_TIMEOUT_SECONDS,
            )
            if processed_ref is None:
                return None
            if _is_url(processed_ref):
                return processed_ref

            output_host_path = self._host_shared_path(processed_ref)
            self._track_shared_file(output_host_path)
            return await self._upload_processed_video(output_host_path, "upscaling", task_id)
        finally:
            self._cleanup_shared_files(input_host_path, output_host_path)

    async def _forward_compression_to_service(self, payload) -> str | None:
        payload_urls = self._payload_reference_video_urls(payload)
        if not payload_urls:
            logger.error(f"Compression payload missing reference video URLs: {self._payload_debug_dump(payload)}")
            return None
        return await self._forward_compression_url_to_service(payload, payload_urls[0])

    async def forward_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> VideoUpscalingProtocol:
        """
        Processes a video upscaling request by downloading, upscaling,
        uploading, and returning a sharing link.
        """
        
        start_time = time.time()

        payload_urls = self._payload_reference_video_urls(synapse.miner_payload)
        task_types = self._payload_task_types(synapse.miner_payload, len(payload_urls))
        task_type = task_types[0] if task_types else "HD24K"
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        logger.info(
            f"✅✅✅ Receiving {task_type} Request from validator: {synapse.dendrite.hotkey} "
            f"with uid: {validator_uid}: round_id : {synapse.round_id} | queries={len(payload_urls)}"
        )
        
        check_version(synapse.version)

        try:
            processed_video_urls = await self._forward_upscaling_payload_to_service(synapse.miner_payload)
            
            if not any(processed_video_urls):
                logger.info(f"💔 Failed to upscaling video 💔")
                return synapse
            
            synapse.miner_response.optimized_video_urls = processed_video_urls
            synapse.miner_response.optimized_video_url = processed_video_urls[0] if processed_video_urls else ""

            processed_time = time.time() - start_time

            logger.info(f"💜 Returning Response, Processed in {processed_time:.2f} seconds 💜")
            
            return synapse
            
        except Exception as e:
            logger.error(f"Failed to process upscaling request: {e}")
            traceback.print_exc()
            return synapse

    async def forward_compression_requests(self, synapse: VideoCompressionProtocol) -> VideoCompressionProtocol:
        """
        Processes a video compression request by downloading, compressing,
        uploading, and returning a sharing link.
        """

        start_time = time.time()

        vmaf_threshold: float = synapse.miner_payload.vmaf_threshold
        target_codec: str = synapse.miner_payload.target_codec
        codec_mode: str = synapse.miner_payload.codec_mode
        target_bitrate: float = synapse.miner_payload.target_bitrate
        payload_urls = self._payload_reference_video_urls(synapse.miner_payload)
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        logger.info(
            f"🛜🛜🛜 Receiving CompressionRequest from validator: {synapse.dendrite.hotkey} "
            f"with uid: {validator_uid} | queries={len(payload_urls)} | VMAF: {vmaf_threshold} | "
            f"Codec: {target_codec} | Mode: {codec_mode} | Bitrate: {target_bitrate} Mbps 🛜🛜🛜"
        )

        check_version(synapse.version)

        try:
            processed_video_urls = await self._forward_compression_payload_to_service(synapse.miner_payload)

            if not any(processed_video_urls):
                logger.info(f"💔 Failed to compress video 💔")
                return synapse

            synapse.miner_response.optimized_video_urls = processed_video_urls
            synapse.miner_response.optimized_video_url = processed_video_urls[0] if processed_video_urls else ""

            processed_time = time.time() - start_time

            logger.info(f"💜 Returning Response, Processed in {processed_time:.2f} seconds 💜")

            return synapse

        except Exception as e:
            logger.error(f"Failed to process compression request: {e}")
            traceback.print_exc()
            return synapse

    async def forward_length_check_requests(self, synapse: LengthCheckProtocol) -> LengthCheckProtocol:

        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        logger.info(f"⭐️⭐️⭐️ Receiving LengthCheckRequest from validator: {synapse.dendrite.hotkey} with uid: {validator_uid} ⭐️⭐️⭐️")

        check_version(synapse.version)

        synapse.max_content_length = MAX_CONTENT_LEN

        return synapse

    async def forward_task_warrant_requests(self, synapse: TaskWarrantProtocol) -> TaskWarrantProtocol:
        """
        Processes a task warrant request by verifying the task type and returning a warrant.
        """

        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        logger.info(f"🌕🌕🌕 Receiving TaskWarrantRequest from validator: {synapse.dendrite.hotkey} with uid: {validator_uid} 🌕🌕🌕")

        check_version(synapse.version)

        synapse.warrant_task = warrant_task

        return synapse

    # ---------------------------------------------------------------------------
    # Polling-based organic compression handlers
    # ---------------------------------------------------------------------------

    async def forward_compression_job_requests(
        self, synapse: VideoCompressionJobProtocol
    ) -> VideoCompressionJobProtocol:
        """Phase-1: Accept a compression job and return a job_id immediately."""
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        logger.info(
            f"📥 Receiving CompressionJob from validator uid {validator_uid} | "
            f"Codec: {synapse.miner_payload.target_codec} | Mode: {synapse.miner_payload.codec_mode}"
        )

        try:
            job_id = synapse.job_id  # assigned by the validator
            if not job_id:
                logger.warning("CompressionJob received with empty job_id, rejecting")
                synapse.job_response = JobKickoffResponse(accepted=False)
                return synapse

            async def _run_compression():
                result = await self._forward_compression_payload_to_service(synapse.miner_payload)
                logger.info(f"CompressionJob processed successfully | job_id={job_id} | result={result}")
                return result

            task = asyncio.create_task(_run_compression())
            self._jobs[job_id] = {"task": task, "result": None, "error": None}
            synapse.job_response = JobKickoffResponse(accepted=True)
            logger.info(f"📤 CompressionJob accepted | job_id={job_id}")
        except Exception as e:
            logger.error(f"Failed to accept compression job: {e}")
            synapse.job_response = JobKickoffResponse(accepted=False)

        return synapse

    async def forward_compression_poll_requests(
        self, synapse: VideoCompressionPollProtocol
    ) -> VideoCompressionPollProtocol:
        """Phase-2: Return the current status (and result URL if done) for a compression job."""
        logger.info(f"⭐️⭐️⭐️ Receiving CompressionPollRequest from validator: {synapse.dendrite.hotkey} with uid: {self.metagraph.hotkeys.index(synapse.dendrite.hotkey)} ⭐️⭐️⭐️")
        job_id = synapse.job_id
        job = self._jobs.get(job_id)

        if job is None:
            logger.warning(f"Compression poll: unknown job_id={job_id}")
            synapse.poll_response = PollResponse(job_id=job_id, status="failed")
            return synapse

        task: asyncio.Task = job["task"]

        if not task.done():
            synapse.poll_response = PollResponse(job_id=job_id, status="processing")
            return synapse

        exc = task.exception()
        if exc is not None:
            logger.error(f"Compression job {job_id} failed: {exc}")
            synapse.poll_response = PollResponse(job_id=job_id, status="failed")
            self._jobs.pop(job_id, None)
            return synapse

        result = task.result() or []
        result_urls = [result] if isinstance(result, str) else list(result)
        result_url = result_urls[0] if result_urls else ""
        logger.info(f"Compression job {job_id} completed | urls={result_urls}")
        synapse.poll_response = PollResponse(
            job_id=job_id,
            status="completed",
            optimized_video_url=result_url,
            optimized_video_urls=result_urls,
        )
        self._jobs.pop(job_id, None)
        return synapse

    async def blacklist_compression_job_requests(
        self, synapse: VideoCompressionJobProtocol
    ) -> Tuple[bool, str]:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return True, "Missing dendrite or hotkey"
        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self._should_blacklist_non_validator(uid):
            return True, "Non-validator hotkey"
        return False, "Hotkey recognized!"

    async def priority_compression_job_requests(
        self, synapse: VideoCompressionJobProtocol
    ) -> float:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return 0.0
        caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    async def blacklist_compression_poll_requests(
        self, synapse: VideoCompressionPollProtocol
    ) -> Tuple[bool, str]:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return True, "Missing dendrite or hotkey"
        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self._should_blacklist_non_validator(uid):
            return True, "Non-validator hotkey"
        return False, "Hotkey recognized!"

    async def priority_compression_poll_requests(
        self, synapse: VideoCompressionPollProtocol
    ) -> float:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return 0.0
        caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    # ---------------------------------------------------------------------------
    # Polling-based organic upscaling handlers
    # ---------------------------------------------------------------------------

    async def forward_upscaling_job_requests(
        self, synapse: VideoUpscalingJobProtocol
    ) -> VideoUpscalingJobProtocol:
        """Phase-1: Accept an upscaling job and return a job_id immediately."""
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        logger.info(
            f"📥 Receiving UpscalingJob from validator uid {validator_uid} | "
            f"task_type: {synapse.miner_payload.task_type}"
        )

        try:
            job_id = synapse.job_id  # assigned by the validator
            if not job_id:
                logger.warning("UpscalingJob received with empty job_id, rejecting")
                synapse.job_response = JobKickoffResponse(accepted=False)
                return synapse

            async def _run_upscaling():
                result = await self._forward_upscaling_payload_to_service(synapse.miner_payload)
                logger.info(f"UpscalingJob processed successfully | job_id={job_id} | result={result}")
                return result

            task = asyncio.create_task(_run_upscaling())
            self._jobs[job_id] = {"task": task, "result": None, "error": None}
            synapse.job_response = JobKickoffResponse(accepted=True)
            logger.info(f"📤 UpscalingJob accepted | job_id={job_id}")
        except Exception as e:
            logger.error(f"Failed to accept upscaling job: {e}")
            synapse.job_response = JobKickoffResponse(accepted=False)

        return synapse

    async def forward_upscaling_poll_requests(
        self, synapse: VideoUpscalingPollProtocol
    ) -> VideoUpscalingPollProtocol:
        """Phase-2: Return the current status (and result URL if done) for an upscaling job."""
        job_id = synapse.job_id
        job = self._jobs.get(job_id)

        if job is None:
            logger.warning(f"Upscaling poll: unknown job_id={job_id}")
            synapse.poll_response = PollResponse(job_id=job_id, status="failed")
            return synapse

        task: asyncio.Task = job["task"]

        if not task.done():
            synapse.poll_response = PollResponse(job_id=job_id, status="processing")
            return synapse

        exc = task.exception()
        if exc is not None:
            logger.error(f"Upscaling job {job_id} failed: {exc}")
            synapse.poll_response = PollResponse(job_id=job_id, status="failed")
            self._jobs.pop(job_id, None)
            return synapse

        result = task.result() or []
        result_urls = [result] if isinstance(result, str) else list(result)
        result_url = result_urls[0] if result_urls else ""
        logger.info(f"Upscaling job {job_id} completed | urls={result_urls}")
        synapse.poll_response = PollResponse(
            job_id=job_id,
            status="completed",
            optimized_video_url=result_url,
            optimized_video_urls=result_urls,
        )
        self._jobs.pop(job_id, None)
        return synapse

    async def blacklist_upscaling_job_requests(
        self, synapse: VideoUpscalingJobProtocol
    ) -> Tuple[bool, str]:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return True, "Missing dendrite or hotkey"
        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self._should_blacklist_non_validator(uid):
            return True, "Non-validator hotkey"
        return False, "Hotkey recognized!"

    async def priority_upscaling_job_requests(
        self, synapse: VideoUpscalingJobProtocol
    ) -> float:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return 0.0
        caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    async def blacklist_upscaling_poll_requests(
        self, synapse: VideoUpscalingPollProtocol
    ) -> Tuple[bool, str]:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return True, "Missing dendrite or hotkey"
        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self._should_blacklist_non_validator(uid):
            return True, "Non-validator hotkey"
        return False, "Hotkey recognized!"

    async def priority_upscaling_poll_requests(
        self, synapse: VideoUpscalingPollProtocol
    ) -> float:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return 0.0
        caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    async def blacklist_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> Tuple[bool, str]:
        """
        Determines whether a request should be blacklisted based on the hotkey status.
        """
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            logger.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if self._should_blacklist_non_validator(uid):
            logger.warning(f"Blacklisting non-validator hotkey {synapse.dendrite.hotkey}")
            return True, "Non-validator hotkey"

        logger.trace(f"Hotkey {synapse.dendrite.hotkey} recognized and allowed.")
        return False, "Hotkey recognized!"

    async def priority_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> float:
        """
        Assigns a priority to requests based on the stake of the requesting entity.
        Higher stakes result in higher priority.
        """
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            logger.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority: float = float(self.metagraph.S[caller_uid])
        
        logger.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
        return priority

    async def blacklist_length_check_requests(self, synapse: LengthCheckProtocol) -> Tuple[bool, str]:
        """
        Determines whether a request should be blacklisted based on the hotkey status.
        """
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            logger.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if self._should_blacklist_non_validator(uid):
            logger.warning(f"Blacklisting non-validator hotkey {synapse.dendrite.hotkey}")
            return True, "Non-validator hotkey"

        logger.trace(f"Hotkey {synapse.dendrite.hotkey} recognized and allowed.")
        return False, "Hotkey recognized!"

    async def priority_length_check_requests(self, synapse: LengthCheckProtocol) -> float:
        """
        Assigns a priority to requests based on the stake of the requesting entity.
        Higher stakes result in higher priority.
        """
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            logger.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority: float = float(self.metagraph.S[caller_uid])
        
        logger.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
        return priority

    async def blacklist_compression_requests(self, synapse: VideoCompressionProtocol) -> Tuple[bool, str]:
        """
        Determines whether a request should be blacklisted based on the hotkey status.
        """
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            logger.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if self._should_blacklist_non_validator(uid):
            logger.warning(f"Blacklisting non-validator hotkey {synapse.dendrite.hotkey}")
            return True, "Non-validator hotkey"

        logger.trace(f"Hotkey {synapse.dendrite.hotkey} recognized and allowed.")
        return False, "Hotkey recognized!"

    async def priority_compression_requests(self, synapse: VideoCompressionProtocol) -> float:
        """
        Assigns a priority to requests based on the stake of the requesting entity.
        Higher stakes result in higher priority.
        """
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            logger.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority: float = float(self.metagraph.S[caller_uid])
        
        logger.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
        return priority

    async def blacklist_task_warrant_requests(self, synapse: TaskWarrantProtocol) -> Tuple[bool, str]:
        """
        Determines whether a request should be blacklisted based on the hotkey status.
        """
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            logger.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"   

        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if self._should_blacklist_non_validator(uid):
            logger.warning(f"Blacklisting non-validator hotkey {synapse.dendrite.hotkey}")
            return True, "Non-validator hotkey"

        logger.trace(f"Hotkey {synapse.dendrite.hotkey} recognized and allowed.")
        return False, "Hotkey recognized!"

    async def priority_task_warrant_requests(self, synapse: TaskWarrantProtocol) -> float:
        """
        Assigns a priority to requests based on the stake of the requesting entity.
        Higher stakes result in higher priority.
        """
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            logger.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority: float = float(self.metagraph.S[caller_uid])

        logger.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
        return priority

if __name__ == "__main__":
    with Miner() as miner:
        while True:
            logger.info(f"Miner running... {time.time()}")
            time.sleep(50)
