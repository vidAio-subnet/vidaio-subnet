import asyncio
import os
import posixpath
import time
import traceback
import uuid
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

UPSCALING_SERVICE_URL = os.getenv("MINER_UPSCALING_SERVICE_URL", "http://localhost:8003").rstrip("/")
COMPRESSION_SERVICE_URL = os.getenv("MINER_COMPRESSION_SERVICE_URL", "http://localhost:8004").rstrip("/")
UPSCALING_SERVICE_TIMEOUT_SECONDS = float(os.getenv("MINER_UPSCALING_SERVICE_TIMEOUT_SECONDS", "1200"))
COMPRESSION_SERVICE_TIMEOUT_SECONDS = float(os.getenv("MINER_COMPRESSION_SERVICE_TIMEOUT_SECONDS", "600"))
MINER_DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("MINER_DOWNLOAD_TIMEOUT_SECONDS", "600"))
MINER_UPLOAD_URL_EXPIRY_SECONDS = int(os.getenv("ORGANIC_PROXY_STORAGE_S3_PRESIGNED_EXPIRY", os.getenv("S3_PRESIGNED_EXPIRY", "3600")))

HOST_SHARED_VOLUME_PATH = Path(
    os.getenv("MINER_SHARED_VOLUME_PATH")
    or os.getenv("ORGANIC_PROXY_SHARED_DIR", "/tmp/vidaio-miner-video-tmp")
).expanduser()
if not HOST_SHARED_VOLUME_PATH.is_absolute():
    HOST_SHARED_VOLUME_PATH = REPO_ROOT / HOST_SHARED_VOLUME_PATH
CONTAINER_SHARED_VOLUME_PATH = os.getenv("MINER_CONTAINER_SHARED_VOLUME_PATH", "/tmp/organic-proxy").rstrip("/")

S3_REGION = os.getenv("ORGANIC_PROXY_STORAGE_S3_REGION", "us-east-1").strip() or "us-east-1"
S3_BUCKET = os.getenv("ORGANIC_PROXY_STORAGE_S3_BUCKET_NAME", "").strip()
S3_ACCESS_KEY_ID = os.getenv("ORGANIC_PROXY_STORAGE_S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("ORGANIC_PROXY_STORAGE_S3_SECRET_ACCESS_KEY", "").strip()
S3_ENDPOINT_URL = (
    os.getenv("ORGANIC_PROXY_STORAGE_S3_ENDPOINT_URL")
    or os.getenv("ORGANIC_PROXY_STORAGE_S3_ENDPOINT")
    or ""
).strip()

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
            missing.append("ORGANIC_PROXY_STORAGE_S3_BUCKET_NAME")
        if not S3_ACCESS_KEY_ID:
            missing.append("ORGANIC_PROXY_STORAGE_S3_ACCESS_KEY_ID")
        if not S3_SECRET_ACCESS_KEY:
            missing.append("ORGANIC_PROXY_STORAGE_S3_SECRET_ACCESS_KEY")

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
            "video_path": video_path,
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
            "video_path": video_path,
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
            logger.error(f"{service_name} service failed: {result.get('error')}")
            return None

        processed_video_url = result.get("output_url") or result.get("output_path")
        if not processed_video_url:
            logger.error(f"{service_name} service returned no output URL/path")
            return None

        logger.info(f"Received {service_name} result from container service")
        return processed_video_url

    async def _forward_upscaling_to_service(self, payload_url: str, task_type: str) -> str | None:
        task_id = uuid.uuid4().hex[:12]
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
            return await self._upload_processed_video(output_host_path, "upscaling", task_id)
        finally:
            self._cleanup_shared_files(input_host_path, output_host_path)

    async def _forward_compression_to_service(self, payload) -> str | None:
        task_id = uuid.uuid4().hex[:12]
        input_host_path = None
        output_host_path = None
        try:
            input_host_path, input_container_path = await self._download_to_shared_volume(
                payload.reference_video_url, task_id
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
            return await self._upload_processed_video(output_host_path, "compression", task_id)
        finally:
            self._cleanup_shared_files(input_host_path, output_host_path)

    async def forward_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> VideoUpscalingProtocol:
        """
        Processes a video upscaling request by downloading, upscaling,
        uploading, and returning a sharing link.
        """
        
        start_time = time.time()

        task_type: str = synapse.miner_payload.task_type      
        payload_url: str = synapse.miner_payload.reference_video_url
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        logger.info(f"✅✅✅ Receiving {task_type} Request from validator: {synapse.dendrite.hotkey} with uid: {validator_uid}: round_id : {synapse.round_id}")
        
        check_version(synapse.version)

        try:
            processed_video_url = await self._forward_upscaling_to_service(payload_url, task_type)
            
            if processed_video_url is None:
                logger.info(f"💔 Failed to upscaling video 💔")
                return synapse
            
            synapse.miner_response.optimized_video_url = processed_video_url

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
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        logger.info(f"🛜🛜🛜 Receiving CompressionRequest from validator: {synapse.dendrite.hotkey} with uid: {validator_uid} | VMAF: {vmaf_threshold} | Codec: {target_codec} | Mode: {codec_mode} | Bitrate: {target_bitrate} Mbps 🛜🛜🛜")

        check_version(synapse.version)

        try:
            processed_video_url = await self._forward_compression_to_service(synapse.miner_payload)

            if processed_video_url is None:
                logger.info(f"💔 Failed to compress video 💔")
                return synapse

            synapse.miner_response.optimized_video_url = processed_video_url

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
                return await self._forward_compression_to_service(synapse.miner_payload)

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

        result_url: str = task.result() or ""
        logger.info(f"Compression job {job_id} completed | url={result_url}")
        synapse.poll_response = PollResponse(
            job_id=job_id, status="completed", optimized_video_url=result_url
        )
        self._jobs.pop(job_id, None)
        return synapse

    async def blacklist_compression_job_requests(
        self, synapse: VideoCompressionJobProtocol
    ) -> Tuple[bool, str]:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return True, "Missing dendrite or hotkey"
        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.metagraph.validator_permit[uid]:
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
        if not self.metagraph.validator_permit[uid]:
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

            payload_url = synapse.miner_payload.reference_video_url
            task_type = synapse.miner_payload.task_type

            async def _run_upscaling():
                return await self._forward_upscaling_to_service(payload_url, task_type)

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

        result_url: str = task.result() or ""
        logger.info(f"Upscaling job {job_id} completed | url={result_url}")
        synapse.poll_response = PollResponse(
            job_id=job_id, status="completed", optimized_video_url=result_url
        )
        self._jobs.pop(job_id, None)
        return synapse

    async def blacklist_upscaling_job_requests(
        self, synapse: VideoUpscalingJobProtocol
    ) -> Tuple[bool, str]:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return True, "Missing dendrite or hotkey"
        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.metagraph.validator_permit[uid]:
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
        if not self.metagraph.validator_permit[uid]:
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

        if not self.metagraph.validator_permit[uid]:
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

        if not self.metagraph.validator_permit[uid]:
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

        if not self.metagraph.validator_permit[uid]:
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

        if not self.metagraph.validator_permit[uid]:
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
