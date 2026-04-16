import time
import uuid
import traceback
import os
import asyncio
from typing import Tuple
from loguru import logger
import bittensor as bt
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
from services.miner_utilities.miner_utils import video_upscaler, video_compressor

from vidaio_subnet_core.utilities.version import check_version

MAX_CONTENT_LEN = ContentLength.FIVE
warrant_task = TaskType.UPSCALING

class Miner(BaseMiner):
    def __init__(self, config: dict | None = None) -> None:
        """
        Initializes the Miner instance.
        """
        super().__init__()
        # Shared in-memory job store for polling-based organic requests.
        # Structure: { job_id: {"task": asyncio.Task, "result": str | None, "error": str | None} }
        self._jobs: dict[str, dict] = {}

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
            processed_video_url = await video_upscaler(payload_url, task_type)
            
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

        payload_url: str = synapse.miner_payload.reference_video_url
        vmaf_threshold: float = synapse.miner_payload.vmaf_threshold
        target_codec: str = synapse.miner_payload.target_codec
        codec_mode: str = synapse.miner_payload.codec_mode
        target_bitrate: float = synapse.miner_payload.target_bitrate
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        logger.info(f"🛜🛜🛜 Receiving CompressionRequest from validator: {synapse.dendrite.hotkey} with uid: {validator_uid} | VMAF: {vmaf_threshold} | Codec: {target_codec} | Mode: {codec_mode} | Bitrate: {target_bitrate} Mbps 🛜🛜🛜")

        check_version(synapse.version)

        try:
            processed_video_url = await video_compressor(payload_url, vmaf_threshold, target_codec, codec_mode, target_bitrate)

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

            payload_url = synapse.miner_payload.reference_video_url
            vmaf_threshold = synapse.miner_payload.vmaf_threshold
            target_codec = synapse.miner_payload.target_codec
            codec_mode = synapse.miner_payload.codec_mode
            target_bitrate = synapse.miner_payload.target_bitrate

            async def _run_compression():
                return await video_compressor(
                    payload_url, vmaf_threshold, target_codec, codec_mode, target_bitrate
                )

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
                return await video_upscaler(payload_url, task_type)

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
