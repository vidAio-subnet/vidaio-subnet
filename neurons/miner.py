import time
import uuid
import traceback
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
)
from services.miner_utilities.chutes_client import call_chute_upscaling, call_chute_compression
from vidaio_subnet_core.utilities import storage_client
from vidaio_subnet_core.utilities.version import check_version

MAX_CONTENT_LEN = ContentLength.FIVE
warrant_task = TaskType.UPSCALING


class Miner(BaseMiner):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()

    async def forward_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> VideoUpscalingProtocol:
        start_time = time.time()

        task_type: str = synapse.miner_payload.task_type
        payload_url: str = synapse.miner_payload.reference_video_url
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        logger.info(f"Receiving {task_type} upscaling request from validator uid: {validator_uid} | round_id: {synapse.round_id}")

        check_version(synapse.version)

        try:
            object_name = f"{uuid.uuid4()}_upscaled.mp4"
            upload_url = await storage_client.get_presigned_put_url(object_name)
            logger.info(f"Calling upscaling chute with payload_url: {payload_url}, upload_url: {upload_url}, task_type: {task_type}")
            processed_video_url = await call_chute_upscaling(
                video_url=payload_url,
                task_type=task_type,
                upload_url=upload_url,
            )

            if processed_video_url is None:
                logger.info("Failed to upscale video via chute")
                return synapse

            download_url = await storage_client.get_presigned_url(object_name)
            synapse.miner_response.optimized_video_url = download_url

            logger.info(f"Upscaling completed in {time.time() - start_time:.2f}s")
            return synapse

        except Exception as e:
            logger.error(f"Failed to process upscaling request: {e}")
            traceback.print_exc()
            return synapse

    async def forward_compression_requests(self, synapse: VideoCompressionProtocol) -> VideoCompressionProtocol:
        start_time = time.time()

        payload_url: str = synapse.miner_payload.reference_video_url
        vmaf_threshold: float = synapse.miner_payload.vmaf_threshold
        target_codec: str = synapse.miner_payload.target_codec
        codec_mode: str = synapse.miner_payload.codec_mode
        target_bitrate: float = synapse.miner_payload.target_bitrate
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        logger.info(f"Receiving compression request from validator uid: {validator_uid} | VMAF: {vmaf_threshold} | Codec: {target_codec} | Mode: {codec_mode} | Bitrate: {target_bitrate} Mbps")

        check_version(synapse.version)

        try:
            object_name = f"{uuid.uuid4()}_compressed.mp4"
            upload_url = await storage_client.get_presigned_put_url(object_name)
            logger.info(f"Calling compression chute with payload_url: {payload_url}, upload_url: {upload_url}, target_codec: {target_codec}, codec_mode: {codec_mode}, target_bitrate: {target_bitrate}")
            processed_video_url = await call_chute_compression(
                video_url=payload_url,
                vmaf_threshold=vmaf_threshold,
                target_codec=target_codec,
                codec_mode=codec_mode,
                target_bitrate=target_bitrate,
                upload_url=upload_url,
            )

            if processed_video_url is None:
                logger.info("Failed to compress video via chute")
                return synapse

            download_url = await storage_client.get_presigned_url(object_name)
            synapse.miner_response.optimized_video_url = download_url

            logger.info(f"Compression completed in {time.time() - start_time:.2f}s")
            return synapse

        except Exception as e:
            logger.error(f"Failed to process compression request: {e}")
            traceback.print_exc()
            return synapse

    async def forward_length_check_requests(self, synapse: LengthCheckProtocol) -> LengthCheckProtocol:
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        logger.info(f"Receiving LengthCheckRequest from validator uid: {validator_uid}")
        check_version(synapse.version)
        synapse.max_content_length = MAX_CONTENT_LEN
        return synapse

    async def forward_task_warrant_requests(self, synapse: TaskWarrantProtocol) -> TaskWarrantProtocol:
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        logger.info(f"Receiving TaskWarrantRequest from validator uid: {validator_uid}")
        check_version(synapse.version)
        synapse.warrant_task = warrant_task
        return synapse

    # --- Blacklist / Priority (shared logic) ---

    def _validate_hotkey(self, synapse) -> Tuple[bool, str]:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return True, "Missing dendrite or hotkey"
        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.metagraph.validator_permit[uid]:
            logger.warning(f"Blacklisting non-validator hotkey {synapse.dendrite.hotkey}")
            return True, "Non-validator hotkey"
        return False, "Hotkey recognized!"

    def _get_priority(self, synapse) -> float:
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            return 0.0
        caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    async def blacklist_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> Tuple[bool, str]:
        return self._validate_hotkey(synapse)

    async def priority_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> float:
        return self._get_priority(synapse)

    async def blacklist_compression_requests(self, synapse: VideoCompressionProtocol) -> Tuple[bool, str]:
        return self._validate_hotkey(synapse)

    async def priority_compression_requests(self, synapse: VideoCompressionProtocol) -> float:
        return self._get_priority(synapse)

    async def blacklist_length_check_requests(self, synapse: LengthCheckProtocol) -> Tuple[bool, str]:
        return self._validate_hotkey(synapse)

    async def priority_length_check_requests(self, synapse: LengthCheckProtocol) -> float:
        return self._get_priority(synapse)

    async def blacklist_task_warrant_requests(self, synapse: TaskWarrantProtocol) -> Tuple[bool, str]:
        return self._validate_hotkey(synapse)

    async def priority_task_warrant_requests(self, synapse: TaskWarrantProtocol) -> float:
        return self._get_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            logger.info(f"Miner running... {time.time()}")
            time.sleep(50)
