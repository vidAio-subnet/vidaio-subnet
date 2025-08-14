import time
import uuid
import traceback
import os
from typing import Tuple
from loguru import logger
import bittensor as bt
from vidaio_subnet_core.base.miner import BaseMiner
from vidaio_subnet_core.protocol import VideoUpscalingProtocol, LengthCheckProtocol, ContentLength, VideoCompressionProtocol, TaskWarrantProtocol, TaskType
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
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        
        logger.info(f"🛜🛜🛜 Receiving CompressionRequest from validator: {synapse.dendrite.hotkey} with uid: {validator_uid} 🛜🛜🛜")

        check_version(synapse.version)
        
        try:
            processed_video_url = await video_compressor(payload_url, vmaf_threshold)
            
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
