import time
import uuid
import traceback
from typing import Tuple
from loguru import logger
import bittensor as bt
from vidaio_subnet_core.base.miner import BaseMiner
from vidaio_subnet_core.protocol import VideoUpscalingProtocol
from services.miner_utilities.miner_utils import download_video, video_upscaler
from vidaio_subnet_core.utilities.minio_client import minio_client

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
        logger.info(f"Receiving Request: {synapse}")
        
        try:
            payload_url: str = synapse.miner_payload.reference_video_url
            payload_video_path: str = await download_video(payload_url)
            processed_video_path: str = await video_upscaler(payload_video_path)
            logger.info(f"Processed video path: {processed_video_path}")

            uploaded_file_id: uuid.UUID = uuid.uuid4()
            object_name: str = f"{uploaded_file_id}.mp4"
            
            await minio_client.upload_file(object_name, processed_video_path)
            logger.info("Video uploaded successfully.")
            
            sharing_link: str | None = await minio_client.get_presigned_url(object_name)
            if not sharing_link:
                logger.error("Upload failed")
                return synapse
            
            logger.info(f"Public download link: {sharing_link}")  
            synapse.miner_response.optimized_video_url = sharing_link

            logger.info(f"Returning Response: {synapse}")
            return synapse
        
        except Exception as e:
            logger.error(f"Failed to process upscaling request: {e}")
            traceback.print_exc()
            return synapse

    async def blacklist_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> Tuple[bool, str]:
        """
        Determines whether a request should be blacklisted based on the hotkey status.
        """
        if not synapse.dendrite or not synapse.dendrite.hotkey:
            logger.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            logger.trace(f"Blacklisting unregistered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit and not self.metagraph.validator_permit[uid]:
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

if __name__ == "__main__":
    with Miner() as miner:
        while True:
            logger.info(f"Miner running... {time.time()}")
            time.sleep(30)
