import time
import uuid
import traceback
import os
from typing import Tuple
from loguru import logger
import bittensor as bt
from vidaio_subnet_core.base.miner import BaseMiner
from vidaio_subnet_core.protocol import VideoUpscalingProtocol
from services.miner_utilities.miner_utils import video_upscaler

from vidaio_subnet_core.utilities.version import check_version


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
        
        task_type: str = synapse.miner_payload.task_type      
        payload_url: str = synapse.miner_payload.reference_video_url
        validator_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        logger.info(f"✅✅✅ Receiving {task_type} Request from validator: {synapse.dendrite.hotkey} with uid: {validator_uid}")
        
        check_version(synapse.version)

        try:
            processed_video_url = await video_upscaler(payload_url, task_type)
            
            if processed_video_url is None:
                return synapse
            
            synapse.miner_response.optimized_video_url = processed_video_url

            logger.info(f"Returning Response: {synapse}")
            
            return synapse
            
        except Exception as e:
            logger.error(f"Failed to process upscaling request: {e}")
            traceback.print_exc()
            return synapse


        try:
            payload_video_path: str = await download_video(payload_url)
            processed_video_name, processed_video_path = await video_upscaler(payload_video_path, task_type)

            logger.info(f"Processed video path: {processed_video_path}")
            if processed_video_path is not None:
                object_name: str = processed_video_name
                
                await minio_client.upload_file(object_name, processed_video_path)
                logger.info("Video uploaded successfully.")
                
                # Delete the local file since we've already uploaded it to MinIO
                if os.path.exists(processed_video_path):
                    os.remove(processed_video_path)
                    logger.info(f"{processed_video_path} has been deleted.")
                else:
                    logger.info(f"{processed_video_path} does not exist.")
                    
                sharing_link: str | None = await minio_client.get_presigned_url(object_name)
                if not sharing_link:
                    logger.error("Upload failed")
                    return synapse
                
                # Schedule the file for deletion after 10 minutes (600 seconds)
                deletion_scheduled = schedule_file_deletion(object_name)
                if deletion_scheduled:
                    logger.info(f"Scheduled deletion of {object_name} after 10 minutes")
                else:
                    logger.warning(f"Failed to schedule deletion of {object_name}")
                
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
