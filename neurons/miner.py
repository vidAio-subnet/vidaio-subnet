import bittensor as bt
from video_subnet_core.base.miner import BaseMiner
from typing import Tuple
import time
from loguru import logger
from video_subnet_core.protocol import VideoUpscalingProtocol
from services.miner_utilities.miner_utils import download_video, video_upscaler
from services.google_drive.google_drive_manager import GoogleDriveManager
import traceback
import asyncio

class Miner(BaseMiner):
    
    def __init__(self, config=None):
        super(Miner, self).__init__()

    async def forward_upscaling_requests(self, synapse: VideoUpscalingProtocol):
        
        try:
            payload_url = synapse.miner_payload.reference_video_url
            allowed_maximum_size = synapse.miner_payload.maximum_optimized_size_mb
            payload_video_path = download_video(payload_url)
            processed_video_path = await video_upscaler(payload_video_path)
            
            gdrive = GoogleDriveManager("video-448518-c557bc123b1b.json")
            
            folder_id = gdrive.create_folder("uploaded_video")
            
            gdrive.list_files(parent_folder_id=folder_id)
            # uploaded_file_id, sharing_link = gdrive.upload_file(
            #     "/Users/mac/Documents/work/video-streaming/vidaio-subnet/services/upscaling/videos/7dacc160-ef37-40a6-a7c4-2da0cc8f2e8b.mp4", 
            #     folder_id
            # )
            
            uploaded_file_id, sharing_link = gdrive.upload_file(processed_video_path, folder_id)
        
            if sharing_link:
                print(f"Public download link: {sharing_link}")  
                synapse.miner_response = sharing_link
                
                # Schedule the deletion of the uploaded file after 60 seconds
                await asyncio.sleep(60)
                gdrive.delete_files(uploaded_file_id)
            
            return synapse
            
        except Exception as e:
            logger.error(f"Failed to run forward_upscaling_request: {e}")
            traceback.print_exc()


    async def blacklist_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (vectornet.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            logger.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                logger.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        logger.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (vectornet.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        logger.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority
    
    
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            logger.info(f"Miner running... {time.time()}")
            time.sleep(30)