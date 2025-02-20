# from vidaio_subnet_core import validating, CONFIG, base, protocol
# import bittensor as bt
# import random
# import httpx
# import asyncio
# from loguru import logger
# import traceback
# import pandas as pd
# from typing import List
# from vidaio_subnet_core.utilities.minio_client import minio_client
# from services.video_scheduler.video_utils import get_4k_vide_path, delete_videos_with_fileid
# from concurrent.futures import ThreadPoolExecutor


# class Validator(base.BaseValidator):
#     def __init__(self):
#         super().__init__()
#         self.miner_manager = validating.managing.MinerManager(
#             uid=self.uid, wallet=self.wallet, metagraph=self.metagraph
#         )
#         logger.info("Initialized miner manager")
#         self.challenge_synthesizer = validating.synthesizing.Synthesizer()
#         logger.info("Initialized challenge synthesizer")
#         self.dendrite = bt.dendrite(wallet=self.wallet)
#         logger.info("Initialized dendrite")
#         self.score_client = httpx.AsyncClient(
#             base_url=f"http://{CONFIG.score.host}:{CONFIG.score.port}"
#         )
#         logger.info(
#             f"Initialized score client with base URL: {CONFIG.score.host}:{CONFIG.score.port}"
#         )
#         self.set_weights_executor = ThreadPoolExecutor(max_workers=1)

#     async def start_epoch(self):
#         logger.info("Starting forward pass")
#         uids = list(range(len(self.metagraph.hotkeys)))
#         logger.debug(f"Initial UIDs: {uids}")
#         uids = self.miner_manager.consume(uids)
#         # uids = [2]
#         logger.info(f"Filtered UIDs after consumption: {uids}")
#         axons = [self.metagraph.axons[uid] for uid in uids]
#         miners = list(zip(axons, uids))
#         batch_size = 20
#         miner_batches = [
#             miners[i : i + batch_size] for i in range(0, len(miners), batch_size)
#         ]
#         logger.info(f"Created {len(miner_batches)} batches of size {batch_size}")

#         for batch_idx, batch in enumerate(miner_batches):
#             logger.info(f"Processing batch {batch_idx + 1}/{len(miner_batches)}")
#             video_id, uploaded_object_name, synapse = await self.challenge_synthesizer.build_protocol()
#             logger.debug(f"Built challenge protocol {synapse.__dict__}")
#             uids = []
#             axons = []
#             for miner in batch:
#                 uids.append(miner[1])
#                 axons.append(miner[0])
#             logger.debug(f"Processing UIDs in batch: {uids}")
#             responses = await self.dendrite.forward(
#                 axons=axons, synapse=synapse, timeout=200
#             )
#             logger.info(f"Received {len(responses)} responses from miners, deleting uploaded_file")
#             logger.info(responses)
#             video_4k_path = get_4k_vide_path(video_id)
            
#             await self.score(uids, responses, video_4k_path)
            
#             # minio_client.delete_file(uploaded_object_name)
#             # delete_videos_with_fileid(video_id)
            
#             logger.debug("Waiting 4 seconds before next batch")
#             await asyncio.sleep(4)

#     async def score(self, uids: list[int], responses: list[protocol.Synapse], reference_4k_path: str):
#         logger.info(f"Starting scoring for {len(uids)} miners")
#         distorted_urls = []
#         print(responses, uids)
#         for uid, response in zip(uids, responses):
#             distorted_urls.append(response.miner_response.optimized_video_url)
#         logger.info(f"distored_urls: {distorted_urls}")
#         score_response = await self.score_client.post(
#             "/score",
#             json = {
#                 "distorted_urls": distorted_urls,
#                 "reference_path": reference_4k_path
#             },
#             timeout=60
#         )
#         response_json = score_response.json()  # Get the full JSON response
#         scores: List[float] = response_json.get("scores", [])  # Extract only the list
#         logger.info(f"Scores: {scores}")
#         logger.info(f"Updating miner manager with {len(scores)} scores")
#         self.miner_manager.step(scores, uids)


#     def set_weights(self):
#         self.current_block = self.subtensor.get_current_block()
#         self.last_update = self.metagraph.last_update[self.uid]
#         uids, weights = self.miner_manager.weights
#         (
#             processed_weight_uids,
#             processed_weights,
#         ) = bt.utils.weight_utils.process_weights_for_netuid(
#             # uids=self.metagraph.uids,
#             uids = uids,
#             weights=weights,
#             netuid=self.config.netuid,
#             subtensor=self.subtensor,
#             metagraph=self.metagraph,
#         )
#         (
#             uint_uids,
#             uint_weights,
#         ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
#             uids=processed_weight_uids, weights=processed_weights
#         )
#         if self.current_block > self.last_update + CONFIG.SUBNET_TEMPO:
#             weight_info = list(zip(uint_uids, uint_weights))
#             weight_info_df = pd.DataFrame(weight_info, columns=["uid", "weight"])
#             logger.info(f"Weight info:\n{weight_info_df.to_markdown()}")
#             logger.info("Actually trying to set weights.")
#             try:
#                 future = self.set_weights_executor.submit(
#                     self.subtensor.set_weights,
#                     netuid=self.config.netuid,
#                     wallet=self.wallet,
#                     uids=uint_uids,
#                     weights=uint_weights,
#                 )
#                 success, msg = future.result(timeout=120)
#                 if not success:
#                     logger.error(f"Failed to set weights: {msg}")
#                 else: 
#                     logger.debug("Set weights successfully 😎")
#             except Exception as e:
#                 logger.error(f"Failed to set weights: {e}")
#                 traceback.print_exc()

#             logger.info(f"Set weights result: {success}")
#         else:
#             logger.info(
#                 f"Not setting weights because current block {self.current_block} is not greater than last update {self.last_update} + tempo {CONFIG.SUBNET_TEMPO}"
#             )


# if __name__ == "__main__":
#     validator = Validator()
#     asyncio.run(validator.run())



from vidaio_subnet_core import validating, CONFIG, base, protocol
import bittensor as bt
import random
import httpx
import asyncio
from loguru import logger
import traceback
import pandas as pd
from typing import List
from vidaio_subnet_core.utilities.minio_client import minio_client
from services.video_scheduler.video_utils import get_4k_vide_path, delete_videos_with_fileid
from concurrent.futures import ThreadPoolExecutor


class Validator(base.BaseValidator):
    """
    Validator class responsible for managing miners, synthesizing challenges, 
    scoring responses, and setting weights in the network. 
    It operates with a batch processing mechanism to handle forward passes and scoring.

    Attributes:
        miner_manager: Manages miner actions such as consumption.
        challenge_synthesizer: Synthesizes challenge protocols for the miners.
        dendrite: Interface for interacting with the dendrite service.
        score_client: Async HTTP client for scoring miners.
        set_weights_executor: Thread pool executor for setting weights.
    """

    def __init__(self):
        super().__init__()
        self.miner_manager = validating.managing.MinerManager(
            uid=self.uid, wallet=self.wallet, metagraph=self.metagraph
        )
        logger.info("Initialized miner manager")
        self.challenge_synthesizer = validating.synthesizing.Synthesizer()
        logger.info("Initialized challenge synthesizer")
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info("Initialized dendrite")
        self.score_client = httpx.AsyncClient(
            base_url=f"http://{CONFIG.score.host}:{CONFIG.score.port}"
        )
        logger.info(
            f"Initialized score client with base URL: {CONFIG.score.host}:{CONFIG.score.port}"
        )
        self.set_weights_executor = ThreadPoolExecutor(max_workers=1)

    async def start_epoch(self) -> None:
        """
        Starts the forward pass for the miners, processes them in batches, 
        builds a challenge protocol, scores the miners, and then deletes the 
        uploaded files after scoring.

        This method involves interacting with the challenge synthesizer, 
        dendrite, and the scoring client in each batch processing loop.
        """
        logger.info("Starting forward pass")
        uids = list(range(len(self.metagraph.hotkeys)))
        logger.debug(f"Initial UIDs: {uids}")
        uids = self.miner_manager.consume(uids)
        logger.info(f"Filtered UIDs after consumption: {uids}")
        
        axons = [self.metagraph.axons[uid] for uid in uids]
        miners = list(zip(axons, uids))
        batch_size = 20
        miner_batches = [
            miners[i : i + batch_size] for i in range(0, len(miners), batch_size)
        ]
        logger.info(f"Created {len(miner_batches)} batches of size {batch_size}")

        for batch_idx, batch in enumerate(miner_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(miner_batches)}")
            video_id, uploaded_object_name, synapse = await self.challenge_synthesizer.build_protocol()
            logger.debug(f"Built challenge protocol {synapse.__dict__}")
            
            uids, axons = zip(*batch)
            logger.debug(f"Processing UIDs in batch: {uids}")
            responses = await self.dendrite.forward(
                axons=axons, synapse=synapse, timeout=200
            )
            logger.info(f"Received {len(responses)} responses from miners, deleting uploaded_file")
            logger.info(responses)
            
            video_4k_path = get_4k_vide_path(video_id)
            await self.score(uids, responses, video_4k_path)
            
            logger.debug("Waiting 4 seconds before next batch")
            await asyncio.sleep(4)

    async def score(
        self, 
        uids: List[int], 
        responses: List[protocol.Synapse], 
        reference_4k_path: str
    ) -> None:
        """
        Scores the miners by comparing their responses with the reference 4k video.
        
        Args:
            uids: List of miner UIDs that are being scored.
            responses: List of Synapse responses from the miners.
            reference_4k_path: Path to the reference 4k video for comparison.
        
        This method sends the responses to the score service, retrieves the scores, 
        and updates the miner manager with the new scores.
        """
        logger.info(f"Starting scoring for {len(uids)} miners")
        distorted_urls = [response.miner_response.optimized_video_url for response in responses]
        logger.info(f"distorted_urls: {distorted_urls}")
        
        score_response = await self.score_client.post(
            "/score",
            json={
                "distorted_urls": distorted_urls,
                "reference_path": reference_4k_path,
            },
            timeout=60,
        )
        
        response_json = score_response.json()  # Get the full JSON response
        scores: List[float] = response_json.get("scores", [])  # Extract only the list
        logger.info(f"Scores: {scores}")
        logger.info(f"Updating miner manager with {len(scores)} scores")
        self.miner_manager.step(scores, uids)

    def set_weights(self) -> None:
        """
        Sets the weights for the network if the current block exceeds the last update
        plus the configured tempo. It retrieves the processed weights and UIDs, 
        then submits the weight-setting operation to a separate thread.

        This method logs the weight information and handles the submission of weights 
        to the subtensor. If the weights are set successfully, it logs the success 
        message; otherwise, it logs an error.
        """
        self.current_block = self.subtensor.get_current_block()
        self.last_update = self.metagraph.last_update[self.uid]
        uids, weights = self.miner_manager.weights

        processed_weight_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
            uids=uids,
            weights=weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        uint_uids, uint_weights = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )

        if self.current_block > self.last_update + CONFIG.SUBNET_TEMPO:
            weight_info = list(zip(uint_uids, uint_weights))
            weight_info_df = pd.DataFrame(weight_info, columns=["uid", "weight"])
            logger.info(f"Weight info:\n{weight_info_df.to_markdown()}")
            logger.info("Actually trying to set weights.")
            
            try:
                future = self.set_weights_executor.submit(
                    self.subtensor.set_weights,
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=uint_uids,
                    weights=uint_weights,
                )
                success, msg = future.result(timeout=120)
                if not success:
                    logger.error(f"Failed to set weights: {msg}")
                else:
                    logger.debug("Set weights successfully 😎")
            except Exception as e:
                logger.error(f"Failed to set weights: {e}")
                traceback.print_exc()

            logger.info(f"Set weights result: {success}")
        else:
            logger.info(
                f"Not setting weights because current block {self.current_block} is not greater than last update {self.last_update} + tempo {CONFIG.SUBNET_TEMPO}"
            )


if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
