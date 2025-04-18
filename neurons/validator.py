from vidaio_subnet_core import validating, CONFIG, base, protocol
import bittensor as bt
import httpx
import asyncio
from loguru import logger
import traceback
import pandas as pd
from typing import List
import requests
from concurrent.futures import ThreadPoolExecutor
from vidaio_subnet_core.utilities.minio_client import minio_client
from vidaio_subnet_core.utilities.wandb_manager import WandbManager
from vidaio_subnet_core.utilities.uids import get_organic_forward_uids
from vidaio_subnet_core.utilities.version import get_version
from services.video_scheduler.video_utils import get_trim_video_path, delete_videos_with_fileid
from services.video_scheduler.redis_utils import get_redis_connection, get_organic_queue_size
import time

class Validator(base.BaseValidator):
    def __init__(self):
        super().__init__()
        self.miner_manager = validating.managing.MinerManager(
            uid=self.uid, wallet=self.wallet, metagraph=self.metagraph
        )
        logger.info("💧 Initialized miner manager 💧")
        
        self.challenge_synthesizer = validating.synthesizing.Synthesizer()
        logger.info("💧 Initialized challenge synthesizer 💧")
        
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info("💧 Initialized dendrite 💧")
        
        self.score_client = httpx.AsyncClient(
            base_url=f"http://{CONFIG.score.host}:{CONFIG.score.port}"
        )
        logger.info(
            f"💧 Initialized score client with base URL: http://{CONFIG.score.host}:{CONFIG.score.port} 💧"
        )
        
        self.set_weights_executor = ThreadPoolExecutor(max_workers=1)
        logger.info("💙 Initialized setting weights executor 💙")

        self.redis_conn = get_redis_connection()
        logger.info("💙 Initialized Redis connection 💙")
        
        self.wandb_manager = WandbManager(validator=self)
        logger.info("🔑 Initialized Wandb Manager 🔑")


        self.organic_gateway_base_url = f"http://localhost:{CONFIG.organic_gateway.port}"

        self.push_result_endpoint = f"http://{CONFIG.video_scheduler.host}:{CONFIG.video_scheduler.port}/api/push_result"


    async def start_epoch(self):
        logger.info("✅✅✅✅✅ Starting forward ✅✅✅✅✅")
        epoch_start_time = time.time()

        miner_uids = self.filter_miners()
        logger.debug(f"Initialized {len(miner_uids)} subnet neurons of total {len(self.metagraph.S)} neurons")

        is_true = await self.should_process_organic()

        uids = self.miner_manager.consume(miner_uids)
        logger.info(f"Filtered UIDs after consumption: {uids}")

        axons = [self.metagraph.axons[uid] for uid in uids]
        miners = list(zip(axons, uids))

        batch_size = CONFIG.bandwidth.requests_per_interval

        miner_batches = [
            miners[i : i + batch_size] for i in range(0, len(miners), batch_size)
        ]
        logger.info(f"Created {len(miner_batches)} batches of size {batch_size}")

        for batch_idx, batch in enumerate(miner_batches):

            is_true = True
            while is_true == True:
                is_true = await self.should_process_organic()

            batch_start_time = time.time()
            logger.info(f"🧩 Processing batch {batch_idx + 1}/{len(miner_batches)} 🧩")
            video_id, uploaded_object_name, synapse = await self.challenge_synthesizer.build_synthetic_protocol()
            synapse.version = get_version()
            logger.debug(f"Built challenge protocol")
            uids = []
            axons = []
            for miner in batch:
                uids.append(miner[1])
                axons.append(miner[0])
            logger.debug(f"Processing UIDs in batch: {uids}")
            responses = await self.dendrite.forward(
                axons=axons, synapse=synapse, timeout=60
            )
            logger.info(f"🎲 Received {len(responses)} responses from miners 🎲")

            reference_video_path = get_trim_video_path(video_id)
            
            await self.score_synthetic(uids, responses, reference_video_path)

            minio_client.delete_file(uploaded_object_name)
            delete_videos_with_fileid(video_id)
            batch_processed_time = time.time() - batch_start_time
            logger.info(f"Completed one batch within {batch_processed_time:.2f} seconds. Waiting 5 seconds before next batch")
            await asyncio.sleep(5)
        
        epoch_processed_time = time.time() - epoch_start_time
        logger.info(f"Completed one epoch within {epoch_processed_time:.2f} seconds")

    async def score_synthetic(self, uids: list[int], responses: list[protocol.Synapse], reference_video_path: str):
        logger.info(f"Starting scoring for {len(uids)} miners")
        logger.info(f"Uids: {uids}")
        distorted_urls = []
        for uid, response in zip(uids, responses):
            distorted_urls.append(response.miner_response.optimized_video_url)

        score_response = await self.score_client.post(
            "/score",
            json = {
                "distorted_urls": distorted_urls,
                "reference_path": reference_video_path
            },
            timeout=1500
        )
        response_json = score_response.json()
        scores: List[float] = response_json.get("scores", [])  
        logger.info(f"Scores: {scores}")
        logger.info(f"Updating miner manager with {len(scores)} miner scores")
        self.miner_manager.step(scores, uids)

    def filter_miners(self):
        min_stake = CONFIG.bandwidth.min_stake
        stake_array = self.metagraph.S
        miner_uids = [i for i, stake in enumerate(stake_array) if stake < min_stake]

        return miner_uids

    async def should_process_organic(self):

        num_organic_chunks = get_organic_queue_size(self.redis_conn)

        logger.info(f"🥒 Checking if an organic query exists 🥒")

        if num_organic_chunks > 0:
            logger.info(f"The organic_queue_size: {num_organic_chunks}, processing organic requests.")
            await self.process_organic_chunks(num_organic_chunks)
            return True
        else:
            logger.info("The organic queue is currently empty, so it will be skipped")
            return False

    async def process_organic_chunks(self, num_organic_chunks):
        organic_start_time = time.time() 

        needed = min(CONFIG.bandwidth.requests_organic_interval, num_organic_chunks)
        
        logger.info(f"🍉 Start processing organic query. need {needed} miners 🍉")

        forward_uids = get_organic_forward_uids(self, needed, CONFIG.bandwidth.min_stake)

        if len(forward_uids) < needed:
            logger.info(f"There are just {len(forward_uids)} miners available, handling {len(forward_uids)} chunks")
            needed = len(forward_uids)

        axon_list = [self.metagraph.axons[uid] for uid in forward_uids]

        logger.info("Building the organic protocol")
        task_ids, original_urls, synapses = await self.challenge_synthesizer.build_organic_protocol(needed)

        if len(task_ids) != needed or len(synapses) != needed:
            logger.error(
                f"Mismatch in organic synapses: {len(task_ids)} != {needed} or {len(synapses)} != {needed}"
            )
            return  # Exit early if there's a mismatch

        logger.info("Updating task status to 'processing'")
        for task_id, original_url in zip(task_ids, original_urls):
            await self.update_task_status(task_id, original_url, "processing")

        logger.info("Performing forward operations asynchronously")
        forward_tasks = [
            self.dendrite.forward(axons=[axon], synapse=synapse, timeout=200)
            for axon, synapse in zip(axon_list, synapses)
        ]

        raw_responses = await asyncio.gather(*forward_tasks)

        responses = [response[0] for response in raw_responses]
        processed_urls = [response.miner_response.optimized_video_url for response in responses]

        logger.info("Updating task status to 'completed' and pushing results")
        for task_id, original_url, processed_url in zip(task_ids, original_urls, processed_urls):
            await self.update_task_status(task_id, original_url, "completed")
            await self.push_result(task_id, original_url, processed_url)

        end_time = time.time()
        total_time = end_time - organic_start_time
        logger.info(f"🍏 Organic chunk processing complete in {total_time:.2f} seconds 🍏")

    async def update_task_status(self, task_id, original_url, status):
        status_update_endpoint = f"{self.organic_gateway_base_url}/admin/task/{task_id}/status"
        status_update_payload = {
            "status": status,
            "original_video_url": original_url
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(status_update_endpoint, json=status_update_payload)
                response.raise_for_status()
                logger.info(f"Successfully updated status to '{status}' for task {task_id}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Error updating status for task {task_id}: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error updating status for task {task_id}: {e}")

    async def push_result(self, task_id, original_url, processed_url):
        result_payload = {
            "processed_video_url": processed_url,
            "original_video_url": original_url,
            "score": 0,
            "task_id": task_id,
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.push_result_endpoint, json=result_payload)
                response.raise_for_status()
                logger.info(f"Successfully pushed result for task {task_id}")
        except httpx.RequestError as e:
            logger.error(f"Error pushing result for task {task_id}: {e}")

    def set_weights(self):
        self.current_block = self.subtensor.get_current_block()
        self.last_update = self.metagraph.last_update[self.uid]
        uids, weights = self.miner_manager.weights
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            # uids=self.metagraph.uids,
            uids = uids,
            weights=weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
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
                    logger.error(f"😠 Failed to set weights: {msg}")
                else: 
                    logger.debug("😎 Set weights successfully ")
            except Exception as e:
                logger.error(f"Failed to set weights: {e}")
                traceback.print_exc()

        else:
            logger.info(
                f"Not setting weights because current block {self.current_block} is not greater than last update {self.last_update} + tempo {CONFIG.SUBNET_TEMPO}"
            )


if __name__ == "__main__":
    validator = Validator()
    time.sleep(200) # wait till the video scheduler is ready
    asyncio.run(validator.run())



