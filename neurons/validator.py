import bittensor as bt
import httpx
import asyncio
import traceback
import pandas as pd
import time
import random
from loguru import logger
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from vidaio_subnet_core import validating, CONFIG, base, protocol
from vidaio_subnet_core.utilities.wandb_manager import WandbManager
from vidaio_subnet_core.utilities.uids import get_organic_forward_uids
from vidaio_subnet_core.utilities.version import get_version
from services.dashboard.server import send_data_to_dashboard
from services.video_scheduler.video_utils import get_trim_video_path, delete_videos_with_fileid
from services.video_scheduler.redis_utils import get_redis_connection, get_organic_queue_size


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

    async def start_synthetic_epoch(self):
        logger.info("✅✅✅✅✅ Starting synthetic forward ✅✅✅✅✅")
        epoch_start_time = time.time()

        miner_uids = self.filter_miners()
        logger.debug(f"Initialized {len(miner_uids)} subnet neurons of total {len(self.metagraph.S)} neurons")

        is_true = await self.should_process_organic()

        uids = self.miner_manager.consume(miner_uids)
        logger.info(f"Filtered UIDs after consumption: {uids}")

        random_uids = uids.copy()
        random.shuffle(random_uids)
        logger.info(f"Randomized UIDs: {random_uids}")

        axons = [self.metagraph.axons[uid] for uid in random_uids]
        miners = list(zip(axons, random_uids))

        batch_size = CONFIG.bandwidth.requests_per_synthetic_interval

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
            payload_url, video_id, uploaded_object_name, synapse = await self.challenge_synthesizer.build_synthetic_protocol()
            synapse.version = get_version()
            logger.debug(f"Built challenge protocol")
            uids = []
            axons = []
            for miner in batch:
                uids.append(miner[1])
                axons.append(miner[0])

            timestamp = datetime.now(timezone.utc).isoformat()

            logger.debug(f"Processing UIDs in batch: {uids}")
            responses = await self.dendrite.forward(
                axons=axons, synapse=synapse, timeout=60
            )
            logger.info(f"🎲 Received {len(responses)} responses from miners 🎲")

            reference_video_path = get_trim_video_path(video_id)
            
            asyncio.create_task(self.score_synthetics(uids, responses, payload_url, reference_video_path, timestamp, video_id, uploaded_object_name,))

            # await self.score_synthetics(uids, responses, payload_url, reference_video_path, timestamp)


            batch_processed_time = time.time() - batch_start_time
            logger.info(f"Completed one batch within {batch_processed_time:.2f} seconds. Waiting 5 seconds before next batch")
            await asyncio.sleep(5)
        
        epoch_processed_time = time.time() - epoch_start_time
        logger.info(f"Completed one epoch within {epoch_processed_time:.2f} seconds")

    async def start_organic_loop(self):

        try:
            is_true = await self.should_process_organic()
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error during process organic requests: {e}")

    async def score_synthetics(self, uids: list[int], responses: list[protocol.Synapse], payload_url: str, reference_video_path: str, timestamp: str, video_id: str, uploaded_object_name: str):

        distorted_urls = []
        for uid, response in zip(uids, responses):
            distorted_urls.append(response.miner_response.optimized_video_url)

        score_response = await self.score_client.post(
            "/score_synthetics",
            json = {
                "uids": uids,
                "distorted_urls": distorted_urls,
                "reference_path": reference_video_path,
                "video_id": video_id,
                "uploaded_object_name": uploaded_object_name
            },
            timeout=1500
        )

        response_data = score_response.json()

        scores = response_data.get("scores", [])
        vmaf_scores = response_data.get("vmaf_scores", [])
        pieapp_scores = response_data.get("pieapp_scores", [])
        reasons = response_data.get("reasons", [])
        
        max_length = max(len(uids), len(scores), len(vmaf_scores), len(pieapp_scores), len(reasons))
        scores.extend([0.0] * (max_length - len(scores)))
        vmaf_scores.extend([0.0] * (max_length - len(vmaf_scores)))
        pieapp_scores.extend([0.0] * (max_length - len(pieapp_scores)))
        reasons.extend(["No reason provided"] * (max_length - len(reasons)))


        logger.info(f"Starting synthetic scoring for {len(uids)} miners")
        logger.info(f"Uids: {uids}")
        for uid, vmaf_score, pieapp_score, score, reason in zip(uids, vmaf_scores, pieapp_scores, scores, reasons):
            logger.info(f"{uid} ** {vmaf_score:.2f} ** {pieapp_score:.2f} ** {score:.4f} || {reason}")
        
        logger.info(f"Updating miner manager with {len(scores)} miner scores")
        
        accumulate_scores = self.miner_manager.step_synthetics(scores, uids)
        miner_hotkeys = [self.metagraph.hotkeys[uid] for uid in uids]
        payload_urls = [payload_url] * len(uids)

        miner_data = {
            "validator_uid": self.my_subnet_uid,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "request_type": "Synthetic",
            "miner_uids": uids,
            "miner_hotkeys": miner_hotkeys,
            "vmaf_scores": vmaf_scores,
            "pieapp_scores": pieapp_scores,
            "final_scores": scores,
            "accumulate_scores": accumulate_scores,
            "status": reasons,
            "task_urls": payload_urls,
            "processed_urls": distorted_urls,
            "timestamp": timestamp
        }
        
        # success = send_data_to_dashboard(miner_data)

    async def score_organics(self, uids: list[int], responses: list[protocol.Synapse], reference_urls: list[str], task_types: list[str], timestamp: str):
        logger.info(f"starting organic scoring for {len(uids)} miners")
        logger.info(f"uids: {uids}")

        distorted_urls = [response.miner_response.optimized_video_url for response in responses]

        # zip and shuffle all lists together to preserve alignment
        combined = list(zip(uids, distorted_urls, reference_urls, task_types))
        random.shuffle(combined)
        uids, distorted_urls, reference_urls, task_types = map(list, zip(*combined))

        score_response = await self.score_client.post(
            "/score_organics",
            json={
                "uids": uids,
                "distorted_urls": distorted_urls,
                "reference_urls": reference_urls,
                "task_types": task_types
            },
            timeout=1500
        )

        response_data = score_response.json()
        scores = response_data.get("scores", [])
        vmaf_scores = response_data.get("vmaf_scores", [])
        pieapp_scores = response_data.get("pieapp_scores", [])
        reasons = response_data.get("reasons", [])

        max_length = max(len(uids), len(scores), len(vmaf_scores), len(pieapp_scores), len(reasons))
        scores.extend([0.0] * (max_length - len(scores)))
        vmaf_scores.extend([0.0] * (max_length - len(vmaf_scores)))
        pieapp_scores.extend([0.0] * (max_length - len(pieapp_scores)))
        reasons.extend(["no reason provided"] * (max_length - len(reasons)))

        for uid, vmaf_score, pieapp_score, score, reason in zip(uids, vmaf_scores, pieapp_scores, scores, reasons):
            logger.info(f"{uid} ** {vmaf_score:.2f} ** {pieapp_score:.2f} ** {score:.4f} || {reason}")

        logger.info(f"updating miner manager with {len(scores)} miner scores")
        accumulate_scores = self.miner_manager.step_organics(scores, uids)

        miner_hotkeys = [self.metagraph.hotkeys[uid] for uid in uids]

        miner_data = {
            "validator_uid": self.my_subnet_uid,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "request_type": "Organic",
            "miner_uids": uids,
            "miner_hotkeys": miner_hotkeys,
            "vmaf_scores": vmaf_scores,
            "pieapp_scores": pieapp_scores,
            "final_scores": scores,
            "accumulate_scores": accumulate_scores,
            "status": reasons,
            "task_urls": reference_urls,
            "processed_urls": distorted_urls,
            "timestamp": timestamp
        }
        
        # success = send_data_to_dashboard(miner_data)


    def filter_miners(self):
        min_stake = CONFIG.bandwidth.min_stake
        stake_array = self.metagraph.S
        miner_uids = [i for i, stake in enumerate(stake_array) if stake < min_stake]

        return miner_uids

    async def should_process_organic(self):

        num_organic_chunks = get_organic_queue_size(self.redis_conn)

        # logger.info(f"🥒 Checking if an organic query exists 🥒")

        if num_organic_chunks > 0:
            logger.info(f"🔷🔷🔷🔷 The organic_queue_size: {num_organic_chunks}, processing organic requests. 🔷🔷🔷🔷")
            await self.process_organic_chunks(num_organic_chunks)
            return True
        else:
            # logger.info("The organic queue is currently empty, so it will be skipped")
            return False

    async def process_organic_chunks(self, num_organic_chunks):
        organic_start_time = time.time() 

        needed = min(CONFIG.bandwidth.requests_per_organic_interval, num_organic_chunks)
        
        logger.info(f"🍉 Start processing organic query. need {needed} miners 🍉")

        forward_uids = get_organic_forward_uids(self, needed, CONFIG.bandwidth.min_stake)

        if len(forward_uids) < needed:
            logger.info(f"There are just {len(forward_uids)} miners available, handling {len(forward_uids)} chunks")
            needed = len(forward_uids)

        axon_list = [self.metagraph.axons[uid] for uid in forward_uids]

        logger.info("Building the organic protocol")
        task_ids, original_urls, task_types, synapses = await self.challenge_synthesizer.build_organic_protocol(needed)

        if len(task_ids) != needed or len(synapses) != needed:
            logger.error(
                f"Mismatch in organic synapses: {len(task_ids)} != {needed} or {len(synapses)} != {needed}"
            )
            return  # Exit early if there's a mismatch

        logger.info("Updating task status to 'processing'")
        for task_id, original_url in zip(task_ids, original_urls):
            await self.update_task_status(task_id, original_url, "processing")

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("💡 Performing forward operations asynchronously 💡")
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

        await self.score_organics(forward_uids.tolist(), responses, original_urls, task_types, timestamp)

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
                # logger.info(f"Successfully updated status to '{status}' for task {task_id}")
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


class WeightSynthesizer:
    def __init__(self, validator: Validator):
        self.weight_manager = validator.set_weights()

    async def run(self):
        while True:
            try:
                logger.info("Running weight_manager...")
                self.weight_manager()  
            except Exception as e:
                logger.error(f"Error in WeightSynthesizer: {e}", exc_info=True)
            await asyncio.sleep(1200)  


if __name__ == "__main__":
    validator = Validator()
    weight_synthesizer = WeightSynthesizer(validator)
    time.sleep(200) # wait till the video scheduler is ready

    async def main():
        # Create separate tasks
        validator_synthetic_task = asyncio.create_task(validator.run_synthetic())
        validator_organic_task = asyncio.create_task(validator.run_organic())
        scheduler_task = asyncio.create_task(weight_synthesizer.run())

        # Wait for both tasks to complete (runs indefinitely in this case)
        await asyncio.gather(validator_task, scheduler_task)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)


