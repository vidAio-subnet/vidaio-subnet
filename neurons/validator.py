import bittensor as bt
import httpx
import asyncio
import traceback
import pandas as pd
import time
import random
import uuid
from loguru import logger
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from vidaio_subnet_core import validating, CONFIG, base, protocol
from vidaio_subnet_core.utilities.wandb_manager import WandbManager
from vidaio_subnet_core.utilities.uids import get_organic_forward_uids
from vidaio_subnet_core.utilities.version import get_version
from vidaio_subnet_core.protocol import LengthCheckProtocol
from services.dashboard.server import send_data_to_dashboard
from services.video_scheduler.video_utils import get_trim_video_path
from services.video_scheduler.redis_utils import get_redis_connection, get_organic_queue_size, set_scheduler_ready
import os

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
        
        self.scheduler_ready_endpoint = f"http://{CONFIG.video_scheduler.host}:{CONFIG.video_scheduler.port}/api/scheduler_ready"

    async def check_scheduler_ready(self) -> bool:
        """
        Check if the video scheduler is ready to process synthetic requests.
        
        Returns:
            bool: True if scheduler is ready, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.scheduler_ready_endpoint, timeout=10)
                response.raise_for_status()
                data = response.json()
                return data.get("ready", False)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error checking scheduler readiness: {e.response.status_code}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Request error checking scheduler readiness: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking scheduler readiness: {e}")
            return False

    async def wait_for_scheduler_ready(self) -> None:
        """
        Wait for the scheduler to be ready before proceeding with synthetic requests.
        """
        logger.info("🔄 Checking if video scheduler is ready...")
        
        while not await self.check_scheduler_ready():
            logger.info("⏳ Waiting for scheduler server to be ready (all synthetic queues need to be populated)...")
            await asyncio.sleep(10)  # Wait 10 seconds before checking again
        
        logger.info("✅ Scheduler is ready! Proceeding with synthetic requests.")

    async def start_synthetic_epoch(self):
        # Wait for scheduler to be ready first
        await self.wait_for_scheduler_ready()
        
        logger.info("✅✅✅✅✅ Starting synthetic forward ✅✅✅✅✅")
        epoch_start_time = time.time()

        miner_uids = self.filter_miners()
        logger.debug(f"Initialized {len(miner_uids)} subnet neurons of total {len(self.metagraph.S)} neurons")

        uids = self.miner_manager.consume(miner_uids)
        logger.info(f"Filtered UIDs after consumption: {uids}")

        random_uids = uids.copy()
        random.shuffle(random_uids)
        logger.info(f"Randomized UIDs: {random_uids}")

        axons = [self.metagraph.axons[uid] for uid in random_uids]

        logger.info(f"Sending LengthCheck requests to {len(axons)} miners")

        version = get_version()
        synapse = LengthCheckProtocol(
            version=version
        )
        responses = await self.dendrite.forward(
            axons=axons, synapse=synapse, timeout=10
        )
        logger.info(f"💊 Received {len(responses)} responses from miners for LengthCheck requests💊")

        content_lengths = []

        for response in responses:
            avail_max_len = response.max_content_length.value
            if avail_max_len is not None:
                content_lengths.append(avail_max_len)
            else:
                content_lengths.append(5)

        logger.info(f"Passed content lengths: {content_lengths}")

        miners = list(zip(axons, random_uids, content_lengths))

        batch_size = CONFIG.bandwidth.requests_per_synthetic_interval

        miner_batches = [
            miners[i : i + batch_size] for i in range(0, len(miners), batch_size)
        ]
        logger.info(f"Created {len(miner_batches)} batches of size {batch_size}")

        for batch_idx, batch in enumerate(miner_batches):

            batch_start_time = time.time()
            logger.info(f"🧩 Processing batch {batch_idx + 1}/{len(miner_batches)} 🧩")
            
            uids = []
            axons = []
            content_lengths = []
            for miner in batch:
                content_lengths.append(miner[2])
                uids.append(miner[1])
                axons.append(miner[0])
            
            round_id = str(uuid.uuid4())
            payload_urls, video_ids, uploaded_object_names, synapses = await self.challenge_synthesizer.build_synthetic_protocol(content_lengths, version, round_id)
            logger.debug(f"Built challenge protocol")

            timestamp = datetime.now(timezone.utc).isoformat()

            logger.debug(f"Processing UIDs in batch: {uids}")
            forward_tasks = [
                self.dendrite.forward(axons=[axon], synapse=synapse, timeout=35)
                for axon, synapse in zip(axons, synapses)
            ]

            raw_responses = await asyncio.gather(*forward_tasks)

            responses = [response[0] for response in raw_responses]
            logger.info(f"🎲 Received {len(responses)} responses from miners 🎲")

            reference_video_paths = []
            for video_id in video_ids:
                reference_video_path = get_trim_video_path(video_id)
                
                # Verify the reference video file exists before scoring
                if not os.path.exists(reference_video_path):
                    logger.warning(f"⚠️ Reference video file missing for video_id {video_id}: {reference_video_path}")
                    # You could either skip this video or create a placeholder
                    # For now, we'll still add it but log the warning
                else:
                    logger.debug(f"✅ Reference video file exists for video_id {video_id}: {reference_video_path}")
                    
                reference_video_paths.append(reference_video_path)
            
            asyncio.create_task(self.score_synthetics(uids, responses, payload_urls, reference_video_paths, timestamp, video_ids, uploaded_object_names, content_lengths, round_id))

            batch_processed_time = time.time() - batch_start_time
            sleep_time = 250 - batch_processed_time
            
            logger.info(f"Completed one batch within {batch_processed_time:.2f} seconds. Waiting {sleep_time:.2f} seconds before next batch")

            await asyncio.sleep(sleep_time)
        
        epoch_processed_time = time.time() - epoch_start_time
        logger.info(f"Completed one epoch within {epoch_processed_time:.2f} seconds")

    async def start_organic_loop(self):
        try:
            is_true = await self.should_process_organic()
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error during process organic requests: {e}")

    async def score_synthetics(
        self, 
        uids: list[int], 
        responses: list[protocol.Synapse], 
        payload_urls: list[str], 
        reference_video_paths: list[str], 
        timestamp: str, 
        video_ids: list[str], 
        uploaded_object_names: list[str], 
        content_lengths: list[int], 
        round_id: str
    ):
        distorted_urls = []
        for uid, response in zip(uids, responses):
            distorted_urls.append(response.miner_response.optimized_video_url)

        score_response = await self.score_client.post(
            "/score_synthetics",
            json = {
                "uids": uids,
                "distorted_urls": distorted_urls,
                "reference_paths": reference_video_paths,
                "video_ids": video_ids,
                "uploaded_object_names": uploaded_object_names,
                "content_lengths": content_lengths
            },
            timeout=240
        )

        response_data = score_response.json()

        quality_scores = response_data.get("quality_scores", [])
        length_scores = response_data.get("length_scores", [])
        final_scores = response_data.get("final_scores", [])
        vmaf_scores = response_data.get("vmaf_scores", [])
        pieapp_scores = response_data.get("pieapp_scores", [])
        reasons = response_data.get("reasons", [])
        
        logger.info(f"Updating miner manager with {len(quality_scores)} miner scores after synthetic requests processing")

        miner_hotkeys = [self.metagraph.hotkeys[uid] for uid in uids]
        
        accumulate_scores, applied_multipliers = self.miner_manager.step_synthetics(
            round_id, uids, miner_hotkeys, vmaf_scores, pieapp_scores, quality_scores, length_scores, final_scores, content_lengths
        )

        max_length = max(
            len(uids),
            len(quality_scores),
            len(length_scores),
            len(final_scores),
            len(vmaf_scores),
            len(pieapp_scores),
            len(reasons),
            len(content_lengths),
            len(applied_multipliers),
            len(accumulate_scores),
        )

        vmaf_scores.extend([0.0] * (max_length - len(vmaf_scores)))
        pieapp_scores.extend([0.0] * (max_length - len(pieapp_scores)))
        quality_scores.extend([0.0] * (max_length - len(quality_scores)))
        length_scores.extend([0.0] * (max_length - len(length_scores)))
        final_scores.extend([0.0] * (max_length - len(final_scores)))
        reasons.extend(["No reason provided"] * (max_length - len(reasons)))
        content_lengths.extend([0.0] * (max_length - len(content_lengths)))
        applied_multipliers.extend([0.0] * (max_length - len(applied_multipliers)))

        logger.info(f"Synthetic scoring results for {len(uids)} miners")
        logger.info(f"Uids: {uids}")

        for uid, vmaf_score, pieapp_score, quality_score, length_score, final_score, reason, content_length, applied_multiplier in zip(
            uids, vmaf_scores, pieapp_scores, quality_scores, length_scores, final_scores, reasons, content_lengths, applied_multipliers
        ):
            logger.info(
                f"{uid} ** VMAF: {vmaf_score:.2f} ** PieAPP: {pieapp_score:.2f} ** Quality: {quality_score:.4f} "
                f"** Length: {length_score:.4f} ** Content Length: {content_length} ** Applied_multiplier {applied_multiplier} ** Final: {final_score:.4f} || {reason}"
            )

        miner_data = {
            "validator_uid": self.my_subnet_uid,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "request_type": "Synthetic",
            "miner_uids": uids,
            "miner_hotkeys": miner_hotkeys,
            "vmaf_scores": vmaf_scores,
            "pieapp_scores": pieapp_scores,
            "quality_scores": quality_scores,
            "length_scores": length_scores,
            "final_scores": final_scores,
            "accumulate_scores": accumulate_scores,
            "applied_multipliers": applied_multipliers,
            "status": reasons,
            "task_urls": payload_urls,
            "processed_urls": distorted_urls,
            "timestamp": timestamp
        }
        
        success = send_data_to_dashboard(miner_data)
        if success:
            logger.info("Data successfully sent to dashboard")
        else:
            logger.info("Failed to send data to dashboard")

    async def score_organics(self, uids: list[int], responses: list[protocol.Synapse], reference_urls: list[str], task_types: list[str], timestamp: str):

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
        scores = response_data.get("final_scores", [])
        vmaf_scores = response_data.get("vmaf_scores", [])
        pieapp_scores = response_data.get("pieapp_scores", [])
        reasons = response_data.get("reasons", [])

        max_length = max(len(uids), len(scores), len(vmaf_scores), len(pieapp_scores), len(reasons))
        scores.extend([0.0] * (max_length - len(scores)))
        vmaf_scores.extend([0.0] * (max_length - len(vmaf_scores)))
        pieapp_scores.extend([0.0] * (max_length - len(pieapp_scores)))
        reasons.extend(["no reason provided"] * (max_length - len(reasons)))

        logger.info(f"organic scoring results for {len(uids)} miners")
        logger.info(f"uids: {uids}")
        for uid, vmaf_score, pieapp_score, score, reason in zip(uids, vmaf_scores, pieapp_scores, scores, reasons):
            logger.info(f"{uid} ** {vmaf_score:.2f} ** {pieapp_score:.2f} ** {score:.4f} || {reason}")

        # Generate round_id for organic scoring
        round_id = f"organic_{int(time.time())}"
        
        logger.info(f"updating miner manager with {len(scores)} miner scores after organic requests processing…")
        accumulate_scores, applied_multipliers = self.miner_manager.step_organics(scores, uids, round_id)

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
            "applied_multipliers": applied_multipliers,
            "status": reasons,
            "task_urls": reference_urls,
            "processed_urls": distorted_urls,
            "timestamp": timestamp
        }
        
        success = send_data_to_dashboard(miner_data)
        if success:
            logger.info("Data successfully sent to dashboard")
        else:
            logger.info("Failed to send data to dashboard")


    def filter_miners(self):
        min_stake = CONFIG.bandwidth.min_stake
        stake_array = self.metagraph.S
        miner_uids = [i for i, stake in enumerate(stake_array) if stake < min_stake]

        return miner_uids

    async def should_process_organic(self):

        num_organic_chunks = get_organic_queue_size(self.redis_conn)

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
            logger.info(f"There are just {len(forward_uids)} miners available for organic, handling {len(forward_uids)} chunks")
            needed = len(forward_uids)

        axon_list = [self.metagraph.axons[uid] for uid in forward_uids]

        task_ids, original_urls, task_types, synapses = await self.challenge_synthesizer.build_organic_protocol(needed)

        if len(task_ids) != needed or len(synapses) != needed:
            logger.error(
                f"Mismatch in organic synapses after building organic protocol: {len(task_ids)} != {needed} or {len(synapses)} != {needed}"
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

        asyncio.create_task(self.score_organics(forward_uids.tolist(), responses, original_urls, task_types, timestamp))

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
            logger.info("Trying to set weights.")
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
        self.validator = validator

    async def run(self):
        while True:
            try:
                logger.info("Running weight_manager...")
                self.validator.set_weights()  
            except Exception as e:
                logger.error(f"Error in WeightSynthesizer: {e}", exc_info=True)
            await asyncio.sleep(1200)  


if __name__ == "__main__":
    validator = Validator()
    weight_synthesizer = WeightSynthesizer(validator)
    # time.sleep(1000) # wait till the video scheduler is ready, # should be removed in production

    set_scheduler_ready(validator.redis_conn, False)
    logger.info("Set scheduler readiness flag to False")

    async def main():
        validator_synthetic_task = asyncio.create_task(validator.run_synthetic())
        validator_organic_task = asyncio.create_task(validator.run_organic())
        weight_setter = asyncio.create_task(weight_synthesizer.run())

        await asyncio.gather(validator_synthetic_task, validator_organic_task, weight_setter)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)


