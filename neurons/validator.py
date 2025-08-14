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
from vidaio_subnet_core.protocol import LengthCheckProtocol, TaskWarrantProtocol, VideoCompressionProtocol, TaskType
from services.dashboard.server import send_data_to_dashboard
from services.video_scheduler.video_utils import get_trim_video_path
from services.video_scheduler.redis_utils import get_redis_connection, get_organic_queue_size, set_scheduler_ready
import os

VMAF_QUALITY_THRESHOLDS = [
    85, #Low
    90, #Medium
    95, #High
]

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

        # Step 1: Send TaskWarrantProtocol requests to determine miner capabilities
        logger.info(f"Sending TaskWarrantProtocol requests to {len(axons)} miners")

        version = get_version()
        task_warrant_synapse = TaskWarrantProtocol(version=version)
        
        task_warrant_responses = await self.dendrite.forward(
            axons=axons, synapse=task_warrant_synapse, timeout=10
        )
        logger.info(f"💊 Received {len(task_warrant_responses)} responses from miners for TaskWarrantProtocol requests💊")

        # Step 2: Group miners based on their task warrant responses
        upscaling_miners = []
        compression_miners = []
        unknown_task_miners = []
        
        for i, response in enumerate(task_warrant_responses):
            uid = random_uids[i]
            axon = axons[i]
            
            if response.warrant_task == TaskType.UPSCALING:
                upscaling_miners.append((axon, uid))
            elif response.warrant_task == TaskType.COMPRESSION:
                compression_miners.append((axon, uid))
            else:
                logger.warning(f"UID {uid} has unknown task warrant: {response.warrant_task}, sending Upscaling Task")
                upscaling_miners.append((axon, uid))
                unknown_task_miners.append(uid)
        
        # Log grouped results in organized lists
        upscaling_uids = [uid for _, uid in upscaling_miners]
        compression_uids = [uid for _, uid in compression_miners]
        
        logger.info(f"🛜 Grouped miners: {len(upscaling_miners)} upscaling, {len(compression_miners)} compression 🛜")
        if upscaling_uids:
            logger.info(f"📈 Upscaling UIDs: {upscaling_uids}")
        if compression_uids:
            logger.info(f"📉 Compression UIDs: {compression_uids}")
        if unknown_task_miners:
            logger.info(f"❓ Unknown task UIDs (defaulted to upscaling): {unknown_task_miners}")

        # Step 3: Send LengthCheckProtocol requests to upscaling miners
        if upscaling_miners:
            logger.info(f"Sending LengthCheckProtocol requests to {len(upscaling_miners)} upscaling miners")
            
            upscaling_start_time = time.time()
            upscaling_axons = [miner[0] for miner in upscaling_miners]
            length_check_synapse = LengthCheckProtocol(version=version)
            
            length_check_responses = await self.dendrite.forward(
                axons=upscaling_axons, synapse=length_check_synapse, timeout=10
            )
            logger.info(f"💊 Received {len(length_check_responses)} responses from upscaling miners for LengthCheckProtocol requests💊")

            # Extract content lengths for upscaling miners
            upscaling_content_lengths = []
            for response in length_check_responses:
                avail_max_len = response.max_content_length.value
                if avail_max_len is not None:
                    upscaling_content_lengths.append(avail_max_len)
                else:
                    upscaling_content_lengths.append(5)

            logger.info(f"Upscaling content lengths: {upscaling_content_lengths}")

            # Create upscaling miner batches with content lengths
            upscaling_miners_with_lengths = []
            for i, (axon, uid) in enumerate(upscaling_miners):
                content_length = upscaling_content_lengths[i] if i < len(upscaling_content_lengths) else 5
                upscaling_miners_with_lengths.append((axon, uid, content_length))

            # Process upscaling miners in batches
            await self.process_upscaling_miners(upscaling_miners_with_lengths, version)

            upscaling_processed_time = time.time() - upscaling_start_time

            logger.info(f"Upscaling tasks processed in {upscaling_processed_time:.2f} seconds")

            await asyncio.sleep(2)

        # Step 4: Process compression miners (placeholder for now)
        if compression_miners:
            logger.info(f"Processing {len(compression_miners)} compression miners")

            compression_start_time = time.time()
            await self.process_compression_miners(compression_miners, version)

            compression_processed_time = time.time() - compression_start_time

            logger.info(f"Compression tasks processed in {compression_processed_time:.2f} seconds")

            await asyncio.sleep(2)

        epoch_processed_time = time.time() - epoch_start_time
        logger.info(f"Completed one epoch within {epoch_processed_time:.2f} seconds")

        await asyncio.sleep(2)

    async def process_upscaling_miners(self, upscaling_miners_with_lengths, version):
        """Process upscaling miners in batches similar to the original implementation."""
        batch_size = CONFIG.bandwidth.requests_per_synthetic_interval

        miner_batches = [
            upscaling_miners_with_lengths[i : i + batch_size] 
            for i in range(0, len(upscaling_miners_with_lengths), batch_size)
        ]
        logger.info(f"Created {len(miner_batches)} upscaling batches of size {batch_size}")

        for batch_idx, batch in enumerate(miner_batches):
            batch_start_time = time.time()
            logger.info(f"🧩 Processing upscaling batch {batch_idx + 1}/{len(miner_batches)} 🧩")
            
            uids = []
            axons = []
            content_lengths = []
            for miner in batch:
                content_lengths.append(miner[2])
                uids.append(miner[1])
                axons.append(miner[0])
            
            round_id = str(uuid.uuid4())
            payload_urls, video_ids, uploaded_object_names, synapses = await self.challenge_synthesizer.build_synthetic_protocol(content_lengths, version, round_id)
            logger.debug(f"Built upscaling challenge protocol")

            timestamp = datetime.now(timezone.utc).isoformat()

            logger.debug(f"Processing upscaling UIDs in batch: {uids}")
            forward_tasks = [
                self.dendrite.forward(axons=[axon], synapse=synapse, timeout=35)
                for axon, synapse in zip(axons, synapses)
            ]

            raw_responses = await asyncio.gather(*forward_tasks)

            responses = [response[0] for response in raw_responses]
            logger.info(f"🎲 Received {len(responses)} upscaling responses from miners 🎲")

            reference_video_paths = []
            for video_id in video_ids:
                reference_video_path = get_trim_video_path(video_id)
                if not os.path.exists(reference_video_path):
                    logger.warning(f"⚠️ Reference video file missing for video_id {video_id}: {reference_video_path}")
                reference_video_paths.append(reference_video_path)
            
            asyncio.create_task(self.score_upscalings(uids, responses, payload_urls, reference_video_paths, timestamp, video_ids, uploaded_object_names, content_lengths, round_id))

            batch_processed_time = time.time() - batch_start_time
            
            sleep_time = 300 - batch_processed_time
            logger.info(f"Completed upscaling batch within {batch_processed_time:.2f} seconds")
            logger.info(f"Sleeping for {sleep_time:.2f} seconds before next upscaling batch")
            
            await asyncio.sleep(sleep_time)

    async def process_compression_miners(self, compression_miners, version):
        """Process compression miners in batches similar to upscaling but with compression protocols."""
        batch_size = CONFIG.bandwidth.requests_per_synthetic_interval

        miner_batches = [
            compression_miners[i : i + batch_size] 
            for i in range(0, len(compression_miners), batch_size)
        ]
        logger.info(f"Created {len(miner_batches)} compression batches of size {batch_size}")

        for batch_idx, batch in enumerate(miner_batches):
            batch_start_time = time.time()
            logger.info(f"🧩 Processing compression batch {batch_idx + 1}/{len(miner_batches)} 🧩")
            
            uids = []
            axons = []
            for miner in batch:
                uids.append(miner[1])
                axons.append(miner[0])
            
            vmaf_threshold = random.choice(VMAF_QUALITY_THRESHOLDS)
            
            round_id = str(uuid.uuid4())

            num_miners = len(uids)

            payload_urls, video_ids, uploaded_object_names, synapses = await self.challenge_synthesizer.build_compression_protocol(vmaf_threshold, num_miners, version, round_id)
            logger.debug(f"Built compression challenge protocol")

            timestamp = datetime.now(timezone.utc).isoformat()

            logger.debug(f"Processing compression UIDs in batch: {uids}")
            forward_tasks = [
                self.dendrite.forward(axons=[axon], synapse=synapse, timeout=40)
                for axon, synapse in zip(axons, synapses)
            ]

            raw_responses = await asyncio.gather(*forward_tasks)

            responses = [response[0] for response in raw_responses]
            logger.info(f"🎲 Received {len(responses)} compression responses from miners 🎲")

            reference_video_paths = []
            for video_id in video_ids:
                reference_video_path = get_trim_video_path(video_id)
                if not os.path.exists(reference_video_path):
                    logger.warning(f"⚠️ Reference video file missing for video_id {video_id}: {reference_video_path}")
                reference_video_paths.append(reference_video_path)
            
            asyncio.create_task(self.score_compressions(uids, responses, payload_urls, reference_video_paths, timestamp, video_ids, uploaded_object_names, vmaf_threshold, round_id))

            batch_processed_time = time.time() - batch_start_time
            sleep_time = 300 - batch_processed_time

            logger.info(f"Completed compression batch within {batch_processed_time:.2f} seconds")
            logger.info(f"Sleeping for {sleep_time:.2f} seconds before next compression batch")
            
            await asyncio.sleep(sleep_time)


    async def start_organic_loop(self):
        try:
            is_true = await self.should_process_organic()
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error during process organic requests: {e}")

    async def score_upscalings(
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

        logger.info("responses: ", responses)

        score_response = await self.score_client.post(
            "/score_upscaling_synthetics",
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
        
        accumulate_scores, applied_multipliers = self.miner_manager.step_synthetics_upscaling(
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
            "task_type": "Upscaling",
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

    async def score_compressions(
        self, 
        uids: list[int], 
        responses: list[protocol.Synapse], 
        payload_urls: list[str], 
        reference_video_paths: list[str], 
        timestamp: str, 
        video_ids: list[str], 
        uploaded_object_names: list[str], 
        vmaf_threshold: float, 
        round_id: str
    ):
        distorted_urls = []
        for uid, response in zip(uids, responses):
            distorted_urls.append(response.miner_response.optimized_video_url)

        score_response = await self.score_client.post(
            "/score_compression_synthetics",  # Using the same endpoint for now
            json = {
                "uids": uids,
                "distorted_urls": distorted_urls,
                "reference_paths": reference_video_paths,
                "video_ids": video_ids,
                "uploaded_object_names": uploaded_object_names,
                "vmaf_threshold": vmaf_threshold
            },
            timeout=240
        )

        response_data = score_response.json()

        compression_rates = response_data.get("compression_rates", [])
        final_scores = response_data.get("final_scores", [])
        vmaf_scores = response_data.get("vmaf_scores", [])
        reasons = response_data.get("reasons", [])
        
        logger.info(f"Updating miner manager with {len(compression_rates)} compression miner scores after synthetic requests processing")

        miner_hotkeys = [self.metagraph.hotkeys[uid] for uid in uids]
        
        # Use the compression rates from the scoring response
        if not compression_rates:
            compression_rates = [0.5] * len(uids)  # Fallback to default if not provided
        
        # Use the updated step_synthetics_compression method with compression context
        accumulate_scores, applied_multipliers = self.miner_manager.step_synthetics_compression(
            round_id, uids, miner_hotkeys, vmaf_scores,
            final_scores, [10] * len(uids), vmaf_threshold, compression_rates
        )

        max_length = max(
            len(uids),
            len(final_scores),
            len(vmaf_scores),
            len(reasons),
            len(applied_multipliers),
            len(accumulate_scores),
        )

        vmaf_scores.extend([0.0] * (max_length - len(vmaf_scores)))
        final_scores.extend([0.0] * (max_length - len(final_scores)))
        reasons.extend(["No reason provided"] * (max_length - len(reasons)))
        compression_rates.extend([0.5] * (max_length - len(compression_rates)))
        applied_multipliers.extend([0.0] * (max_length - len(applied_multipliers)))

        logger.info(f"Compression scoring results for {len(uids)} miners")
        logger.info(f"Uids: {uids}")

        for uid, vmaf_score, final_score, reason, compression_rate, applied_multiplier in zip(
            uids, vmaf_scores, final_scores, reasons, compression_rates, applied_multipliers
        ):
            logger.info(
                f"{uid} ** VMAF: {vmaf_score:.2f} "
                f"** VMAF Threshold: {vmaf_threshold} ** Compression Rate: {compression_rate:.4f} ** Applied_multiplier {applied_multiplier} ** Final: {final_score:.4f} || {reason}"
            )

        miner_data = {
            "validator_uid": self.my_subnet_uid,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "request_type": "Synthetic",
            "task_type": "Compression",
            "miner_uids": uids,
            "miner_hotkeys": miner_hotkeys,
            "vmaf_scores": vmaf_scores,
            "vmaf_threshold": vmaf_threshold,
            "compression_rates": compression_rates,
            "final_scores": final_scores,
            "accumulate_scores": accumulate_scores,
            "applied_multipliers": applied_multipliers,
            "status": reasons,
            "task_urls": payload_urls,
            "processed_urls": distorted_urls,
            "timestamp": timestamp
        }
        
        # success = send_data_to_dashboard(miner_data)
        # if success:
        #     logger.info("Compression data successfully sent to dashboard")
        # else:
        #     logger.info("Failed to send compression data to dashboard")

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
            self.dendrite.forward(axons=[axon], synapse=synapse, timeout=35)
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


