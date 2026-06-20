import os
import time
import uuid
import httpx
import random
import asyncio
import traceback
import pandas as pd
import bittensor as bt
import tempfile
import aiohttp
from loguru import logger
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from vidaio_subnet_core.utilities.version import get_version
from vidaio_subnet_core import validating, CONFIG, base, protocol
from vidaio_subnet_core.utilities.wandb_manager import WandbManager
from services.video_scheduler.video_utils import get_trim_video_path, get_perumted_video_path
from vidaio_subnet_core.utilities.uids import get_organic_forward_uids
from vidaio_subnet_core.protocol import (
    LengthCheckProtocol,
    TaskWarrantProtocol,
    TaskType,
    VideoCompressionJobProtocol,
    VideoCompressionPollProtocol,
    VideoUpscalingJobProtocol,
    VideoUpscalingPollProtocol,
    MinerResponse,
)
from vidaio_subnet_core.validating.managing.sql_schemas import MinerMetadata, MinerPerformanceHistory, Base
from sqlalchemy import desc, asc, func, select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from services.dashboard.server import send_upscaling_data_to_dashboard, send_compression_data_to_dashboard
from services.video_scheduler.redis_utils import (
    get_redis_connection, 
    get_organic_upscaling_queue_size, 
    get_organic_compression_queue_size, 
    set_scheduler_ready
)
from typing import List, Tuple, Any
from enum import IntEnum


class VMAF_QUALITY_THRESHOLD(IntEnum):
    LOW = 85
    MEDIUM = 89
    HIGH = 93


TARGET_CODECS = [
    "av1",        # AV1 codec (protocol standard name)
    "hevc",       # H.265/HEVC (protocol standard name) 
    # "h264",       # H.264/AVC (protocol standard name)
    # "vp9",        # VP9 (protocol standard name)
]

# Codec encoding modes
CODEC_MODES = [
    "CRF",        # Constant Rate Factor (quality-based)
    "VBR",        # Variable Bitrate
]

# Target bitrates in Mbps for CBR/VBR modes
TARGET_BITRATES = [
    5.0,         # 5 Mbps
    8.0,         # 8 Mbps
    10.0,         # 10 Mbps
]


SLEEP_TIME_LOW = 60 * 5 # 5 minutes
SLEEP_TIME_HIGH = 60 * 6 # 6 minutes

try:
    SYNTHETIC_QUERIES_PER_MINER = max(1, int(os.getenv("SYNTHETIC_QUERIES_PER_MINER", "5")))
except ValueError:
    logger.warning("Invalid SYNTHETIC_QUERIES_PER_MINER value, defaulting to 5")
    SYNTHETIC_QUERIES_PER_MINER = 5

UPSCALING_SCORING_BATCH_SIZE = 5

try:
    ORGANIC_QUERIES_PER_MINER = max(
        1, int(os.getenv("ORGANIC_QUERIES_PER_MINER", str(SYNTHETIC_QUERIES_PER_MINER)))
    )
except ValueError:
    logger.warning("Invalid ORGANIC_QUERIES_PER_MINER value, defaulting to SYNTHETIC_QUERIES_PER_MINER")
    ORGANIC_QUERIES_PER_MINER = SYNTHETIC_QUERIES_PER_MINER

class Validator(base.BaseValidator):
    def __init__(self):
        super().__init__()
        self.miner_manager = validating.managing.MinerManager(
            uid=self.uid, config=self.config, wallet=self.wallet, metagraph=self.metagraph
        )
        logger.info("💧 Initialized miner manager 💧")
        
        self.challenge_synthesizer = validating.synthesizing.Synthesizer()
        logger.info("💧 Initialized challenge synthesizer 💧")
        
        self.dendrite = bt.Dendrite(wallet=self.wallet)
        logger.info("💧 Initialized dendrite 💧")
        
        self.score_client_upscaling = httpx.AsyncClient(
            base_url=f"http://{CONFIG.score.host}:{CONFIG.score.upscaling_score_port}"
        )
        logger.info(
            f"💧 Initialized upscaling score client with base URL: http://{CONFIG.score.host}:{CONFIG.score.upscaling_score_port} 💧"
        )
        
        self.score_client_compression = httpx.AsyncClient(
            base_url=f"http://{CONFIG.score.host}:{CONFIG.score.compression_score_port}"
        )
        logger.info(
            f"💧 Initialized compression score client with base URL: http://{CONFIG.score.host}:{CONFIG.score.compression_score_port} 💧"
        )

        self.score_client_upscaling_organics = httpx.AsyncClient(
            base_url=f"http://{CONFIG.score.host}:{CONFIG.score.upscaling_organics_score_port}"
        )
        logger.info(
            f"💧 Initialized organic upscaling score client with base URL: http://{CONFIG.score.host}:{CONFIG.score.upscaling_organics_score_port} 💧"
        )
        
        self.score_client_compression_organics = httpx.AsyncClient(
            base_url=f"http://{CONFIG.score.host}:{CONFIG.score.compression_organics_score_port}"
        )
        logger.info(
            f"💧 Initialized organic compression score client with base URL: http://{CONFIG.score.host}:{CONFIG.score.compression_organics_score_port} 💧"
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
            await asyncio.sleep(10)
        
        logger.info("✅ Scheduler is ready! Proceeding with synthetic requests.")

    async def download_video(self, video_url: str, verbose: bool = False) -> tuple[str, float]:
        """
        Download a video from the given URL and save it to a temporary file.

        Args:
            video_url (str): The URL of the video to download.
            verbose (bool): Whether to show download progress.

        Returns:
            tuple[str, float]: A tuple containing the path to the downloaded video file
                            and the time taken to download it.

        Raises:
            Exception: If the download fails or takes longer than the timeout.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid_temp:
                file_path = vid_temp.name  # Path to the temporary file
            if verbose:
                logger.info(f"Downloading video from {video_url} to {file_path}")

            timeout = aiohttp.ClientTimeout(sock_connect=30, total=300)

            start_time = time.time()  # Record start time

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(video_url, allow_redirects=True) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download video. HTTP status: {response.status}")

                    with open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(2 * 1024 * 1024):
                            f.write(chunk)

            end_time = time.time()  # Record end time
            download_time = end_time - start_time  # Calculate download duration

            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise Exception(f"Download failed or file is empty: {file_path}")

            if verbose:
                logger.info(f"File successfully downloaded to: {file_path}")
                logger.info(f"Download time: {download_time:.2f} seconds")

            return file_path, download_time
        except aiohttp.ServerTimeoutError:
            raise Exception("Download failed: Server connection timed out")
        except asyncio.TimeoutError:
            raise Exception("Download timed out")
        except aiohttp.ClientError as e:
            raise Exception(f"Download failed due to a network error: {type(e).__name__}: {repr(e)}")
    
    async def refresh_miner_manager(self, miner_uids: list[int]):
        if not miner_uids:
            return

        miner_hotkeys = [self.metagraph.hotkeys[uid] for uid in miner_uids]

        # Single query to fetch all miners
        stmt = select(MinerMetadata).where(MinerMetadata.uid.in_(miner_uids))
        result = self.miner_manager.session.execute(stmt)
        miners = {miner.uid: miner for miner in result.scalars().all()}

        delete_uids = []

        for uid, latest_hotkey in zip(miner_uids, miner_hotkeys):
            miner = miners.get(uid)
            if miner is None:
                continue

            if miner.hotkey != latest_hotkey:
                delete_uids.append(uid)
                logger.info(
                    f"UID {uid} hotkey changed from {miner.hotkey} → {latest_hotkey}"
                )
                miner.hotkey = latest_hotkey  # update record in-memory

        if delete_uids:
            delete_stmt = delete(MinerPerformanceHistory).where(
                MinerPerformanceHistory.uid.in_(delete_uids)
            )
            self.miner_manager.session.execute(delete_stmt)

            delete_stmt = delete(MinerMetadata).where(MinerMetadata.uid.in_(delete_uids))
            self.miner_manager.session.execute(delete_stmt)

        self.miner_manager.session.commit()
        self.miner_manager.sync_miner_chain_metadata(miner_uids)

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

        logger.info(f"Sending TaskWarrantProtocol requests to {len(axons)} miners")

        version = get_version()
        task_warrant_synapse = TaskWarrantProtocol(version=version)
        
        task_warrant_responses = await self.dendrite.forward(
            axons=axons, synapse=task_warrant_synapse, timeout=60
        )
        logger.info(f"💊 Received {len(task_warrant_responses)} responses from miners for TaskWarrantProtocol requests💊")

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
                unknown_task_miners.append(uid)
        
        if unknown_task_miners:
            logger.info(f"🔍 Checking database for {len(unknown_task_miners)} unknown task warrant miners")
            db_task_types = self.miner_manager.get_miner_processing_task_types(unknown_task_miners)
            
            resolved_from_db = []
            defaulted_to_upscaling = []
            
            for i, uid in enumerate(unknown_task_miners):
                axon = axons[random_uids.index(uid)]
                
                if uid in db_task_types:
                    db_task_type = db_task_types[uid]
                    if db_task_type == "upscaling":
                        upscaling_miners.append((axon, uid))
                        resolved_from_db.append(f"{uid}(upscaling)")
                    elif db_task_type == "compression":
                        compression_miners.append((axon, uid))
                        resolved_from_db.append(f"{uid}(compression)")
                    else:
                        upscaling_miners.append((axon, uid))
                        defaulted_to_upscaling.append(f"{uid}(unknown_db_value:{db_task_type})")
                else:
                    upscaling_miners.append((axon, uid))
                    defaulted_to_upscaling.append(f"{uid}(no_db_record)")
            
        upscaling_uids = [uid for _, uid in upscaling_miners]
        compression_uids = [uid for _, uid in compression_miners]
        all_uids = upscaling_uids + compression_uids
        
        await self.refresh_miner_manager(all_uids)
        
        logger.info(f"🛜 Grouped miners: {len(upscaling_miners)} upscaling, {len(compression_miners)} compression 🛜")
        if upscaling_uids:
            logger.info(f"📈 Upscaling UIDs: {upscaling_uids}")
        if compression_uids:
            logger.info(f"📉 Compression UIDs: {compression_uids}")
        if unknown_task_miners:
            logger.info(f"❓ Unknown task UIDs processed: {unknown_task_miners}")
        
        disable_synthetics = os.getenv("DISABLE_SYNTHETICS", "False").lower() == "true"
        if disable_synthetics:
            logger.info("Synthetic epoch disabled by DISABLE_SYNTHETICS env var, only ran TaskWarrant requests to gauge miner task types necessary to execute organics, sleeping for 30 minutes before refreshing")
            await asyncio.sleep(1800)
            return

        logger.info("Sleeping for 3 minutes before starting epoch")
        await asyncio.sleep(180)

        # ---- Run upscaling & compression epochs in parallel ---- #
        async def _run_upscaling():
            if not upscaling_miners:
                return
            
            logger.info("🧩 Sleeping for 3 minutes before starting upscaling epoch")
            await asyncio.sleep(180)
            
            logger.info(f"Sending LengthCheckProtocol requests to {len(upscaling_miners)} upscaling miners")

            upscaling_start_time = time.time()
            upscaling_axons = [miner[0] for miner in upscaling_miners]
            length_check_synapse = LengthCheckProtocol(version=version)

            length_check_responses = await self.dendrite.forward(
                axons=upscaling_axons, synapse=length_check_synapse, timeout=10
            )
            logger.info(f"💊 Received {len(length_check_responses)} responses from upscaling miners for LengthCheckProtocol requests💊")

            upscaling_content_lengths = []
            for response in length_check_responses:
                avail_max_len = response.max_content_length.value
                if avail_max_len == 5:
                    upscaling_content_lengths.append(avail_max_len)
                else:
                    upscaling_content_lengths.append(10)

            logger.info(f"Upscaling content lengths: {upscaling_content_lengths}")

            upscaling_miners_with_lengths = []
            for i, (axon, uid) in enumerate(upscaling_miners):
                content_length = upscaling_content_lengths[i] if i < len(upscaling_content_lengths) else 10
                upscaling_miners_with_lengths.append((axon, uid, content_length))

            logger.info(f"Sleeping for 3 minutes before querying upscaling miners")
            await asyncio.sleep(180)

            await self.process_upscaling_miners(upscaling_miners_with_lengths, version)

            upscaling_processed_time = time.time() - upscaling_start_time
            logger.info(f"Upscaling tasks processed in {upscaling_processed_time:.2f} seconds")

        async def _run_compression():
            if not compression_miners:
                return
            logger.info(f"Processing {len(compression_miners)} compression miners")

            compression_start_time = time.time()
            await self.process_compression_miners(compression_miners, version)

            compression_processed_time = time.time() - compression_start_time
            logger.info(f"Compression tasks processed in {compression_processed_time:.2f} seconds")

        results = await asyncio.gather(
            _run_upscaling(), _run_compression(), return_exceptions=True
        )
        for label, result in zip(["upscaling", "compression"], results):
            if isinstance(result, BaseException):
                logger.error(f"Parallel {label} epoch failed: {result}", exc_info=result)

        epoch_processed_time = time.time() - epoch_start_time
        logger.info(f"Completed one epoch within {epoch_processed_time:.2f} seconds")

        logger.info(f"Sleeping for 3-5 hours before starting next cycle")
        await asyncio.sleep(random.randint(3600 * 3, 3600 * 5)) 

    #   --------------------------------------------------------------------------- #
    #  Helper – turn the raw list of (axon, uid) into a dict {uid: (axon, uid)}
    # --------------------------------------------------------------------------- #
    def _miner_lookup(self, miners: List[Tuple[Any, int]]) -> dict[int, Tuple[Any, int]]:
        """{uid: (axon, uid)} – fast lookup while preserving the original tuple."""
        return {uid: (axon, uid) for axon, uid in miners}


    async def create_miner_batches(
        self,
        miners: List[Tuple[Any, int]],
        batch_size: int,
        task_type: str = "upscaling",
    ) -> List[List[Tuple[int, Tuple[Any, int]]]]:
        """
        Build batches of miners for *task_type* (upscaling / compression).

        Rules
        -----
        1. Miners with **fewest** performance-history rows go first.
        2. Miners whose **most recent entry** is older than 2 h are eligible, subject to 70% probability of selection in batching.
        3. Miners that already have >= CONFIG.score.max_performance_records rows
        **and** whose latest entry is < 2 h are **skipped**.
        """
        session = self.miner_manager.session
        hours_threshold = datetime.utcnow() - timedelta(hours=CONFIG.score.synthetics_hours_threshold)
        max_records = CONFIG.score.max_performance_records

        info_new_uids = []

        # ------------------------------------------------------------------- #
        # 1. One-shot aggregated stats per UID (count + latest timestamp)
        # ------------------------------------------------------------------- #
        uid_stats_subq = (
            session.query(
                MinerPerformanceHistory.uid,
                func.count(MinerPerformanceHistory.id).label("record_count"),
                func.max(MinerPerformanceHistory.timestamp).label("latest_timestamp"),
            )
            .filter(
                MinerPerformanceHistory.processed_task_type == task_type,
                MinerPerformanceHistory.uid.in_([uid for _axon, uid in miners]),
            )
            .group_by(MinerPerformanceHistory.uid)
            .subquery()
        )

        uid_stats = session.query(
            uid_stats_subq.c.uid,
            uid_stats_subq.c.record_count,
            uid_stats_subq.c.latest_timestamp,
        ).all()

        # uid → (record_count, latest_timestamp)
        uid_info: dict[int, Tuple[int, datetime | None]] = {
            row.uid: (row.record_count or 0, row.latest_timestamp) for row in uid_stats
        }

        # ------------------------------------------------------------------- #
        # 2. Miners that have *zero* rows are the highest priority
        # ------------------------------------------------------------------- #
        all_uids = {uid for _axon, uid in miners}
        for uid in all_uids - uid_info.keys():
            uid_info[uid] = (0, None)

        # ------------------------------------------------------------------- #
        # 3. Build a fast lookup {uid: original_tuple}
        # ------------------------------------------------------------------- #
        miner_by_uid = self._miner_lookup(miners)

        # ------------------------------------------------------------------- #
        # 4. Filter + priority key
        # ------------------------------------------------------------------- #
        eligible: List[Tuple[Tuple[int, datetime], Tuple[Any, int]]] = []

        for uid in all_uids:
            miner_tuple = miner_by_uid[uid]
            rec_cnt, latest_ts = uid_info[uid]

            # ---- Rule 3: skip capped miners with recent performance record ----
            if rec_cnt >= max_records and latest_ts and latest_ts >= hours_threshold:
                logger.info(
                    f"Skipping miner uid={uid}: {rec_cnt} records, last update {latest_ts}"
                )
                continue

            # ---- Rule 2: include if never seen, below capacity or stale ----
            if rec_cnt < max_records or not latest_ts \
                or (latest_ts < hours_threshold and random.random() < CONFIG.score.synthetics_select_probability):
                priority = (
                    rec_cnt,                                   # fewer records first
                    latest_ts if latest_ts else datetime.min,  # oldest first
                )
                if rec_cnt == 0:
                    info_new_uids.append(uid) 
                eligible.append((priority, miner_tuple))

        # ------------------------------------------------------------------- #
        # 5. Sort & batch
        # ------------------------------------------------------------------- #
        eligible.sort(key=lambda x: x[0])                     # by priority tuple
        sorted_miners = [(prio[0], miner) for prio, miner in eligible]

        batches = [
            sorted_miners[i : i + batch_size]
            for i in range(0, len(sorted_miners), batch_size)
        ]

        logger.info(
            f"Created {len(batches)} {task_type} batches (size≤{batch_size}). "
            f"Selected {len(sorted_miners)}/{len(miners)} miners."
            f" New UIDs with no history: {info_new_uids}"
        )
        if eligible:
            top_uid = eligible[0][1][1]   # (axon, uid) → uid
            logger.info(
                f"Top-priority miner uid={top_uid} – records={uid_info[top_uid][0]}"
            )

        return batches

    async def call_miner_batch(self, axons, synapse, batch, timeout=60):
        start = time.perf_counter()

        try:
            raw = await self.dendrite.forward(
                axons=axons,
                synapse=synapse,
                timeout=timeout,
            )
            duration = (time.perf_counter() - start) * 1000  # ms

            logger.info(f"💊 Received {len(raw)} responses from miners in {duration} ms💊")
            return raw
        except Exception as e:
            logger.error(f"Unexpected error calling miner batch: {e}", exc_info=True)
            return []

    async def call_miner(self, axon, synapse, uid, timeout=60):
        start = time.perf_counter()

        try:
            raw = await self.dendrite.forward(
                axons=[axon],
                synapse=synapse,
                timeout=timeout,
            )
            duration = (time.perf_counter() - start) * 1000  # ms

            try:
                result = raw[0]
            except Exception as e:
                logger.error(
                    f"UID {uid} → invalid response structure after {duration:.2f} ms | {e}",
                    exc_info=True
                )
                return {
                    "uid": uid,
                    "result": None,
                    "error": e,
                    "latency_ms": duration,
                }

            logger.info(f"UID {uid} → success | {duration:.2f} ms")

            return {
                "uid": uid,
                "result": result,
                "error": None,
                "latency_ms": duration,
            }

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000  # ms

            logger.error(
                f"UID {uid} → failure after {duration:.2f} ms | {type(e).__name__}: {e}",
                exc_info=True
            )

            return {
                "uid": uid,
                "result": None,  # EXACT same shape as original gather usage
                "error": e,
                "latency_ms": duration,
            }

    def _response_video_urls(self, response, expected_count: int) -> list[str]:
        miner_response = getattr(response, "miner_response", None)
        if miner_response is None:
            return [""] * expected_count

        urls = list(getattr(miner_response, "optimized_video_urls", []) or [])
        legacy_url = getattr(miner_response, "optimized_video_url", "")
        if not urls and legacy_url:
            urls = [legacy_url]

        if len(urls) < expected_count:
            urls.extend([""] * (expected_count - len(urls)))

        return urls[:expected_count]
    
    async def call_miner_polling(
        self,
        axon,
        synapse_job,
        uid: int,
        PollProtocolClass,
        poll_interval: float = 60.0,
        max_polls: int = 20,
    ):
        """Polling-based miner call for organic requests.

        Phase 1: Sends the job kick-off synapse and receives a ``job_id``
        immediately (30s timeout — just long enough for a network round-trip).
        Phase 2: Polls at ``poll_interval`` second intervals until the miner
        reports ``completed`` or ``failed``, or ``max_polls`` is exhausted
        (default: 20 * 60s = 20-minute budget).

        Args:
            axon: Target miner axon.
            synapse_job: A ``VideoCompressionJobProtocol`` or
                ``VideoUpscalingJobProtocol`` instance with the task payload.
            uid: Miner UID (used for logging).
            PollProtocolClass: ``VideoCompressionPollProtocol`` or
                ``VideoUpscalingPollProtocol``.
            poll_interval: Seconds between polls (default 20).
            max_polls: Maximum poll attempts (default 60 → 20-min budget).

        Returns:
            Dict with keys ``uid``, ``result`` (synapse with populated
            ``miner_response.optimized_video_url``), ``error``, ``latency_ms``.
        """
        start = time.perf_counter()

        def _empty(error_msg):
            return {
                "uid": uid,
                "result": synapse_job,
                "error": error_msg,
                "latency_ms": (time.perf_counter() - start) * 1000,
            }

        # ---- Phase 1: kick-off ----
        # The validator owns the job_id — assign it before sending.
        job_id = str(uuid.uuid4())
        synapse_job.job_id = job_id
        logger.info(f"UID {uid} → sending job kick-off | job_id={job_id} ")
        try:
            raw = await self.dendrite.forward(
                axons=[axon], synapse=synapse_job, timeout=60
            )
            accepted = raw[0].job_response.accepted
        except Exception as e:
            logger.error(f"UID {uid} → polling kick-off failed: {e}")
            return _empty(str(e))

        if not accepted:
            logger.warning(
                f"UID {uid} → miner did not accept the job (job_id={job_id!r})"
            )
            return _empty("not accepted")

        logger.info(f"UID {uid} → job kick-off accepted | job_id={job_id}")

        # ---- Phase 2: poll loop ----
        for poll_num in range(1, max_polls + 1):
            await asyncio.sleep(poll_interval)
            try:
                poll_syn = PollProtocolClass(job_id=job_id)
                raw = await self.dendrite.forward(
                    axons=[axon], synapse=poll_syn, timeout=15
                )
                resp = raw[0].poll_response
            except Exception as e:
                logger.warning(f"UID {uid} | poll #{poll_num} error: {e}")
                continue

            logger.info(f"UID {uid} | poll #{poll_num} → status={resp.status}")

            if resp.status == "completed":
                # Shim result so all downstream code can access
                # .miner_response.optimized_video_url unchanged.
                response_urls = list(getattr(resp, "optimized_video_urls", []) or [])
                legacy_url = getattr(resp, "optimized_video_url", "")
                if not response_urls and legacy_url:
                    response_urls = [legacy_url]
                synapse_job.miner_response = MinerResponse(
                    optimized_video_urls=response_urls
                )
                return {
                    "uid": uid,
                    "result": synapse_job,
                    "error": None,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
            elif resp.status == "failed":
                logger.error(f"UID {uid} → job {job_id} reported failed")
                return _empty("miner reported failed")
            # status == "processing" → keep polling

        logger.error(
            f"UID {uid} → polling timed out after {max_polls} attempts for job {job_id}"
        )
        return _empty("polling timeout")

    async def process_upscaling_miners(self, upscaling_miners_with_lengths, version):
        """Process all upscaling miners concurrently with the same bundled challenge."""
        upscaling_miners = [(axon, uid) for axon, uid, _ in upscaling_miners_with_lengths]
        upscaling_content_lengths = {uid: length for _, uid, length in upscaling_miners_with_lengths}

        num_miners = len(upscaling_miners)
        if num_miners == 0:
            return

        query_count = SYNTHETIC_QUERIES_PER_MINER
        shared_content_length = min(upscaling_content_lengths.values()) if upscaling_content_lengths else 5
        content_lengths = [shared_content_length] * query_count
        uids = [uid for _, uid in upscaling_miners]
        axons = [axon for axon, _ in upscaling_miners]

        logger.info(
            f"🧩 Processing {num_miners} upscaling miners concurrently with "
            f"{query_count} shared synthetic queries 🧩"
        )
        batch_start_time = time.time()
        round_id = str(uuid.uuid4())

        payload_urls, video_ids, uploaded_object_names, synapses, task_types = await self.challenge_synthesizer.build_synthetic_protocol(
            content_lengths, version, round_id, bundle_protocols=True
        )
        query_count = len(payload_urls)
        logger.info(f"Built upscaling challenge protocol: payload URLs: {payload_urls}\nvideo IDs: {video_ids}")

        timestamp = datetime.now(timezone.utc).isoformat()
        responses = await self.call_miner_batch(
            axons,
            synapses[0],
            num_miners,
            timeout=max(90, 90 * query_count),
        )

        logger.info(f"🎲 Received {len(responses)} upscaling responses from miners 🎲")

        reference_video_paths = []
        for video_id in video_ids:
            reference_video_path = get_trim_video_path(video_id)
            if not os.path.exists(reference_video_path):
                logger.warning(f"⚠️ Reference video file missing for video_id {video_id}: {reference_video_path}")
            reference_video_paths.append(reference_video_path)

        flat_uids = []
        flat_distorted_urls = []
        flat_payload_urls = []
        flat_reference_paths = []
        flat_video_ids = []
        flat_uploaded_object_names = []
        flat_content_lengths = []
        flat_task_types = []

        for response_idx, uid in enumerate(uids):
            response = responses[response_idx] if response_idx < len(responses) else None
            response_urls = self._response_video_urls(response, query_count)
            for query_idx, distorted_url in enumerate(response_urls):
                flat_uids.append(uid)
                flat_distorted_urls.append(distorted_url)
                flat_payload_urls.append(payload_urls[query_idx])
                flat_reference_paths.append(reference_video_paths[query_idx])
                flat_video_ids.append(video_ids[query_idx])
                flat_uploaded_object_names.append(uploaded_object_names[query_idx])
                flat_content_lengths.append(content_lengths[query_idx])
                flat_task_types.append(task_types[query_idx])

        total_scoring_responses = len(flat_uids)
        scoring_batch_size = UPSCALING_SCORING_BATCH_SIZE
        scoring_batch_count = (total_scoring_responses + scoring_batch_size - 1) // scoring_batch_size
        logger.info(
            f"Scoring {total_scoring_responses} upscaling query responses from "
            f"{num_miners} miners in {scoring_batch_count} batches of up to {scoring_batch_size}"
        )

        for batch_idx, batch_start in enumerate(range(0, total_scoring_responses, scoring_batch_size), start=1):
            batch_end = min(batch_start + scoring_batch_size, total_scoring_responses)
            batch_uids = flat_uids[batch_start:batch_end]
            batch_timer = time.time()
            logger.info(
                f"Scoring upscaling batch {batch_idx}/{scoring_batch_count}: "
                f"responses {batch_start + 1}-{batch_end}/{total_scoring_responses}, "
                f"uids={batch_uids}"
            )

            await self.score_upscalings(
                batch_uids,
                [],
                flat_payload_urls[batch_start:batch_end],
                flat_reference_paths[batch_start:batch_end],
                timestamp,
                flat_video_ids[batch_start:batch_end],
                flat_uploaded_object_names[batch_start:batch_end],
                flat_content_lengths[batch_start:batch_end],
                flat_task_types[batch_start:batch_end],
                round_id,
                distorted_urls=flat_distorted_urls[batch_start:batch_end],
            )

            logger.info(
                f"Completed upscaling scoring batch {batch_idx}/{scoring_batch_count} "
                f"in {time.time() - batch_timer:.2f} seconds"
            )

        batch_processed_time = time.time() - batch_start_time
        logger.info(f"Completed upscaling batch within {batch_processed_time:.2f} seconds")
        logger.info(f"Scored {total_scoring_responses} upscaling query responses from {num_miners} miners")

    async def process_compression_miners(self, compression_miners, version):
        """Process all compression miners concurrently with the same bundled challenge."""
        num_miners = len(compression_miners)
        if num_miners == 0:
            return

        query_count = SYNTHETIC_QUERIES_PER_MINER
        logger.info(
            f"🧩 Processing {num_miners} compression miners concurrently with "
            f"{query_count} shared synthetic queries 🧩"
        )
        
        round_id = str(uuid.uuid4())

        uids = []
        axons = []
        for axon, uid in compression_miners:
            uids.append(uid)
            axons.append(axon)

        vmaf_threshold = random.choice(list(VMAF_QUALITY_THRESHOLD))
        vmaf_thresholds = [vmaf_threshold] * query_count
        target_codec = random.choice(TARGET_CODECS)
        codec_mode = random.choice(CODEC_MODES)
        target_bitrate = random.choice(TARGET_BITRATES)

        payload_urls, video_ids, uploaded_object_names, synapses = await self.challenge_synthesizer.build_compression_protocol(
            vmaf_thresholds,
            query_count,
            version,
            round_id,
            target_codec,
            codec_mode,
            target_bitrate,
            bundle_protocols=True,
        )
        query_count = len(payload_urls)
        logger.warning(
            f"Built compression challenge protocol for {num_miners} miners, "
            f"{query_count} queries, VMAF threshold {vmaf_thresholds[0]}, codec {target_codec}, "
            f"mode {codec_mode}, bitrate {target_bitrate} Mbps"
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.debug(f"Processing compression UIDs: {uids}")
        
        batch_start_time = time.time()
        
        responses = await self.call_miner_batch(
            axons,
            synapses[0],
            num_miners,
            timeout=max(135, 135 * query_count),
        )

        logger.info(f"🎲 Received {len(responses)} compression responses from miners 🎲")

        reference_video_paths = []
        for video_id in video_ids:
            reference_video_path = get_perumted_video_path(video_id)
            if not os.path.exists(reference_video_path):
                logger.warning(f"⚠️ Reference video file missing for video_id {video_id}: {reference_video_path}")
            reference_video_paths.append(reference_video_path)

        flat_uids = []
        flat_distorted_urls = []
        flat_payload_urls = []
        flat_reference_paths = []
        flat_video_ids = []
        flat_uploaded_object_names = []
        flat_vmaf_thresholds = []

        for response_idx, uid in enumerate(uids):
            response = responses[response_idx] if response_idx < len(responses) else None
            response_urls = self._response_video_urls(response, query_count)
            for query_idx, distorted_url in enumerate(response_urls):
                flat_uids.append(uid)
                flat_distorted_urls.append(distorted_url)
                flat_payload_urls.append(payload_urls[query_idx])
                flat_reference_paths.append(reference_video_paths[query_idx])
                flat_video_ids.append(video_ids[query_idx])
                flat_uploaded_object_names.append(uploaded_object_names[query_idx])
                flat_vmaf_thresholds.append(getattr(vmaf_thresholds[query_idx], "value", vmaf_thresholds[query_idx]))

        
        # Concurrently start all downloads that have valid URLs,
        # limited by a semaphore to prevent timeouts on large fleets
        semaphore = asyncio.Semaphore(15)

        async def _download_with_semaphore(uid_val, url_str):
            logger.info(f"UID {uid_val}: Queuing download for distorted video")
            async with semaphore:
                try:
                    logger.info(f"UID {uid_val}: Starting download for distorted video")
                    return await self.download_video(url_str)
                except Exception as e:
                    return e

        download_tasks = []
        for uid_val, url in zip(flat_uids, flat_distorted_urls):
            if not url or len(url) < 10:
                logger.error(f"UID {uid_val}: invalid or missing distorted video download URL: {url}")
                # Create a task that simply returns None
                download_tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
            else:
                download_tasks.append(
                    asyncio.create_task(_download_with_semaphore(uid_val, url))
                )
        
        # Wait for all downloads to finish
        download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Process the results, handling any exceptions
        distorted_file_paths = []
        for uid_val, url, result in zip(flat_uids, flat_distorted_urls, download_results):
            if isinstance(result, Exception):
                logger.error(f"UID {uid_val}: Failed to download distorted video from {url} - {str(result)}")
                distorted_file_paths.append(None)
            elif result is None:
                # Already logged in the loop above
                distorted_file_paths.append(None)
            elif isinstance(result, tuple) and len(result) >= 1:
                distorted_file_paths.append(result[0])
            else:
                logger.error(f"UID {uid_val}: Unexpected result from download_video for {url} - {result}")
                distorted_file_paths.append(None)


        batch_processed_time = time.time() - batch_start_time
        logger.info(f"Completed compression gathering and downloading within {batch_processed_time:.2f} seconds")

        logger.info(f"Scoring {len(flat_uids)} compression query responses from {num_miners} miners")
        await self.score_compressions(
            flat_uids,
            [],
            distorted_file_paths,
            flat_payload_urls,
            flat_reference_paths,
            timestamp,
            flat_video_ids,
            flat_uploaded_object_names,
            flat_vmaf_thresholds,
            target_codec,
            codec_mode,
            target_bitrate,
            round_id,
            distorted_urls=flat_distorted_urls,
        )

        for path in distorted_file_paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.error(f"Error deleting distorted video file {path}: {e}")

        try:
            for reference_video_path in set(reference_video_paths):
                if os.path.exists(reference_video_path):
                    os.unlink(reference_video_path)
        except Exception as e:
            logger.error(f"Error deleting reference video file: {e}")

    async def start_organic_loop(self):
        """Start organic processing loop for both upscaling and compression tasks asynchronously."""
        try:
            # Create tasks for both upscaling and compression processing
            upscaling_task = asyncio.create_task(self._process_organic_upscaling_loop())
            compression_task = asyncio.create_task(self._process_organic_compression_loop())
            
            # Wait for both tasks to complete (they run indefinitely)
            await asyncio.gather(upscaling_task, compression_task)
        except Exception as e:
            logger.error(f"Error during organic processing loop: {e}")

    async def _process_organic_upscaling_loop(self):
        """Process organic upscaling tasks in a loop."""
        while True:
            try:
                await self.should_process_organic_upscaling()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error during organic upscaling processing: {e}")
                await asyncio.sleep(5)

    async def _process_organic_compression_loop(self):
        """Process organic compression tasks in a loop."""
        while True:
            try:
                await self.should_process_organic_compression()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error during organic compression processing: {e}")
                await asyncio.sleep(5)

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
        task_types: list[str],
        round_id: str,
        distorted_urls: list[str] | None = None,
    ):
        if distorted_urls is None:
            distorted_urls = []
            for uid, response in zip(uids, responses):
                distorted_urls.append(response.miner_response.optimized_video_url)

        logger.info(f"payloads: {payload_urls}\nresponses: {distorted_urls}")

        score_response = await self.score_client_upscaling.post(
            "/score_upscaling_synthetics",
            json = {
                "uids": uids,
                "payload_urls": payload_urls,
                "distorted_urls": distorted_urls,
                "reference_paths": reference_video_paths,
                "video_ids": video_ids,
                "uploaded_object_names": uploaded_object_names,
                "content_lengths": content_lengths,
                "task_types": task_types
            },
            timeout=600
        )

        response_data = score_response.json()

        quality_scores = response_data.get("quality_scores", [])
        length_scores = response_data.get("length_scores", [])
        final_scores = response_data.get("final_scores", [])
        vmaf_scores = response_data.get("vmaf_scores", [])
        pieapp_scores = response_data.get("pieapp_scores", [])
        reasons = response_data.get("reasons", [])
        
        logger.info(f"Updating miner manager with {len(quality_scores)} upscaling scores after synthetic requests processing")

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

        logger.info(f"Synthetic upscaling scoring results for {len(uids)} query responses")
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
            "processing_task_type": "upscaling",
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
        
        success = send_upscaling_data_to_dashboard(miner_data)
        if success:
            logger.info("Data successfully sent to dashboard")
        else:
            logger.info("Failed to send data to dashboard")

    async def score_compressions(
        self,
        uids: list[int],
        responses: list[protocol.Synapse],
        distorted_file_paths: list[str],
        payload_urls: list[str],
        reference_video_paths: list[str],
        timestamp: str,
        video_ids: list[str],
        uploaded_object_names: list[str],
        vmaf_thresholds: List[float],
        target_codec: str,
        codec_mode: str,
        target_bitrate: float,
        round_id: str,
        distorted_urls: list[str] | None = None,
    ):
        if distorted_urls is None:
            distorted_urls = []
            for uid, response in zip(uids, responses):
                distorted_urls.append(response.miner_response.optimized_video_url)

        logger.info(f"payload: {payload_urls[0]}\nresponses: {distorted_urls}")

        score_response = await self.score_client_compression.post(
            "/score_compression_synthetics",
            json = {
                "uids": uids,
                "distorted_file_paths": distorted_file_paths,
                "reference_paths": reference_video_paths,
                "video_ids": video_ids,
                "uploaded_object_names": uploaded_object_names,
                "vmaf_thresholds": vmaf_thresholds,
                "distorted_urls": distorted_urls,
                "target_codec": target_codec,
                "codec_mode": codec_mode,
                "target_bitrate": target_bitrate
            },
            timeout=600
        )

        response_data = score_response.json()

        compression_rates = response_data.get("compression_rates", [])
        final_scores = response_data.get("final_scores", [])
        vmaf_scores = response_data.get("vmaf_scores", [])
        reasons = response_data.get("reasons", [])
        
        logger.info(f"Updating miner manager with {len(compression_rates)} compression miner scores after synthetic requests processing")

        miner_hotkeys = [self.metagraph.hotkeys[uid] for uid in uids]
        
        if not compression_rates:
            compression_rates = [0.5] * len(uids)
        
        accumulate_scores, applied_multipliers = self.miner_manager.step_synthetics_compression(
            round_id, uids, miner_hotkeys, vmaf_scores,
            final_scores, [10] * len(uids), vmaf_thresholds, compression_rates
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


        logger.info(f"Synthetic compression scoring results for {len(uids)} miners")
        logger.info(f"Uids: {uids}")

        for uid, vmaf_score, final_score, reason, compression_rate, applied_multiplier, vmaf_threshold in zip(
            uids, vmaf_scores, final_scores, reasons, compression_rates, applied_multipliers, vmaf_thresholds
        ):
            logger.info(
                f"{uid} ** VMAF: {vmaf_score:.2f} "
                f"** VMAF Threshold: {vmaf_threshold} ** Compression Rate: {compression_rate:.4f} ** Final: {final_score:.4f} || {reason}"
            )

        miner_data = {
            "validator_uid": self.my_subnet_uid,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "request_type": "Synthetic",
            "processing_task_type": "compression",
            "miner_uids": uids,
            "miner_hotkeys": miner_hotkeys,
            "vmaf_scores": vmaf_scores,
            "vmaf_thresholds": vmaf_thresholds,
            "compression_rates": compression_rates,
            "final_scores": final_scores,
            "accumulate_scores": accumulate_scores,
            "applied_multipliers": applied_multipliers,
            "status": reasons,
            "task_urls": payload_urls,
            "processed_urls": distorted_urls,
            "timestamp": timestamp
        }
        
        success = send_compression_data_to_dashboard(miner_data)
        if success:
            logger.info("Compression data successfully sent to dashboard")
        else:
            logger.info("Failed to send compression data to dashboard")

    async def score_organics_upscaling(
        self,
        uids: list[int],
        responses: list[protocol.Synapse],
        reference_urls: list[str],
        task_types: list[str],
        timestamp: str,
        distorted_urls: list[str] | None = None,
    ):
        """Score organic upscaling tasks."""
        if distorted_urls is None:
            distorted_urls = [response.miner_response.optimized_video_url for response in responses]

        combined = list(zip(uids, distorted_urls, reference_urls, task_types))
        if not combined:
            logger.warning("No organic upscaling responses available for scoring")
            return
        random.shuffle(combined)
        uids, distorted_urls, reference_urls, task_types = map(list, zip(*combined))

        num_pairs_to_validate = min(1, len(combined))
        selected_indices = random.sample(range(len(combined)), num_pairs_to_validate)
        
        selected_uids = [uids[i] for i in selected_indices]
        selected_distorted_urls = [distorted_urls[i] for i in selected_indices]
        selected_reference_urls = [reference_urls[i] for i in selected_indices]
        selected_task_types = [task_types[i] for i in selected_indices]

        logger.info(f"Randomly selected {len(selected_uids)} pairs out of {len(uids)} total pairs for validation")

        score_response = await self.score_client_upscaling_organics.post(
            "/score_organics_upscaling",
            json={
                "uids": selected_uids,
                "distorted_urls": selected_distorted_urls,
                "reference_urls": selected_reference_urls,
                "task_types": selected_task_types
            },
            timeout=180
        )

        response_data = score_response.json()
        scores = response_data.get("final_scores", [])
        vmaf_scores = response_data.get("vmaf_scores", [])
        pieapp_scores = response_data.get("pieapp_scores", [])
        quality_scores = response_data.get("quality_scores", [])
        length_scores = response_data.get("length_scores", [])
        reasons = response_data.get("reasons", [])

        max_length = max(len(selected_uids), len(scores), len(vmaf_scores), len(pieapp_scores), len(reasons))
        scores.extend([0.0] * (max_length - len(scores)))
        vmaf_scores.extend([0.0] * (max_length - len(vmaf_scores)))
        pieapp_scores.extend([0.0] * (max_length - len(pieapp_scores)))
        quality_scores.extend([0.0] * (max_length - len(quality_scores)))
        length_scores.extend([0.0] * (max_length - len(length_scores)))
        reasons.extend(["no reason provided"] * (max_length - len(reasons)))

        logger.info(f"organic upscaling scoring results for {len(selected_uids)} miners")
        logger.info(f"uids: {selected_uids}")
        for uid, vmaf_score, score, reason in zip(selected_uids, vmaf_scores, scores, reasons):
            logger.info(f"{uid} ** {vmaf_score:.2f} ** {score:.4f} || {reason}")

        round_id = f"organic_upscaling_{int(time.time())}"
        
        logger.info(f"updating miner manager with {len(scores)} miner scores after organic upscaling requests processing…")
        accumulate_scores, applied_multipliers = self.miner_manager.step_organic_upscaling(scores, selected_uids, round_id)

        miner_hotkeys = [self.metagraph.hotkeys[uid] for uid in selected_uids]

        miner_data = {
            "validator_uid": self.my_subnet_uid,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "request_type": "Organic",
            "processing_task_type": "upscaling",
            "miner_uids": selected_uids,
            "miner_hotkeys": miner_hotkeys,
            "vmaf_scores": vmaf_scores,
            "pieapp_scores": pieapp_scores,
            "quality_scores": quality_scores,
            "length_scores": length_scores,
            "final_scores": scores,
            "accumulate_scores": accumulate_scores,
            "applied_multipliers": applied_multipliers,
            "status": reasons,
            "task_urls": selected_reference_urls,
            "processed_urls": selected_distorted_urls,
            "timestamp": timestamp
        }
        
        success = send_upscaling_data_to_dashboard(miner_data)
        if success:
            logger.info("Data successfully sent to dashboard")
        else:
            logger.info("Failed to send data to dashboard")

    async def score_organics_compression(
        self,
        uids: list[int],
        responses: list[protocol.Synapse],
        reference_urls: list[str],
        vmaf_thresholds: list[float],
        target_codecs: list[str],
        codec_modes: list[str],
        target_bitrates: list[float],
        timestamp: str,
        distorted_urls: list[str] | None = None,
    ):
        """Score organic compression tasks."""
        if distorted_urls is None:
            distorted_urls = [response.miner_response.optimized_video_url for response in responses]

        combined = list(zip(uids, distorted_urls, reference_urls, vmaf_thresholds, target_codecs, codec_modes, target_bitrates))
        if not combined:
            logger.warning("No organic compression responses available for scoring")
            return
        random.shuffle(combined)
        uids, distorted_urls, reference_urls, vmaf_thresholds, target_codecs, codec_modes, target_bitrates = map(list, zip(*combined))

        num_pairs_to_validate = min(5, len(combined))
        selected_indices = random.sample(range(len(combined)), num_pairs_to_validate)

        selected_uids = [uids[i] for i in selected_indices]
        selected_distorted_urls = [distorted_urls[i] for i in selected_indices]
        selected_reference_urls = [reference_urls[i] for i in selected_indices]
        selected_vmaf_thresholds = [vmaf_thresholds[i] for i in selected_indices]
        selected_target_codecs = [target_codecs[i] for i in selected_indices]
        selected_codec_modes = [codec_modes[i] for i in selected_indices]
        selected_target_bitrates = [target_bitrates[i] for i in selected_indices]

        logger.info(f"Randomly selected {len(selected_uids)} pairs out of {len(uids)} total pairs for compression validation")

        score_response = await self.score_client_compression_organics.post(
            "/score_organics_compression",
            json={
                "uids": selected_uids,
                "distorted_urls": selected_distorted_urls,
                "reference_urls": selected_reference_urls,
                "vmaf_thresholds": selected_vmaf_thresholds,
                "target_codecs": selected_target_codecs,
                "codec_modes": selected_codec_modes,
                "target_bitrates": selected_target_bitrates
            },
            timeout=180
        )

        response_data = score_response.json()
        scores = response_data.get("final_scores", [])
        vmaf_scores = response_data.get("vmaf_scores", [])
        compression_rates = response_data.get("compression_rates", [])
        reasons = response_data.get("reasons", [])

        max_length = max(len(selected_uids), len(scores), len(vmaf_scores), len(compression_rates), len(reasons))
        scores.extend([0.0] * (max_length - len(scores)))
        vmaf_scores.extend([0.0] * (max_length - len(vmaf_scores)))
        compression_rates.extend([0.5] * (max_length - len(compression_rates)))
        reasons.extend(["no reason provided"] * (max_length - len(reasons)))

        round_id = f"organic_compression_{int(time.time())}"
        
        logger.info(f"updating miner manager with {len(scores)} miner scores after organic compression requests processing…")
        accumulate_scores, applied_multipliers = self.miner_manager.step_organic_compression(scores, selected_uids, vmaf_scores, compression_rates, selected_vmaf_thresholds, round_id)

        logger.info(f"organic compression scoring results for {len(selected_uids)} miners")
        logger.info(f"uids: {selected_uids}")
        for uid, vmaf_score, final_score, reason, compression_rate, applied_multiplier, vmaf_threshold in zip(
            selected_uids, vmaf_scores, scores, reasons, compression_rates, applied_multipliers, selected_vmaf_thresholds
        ):
            logger.info(
                f"{uid} ** VMAF: {vmaf_score:.2f} "
                f"** VMAF Threshold: {vmaf_threshold} ** Compression Rate: {compression_rate:.4f} ** Final: {final_score:.4f} || {reason}"
            )

        miner_hotkeys = [self.metagraph.hotkeys[uid] for uid in selected_uids]

        miner_data = {
            "validator_uid": self.my_subnet_uid,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "request_type": "Organic",
            "processing_task_type": "compression",
            "miner_uids": selected_uids,
            "miner_hotkeys": miner_hotkeys,
            "vmaf_scores": vmaf_scores,
            "vmaf_thresholds": selected_vmaf_thresholds,
            "compression_rates": compression_rates,
            "final_scores": scores,
            "accumulate_scores": accumulate_scores,
            "applied_multipliers": applied_multipliers,
            "status": reasons,
            "task_urls": selected_reference_urls,
            "processed_urls": selected_distorted_urls,
            "timestamp": timestamp
        }
        
        success = send_compression_data_to_dashboard(miner_data)
        if success:
            logger.info("Compression data successfully sent to dashboard")
        else:
            logger.info("Failed to send compression data to dashboard")

    def filter_miners(self):
        min_stake = CONFIG.bandwidth.min_stake
        stake_array = self.metagraph.S
        miner_uids = [i for i, stake in enumerate(stake_array) if stake < min_stake]

        return self._deduplicate_uids_by_ip(miner_uids, "synthetic miner filter")

    def _axon_ip_address(self, uid: int) -> str:
        axon = self.metagraph.axons[uid]
        ip_value = None
        for attr in ("ip_str", "external_ip", "ip"):
            candidate = getattr(axon, attr, None)
            if callable(candidate):
                try:
                    candidate = candidate()
                except Exception as e:
                    logger.debug(f"Unable to resolve axon {attr}: {e}")
                    candidate = None

            if candidate is not None and candidate != "":
                ip_value = candidate
                break

        if ip_value is None:
            return ""
        return str(ip_value)

    def _deduplicate_uids_by_ip(self, uids: list[int], context: str) -> list[int]:
        seen_ips: dict[str, int] = {}
        selected = []

        for uid in uids:
            ip_address = self._axon_ip_address(uid)
            if ip_address and ip_address in seen_ips:
                logger.warning(
                    f"Skipping UID {uid} during {context}: duplicate IP {ip_address} "
                    f"already used by UID {seen_ips[ip_address]}"
                )
                continue

            if ip_address:
                seen_ips[ip_address] = uid
            selected.append(uid)

        removed = len(uids) - len(selected)
        if removed:
            logger.info(f"Removed {removed} duplicate-IP UIDs during {context}")

        return selected

    async def should_process_organic_upscaling(self):
        """Check if organic upscaling tasks should be processed."""
        num_organic_upscaling_chunks = get_organic_upscaling_queue_size(self.redis_conn)

        if num_organic_upscaling_chunks > 0:
            logger.info(f"🔷 | UPSCALING | The organic_upscaling_queue_size: {num_organic_upscaling_chunks}, processing organic upscaling requests. 🔷")
            await self.process_organic_upscaling_chunks(num_organic_upscaling_chunks)
            return True
        else:
            return False

    async def should_process_organic_compression(self):
        """Check if organic compression tasks should be processed."""
        num_organic_compression_chunks = get_organic_compression_queue_size(self.redis_conn)

        if num_organic_compression_chunks > 0:
            logger.info(f"🔷 | COMPRESSION | The organic_compression_queue_size: {num_organic_compression_chunks}, processing organic compression requests. 🔷")
            await self.process_organic_compression_chunks(num_organic_compression_chunks)
            return True
        else:
            return False

    async def process_organic_upscaling_chunks(self, num_organic_chunks):
        """Process organic upscaling chunks."""
        organic_start_time = time.time() 

        needed = min(CONFIG.bandwidth.requests_per_organic_interval, num_organic_chunks)
        batch_size = ORGANIC_QUERIES_PER_MINER
        miner_count_needed = (needed + batch_size - 1) // batch_size

        logger.info(
            f"☘️ | UPSCALING | Start processing {needed} organic chunks "
            f"with up to {batch_size} chunks per miner ☘️"
        )

        forward_uids = get_organic_forward_uids(self, miner_count_needed, "upscaling", CONFIG.bandwidth.min_stake)

        if len(forward_uids) < miner_count_needed:
            needed = min(needed, len(forward_uids) * batch_size)
            logger.info(
                f"There are just {len(forward_uids)} miners available for organic upscaling, "
                f"handling {needed} chunks"
            )

        if needed <= 0 or len(forward_uids) == 0:
            logger.warning("No available miners for organic upscaling")
            return

        task_ids, original_urls, task_types, synapses = await self.challenge_synthesizer.build_organic_upscaling_protocol(
            needed, bundle_size=batch_size
        )

        if len(task_ids) != needed or len(synapses) > len(forward_uids):
            logger.error(
                f"Mismatch in organic upscaling synapses after building organic protocol: "
                f"task_ids={len(task_ids)}, needed={needed}, synapses={len(synapses)}, uids={len(forward_uids)}"
            )
            return

        forward_uids = forward_uids[:len(synapses)]
        axon_list = [self.metagraph.axons[uid] for uid in forward_uids]

        logger.info("Updating task status to 'processing' for upscaling")
        for task_id, original_url in zip(task_ids, original_urls):
            await self.update_task_status(task_id, original_url, "processing")

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("🌜 | UPSCALING | Performing polling-based forward operations for upscaling 🌜")
        forward_tasks = [
            self.call_miner_polling(axon, synapse, uid, VideoUpscalingPollProtocol)
            for uid, axon, synapse in zip(forward_uids, axon_list, synapses)
        ]
        raw_responses = await asyncio.gather(*forward_tasks)
        responses = [response['result'] for response in raw_responses]

        flat_uids = []
        processed_urls = []
        for batch_idx, response in enumerate(responses):
            query_count = len(synapses[batch_idx].miner_payload.reference_video_urls)
            response_urls = self._response_video_urls(response, query_count)
            for processed_url in response_urls:
                flat_uids.append(int(forward_uids[batch_idx]))
                processed_urls.append(processed_url)

        logger.info(f"Processing organic upscaling chunks with uids: {forward_uids.tolist()}")
        logger.info("Updating task status to 'completed' and pushing results for upscaling")
        for task_id, original_url, processed_url in zip(task_ids, original_urls, processed_urls):
            await self.update_task_status(task_id, original_url, "completed")
            await self.push_result(task_id, original_url, processed_url)

        asyncio.create_task(
            self.score_organics_upscaling(
                flat_uids,
                [],
                original_urls,
                task_types,
                timestamp,
                distorted_urls=processed_urls,
            )
        )

        end_time = time.time()
        total_time = end_time - organic_start_time
        logger.info(f"🍏 Organic upscaling chunk processing complete in {total_time:.2f} seconds 🍏")

    async def process_organic_compression_chunks(self, num_organic_chunks):
        """Process organic compression chunks."""
        organic_start_time = time.time() 

        needed = min(CONFIG.bandwidth.requests_per_organic_interval, num_organic_chunks)
        batch_size = ORGANIC_QUERIES_PER_MINER
        miner_count_needed = (needed + batch_size - 1) // batch_size

        logger.info(
            f"☘️ | COMPRESSION | Start processing {needed} organic chunks "
            f"with up to {batch_size} chunks per miner ☘️"
        )

        forward_uids = get_organic_forward_uids(self, miner_count_needed, "compression", CONFIG.bandwidth.min_stake)

        if len(forward_uids) < miner_count_needed:
            needed = min(needed, len(forward_uids) * batch_size)
            logger.info(
                f"There are just {len(forward_uids)} miners available for organic compression, "
                f"handling {needed} chunks"
            )

        if needed <= 0 or len(forward_uids) == 0:
            logger.warning("No available miners for organic compression")
            return

        (
            task_ids,
            original_urls,
            vmaf_thresholds,
            target_codecs,
            codec_modes,
            target_bitrates,
            synapses,
        ) = await self.challenge_synthesizer.build_organic_compression_protocol(
            needed, bundle_size=batch_size
        )

        if len(task_ids) != needed or len(synapses) > len(forward_uids):
            logger.error(
                f"Mismatch in organic compression synapses after building organic protocol: "
                f"task_ids={len(task_ids)}, needed={needed}, synapses={len(synapses)}, uids={len(forward_uids)}"
            )
            return

        forward_uids = forward_uids[:len(synapses)]
        axon_list = [self.metagraph.axons[uid] for uid in forward_uids]

        logger.info(f"Processing organic compression chunks with uids: {forward_uids.tolist()}")
        logger.info("Updating task status to 'processing' for compression")
        for task_id, original_url in zip(task_ids, original_urls):
            await self.update_task_status(task_id, original_url, "processing")

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("🌜 | COMPRESSION | Performing polling-based forward operations for compression 🌜")

        forward_tasks = [
            self.call_miner_polling(axon, synapse, uid, VideoCompressionPollProtocol)
            for uid, axon, synapse in zip(forward_uids, axon_list, synapses)
        ]
        raw_responses = await asyncio.gather(*forward_tasks)
        responses = [response['result'] for response in raw_responses]

        flat_uids = []
        processed_urls = []
        for batch_idx, response in enumerate(responses):
            query_count = len(synapses[batch_idx].miner_payload.reference_video_urls)
            response_urls = self._response_video_urls(response, query_count)
            for processed_url in response_urls:
                flat_uids.append(int(forward_uids[batch_idx]))
                processed_urls.append(processed_url)

        logger.info("Updating task status to 'completed' and pushing results for compression")
        for task_id, original_url, processed_url in zip(task_ids, original_urls, processed_urls):
            await self.update_task_status(task_id, original_url, "completed")
            await self.push_result(task_id, original_url, processed_url)

        asyncio.create_task(
            self.score_organics_compression(
                flat_uids,
                [],
                original_urls,
                vmaf_thresholds,
                target_codecs,
                codec_modes,
                target_bitrates,
                timestamp,
                distorted_urls=processed_urls,
            )
        )

        end_time = time.time()
        total_time = end_time - organic_start_time
        logger.info(f"🍏 Organic compression chunk processing complete in {total_time:.2f} seconds 🍏")


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

    async def set_weights(self):
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
                success, msg = await asyncio.wait_for(asyncio.wrap_future(future), timeout=120)
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
        dev_mode = os.getenv("DEV_MODE", "False").lower() == "true"
        while True:
            try:
                if dev_mode:
                    logger.info("Not running weight_manager (DEV_MODE)...")
                else:
                    logger.info("Running weight_manager...")
                    await self.validator.set_weights()
            except Exception as e:
                logger.error(f"Error in WeightSynthesizer: {e}", exc_info=True)
            await asyncio.sleep(1200)  


if __name__ == "__main__":
    validator = Validator()
    weight_synthesizer = WeightSynthesizer(validator)
    dev_mode = os.getenv("DEV_MODE", "False").lower() == "true"
    time.sleep(10 if dev_mode else 1300) # wait till the video scheduler is ready

    set_scheduler_ready(validator.redis_conn, False)
    logger.info("Set scheduler readiness flag to False")

    async def main():
        validator_synthetic_task = asyncio.create_task(validator.run_synthetic())
        validator_organic_task = asyncio.create_task(validator.run_organic())
        weight_setter = asyncio.create_task(weight_synthesizer.run())
        wandb_maintenance_task = asyncio.create_task(
            validator.wandb_manager.run_maintenance()
        )

        await asyncio.gather(
            validator_synthetic_task,
            validator_organic_task,
            weight_setter,
            wandb_maintenance_task,
        )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        validator.wandb_manager.finish()
