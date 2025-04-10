import asyncio
import httpx
from typing import Tuple, Dict, List
from ...protocol import VideoUpscalingProtocol, MinerPayload
from ...global_config import CONFIG
from loguru import logger

class Synthesizer:
    def __init__(self):
        self.session = httpx.AsyncClient(
            base_url=f"http://{CONFIG.video_scheduler.host}:{CONFIG.video_scheduler.port}"
        )
        self.max_retries = 20
        self.retry_delay = 10  # seconds

    async def build_synthetic_protocol(self) -> Tuple[str, str, VideoUpscalingProtocol]:
        """Fetches the synthetic video chunk and builds the video compression protocol.
        
        Returns:
            Tuple[str, str, VideoUpscalingProtocol]: A tuple containing the video ID, 
            uploaded object name, and the corresponding VideoUpscalingProtocol instance.
        
        Raises:
            httpx.HTTPStatusError: If the request to the video scheduler fails.
            RuntimeError: If max retries exceeded without valid response.
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.session.get("/api/get_synthetic_chunk")
                response.raise_for_status()
                data = response.json()

                # check if response data is None or empty
                if not data or not data.get("chunk"):
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}: No chunk available, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunk: Dict = data["chunk"]
                logger.info(f"received synthetic chunk from video-scheduler api")

                # validate required fields
                required_fields = ["video_id", "uploaded_object_name", "sharing_link", "task_type"]
                if not all(field in chunk for field in required_fields):
                    logger.info(f"Missing required fields in chunk data: {chunk}")
                    await asyncio.sleep(self.retry_delay)
                    continue

                video_id = chunk["video_id"]
                uploaded_object_name = chunk["uploaded_object_name"]
                sharing_link = chunk["sharing_link"]
                task_type = chunk["task_type"]

                return video_id, uploaded_object_name, VideoUpscalingProtocol(
                    miner_payload=MinerPayload(
                        reference_video_url=sharing_link,
                        task_type=task_type
                    ),
                )

            except httpx.HTTPStatusError as e:
                logger.info(f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                logger.info(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to get valid response after {self.max_retries} attempts")

    async def build_organic_protocol(self, needed: int):
        for attempt in range(self.max_retries):
            try:
                response = await self.session.get(f"/api/get_organic_chunks?needed={needed}")
                response.raise_for_status()
                data = response.json()

                # check if response data is None or empty
                if not data or not data.get("chunks"):
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}: No chunk available, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunks: List[Dict] = data["chunks"]
                logger.info(f"received organic chunks from video-scheduler api")

                # validate required fields
                required_fields = ["url", "chunk_id", "task_id", "resolution_type"]
                should_retry = False
                for chunk in chunks:
                    if not all(field in chunk for field in required_fields):
                        logger.info(f"Missing required fields in chunk data: {chunk}")
                        should_retry = True
                        break
                if should_retry:
                    logger.info(f"Failed to fetching organic chunks, retrying after {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                
                organic_synapses = []
                task_ids = []
                original_urls = []
                for chunk in chunks:
                    payload_url = chunk["url"]
                    chunk_id = chunk["chunk_id"]
                    task_id = chunk["task_id"]
                    resolution_type = chunk["resolution_type"]

                    synapse = VideoUpscalingProtocol(
                        miner_payload=Minerpayload(
                            reference_video_url=payload_url,
                            task_type=chunk['resolution_type']
                        ),
                    )

                    task_ids.append(task_id)
                    organic_synapses.append(synsapse)
                    original_urls.append(payload_url)

                return task_ids, original_urls, organic_synapses

            except httpx.HTTPStatusError as e:
                logger.info(f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                logger.info(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to get valid response after {self.max_retries} attempts")



