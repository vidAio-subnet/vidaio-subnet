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

                if not data or not data.get("chunk"):
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}: No chunk available, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunk: Dict = data["chunk"]
                logger.info("Received synthetic chunk from video-scheduler API")

                required_fields = ["video_id", "uploaded_object_name", "sharing_link", "task_type"]
                if not all(field in chunk for field in required_fields):
                    logger.info(f"Missing required fields in chunk data: {chunk}")
                    await asyncio.sleep(self.retry_delay)
                    continue

                return (
                    chunk["video_id"],
                    chunk["uploaded_object_name"],
                    VideoUpscalingProtocol(
                        miner_payload=MinerPayload(
                            reference_video_url=chunk["sharing_link"],
                            task_type=chunk["task_type"]
                        ),
                    )
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

                if not data or not data.get("chunks"):
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}: No chunks available, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunks: List[Dict] = data["chunks"]
                logger.info("Received organic chunks from video-scheduler API")

                required_fields = ["url", "chunk_id", "task_id", "resolution_type"]
                if any(not all(field in chunk for field in required_fields) for chunk in chunks):
                    logger.info("Missing required fields in some chunk data, retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                task_ids = []
                original_urls = []
                organic_synapses = []
                task_types = []

                for chunk in chunks:
                    synapse = VideoUpscalingProtocol(
                        miner_payload=MinerPayload(
                            reference_video_url=chunk["url"],
                            task_type=chunk["resolution_type"]
                        ),
                    )
                    task_ids.append(chunk["task_id"])
                    original_urls.append(chunk["url"])
                    task_types.append(chunk["resolution_type"])

                    organic_synapses.append(synapse)

                return task_ids, original_urls, task_types, organic_synapses

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
