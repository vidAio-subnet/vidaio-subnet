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

    async def build_synthetic_protocol(self, content_lengths: list[int], version) -> Tuple[list[str], list[str], list[str], list[VideoUpscalingProtocol]]:
        """Fetches the synthetic video chunk and builds the video compression protocol.
        
        Args:
            content_lengths: List of content lengths to send to the endpoint
            version: Protocol version
            
        Returns:
            Tuple[list[str], list[str], list[str], list[VideoUpscalingProtocol]]: A tuple containing 
            the payload URLs, video IDs, uploaded object names, and the corresponding 
            VideoUpscalingProtocol instances.
        
        Raises:
            httpx.HTTPStatusError: If the request to the video scheduler fails.
            RuntimeError: If max retries exceeded without valid response.
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.session.post(
                    "/api/get_synthetic_chunks",
                    json={"content_lengths": content_lengths}                               
                )
                response.raise_for_status()
                data = response.json()

                if not data or not data.get("chunks"):
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}: No chunk available, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunks: Dict = data["chunks"]
                logger.info("Received synthetic chunks from video-scheduler API")

                required_fields = ["video_ids", "uploaded_object_names", "sharing_links", "task_types"]
                if not all(field in chunks for field in required_fields):
                    logger.info(f"Missing required fields in chunk data: {chunks}")
                    await asyncio.sleep(self.retry_delay)
                    continue
                
                payload_urls = chunks["sharing_links"]
                video_ids = chunks["video_ids"]
                uploaded_object_names = chunks["uploaded_object_names"]
                task_types = chunks["task_types"]

                # Ensure all lists have the same length
                if not (len(payload_urls) == len(task_types) == len(video_ids) == len(uploaded_object_names)):
                    logger.warning(f"Inconsistent data lengths: payload_urls={len(payload_urls)}, task_types={len(task_types)}, video_ids={len(video_ids)}, uploaded_object_names={len(uploaded_object_names)}")
                    await asyncio.sleep(self.retry_delay)
                    continue

                synapses = []
                # Use zip to iterate through multiple lists simultaneously
                for payload_url, task_type in zip(payload_urls, task_types):
                    synapse = VideoUpscalingProtocol(
                        miner_payload=MinerPayload(
                            reference_video_url=payload_url,
                            task_type=task_type,
                        ),
                        version=version
                    )
                    synapses.append(synapse)

                return payload_urls, video_ids, uploaded_object_names, synapses
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
                    
        raise RuntimeError(f"Failed to get synthetic chunk after {self.max_retries} attempts")

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
