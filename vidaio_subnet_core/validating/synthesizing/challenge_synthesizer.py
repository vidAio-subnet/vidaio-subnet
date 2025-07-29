import asyncio
import httpx
from typing import Tuple, Dict, List
from ...protocol import VideoUpscalingProtocol, MinerPayload, VideoCompressionProtocol, CompressionMinerPayload
from ...global_config import CONFIG
from loguru import logger

class Synthesizer:
    def __init__(self):
        self.session = httpx.AsyncClient(
            base_url=f"http://{CONFIG.video_scheduler.host}:{CONFIG.video_scheduler.port}",
            timeout=30.0
        )
        self.max_retries = 20
        self.retry_delay = 10  

    async def build_synthetic_protocol(self, content_lengths: list[int], version, round_id) -> Tuple[list[str], list[str], list[str], list[VideoUpscalingProtocol]]:
        """Fetches synthetic video chunks and builds the video compression protocols.
        
        Args:
            content_lengths: List of requested content durations in seconds
            version: Version of the protocol to use
            
        Returns:
            Tuple containing lists of:
            - payload URLs
            - video IDs
            - uploaded object names
            - VideoUpscalingProtocol instances
        
        Raises:
            httpx.HTTPStatusError: If the request to the video scheduler fails
            RuntimeError: If max retries exceeded without valid response
        """
        for attempt in range(self.max_retries):
            try:
                # Send the correct request with content_lengths
                response = await self.session.post(
                    "/api/get_synthetic_chunks",
                    json={"content_lengths": content_lengths}
                )
                response.raise_for_status()
                data = response.json()

                # Check if we have valid chunks
                if not data or not data.get("chunks") or not any(chunk for chunk in data.get("chunks", []) if chunk is not None):
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}: No valid chunks available, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunks = data["chunks"]
                logger.info(f"Received {len([c for c in chunks if c is not None])}/{len(chunks)} valid synthetic chunks")

                payload_urls = []
                video_ids = []
                uploaded_object_names = []
                synapses = []

                # Process only non-None chunks
                valid_chunks = [chunk for chunk in chunks if chunk is not None]
                
                if not valid_chunks:
                    logger.warning("All chunks were None, retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                for chunk in valid_chunks:
                    # Validate chunk data
                    required_fields = ["video_id", "uploaded_object_name", "sharing_link", "task_type"]
                    if not all(field in chunk for field in required_fields):
                        logger.warning(f"Missing required fields in chunk data: {chunk}")
                        continue

                    payload_urls.append(chunk["sharing_link"])
                    video_ids.append(chunk["video_id"])
                    uploaded_object_names.append(chunk["uploaded_object_name"])
                    
                    synapse = VideoUpscalingProtocol(
                        miner_payload=MinerPayload(
                            reference_video_url=chunk["sharing_link"],
                            task_type=chunk["task_type"]
                        ),
                        version=version,
                        round_id=round_id
                    )
                    synapses.append(synapse)

                if not synapses:
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: No valid protocols could be created, retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                    
                # Return the results if we have valid data
                return payload_urls, video_ids, uploaded_object_names, synapses
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}", exc_info=True)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
                    
        raise RuntimeError(f"Failed to get synthetic chunks after {self.max_retries} attempts")

    async def build_compression_protocol(self, vmaf_thresholds: list[float], version, round_id) -> Tuple[list[str], list[str], list[str], list[VideoCompressionProtocol]]:
        """Fetches synthetic video chunks and builds the video compression protocols.
        
        Args:
            vmaf_thresholds: List of VMAF thresholds for compression quality control
            version: Version of the protocol to use
            round_id: Unique identifier for this round
            
        Returns:
            Tuple containing lists of:
            - payload URLs
            - video IDs
            - uploaded object names
            - VideoCompressionProtocol instances
        
        Raises:
            httpx.HTTPStatusError: If the request to the video scheduler fails
            RuntimeError: If max retries exceeded without valid response
        """
        for attempt in range(self.max_retries):
            try:
                # Send the correct request with content_lengths (using default 5s for compression)
                content_lengths = [10] * len(vmaf_thresholds)  # Default 5s for compression tasks
                response = await self.session.post(
                    "/api/get_synthetic_chunks",
                    json={"content_lengths": content_lengths}
                )
                response.raise_for_status()
                data = response.json()

                # Check if we have valid chunks
                if not data or not data.get("chunks") or not any(chunk for chunk in data.get("chunks", []) if chunk is not None):
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}: No valid chunks available for compression, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunks = data["chunks"]
                logger.info(f"Received {len([c for c in chunks if c is not None])}/{len(chunks)} valid compression chunks")

                payload_urls = []
                video_ids = []
                uploaded_object_names = []
                synapses = []

                # Process only non-None chunks
                valid_chunks = [chunk for chunk in chunks if chunk is not None]
                
                if not valid_chunks:
                    logger.warning("All compression chunks were None, retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                for i, chunk in enumerate(valid_chunks):
                    # Validate chunk data
                    required_fields = ["video_id", "uploaded_object_name", "sharing_link"]
                    if not all(field in chunk for field in required_fields):
                        logger.warning(f"Missing required fields in compression chunk data: {chunk}")
                        continue

                    payload_urls.append(chunk["sharing_link"])
                    video_ids.append(chunk["video_id"])
                    uploaded_object_names.append(chunk["uploaded_object_name"])
                    
                    # Use the corresponding VMAF threshold, or default to 90.0
                    vmaf_threshold = vmaf_thresholds[i] if i < len(vmaf_thresholds) else 90.0
                    
                    synapse = VideoCompressionProtocol(
                        miner_payload=CompressionMinerPayload(
                            reference_video_url=chunk["sharing_link"],
                            vmaf_threshold=vmaf_threshold
                        ),
                        version=version,
                        round_id=round_id
                    )
                    synapses.append(synapse)

                if not synapses:
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: No valid compression protocols could be created, retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                    
                # Return the results if we have valid data
                return payload_urls, video_ids, uploaded_object_names, synapses
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}", exc_info=True)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
                    
        raise RuntimeError(f"Failed to get compression chunks after {self.max_retries} attempts")

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

    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()