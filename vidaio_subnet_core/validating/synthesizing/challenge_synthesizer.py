import asyncio
import httpx
from typing import Tuple, Dict, List
from ...protocol import VideoUpscalingProtocol, UpscalingMinerPayload, VideoCompressionProtocol, CompressionMinerPayload
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

    async def build_synthetic_protocol(self, content_lengths: list[int], version, round_id) -> Tuple[list[str], list[str], list[str], list[VideoUpscalingProtocol], list[str]]:
        """Fetches synthetic video chunks and builds the video upscaling protocols.
        
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
        # Count occurrences of each content length and calculate required chunks
        from collections import Counter
        content_counts = Counter(content_lengths)
        
        # Get miners per task from configuration
        miners_per_task = CONFIG.bandwidth.miners_per_task
        
        # Calculate required chunks (one chunk per miners_per_task miners of same length)
        required_chunks = []
        for length, count in content_counts.items():
            chunks_needed = (count + miners_per_task - 1) // miners_per_task  # Ceiling division to ensure enough chunks
            required_chunks.extend([length] * chunks_needed)
        
        logger.info(f"Original content_lengths: {content_lengths}")
        logger.info(f"Using {miners_per_task} miners per task")
        logger.info(f"Optimized chunk request: {required_chunks} (reduced from {len(content_lengths)} to {len(required_chunks)} chunks)")
        
        for attempt in range(self.max_retries):
            try:
                # Send the optimized request with required_chunks
                response = await self.session.post(
                    "/api/get_synthetic_chunks",
                    json={"content_lengths": required_chunks}
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
                task_types = []
                synapses = []

                # Process only non-None chunks
                valid_chunks = [chunk for chunk in chunks if chunk is not None]
                
                if not valid_chunks:
                    logger.warning("All chunks were None, retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                # Group chunks by content length for efficient protocol generation
                chunks_by_length = {}
                chunk_assignment_index = 0
                
                for chunk in valid_chunks:
                    # Validate chunk data
                    required_fields = ["video_id", "uploaded_object_name", "sharing_link", "task_type"]
                    if not all(field in chunk for field in required_fields):
                        logger.warning(f"Missing required fields in chunk data: {chunk}")
                        continue
                    
                    # Assign chunks to lengths based on required_chunks order
                    if chunk_assignment_index < len(required_chunks):
                        chunk_length = required_chunks[chunk_assignment_index]
                        chunk_assignment_index += 1
                    else:
                        logger.warning(f"More chunks received than expected, skipping chunk")
                        continue
                    
                    if chunk_length not in chunks_by_length:
                        chunks_by_length[chunk_length] = []
                    chunks_by_length[chunk_length].append(chunk)

                # Generate protocols in the exact order of original content_lengths
                length_usage_count = {length: 0 for length in content_counts.keys()}
                chunk_usage_count = {length: 0 for length in content_counts.keys()}
                
                for length in content_lengths:
                    available_chunks = chunks_by_length.get(length, [])
                    
                    if not available_chunks:
                        logger.warning(f"No chunks available for length {length}s")
                        continue
                    
                    # Determine which chunk to use (one chunk per miners_per_task miners)
                    chunk_index = chunk_usage_count[length] // miners_per_task
                    
                    if chunk_index >= len(available_chunks):
                        # If we run out of chunks, cycle back to first chunk
                        chunk_index = 0
                        logger.warning(f"Cycling back to first chunk for length {length}s")
                    
                    chunk = available_chunks[chunk_index]
                    
                    try:
                        payload_urls.append(chunk["sharing_link"])
                        video_ids.append(chunk["video_id"])
                        task_types.append(chunk["task_type"])
                        uploaded_object_names.append(chunk["uploaded_object_name"])
                        
                        synapse = VideoUpscalingProtocol(
                            miner_payload=UpscalingMinerPayload(
                                reference_video_url=chunk["sharing_link"],
                                task_type=chunk["task_type"]
                            ),
                            version=version,
                            round_id=round_id
                        )
                        synapses.append(synapse)
                        
                        chunk_usage_count[length] += 1
                        
                    except Exception as e:
                        logger.error(f"Error creating protocol for length {length}: {e}")
                        continue

                if not synapses:
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: No valid protocols could be created, retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                
                logger.info(f"Successfully created {len(synapses)} protocols from {len(valid_chunks)} chunks")
                # Return the results if we have valid data
                return payload_urls, video_ids, uploaded_object_names, synapses, task_types
                
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

    async def build_compression_protocol(self, vmaf_threshold: float, num_miners: int, version, round_id) -> Tuple[list[str], list[str], list[str], list[VideoCompressionProtocol]]:
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
        # Get miners per task from configuration
        miners_per_task = CONFIG.bandwidth.miners_per_task
        
        # Calculate required chunks (one chunk per miners_per_task protocols)
        num_protocols = num_miners
        num_needed = (num_protocols + miners_per_task - 1) // miners_per_task  # Ceiling division
        
        logger.info(f"Vmaf_threshold: {vmaf_threshold}")
        logger.info(f"Using {miners_per_task} miners per task")
        logger.info(f"Optimized chunk request: {num_needed} chunks (reduced from {num_protocols} protocols)")
        
        for attempt in range(self.max_retries):
            try:
                # Send the optimized request with calculated num_needed
                response = await self.session.post(
                    "/api/get_compression_chunks",
                    json={"num_needed": num_needed}
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

                # Generate protocols in order, reusing chunks based on miners_per_task
                for i in range(num_miners):
                    # Determine which chunk to use (one chunk per miners_per_task protocols)
                    chunk_index = i // miners_per_task
                    
                    if chunk_index >= len(valid_chunks):
                        # If we run out of chunks, cycle back to first chunk
                        chunk_index = 0
                        logger.warning(f"Cycling back to first chunk for protocol {i}")
                    
                    chunk = valid_chunks[chunk_index]
                    
                    # Validate chunk data
                    required_fields = ["video_id", "uploaded_object_name", "sharing_link"]
                    if not all(field in chunk for field in required_fields):
                        logger.warning(f"Missing required fields in compression chunk data: {chunk}")
                        continue

                    try:
                        payload_urls.append(chunk["sharing_link"])
                        video_ids.append(chunk["video_id"])
                        uploaded_object_names.append(chunk["uploaded_object_name"])
                        
                        synapse = VideoCompressionProtocol(
                            miner_payload=CompressionMinerPayload(
                                reference_video_url=chunk["sharing_link"],
                                vmaf_threshold=vmaf_threshold
                            ),
                            version=version,
                            round_id=round_id
                        )
                        synapses.append(synapse)
                        
                    except Exception as e:
                        logger.error(f"Error creating compression protocol {i}: {e}")
                        continue

                if not synapses:
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: No valid compression protocols could be created, retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                
                logger.info(f"Successfully created {len(synapses)} compression protocols from {len(valid_chunks)} chunks")
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

    async def build_organic_upscaling_protocol(self, needed: int):
        for attempt in range(self.max_retries):
            try:
                response = await self.session.get(f"/api/get_organic_upscaling_chunks?needed={needed}")
                response.raise_for_status()
                data = response.json()

                if not data or not data.get("chunks"):
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}: No chunks available, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunks: List[Dict] = data["chunks"]
                logger.info("Received organic upscaling chunks from video-scheduler API")

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
                        miner_payload=UpscalingMinerPayload(
                            reference_video_url=chunk["url"],
                            task_type=chunk["resolution_type"],
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

    async def build_organic_compression_protocol(self, needed: int):
        for attempt in range(self.max_retries):
            try:
                response = await self.session.get(f"/api/get_organic_compression_chunks?needed={needed}")
                response.raise_for_status()
                data = response.json()

                if not data or not data.get("chunks"):
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}: No chunks available, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunks: List[Dict] = data["chunks"]
                logger.info("Received organic compression chunks from video-scheduler API")

                required_fields = ["url", "chunk_id", "task_id", "compression_type"]
                if any(not all(field in chunk for field in required_fields) for chunk in chunks):
                    logger.info("Missing required fields in some chunk data, retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                task_ids = []
                original_urls = []
                organic_synapses = []
                vmaf_thresholds = []

                for chunk in chunks:
                    vmaf_threshold = None
                    if chunk["compression_type"] == "High":
                        vmaf_threshold = 95
                    elif chunk["compression_type"] == "Medium":
                        vmaf_threshold = 90
                    elif chunk["compression_type"] == "Low":
                        vmaf_threshold = 85

                    if vmaf_threshold is None:
                        logger.info(f"Invalid compression type: {chunk['compression_type']}")
                        continue

                    synapse = VideoCompressionProtocol(
                        miner_payload=CompressionMinerPayload(
                            reference_video_url=chunk["url"],
                            vmaf_threshold=vmaf_threshold
                        ),
                    )
                    task_ids.append(chunk["task_id"])
                    original_urls.append(chunk["url"])
                    vmaf_thresholds.append(vmaf_threshold)

                    organic_synapses.append(synapse)

                return task_ids, original_urls, vmaf_thresholds, organic_synapses

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