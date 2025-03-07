import asyncio
import httpx
from typing import Tuple, Dict
from ...protocol import VideoUpscalingProtocol, MinerPayload
from ...global_config import CONFIG

class Synthesizer:
    def __init__(self):
        self.session = httpx.AsyncClient(
            base_url=f"http://{CONFIG.video_scheduler.host}:{CONFIG.video_scheduler.port}"
        )
        self.max_retries = 20
        self.retry_delay = 10  # seconds

    async def build_protocol(self) -> Tuple[str, str, VideoUpscalingProtocol]:
        """Fetches the prioritized video chunk and builds the video compression protocol.
        
        Returns:
            Tuple[str, str, VideoUpscalingProtocol]: A tuple containing the video ID, 
            uploaded object name, and the corresponding VideoUpscalingProtocol instance.
        
        Raises:
            httpx.HTTPStatusError: If the request to the video scheduler fails.
            RuntimeError: If max retries exceeded without valid response.
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.session.get("/api/get_prioritized_chunk")
                response.raise_for_status()
                data = response.json()

                # check if response data is None or empty
                if not data or not data.get("chunk"):
                    print(f"Attempt {attempt + 1}/{self.max_retries}: No chunk available, waiting {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue

                chunk: Dict = data["chunk"]
                print(f"received chunk from video-scheduler api: {chunk}")

                # validate required fields
                required_fields = ["video_id", "uploaded_object_name", "sharing_link", "task_type"]
                if not all(field in chunk for field in required_fields):
                    print(f"Missing required fields in chunk data: {chunk}")
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
                print(f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to get valid response after {self.max_retries} attempts")

