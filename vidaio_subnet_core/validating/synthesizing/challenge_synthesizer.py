from ...protocol import VideoUpscalingProtocol, MinerPayload
from ...global_config import CONFIG
import httpx
from typing import Dict, Tuple
from vidaio_subnet_core.utilities.version import get_version

class Synthesizer:
    def __init__(self):
        self.session = httpx.AsyncClient(
            base_url=f"http://{CONFIG.video_scheduler.host}:{CONFIG.video_scheduler.port}"
        )

    async def build_protocol(self) -> Tuple[str, VideoUpscalingProtocol]:
        """Fetches the prioritized video chunk and builds the video compression protocol.

        Returns:
            Tuple[str, VideoUpscalingProtocol]: A tuple containing the uploaded video ID and the
            corresponding VideoUpscalingProtocol instance.
        
        Raises:
            httpx.HTTPStatusError: If the request to the video scheduler fails.
        """
        try:
            response = await self.session.get("/api/get_prioritized_chunk")
            response.raise_for_status()
            data = response.json()
            chunk: Dict = data["chunk"]

            print(f"received chunk from video-scheduler api: {chunk}")
            video_id = chunk["video_id"]
            uploaded_object_name = chunk["uploaded_object_name"]
            sharing_link = chunk["sharing_link"]
            task_type = chunk["task_type"]

            version = get_version()
            
            return video_id, uploaded_object_name, VideoUpscalingProtocol(
                miner_payload=MinerPayload(reference_video_url=sharing_link, task_type = task_type),
                version = version,
            )
        except httpx.HTTPStatusError as e:
            print(f"Error fetching prioritized chunk: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
