from ...protocol import VideoCompressionProtocol, MinerPayload
from ...global_config import CONFIG
import httpx


class Synthesizer:
    def __init__(self):
        self.session = httpx.AsyncClient(
            base_url=f"http://{CONFIG.video_scheduler.host}:{CONFIG.video_scheduler.port}"
        )

    async def build_protocol(self) -> VideoCompressionProtocol:
        response = await self.session.get("/api/get_prioritized_chunk")
        response.raise_for_status()
        data = response.json()
        print(data)
        chunk_url = data["chunk_url"]
        
        video_reference_url_4k = chunk_url + "_4k.mp4"
        video_reference_url_hd = chunk_url + "_hd.mp4"
        
        return video_reference_url_4k, VideoCompressionProtocol(
            miner_payload=MinerPayload(reference_video_url=video_reference_url_hd)
        ) 
