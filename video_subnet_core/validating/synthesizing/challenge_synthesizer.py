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
        data = await response.json()
        chunk_url = data["chunk_url"]
        return VideoCompressionProtocol(
            miner_payload=MinerPayload(reference_video_url=chunk_url)
        )
