from ...protocol import VideoCompressionProtocol, MinerPayload
import aiohttp
import os


class Synthesizer:
    def __init__(self):
        self.session = aiohttp.ClientSession(
            base_url=f"http://{os.getenv('VIDEO_SCHEDULER_HOST')}:{os.getenv('VIDEO_SCHEDULER_PORT')}"
        )

    async def build_protocol(self) -> VideoCompressionProtocol:
        response = await self.session.get("/api/get_prioritized_chunk")
        response.raise_for_status()
        data = await response.json()
        chunk_url = data["chunk_url"]
        return VideoCompressionProtocol(
            miner_payload=MinerPayload(reference_video_url=chunk_url)
        )
