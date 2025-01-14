from pydantic import BaseModel, Field
from bittensor import Synapse


class MinerPayload(BaseModel):
    reference_video_url: str = Field(
        description="The URL of the reference video to be compressed", default=""
    )
    maximum_compressed_size_mb: int = Field(
        description="The maximum size of the compressed video in MB", default=100
    )


class MinerResponse(BaseModel):
    compressed_video_url: str = Field(
        description="The URL of the compressed video", default=""
    )


class ScoringPayload(BaseModel):
    reference_video_url: str = Field(
        description="The URL of the reference video", default=""
    )
    compressed_video_url: str = Field(
        description="The URL of the compressed video", default=""
    )


class ScoringResponse(BaseModel):
    score: float = Field(description="The score of the compressed video", default=0.0)


class VideoCompressionProtocol(Synapse):
    miner_payload: MinerPayload = Field(
        description="The payload for the miner. Can not modify this field",
        default=MinerPayload(),
        frozen=True,
    )
    miner_response: MinerResponse = Field(
        description="The response from the miner", default=MinerResponse()
    )

    @property
    def scoring_payload(self) -> ScoringPayload:
        return ScoringPayload(
            reference_video_url=self.miner_payload.reference_video_url,
            compressed_video_url=self.miner_response.compressed_video_url,
        )
