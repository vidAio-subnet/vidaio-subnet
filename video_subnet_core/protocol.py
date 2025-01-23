from pydantic import BaseModel, Field
from bittensor import Synapse
from typing import Optional 


class MinerPayload(BaseModel):
    reference_video_url: str = Field(
        description="The URL of the reference video to be optimized",
        default="",
        min_length=1 
    )
    maximum_optimized_size_mb: int = Field(
        description="The maximum size of the optimized video in MB",
        default=100,
        gt=0  
    )


class MinerResponse(BaseModel):
    optimized_video_url: str = Field(
        description="The URL of the processed video (compressed/upscaled)",
        default="",
        min_length=1  
    )


class ScoringPayload(BaseModel):
    reference_video_url: str = Field(
        description="The URL of the reference video",
        default="",
        min_length=1  
    )
    optimized_video_url: str = Field(
        description="The URL of the processed video (compressed/upscaled)",
        default="",
        min_length=1  
    )


class ScoringResponse(BaseModel):
    score: float = Field(
        description="Quality score of the processed video",
        default=0.0,
        ge=0.0,  
        le=1.0  
    )


class VideoCompressionProtocol(Synapse):
    """Protocol for video compression operations."""
    
    miner_payload: MinerPayload = Field(
        description="The payload for the miner. Cannot be modified after initialization.",
        default_factory=MinerPayload,
        frozen=True
    )
    miner_response: MinerResponse = Field(
        description="The response from the miner",
        default_factory=MinerResponse
    )

    @property
    def scoring_payload(self) -> ScoringPayload:
        """Generate scoring payload from miner payload and response."""
        return ScoringPayload(
            reference_video_url=self.miner_payload.reference_video_url,
            optimized_video_url=self.miner_response.optimized_video_url
        )


class VideoUpscalingProtocol(Synapse):
    """Protocol for video upscaling operations."""
    
    miner_payload: MinerPayload = Field(
        description="The payload for the miner. Cannot be modified after initialization.",
        default_factory=MinerPayload,
        frozen=True
    )
    miner_response: MinerResponse = Field(
        description="The response from the miner",
        default_factory=MinerResponse
    )

    @property
    def scoring_payload(self) -> ScoringPayload:
        """Generate scoring payload from miner payload and response."""
        return ScoringPayload(
            reference_video_url=self.miner_payload.reference_video_url,
            optimized_video_url=self.miner_response.optimized_video_url
        )
