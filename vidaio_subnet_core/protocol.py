from pydantic import BaseModel, Field
from bittensor import Synapse
from typing import Optional 
from enum import Enum

class Version(BaseModel):
    major: int
    minor: int
    patch: int

class ContentLength(Enum):
    """
    Enumeration of allowed video content lengths in seconds.
    These represent the maximum duration of video content that miners can process efficiently.
    """
    FIVE = 5    
    TEN = 10    
    TWENTY = 20 
    # FORTY = 40  
    # EIGHTY = 80    
    # ONE_SIXTY = 160  
    # THREE_TWENTY = 320 


class MinerPayload(BaseModel):
    reference_video_url: str = Field(
        description="The URL of the reference video to be optimized",
        default="",
        min_length=1,
    )
    maximum_optimized_size_mb: int = Field(
        description="The maximum size of the optimized video in MB",
        default=100,
        gt=0,
    )
    task_type: str = Field(
        description="The type of task: HD24K, SD2HD, SD24K, 4K28K",
        default="HD24K",
    )


class MinerResponse(BaseModel):
    optimized_video_url: str = Field(
        description="The URL of the processed video (compressed/upscaled)",
        default="",
    )


class ScoringPayload(BaseModel):
    reference_video_url: str = Field(
        description="The URL of the reference video",
        default="",
        min_length=1, 
    )
    optimized_video_url: str = Field(
        description="The URL of the processed video (compressed/upscaled)",
        default="",
        min_length=1,
    )


class ScoringResponse(BaseModel):
    score: float = Field(
        description="Quality score of the processed video",
        default=0.0,
        ge=0.0,  
        le=1.0,  
    )


class VideoCompressionProtocol(Synapse):
    """Protocol for video compression operations."""
    
    miner_payload: MinerPayload = Field(
        description="The payload for the miner. Cannot be modified after initialization.",
        default_factory=MinerPayload,
        frozen=True,
    )
    miner_response: MinerResponse = Field(
        description="The response from the miner",
        default_factory=MinerResponse,
    )

    @property
    def scoring_payload(self) -> ScoringPayload:
        """Generate scoring payload from miner payload and response."""
        return ScoringPayload(
            reference_video_url=self.miner_payload.reference_video_url,
            optimized_video_url=self.miner_response.optimized_video_url,
        )


class VideoUpscalingProtocol(Synapse):
    """Protocol for video upscaling operations."""
    
    version: Optional[Version] = None
    
    miner_payload: MinerPayload = Field(
        description="The payload for the miner. Cannot be modified after initialization.",
        default_factory=MinerPayload,
        frozen=True,
    )
    miner_response: MinerResponse = Field(
        description="The response from the miner",
        default_factory=MinerResponse,
    )

    @property
    def scoring_payload(self) -> ScoringPayload:
        """Generate scoring payload from miner payload and response."""
        return ScoringPayload(
            reference_video_url=self.miner_payload.reference_video_url,
            optimized_video_url=self.miner_response.optimized_video_url,
        )


class LengthCheckProtocol(Synapse):
    """
    Protocol for verifying and enforcing maximum video content length constraints.
    
    This protocol ensures that video processing requests don't exceed the miner's
    capacity to handle content within a reasonable timeframe. Miners can specify
    their maximum supported content length from the predefined options.
    
    Attributes:
        version (Optional[Version]): The version of the protocol implementation.
        max_content_length (ContentLength): Maximum video duration in seconds that
            miners can process, must be one of the predefined values (5, 10, or 20).
    """
    
    version: Optional[Version] = None
    max_content_length: ContentLength = Field(
        description="Maximum content length miner can process in 60 seconds",
        default=ContentLength.FIVE,
    )