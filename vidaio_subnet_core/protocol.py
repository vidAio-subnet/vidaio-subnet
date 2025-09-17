from pydantic import BaseModel, Field
from bittensor import Synapse
from typing import Optional 
from enum import Enum, IntEnum

class Version(BaseModel):
    major: int
    minor: int
    patch: int
    
class ContentLength(IntEnum):
    """
    Enumeration of allowed video content lengths in seconds.
    These represent the maximum duration of video content that miners can process efficiently.
    """
    FIVE = 5
    TEN = 10
    # TWENTY = 20    
    # FORTY = 40  
    # EIGHTY = 80    
    # ONE_SIXTY = 160  
    # THREE_TWENTY = 320 

class TaskType(IntEnum):
    """
    Enumeration of allowed task types that miners can handle.
    These represent the types of video processing tasks that miners can warrant.
    """
    COMPRESSION = 1
    UPSCALING = 2

class UpscalingMinerPayload(BaseModel):
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

class CompressionMinerPayload(BaseModel):
    reference_video_url: str = Field(
        description="The URL of the reference video to be compressed",
        default="",
        min_length=1,
    )
    vmaf_threshold: float = Field(
        description="The VMAF threshold for quality control during compression",
        default=90.0,
        ge=0.0,
        le=100.0,
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
    
    version: Optional[Version] = None
    round_id: Optional[str] = None
    
    miner_payload: CompressionMinerPayload = Field(
        description="The payload for the compression miner. Cannot be modified after initialization.",
        default_factory=CompressionMinerPayload,
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

    round_id: Optional[str] = None
    
    miner_payload: UpscalingMinerPayload = Field(
        description="The payload for the upscaling miner. Cannot be modified after initialization.",
        default_factory=UpscalingMinerPayload,
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
    Protocol for verifying and enforcing maximum content length constraints.
    
    This protocol ensures that content processing requests don't exceed the miner's
    capacity to handle content within a reasonable timeframe. Miners can specify
    their maximum supported content length from the predefined options.
    
    Attributes:
        version (Optional[Version]): The version of the protocol implementation.
        max_content_length (ContentLength): Maximum content length that
            miners can process, must be one of the predefined values (5, 10, or 20).
    """
    
    version: Optional[Version] = None
    max_content_length: ContentLength = Field(
        description="Maximum content length miner can process (5, 10, or 20)",
        default=ContentLength.FIVE
    )


class TaskWarrantProtocol(Synapse):
    """
    Protocol for verifying and warranting task types that miners can handle.
    
    This protocol ensures that miners can specify which types of video processing
    tasks they are capable of handling. This helps in task distribution and
    ensures miners only receive tasks they can process.
    
    Attributes:
        version (Optional[Version]): The version of the protocol implementation.
        warrant_task (Optional[TaskType]): The type of task the miner can handle,
            must be one of the predefined values (COMPRESSION or UPSCALING).
            Will be None if miner doesn't respond, allowing fallback to performance history.
    """
    
    version: Optional[Version] = None
    warrant_task: Optional[TaskType] = Field(
        description="Type of task miner can handle: COMPRESSION or UPSCALING",
        default=None
    )