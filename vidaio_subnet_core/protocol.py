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
    target_codec: str = Field(
        description="The target codec for compression (e.g., av1, hevc, h264, vp9)",
        default="av1",
    )
    codec_mode: str = Field(
        description="Codec mode: CBR (Constant Bitrate), VBR (Variable Bitrate), or CRF (Constant Rate Factor)",
        default="CRF",
    )
    target_bitrate: float = Field(
        description="Target bitrate in Mbps (megabits per second)",
        default=10.0,
        gt=0.0,
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


# =============================================================================
# TEE (Trusted Execution Environment) Protocol Definitions
# =============================================================================
# These protocols enable secure video processing where miners cannot see
# video URLs, content, or storage locations. All sensitive data is encrypted
# and only decrypted inside Intel SGX enclaves.


class EncryptedTaskPayload(BaseModel):
    """
    Fully encrypted task payload for TEE-protected video processing.
    
    This payload is encrypted by the validator and can only be decrypted
    inside the miner's SGX enclave. It contains:
    - The video URL to process
    - Storage credentials for uploading results
    - Task parameters (upscaling/compression settings)
    - Result object key (where to upload the processed video)
    
    The miner never sees any of this data in plaintext.
    """
    encrypted_blob: str = Field(
        description="Base64-encoded AES-GCM encrypted JSON containing all task data",
        default="",
    )
    nonce: str = Field(
        description="Base64-encoded nonce for AES-GCM decryption",
        default="",
    )
    session_key_id: str = Field(
        description="Identifier for the session key established during attestation",
        default="",
    )


class TEEMinerResponse(BaseModel):
    """
    Miner response for TEE-protected operations.
    
    Unlike regular MinerResponse, this does NOT contain any video URLs.
    The miner only reports success/failure and processing statistics.
    The validator retrieves results directly from storage using the
    result_object_key it specified in the encrypted payload.
    """
    success: bool = Field(
        description="Whether the video processing completed successfully",
        default=False,
    )
    processing_time_seconds: float = Field(
        description="Time taken to process the video in seconds",
        default=0.0,
        ge=0.0,
    )
    error_message: Optional[str] = Field(
        description="Error details if processing failed",
        default=None,
    )
    result_checksum: str = Field(
        description="SHA-256 checksum of the uploaded result for verification",
        default="",
    )


class TEECapabilities(BaseModel):
    """
    TEE capabilities reported by the miner.
    
    Used during attestation to communicate what the miner's enclave supports.
    """
    sgx_supported: bool = Field(
        description="Whether Intel SGX is available and functional",
        default=False,
    )
    attestation_type: str = Field(
        description="Type of attestation supported: 'dcap', 'epid', or 'none'",
        default="none",
    )
    enclave_version: str = Field(
        description="Version of the miner enclave code",
        default="",
    )
    max_enclave_memory_mb: int = Field(
        description="Maximum enclave memory available in MB",
        default=0,
    )


class TEEAttestationProtocol(Synapse):
    """
    Protocol for SGX attestation exchange between validator and miner.
    
    Flow:
    1. Validator sends challenge to miner
    2. Miner generates SGX Quote embedding the challenge
    3. Miner returns quote and capabilities
    4. Validator verifies quote and provisions session key
    
    This establishes trust that the miner is running inside a genuine
    SGX enclave with the expected code (MRENCLAVE).
    """
    
    version: Optional[Version] = None
    
    # Validator -> Miner: Random challenge
    challenge: str = Field(
        description="Base64-encoded random challenge from validator (32 bytes)",
        default="",
    )
    
    # Miner -> Validator: SGX Quote
    sgx_quote: str = Field(
        description="Base64-encoded SGX Quote proving enclave identity",
        default="",
    )
    
    # Miner -> Validator: Enclave capabilities
    capabilities: TEECapabilities = Field(
        description="TEE capabilities of the miner",
        default_factory=TEECapabilities,
    )
    
    # Validator -> Miner: Encrypted session key (after verification)
    encrypted_session_key: str = Field(
        description="Base64-encoded encrypted session key (set after quote verification)",
        default="",
    )
    session_key_id: str = Field(
        description="Identifier for the provisioned session key",
        default="",
    )
    
    # Attestation result
    attestation_success: bool = Field(
        description="Whether attestation was successful",
        default=False,
    )
    attestation_error: Optional[str] = Field(
        description="Error message if attestation failed",
        default=None,
    )


class TEEVideoUpscalingProtocol(Synapse):
    """
    TEE-protected protocol for video upscaling operations.
    
    Unlike VideoUpscalingProtocol, all sensitive data (video URL, storage
    credentials, result location) is encrypted and only decrypted inside
    the miner's SGX enclave. The miner never sees the video URL or content.
    
    The response contains only success/failure status and a checksum -
    no video URLs are exposed to the miner.
    """
    
    version: Optional[Version] = None
    round_id: Optional[str] = None
    
    # Encrypted payload containing all task data
    encrypted_payload: EncryptedTaskPayload = Field(
        description="Encrypted task payload (URL, credentials, params) - decrypted only in enclave",
        default_factory=EncryptedTaskPayload,
        frozen=True,
    )
    
    # Miner response with NO URLs
    miner_response: TEEMinerResponse = Field(
        description="Response from miner containing only status and checksum",
        default_factory=TEEMinerResponse,
    )


class TEEVideoCompressionProtocol(Synapse):
    """
    TEE-protected protocol for video compression operations.
    
    Unlike VideoCompressionProtocol, all sensitive data (video URL, storage
    credentials, result location, compression parameters) is encrypted and
    only decrypted inside the miner's SGX enclave.
    
    The response contains only success/failure status and a checksum -
    no video URLs are exposed to the miner.
    """
    
    version: Optional[Version] = None
    round_id: Optional[str] = None
    
    # Encrypted payload containing all task data
    encrypted_payload: EncryptedTaskPayload = Field(
        description="Encrypted task payload (URL, credentials, params) - decrypted only in enclave",
        default_factory=EncryptedTaskPayload,
        frozen=True,
    )
    
    # Miner response with NO URLs
    miner_response: TEEMinerResponse = Field(
        description="Response from miner containing only status and checksum",
        default_factory=TEEMinerResponse,
    )