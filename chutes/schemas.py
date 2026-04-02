from typing import Any
from pydantic import BaseModel


class UpscalingInput(BaseModel):
    video_url: str
    task_type: str  # HD24K, SD2HD, SD24K, 4K28K
    upload_url: str = ""  # presigned PUT URL for result upload (optional in local test)


class CompressionInput(BaseModel):
    video_url: str
    vmaf_threshold: float
    target_codec: str  # av1, hevc, h264, vp9
    codec_mode: str  # CRF, CBR, VBR
    target_bitrate: float  # Mbps
    upload_url: str = ""  # presigned PUT URL for result upload (optional in local test)


class ProcessOutput(BaseModel):
    success: bool
    output_url: str | None = None
    output_video_b64: str | None = None  # base64-encoded result video (local test fallback)
    error: str | None = None
