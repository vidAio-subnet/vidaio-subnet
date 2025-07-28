from pydantic import BaseModel, Field


class VideoSchedulerConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    max_synthetic_queue_size: int = Field(default=150)  # Increased from 50
    refill_threshold: int = Field(default=70)  # Increased from 25
    refill_target: int = Field(default=100)  # Increased from 30
    min_video_len: int = Field(default=25)
    max_video_len: int = Field(default=30)  # Set to 30 for optimized video duration
    pexels_max_size: int = Field(default=150)  # Increased from 110
    pexels_threshold: int = Field(default=120)  # Increased from 90
    weight_hd_to_4k: float = Field(default=0.4)
    weight_sd_to_hd: float = Field(default=0.3)
    weight_sd_to_4k: float = Field(default=0.3)
    weight_4k_to_8k: float = Field(default=0.1)
    weight_hd_to_8k: float = Field(default=0.1)
