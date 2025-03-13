from pydantic import BaseModel, Field


class VideoSchedulerConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    max_synthetic_queue_size: int = Field(default=50)
    refill_threshold: int = Field(default=3)
    refill_target: int = Field(default=10)
    clip_duration: int = Field(default=1)
    min_video_len: int = Field(default=10)
    max_video_len: int = Field(default=30)
    pexels_max_size: int = Field(default = 100)
    pexels_threshold: int = Field(default = 85)
    weight_hd_to_4k: int = Field(default = 0.4)
    weight_sd_to_hd: int = Field(default = 0.2)
    weight_sd_to_4k: int = Field(default = 0.2)
    weight_4k_to_8k: int = Field(default = 0.2)

