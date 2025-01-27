from pydantic import BaseModel, Field


class VideoSchedulerConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    max_synthetic_queue_size: int = Field(default=4)
    refill_threshold: int = Field(default=2)
    refill_target: int = Field(default=4)
    clip_duration: int = Field(default=1)
    min_video_len: int = Field(default=1)
    max_video_len: int = Field(default=5)
