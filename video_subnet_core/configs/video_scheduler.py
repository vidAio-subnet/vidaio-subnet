from pydantic import BaseModel, Field


class VideoSchedulerConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    max_synthetic_queue_size: int = Field(default=30)
    refill_threshold: int = Field(default=27)
    refill_target: int = Field(default=30)
    clip_duration: int = Field(default=1)
    min_video_len: int = Field(default=1)
    max_video_len: int = Field(default=20)
