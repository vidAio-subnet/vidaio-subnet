from pydantic import BaseModel, Field
import os


class RedisConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    organic_queue_key: str = Field(default="organic_queue")
    synthetic_queue_key: str = Field(default="synthetic_queue")
    pexels_video_ids_key: str = Field(default = "pexels_queue")
    miner_manager_key: str = Field(default="miner_manager")
