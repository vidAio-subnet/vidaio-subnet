from pydantic import BaseModel, Field
import os


class RedisConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    delete_after_second: int = Field(default=600)
    redis_ttl: int = Field(default= 60 * 60 * 6) # 6 hours
    organic_upscaling_queue_key: str = Field(default="organic_upscaling_queue")
    organic_compression_queue_key: str = Field(default="organic_compression_queue")
    synthetic_5s_clip_queue_key: str = Field(default="5s_clips")
    synthetic_10s_clip_queue_key: str = Field(default="10s_clips")
    synthetic_20s_clip_queue_key: str = Field(default="20s_clips")
    synthetic_compression_queue_key: str = Field(default="compression_queue")
    pexels_video_ids_key: str = Field(default = "pexels_queue")
    youtube_video_ids_key: str = Field(default="youtube_queue")
    miner_manager_key: str = Field(default="miner_manager")
    file_deletioin_key: str = Field(default="minio_file_deletion_queue")
