from pydantic import BaseModel, Field
import os


class RedisConfig(BaseModel):
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    organic_queue_key: str = Field(default="organic_queue", env="ORGANIC_QUEUE_KEY")
    synthetic_queue_key: str = Field(
        default="synthetic_queue", env="SYNTHETIC_QUEUE_KEY"
    )
    miner_manager_key: str = Field(default="miner_manager", env="MINER_MANAGER_KEY")
