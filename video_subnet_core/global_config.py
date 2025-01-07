from pydantic_settings import BaseSettings
from .configs import (
    RedisConfig,
    VideoSchedulerConfig,
    BandwidthConfig,
    ScoreConfig,
    SQLConfig,
)
from loguru import logger


class GlobalConfig(BaseSettings):
    redis: RedisConfig = RedisConfig()
    video_scheduler: VideoSchedulerConfig = VideoSchedulerConfig()
    bandwidth: BandwidthConfig = BandwidthConfig()
    score: ScoreConfig = ScoreConfig()
    sql: SQLConfig = SQLConfig(url="sqlite:///video_subnet.db")

    class Config:
        env_nested_delimiter = "__"


CONFIG = GlobalConfig()
logger.info(f"GlobalConfig: {CONFIG}")
