from pydantic_settings import BaseSettings
from .configs import (
    RedisConfig,
    VideoSchedulerConfig,
    BandwidthConfig,
    ScoreConfig,
    SQLConfig,
    StorageConfig,
    OrganicGatewayConfig,
    DashboardConfig,
    ChutesConfig,
)
from loguru import logger


class GlobalConfig(BaseSettings):
    redis: RedisConfig = RedisConfig()
    video_scheduler: VideoSchedulerConfig = VideoSchedulerConfig()
    bandwidth: BandwidthConfig = BandwidthConfig()
    score: ScoreConfig = ScoreConfig()
    sql: SQLConfig = SQLConfig(url="sqlite:///video_subnet_validator.db")
    storage: StorageConfig = StorageConfig()
    organic_gateway: OrganicGatewayConfig = OrganicGatewayConfig()
    dashboard: DashboardConfig = DashboardConfig()
    chutes: ChutesConfig = ChutesConfig()
    SUBNET_TEMPO: int = 100

    class Config:
        env_nested_delimiter = "__"


CONFIG = GlobalConfig()
logger.info(f"Configuration loaded correctly")
