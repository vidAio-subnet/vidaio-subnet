from pydantic import BaseModel
from .configs import RedisConfig, VideoSchedulerConfig, BandwidthConfig, ScoreConfig


class GlobalConfig(BaseModel):
    redis: RedisConfig = RedisConfig()
    video_scheduler: VideoSchedulerConfig = VideoSchedulerConfig()
    bandwidth: BandwidthConfig = BandwidthConfig()
    score: ScoreConfig = ScoreConfig()


CONFIG = GlobalConfig()
