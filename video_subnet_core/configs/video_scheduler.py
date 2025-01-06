from pydantic import BaseModel, Field


class VideoSchedulerConfig(BaseModel):
    host: str = Field(default="localhost", env="VIDEO_SCHEDULER_HOST")
    port: int = Field(default=8000, env="VIDEO_SCHEDULER_PORT")
