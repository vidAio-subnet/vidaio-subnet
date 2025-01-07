from pydantic import BaseModel, Field


class VideoSchedulerConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
