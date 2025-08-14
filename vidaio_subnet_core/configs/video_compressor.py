from pydantic import BaseModel, Field


class VideoCompressorConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=29116)