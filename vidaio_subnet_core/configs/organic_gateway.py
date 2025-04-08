from pydantic import BaseModel, Field


class VideoUpscalerConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=29116)
