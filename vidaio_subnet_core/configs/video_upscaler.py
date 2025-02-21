from pydantic import BaseModel, Field


class VideoUpscalerConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=29115)