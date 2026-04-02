from pydantic import BaseModel, Field


class ChutesConfig(BaseModel):
    api_key: str = Field(default="")
    upscaling_slug: str = Field(default="")
    compression_slug: str = Field(default="")
    request_timeout: int = Field(default=600)
