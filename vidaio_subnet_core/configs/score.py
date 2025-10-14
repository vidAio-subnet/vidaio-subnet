from pydantic import BaseModel, Field


class ScoreConfig(BaseModel):
    decay_factor: float = Field(default=0.9)
    host: str = Field(default="localhost")
    port: int = Field(default=8201)
    vmaf_threshold: float = Field(default=0.5)
    vmaf_sample_count: int = Field(default=10)
    pieapp_threshold: float = Field(default=1.0)
    pieapp_sample_count: int = Field(default=4)
