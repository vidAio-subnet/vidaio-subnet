from pydantic import BaseModel, Field


class ScoreConfig(BaseModel):
    decay_factor: float = Field(default=0.95)
    host: str = Field(default="localhost")
    port: int = Field(default=8201)
    vmaf_threshold: float = Field(default=0.5)
    sample_count: int = Field(default=7)
