from pydantic import BaseModel, Field


class ScoreConfig(BaseModel):
    decay_factor: float = Field(default=0.9, env="DECAY_FACTOR")
