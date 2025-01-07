from pydantic import BaseModel, Field


class ScoreConfig(BaseModel):
    decay_factor: float = Field(default=0.9, env="DECAY_FACTOR")
    host: str = Field(default="http://localhost", env="SCORE_HOST")
    port: int = Field(default=8200, env="SCORE_PORT")
