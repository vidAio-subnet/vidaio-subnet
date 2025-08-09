from pydantic import BaseModel, Field


class BandwidthConfig(BaseModel):
    total_requests: int = Field(default=64)
    request_interval: int = Field(default=500)
    requests_per_synthetic_interval: int = Field(default=20)
    requests_per_organic_interval: int = Field(default = 30)
    miners_per_task: int = Field(default=2)
    min_stake: int = Field(default=25000)
