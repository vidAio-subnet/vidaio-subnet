from pydantic import BaseModel, Field


class BandwidthConfig(BaseModel):
    total_requests: int = Field(default=64)
    request_interval: int = Field(default=60)
    synthetic_request_per_interval: int = Field(default=5)
    min_stake: int = Field(default=10000)
