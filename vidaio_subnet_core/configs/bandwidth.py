from pydantic import BaseModel, Field


class BandwidthConfig(BaseModel):
    total_requests: int = Field(default=64)
    request_interval: int = Field(default=4000)
    requests_per_interval: int = Field(default=20)
    min_stake: int = Field(default=20000)
