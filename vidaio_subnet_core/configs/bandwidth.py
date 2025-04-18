from pydantic import BaseModel, Field


class BandwidthConfig(BaseModel):
    total_requests: int = Field(default=64)
    request_interval: int = Field(default=500)
    requests_per_interval: int = Field(default=20)
    requests_organic_interval: int = Field(default = 10)
    requests_organic_interval: int = Field(default = 1)
    min_stake: int = Field(default=25000)
