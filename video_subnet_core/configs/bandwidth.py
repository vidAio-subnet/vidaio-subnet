from pydantic import BaseModel, Field


class BandwidthConfig(BaseModel):
    total_requests: int = Field(default=1000000, env="TOTAL_REQUESTS")
    request_interval: int = Field(default=60, env="REQUEST_INTERVAL")
    synthetic_request_per_interval: int = Field(
        default=5, env="SYNTHETIC_REQUEST_PER_INTERVAL"
    )
    min_stake: int = Field(default=10000, env="MIN_STAKE")
