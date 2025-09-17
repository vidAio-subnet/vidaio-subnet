from pydantic import BaseModel, Field


class OrganicGatewayConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=29996)
