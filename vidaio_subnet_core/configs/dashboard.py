from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
import os

load_dotenv()

class DashboardConfig(BaseModel):
    endpoint: str = Field(default=os.getenv("DASHBOARD_ENDPOINT", "localhost:9001"))
    timeout: int = Field(default=10)
    max_retries: int = Field(default=3)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=29117)
    token: str = Field(default=os.getenv("DASHBOARD_TOKEN", ""))
    api_key: str = Field(default=os.getenv("DASHBOARD_API_KEY", ""))