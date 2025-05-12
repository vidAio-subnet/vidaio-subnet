from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
import os

load_dotenv()

class DashboardConfig(BaseModel):
    endpoint: str = Field(default=os.getenv("DASHBOARD_ENDPOINT", "localhost:9001"))