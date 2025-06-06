from pydantic import BaseModel, Field


class SQLConfig(BaseModel):
    url: str = Field(default="sqlite:///video_subnet_validator.db")
