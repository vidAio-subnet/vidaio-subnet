from pydantic import BaseModel, Field


class SQLConfig(BaseModel):
    url: str = Field(env="SQL_URL")
