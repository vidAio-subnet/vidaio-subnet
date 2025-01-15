from pydantic import BaseModel, Field


class MinioConfig(BaseModel):
    endpoint: str = Field(default="localhost:9000")
    access_key: str = Field(default="minioadmin")
    secret_key: str = Field(default="minioadmin")
    secure: bool = Field(default=False)
    bucket_name: str = Field(default="video-subnet")
