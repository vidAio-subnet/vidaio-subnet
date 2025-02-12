from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

class MinioConfig(BaseModel):
    endpoint: str = Field(default=os.getenv("S3_COMPATIBLE_ENDPOINT", "localhost:9000"))
    access_key: str = Field(default=os.getenv("S3_COMPATIBLE_ACCESS_KEY", "minioadmin"))
    secret_key: str = Field(default=os.getenv("S3_COMPATIBLE_SECRET_KEY", "minioadmin"))
    secure: bool = Field(default=True)
    bucket_name: str = Field(default=os.getenv("BUCKET_NAME", "vidaio-subnet"))
